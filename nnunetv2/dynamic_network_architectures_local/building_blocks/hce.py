from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableBias(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class RPReLU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.shift_in = nn.Parameter(torch.zeros(channels))
        self.prelu = nn.PReLU(channels)
        self.shift_out = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is typically [B, D*H*W, C] (flattened in MSGDC branches).
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return self.prelu((x - self.shift_in).transpose(-1, -2)).transpose(-1, -2) + self.shift_out


class MSGDC(nn.Module):
    """Multi-Scale Dilated Convolution (MSGDC)."""

    def __init__(
        self,
        channels: int,
        dilation_rates: Sequence[int] = (1, 3, 5),
        kernel_size: int = 3,
        stride: int = 1,
        padding: str | int = "same",
    ):
        super().__init__()
        self.bias = LearnableBias(channels)
        self.convs = nn.ModuleList(
            [
                nn.Conv3d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=int(d),
                    bias=True,
                )
                for d in dilation_rates
            ]
        )
        self.norm = nn.LayerNorm(channels)
        self.acts = nn.ModuleList([RPReLU(channels) for _ in dilation_rates])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        x = self.bias(x)

        branch_outs = []
        for conv, act in zip(self.convs, self.acts):
            y = conv(x).permute(0, 2, 3, 4, 1).flatten(1, 3)  # [B, D*H*W, C]
            y = act(y)
            branch_outs.append(y)

        y = self.norm(branch_outs[0] + branch_outs[1] + branch_outs[2])
        return y.permute(0, 2, 1).view(b, c, d, h, w).contiguous()


class ConvBNReLU6(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilate: int = 1,
        bias: bool = False,
        groups: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilate,
                bias=bias,
                groups=groups,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU6(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ICCBlock(nn.Module):
    """Inter-Channel Communication (ICC) implemented by 3D channel shuffle + 1×1×1 mixing."""

    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.groups = groups
        self.conv = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm3d(channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        group_channels = c // self.groups
        out = x.view(b, self.groups, group_channels, d, h, w)
        out = out.permute(0, 2, 1, 3, 4, 5).contiguous().view(b, c, d, h, w)
        out = self.conv(out)
        out = self.norm(out)
        out = self.act(out)
        return out + x


class HCEBlock(nn.Module):
    def __init__(
        self,
        stage_channels: int,
        neighbor_channels: int,
        attention_type: Optional[str] = "SE",
        reduction: int = 16,
        enable_icc: bool = False,
        enable_msgdc: bool = True,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.fuse = ConvBNReLU6(stage_channels + neighbor_channels, stage_channels, 3, 1, 1)
        self.enable_msgdc = enable_msgdc
        if enable_msgdc:
            self.msgdc = MSGDC(stage_channels)
        self.drop = nn.Dropout(dropout_p)

        if attention_type == "SE":
            self.attention = SEBlock(stage_channels, reduction=reduction)
        else:
            self.attention = None

        self.enable_icc = enable_icc
        if enable_icc:
            self.icc = ICCBlock(stage_channels)

    def forward(self, x: torch.Tensor, neighbor: torch.Tensor) -> torch.Tensor:
        x_cat = torch.cat([x, neighbor], 1)
        x_fused = self.fuse(x_cat)
        if self.enable_msgdc:
            x_out = self.msgdc(x_fused) + x
        else:
            x_out = x_fused + x
        if self.attention is not None:
            x_out = self.attention(x_out)
        if self.enable_icc:
            x_out = self.icc(x_out)
        return self.drop(x_out)


class HCE(nn.Module):
    """Hierarchical Context Enhancement (HCE) module applied to all skip features."""

    def __init__(
        self,
        channels: Optional[Sequence[int]] = None,
        attention_type: str = "SE",
        enable_msgdc: Optional[Sequence[bool]] = None,
        attention_types: Optional[Sequence[Optional[str]]] = None,
        reduction_factors: Optional[Sequence[int]] = None,
        enable_icc: Optional[Sequence[bool]] = None,
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 256, 320, 320, 320]
        self.channels = list(channels)
        num_stages = len(self.channels)

        def _validated_list(name: str, values: Sequence, expected_len: int):
            values = list(values)
            if len(values) != expected_len:
                raise ValueError(f"{name} must have length {expected_len}, but got {len(values)}")
            return values

        if attention_types is None:
            attention_types = []
            for stage in range(num_stages):
                attention_types.append(None if stage == 0 else attention_type)
        else:
            attention_types = _validated_list("attention_types", attention_types, num_stages)

        if reduction_factors is None:
            reduction_factors = []
            for stage in range(num_stages):
                if attention_types[stage] == "SE":
                    # Paper-aligned: use a smaller reduction at shallow stages (1–2) to better preserve boundary details.
                    reduction_factors.append(8 if stage in (1, 2) else 16)
                else:
                    reduction_factors.append(16)
        else:
            reduction_factors = _validated_list("reduction_factors", reduction_factors, num_stages)

        if enable_msgdc is None:
            enable_msgdc = [True] * num_stages
        else:
            enable_msgdc = [bool(v) for v in _validated_list("enable_msgdc", enable_msgdc, num_stages)]

        if enable_icc is None:
            enable_icc = [stage >= 3 for stage in range(num_stages)]
        else:
            enable_icc = [bool(v) for v in _validated_list("enable_icc", enable_icc, num_stages)]

        self.blocks = nn.ModuleList()
        for idx, ch in enumerate(self.channels):
            if idx == 0:
                neighbor_ch = self.channels[idx + 1]
            elif idx == num_stages - 1:
                neighbor_ch = self.channels[idx - 1]
            else:
                neighbor_ch = self.channels[idx - 1] + self.channels[idx + 1]

            self.blocks.append(
                HCEBlock(
                    stage_channels=ch,
                    neighbor_channels=neighbor_ch,
                    attention_type=attention_types[idx],
                    reduction=reduction_factors[idx],
                    enable_icc=enable_icc[idx],
                    enable_msgdc=enable_msgdc[idx],
                )
            )

    def forward(self, feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        feats = list(feats)
        for i in range(len(feats)):
            if i == 0:
                fused = F.interpolate(feats[1], feats[i].shape[2:], mode="trilinear", align_corners=True)
                feats[i] = self.blocks[i](feats[i], fused)
            elif i == len(feats) - 1:
                fused = F.interpolate(feats[-2], feats[i].shape[2:], mode="trilinear", align_corners=True)
                feats[i] = self.blocks[i](feats[i], fused)
            else:
                down = F.interpolate(feats[i - 1], feats[i].shape[2:], mode="trilinear", align_corners=True)
                up = F.interpolate(feats[i + 1], feats[i].shape[2:], mode="trilinear", align_corners=True)
                feats[i] = self.blocks[i](feats[i], torch.cat([down, up], 1))
        return feats
