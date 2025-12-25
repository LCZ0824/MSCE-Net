# 索引

## 产品文档
无。

## 前端文档
无。

## 后端文档
无。

## 当前任务文档
无。

## 全局重要记忆
- 本包包含 HCE 模块及其在 nnU-Net v2 中的最小集成。
- HCE 实现路径：`nnunetv2/dynamic_network_architectures_local/building_blocks/hce.py`。
- 集成路径：`nnunetv2/dynamic_network_architectures_local/architectures/unet.py`（`PlainConvUNet` 在 skip（跳接）特征上应用 HCE）。
- 开源包包含 `ckpt/` 权重与 `CDEMRIS_RA/` 外部标注文件，发布前需确认公开授权。
