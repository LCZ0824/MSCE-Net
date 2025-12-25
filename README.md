# MSCE-Net Open-Source Minimal Package (HCE Module + nnU-Net v2 Reference Integration)

This directory provides the HCE (Hierarchical Context Enhancement) module of MSCE-Net and its reference integration point in nnU-Net v2 for ease of understanding and reuse.

## Included
- HCE module implementation
- nnU-Net v2 reference integration (applied to skip features)
- External annotations
- Weights with results fully consistent with the paper

## Folder Structure
- `nnunetv2/dynamic_network_architectures_local/building_blocks/hce.py`
  HCE module implementation.
- `nnunetv2/dynamic_network_architectures_local/architectures/unet.py`
  nnU-Net v2 reference integration: apply HCE to skip features in `PlainConvUNet`; the training/inference pipeline follows nnU-Net v2 conventions.
- `ckpt/`
  Training weights. Currently includes `checkpoint_best.pth` and `checkpoint_final.pth`. For reference only; verify and adapt as needed.

## How to Use (Reference)
1. Prepare the nnU-Net v2 source code (editable install recommended).
2. Copy/overwrite the following two files into the corresponding paths of your `nnunetv2/` source tree:
   - `nnunetv2/dynamic_network_architectures_local/building_blocks/hce.py`
   - `nnunetv2/dynamic_network_architectures_local/architectures/unet.py`
3. Train/infer with the standard nnU-Net v2 workflow.
4. If you load weights from `ckpt/`, ensure the model structure, plans, preprocessing, and training configuration are consistent; otherwise results may deviate.

## Default Implementation Settings (Reference)
- MSGDC (Multi-Scale Dilated Convolution): three parallel `3x3x3` dilated convolutions with dilation rates `{1,3,5}`, with `LearnableBias + RPReLU + LayerNorm`.
- SE (Squeeze-and-Excitation): disabled at stage 0; `reduction=8` for stages 1-2; `reduction=16` for stages 3-6.
- ICC (Inter-Channel Communication): enabled only for stages >=3 with `groups=4`, with `1x1x1 Conv + BN + ReLU` residual mixing.
- Dropout: `p=0.1`.

## License and Attribution
This package is released under the Apache License 2.0 (see `LICENSE`).
nnU-Net v2 is an independent open-source project. This package only contains minimal integration code; please comply with the nnU-Net v2 license when redistributing derived files.
