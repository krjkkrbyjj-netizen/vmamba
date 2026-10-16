# Robustness of Scanning Sequence in Visual Mamba-Based Object Detection Model

## Description

This repository contains the official implementation of the research presented in the paper *"Robustness of Scanning Sequence in Visual Mamba-Based Object Detection Model"*. 

The project investigates the robustness of scanning sequence strategies in Visual Mamba (VMamba) architectures for object detection tasks. It implements a selective scan mechanism accelerated by Triton kernels, supporting multiple scanning modes including cross-scan, unidirectional, and bidirectional scanning patterns. The codebase builds upon the VMamba backbone integrated with detection heads, enabling systematic evaluation of how different scanning sequences affect detection performance across diverse datasets.

Key features:
- **Triton-accelerated selective scan**: Custom CUDA kernels via Triton for efficient cross-scan operations (`simple_triton.py`)
- **Multiple scanning strategies**: Cross-scan (4 directions), unidirectional, and bidirectional scanning modes
- **Flexible tensor layouts**: Supports both channel-first `(B, C, H, W)` and channel-last `(B, H, W, C)` formats
- **Multi-dataset compatibility**: Validated on PASCAL VOC, KITTI, ImageNet-1K, VOC2012, and VisDrone datasets

## Dataset Information

The following datasets are used in this study. All datasets are publicly available and should be downloaded separately following their respective licensing terms.

| Dataset | Type | License | Preprocessing |
|---------|------|---------|---------------|
| **PASCAL VOC** (2007+2012) | Object detection | Creative Commons Attribution-ShareAlike 4.0 | Standard train/val/test split; images resized to 512×512 |
| **KITTI** | Autonomous driving (2D detection) | CC BY-NC-SA 3.0 | Left color images; bounding box annotations for Car, Pedestrian, Cyclist |
| **ImageNet-1K** | Image classification (pre-training) | Custom (non-commercial research) | Resolution 224×224; standard normalization |
| **VOC2012** | Semantic/object detection | Creative Commons Attribution-ShareAlike 4.0 | Augmented with SBD dataset for additional annotations |
| **VisDrone** | Drone-captured object detection | CC BY-NC 4.0 | 10,209 images; 10 object categories; images at native resolution |

**Data preparation steps:**

1. Download each dataset from its official source.
2. Organize datasets in the following directory structure:
