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
data/
├── VOCdevkit/
│ ├── VOC2007/
│ └── VOC2012/
├── KITTI/
│ ├── images/
│ └── labels/
├── imagenet/
│ ├── train/
│ └── val/
└── VisDrone/
├── VisDrone2019-DET-train/
├── VisDrone2019-DET-val/
└── VisDrone2019-DET-test/
3. Run preprocessing scripts (if applicable):
bash
python tools/prepare_voc.py --data-root data/VOCdevkit
python tools/prepare_kitti.py --data-root data/KITTI
## Code Information

### Directory Structure
vmamba/
├── models/
│ ├── backbone/
│ │ ├── simple_triton.py # Triton-accelerated selective scan (core kernel)
│ │ ├── vmamba.py # VMamba backbone implementation
│ │ └── ...
│ ├── detection/
│ │ ├── detector.py # Object detection head
│ │ └── ...
│ └── ...
├── tools/
│ ├── train.py # Training entry point
│ ├── test.py # Evaluation entry point
│ └── ...
├── configs/
│ ├── voc_config.py # PASCAL VOC configuration
│ ├── kitti_config.py # KITTI configuration
│ ├── imagenet_config.py # ImageNet pre-training configuration
│ ├── voc2012_config.py # VOC2012 configuration
│ └── visdrone_config.py # VisDrone configuration
├── data/ # Dataset directory (user-managed)
├── requirements.txt # Python dependencies
└── README.md # This file
### Key Modules

| Module | Description |
|--------|-------------|
| `simple_triton.py` | Core Triton JIT kernel for cross-scan and merge operations; includes both forward and backward passes via `torch.autograd.Function` |
| `vmamba.py` | VMamba backbone with selective state space model (SSM) blocks |
| `detector.py` | Detection head supporting multiple scanning sequence configurations |
| `train.py` | Training script with configurable hyperparameters |
| `test.py` | Evaluation script reporting mAP, precision, recall |

## Usage Instructions

### Installation
bash
Clone the repository
git clone https://github.com/krjkkrbyjj-netizen/vmamba.git
cd vmamba
Create a conda environment (recommended)
conda create -n vmamba python=3.9 -y
conda activate vmamba
Install dependencies
pip install -r requirements.txt
Install Triton (for CUDA acceleration, optional but recommended)
pip install triton>=2.1
### Training
bash
Train on PASCAL VOC
python tools/train.py --config configs/voc_config.py --batch-size 16 --epochs 50
Train on KITTI
python tools/train.py --config configs/kitti_config.py --batch-size 8 --epochs 40
Train on VisDrone
python tools/train.py --config configs/visdrone_config.py --batch-size 4 --epochs 60
Train with specific scanning strategy
python tools/train.py --config configs/voc_config.py --scan-mode cross_scan --epochs 50
### Evaluation
bash
Evaluate on test set
python tools/test.py --config configs/voc_config.py --checkpoint checkpoints/best_model.pth
Evaluate with different scanning sequences for robustness comparison
python tools/test.py --config configs/voc_config.py --checkpoint checkpoints/best_model.pth --compare-scans
### Loading Pre-trained Weights
python
from models.backbone.vmamba import VMambaBackbone
model = VMambaBackbone(pretrained=True, scan_mode='cross_scan')
## Requirements

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **CUDA**: 11.7 or higher (for Triton acceleration)
- **GPU**: NVIDIA GPU with compute capability ≥ 7.0 (Volta or newer)

### Python Dependencies
torch>=2.0.0
torchvision>=0.15.0
triton>=2.1.0 # Optional but recommended for CUDA acceleration
mmcv>=2.0.0
mmengine>=0.7.0
numpy>=1.21.0
tqdm>=4.64.0
pyyaml>=6.0
matplotlib>=3.5.0
opencv-python>=4.6.0
Install all dependencies:
bash
pip install -r requirements.txt
## Methodology

### Selective Scan Mechanism

The core contribution of this work is the systematic evaluation of scanning sequence robustness in Visual Mamba-based object detection. The selective scan mechanism processes 2D feature maps through multiple scanning paths:

1. **Cross-scan (4 directions)**: Top-left→bottom-right, top-right→bottom-left, bottom-left→top-right, bottom-right→top-left
2. **Unidirectional scan**: Single-direction sequential processing
3. **Bidirectional scan**: Forward and reverse direction only

### Cross-Scan Operation

For an input feature map $X \in \mathbb{R}^{B \times C \times H \times W}$, the cross-scan expands it into 4 views:

$$Y = \text{Stack}(X_{\rightarrow}, X_{\downarrow}, X_{\leftarrow}, X_{\uparrow})$$

where each view applies a different spatial traversal order. The merge operation aggregates all 4 views back into the original feature space.

### Implementation Details

- **Triton kernel**: `triton_cross_scan_flex` implements the scan/merge operations with configurable layouts, scanning modes, and block sizes (BC=1, BH=32, BW=32)
- **Autograd integration**: `CrossScanTritonF` and `CrossMergeTritonF` wrap Triton kernels as `torch.autograd.Function` for seamless backpropagation
- **Fallback**: If Triton is unavailable, the code automatically falls back to pure PyTorch implementations (`CrossScanF`, `CrossMergeF`)

### Experimental Design

The robustness evaluation follows this pipeline:
1. Train detection models with different scanning sequences on each dataset
2. Measure performance degradation under input perturbations (rotation, scaling, occlusion)
3. Compare cross-scan vs. unidirectional vs. bidirectional strategies across datasets

## Citation

If you use this code or dataset preparation pipeline in your research, please cite:
bibtex
@article{anonymous2024robustness,
title={Robustness of Scanning Sequence in Visual Mamba-Based Object Detection Model},
author={Anonymous},
journal={PeerJ Computer Science},
year={2024},
note={Under review}
}
*Note: Full citation will be updated upon acceptance.*

## Contribution Guidelines

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Commit your changes**: `git commit -m 'Add some feature'`
4. **Push to the branch**: `git push origin feature/your-feature-name`
5. **Open a Pull Request**

### Code Style
- Python code should follow [PEP 8](https://pep8.org/)
- Add docstrings to all public functions and classes
- All comments must be written in **English**
- Run `flake8` or `black` before submitting

### Reporting Issues
- Use the GitHub Issues tab to report bugs
- Include your environment details (OS, CUDA version, Python version, GPU model)
- Provide a minimal reproducible example

### Contact

For questions or collaboration inquiries, please contact: chenxinran@gml.ac.cn
