# VMamba: Robustness of Scanning Sequence in Visual Mamba-Based Object Detection

## Description

This repository contains the official implementation of the paper **"Robustness of Scanning Sequence in Visual Mamba-Based Object Detection Model"** (submitted to PeerJ Computer Science). 

We introduce a **unified disentanglement framework** for Vision Mamba-based object detection that explicitly separates **scanning topology** from **sequence multiplicity**. Through extensive controlled experiments on PASCAL VOC (2D detection), KITTI (3D detection), ImageNet-1K (classification), and VOC2012 (semantic segmentation), we demonstrate that:

- Detection performance remains consistently robust across five canonical scanning topologies (raster, cross, zigzag, slanted, spiral) and varied sequence counts (K ∈ {1, 2, 4, 8}).
- **Patch granularity** is the dominant factor governing detection efficacy, overshadowing the marginal gains from sophisticated traversal orders.
- Under constrained computational budgets, prioritizing finer patch granularity over complex multi-scan designs yields superior performance.

All source-code comments (including `simple_triton.py`) are written in English. The codebase is permanently archived on Zenodo: [10.5281/zenodo.21783278](https://doi.org/10.5281/zenodo.21783278).

## Dataset Information

| Dataset | Task | Source | License |
|---------|------|--------|---------|
| PASCAL VOC 2007 & 2012 | 2D Detection | [Official Site](http://host.robots.ox.ac.uk/pascal/VOC/) | Challenge usage terms; images from Flickr |
| KITTI Object Detection | 3D Detection | [Official Site](https://www.cvlibs.net/datasets/kitti/) | CC BY-NC-SA 3.0 |
| ImageNet-1K | Classification Pretraining | [Official Site](https://image-net.org/challenges/LSVRC/2012/) | ImageNet Terms of Access (non-commercial research) |
| PASCAL VOC2012 | Semantic Segmentation | [Official Site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) | Challenge usage terms; images from Flickr |
| VisDrone | Small Object Detection | [Official Site](http://aiskyeye.com/) | Academic license |

**Data Preprocessing:**
- All images resized to 512×512, normalized with ImageNet statistics (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]).
- Bounding boxes scaled proportionally to resized dimensions.
- KITTI point clouds voxelized with grid size 0.05m×0.05m×0.05m, filtered to range [-3,1]m height, [-40,40]m width, [0,70]m depth.
- VOC2012 segmentation: random crop to 512×512 (training), center crop (inference).

## Code Information

## Data Availability
The unified disentanglement framework and all source code supporting the findings of this study are permanently archived at Zenodo: https://doi.org/10.5281/zenodo.21783278

The codebase is also available on GitHub: https://github.com/krjkkrbyjj-netizen/vmamba
1. PASCAL VOC 2007 & 2012 (2D Detection)
•
Source: http://host.robots.ox.ac.uk/pascal/VOC/
￼
•
License: VOC challenge official usage terms. Images are sourced from Flickr and subject to Flickr terms of use.
•
Usage in this work:
•
Training: Union of VOC2007 and VOC2012 trainval splits
•
Evaluation: VOC2007 test set
•
Citation: Everingham et al., The PASCAL Visual Object Classes Challenge: A Retrospective, International Journal of Computer Vision (IJCV), 2015.
2. KITTI Object Detection (3D Detection)
•
Source: https://www.cvlibs.net/datasets/kitti/
￼
•
License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)
•
Registration required: Users must register at the KITTI portal and declare their purpose of use before downloading.
•
Usage in this work:
•
Official KITTI split: 3,712 training + 3,769 validation samples
•
Metrics: AP_BEV and AP_3D under AP11 and AP40 protocols
•
Citation: Geiger et al., Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite, Conference on Computer Vision and Pattern Recognition (CVPR), 2012. DOI: 10.1177/0278364913491297
3. ImageNet-1K (Backbone Pretraining)
•
Source: https://image-net.org/challenges/LSVRC/2012/
•
License: ImageNet Terms of Access — non-commercial research and educational use only. Users must register and agree to the Terms of Access before downloading.
•
Usage in this work: VMamba backbone pretrained on ImageNet-1K (1,281,167 training images across 1,000 categories).
4. PASCAL VOC2012 (Semantic Segmentation)
•
Source: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
￼
•
License: VOC challenge official usage terms; images sourced from Flickr subject to Flickr terms of use.
•
Usage in this work: 20 semantic classes; images resized to 512×512 during training/inference.
Code Information
•
simple_triton.py: Triton-based implementation of five scanning topologies (spatial-first, channel-first, zigzag, spiral, hybrid).
•
train_voc.py: Training pipeline for 2D detection on PASCAL VOC.
•
train_kitti.py: Training pipeline for monocular 3D detection on KITTI.
•
train_seg.py: Training pipeline for semantic segmentation on VOC2012.
•
eval.py: Evaluation script computing mAP@50:95, AP50/AP70/AP90, AP_BEV, AP_3D, Top-1 Accuracy, and mIoU.
•
configs/: YAML configuration files for all experiment setups.
Usage Instructions
Step 1: Clone and install dependencies
bash
￼

git clone
 https://github.com/krjkkrbyjj-netizen/vmamba.git
cd
 vmamba
pip install -r requirements.txt
￼
￼
Step 2: Download datasets
Download each dataset from its official source (see Dataset Information above) and organize them in the following directory structure:

./data/
├── VOCdevkit/
│   ├── VOC2007/
│   └── VOC2012/
├── KITTI/
│   ├── training/
│   └── validation/
├── ImageNet/
│   └── ILSVRC2012/
└── VOC2012_SEG/
    └── VOC2012/

Step 3: Train models
bash

# 2D detection on PASCAL VOC
python train_voc.py --scan zigzag --patch_size 16 --batch_size 8

# Monocular 3D detection on KITTI
python train_kitti.py --scan spatial_first --patch_size 16 --batch_size 4

# Semantic segmentation on VOC2012
python train_seg.py --scan spiral --patch_size 16

Step 4: Evaluate trained models
bash

python eval.py --checkpoint outputs/best.pth --dataset voc

Requirements
•
Python >= 3.8
•
PyTorch >= 2.0
•
CUDA >= 11.8
•
triton >= 2.1
•
mmcv >= 2.0
•
mmengine >= 0.9
•
numpy >= 1.21
•
See requirements.txt for the full dependency list.
Methodology
1.
Patch Partition: Input images are divided into patches of configurable granularity (8/16/32 pixels).
2.
Scanning Topologies: Five scanning strategies are implemented: spatial-first, channel-first, zigzag, spiral, and hybrid.
3.
VMamba Encoding: Patch sequences are forwarded through the VMamba encoder with Visual State Space (VSS) blocks.
4.
Task-Specific Heads: Faster R-CNN for 2D detection, SECOND-style head for 3D detection, and FCN for semantic segmentation.
5.
Evaluation: Standard metrics are reported per dataset specification.
Data Preprocessing
•
All image inputs: Resized to 512×512 and normalized using ImageNet statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]). Bounding boxes are scaled proportionally to the resized image dimensions.
•
KITTI point clouds: Voxelized with a fixed grid size of 0.05m × 0.05m × 0.05m, filtered to a spatial range of [-3, 1]m in height, [-40, 40]m in width, and [0, 70]m in depth.
•
VOC2012 segmentation: Randomly cropped to 512×512 during training, center-cropped at inference.
License
MIT License
Copyright (c) 2026 Dongfu Yin, Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ)
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
