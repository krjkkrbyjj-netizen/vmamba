# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'
import sys
import os

# 添加 detection 文件夹到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import argparse
import numpy as np
import torch
from timm.utils import AverageMeter
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from erf.resnet_for_erf import resnet18, resnet50,resnet101,resnet152
from erf.vmamba_for_erf import VMambaForERF
from torch import optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.datasets import VOCDetection

def parse_args():
    parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)
    parser.add_argument('--model', default='mobilemamba', type=str, help='model name')
    parser.add_argument('--weights', default=None, type=str, help='path to weights file. For resnet101/152, ignore this arg to download from torchvision')
    parser.add_argument('--data_path', default='/exp_zbc/ImageNet/imagenet/', type=str, help='dataset path')
    parser.add_argument('--save_path', default='vmamba_origin4_imgnet.npy', type=str, help='path to save the ERF matrix (.npy file)')
    parser.add_argument('--num_images', default=50, type=int, help='num of images to use')
    args = parser.parse_args()
    return args


def get_input_grad(model, samples):
    outputs = model(samples)
    output = outputs[3];  #使用不同模块的输出，这里有四个模块output
    out_size = output.size()
    central_point = torch.nn.functional.relu(output[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map

# 定义模型参数
# model_params = {
#     'img_size': 224,
#     'in_chans': 3,
#     'num_classes': 80,
#     'stages': ['s', 's', 's'],
#     'embed_dim': [200, 376, 448],
#     'global_ratio': [0.8, 0.7, 0.6],
#     'local_ratio': [0.2, 0.2, 0.3],
#     'depth': [2, 3, 2],
#     'kernels': [7, 5, 3],
#     'down_ops': [['subsample', 2], ['subsample', 2], ['']],
#     'distillation': False,
#     'drop_path': 0.03,
#     'ssm_ratio': 2,
#     'forward_type': "v2",
#     'sync_bn': False,
#     'pretrained' : '/exp_zbc/VMamba/detection/work_dirs/faster_rcnn_mobilemamba_b1_fpn_1x_voc/20250324_005638_bs=8_1/epoch_12.pth',
#     'out_indices': (1, 2, 3),
#     'frozen_stages': -1,
#     'norm_eval': True,
# }

model_params={
        'out_indices': (0, 1, 2, 3),
        # 'pretrained' : "/exp_zbc/VMamba/vssm1_tiny_0230s_ckpt_epoch_264.pth",
        # copied from classification/configs/vssm/vssm_tiny_224.yaml
        'dims': 96,
        'depths': (2, 2, 9, 2),
        'ssm_d_state': 16,
        'ssm_dt_rank': "auto",
        'ssm_ratio': 2.0,
        'mlp_ratio': 0.0,
        'downsample_version': "v1",
        'patchembed_version': "v1",
        # forward_type="v0", # if you want exactly the same
}


def main(args):
    #   ================================= transform: resize to 1024x1024
    # t = [
    #     transforms.Resize((1024, 1024), interpolation=Image.BICUBIC),
    #     transforms.ToTensor(),
    #     transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # ]
    # transform = transforms.Compose(t)
    #
    # print("reading from datapath", args.data_path)
    # # root = os.path.join(args.data_path, 'val')
    # # dataset = datasets.ImageFolder(root, transform=transform)
    # # 加载CIFAR-10数据集
    # dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # # nori_root = os.path.join('/home/dingxiaohan/ndp/', 'imagenet.val.nori.list')
    # # from nori_dataset import ImageNetNoriDataset      # Data source on our machines. You will never need it.
    # # dataset = ImageNetNoriDataset(nori_root, transform=transform)
    #
    # sampler_val = torch.utils.data.SequentialSampler(dataset)
    # data_loader_val = torch.utils.data.DataLoader(dataset, sampler=sampler_val,
    #     batch_size=1, num_workers=1, pin_memory=True, drop_last=False)

    # 数据预处理 ============VOc数据集=================================================================
    # transform = transforms.Compose([
    #     transforms.Resize((1024, 1024), interpolation=Image.BICUBIC),  # 调整图片大小为1024x1024
    #     transforms.ToTensor(),  # 将图片转换为Tensor
    #     transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)  # 图像标准化
    # ])
    #
    # # 加载VOC数据集
    # # 默认情况下，VOC数据集包含训练集和测试集，你可以指定哪个子集进行加载
    # dataset = VOCDetection(root=args.data_path, year='2012', image_set='val', download=True, transform=transform)
    #
    # # 使用 DataLoader 加载数据集
    # sampler_val = torch.utils.data.SequentialSampler(dataset)
    # data_loader_val = torch.utils.data.DataLoader(dataset, sampler=sampler_val,
    #                                               batch_size=1, num_workers=1, pin_memory=True, drop_last=False)



    #=====================imagenet-val数据集===========================================
    from torch.utils.data import DataLoader, RandomSampler
    # 选择 ImageNet 数据集
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize to match ImageNet input size
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    # 加载 ImageNet 数据集
    dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform)
    data_loader_val = DataLoader(dataset, sampler=RandomSampler(dataset), pin_memory=True)

    if args.model == 'resnet18':
        model = resnet18(pretrained= False)
    elif args.model == 'resnet50':
        model = resnet50(pretrained= False)
    elif args.model == 'resnet101':
        model = resnet101(pretrained= False)
    elif args.model == 'resnet152':
        model = resnet152(pretrained= False)
    elif args.model == 'mobilemamba':
        # 实例化模型
        model = VMambaForERF(**model_params)
    else:
        raise ValueError('Unsupported model. Please add it here.')

    if args.weights is not None:
        print('load weights')
        weights = torch.load(args.weights, map_location='cpu')
        if 'model' in weights:
            weights = weights['model']
        if 'state_dict' in weights:
            weights = weights['state_dict']
        model.load_state_dict(weights)
        print('loaded')

    model.cuda()
    model.eval()    #   fix BN and droppath

    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

    meter = AverageMeter()
    optimizer.zero_grad()

    for i, (samples, _) in enumerate(data_loader_val):

        if meter.count == args.num_images:
            np.save(args.save_path, meter.avg)
            exit()

        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, samples)

        if np.isnan(np.sum(contribution_scores)):
            print('got NAN, next image')
            continue
        else:
            print(f'accumulate {i}')
            meter.update(contribution_scores)



if __name__ == '__main__':
    args = parse_args()
    main(args)