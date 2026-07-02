_base_ = [
    '../swin/faster-rcnn_swin-t-p4-w7_fpn_1x_visdrone.py'
]

model = dict(
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="/root/autodl-tmp/vmamba/detection/pretrain/vssmtiny_dp01_ckpt_epoch_292.pth",#"../vssm1_tiny_0230s_ckpt_epoch_264.pth",
        # copied from classification/configs/vssm/vssm_tiny_224.yaml
        dims=96,
        depths=(2, 2, 9, 2),
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
        # forward_type="v0", # if you want exactly the same
        # ============ Cross Scan Configuration (Default) ============
        scan_type="permute",  # cross, permute
        scan_k=4,
        scan_learnable=False,
    ),
    neck=dict(in_channels=[96, 192, 384, 768])
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
)

# train_dataloader = dict(batch_size=2) # as gpus=8

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# swanlab visualizer
custom_imports = dict(  # 引入SwanLab作为日志记录器，对于部分不支持custom_imports的项目可以直接初始化SwanlabVisBackend并加入vis_backends
    imports=["swanlab.integration.mmengine"], allow_failed_imports=False
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="SwanlabVisBackend",
        init_kwargs={  # swanlab.init 参数
            "project": "mamba_visdrone",
            "experiment_name": "permutek4_bs2_sgd0.02_pretrain",  # 实验名称
            "description": "permute的不同san_k实验，当前是4",  # 实验的描述信息
        },
    ),
]

visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

# 设置工作目录
work_dir = './work_dirs/visdrone/faster_rcnn_vssm_fpn_visdrone_tiny_cross_scan/permutek4_bs2_sgd0.02_pretrain'

