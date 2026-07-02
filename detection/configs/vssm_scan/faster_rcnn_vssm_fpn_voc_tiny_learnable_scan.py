_base_ = [
    '../swin/faster-rcnn_swin-t-p4-w7_fpn_1x_voc.py'
]

# Note: For learnable scan with auxiliary loss, use VMambaTwoStageDetector
# to collect auxiliary losses from the backbone (Faster R-CNN is two-stage)
model = dict(
    type='VMambaTwoStageDetector',
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="",
        # copied from classification/configs/vssm/vssm_tiny_224.yaml
        dims=96,
        depths=(2, 2, 9, 2),
        ssm_d_state=16,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        mlp_ratio=0.0,
        downsample_version="v1",
        patchembed_version="v1",
        # ============ Learnable Scan Configuration ============
        scan_type="learnable_v2",   # 'learnable' 或 'learnable_v2', 'content_adaptive'
        scan_k=4,
        scan_learnable=True,
        scan_diversity_weight=0.01,   # auxiliary loss weight for diversity多样性损失权重
        scan_coverage_weight=0.01,   # auxiliary loss weight for coverage覆盖率损失权重
        use_independent_learnable=False,   # True: 每个SS2D独立参数, False: 共享参数
        scan_share_level="global",  # "stage"或 "global"
    ),
    neck=dict(in_channels=[96, 192, 384, 768])
)

# swanlab visualizer
custom_imports = dict(  # 引入SwanLab作为日志记录器，对于部分不支持custom_imports的项目可以直接初始化SwanlabVisBackend并加入vis_backends
    imports=["swanlab.integration.mmengine"], allow_failed_imports=False
)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type="SwanlabVisBackend",
        init_kwargs={ # swanlab.init 参数
            "project": "voc",
            "experiment_name": "voc_learnabel_k4_lr0.0001_adamw_bs2_epoch13",  # 实验名称
            "description": "voc数据集上的可学习扫描方法测试，还有对应的损失函数",  # 实验的描述信息
        },
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ============ Enable AMP (Mixed Precision Training) ============
# optim_wrapper = dict(type='AmpOptimWrapper')

# train_dataloader = dict(batch_size=2) # as gpus=8


# - scan_diversity_weight: 建议范围0.001 ~ 0.1
# - 太小: 扫描方向会趋同
# - 太大: 可能干扰主任务训练
# - scan_coverage_weight: 建议范围0.001~ 0.1
# - 太小: 扫描分布不均匀
# - 太大: 可能过于平滑, 失去适应性