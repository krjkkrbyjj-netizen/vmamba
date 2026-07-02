_base_ = [
    '../swin/faster-rcnn_swin-t-p4-w7_fpn_1x_voc.py'
]

model = dict(
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
        # ============ Blockwise Scan Configuration ============
        scan_type="blockwise",
        scan_k=4,
        scan_learnable=False,
    ),
    neck=dict(in_channels=[96, 192, 384, 768])
)

# train_dataloader = dict(batch_size=2) # as gpus=8
