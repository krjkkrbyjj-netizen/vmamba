_base_ = [
    '../_base_/models/pgd.py',
    '../_base_/datasets/kitti_cfg.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]
model = dict(
    # type = 'FCOS3D',
    data_preprocessor=dict(
        type='DataDetPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(frozen_stages=0),
    neck=dict(start_level=0, num_outs=4),
    bbox_head=dict(
        num_classes=3,
        bbox_code_size=7,
        pred_attrs=False,
        pred_velo=False,
        pred_bbox2d=True,
        use_onlyreg_proj=True,
        strides=(4, 8, 16, 32),
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 1e8)),
        group_reg_dims=(2, 1, 3, 1, 16,
                        4),  # offset, depth, size, rot, kpts, bbox2d
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, ),  # rot
            (256, ),  # kpts
            (256, )  # bbox2d
        ),
        centerness_branch=(256, ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        use_depth_classifier=True,
        depth_branch=(256, ),
        depth_range=(0, 70),
        depth_unit=10,
        division='uniform',
        depth_bins=8,
        pred_keypoints=True,
        weight_dim=1,
        loss_depth=dict(
            type='DetUncertainSmoothL1Loss', alpha=1.0, beta=3.0,
            loss_weight=1.0),
        bbox_coder=dict(
            type='PGDDetBBoxCoder',
            base_depths=((28.01, 16.32), ),
            base_dims=((0.8, 1.73, 0.6), (1.76, 1.73, 0.6), (3.9, 1.56, 1.6)),
            code_size=7)),
    # set weight 1.0 for base 7 dims (offset, depth, size, rot)
    # 0.2 for 16-dim keypoint offsets and 1.0 for 4-dim 2D distance targets
    train_cfg=dict(code_weight=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0
    ]),
    test_cfg=dict(nms_pre=100, nms_thr=0.05, score_thr=0.001, max_per_img=20))


backend_args = None
train_pipeline = [
    dict(type='DetLoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadDetAnnotations',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize', scale=(1242, 375), keep_ratio=False),
    dict(type='DetRandomFlip', flip_ratio_bev_horizontal=0.5),
    dict(
        type='DetPackInputs',
        keys=[
            'img', 'gt_bboxes', 'gt_bboxes_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers_2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='DetLoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale_factor=1.0),
    dict(type='DetPackInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=1, num_workers=2, dataset=dict(pipeline=train_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    # optimizer=dict(
    #         _delete_=True,
    #         type='AdamW',
    #         lr=0.0001,
    #         betas=(0.9, 0.999),
    #         weight_decay=0.01),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning rate
param_scheduler= [
    # dict(
    #     type='ConstantLR',
    #     factor=0.5,  # 学习率衰减因子
    #     begin=49,  # 从第0步开始
    #     end=50,
    #     by_epoch=True,  # 按epoch更新学习率
    #     verbose=True  # 打印每次更新的学习率
    # ),
    dict(
        type='LinearLR',
        start_factor=1.0/3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=64,
        by_epoch=True,
        milestones=[32, 44],
        gamma=0.1),

    # dict(
    #     type='CosineAnnealingLR',
    #     T_max=56,
    #     eta_min=1e-9,
    #     begin=48 ,
    #     end=56,
    #     by_epoch=True)
]

train_cfg = dict(max_epochs=48, val_interval=4)
auto_scale_lr = dict(base_batch_size=12)

# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
# model = dict(
#     # type='MaskRCNN',
#     type = 'FCOS3D',
#     backbone=dict(
#         _delete_=True,
#         type='SwinTransformer',
#         embed_dims=96,
#         depths=[2, 2, 6, 2],
#         num_heads=[3, 6, 12, 24],
#         window_size=7,
#         mlp_ratio=4,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.,
#         attn_drop_rate=0.,
#         drop_path_rate=0.2,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         with_cp=False,
#         convert_weights=True,
#         init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
#     neck=dict(in_channels=[96, 192, 384, 768]))

# max_epochs = 1
# train_cfg = dict(max_epochs=max_epochs)


#================================nuscenes_train_cfg=====================================
# lr = 0.0001
# # optimizer
# optim_wrapper = dict(
#     type='OptimWrapper',
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }),
#     optimizer=dict(
#         _delete_=True,
#         type='AdamW',
#         lr=lr,
#         betas=(0.9, 0.999),
#         weight_decay=0.01))
# batch_size = 1
# # num_iter = 168780/batch_size
# num_iter = 2000/batch_size
# # learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
#         end=100),
#     # dict(
#     #     type='MultiStepLR',
#     #     begin=0,
#     #     end=max_epochs,
#     #     by_epoch=True,
#     #     milestones=[8, 11],
#     #     gamma=0.1),
#
#     #余弦退火
#     dict(
#         type='CosineAnnealingLR',
#         T_max=num_iter,
#         eta_min=lr * 1e-4,
#         begin=100 ,
#         end=num_iter,
#         by_epoch=False)
# ]
