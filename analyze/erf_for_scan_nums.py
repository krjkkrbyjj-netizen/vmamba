# this is only a script !
if __name__ == "__main__":
    import os
    from utils import visualize, EffectiveReceiptiveField, BuildModels

    simpnorm = EffectiveReceiptiveField.simpnorm

    HOME = os.environ["HOME"].rstrip("/")

    showpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./show").rstrip("/")
    # Ensure the directory exists
    os.makedirs(showpath, exist_ok=True)  # This will create the directory if it doesn't exist
    # data_path = "/media/Disk1/Dataset/ImageNet_ILSVRC2012"
    data_path = "/root/autodl-tmp/data"
    # data_path = "/root/autodl-tmp/data/imagenet"
    cfg = {
        'out_indices': (0, 1, 2, 3),
        'dims': 96,
        'depths': (2, 2, 9, 2),
        'ssm_d_state': 16,
        'ssm_dt_rank': "auto",
        'ssm_ratio': 2.0,
        'mlp_ratio': 0.0,
        'downsample_version': "v1",
        'patchembed_version': "v1",
    }
    ckpt_1_1 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250401_003808_1_plainmamba(1-1)_new/epoch_12.pth"
    ckpt_1_2 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250324_193905_1_plainmamba(1.2)/epoch_12.pth"
    ckpt_1_3 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250325_104940_1_plainmamba(1.3)/epoch_12.pth"
    ckpt_1_4 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250325_224843_1_plainmamba(1.4)/epoch_12.pth"
    ckpt_1_5 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250327_005555_1_plainmamba(1.5)/epoch_12.pth"
    ckpt_1_6 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250327_152742_1_plainmamba(1.6)/epoch_12.pth"
    ckpt_1_7 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250328_113039_1_plainmamba(1.7)/epoch_12.pth"
    ckpt_1_8 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250329_000422_1_plainmamba(1.8)/epoch_12.pth"
    ckpt_2 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250401_163826_1_plainmamba(1-2)_new/epoch_12.pth"
    ckpt_3 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250402_113034_1_plainmamba(1-3)_new/epoch_12.pth"
    ckpt_4 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250403_084730_1_plainmamba(1-4)_new/epoch_12.pth"
    ckpt_5 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250404_163605_1_plainmamba(1-5)_new/epoch_12.pth"
    ckpt_6 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250405_181719_1_plainmamba(1-6)_new/epoch_12.pth"
    ckpt_7 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250406_211019_1_plainmamba(1-7)_new/epoch_12.pth"
    ckpt_8 = "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250408_010721_1_plainmamba(1-8)_new/epoch_12.pth"

    # main results
    if False:
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", cfg=cfg, ckpt=ckpt_1_1),
                                                         size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        results_after = [
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex", cfg=cfg, ckpt=ckpt_1_1), size=1024,
                data_path=data_path, norms=simpnorm), ""),

        ]
        visualize.visualize_snsmaps(results_before + results_after, savefig=f"{showpath}/erf_main.jpg", rows=2,
                                    sticks=False, figsize=(10, 10.75), cmap='RdYlGn')

    if False:
        # 定义所有的 checkpoint 文件路径
        ckpt_files = [
            "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250401_003808_1_plainmamba(1-1)_new/epoch_12.pth",
            "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250324_193905_1_plainmamba(1.2)/epoch_12.pth",
            "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250325_104940_1_plainmamba(1.3)/epoch_12.pth",
            "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250325_224843_1_plainmamba(1.4)/epoch_12.pth",
            "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250327_005555_1_plainmamba(1.5)/epoch_12.pth",
            "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250327_152742_1_plainmamba(1.6)/epoch_12.pth",
            "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250328_113039_1_plainmamba(1.7)/epoch_12.pth",
            "/root/autodl-tmp/vmamba/detection/work_dirs/faster_rcnn_vssm_fpn_voc_tiny/20250329_000422_1_plainmamba(1.8)/epoch_12.pth"
        ]

        # 使用 for 循环来进行推理并存储结果
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(only_backbone=True, scale="flex", cfg=cfg, ckpt=ckpt),
                size=1024, data_path=data_path, norms=simpnorm), "")
            for ckpt in ckpt_files
        ]

        results_after = [
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex", cfg=cfg, ckpt=ckpt),
                size=1024, data_path=data_path, norms=simpnorm), "")
            for ckpt in ckpt_files
        ]

        # 合并所有结果
        all_results = results_before + results_after

        # 绘制所有结果
        visualize.visualize_snsmaps(all_results, savefig=f"{showpath}/erf_main_(a-h).jpg", rows=2,
                                    sticks=False, figsize=(10, 10.75), cmap='RdYlGn')

    # import pickle
    # # 运行推理
    # results_before1 = [
    #     (EffectiveReceiptiveField.get_input_grad_avg(
    #         BuildModels.build_vmamba(only_backbone=True, scale="flex", cfg=cfg, ckpt=ckpt_8),
    #         size=1024, data_path=data_path, norms=simpnorm), "Before")
    # ]
    # results_after1 = [
    #     (EffectiveReceiptiveField.get_input_grad_avg(
    #         BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex", cfg=cfg, ckpt=ckpt_8),
    #         size=1024, data_path=data_path, norms=simpnorm), "After")
    # ]
    # # 保存第一次的结果
    # with open(f"{showpath}/results_before_voc8.pkl", "wb") as f:
    #     pickle.dump(results_before1, f)
    #
    # with open(f"{showpath}/results_after_voc8.pkl", "wb") as f:
    #     pickle.dump(results_after1, f)

    import pickle
    import matplotlib.pyplot as plt
    from utils import visualize
    # 定义文件后缀列表
    result_suffixes = ["1", "2", "3", "4", "5", "6", "7", "8"]  # 你可以根据实际需求修改这个列表
    # 用于存储所有结果的列表
    all_results_before = []
    all_results_after = []
    # 循环加载不同的结果文件并拼接
    for suffix in result_suffixes:
        # 加载每个结果文件
        with open(f"{showpath}/results_before_voc{suffix}.pkl", "rb") as f:
            results_before = pickle.load(f)
        with open(f"{showpath}/results_after_voc{suffix}.pkl", "rb") as f:
            results_after = pickle.load(f)

        # 按照顺序合并 results_before 和 results_after
        all_results_before.extend(results_before)  # 使用 extend 来合并列表
        all_results_after.extend(results_after)

    # 合并 all_results_before 和 all_results_after
    all_results = all_results_before + all_results_after
    # 绘制所有结果
    visualize.visualize_snsmaps(all_results, savefig=f"{showpath}/erf_main_nums.jpg", rows=2,
                                sticks=False, figsize=(10, 10.75), cmap='RdYlGn')

    # import pickle
    # import matplotlib.pyplot as plt
    # from utils import visualize
    # # 定义文件后缀列表
    # result_suffixes = ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8"]  # 你可以根据实际需求修改这个列表
    # # 用于存储所有结果的列表
    # all_results_before = []
    # all_results_after = []
    # # 循环加载不同的结果文件并拼接
    # for suffix in result_suffixes:
    #     # 加载每个结果文件
    #     with open(f"{showpath}/results_before_voc{suffix}.pkl", "rb") as f:
    #         results_before = pickle.load(f)
    #     with open(f"{showpath}/results_after_voc{suffix}.pkl", "rb") as f:
    #         results_after = pickle.load(f)
    #
    #     # 按照顺序合并 results_before 和 results_after
    #     all_results_before.extend(results_before)  # 使用 extend 来合并列表
    #     all_results_after.extend(results_after)
    #
    # # 合并 all_results_before 和 all_results_after
    # all_results = all_results_before + all_results_after
    # # 绘制所有结果
    # visualize.visualize_snsmaps(all_results, savefig=f"{showpath}/erf_main_(a-h).jpg", rows=2,
    #                             sticks=False, figsize=(10, 10.75), cmap='RdYlGn')


    # main results
    if False:
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_resnet_mmpretrain(only_backbone=True, scale="r50"), size=1024, data_path=data_path,
                norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_convnext(only_backbone=True, scale="tiny"),
                                                         size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_swin_mmpretrain(only_backbone=True, scale="tiny"), size=1024, data_path=data_path,
                norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_deit_mmpretrain(only_backbone=True, scale="small"), size=1024, data_path=data_path,
                norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_hivit_mmpretrain(only_backbone=True, scale="tiny", size=1024), size=1024,
                data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="tv2"),
                                                         size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        results_after = [
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_resnet_mmpretrain(with_ckpt=True, only_backbone=True, scale="r50"), size=1024,
                data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_convnext(with_ckpt=True, only_backbone=True, scale="tiny"), size=1024,
                data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_swin_mmpretrain(with_ckpt=True, only_backbone=True, scale="tiny"), size=1024,
                data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_deit_mmpretrain(with_ckpt=True, only_backbone=True, scale="small"), size=1024,
                data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_hivit_mmpretrain(with_ckpt=True, only_backbone=True, scale="tiny", size=1024),
                size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="tv2"), size=1024,
                data_path=data_path, norms=simpnorm), ""),
        ]
        visualize.visualize_snsmaps(results_before + results_after, savefig=f"{showpath}/erf_main.jpg", rows=2,
                                    sticks=False, figsize=(10, 10.75), cmap='RdYlGn')

    # erf for tv2, initv1, initv2;
    if False:
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="tv2"),
                                                         size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=3,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v05_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d",
                                                                                           ssm_init="v1"),
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=3,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v05_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d",
                                                                                           ssm_init="v2"),
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
        ]
        results_after = [
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="tv2"), size=1024,
                data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex",
                                         cfg=dict(dims=96, depths=[2, 2, 8, 2], ssm_d_state=1, ssm_dt_rank="auto",
                                                  ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False,
                                                  forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3",
                                                  patchembed_version="v2", norm_layer="ln2d", ssm_init="v1"),
                                         ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/classification/vssms/vssm1_tiny_0230s_init1/ckpt_epoch_261.pth",
                                         key="model_ema",
                                         ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex",
                                         cfg=dict(dims=96, depths=[2, 2, 8, 2], ssm_d_state=1, ssm_dt_rank="auto",
                                                  ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False,
                                                  forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3",
                                                  patchembed_version="v2", norm_layer="ln2d", ssm_init="v2"),
                                         ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/classification/vssms/vssm1_tiny_0230s_init2/ckpt_epoch_264.pth",
                                         key="model_ema",
                                         ), size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        visualize.visualize_snsmaps(results_before + results_after, savefig=f"{showpath}/erf_initv0v1v2.jpg", rows=2,
                                    sticks=False, figsize=(10, 10.75), cmap='RdYlGn')

    # erf for tv2, unidi, bidi, cas2d; ndw, unidindw, bidindw, cas2dndw;
    if False:
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="tv2"),
                                                         size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=3,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v051d_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=3,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v052d_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=3,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v052dc_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),

            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=-1,
                                                                                           ssm_conv_bias=True,
                                                                                           forward_type="v05_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=-1,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v051d_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=-1,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v052d_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=-1,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v052dc_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
        ]
        results_after = [
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="tv2"), size=1024,
                data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=3,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v051d_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230ab1d/ckpt_epoch_272.pth",
                                                                                  key="model_ema",
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=3,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v052d_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230ab2d/ckpt_epoch_269.pth",
                                                                                  key="model_ema",
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=3,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v052dc_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230ab2dc/ckpt_epoch_250.pth",
                                                                                  key="model_ema",
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),

            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=-1,
                                                                                           ssm_conv_bias=True,
                                                                                           forward_type="v05_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230s_ndw/ckpt_epoch_237.pth",
                                                                                  key="model_ema",
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=-1,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v051d_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230ab1d_ndw/ckpt_epoch_272.pth",
                                                                                  key="model_ema",
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=-1,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v052d_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230ab2d_ndw/ckpt_epoch_268.pth",
                                                                                  key="model_ema",
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex",
                                                                                  cfg=dict(dims=96, depths=[2, 2, 8, 2],
                                                                                           ssm_d_state=1,
                                                                                           ssm_dt_rank="auto",
                                                                                           ssm_ratio=1.0, ssm_conv=-1,
                                                                                           ssm_conv_bias=False,
                                                                                           forward_type="v052dc_noz",
                                                                                           mlp_ratio=4.0,
                                                                                           downsample_version="v3",
                                                                                           patchembed_version="v2",
                                                                                           norm_layer="ln2d"),
                                                                                  ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230ab2dc_ndw/ckpt_epoch_263.pth",
                                                                                  key="model_ema",
                                                                                  ), size=1024, data_path=data_path,
                                                         norms=simpnorm), ""),
        ]
        visualize.visualize_snsmaps(results_before + results_after, savefig=f"{showpath}/erf_scanmethod.jpg", rows=2,
                                    sticks=False, figsize=(10, 10.75), cmap='RdYlGn')

    # erf for training stage for tv2
    if False:
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="tv2"),
                                                         size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex",
                                         cfg=dict(dims=96, depths=[2, 2, 8, 2], ssm_d_state=1, ssm_dt_rank="auto",
                                                  ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False,
                                                  forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3",
                                                  patchembed_version="v2", norm_layer="ln2d"),
                                         ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/vssm1_tiny_0230s/20240426174619/ckpt_epoch_2.pth",
                                         key="model_ema",
                                         ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex",
                                         cfg=dict(dims=96, depths=[2, 2, 8, 2], ssm_d_state=1, ssm_dt_rank="auto",
                                                  ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False,
                                                  forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3",
                                                  patchembed_version="v2", norm_layer="ln2d"),
                                         ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/vssm1_tiny_0230s/20240426174619/ckpt_epoch_5.pth",
                                         key="model_ema",
                                         ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex",
                                         cfg=dict(dims=96, depths=[2, 2, 8, 2], ssm_d_state=1, ssm_dt_rank="auto",
                                                  ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False,
                                                  forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3",
                                                  patchembed_version="v2", norm_layer="ln2d"),
                                         ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/vssm1_tiny_0230s/20240426174619/ckpt_epoch_8.pth",
                                         key="model_ema",
                                         ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex",
                                         cfg=dict(dims=96, depths=[2, 2, 8, 2], ssm_d_state=1, ssm_dt_rank="auto",
                                                  ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False,
                                                  forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3",
                                                  patchembed_version="v2", norm_layer="ln2d"),
                                         ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/vssm1_tiny_0230s/20240426174619/ckpt_epoch_11.pth",
                                         key="model_ema",
                                         ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex",
                                         cfg=dict(dims=96, depths=[2, 2, 8, 2], ssm_d_state=1, ssm_dt_rank="auto",
                                                  ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False,
                                                  forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3",
                                                  patchembed_version="v2", norm_layer="ln2d"),
                                         ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/vssm1_tiny_0230s/20240426174619/ckpt_epoch_29.pth",
                                         key="model_ema",
                                         ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="tv2"), size=1024,
                data_path=data_path, norms=simpnorm), ""),
        ]
        visualize.visualize_snsmaps(results_before, savefig=f"{showpath}/erf_trainprocess.jpg", rows=1, sticks=False,
                                    figsize=(10, 10.75), cmap='RdYlGn')

    # erf for vim
    if False:
        from analyze_for_vim import ExtraDev

        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(
                ExtraDev.build_vim_for_throughput(only_backbone=True, size=1024), size=1024, data_path=data_path,
                norms=simpnorm), ""),
        ]
        results_after = [
            (EffectiveReceiptiveField.get_input_grad_avg(
                ExtraDev.build_vim_for_throughput(with_ckpt=True, only_backbone=True, size=1024), size=1024,
                data_path=data_path, norms=simpnorm), ""),
        ]
        visualize.visualize_snsmaps(results_before + results_after, savefig=f"{showpath}/erf_vimmethods.jpg", rows=2,
                                    sticks=False, figsize=(10, 10.75), cmap='RdYlGn')

    # erf for s4nd
    if False:
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="tv2"),
                                                         size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_s4nd(only_backbone=True, scale="ctiny"),
                                                         size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        results_after = [
            (EffectiveReceiptiveField.get_input_grad_avg(
                BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="tv2"), size=1024,
                data_path=data_path, norms=simpnorm), ""),
            # (after_s4nd, ""),
        ]
        visualize.visualize_snsmaps(results_before + results_after, savefig=f"{showpath}/erf_s4ndmethods.jpg", rows=2,
                                    sticks=False, figsize=(10, 10.75), cmap='RdYlGn')




