import os
from functools import partial
import torch

from .vmamba import VSSM
from .vmamba import Backbone_VSSM
from .scan_order import ScanOrderGenerator
# 从 vmamba 导入新的 aux_loss_cache（用于内联可学习扫描）
from .vmamba import aux_loss_cache
# 注释掉旧的导入，避免冲突
# from .learnable_scan import LearnableScanOrder, LearnableScanOrderV2, aux_loss_cache


def build_vssm_model(config, **kwargs):
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        model = VSSM(
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            # ===================
            posembed=config.MODEL.VSSM.POSEMBED,
            imgsize=config.DATA.IMG_SIZE,
            # ===================
            # scan configuration (optional, with defaults)
            scan_type=getattr(config.MODEL.VSSM, 'SCAN_TYPE', 'cross'),
            scan_k=getattr(config.MODEL.VSSM, 'SCAN_K', 4),
            scan_learnable=getattr(config.MODEL.VSSM, 'SCAN_LEARNABLE', False),
            scan_diversity_weight=getattr(config.MODEL.VSSM, 'SCAN_DIVERSITY_WEIGHT', 0.0),
            scan_coverage_weight=getattr(config.MODEL.VSSM, 'SCAN_COVERAGE_WEIGHT', 0.0),
        )
        return model

    return None


def build_model(config, is_pretrain=False):
    model = None
    if model is None:
        model = build_vssm_model(config)
    if model is None:
        from .simvmamba import simple_build
        model = simple_build(config.MODEL.TYPE)
    return model




