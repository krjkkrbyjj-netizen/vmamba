import os
from functools import partial
from typing import Callable

import torch
from torch import nn
from torch.utils import checkpoint

from mmengine.model import BaseModule
from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG


def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

build = import_abspy(
    "models",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)
Backbone_VSSM: nn.Module = build.vmamba.Backbone_VSSM

# 导入辅助损失缓存（用于收集可学习扫描的损失）
# 必须在 import_abspy 之后导入，因为需要先将 classification 加入路径
aux_loss_cache = build.aux_loss_cache

@MODELS_MMSEG.register_module()
@MODELS_MMDET.register_module()
class MM_VSSM(BaseModule, Backbone_VSSM):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_VSSM.__init__(self, *args, **kwargs)


# ==================== 支持可学习扫描辅助损失的 Detector ====================

from mmdet.models.detectors import SingleStageDetector, TwoStageDetector


@MODELS_MMDET.register_module()
class VMambaSingleStageDetector(SingleStageDetector):
    """
    支持 VMamba backbone 辅助损失的单阶段检测器

    当 backbone 使用可学习扫描并配置了辅助损失权重时，
    会自动收集并添加到总损失中

    Config 示例:
        model = dict(
            type='VMambaSingleStageDetector',
            backbone=dict(
                type='MM_VSSM',
                ...
                # 在 backbone 中配置可学习扫描的辅助损失权重
            ),
            ...
        )
    """

    def loss(self, batch_inputs, batch_data_samples):
        """重写 loss 方法，收集 backbone 中的辅助损失"""
        x = self.extract_feat(batch_inputs)

        # 在 extract_feat 之后立即收集辅助损失（避免后续操作清空缓存）
        scan_aux_losses = aux_loss_cache.collect()

        losses = self.bbox_head.loss(x, batch_data_samples)

        # 添加辅助损失到总损失
        if scan_aux_losses:
            losses.update(scan_aux_losses)


        return losses


@MODELS_MMDET.register_module()
class VMambaTwoStageDetector(TwoStageDetector):
    """
    支持 VMamba backbone 辅助损失的两阶段检测器 (如 Mask R-CNN)

    Config 示例:
        model = dict(
            type='VMambaTwoStageDetector',
            backbone=dict(type='MM_VSSM', ...),
            ...
        )
    """

    def loss(self, batch_inputs, batch_data_samples):
        """重写 loss 方法，收集 backbone 中的辅助损失"""

        x = self.extract_feat(batch_inputs)


        # 在 extract_feat 之后立即收集辅助损失（避免后续操作清空缓存）
        scan_aux_losses = aux_loss_cache.collect()

        losses = dict()

        # RPN 损失
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_data_samples = batch_data_samples
            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            for data_sample, rpn_results in zip(batch_data_samples, rpn_results_list):
                data_sample.proposals = rpn_results
            losses.update(rpn_losses)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # ROI 损失
        roi_losses = self.roi_head.loss(x, rpn_results_list, batch_data_samples)
        losses.update(roi_losses)

        # 添加辅助损失到总损失
        if scan_aux_losses:
            losses.update(scan_aux_losses)

        return losses


# def import_abspy(name="backbones", path="detection/"):
#     import sys
#     import importlib
#     path = os.path.abspath(path)
#     assert os.path.isdir(path)
#     sys.path.insert(0, path)
#     module = importlib.import_module(name)
#     sys.path.pop(0)
#     return module
#
# build = import_abspy(
#     "backbones",
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), "../detection/"),
# )
# MobileMamba: nn.Module = build.mobilemamba.MobileMamba
#
# @MODELS_MMSEG.register_module()
# @MODELS_MMDET.register_module()
# class MB_MAMBA(BaseModule, MobileMamba):
#     def __init__(self, *args, **kwargs):
#         BaseModule.__init__(self)
#         MobileMamba.__init__(self, *args, **kwargs)