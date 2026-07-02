import os
import time
import json
from os import path as osp
from typing import Dict, List, Optional, Union

import mmengine
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from nuscenes.eval.detection.config import config_factory
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval


from mmdet3d.registry import METRICS as TREEDMETRICS
from mmdet.registry import METRICS
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric


@METRICS.register_module()
@TREEDMETRICS.register_module()
class NuScenesDetMetric(NuScenesMetric):
    pass