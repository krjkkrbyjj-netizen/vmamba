from mmdet3d.registry import METRICS as TREEDMETRICS
from mmdet.registry import METRICS
from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric


@METRICS.register_module()
@TREEDMETRICS.register_module()
class KittiDetMetric(KittiMetric):
    pass