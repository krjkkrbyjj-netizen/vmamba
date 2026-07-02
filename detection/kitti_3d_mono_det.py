from mmdet3d.datasets.kitti_dataset import KittiDataset
from mmdet.registry import DATASETS as MMDETDATASETS
@MMDETDATASETS.register_module()
class Kitti3DDetDataset(KittiDataset):
    pass