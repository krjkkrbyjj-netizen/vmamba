# from mmdet3d.configs._base_.models.fcos3d import FCOSMono3D
from mmdet3d.models.detectors.fcos_mono3d import FCOSMono3D
from mmdet3d.models.data_preprocessors.data_preprocessor import \
    Det3DDataPreprocessor
from mmdet3d.models.dense_heads.fcos_mono3d_head import FCOSMono3DHead
from mmdet3d.models.detectors.fcos_mono3d import FCOSMono3D
from mmdet3d.models.task_modules.coders.fcos3d_bbox_coder import \
    FCOS3DBBoxCoder
from mmdet.registry import MODELS
from mmdet.registry import TASK_UTILS

from mmdet3d.models.dense_heads.pgd_head import PGDHead
from mmdet3d.models.task_modules.coders.pgd_bbox_coder import PGDBBoxCoder

#===================================fcos3d================================================
@MODELS.register_module()
class FCOS3D(FCOSMono3D):
    pass

@MODELS.register_module()
class FCOS3DHead(FCOSMono3DHead):
    pass

@MODELS.register_module()
@TASK_UTILS.register_module()
class FCOSDetBBoxCoder(FCOS3DBBoxCoder):
    pass


#==================================pgd=====================================================
@MODELS.register_module()
class PGDDetHead(PGDHead):
    pass

@MODELS.register_module()
@TASK_UTILS.register_module()
class PGDDetBBoxCoder(PGDBBoxCoder):
    pass
