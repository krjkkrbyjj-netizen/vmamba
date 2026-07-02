from mmdet3d.datasets.transforms.loading import *
from mmdet3d.datasets.transforms.transforms_3d import *
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.models.losses.uncertain_smooth_l1_loss import  UncertainSmoothL1Loss

from mmdet.registry import DATASETS as MMDETDATASETS
from mmdet.registry import TRANSFORMS
from mmdet.registry import MODELS

@MMDETDATASETS.register_module()
@TRANSFORMS.register_module()
class DetLoadImageFromFile(LoadImageFromFileMono3D):
    pass

@TRANSFORMS.register_module()
class LoadDetAnnotations(LoadAnnotations3D):
    pass

@TRANSFORMS.register_module()
class DetRandomFlip(RandomFlip3D):
    pass

@TRANSFORMS.register_module()
class DetPackInputs(Pack3DDetInputs):
    pass

@MODELS.register_module()
class DataDetPreprocessor(Det3DDataPreprocessor):
    pass

@MODELS.register_module()
class DetUncertainSmoothL1Loss(UncertainSmoothL1Loss):
    pass

from mmengine.runner import Runner
from mmengine.runner import checkpoint

class Runner_new(Runner):
    def load_or_resume(self) -> None:
        """load or resume checkpoint."""
        if self._has_loaded:
            return None

        # decide to load from checkpoint or resume from checkpoint
        resume_from = None
        if self._resume and self._load_from is None:
            # auto resume from the latest checkpoint
            resume_from = checkpoint.find_latest_checkpoint(self.work_dir)
            self.logger.info(
                f'Auto resumed from the latest checkpoint {resume_from}.')
        elif self._resume and self._load_from is not None:
            # resume from the specified checkpoint
            resume_from = self._load_from

        if resume_from is not None:
            self.resume(resume_from, resume_optimizer=False,resume_param_scheduler=False)
            self._has_loaded = True
        elif self._load_from is not None:
            self.load_checkpoint(self._load_from)
            self._has_loaded = True



