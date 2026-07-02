# A script to visualize the ERF.
import sys
import os

# # # 获取当前脚本所在的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # # 获取上一级目录（即 project 目录）
# root_dir = os.path.dirname(os.path.dirname(current_dir))
# #
# # # 将项目根目录添加到 Python 搜索路径中
# if root_dir not in sys.path:
#     sys.path.append(root_dir)
# print(sys.path)
# 获取项目的根目录
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)
# from .backbones.mobilemamba import MobileMamba
import sys
import os
import  torch.nn as nn
# 添加 detection 文件夹到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../classification/"),
)
Backbone_VSSM: nn.Module = build.vmamba.Backbone_VSSM


class VMambaForERF(Backbone_VSSM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x

        return outs