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

# 添加 detection 文件夹到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backbones.mobilemamba import MobileMamba


class MobileMambaForERF(MobileMamba):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        return x