import torch
import numpy as np
def sort_patches_by_offset(x:torch.Tensor, offset_map:torch.Tensor):
    """
    根据给定的偏移量图对patches进行排序。
    参数:
        x: 输入特征图 (B, C, H, W)
        offset_map: 偏移量图 (B, 1, H, W)，用于决定patch的排序顺序

    返回:
        sorted_patches: 排序后的patches (B, C, H, W) 保持原空间结构但内部patches按偏移量排序
    """
    B, C, H, W = x.shape
    # 确保offset_map和x有相同的batch size、height和width
    assert x.size(0) == offset_map.size(0) and x.size(2) == offset_map.size(2) and x.size(3) == offset_map.size(
        3), "尺寸不匹配"
    # 展开offset_map以便于排序
    offset_values = offset_map.view(B, -1)
    # 获取排序索引
    _, indices = torch.sort(offset_values, dim=1, descending=True)
    # 扩展indices以匹配x的形状
    indices = indices.unsqueeze(1).expand(-1, C, -1)
    # 根据计算得到的索引对patches进行排序
    sorted_patches = torch.gather(x.view(B, C, -1), dim=2, index=indices).view(B, C, H, W)

    return sorted_patches, indices


def generate_sequential_map(x:torch.Tensor):
    """
    根据输入的特征图 x 生成一个大小相同但内容为从左到右、从上到下依次递增的整数的映射图。
    参数:
        x: 输入的特征图 (B, C, H, W)
    返回:
        sequential_map: 形状与 x 相同的映射图，但是内容是从 1 开始的连续整数
    """
    B, C, H, W = x.shape
    # 创建一个从 1 开始的连续整数序列
    sequential_array = torch.arange(start=1, end=H * W + 1).view(1, 1, H, W)
    # 如果输入的批次大小大于 1，则需要复制该序列以匹配批次大小
    sequential_map = sequential_array.repeat(B, C, 1, 1)

    return sequential_map


def restore_original_order(sorted_patches, indices):
    """
    根据排序前的索引 indices 恢复 sorted_patches 到原始顺序。

    参数:
        sorted_patches: 排序后的 patches (B, C, H, W)
        indices: 排序时用到的索引 (B, C, H*W)

    返回:
        restored_patches: 恢复到原始顺序的 patches (B, C, H, W)
    """
    B, C, H, W = sorted_patches.shape


    # 将 indices 反转以恢复原始顺序
    _, original_indices = torch.sort(indices, dim=2)  # (B, C, H*W)

    # 扩展索引到所有通道
    original_indices = original_indices.expand(-1, C, -1)  # (B, C, H*W)

    # 将 sorted_patches 展平后 gather 恢复原始顺序
    sorted_patches_flat = sorted_patches.view(B, C, -1)  # (B, C, H*W)
    restored_patches_flat = torch.gather(sorted_patches_flat, dim=2, index=original_indices)  # (B, C, H*W)

    # 恢复成图像格式
    restored_patches = restored_patches_flat.view(B, C, H, W)

    return restored_patches
# 示例使用
if __name__ =="__main__":
    x: torch.Tensor = torch.randn(1, 1, 3, 4)  # 假设输入为2个batch，3通道，4x4大小的patch
    offset_map: torch.Tensor = torch.rand(1, 1, 3, 4)  # 学习得到的偏移量图

    sorted_patches, indices = sort_patches_by_offset(x, offset_map)
    print(indices.shape)

    # restort_patches = torch.gather(sorted_patches.view(1,1,-1), dim=2, index=indices).view(1, 1, 3, 4)
    restort_patches = restore_original_order(sorted_patches, indices)



    print("sorted before patches:\n", x)
    print("offset_map:\n", offset_map)
    print("indices:\n", indices)
    print("Sorted patches:\n", sorted_patches)

    print("restort patches:\n", restort_patches)

    # sequential_map = generate_sequential_map(x)
    # print("Sequential map:\n", sequential_map)