import torch
import warnings

WITH_TRITON = True  # 启用 Triton 加速（如果可用）
# WITH_TRITON = False
try:
    import triton
    import triton.language as tl
except:
    WITH_TRITON = False
    warnings.warn("Triton not installed, fall back to pytorch implements.")

# to make sure cached_property can be loaded for triton
if WITH_TRITON:
    try:
        from functools import cached_property
    except:
        warnings.warn("if you are using py37, add this line to functools.py: "
                      "cached_property = lambda func: property(lru_cache()(func))")


# =============================================================================================================================================
# 根据顺序展开图像 (支持4D Tensor, GPU 加速)
def flatten_by_order(image, indices):
    # 假设输入的 image 是四维 (B, C, H, W) 张量
    B, C, H, W = image.shape

    # indices 已经是张量，且是在合适的设备上 (len(order), 2)
    i_indices = indices[:, 0]
    j_indices = indices[:, 1]

    # 扩展批次和通道的维度，得到 (B, C, len(order))
    flattened = image[:, :, i_indices, j_indices]  # (B, C, len(order))

    return flattened


# 将一维序列还原为二维图形 (支持4D Tensor, GPU 加速)
def restore_from_flattened(hw_shape, flattened, indices):
    B, C = flattened.shape[0], flattened.shape[1]
    H, W = hw_shape[-2], hw_shape[-1]

    # 创建一个全零的张量
    restored_image = torch.zeros((B, C, H, W), dtype=flattened.dtype, device=flattened.device)

    # 使用索引方式直接赋值 (B, C, len(order)) 映射到 (B, C, H, W)
    i_indices = indices[:, 0]
    j_indices = indices[:, 1]
    restored_image[:, :, i_indices, j_indices] = flattened

    return restored_image


# torch implementation ========================================
# 逆索引缓存：避免重复计算 torch.sort
_inverse_indices_cache = {}


def _get_inverse_indices(indices: torch.Tensor) -> torch.Tensor:
    """
    计算并缓存逆索引。对于固定扫描顺序，逆索引只需计算一次。
    indices: (K, L) - 扫描索引
    返回: (K, L) - 逆索引
    """
    # 使用 (device, shape, data_ptr) 作为缓存 key
    cache_key = (indices.device, indices.shape, indices.data_ptr())
    if cache_key in _inverse_indices_cache:
        return _inverse_indices_cache[cache_key]

    # 计算逆索引
    _, inverse = torch.sort(indices, dim=-1)
    _inverse_indices_cache[cache_key] = inverse

    # 限制缓存大小，防止内存泄漏
    if len(_inverse_indices_cache) > 100:
        # 删除最老的一半
        keys_to_remove = list(_inverse_indices_cache.keys())[:50]
        for key in keys_to_remove:
            del _inverse_indices_cache[key]

    return inverse


# def cross_scan_fwd(x: torch.Tensor, indices=None, in_channel_first=True, out_channel_first=True, scans=0):
#     if in_channel_first:
#         B, C, H, W = x.shape
#         L = H * W
#         # flat_x = x.view(B, C, L)  # (B, C, H*W)
#
#         # y=x.new_zeros((B,4,C,H*W))
#         # y[:, 0, :, :] = flatten_by_order(x, indices[0])
#         # y[:, 1, :, :] = flatten_by_order(x, indices[1])
#         # y[:, 2, :, :] = flatten_by_order(x, indices[2])
#         # y[:, 3, :, :] = flatten_by_order(x, indices[3])
#
#         # 检查 indices 格式
#         if indices.dim() == 3 and indices.shape[-1] == 2:
#             # (K, L, 2) 格式 - 2D 坐标格式 (来自 get_permute_order)
#             K = indices.shape[0]
#             y = x.new_zeros((B, K, C, L))
#             for k in range(K):
#                 i_indices = indices[k, :, 0]  # (L,)
#                 j_indices = indices[k, :, 1]  # (L,)
#                 y[:, k, :, :] = x[:, :, i_indices, j_indices]  # (B, C, L)
#         elif indices.dim() == 2:
#             # (K, L) 格式 - 1D 索引格式 (来自 ScanOrderGenerator)
#             K = indices.shape[0]
#             flat_x = x.view(B, C, L)
#             y = flat_x[:, :, indices]  # (B, C, K, L)
#             y = y.permute(0, 2, 1, 3)  # (B, K, C, L)
#         else:
#             # (B, K, L) 格式 - 每个 batch 可能有不同索引
#             K = indices.shape[1]
#             flat_x = x.view(B, C, L)
#             indices_expanded = indices.unsqueeze(2).expand(-1, -1, C, -1)  # (B, K, C, L)
#             flat_x_expanded = flat_x.unsqueeze(1).expand(B, K, C, L)
#             y = torch.gather(flat_x_expanded, dim=3, index=indices_expanded)  # (B, K, C, L)
#
#     if in_channel_first and (not out_channel_first):
#         y = y.permute(0, 3, 1, 2).contiguous()
#     elif (not in_channel_first) and out_channel_first:
#         y = y.permute(0, 2, 3, 1).contiguous()
#
#     return y
# ========== 优化版本 cross_scan_fwd ==========
def cross_scan_fwd(x: torch.Tensor, indices=None, in_channel_first=True, out_channel_first=True, scans=0):
    """
    优化版 cross_scan_fwd

    优化点：
    1. 消除 flatten_by_order 函数调用开销（内联处理）
    2. 减少重复的索引计算
    3. 支持动态 K 值（不再硬编码为4）
    """
    # 添加调试信息
    # print(f"[DEBUG] x shape: {x.shape}")
    # print(f"[DEBUG] indices shape: {indices.shape}")
    # print(f"[DEBUG] scans parameter: {scans}")
    if in_channel_first:
        B, C, H, W = x.shape
        K = indices.shape[0]  # 动态获取方向数
        L = H * W
        # print(indices.shape)

        # 预分配输出张量
        y = x.new_zeros((B, K, C, L))

        # 批量处理所有方向，消除函数调用开销
        for k in range(K):
            i_idx = indices[k, :, 0]  # (L,) - 行索引
            j_idx = indices[k, :, 1]  # (L,) - 列索引
            y[:, k, :, :] = x[:, :, i_idx, j_idx]  # (B, C, L)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


# def cross_merge_fwd(y: torch.Tensor, indices=None, in_channel_first=True, out_channel_first=True, scans=0):
#     if out_channel_first:
#         B, K, D, H, W = y.shape
#         L = H * W
#         y = y.view(B, K, D, L)
#
#         # z = restore_from_flattened((H, W), y[:, 0, :, :], indices[0]).view(B, D, -1)
#         # z += restore_from_flattened((H, W), y[:, 1, :, :], indices[1]).view(B, D, -1)
#         # z += restore_from_flattened((H, W), y[:, 2, :, :], indices[2]).view(B, D, -1)
#         # z += restore_from_flattened((H, W), y[:, 3, :, :], indices[3]).view(B, D, -1)
#
#         # 检查 indices 格式
#         if indices.dim() == 3 and indices.shape[-1] == 2:
#             # (K, L, 2) 格式 - 2D 坐标格式 (来自 get_permute_order)
#             z = y.new_zeros((B, D, H, W))
#             for k in range(K):
#                 i_indices = indices[k, :, 0]  # (L,)
#                 j_indices = indices[k, :, 1]  # (L,)
#                 z[:, :, i_indices, j_indices] += y[:, k, :, :]  # scatter add
#             z = z.view(B, D, L)
#         elif indices.dim() == 2:
#             # (K, L) 格式 - 1D 索引格式 (来自 ScanOrderGenerator)
#             # 获取逆索引（使用缓存）
#             re_indices = _get_inverse_indices(indices)  # (K, L)
#             z = y.new_zeros((B, D, L))
#             for k in range(K):
#                 z += y[:, k, :, :][:, :, re_indices[k]]  # (B, D, L)
#         else:
#             # (B, K, L) 格式 - 每个 batch 可能有不同索引
#             _, re_indices = torch.sort(indices, dim=2)
#             re_indices_expanded = re_indices.unsqueeze(2).expand(-1, -1, D, -1)  # (B, K, D, L)
#             z = torch.gather(y, dim=3, index=re_indices_expanded)  # (B, K, D, L)
#             z = z.sum(dim=1)  # (B, D, L)
#
#     if in_channel_first and (not out_channel_first):
#         z = z.permute(0, 2, 1).contiguous()
#     elif (not in_channel_first) and out_channel_first:
#         z = z.permute(0, 2, 1).contiguous()
#
#     return z
# ========== 优化版本 cross_merge_fwd ==========
def cross_merge_fwd(y: torch.Tensor, indices=None, in_channel_first=True, out_channel_first=True, scans=0):
    """
    优化版 cross_merge_fwd

    优化点：
    1. 只创建一次零张量（原来创建4次）
    2. 使用 in-place 累加（+=）减少内存分配
    3. 消除 restore_from_flattened 函数调用（内联处理）
    4. 支持动态 K 值
    """
    if out_channel_first:
        B, K, D, H, W = y.shape
        L = H * W
        y = y.view(B, K, D, L)

        # 优化：只创建一次零张量，避免重复分配内存
        z = y.new_zeros((B, D, H, W))

        # 批量累加所有方向，使用 in-place 操作
        for k in range(K):
            i_idx = indices[k, :, 0]  # (L,) - 行索引
            j_idx = indices[k, :, 1]  # (L,) - 列索引
            z[:, :, i_idx, j_idx] += y[:, k, :, :]  # in-place 累加，避免额外内存

        z = z.view(B, D, -1)

    if in_channel_first and (not out_channel_first):
        z = z.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        z = z.permute(0, 2, 1).contiguous()

    return z

class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, indices=None, in_channel_first=True, out_channel_first=True, one_by_one=False,
                scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.indices = indices

        B, C, H, W = x.shape
        if not in_channel_first:
            B, H, W, C = x.shape
        ctx.shape = (B, C, H, W)

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        y = _fn(x, indices, in_channel_first, out_channel_first, scans)

        return y

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        indices = ctx.indices

        ys = ys.view(B, -1, C, H, W) if out_channel_first else ys.view(B, H, W, -1, C)
        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, indices, in_channel_first, out_channel_first, scans)

        y = y.view(B, -1, H, W) if in_channel_first else y.view(B, H, W, -1)

        return y, None, None, None, None, None


class CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, indices=None, in_channel_first=True, out_channel_first=True, one_by_one=False,
                scans=0):
        # x: (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        # y: (B, 4, C, H * W) | (B, H * W, 4, C)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.indices = indices

        B, K, C, H, W = ys.shape
        if not out_channel_first:
            B, H, W, K, C = ys.shape
        ctx.shape = (B, C, H, W)
        ctx.K = K  # 保存 K 用于反向传播

        _fn = cross_merge1b1_fwd if one_by_one else cross_merge_fwd
        y = _fn(ys, indices, in_channel_first, out_channel_first, scans)

        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, h, w)
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        K = ctx.K  # 获取保存的 K
        indices = ctx.indices

        if in_channel_first:
            x = x.view(B, C, H, W)
        else:
            x = x.view(B, H, W, C)

        _fn = cross_scan1b1_fwd if one_by_one else cross_scan_fwd
        x = _fn(x, indices, in_channel_first, out_channel_first, scans)
        x = x.view(B, K, C, H, W) if out_channel_first else x.view(B, H, W, K, C)  # 使用实际的 K

        return x, None, None, None, None, None


# triton implements ========================================

@triton.jit
def triton_cross_scan_flex(
        x: tl.tensor,  # (B, C, H, W) | (B, H, W, C) | (B, 4, C, H, W) | (B, H, W, 4, C)
        y: tl.tensor,  # (B, 4, C, H, W) | (B, H, W, 4, C)
        x_layout: tl.constexpr,
        y_layout: tl.constexpr,
        operation: tl.constexpr,
        onebyone: tl.constexpr,
        scans: tl.constexpr,
        BC: tl.constexpr,
        BH: tl.constexpr,
        BW: tl.constexpr,
        DC: tl.constexpr,
        DH: tl.constexpr,
        DW: tl.constexpr,
        NH: tl.constexpr,
        NW: tl.constexpr,
):
    # x_layout = 0
    # y_layout = 1 # 0 BCHW, 1 BHWC
    # operation = 0 # 0 scan, 1 merge
    # onebyone = 0 # 0 false, 1 true
    # scans = 0 # 0 cross scan, 1 unidirectional, 2 bidirectional

    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    pos_h = (i_h * BH + tl.arange(0, BH)[:, None])
    pos_w = (i_w * BW + tl.arange(0, BW)[None, :])
    neg_h = (DH - i_h * BH - 1 - tl.arange(0, BH)[:, None])
    neg_w = (DW - i_w * BW - 1 - tl.arange(0, BW)[None, :])
    if scans == 0:
        # none; trans; flip; trans + flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = pos_w * DH + pos_h  # trans
        HWRoute2 = neg_h * DW + neg_w  # flip
        HWRoute3 = neg_w * DH + neg_h  # trans + flip
    elif scans == 1:
        # none; none; none; none;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = HWRoute0
        HWRoute3 = HWRoute0
    elif scans == 2:
        # none; none; flip; flip;
        HWRoute0 = pos_h * DW + pos_w
        HWRoute1 = HWRoute0
        HWRoute2 = neg_h * DW + neg_w  # flip
        HWRoute3 = HWRoute2

    _tmp1 = DC * DH * DW

    y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
    if y_layout == 0:
        p_y1 = y_ptr_base + HWRoute0
        p_y2 = y_ptr_base + _tmp1 + HWRoute1
        p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
        p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
    else:
        p_y1 = y_ptr_base + HWRoute0 * 4 * DC
        p_y2 = y_ptr_base + DC + HWRoute1 * 4 * DC
        p_y3 = y_ptr_base + 2 * DC + HWRoute2 * 4 * DC
        p_y4 = y_ptr_base + 3 * DC + HWRoute3 * 4 * DC

    if onebyone == 0:
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x = x_ptr_base + HWRoute0
        else:
            p_x = x_ptr_base + HWRoute0 * DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y3 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y4 + _idx_y, _x, mask=_mask_hw)
        elif operation == 1:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hw)
                _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hw)
                tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)

    else:
        x_ptr_base = x + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
        if x_layout == 0:
            p_x1 = x_ptr_base + HWRoute0
            p_x2 = p_x1 + _tmp1
            p_x3 = p_x2 + _tmp1
            p_x4 = p_x3 + _tmp1
        else:
            p_x1 = x_ptr_base + HWRoute0 * 4 * DC
            p_x2 = p_x1 + DC
            p_x3 = p_x2 + DC
            p_x4 = p_x3 + DC

        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=_mask_hw), mask=_mask_hw)
                tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=_mask_hw), mask=_mask_hw)
        else:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y), mask=_mask_hw)
                tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y), mask=_mask_hw)
                tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y), mask=_mask_hw)
                tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y), mask=_mask_hw)


class CrossScanTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if one_by_one:
            if in_channel_first:
                B, _, C, H, W = x.shape
            else:
                B, H, W, _, C = x.shape
        else:
            if in_channel_first:
                B, C, H, W = x.shape
            else:
                B, H, W, C = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)

        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)

        y = x.new_empty((B, 4, C, H * W)) if out_channel_first else x.new_empty((B, H * W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y,
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return y

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        if one_by_one:
            x = y.new_empty((B, 4, C, H, W)) if in_channel_first else y.new_empty((B, H, W, 4, C))
        else:
            x = y.new_empty((B, C, H, W)) if in_channel_first else y.new_empty((B, H, W, C))

        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(),
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x, None, None, None, None


class CrossMergeTritonF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        if out_channel_first:
            B, _, C, H, W = y.shape
        else:
            B, H, W, _, C = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        if one_by_one:
            x = y.new_empty((B, 4, C, H * W)) if in_channel_first else y.new_empty((B, H * W, 4, C))
        else:
            x = y.new_empty((B, C, H * W)) if in_channel_first else y.new_empty((B, H * W, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x, y.contiguous(),
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 1, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return x

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = x.new_empty((B, 4, C, H, W)) if out_channel_first else x.new_empty((B, H, W, 4, C))
        triton_cross_scan_flex[(NH * NW, NC, B)](
            x.contiguous(), y,
            (0 if in_channel_first else 1), (0 if out_channel_first else 1), 0, (0 if not one_by_one else 1), scans,
            BC, BH, BW, C, H, W, NH, NW
        )
        return y, None, None, None, None, None


# ==================== Triton kernels for custom indices ====================
# 支持自定义扫描索引的 Triton 实现

if WITH_TRITON:
    @triton.jit
    def triton_indexed_scan_kernel(
            x_ptr,  # input: (B, C, L)
            y_ptr,  # output: (B, K, C, L)
            indices_ptr,  # indices: (K, L)
            B: tl.constexpr,
            K: tl.constexpr,
            C: tl.constexpr,
            L: tl.constexpr,
            BLOCK_L: tl.constexpr,
    ):
        """
        Triton kernel for indexed scan (gather operation)
        For each k, gather x values at positions indices[k]
        """
        # 程序 ID: (batch, k, channel, L_block)
        pid_b = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_c = tl.program_id(2)

        # 计算 L 维度的偏移
        offs_l = tl.arange(0, BLOCK_L)
        mask_l = offs_l < L

        # 加载 indices[k, :]
        indices_offset = pid_k * L + offs_l
        src_indices = tl.load(indices_ptr + indices_offset, mask=mask_l, other=0)

        # 计算源位置: x[b, c, src_indices]
        x_offset = pid_b * C * L + pid_c * L + src_indices
        x_vals = tl.load(x_ptr + x_offset, mask=mask_l, other=0.0)

        # 存储到 y[b, k, c, :]
        y_offset = pid_b * K * C * L + pid_k * C * L + pid_c * L + offs_l
        tl.store(y_ptr + y_offset, x_vals, mask=mask_l)


    @triton.jit
    def triton_indexed_merge_kernel(
            y_ptr,  # input: (B, K, C, L)
            x_ptr,  # output: (B, C, L)
            re_indices_ptr,  # inverse indices: (K, L)
            B: tl.constexpr,
            K: tl.constexpr,
            C: tl.constexpr,
            L: tl.constexpr,
            BLOCK_L: tl.constexpr,
    ):
        """
        Triton kernel for indexed merge (scatter-sum operation)
        For each k, scatter y values back to original positions and sum
        """
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)

        offs_l = tl.arange(0, BLOCK_L)
        mask_l = offs_l < L

        # 初始化累加器
        acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

        # 对每个 k 进行累加
        for k in range(K):
            # 加载 re_indices[k, :]
            re_indices_offset = k * L + offs_l
            dst_indices = tl.load(re_indices_ptr + re_indices_offset, mask=mask_l, other=0)

            # 加载 y[b, k, c, :]
            y_offset = pid_b * K * C * L + k * C * L + pid_c * L + offs_l
            y_vals = tl.load(y_ptr + y_offset, mask=mask_l, other=0.0)

            # 根据 dst_indices 进行 scatter (使用 gather 的逆操作思路)
            # 注意：这里我们用 dst_indices 作为源位置来 gather
            # 因为 re_indices 是逆索引，所以 y[..., dst_indices] 就是还原后的位置
            acc += tl.load(y_ptr + pid_b * K * C * L + k * C * L + pid_c * L + dst_indices,
                           mask=mask_l, other=0.0)

        # 存储结果
        x_offset = pid_b * C * L + pid_c * L + offs_l
        tl.store(x_ptr + x_offset, acc, mask=mask_l)


    class CrossScanTritonIndexedF(torch.autograd.Function):
        """支持自定义索引的 Triton scan 实现"""

        @staticmethod
        def forward(ctx, x: torch.Tensor, indices: torch.Tensor, in_channel_first=True, out_channel_first=True,
                    one_by_one=False, scans=0):
            assert in_channel_first and out_channel_first, "Triton indexed scan only supports channel_first"
            assert indices is not None and indices.dim() == 2, "indices must be (K, L)"

            B, C, H, W = x.shape
            L = H * W
            K = indices.shape[0]

            ctx.save_for_backward(indices)
            ctx.shape = (B, C, H, W)
            ctx.K = K

            # 展平输入
            flat_x = x.view(B, C, L).contiguous()
            y = torch.empty((B, K, C, L), dtype=x.dtype, device=x.device)

            # 选择 block size
            BLOCK_L = triton.next_power_of_2(L)
            if BLOCK_L > 4096:
                BLOCK_L = 4096

            # 计算 grid
            grid = (B, K, C)

            triton_indexed_scan_kernel[grid](
                flat_x, y, indices,
                B, K, C, L, BLOCK_L
            )

            return y

        @staticmethod
        def backward(ctx, grad_y: torch.Tensor):
            indices, = ctx.saved_tensors
            B, C, H, W = ctx.shape
            K = ctx.K
            L = H * W

            # 计算逆索引
            re_indices = _get_inverse_indices(indices)

            grad_x = torch.empty((B, C, L), dtype=grad_y.dtype, device=grad_y.device)

            BLOCK_L = triton.next_power_of_2(L)
            if BLOCK_L > 4096:
                BLOCK_L = 4096

            grid = (B, C)

            triton_indexed_merge_kernel[grid](
                grad_y.contiguous(), grad_x, re_indices,
                B, K, C, L, BLOCK_L
            )

            grad_x = grad_x.view(B, C, H, W)
            return grad_x, None, None, None, None, None


    class CrossMergeTritonIndexedF(torch.autograd.Function):
        """支持自定义索引的 Triton merge 实现"""

        @staticmethod
        def forward(ctx, y: torch.Tensor, indices: torch.Tensor, in_channel_first=True, out_channel_first=True,
                    one_by_one=False, scans=0):
            assert in_channel_first and out_channel_first, "Triton indexed merge only supports channel_first"
            assert indices is not None and indices.dim() == 2, "indices must be (K, L)"

            B, K, C, H, W = y.shape
            L = H * W

            ctx.save_for_backward(indices)
            ctx.shape = (B, C, H, W)
            ctx.K = K

            # 计算逆索引
            re_indices = _get_inverse_indices(indices)

            y_flat = y.view(B, K, C, L).contiguous()
            x = torch.empty((B, C, L), dtype=y.dtype, device=y.device)

            BLOCK_L = triton.next_power_of_2(L)
            if BLOCK_L > 4096:
                BLOCK_L = 4096

            grid = (B, C)

            triton_indexed_merge_kernel[grid](
                y_flat, x, re_indices,
                B, K, C, L, BLOCK_L
            )

            return x

        @staticmethod
        def backward(ctx, grad_x: torch.Tensor):
            indices, = ctx.saved_tensors
            B, C, H, W = ctx.shape
            K = ctx.K
            L = H * W

            grad_x_flat = grad_x.view(B, C, L).contiguous()
            grad_y = torch.empty((B, K, C, L), dtype=grad_x.dtype, device=grad_x.device)

            BLOCK_L = triton.next_power_of_2(L)
            if BLOCK_L > 4096:
                BLOCK_L = 4096

            grid = (B, K, C)

            triton_indexed_scan_kernel[grid](
                grad_x_flat, grad_y, indices,
                B, K, C, L, BLOCK_L
            )

            grad_y = grad_y.view(B, K, C, H, W)
            return grad_y, None, None, None, None, None

else:
    # Triton 不可用时的占位符
    CrossScanTritonIndexedF = None
    CrossMergeTritonIndexedF = None


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_scan_fn(x: torch.Tensor, indices=None, in_channel_first=True, out_channel_first=True, one_by_one=False,
                  scans=0, force_torch=False):
    """
    Cross scan function with optional Triton acceleration.

    Args:
        x: Input tensor (B, C, H, W) or (B, H, W, C)
        indices: Custom scan indices (K, L) or None for default cross scan
        in_channel_first: Input channel first format
        out_channel_first: Output channel first format
        one_by_one: One by one mode
        scans: Scan mode (0: cross, 1: unidi, 2: bidi)
        force_torch: Force PyTorch implementation (disable Triton)

    Returns:
        Scanned tensor (B, K, C, L) or (B, L, K, C)
    """
    # if indices is None:
    #     # 无自定义索引：使用原始 Triton cross scan（如果可用）
    #     use_triton = WITH_TRITON and x.is_cuda and (not force_torch)
    #     CSF = CrossScanTritonF if use_triton else CrossScanF
    # else:
    #     # 有自定义索引：使用 Triton indexed scan（如果可用且满足条件）
    #     can_use_triton_indexed = (
    #             WITH_TRITON and x.is_cuda and (not force_torch) and
    #             in_channel_first and out_channel_first and
    #             indices.dim() == 2 and
    #             CrossScanTritonIndexedF is not None
    #     )
    #     if can_use_triton_indexed:
    #         CSF = CrossScanTritonIndexedF
    #     else:
    #         CSF = CrossScanF
    CSF = CrossScanF

    with torch.cuda.device(x.device):
        return CSF.apply(x, indices, in_channel_first, out_channel_first, one_by_one, scans)


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def cross_merge_fn(y: torch.Tensor, indices=None, in_channel_first=True, out_channel_first=True, one_by_one=False,
                   scans=0, force_torch=False):
    """
    Cross merge function with optional Triton acceleration.

    Args:
        y: Input tensor (B, K, C, H, W) or (B, H, W, K, C)
        indices: Custom scan indices (K, L) or None for default cross merge
        in_channel_first: Input channel first format
        out_channel_first: Output channel first format
        one_by_one: One by one mode
        scans: Scan mode (0: cross, 1: unidi, 2: bidi)
        force_torch: Force PyTorch implementation (disable Triton)

    Returns:
        Merged tensor (B, C, L) or (B, L, C)
    """
    # if indices is None:
    #     # 无自定义索引：使用原始 Triton cross merge（如果可用）
    #     use_triton = WITH_TRITON and y.is_cuda and (not force_torch)
    #     CMF = CrossMergeTritonF if use_triton else CrossMergeF
    # else:
    #     # 有自定义索引：使用 Triton indexed merge（如果可用且满足条件）
    #     can_use_triton_indexed = (
    #             WITH_TRITON and y.is_cuda and (not force_torch) and
    #             in_channel_first and out_channel_first and
    #             indices.dim() == 2 and
    #             CrossMergeTritonIndexedF is not None
    #     )
    #     if can_use_triton_indexed:
    #         CMF = CrossMergeTritonIndexedF
    #     else:
    #         CMF = CrossMergeF
    CMF = CrossMergeF

    with torch.cuda.device(y.device):
        return CMF.apply(y, indices, in_channel_first, out_channel_first, one_by_one, scans)


# checks =================================================================

class CHECK:
    def check_csm_triton():
        B, C, H, W = 256, 192, 56, 57
        dtype = torch.float16
        dtype = torch.float32
        x = torch.randn((B, C, H, W), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        y = torch.randn((B, 4, C, H, W), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        x1 = x.clone().detach().requires_grad_(True)
        y1 = y.clone().detach().requires_grad_(True)

        def cross_scan(x: torch.Tensor):
            B, C, H, W = x.shape
            L = H * W
            xs = torch.stack([
                x.view(B, C, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L),
                torch.flip(x.contiguous().view(B, C, L), dims=[-1]),
                torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs

        def cross_merge(out_y: torch.Tensor):
            B, K, D, H, W = out_y.shape
            L = H * W
            out_y = out_y.view(B, K, D, L)
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
            return y

        def cross_scan_1b1(x: torch.Tensor):
            B, K, C, H, W = x.shape
            L = H * W
            xs = torch.stack([
                x[:, 0].view(B, C, L),
                torch.transpose(x[:, 1], dim0=2, dim1=3).contiguous().view(B, C, L),
                torch.flip(x[:, 2].contiguous().view(B, C, L), dims=[-1]),
                torch.flip(torch.transpose(x[:, 3], dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs

        def unidi_scan(x):
            B, C, H, W = x.shape
            x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
            return x

        def unidi_merge(ys):
            B, K, C, H, W = ys.shape
            return ys.view(B, 4, -1, H * W).sum(1)

        def bidi_scan(x):
            B, C, H, W = x.shape
            x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            x = torch.cat([x, x.flip(dims=[-1])], dim=1)
            return x

        def bidi_merge(ys):
            B, K, D, H, W = ys.shape
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            return ys.contiguous().sum(1)

        if True:
            res0 = triton.testing.do_bench(lambda: cross_scan(x))
            res1 = triton.testing.do_bench(lambda: cross_scan_fn(x, True, True, False))
            # res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x))
            res3 = triton.testing.do_bench(lambda: cross_merge(y))
            res4 = triton.testing.do_bench(lambda: cross_merge_fn(y, True, True, False))
            # res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y))
            # print(res0, res1, res2, res3, res4, res5)
            print(res0, res1, res3, res4)
            res0 = triton.testing.do_bench(lambda: cross_scan(x).sum().backward())
            res1 = triton.testing.do_bench(lambda: cross_scan_fn(x, True, True, False).sum().backward())
            # res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x).sum().backward())
            res3 = triton.testing.do_bench(lambda: cross_merge(y).sum().backward())
            res4 = triton.testing.do_bench(lambda: cross_merge_fn(y, True, True, False).sum().backward())
            # res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y).sum().backward())
            # print(res0, res1, res2, res3, res4, res5)
            print(res0, res1, res3, res4)

        print("test cross scan")
        for (cs0, cm0, cs1, cm1) in [
            # channel_first -> channel_first
            (cross_scan, cross_merge, cross_scan_fn, cross_merge_fn),
            (unidi_scan, unidi_merge, lambda x: cross_scan_fn(x, scans=1), lambda x: cross_merge_fn(x, scans=1)),
            (bidi_scan, bidi_merge, lambda x: cross_scan_fn(x, scans=2), lambda x: cross_merge_fn(x, scans=2)),

            # flex: BLC->BCL; BCL->BLC; BLC->BLC;
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x.permute(0, 2, 3, 1), in_channel_first=False),
             lambda x: cross_merge_fn(x, in_channel_first=False).permute(0, 2, 1)),
            (cross_scan, cross_merge, lambda x: cross_scan_fn(x, out_channel_first=False).permute(0, 2, 3, 1),
             lambda x: cross_merge_fn(x.permute(0, 3, 4, 1, 2), out_channel_first=False)),
            (cross_scan, cross_merge,
             lambda x: cross_scan_fn(x.permute(0, 2, 3, 1), in_channel_first=False, out_channel_first=False).permute(0,
                                                                                                                     2,
                                                                                                                     3,
                                                                                                                     1),
             lambda x: cross_merge_fn(x.permute(0, 3, 4, 1, 2), in_channel_first=False,
                                      out_channel_first=False).permute(0, 2, 1)),

            # previous
            # (cross_scan, cross_merge, lambda x: CrossScanTriton.apply(x), lambda x: CrossMergeTriton.apply(x)),
            # (unidi_scan, unidi_merge, lambda x: getCSM(1)[0].apply(x), lambda x: getCSM(1)[1].apply(x)),
            # (bidi_scan, bidi_merge, lambda x: getCSM(2)[0].apply(x), lambda x: getCSM(2)[1].apply(x)),
        ]:
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            o0 = cs0(x)
            o1 = cs1(x1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((x.grad - x1.grad).abs().max())
            o0 = cm0(y)
            o1 = cm1(y1)
            o0.backward(x.view(B, C, H * W))
            o1.backward(x.view(B, C, H * W))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            print("===============", flush=True)

        print("test cross scan one by one")
        for (cs0, cs1) in [
            (cross_scan_1b1, lambda x: cross_scan_fn(x, one_by_one=True)),
            # (cross_scan_1b1, lambda x: CrossScanTriton1b1.apply(x)),
        ]:
            o0 = cs0(y)
            o1 = cs1(y1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            print("===============", flush=True)

    def check_csm_scan3():
        if False:
            x = torch.arange(0, 16).view(1, 1, 4, 4).cuda()
            out1 = cross_scan_fn(x, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out2 = cross_merge_fn(out1, scans=3, force_torch=True).view(1, 1, 4, 4)
            out4 = cross_merge_fn(out1, one_by_one=True, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out3 = cross_scan_fn(out4, one_by_one=True, scans=3, force_torch=True).view(1, 4, 1, 4, 4)
            out5 = cross_scan_fn(x.view(1, 4, 4, 1), in_channel_first=False, out_channel_first=False, scans=3,
                                 force_torch=True).view(1, 4, 4, 4, 1)
            out6 = cross_merge_fn(out5, in_channel_first=False, out_channel_first=False, scans=3,
                                  force_torch=True).view(1, 4, 4, 1)
            out8 = cross_merge_fn(out5, in_channel_first=False, out_channel_first=False, one_by_one=True, scans=3,
                                  force_torch=True).view(1, 4, 4, 4, 1)
            out7 = cross_scan_fn(out8, in_channel_first=False, out_channel_first=False, one_by_one=True, scans=3,
                                 force_torch=True).view(1, 4, 4, 4, 1)
            print(out1.view(4, -1))
            print(out2.view(-1))
            print(out3.view(4, -1))
            print(out4.view(4, -1))
            print(out5.view(-1, 4).t())
            print(out6.view(-1))
            print(out7.view(-1, 4).t())
            print(out8.view(-1, 4).t())

        B, C, H, W = 27, 253, 57, 58
        x = torch.randn((B, C, H, W)).cuda()

        for scans in [0, 1, 2, 3]:
            o1 = cross_scan_fn(x, scans=scans, force_torch=True).view(B, 4, C, H, W)
            print((cross_scan_fn(x, scans=scans) == cross_scan_fn(x, scans=scans, force_torch=True)).all())
            print((cross_merge_fn(o1, scans=scans) == cross_merge_fn(o1, scans=scans, force_torch=True)).all())

            kwargs = dict(in_channel_first=False, out_channel_first=False)
            x2 = x.permute(0, 2, 3, 1).contiguous()
            o2 = o1.permute(0, 3, 4, 1, 2).contiguous()
            print((cross_scan_fn(x, scans=scans, **kwargs) == cross_scan_fn(x, scans=scans, force_torch=True,
                                                                            **kwargs)).all())
            print((cross_merge_fn(o2, scans=scans, **kwargs) == cross_merge_fn(o2, scans=scans, force_torch=True,
                                                                               **kwargs)).all())

        breakpoint()


if __name__ == "__main__":
    CHECK.check_csm_scan3()
    CHECK.check_csm_triton()




