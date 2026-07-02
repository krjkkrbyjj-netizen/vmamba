import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from mamba2.ssd_minimal import selective_scan_chunk_fn

# 导入扫描顺序生成器
# try:
#     from .scan_order import ScanOrderGenerator
#     from .learnable_scan import LearnableScanOrder, aux_loss_cache
# except:
#     from scan_order import ScanOrderGenerator
#     from learnable_scan import LearnableScanOrder, aux_loss_cache


# from compute_attn import *  #在compute_attn里面导入计算注意力矩阵的函数

# ==================== 辅助损失缓存（用于内联可学习扫描）====================
class _InlineScanAuxLossCache:
    """
    辅助损失缓存 - 用于内联可学习扫描方法

    内联可学习扫描方法（get_learnable_order等）使用这个缓存
    与 learnable_scan.py 中的 aux_loss_cache 独立

    Usage:
        # 在训练循环中收集辅助损失
        from classification.models.vmamba import inline_scan_aux_loss_cache

        # Forward pass
        output = model(x)

        # 收集辅助损失
        aux_losses = inline_scan_aux_loss_cache.collect()

        # 添加到总损失
        total_loss = main_loss
        for name, loss in aux_losses.items():
            total_loss += loss
    """
    _losses = {}
    _collect_count = 0  # 追踪 collect() 被调用的次数

    @classmethod
    def register(cls, name: str, loss: torch.Tensor):
        """注册一个辅助损失（累加模式）"""
        # 确保 _losses 字典存在（DDP环境下可能不存在）
        if not hasattr(cls, '_losses'):
            cls._losses = {}

        if not name.startswith('loss_'):
            name = 'loss_' + name
        # 累加损失而不是覆盖（因为多个SS2D block会注册同名损失）
        if name in cls._losses:
            cls._losses[name] = cls._losses[name] + loss
        else:
            cls._losses[name] = loss

    @classmethod
    def collect(cls):
        """收集所有损失并清空缓存"""
        # 安全地访问和更新 _collect_count（DDP环境下可能不存在）
        if not hasattr(cls, '_collect_count'):
            cls._collect_count = 0
        cls._collect_count += 1

        # 确保 _losses 字典存在
        if not hasattr(cls, '_losses'):
            cls._losses = {}

        losses = cls._losses.copy()
        cls._losses.clear()
        return losses

    @classmethod
    def clear(cls):
        """清空缓存"""
        # 确保 _losses 字典存在（DDP环境下可能不存在）
        if not hasattr(cls, '_losses'):
            cls._losses = {}
        cls._losses.clear()



# 导出缓存实例供外部使用
inline_scan_aux_loss_cache = _InlineScanAuxLossCache
# 为detection模块提供简短别名
aux_loss_cache = inline_scan_aux_loss_cache


def compute_scan_diversity_loss(scores: torch.Tensor) -> torch.Tensor:
    """
    多样性损失 - 鼓励不同扫描方向学习不同的模式

    Args:
        scores: (B, k, L) 各方向的 score
    Returns:
        loss: 多样性损失
    """
    B, k, L = scores.shape
    scores_norm = F.normalize(scores, dim=-1)
    sim_matrix = torch.bmm(scores_norm, scores_norm.transpose(1, 2))
    mask = torch.eye(k, device=scores.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim_matrix.masked_fill(mask, 0)
    diversity_loss = off_diag.abs().mean()

    # 处理 inf/nan 情况
    if torch.isnan(diversity_loss) or torch.isinf(diversity_loss):
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

    return diversity_loss


def compute_scan_coverage_loss(scores: torch.Tensor) -> torch.Tensor:
    """
    覆盖均匀性损失 - 鼓励分数分布更加均匀

    Args:
        scores: (B, k, L) 各方向的 score
    Returns:
        loss: 覆盖损失
    """
    B, k, L = scores.shape

    # 使用 log_softmax 获得数值稳定的对数概率
    log_probs = F.log_softmax(scores, dim=-1)  # (B, k, L)

    # 目标：均匀分布
    uniform = torch.ones_like(log_probs) / L

    # 计算 KL 散度：KL(softmax(scores) || uniform)
    kl_div = F.kl_div(log_probs, uniform, reduction='batchmean')

    # 处理 inf/nan 情况
    if torch.isnan(kl_div) or torch.isinf(kl_div):
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

    return kl_div

# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1).contiguous()).view(B, C, H, W).contiguous()
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C).contiguous()).view(B, H, W, C).contiguous()
        else:
            raise NotImplementedError


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


# support: v0, v0seq
class SS2Dv0:
    def __initv0__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            # ======================
            dropout=0.0,
            # ======================
            seq=False,
            force_fp32=True,
            **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 1
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj, A, D ============================
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=k_group,
        )

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x: torch.Tensor, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        selective_scan = partial(selective_scan_fn, backend="mamba")

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if hasattr(self, "x_proj_bias"):
            x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.contiguous()  # (b, k, d_state, l)
        Cs = Cs.contiguous()  # (b, k, d_state, l)

        As = -self.A_logs.float().exp()  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# support: v01-v05; v051d,v052d,v052dc;
# postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
# history support: v2,v3;v31d,v32d,v32dc;
class SS2Dv2:
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=True,
            # ======================
            # scan configuration ===
            scan_type="cross",
            # cross, permute, diagonal, zigzag, spiral, efficient, hilbert, blockwise, random, learnable, learnable_v2, content_adaptive
            scan_k=4,  # number of scan directions
            scan_learnable=False,  # whether to use learnable scan
            scan_diversity_weight=0.0,  # auxiliary loss weight for diversity (only for learnable scan)
            scan_coverage_weight=0.0,  # auxiliary loss weight for coverage (only for learnable scan)
            use_independent_learnable=False,  # True: 每个SS2D独立参数, False: 共享参数
            scan_force_torch=False,  # force PyTorch implementation (disable Triton)
            # ======================
            # 共享网络参数 =========
            stage_index=0,  # stage索引，用于获取对应的共享网络
            shared_learnable_nets=None,  # 共享的可学习扫描网络字典
            shared_learnable_params=None,  # 共享的可学习扫描参数字典
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4  # 使用 scan_k 作为扫描方向数
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        self.scan_force_torch = scan_force_torch  # store for forward_corev2
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
                        scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
                        scan_force_torch=self.scan_force_torch),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex",
                        scan_force_torch=self.scan_force_torch),
            v04=partial(self.forward_corev2, force_fp32=False, scan_force_torch=self.scan_force_torch),
            # selective_scan_backend="oflex", scan_mode="cross2d"
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_force_torch=self.scan_force_torch),
            # selective_scan_backend="oflex", scan_mode="cross2d"
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi",
                          scan_force_torch=self.scan_force_torch),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi",
                          scan_force_torch=self.scan_force_torch),
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d",
                           scan_force_torch=self.scan_force_torch),
            v052d3=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode=3,
                           scan_force_torch=self.scan_force_torch),  # debug
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex",
                       scan_force_torch=self.scan_force_torch),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex",
                       scan_force_torch=self.scan_force_torch),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # self.conv_offset = nn.Sequential(
        #     nn.Conv2d(in_channels=self.d_inner,
        #         out_channels=self.d_inner,
        #         groups=self.d_inner,
        #         bias=conv_bias,
        #         kernel_size=d_conv,
        #         padding=(d_conv - 1) // 2,
        #         **factory_kwargs,),
        #     ChannelAttention(self.d_inner),
        #     nn.GELU(),
        #     nn.Conv2d(self.d_inner, self.d_inner, 1, 1, 0, bias=False),
        # )
        # # 通道注意力部分
        # reduced_dim = max(self.d_inner// 16, 4)
        # self.channel_proj1 = nn.Conv2d(self.d_inner, reduced_dim, kernel_size=1)
        # self.channel_proj2 = nn.Conv2d(reduced_dim, self.d_inner, kernel_size=1)

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros(
                (self.k_group * self.d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

        # Initialize scan generator =======================================
        self.scan_type = scan_type
        self.scan_learnable = scan_learnable
        self.scan_generator = None

        # 保存辅助损失权重（用于内联可学习扫描）
        self.scan_diversity_weight = scan_diversity_weight
        self.scan_coverage_weight = scan_coverage_weight

        # if scan_learnable or scan_type == "learnable":
        #     # Use learnable scan order
        #     self.scan_generator = LearnableScanOrder(
        #         dim=self.d_inner,
        #         k=scan_k,
        #         diversity_weight=scan_diversity_weight,
        #         coverage_weight=scan_coverage_weight,
        #     )
        # else:
        #     # Use fixed scan order generator (including cross mode)
        #     self.scan_generator = ScanOrderGenerator(
        #         mode=scan_type,
        #         k=scan_k,
        #     )
        # ========== 可学习扫描网络初始化（三种模式）==========
        # 模式1: use_independent_learnable=True - 每个SS2D独立参数（在__init__中创建）
        # 模式2: use_independent_learnable=False + shared_learnable_nets!=None - 使用VSSM传入的共享网络（stage级别共享）
        # 模式3: use_independent_learnable=False + shared_learnable_nets==None - 懒加载（所有SS2D共享，已废弃）
        self.use_independent_learnable = use_independent_learnable
        self.stage_index = stage_index
        # 注意：不直接保存引用，避免参数重复注册
        # 而是保存一个访问函数
        if shared_learnable_nets is not None:
            # 创建访问函数，避免直接引用导致参数重复
            self._get_shared_nets = lambda: shared_learnable_nets
            self._get_shared_params = lambda: shared_learnable_params
        else:
            self._get_shared_nets = lambda: None
            self._get_shared_params = lambda: None
        self.learnable_score_net = None
        self.learnable_offset_nets = None
        self.learnable_pos_bias = None
        self.content_net = None
        self.scan_heads = None
        self.direction_embeds = None

        if self.use_independent_learnable and scan_type in ["learnable", "learnable_v2", "content_adaptive"]:
            # 模式1：每个SS2D独立参数
            if scan_type == "learnable":
                # 轻量级网络：depthwise conv + pointwise conv
                self.learnable_score_net = nn.Sequential(
                    nn.Conv2d(self.d_inner, self.d_inner, 3, 1, 1, groups=self.d_inner, bias=False),
                    nn.BatchNorm2d(self.d_inner),
                    nn.GELU(),
                    nn.Conv2d(self.d_inner, self.k_group, 1, bias=True)
                )
                self._init_learnable_weights(self.learnable_score_net)

            elif scan_type == "learnable_v2":
                # 为每个方向创建独立的网络
                self.learnable_offset_nets = nn.ModuleList()
                for i in range(self.k_group):
                    net = nn.Sequential(
                        nn.Conv2d(self.d_inner, self.d_inner // 4, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(self.d_inner // 4),
                        nn.GELU(),
                        nn.Conv2d(self.d_inner // 4, 1, 1, bias=True)
                    )
                    # 不同方向使用不同的初始化
                    for m in net.modules():
                        if isinstance(m, nn.Conv2d):
                            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                            if m.bias is not None:
                                nn.init.constant_(m.bias, 0.1 * i)
                        elif isinstance(m, nn.BatchNorm2d):
                            nn.init.ones_(m.weight)
                            nn.init.zeros_(m.bias)
                    self.learnable_offset_nets.append(net)
                # 可学习的位置偏置
                self.learnable_pos_bias = nn.Parameter(torch.randn(self.k_group, 1, 1))

            elif scan_type == "content_adaptive":
                # 内容分析网络
                self.content_net = nn.Sequential(
                    nn.Conv2d(self.d_inner, self.d_inner, 3, 1, 1, groups=self.d_inner, bias=False),
                    nn.BatchNorm2d(self.d_inner),
                    nn.GELU(),
                    nn.Conv2d(self.d_inner, self.d_inner // 2, 1, bias=False),
                    nn.BatchNorm2d(self.d_inner // 2),
                    nn.GELU(),
                )
                # k个方向的扫描头
                self.scan_heads = nn.ModuleList([
                    nn.Conv2d(self.d_inner // 2, 1, 1, bias=True)
                    for _ in range(self.k_group)
                ])
                # 可学习的方向嵌入
                self.direction_embeds = nn.Parameter(
                    torch.randn(self.k_group, self.d_inner // 2, 1, 1) * 0.02
                )
                self._init_learnable_weights(self.content_net)
                for head in self.scan_heads:
                    self._init_learnable_weights(head)
    def _init_learnable_weights(self, module):
        """初始化可学习扫描网络的权重"""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            selective_scan_backend=None,
            # ==============================
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode,
                                                                                                       str) else scan_mode  # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        # print(f"x,shape:B:{B}, D:{D}, H:{H}, W:{W}")
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        # if H == 96 and W == 312:
        #     indices = torch.load("plainmamba/permute_orders_96_312.pth")
        def get_cross_order(hw_shape, k=4):
            H, W = hw_shape
            orders = []

            # # 模式0: 左上角 → 逐行扫描 (行主序)
            # order0 = []
            # for i in range(H):
            #     for j in range(W):
            #         order0.append(i * W + j)  # 标准行主序
            # orders.append(order0)
            #
            # # 模式1: 左上角 → 逐列扫描 (列主序)
            # order1 = []
            # for j in range(W):
            #     for i in range(H):
            #         order1.append(i * W + j)  # 标准列主序
            # orders.append(order1)
            #
            # # 模式2: 右上角 → 逐行扫描 (从右向左扫描每行)
            # order2 = []
            # for i in range(H):
            #     for j in range(W - 1, -1, -1):  # j从右向左
            #         order2.append(i * W + j)
            # orders.append(order2)
            #
            # # 模式3: 右上角 → 逐列扫描 (从最右列向左逐列)
            # order3 = []
            # for j in range(W - 1, -1, -1):  # j从右向左
            #     for i in range(H):
            #         order3.append(i * W + j)
            # orders.append(order3)
            #
            # # 模式4: 左下角 → 逐行扫描 (从下向上扫描每行)
            # order4 = []
            # for i in range(H - 1, -1, -1):  # i从下向上
            #     for j in range(W):
            #         order4.append(i * W + j)
            # orders.append(order4)
            #
            # # 模式5: 左下角 → 逐列扫描 (从左向右逐列，从下向上)
            # order5 = []
            # for j in range(W):
            #     for i in range(H - 1, -1, -1):  # i从下向上
            #         order5.append(i * W + j)
            # orders.append(order5)
            #
            # # 模式6: 右下角 → 逐行扫描 (从下向上，从右向左)
            # order6 = []
            # for i in range(H - 1, -1, -1):  # i从下向上
            #     for j in range(W - 1, -1, -1):  # j从右向左
            #         order6.append(i * W + j)
            # orders.append(order6)
            #
            # # 模式7: 右下角 → 逐列扫描 (从右向左逐列，从下向上)
            # order7 = []
            # for j in range(W - 1, -1, -1):  # j从右向左
            #     for i in range(H - 1, -1, -1):  # i从下向上
            #         order7.append(i * W + j)
            # orders.append(order7)

            # 模式1: 左上角 → 逐行扫描 (行主序)
            if k >= 1:
                order1 = [(i, j) for i in range(H) for j in range(W)]
                orders.append(order1)

            # 模式2: 左上角 → 逐列扫描 (列主序)
            if k >= 2:
                order2 = [(i, j) for j in range(W) for i in range(H)]
                orders.append(order2)

            # 模式3: 右上角 → 逐行扫描 (从右向左扫描每行)
            if k >= 3:
                order3 = [(i, j) for i in range(H) for j in range(W - 1, -1, -1)]
                orders.append(order3)

            # 模式4: 右上角 → 逐列扫描 (从最右列向左逐列)
            if k >= 4:
                order4 = [(i, j) for j in range(W - 1, -1, -1) for i in range(H)]
                orders.append(order4)

            # 模式5-8: 逆序
            if k >= 5:
                orders.append(order1[::-1])
            if k >= 6:
                orders.append(order2[::-1])
            if k >= 7:
                orders.append(order3[::-1])
            if k == 8:
                orders.append(order4[::-1])
            return orders

        # -------------------plainmamba--------------------------------------
        def get_permute_order(hw_shape, dh=0, dw=0, k=8):
            # H, W = hw_shape
            # L = H * W
            # orders = []
            # # Method 1: Left to Right, even rows
            # if k>=1:
            #     order1 = []
            #     for i in range(H):
            #         if i % 2 == 0:
            #             for j in range(W):
            #                 order1.append((dh + i, dw + j))
            #         else:
            #             for j in range(W - 1, -1, -1):
            #                 order1.append((dh + i, dw + j))
            #     orders.append(order1)
            # # Method 2: Right to Left, even rows
            # if k>=2:
            #     order2 = []
            #     for i in range(H):
            #         if i % 2 == 0:
            #             for j in range(W - 1, -1, -1):
            #                 order2.append((i, j))
            #         else:
            #             for j in range(W):
            #                 order2.append((i, j))
            #     orders.append(order2)
            # # # Method 3: Column-first, even columns down, odd columns up
            # if k>=3:
            #     order3 = []
            #     for j in range(W):
            #         if j % 2 == 0:
            #             for i in range(H):
            #                 order3.append((i, j))
            #         else:
            #             for i in range(H - 1, -1, -1):
            #                 order3.append((i, j))
            #     orders.append(order3)
            # # # # Method 4: Column-first, odd columns down, even columns up
            # if k>=4:
            #     order4 = []
            #     for j in range(W):
            #         if j % 2 == 0:
            #             for i in range(H - 1, -1, -1):
            #                 order4.append((i, j))
            #         else:
            #             for i in range(H):
            #                 order4.append((i, j))
            #     orders.append(order4)
            """蛇形扫描 - 优化版（列表推导式，快10-20倍）"""
            H, W = hw_shape
            orders = []

            # Method 1: 水平蛇形（偶数行左→右，奇数行右→左）
            if k >= 1:
                order1 = [
                    (dh + i, dw + j)
                    for i in range(H)
                    for j in (range(W) if i % 2 == 0 else range(W - 1, -1, -1))
                ]
                orders.append(order1)

            # Method 2: 水平蛇形（偶数行右→左，奇数行左→右）
            if k >= 2:
                order2 = [
                    (i, j)
                    for i in range(H)
                    for j in (range(W - 1, -1, -1) if i % 2 == 0 else range(W))
                ]
                orders.append(order2)

            # Method 3: 垂直蛇形（偶数列上→下，奇数列下→上）
            if k >= 3:
                order3 = [
                    (i, j)
                    for j in range(W)
                    for i in (range(H) if j % 2 == 0 else range(H - 1, -1, -1))
                ]
                orders.append(order3)

            # Method 4: 垂直蛇形（偶数列下→上，奇数列上→下）
            if k >= 4:
                order4 = [
                    (i, j)
                    for j in range(W)
                    for i in (range(H - 1, -1, -1) if j % 2 == 0 else range(H))
                ]
                orders.append(order4)
            # Methods 5-8: 逆序
            if k >= 5:
                orders.append(order1[::-1])
            if k >= 6:
                orders.append(order2[::-1])
            if k >= 7:
                orders.append(order3[::-1])
            if k == 8:
                orders.append(order4[::-1])

            return orders

        def get_permute_order_fast(hw_shape, dh=0, dw=0, k=8):
            """高效版本 - 使用 NumPy 向量化"""
            H, W = hw_shape
            L = H * W

            # 创建基础网格
            rows = np.arange(H)
            cols = np.arange(W)

            # Method 1: 蛇形行扫描 (偶数行左→右, 奇数行右→左)
            # 创建 (H, W) 的索引矩阵
            if k>=1:
                base = np.zeros((H, W, 2), dtype=np.int64)
                base[:, :, 0] = rows[:, None] + dh  # i indices
                base[:, :, 1] = cols[None, :] + dw  # j indices
                # 奇数行翻转
                base[1::2, :, 1] = base[1::2, ::-1, 1]
                order1 = base.reshape(-1, 2)
                orders = [order1]

            # Method 2: 蛇形行扫描 (偶数行右→左, 奇数行左→右)
            if k>=2:
                base2 = np.zeros((H, W, 2), dtype=np.int64)
                base2[:, :, 0] = rows[:, None]
                base2[:, :, 1] = cols[None, :]
                base2[0::2, :, 1] = base2[0::2, ::-1, 1]  # 偶数行翻转
                order2 = base2.reshape(-1, 2)
                orders = [order1, order2]

            # Method 3: 蛇形列扫描 (偶数列上→下, 奇数列下→上)
            if k>=3:
                base3 = np.zeros((W, H, 2), dtype=np.int64)  # 注意转置
                base3[:, :, 0] = rows[None, :]  # i
                base3[:, :, 1] = cols[:, None]  # j
                base3[1::2, :, 0] = base3[1::2, ::-1, 0]  # 奇数列翻转 i
                order3 = base3.reshape(-1, 2)
                orders = [order1, order2, order3]


            # Method 4: 蛇形列扫描 (偶数列下→上, 奇数列上→下)
            if k>=4:
                base4 = np.zeros((W, H, 2), dtype=np.int64)
                base4[:, :, 0] = rows[None, :]
                base4[:, :, 1] = cols[:, None]
                base4[0::2, :, 0] = base4[0::2, ::-1, 0]  # 偶数列翻转 i
                order4 = base4.reshape(-1, 2)
                orders = [order1, order2, order3, order4]

            # 根据 k 选择返回
            if k == 8:
                orders = [order1, order2, order3, order4,
                          order1[::-1].copy(), order2[::-1].copy(),
                          order3[::-1].copy(), order4[::-1].copy()]

            return orders

        def get_diagonal_order(hw_shape, k=8):
            # def main_diagonal_order(hw_shape):
            #     H, W = hw_shape
            #     order = []
            #     for diag in range(H + W - 1):
            #         for i in range(max(0, diag - W + 1), min(H, diag + 1)):
            #             j = diag - i
            #             order.append((i, j))
            #     return order
            #
            # def main_diagonal_order_rev(hw_shape):
            #     H, W = hw_shape
            #     order = []
            #     for diag in range(H + W - 1):
            #         for j in range(max(0, diag - H + 1), min(W, diag + 1)):
            #             i = diag - j
            #             order.append((i, j))
            #     return order
            #
            # def second_diagonal_order(hw_shape):
            #     H, W = hw_shape
            #     order = []
            #     for diag in range(-(H - 1), W):
            #         for i in range(min(H - 1, W - 1 - diag), max(-1, -diag - 1), -1):
            #             j = diag + i
            #             order.append((i, j))
            #     return order
            #
            # def second_diagonal_order_rev(hw_shape):
            #     H, W = hw_shape
            #     order = []
            #     for diag in range(-(H - 1), W):
            #         for i in range(max(0, -diag), min(H, W - diag)):
            #             j = diag + i
            #             order.append((i, j))
            #     return order
            #
            # orders = []
            # order1 = main_diagonal_order(hw_shape)
            # orders.append(order1)
            # # order2 = main_diagonal_order_rev(hw_shape)
            # # orders.append(order2)
            # # order3 = second_diagonal_order(hw_shape)
            # # orders.append(order3)
            # # order4 = second_diagonal_order_rev(hw_shape)
            # # orders.append(order4)
            # # orders.append(order1[::-1])
            """对角线扫描 - 优化版（列表推导式）"""

            def main_diagonal_order(hw_shape):
                H, W = hw_shape
                return [
                    (i, diag - i)
                    for diag in range(H + W - 1)
                    for i in range(max(0, diag - W + 1), min(H, diag + 1))
                ]

            def main_diagonal_order_rev(hw_shape):
                H, W = hw_shape
                return [
                    (diag - j, j)
                    for diag in range(H + W - 1)
                    for j in range(max(0, diag - H + 1), min(W, diag + 1))
                ]

            def second_diagonal_order(hw_shape):
                H, W = hw_shape
                return [
                    (i, diag + i)
                    for diag in range(-(H - 1), W)
                    for i in range(min(H - 1, W - 1 - diag), max(-1, -diag - 1), -1)
                ]

            def second_diagonal_order_rev(hw_shape):
                H, W = hw_shape
                return [
                    (i, diag + i)
                    for diag in range(-(H - 1), W)
                    for i in range(max(0, -diag), min(H, W - diag))
                ]

            orders = []
            if k >= 1:
                orders.append(main_diagonal_order(hw_shape))
            if k >= 2:
                orders.append(main_diagonal_order_rev(hw_shape))
            if k >= 3:
                orders.append(second_diagonal_order(hw_shape))
            if k >= 4:
                orders.append(second_diagonal_order_rev(hw_shape))
            if k >= 5:
                orders.append(orders[0][::-1])
            if k >= 6:
                orders.append(orders[1][::-1])
            if k >= 7:
                orders.append(orders[2][::-1])
            if k == 8:
                orders.append(orders[3][::-1])
            return orders

        def get_zigzag_order(hw_shape, k=8):
            # def get_zigzag_diagonal_order(hw_shape, rev: bool):
            #     H, W = hw_shape
            #     order = []
            #     for diag in range(H + W - 1):
            #         if diag % 2 == rev:
            #             for i in range(max(0, diag - W + 1), min(H, diag + 1)):
            #                 j = diag - i
            #                 order.append((i, j))
            #         else:
            #             for i in range(min(H, diag + 1) - 1, max(0, diag - W + 1) - 1, -1):
            #                 j = diag - i
            #                 order.append((i, j))
            #     return order
            #
            # def second_diagonal_order(hw_shape, rev: bool):
            #     H, W = hw_shape
            #     order = []
            #     for diag in range(H + W - 1):
            #         if diag % 2 == rev:
            #             for j in range(max(0, diag - H + 1), min(W, diag + 1)):
            #                 i = diag - j
            #                 if 0 <= i < H:
            #                     order.append((H - 1 - i, j))
            #         else:
            #             for j in range(max(0, diag - W + 1), min(H, diag + 1)):
            #                 i = diag - j
            #                 if 0 <= j < H:
            #                     order.append((H - 1 - j, i))
            #     return order
            #
            # order = []
            # order1 = get_zigzag_diagonal_order(hw_shape, False)
            # order.append(order1)
            # # order2 = get_zigzag_diagonal_order(hw_shape, True)
            # # order.append(order2)
            # # order3 = second_diagonal_order(hw_shape, False)
            # # order.append(order3)
            # # order4 = second_diagonal_order(hw_shape, True)
            # # order.append(order4)
            """锯齿形对角线扫描 - 优化版（列表推导式）"""

            def get_zigzag_diagonal_order(hw_shape, rev: bool):
                H, W = hw_shape
                result = []
                for diag in range(H + W - 1):
                    if diag % 2 == rev:
                        result.extend([
                            (i, diag - i)
                            for i in range(max(0, diag - W + 1), min(H, diag + 1))
                        ])
                    else:
                        result.extend([
                            (i, diag - i)
                            for i in range(min(H, diag + 1) - 1, max(0, diag - W + 1) - 1, -1)
                        ])
                return result

            def second_diagonal_order(hw_shape, rev: bool):
                H, W = hw_shape
                result = []
                for diag in range(H + W - 1):
                    if diag % 2 == rev:
                        result.extend([
                            (H - 1 - i, j)
                            for j in range(max(0, diag - H + 1), min(W, diag + 1))
                            for i in [diag - j]
                            if 0 <= i < H
                        ])
                    else:
                        result.extend([
                            (H - 1 - j, i)
                            for j in range(max(0, diag - W + 1), min(H, diag + 1))
                            for i in [diag - j]
                            if 0 <= j < H
                        ])
                return result

            orders = []
            if k >= 1:
                orders.append(get_zigzag_diagonal_order(hw_shape, False))
            if k >= 2:
                orders.append(get_zigzag_diagonal_order(hw_shape, True))
            if k >= 3:
                orders.append(second_diagonal_order(hw_shape, False))
            if k >= 4:
                orders.append(second_diagonal_order(hw_shape, True))
            if k >= 5:
                orders.append(orders[0][::-1])
            if k >= 6:
                orders.append(orders[1][::-1])
            if k >= 7:
                orders.append(orders[2][::-1])
            if k == 8:
                orders.append(orders[3][::-1])
            return orders

        def get_spiral_order(hw_shape, k=8):
            # H, W = hw_shape
            # orders = []
            # direction_order = {
            #     'left_to_right': [0, 1, 2, 3],
            #     'top_to_bottom': [1, 2, 3, 0],
            #     'right_to_left': [2, 3, 0, 1],
            #     'bottom_to_top': [3, 0, 1, 2]
            # }
            #
            # def spiral_indices(H, W, order):
            #     current_order = direction_order.get(order, direction_order['left_to_right'])
            #     top, bottom, left, right = 0, H - 1, 0, W - 1
            #     order = []
            #     while top <= bottom and left <= right:
            #         for index, direction in enumerate(current_order):
            #             if direction == 0 and top <= bottom:  # 从左到右
            #                 for j in range(left, right + 1):
            #                     order.append((top, j))
            #                 top += 1
            #             elif direction == 1 and left <= right:  # 从上到下
            #                 for i in range(top, bottom + 1):
            #                     order.append((i, right))
            #                 right -= 1
            #             elif direction == 2 and top <= bottom:  # 从右到左
            #                 for j in range(right, left - 1, -1):
            #                     order.append((bottom, j))
            #                 bottom -= 1
            #             elif direction == 3 and left <= right:  # 从下到上
            #                 for i in range(bottom, top - 1, -1):
            #                     order.append((i, left))
            #                 left += 1
            #     return order
            #
            # order1 = spiral_indices(H, W, 'left_to_right')
            # orders.append(order1)
            # # order2 = spiral_indices(H, W, 'top_to_bottom')
            # # orders.append(order2)
            # # order3 = spiral_indices(H, W, 'right_to_left')
            # # orders.append(order3)
            # # order4 = spiral_indices(H, W, 'bottom_to_top')
            # # orders.append(order4)
            # return orders
            """螺旋扫描 - 优化版（减少append）"""
            H, W = hw_shape
            orders = []
            direction_order = {
                'left_to_right': [0, 1, 2, 3],
                'top_to_bottom': [1, 2, 3, 0],
                'right_to_left': [2, 3, 0, 1],
                'bottom_to_top': [3, 0, 1, 2]
            }

            def spiral_indices(H, W, order_key):
                current_order = direction_order.get(order_key, direction_order['left_to_right'])
                top, bottom, left, right = 0, H - 1, 0, W - 1
                result = []
                while top <= bottom and left <= right:
                    for direction in current_order:
                        if direction == 0 and top <= bottom:  # 从左到右
                            result.extend([(top, j) for j in range(left, right + 1)])
                            top += 1
                        elif direction == 1 and left <= right:  # 从上到下
                            result.extend([(i, right) for i in range(top, bottom + 1)])
                            right -= 1
                        elif direction == 2 and top <= bottom:  # 从右到左
                            result.extend([(bottom, j) for j in range(right, left - 1, -1)])
                            bottom -= 1
                        elif direction == 3 and left <= right:  # 从下到上
                            result.extend([(i, left) for i in range(bottom, top - 1, -1)])
                            left += 1
                return result

            if k >= 1:
                orders.append(spiral_indices(H, W, 'left_to_right'))
            if k >= 2:
                orders.append(spiral_indices(H, W, 'top_to_bottom'))
            if k >= 3:
                orders.append(spiral_indices(H, W, 'right_to_left'))
            if k >= 4:
                orders.append(spiral_indices(H, W, 'bottom_to_top'))
            if k >= 5:
                orders.append(orders[0][::-1])
            if k >= 6:
                orders.append(orders[1][::-1])
            if k >= 7:
                orders.append(orders[2][::-1])
            if k == 8:
                orders.append(orders[3][::-1])
            return orders

        def get_efficient_order(hw_shape, k=8):
            # H, W = hw_shape
            # orders = []
            # # =======1========
            # order1 = []
            # for i in range(H):
            #     for j in range(0, W, 2):
            #         order1.append((i, j))
            #     for j in range(1, W, 2):
            #         order1.append((i, j))
            # orders.append(order1)
            # # #=======2=======
            # # order2 = []
            # # for j in range(W):
            # #     for i in range(0, H, 2):
            # #         order2.append((i, j))
            # #     for i in range(1, H, 2):
            # #         order2.append((i, j))
            # # orders.append(order2)
            # # # #========3=======
            # # order3 = []
            # # for i in range(H):
            # #     for j in range(W-1, -1, -2):
            # #         order3.append((i, j))
            # #     for j in range(W-2, -1, -2):
            # #         order3.append((i, j))
            # # orders.append(order3)
            # # # #========4========
            # # order4 = []
            # # for j in range(W-1, -1, -1):
            # #     for i in range(0, H, 2):
            # #         order4.append((i, j))
            # #     for i in range(1, H, 2):
            # #         order4.append((i, j))
            # # orders.append(order4)
            #
            # return orders
            """先偶后奇扫描 - 优化版（列表推导式）"""
            H, W = hw_shape
            orders = []

            # Method 1: 行优先，每行先偶数列后奇数列
            if k >= 1:
                order1 = [
                    (i, j)
                    for i in range(H)
                    for j in list(range(0, W, 2)) + list(range(1, W, 2))
                ]
                orders.append(order1)

            # Method 2: 列优先，每列先偶数行后奇数行
            if k >= 2:
                order2 = [
                    (i, j)
                    for j in range(W)
                    for i in list(range(0, H, 2)) + list(range(1, H, 2))
                ]
                orders.append(order2)

            # Method 3: 行优先，每行从右到左（先奇数列后偶数列）
            if k >= 3:
                order3 = [
                    (i, j)
                    for i in range(H)
                    for j in list(range(W - 1, -1, -2)) + list(range(W - 2, -1, -2))
                ]
                orders.append(order3)

            # Method 4: 列优先，从右到左（先偶数行后奇数行）
            if k >= 4:
                order4 = [
                    (i, j)
                    for j in range(W - 1, -1, -1)
                    for i in list(range(0, H, 2)) + list(range(1, H, 2))
                ]
                orders.append(order4)

            if k >= 5:
                orders.append(order1[::-1])
            if k >= 6:
                orders.append(order2[::-1])
            if k >= 7:
                orders.append(order3[::-1])
            if k == 8:
                orders.append(order4[::-1])

            return orders

        # def hilbert_curve_order(size):
        #     orders = []
        #
        #     order1 = []
        #
        #     def hilbert_recursive(n, x, y, xi, xj, yi, yj, i):
        #         if n == 0:
        #             px = x + (xi + yi) // 2
        #             py = y + (xj + yj) // 2
        #             order1.append((px+i, py))
        #         else:
        #             hilbert_recursive(n - 1, x, y, yi // 2, yj // 2, xi // 2, xj // 2)
        #             hilbert_recursive(n - 1, x + xi // 2, y + xj // 2, xi // 2, xj // 2, yi // 2, yj // 2)
        #             hilbert_recursive(n - 1, x + xi // 2 + yi // 2, y + xj // 2 + yj // 2, xi // 2, xj // 2, yi // 2,
        #                               yj // 2)
        #             hilbert_recursive(n - 1, x + xi // 2 + yi, y + xj // 2 + yj, -yi // 2, -yj // 2, -xi // 2, -xj // 2)
        #
        #     order = int(np.log2(size))
        #     hilbert_recursive(order, 0, 0, size, 0, 0, size, 0)
        #     hilbert_recursive(order, 0, 0, size, 0, 0, size, size)
        #     orders.append(order1)
        #     return orders

        def hilbert_curve_order(width, height, direction='up'):
            """
            Generate Hilbert curve points with specified start direction.

            :param width: Width of the space.
            :param height: Height of the space (must be a power of two).
            :param direction: Start direction ('right', 'left', 'up', 'down').
            :return: List of points on the Hilbert curve.
            """
            if height & (height - 1) != 0:  # Check if height is a power of two
                raise ValueError("高度必须是 2 的幂")
            if width % height != 0:  # Check if width is an integer multiple of height
                raise ValueError("宽度必须是高度的整数倍")

            block_width = height
            num_blocks = width // block_width

            all_points = []
            all_point = []

            def hilbert_recursive(n, x, y, xi, xj, yi, yj, order_list):
                if n == 0:
                    px = x + (xi + yi) // 2
                    py = y + (xj + yj) // 2
                    order_list.append((py, px))
                else:
                    hilbert_recursive(n - 1, x, y, yi // 2, yj // 2, xi // 2, xj // 2, order_list)
                    hilbert_recursive(n - 1, x + xi // 2, y + xj // 2, xi // 2, xj // 2, yi // 2, yj // 2, order_list)
                    hilbert_recursive(n - 1, x + xi // 2 + yi // 2, y + xj // 2 + yj // 2, xi // 2, xj // 2, yi // 2,
                                      yj // 2, order_list)
                    hilbert_recursive(n - 1, x + xi // 2 + yi, y + xj // 2 + yj, -yi // 2, -yj // 2, -xi // 2, -xj // 2,
                                      order_list)

            for block_idx in range(num_blocks):
                offset_x = block_idx * block_width
                order1 = []

                if direction == 'right':
                    hilbert_recursive(int(np.log2(block_width)), 0, 0, block_width, 0, 0, block_width, order1)
                elif direction == 'left':
                    hilbert_recursive(int(np.log2(block_width)), 0, 0, -block_width, 0, 0, block_width, order1)
                elif direction == 'up':
                    hilbert_recursive(int(np.log2(block_width)), 0, 0, 0, -block_width, block_width, 0, order1)
                elif direction == 'down':
                    hilbert_recursive(int(np.log2(block_width)), 0, 0, 0, block_width, block_width, 0, order1)
                else:
                    raise ValueError("Unsupported direction. Choose from 'right', 'left', 'up', 'down'.")

                adjusted_points = [(p[0] + offset_x, p[1]) for p in order1]
                all_point.extend(adjusted_points)

            all_points.append(all_point)
            return all_points

        def get_blockwise_order(hw_shape, block_h=8, block_w=8, scan=None, k=4):
            # h, w = hw_shape
            # # print(f'block: H{h}, W{w}')
            # block_h, block_w = h // 8, w // 8
            # orders = []
            # order1 = []
            # for i in range(0, h, block_h):
            #     for j in range(0, w, block_w):
            #         h_block = min(h - i, block_h)
            #         w_block = min(w - j, block_w)
            #         order = get_permute_order((h_block, w_block), i, j)
            #         order1.extend(order)
            #
            # orders.append(order1)
            # return orders
            """块扫描 - 优化版（列表推导式）

            返回 shape (k, H*W, 2) 的坐标列表
            - k 个扫描方向
            - 每个方向包含 H*W 个坐标
            """
            h, w = hw_shape
            # print(f'block: H{h}, W{w}')
            block_h, block_w = h//8, w//8
            k = self.k_group

            # 为k个方向分别创建列表
            orders = [[] for _ in range(k)]

            # 遍历所有块
            for i in range(0, h, block_h):
                for j in range(0, w, block_w):
                    h_block = min(h-i, block_h)
                    w_block = min(w-j, block_w)
                    # 获取当前块的k个扫描方向
                    block_orders = get_permute_order((h_block, w_block), i, j, k=k)

                    # 把每个块的第k_idx个方向添加到整体的第k_idx个方向
                    for k_idx in range(k):
                        orders[k_idx].extend(block_orders[k_idx])

            return orders  # shape: (k, H*W, 2)

        # 随机的
        def get_random_order(hw_shape, k=8 ):
            # H, W = hw_shape
            # # 1. 先生成所有 (h, w) 坐标对
            # orders = []
            # order = []
            # for h in range(H):
            #     for w in range(W):
            #         order.append([h, w])  # 每个是 [h, w]
            # # 2. 转为 numpy 或直接 shuffle
            # import random
            # random.shuffle(order)  # 随机打乱
            # # 3. 包装成 list[list]：这里是一个方向，所以是 [coords]
            # orders.append(order)  # list of list of list: shape (1 , H*W, 2)
            #
            # return orders
            """随机扫描 - 优化版（列表推导式，支持多方向）"""
            H, W = hw_shape
            import random

            if k is None:
                k = self.k_group  # 使用类的k_group属性

            # 生成基础坐标列表
            base_coords = [(h, w) for h in range(H) for w in range(W)]

            orders = []
            for _ in range(k):
                # 为每个方向生成不同的随机排列
                order = base_coords.copy()
                random.shuffle(order)
                orders.append(order)

            return orders  # 返回格式: list of list, shape (k, H*W, 2)


        def get_permute_order_fast_1d(hw_shape, k=8):
            """高效版本 - 直接返回 1D 索引 (K, L)，避免后续转换"""
            H, W = hw_shape
            L = H * W

            # 创建基础行主序索引
            base_idx = np.arange(L).reshape(H, W)

            # Method 1: 蛇形行扫描 (偶数行左→右, 奇数行右→左)
            order1 = base_idx.copy()
            order1[1::2] = order1[1::2, ::-1]
            order1 = order1.ravel()

            # Method 2: 蛇形行扫描 (偶数行右→左, 奇数行左→右)
            order2 = base_idx.copy()
            order2[0::2] = order2[0::2, ::-1]
            order2 = order2.ravel()

            # Method 3: 蛇形列扫描 (偶数列上→下, 奇数列下→上)
            order3 = base_idx.T.copy()  # 转置得到列主序
            order3[1::2] = order3[1::2, ::-1]
            order3 = order3.ravel()

            # Method 4: 蛇形列扫描 (偶数列下→上, 奇数列上→下)
            order4 = base_idx.T.copy()
            order4[0::2] = order4[0::2, ::-1]
            order4 = order4.ravel()

            # 根据 k 选择返回
            if k == 1:
                orders = np.stack([order1])
            elif k == 2:
                orders = np.stack([order1, order2])
            elif k == 4:
                orders = np.stack([order1, order2, order3, order4])
            elif k == 8:
                orders = np.stack([order1, order2, order3, order4,
                                   order1[::-1], order2[::-1], order3[::-1], order4[::-1]])
            else:
                orders = np.stack([order1, order2, order3, order4])

            return orders  # shape: (K, L)

        # ========== 可学习的扫描方法 ==========
        # 注意：可学习扫描需要输入特征图x，输出是基于内容的动态扫描顺序
        def get_learnable_order(x, k=None):
            """
            可学习的扫描顺序 - 基于输入特征图动态生成扫描顺序

            支持两种模式：
            1. use_independent_learnable=True: 每个SS2D实例有独立参数
            2. use_independent_learnable=False: 懒加载，所有step共享参数

            Args:
                x: (B, C, H, W) 输入特征图
                k: 扫描方向数，如果为None则使用self.k_group

            Returns:
                orders: (k, H*W, 2) - 2D坐标形式
            """
            B, C, H, W = x.shape
            if k is None:
                k = self.k_group
            L = H * W

            # 选择网络：独立参数 / stage共享 / global共享 / 懒加载（已废弃）
            if self.use_independent_learnable:
                # 模式1：使用__init__中初始化的独立网络
                score_net = self.learnable_score_net
                scores = score_net(x)  # (B, k, H, W)
            elif self._get_shared_nets() is not None:
                # 检查是否为全局共享模式（方案1：共享Depthwise + 独立Pointwise）
                if "global_depthwise" in self._get_shared_nets():
                    # 模式2a：全局共享模式
                    B, C, H, W = x.shape
                    depthwise = self._get_shared_nets()["global_depthwise"]
                    bn = self._get_shared_nets()["global_bn"]
                    act = self._get_shared_nets()["global_act"]
                    stage_key = f"stage_{self.stage_index}"
                    pointwise = self._get_shared_nets()[f"{stage_key}_pointwise"]

                    # 使用参数切片处理Depthwise和BN
                    weight_dw = depthwise.weight[:C, :, :, :]
                    x = torch.nn.functional.conv2d(x, weight_dw, None, stride=1, padding=1, groups=C)

                    weight_bn = bn.weight[:C]
                    bias_bn = bn.bias[:C]
                    running_mean = bn.running_mean[:C]
                    running_var = bn.running_var[:C]
                    x = torch.nn.functional.batch_norm(x, running_mean, running_var,
                                                       weight_bn, bias_bn,
                                                       training=bn.training,
                                                       momentum=bn.momentum,
                                                       eps=bn.eps)
                    x = act(x)
                    scores = pointwise(x)  # 独立的Pointwise，维度匹配
                else:
                    # 模式2b：stage级别共享（4个网络）
                    stage_key = f"stage_{self.stage_index}"
                    score_net = self._get_shared_nets()[stage_key]
                    scores = score_net(x)  # (B, k, H, W)
            else:
                # 模式3：懒加载（已废弃，仅用于向后兼容）
                if not hasattr(self, '_shared_learnable_score_net'):
                    self._shared_learnable_score_net = nn.Sequential(
                        nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False),
                        nn.BatchNorm2d(C),
                        nn.GELU(),
                        nn.Conv2d(C, k, 1, bias=True)
                    ).to(x.device)
                    self._init_learnable_weights(self._shared_learnable_score_net)
                score_net = self._shared_learnable_score_net
                scores = score_net(x)  # (B, k, H, W)

            # 将scores展平为 (B, k, L)
            scores = scores.view(B, k, L)  # (B, k, L)

            # 训练时计算辅助损失
            if self.training and (self.scan_diversity_weight > 0 or self.scan_coverage_weight > 0):
                if self.scan_diversity_weight > 0:
                    diversity_loss = compute_scan_diversity_loss(scores) * self.scan_diversity_weight
                    _InlineScanAuxLossCache.register('inline_scan_diversity', diversity_loss)

                if self.scan_coverage_weight > 0:
                    coverage_loss = compute_scan_coverage_loss(scores) * self.scan_coverage_weight
                    _InlineScanAuxLossCache.register('inline_scan_coverage', coverage_loss)

            # 按score降序排列得到1D索引 (B, k, L)
            indices_1d = scores.argsort(dim=-1, descending=True)  # (B, k, L)

            # 转换1D索引到2D坐标（完全在GPU上计算）
            # 取第一个batch
            indices_1d = indices_1d[0]  # (k, L)

            # 转换为2D坐标
            rows = indices_1d // W  # (k, L)
            cols = indices_1d % W   # (k, L)

            # 堆叠为 (k, L, 2) 格式
            orders = torch.stack([rows, cols], dim=-1)  # (k, L, 2)

            return orders  # 返回 tensor (k, H*W, 2)

        # ========== 可学习的扫描方法 V2 - 基于偏移预测 ==========
        def get_learnable_order_v2(x, k=None):
            """
            可学习的扫描顺序 V2 - 基于偏移预测

            每个方向使用独立的网络预测重要性分数

            支持两种模式：
            1. use_independent_learnable=True: 每个SS2D实例有独立参数
            2. use_independent_learnable=False: 懒加载，所有step共享参数

            Args:
                x: (B, C, H, W) 输入特征图
                k: 扫描方向数

            Returns:
                orders: (k, H*W, 2) - 2D坐标形式
            """
            B, C, H, W = x.shape
            if k is None:
                k = self.k_group
            L = H * W

            # 选择网络：独立参数 / stage共享 / global共享 / 懒加载（已废弃）
            if self.use_independent_learnable:
                # 模式1：使用__init__中初始化的独立网络
                offset_nets = self.learnable_offset_nets
                pos_bias = self.learnable_pos_bias
            elif self._get_shared_nets() is not None:
                # 检查是否为全局共享模式（方案1）
                if "global_conv1_dir0" in self._get_shared_nets():
                    # 模式2a：全局共享模式（共享特征提取 + 独立输出层）
                    # 创建一个包装类来模拟offset_nets的接口
                    class GlobalOffsetNetsWrapper:
                        def __init__(self, shared_nets, stage_index, C, scan_k):
                            self.shared_nets = shared_nets
                            self.stage_index = stage_index
                            self.C = C
                            self.scan_k = scan_k

                        def __getitem__(self, idx):
                            def forward_fn(x):
                                # 使用共享的conv1和bn1（参数切片）
                                conv1 = self.shared_nets[f"global_conv1_dir{idx}"]
                                bn1 = self.shared_nets[f"global_bn1_dir{idx}"]
                                act = self.shared_nets[f"global_act_dir{idx}"]
                                stage_key = f"stage_{self.stage_index}"
                                conv2 = self.shared_nets[f"{stage_key}_conv2_dir{idx}"]

                                out_c = self.C // 4
                                weight1 = conv1.weight[:out_c, :self.C, :, :]
                                x = torch.nn.functional.conv2d(x, weight1, None, stride=1, padding=1)

                                weight_bn = bn1.weight[:out_c]
                                bias_bn = bn1.bias[:out_c]
                                running_mean = bn1.running_mean[:out_c]
                                running_var = bn1.running_var[:out_c]
                                x = torch.nn.functional.batch_norm(x, running_mean, running_var,
                                                                   weight_bn, bias_bn,
                                                                   training=bn1.training,
                                                                   momentum=bn1.momentum,
                                                                   eps=bn1.eps)
                                x = act(x)
                                x = conv2(x)  # 独立的输出层，维度匹配
                                return x
                            return forward_fn

                    offset_nets = GlobalOffsetNetsWrapper(self._get_shared_nets(), self.stage_index, C, k)
                    pos_bias = self._get_shared_params()["global_pos_bias"]
                else:
                    # 模式2b：stage级别共享（4个网络）
                    stage_key = f"stage_{self.stage_index}"
                    offset_nets = self._get_shared_nets()[stage_key]
                    pos_bias = self._get_shared_params()[f"{stage_key}_pos_bias"]
            else:
                # 模式3：懒加载（已废弃，仅用于向后兼容）
                if not hasattr(self, '_shared_learnable_offset_nets'):
                    self._shared_learnable_offset_nets = nn.ModuleList()
                    for i in range(k):
                        net = nn.Sequential(
                            nn.Conv2d(C, C // 4, 3, 1, 1, bias=False),
                            nn.BatchNorm2d(C // 4),
                            nn.GELU(),
                            nn.Conv2d(C // 4, 1, 1, bias=True)
                        ).to(x.device)
                        # 不同方向使用不同的初始化
                        for m in net.modules():
                            if isinstance(m, nn.Conv2d):
                                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                                if m.bias is not None:
                                    nn.init.constant_(m.bias, 0.1 * i)
                            elif isinstance(m, nn.BatchNorm2d):
                                nn.init.ones_(m.weight)
                                nn.init.zeros_(m.bias)
                        self._shared_learnable_offset_nets.append(net)
                    self._shared_learnable_pos_bias = nn.Parameter(torch.randn(k, 1, 1).to(x.device))
                offset_nets = self._shared_learnable_offset_nets
                pos_bias = self._shared_learnable_pos_bias

            # 生成每个方向的score并收集
            all_scores = []  # 用于计算辅助损失
            all_indices = []  # 收集所有方向的索引
            for i in range(k):
                # 预测score map（动态网络会自动处理不同维度）
                score = offset_nets[i](x)  # (B, 1, H, W)
                score = score.view(B, L)  # (B, L)

                # 添加位置编码偏置
                y_coords = torch.arange(H, device=x.device).float().view(H, 1).expand(H, W)
                x_coords = torch.arange(W, device=x.device).float().view(1, W).expand(H, W)

                # 不同方向的位置偏好
                if i % 4 == 0:
                    pos_bias_val = y_coords + x_coords  # 左上优先
                elif i % 4 == 1:
                    pos_bias_val = y_coords - x_coords  # 右上优先
                elif i % 4 == 2:
                    pos_bias_val = -y_coords + x_coords  # 左下优先
                else:
                    pos_bias_val = -y_coords - x_coords  # 右下优先

                if i >= 4:
                    pos_bias_val = -pos_bias_val  # 反向

                score = score + pos_bias_val.view(1, L) * pos_bias[i]
                all_scores.append(score)  # (B, L)

                # 排序并转换为2D坐标（完全在GPU上）
                idx_1d = score[0].argsort(descending=True)  # 取第一个batch, (L,)
                rows = idx_1d // W  # (L,)
                cols = idx_1d % W   # (L,)
                coords = torch.stack([rows, cols], dim=-1)  # (L, 2)
                all_indices.append(coords)

            # 堆叠所有方向的索引
            orders = torch.stack(all_indices, dim=0)  # (k, L, 2)

            # 训练时计算辅助损失
            if self.training and (self.scan_diversity_weight > 0 or self.scan_coverage_weight > 0):
                # 合并所有scores: (B, k, L)
                scores = torch.stack(all_scores, dim=1)  # (B, k, L)

                if self.scan_diversity_weight > 0:
                    diversity_loss = compute_scan_diversity_loss(scores) * self.scan_diversity_weight
                    _InlineScanAuxLossCache.register('inline_scan_diversity_v2', diversity_loss)

                if self.scan_coverage_weight > 0:
                    coverage_loss = compute_scan_coverage_loss(scores) * self.scan_coverage_weight
                    _InlineScanAuxLossCache.register('inline_scan_coverage_v2', coverage_loss)

            return orders  # (k, H*W, 2)

        # ========== 内容自适应扫描方法 ==========
        def get_content_adaptive_order(x, k=None):
            """
            内容自适应扫描顺序 - 根据图像内容自适应调整扫描顺序

            支持两种模式：
            1. use_independent_learnable=True: 每个SS2D实例有独立参数
            2. use_independent_learnable=False: 懒加载，所有step共享参数

            Args:
                x: (B, C, H, W) 输入特征图
                k: 扫描方向数

            Returns:
                orders: (k, H*W, 2) - 2D坐标形式
            """
            B, C, H, W = x.shape
            if k is None:
                k = self.k_group
            L = H * W

            # 选择网络：独立参数 / stage共享 / global共享 / 懒加载（已废弃）
            if self.use_independent_learnable:
                # 模式1：使用__init__中初始化的独立网络
                content_net = self.content_net
                scan_heads = self.scan_heads
                direction_embeds = self.direction_embeds
            elif self._get_shared_nets() is not None:
                # 检查是否为全局共享模式（方案1）
                if "global_content_dw" in self._get_shared_nets():
                    # 模式2a：全局共享模式（共享内容提取 + 独立扫描头）
                    # 创建包装类来处理参数切片
                    class GlobalContentNetWrapper:
                        def __init__(self, shared_nets, C):
                            self.dw = shared_nets["global_content_dw"]
                            self.bn = shared_nets["global_content_bn"]
                            self.act = shared_nets["global_content_act"]
                            self.pw = shared_nets["global_content_pw"]
                            self.C = C

                        def __call__(self, x):
                            # Depthwise
                            weight_dw = self.dw.weight[:self.C, :, :, :]
                            x = torch.nn.functional.conv2d(x, weight_dw, None, stride=1, padding=1, groups=self.C)

                            # BatchNorm
                            weight_bn = self.bn.weight[:self.C]
                            bias_bn = self.bn.bias[:self.C]
                            running_mean = self.bn.running_mean[:self.C]
                            running_var = self.bn.running_var[:self.C]
                            x = torch.nn.functional.batch_norm(x, running_mean, running_var,
                                                               weight_bn, bias_bn,
                                                               training=self.bn.training,
                                                               momentum=self.bn.momentum,
                                                               eps=self.bn.eps)
                            x = self.act(x)

                            # Pointwise
                            weight_pw = self.pw.weight[:self.C, :self.C, :, :]
                            x = torch.nn.functional.conv2d(x, weight_pw, None, stride=1, padding=0)
                            return x

                    class GlobalScanHeadsWrapper:
                        def __init__(self, shared_nets, stage_index, scan_k):
                            self.shared_nets = shared_nets
                            self.stage_index = stage_index
                            self.scan_k = scan_k

                        def __getitem__(self, idx):
                            stage_key = f"stage_{self.stage_index}"
                            return self.shared_nets[f"{stage_key}_head_dir{idx}"]

                    content_net = GlobalContentNetWrapper(self._get_shared_nets(), C)
                    scan_heads = GlobalScanHeadsWrapper(self._get_shared_nets(), self.stage_index, k)
                    direction_embeds = self._get_shared_params()["global_pos_bias"]
                else:
                    # 模式2b：stage级别共享（4个网络）
                    stage_key = f"stage_{self.stage_index}"
                    content_net = self._get_shared_nets()[f"{stage_key}_content"]
                    scan_heads = self._get_shared_nets()[f"{stage_key}_heads"]
                    direction_embeds = self._get_shared_params()[f"{stage_key}_embeds"]
            else:
                # 模式3：懒加载（已废弃，仅用于向后兼容）
                if not hasattr(self, '_shared_content_net'):
                    self._shared_content_net = nn.Sequential(
                        nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False),
                        nn.BatchNorm2d(C),
                        nn.GELU(),
                        nn.Conv2d(C, C // 2, 1, bias=False),
                        nn.BatchNorm2d(C // 2),
                        nn.GELU(),
                    ).to(x.device)
                    # k个方向的扫描头
                    self._shared_scan_heads = nn.ModuleList([
                        nn.Conv2d(C // 2, 1, 1, bias=True).to(x.device)
                        for _ in range(k)
                    ])
                    # 可学习的方向嵌入
                    self._shared_direction_embeds = nn.Parameter(
                        torch.randn(k, C // 2, 1, 1, device=x.device) * 0.02
                    )
                    # 初始化权重
                    self._init_learnable_weights(self._shared_content_net)
                    for head in self._shared_scan_heads:
                        self._init_learnable_weights(head)
                content_net = self._shared_content_net
                scan_heads = self._shared_scan_heads
                direction_embeds = self._shared_direction_embeds

            # 提取内容特征（动态网络会自动处理不同维度）
            content_feat = content_net(x)  # (B, C, H, W)

            # 生成每个方向的score并收集
            all_scores = []  # 用于计算辅助损失
            all_indices = []  # 收集所有方向的索引
            for i in range(k):
                # 添加方向嵌入
                feat = content_feat + direction_embeds[i]  # (B, C//2, H, W)

                # 生成score
                score = scan_heads[i](feat)  # (B, 1, H, W)
                score = score.view(B, L)  # (B, L)
                all_scores.append(score)

                # 排序并转换为2D坐标（完全在GPU上）
                idx_1d = score[0].argsort(descending=True)  # 取第一个batch, (L,)
                rows = idx_1d // W  # (L,)
                cols = idx_1d % W   # (L,)
                coords = torch.stack([rows, cols], dim=-1)  # (L, 2)
                all_indices.append(coords)

            # 堆叠所有方向的索引
            orders = torch.stack(all_indices, dim=0)  # (k, L, 2)

            # 训练时计算辅助损失
            if self.training and (self.scan_diversity_weight > 0 or self.scan_coverage_weight > 0):
                # 合并所有scores: (B, k, L)
                scores = torch.stack(all_scores, dim=1)  # (B, k, L)

                if self.scan_diversity_weight > 0:
                    diversity_loss = compute_scan_diversity_loss(scores) * self.scan_diversity_weight
                    _InlineScanAuxLossCache.register('inline_scan_diversity_ca', diversity_loss)

                if self.scan_coverage_weight > 0:
                    coverage_loss = compute_scan_coverage_loss(scores) * self.scan_coverage_weight
                    _InlineScanAuxLossCache.register('inline_scan_coverage_ca', coverage_loss)

            return orders  # (k, H*W, 2)

        #Generate scan indices using scan generator
        # if self.scan_learnable or self.scan_type == "learnable":
        #     # LearnableScanOrder: input (B, C, H, W), output (B, k, L)
        #     indices = self.scan_generator(x)
        # else:
        #     # ScanOrderGenerator: input (H, W, device), output (k, L)
        #     indices = self.scan_generator(H, W, device=x.device)

        # # ========== 关键优化：indices缓存机制 ==========
        # # 检查是否已经缓存了这个尺寸的indices
        # cache_key = (H, W)
        # if not hasattr(self, '_indices_cache'):
        #     self._indices_cache = {}
        #
        # if cache_key not in self._indices_cache:
        #     # 第一次遇到这个尺寸，计算并缓存
        #     # print(f'[Cache Miss] Computing indices for H={H}, W={W}')
        #     if self.scan_type=="cross":
        #         indices = get_cross_order((H, W), k=self.k_group )
        #     elif self.scan_type=="permute":
        #         indices = get_permute_order((H, W), k=self.k_group)
        #     elif self.scan_type=="diagonal":
        #         indices = get_diagonal_order((H, W), k=self.k_group)
        #     elif self.scan_type == "zigzag":
        #         indices = get_zigzag_order((H, W), k=self.k_group)
        #     elif self.scan_type == "spiral":
        #         indices = get_spiral_order((H, W), k=self.k_group)
        #     elif self.scan_type == "efficient":
        #         indices = get_efficient_order((H, W), k=self.k_group)
        #     elif self.scan_type == "blockwise":
        #         indices = get_blockwise_order((H, W), k=self.k_group)
        #     # indices = hilbert_curve_order(W, H)
        #     elif self.scan_type == "random":
        #         print("random")
        #         indices = get_random_order((H, W), k=self.k_group)
        #     else:
        #         print("###=====================###")
        #         indices = get_cross_order((H, W), k=self.k_group)
        #     indices = torch.tensor(indices, device='cuda')
        #     self._indices_cache[cache_key] = indices
        # else:
        #     # 使用缓存，避免重复计算！
        #     indices = self._indices_cache[cache_key]
        #
        # # ========== 缓存机制结束 ==========
        # ========== 可学习扫描方法（不使用缓存，每次动态计算）==========
        # 调试输出
        # print(f"[DEBUG] scan_type={self.scan_type}, type={type(self.scan_type)}")

        if self.scan_type in ["learnable", "learnable_v2", "content_adaptive"]:
            # print(f"[DEBUG] Using learnable scan: {self.scan_type}, training={self.training}")
            if self.scan_type == "learnable":
                indices = get_learnable_order(x)
            elif self.scan_type == "learnable_v2":
                indices = get_learnable_order_v2(x)
            elif self.scan_type == "content_adaptive":
                indices = get_content_adaptive_order(x)
            # 函数已经直接返回tensor，无需转换
        else:
            # ========== 固定扫描方法：使用indices缓存机制 ==========
            # 检查是否已经缓存了这个尺寸的indices
            cache_key = (H, W)
            if not hasattr(self, '_indices_cache'):
                self._indices_cache = {}

            if cache_key not in self._indices_cache:
                # 第一次遇到这个尺寸，计算并缓存
                # print(f'[Cache Miss] Computing indices for H={H}, W={W}')
                if self.scan_type == "cross":
                    indices = get_cross_order((H, W), k=self.k_group)
                elif self.scan_type == "permute":
                    indices = get_permute_order((H, W), k=self.k_group)
                elif self.scan_type == "diagonal":
                    indices = get_diagonal_order((H, W), k=self.k_group)
                elif self.scan_type == "zigzag":
                    indices = get_zigzag_order((H, W), k=self.k_group)
                elif self.scan_type == "spiral":
                    indices = get_spiral_order((H, W), k=self.k_group)
                elif self.scan_type == "efficient":
                    indices = get_efficient_order((H, W), k=self.k_group)
                elif self.scan_type == "blockwise":
                    indices = get_blockwise_order((H, W), k=self.k_group)
                # indices = hilbert_curve_order(W, H)
                elif self.scan_type == "random":
                    print("random")
                    indices = get_random_order((H, W), k=self.k_group)
                else:
                    print("###=====================###")
                    indices = get_cross_order((H, W), k=self.k_group)
                indices = torch.tensor(indices, device='cuda')
                self._indices_cache[cache_key] = indices
            else:
                # 使用缓存，避免重复计算！
                indices = self._indices_cache[cache_key]

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        x_proj_bias = getattr(self, "x_proj_bias", None)
        xs = cross_scan_fn(x, indices, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                           force_torch=scan_force_torch)  # 添加indices
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1),
                             bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K , -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.contiguous().view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
        Ds = self.Ds.to(torch.float)  # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)

        # save_dict = {}
        # #计算全矩阵
        # if compute_attn_matrix:
        #     xai_matrix = compute_attn_matrix_fn(delta, delta_bias, A=A, B=B, C=C, L=L, x_shape=x.shape)
        #     save_dict["attention_matrix"] = xai_matrix
        #     xai_vector = compute_middle_attn_vector(delta, delta_bias, A=A, B=B, C=C, L=L, x_shape=x.shape)
        #     save_dict["xai_vector"] = xai_vector
        # else:
        #     xai_vector = compute_middle_attn_vector(delta, delta_bias, A=A, B=B, C=C, L=L, x_shape=x.shape)
        #     save_dict["xai_vector"] = xai_vector
        #
        # save_dict["z"] = z
        # save_dict["conv1d_out"] = conv1d_out

        y: torch.Tensor = cross_merge_fn(ys, indices, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                         force_torch=scan_force_torch)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=xs, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y, H=H, W=W,
            ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # (B, L, C)
        y = out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value


# support: xv1a,xv2a,xv3a;
# postfix: _cpos;_ocov;_ocov2;_ca,_ca1;_act;_mul;_onsigmoid,_onsoftmax,_ondwconv3,_onnone;
class SS2Dv3:
    def __initxv__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_inner = d_inner
        k_group = 4
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardxv

        # tags for forward_type ==============================
        checkpostfix = SS2Dv2.checkpostfix
        self.out_norm, forward_type = SS2Dv2.get_outnorm(forward_type, d_inner, channel_first)
        self.omul, forward_type = checkpostfix("_mul", forward_type)
        self.oact, forward_type = checkpostfix("_act", forward_type)
        self.f_omul = nn.Identity() if self.omul else None
        self.out_act = nn.GELU() if self.oact else nn.Identity()

        mode = forward_type[:4]
        assert mode in ["xv1a", "xv2a", "xv3a"]

        self.forward = partial(self.forwardxv, mode=mode)
        self.dts_dim = dict(xv1a=self.dt_rank, xv2a=self.d_inner, xv3a=4 * self.dt_rank)[mode]
        d_inner_all = d_inner + self.dts_dim + 8 * d_state
        self.in_proj = Linear(d_model, d_inner_all, bias=bias)

        # conv =======================================
        self.cpos = False
        self.iconv = False
        self.oconv = False
        self.oconv2 = False
        if self.with_dconv:
            cact, forward_type = checkpostfix("_ca", forward_type)
            cact1, forward_type = checkpostfix("_ca1", forward_type)
            self.cact = nn.SiLU() if cact else nn.Identity()
            self.cact = nn.GELU() if cact1 else self.cact

            self.oconv2, forward_type = checkpostfix("_ocov2", forward_type)
            self.oconv, forward_type = checkpostfix("_ocov", forward_type)
            self.cpos, forward_type = checkpostfix("_cpos", forward_type)
            self.iconv = (not self.oconv) and (not self.oconv2)

            if self.iconv:
                self.conv2d = nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    groups=d_model,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                )
            if self.oconv:
                self.oconv2d = nn.Conv2d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    groups=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                )
            if self.oconv2:
                self.conv2d = nn.Conv2d(
                    in_channels=d_inner_all,
                    out_channels=d_inner_all,
                    groups=d_inner_all,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                )

        # out proj =======================================
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

        if forward_type.startswith("xv2"):
            del self.dt_projs_weight
            self.dt_projs_weight = None

    def forwardxv(self, x: torch.Tensor, **kwargs):
        B, (H, W) = x.shape[0], (x.shape[2:4] if self.channel_first else x.shape[1:3])
        L = H * W
        force_fp32 = False
        delta_softplus = True
        out_norm = self.out_norm
        to_dtype = True

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex=True, backend=None)

        if self.iconv:
            x = self.cact(self.conv2d(x))  # (b, d, h, w)
        elif self.cpos:
            x = x + self.conv2d(x)  # (b, d, h, w)

        x = self.in_proj(x)

        if self.oconv2:
            x = self.conv2d(x)  # (b, d, h, w)

        us, dts, Bs, Cs = x.split([self.d_inner, self.dts_dim, 4 * self.d_state, 4 * self.d_state],
                                  dim=(1 if self.channel_first else -1))

        _us = us
        # Bs, Cs = Bs.view(B, H, W, 4, -1), Cs.view(B, H, W, 4, -1)
        # Bs, Cs = Bs.view(B, 4, -1, H, W), Cs.view(B, 4, -1, H, W)
        us = cross_scan_fn(us.contiguous(), in_channel_first=self.channel_first, out_channel_first=True).view(B, -1, L)
        Bs = cross_scan_fn(Bs.contiguous(), in_channel_first=self.channel_first, out_channel_first=True,
                           one_by_one=True).view(B, 4, -1, L)
        Cs = cross_scan_fn(Cs.contiguous(), in_channel_first=self.channel_first, out_channel_first=True,
                           one_by_one=True).view(B, 4, -1, L)
        dts = cross_scan_fn(dts.contiguous(), in_channel_first=self.channel_first, out_channel_first=True,
                            one_by_one=(self.dts_dim == 4 * self.dt_rank)).view(B, L, -1)
        if self.dts_dim == self.dt_rank:
            dts = F.conv1d(dts, self.dt_projs_weight.view(4 * self.d_inner, self.dt_rank, 1), None, groups=4)
        elif self.dts_dim == 4 * self.dt_rank:
            dts = F.conv1d(dts, self.dt_projs_weight.view(4 * self.d_inner, self.dt_rank, 1), None, groups=4)

        As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
        Ds = self.Ds.to(torch.float)  # (K * c)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)  # (K * c)

        if force_fp32:
            us, dts, Bs, Cs = to_fp32(us, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            us, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, 4, -1, H, W)
        y: torch.Tensor = cross_merge_fn(ys.contiguous(), in_channel_first=self.channel_first, out_channel_first=True)
        y = y.view(B, -1, H, W) if self.channel_first else y.view(B, H, W, -1)
        y = out_norm(y)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=us, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y,
            ))

        y = (y.to(x.dtype) if to_dtype else y)

        y = self.out_act(y)

        if self.omul:
            y = y * _us

        if self.oconv:
            y = y + self.cact(self.oconv2d(_us))

        out = self.dropout(self.out_proj(y))
        return out


# mamba2 support ================================
class SS2Dm0:
    def __initm0__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,  # now with mamba2, dstate should be bigger...
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.GELU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v2",
            # ======================
            forward_type="m0",
            # ======================
            with_initial_state=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        assert d_inner % dt_rank == 0
        self.with_dconv = d_conv > 1
        Linear = nn.Linear
        self.forward = self.forwardm0

        # tags for forward_type ==============================
        checkpostfix = SS2Dv2.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = SS2Dv2.get_outnorm(forward_type, d_inner, False)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            m0=partial(self.forward_corem0, force_fp32=False, dstate=d_state),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Sequential(
                Permute(0, 3, 1, 2),  # relate with channel_first and conv
                nn.Conv2d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    groups=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                ),
                Permute(0, 2, 3, 1),
            )

            # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, dt_rank)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

        # init state ============================
        self.initial_state = None
        if with_initial_state:
            self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)),
                                              requires_grad=False)

    def forward_corem0(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            chunk_size=64,
            dstate=64,
            # ==============================
            selective_scan_backend=None,
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        assert scan_mode in ["unidi", "bidi", "cross2d"]
        assert selective_scan_backend in [None, "triton", "torch"]
        x_proj_bias = getattr(self, "x_proj_bias", None)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        N = dstate
        B, H, W, RD = x.shape
        K, R = self.A_logs.shape
        K, R, D = self.Ds.shape
        assert RD == R * D
        L = H * W
        KR = K * R
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        initial_state = None
        if self.initial_state is not None:
            assert self.initial_state.shape[-1] == dstate
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
        xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode,
                           force_torch=scan_force_torch)  # (B, H, W, 4, D)
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
        xs = xs.contiguous().view(B, L, KR, D)
        dts = dts.contiguous().view(B, L, KR)
        Bs = Bs.contiguous().view(B, L, K, N)
        Cs = Cs.contiguous().view(B, L, K, N)
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        As = -self.A_logs.to(torch.float).exp().view(KR)
        Ds = self.Ds.to(torch.float).view(KR, D)
        dt_bias = self.dt_projs_bias.view(KR)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys, final_state = selective_scan_chunk_fn(
            xs, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=dt_bias,
            initial_states=initial_state, dt_softplus=True, return_final_states=True,
            backend=selective_scan_backend,
        )
        y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False,
                                         scans=_scan_mode, force_torch=scan_force_torch)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=self.Ds,
                us=xs, dts=dts, delta_bias=self.dt_projs_bias,
                initial_state=self.initial_state, final_satte=final_state,
                ys=ys, y=y, H=H, W=W,
            ))
        if self.initial_state is not None:
            self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

        y = self.out_norm(y.view(B, H, W, -1))

        return y.to(x.dtype)

    def forwardm0(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class SS2D(nn.Module, SS2Dv0, SS2Dv2, SS2Dv3, SS2Dm0):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=False,
            # ======================
            **kwargs,
    ):
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        elif forward_type.startswith("m"):
            self.__initm0__(**kwargs)
        else:
            self.__initv2__(**kwargs)


# =====================================================
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # =============================
            # scan configuration ===
            scan_type="cross",
            scan_k=4,
            scan_learnable=False,
            scan_diversity_weight=0.0,
            scan_coverage_weight=0.0,
            use_independent_learnable=False,
            scan_force_torch=False,  # force PyTorch implementation (disable Triton)
            # =============================
            # 共享网络参数
            stage_index=0,
            shared_learnable_nets=None,
            shared_learnable_params=None,
            # =============================
            _SS2D: type = SS2D,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
                # ==========================
                # scan configuration
                scan_type=scan_type,
                scan_k=scan_k,
                scan_learnable=scan_learnable,
                scan_diversity_weight=scan_diversity_weight,
                scan_coverage_weight=scan_coverage_weight,
                use_independent_learnable=use_independent_learnable,
                scan_force_torch=scan_force_torch,
                # 传递共享网络
                stage_index=stage_index,
                shared_learnable_nets=shared_learnable_nets,
                shared_learnable_params=shared_learnable_params,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version: str = "v2",  # "v1", "v2", "v3"
            patchembed_version: str = "v1",  # "v1", "v2"
            use_checkpoint=False,
            # =========================
            posembed=False,
            imgsize=224,
            _SS2D=SS2D,
            # =========================
            # scan configuration
            scan_type="cross",
            scan_k=4,
            scan_learnable=False,
            scan_diversity_weight=0.0,
            scan_coverage_weight=0.0,
            use_independent_learnable=False,  # True: 每个SS2D独立参数, False: 所有SS2D共享参数
            scan_share_level="stage",  # "block": 每个block独立, "stage": 每个stage共享, "global": 全局共享
            scan_force_torch=False,  # force PyTorch implementation (disable Triton)
            # =========================
            **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.scan_type = scan_type
        self.scan_k = scan_k
        self.use_independent_learnable = use_independent_learnable
        self.scan_share_level = scan_share_level  # 新增：共享级别
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer,
                                             channel_first=self.channel_first)

        # ========== 创建共享的可学习扫描网络 ==========
        # 支持三种共享级别：
        # - "block": 每个SS2D block独立参数（15个网络）
        # - "stage": 每个stage共享参数（4个网络）
        # - "global": 全局共享参数（1个网络 + 维度适配器）
        self.shared_learnable_nets = None
        self.shared_learnable_params = None  # 用于存储Parameter（nn.ParameterDict）
        if not self.use_independent_learnable and scan_type in ["learnable", "learnable_v2", "content_adaptive"]:
            self.shared_learnable_nets = nn.ModuleDict()
            self.shared_learnable_params = nn.ParameterDict()  # 存储Parameter

            if self.scan_share_level == "global":
                # ========== 全局共享模式：共享Depthwise + 独立Pointwise ==========
                # Depthwise参数少，共享它；Pointwise参数多，每个stage独立
                # 这样既减少参数，又保持灵活性

                # 计算最大维度
                max_dim = int(max(dims) * ssm_ratio)

                if scan_type == "learnable":
                    # 方案1：共享Depthwise + 独立Pointwise
                    # 共享的Depthwise和BN（使用最大维度）
                    self.shared_learnable_nets["global_depthwise"] = nn.Conv2d(
                        max_dim, max_dim, 3, 1, 1, groups=max_dim, bias=False
                    )
                    self.shared_learnable_nets["global_bn"] = nn.BatchNorm2d(max_dim)
                    self.shared_learnable_nets["global_act"] = nn.GELU()

                    # 为每个stage创建独立的Pointwise
                    for i_layer in range(self.num_layers):
                        stage_dim = int(dims[i_layer] * ssm_ratio)
                        stage_key = f"stage_{i_layer}"
                        self.shared_learnable_nets[f"{stage_key}_pointwise"] = nn.Conv2d(
                            stage_dim, scan_k, 1, bias=True
                        )

                elif scan_type == "learnable_v2":
                    # 方案1：共享前面的卷积层 + 独立的最后一层
                    # 为每个方向创建共享的特征提取器
                    for direction_idx in range(scan_k):
                        # 共享的特征提取（使用最大维度）
                        self.shared_learnable_nets[f"global_conv1_dir{direction_idx}"] = nn.Conv2d(
                            max_dim, max_dim // 4, 3, 1, 1, bias=False
                        )
                        self.shared_learnable_nets[f"global_bn1_dir{direction_idx}"] = nn.BatchNorm2d(max_dim // 4)
                        self.shared_learnable_nets[f"global_act_dir{direction_idx}"] = nn.GELU()

                        # 为每个stage创建独立的输出层
                        for i_layer in range(self.num_layers):
                            stage_dim = int(dims[i_layer] * ssm_ratio)
                            stage_key = f"stage_{i_layer}"
                            self.shared_learnable_nets[f"{stage_key}_conv2_dir{direction_idx}"] = nn.Conv2d(
                                stage_dim // 4, 1, 1, bias=True
                            )

                    # 全局位置偏置
                    self.shared_learnable_params["global_pos_bias"] = torch.randn(scan_k, 1, 1)

                elif scan_type == "content_adaptive":
                    # 方案1：共享内容提取 + 独立的扫描头
                    # 共享的内容提取网络（使用最大维度）
                    self.shared_learnable_nets["global_content_dw"] = nn.Conv2d(
                        max_dim, max_dim, 3, 1, 1, groups=max_dim, bias=False
                    )
                    self.shared_learnable_nets["global_content_bn"] = nn.BatchNorm2d(max_dim)
                    self.shared_learnable_nets["global_content_act"] = nn.GELU()
                    self.shared_learnable_nets["global_content_pw"] = nn.Conv2d(max_dim, max_dim, 1, bias=False)

                    # 为每个stage和每个方向创建独立的扫描头
                    for i_layer in range(self.num_layers):
                        stage_dim = int(dims[i_layer] * ssm_ratio)
                        stage_key = f"stage_{i_layer}"
                        for direction_idx in range(scan_k):
                            self.shared_learnable_nets[f"{stage_key}_head_dir{direction_idx}"] = nn.Conv2d(
                                stage_dim, 1, 1, bias=True
                            )

                    # 全局位置偏置
                    self.shared_learnable_params["global_pos_bias"] = torch.randn(scan_k, 1, 1)

            else:  # self.scan_share_level == "stage"
                # ========== Stage级别共享模式：4个网络（每个stage一个）==========
                for i_layer in range(self.num_layers):
                    stage_dim = int(dims[i_layer] * ssm_ratio)  # d_inner for this stage
                    stage_key = f"stage_{i_layer}"

                    if scan_type == "learnable":
                        self.shared_learnable_nets[stage_key] = nn.Sequential(
                            nn.Conv2d(stage_dim, stage_dim, 3, 1, 1, groups=stage_dim, bias=False),
                            nn.BatchNorm2d(stage_dim),
                            nn.GELU(),
                            nn.Conv2d(stage_dim, scan_k, 1, bias=True)
                        )
                    elif scan_type == "learnable_v2":
                        # 为每个方向创建独立的网络
                        offset_nets = nn.ModuleList()
                        for _ in range(scan_k):
                            net = nn.Sequential(
                                nn.Conv2d(stage_dim, stage_dim // 4, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(stage_dim // 4),
                                nn.GELU(),
                                nn.Conv2d(stage_dim // 4, 1, 1, bias=True)
                            )
                            offset_nets.append(net)
                        self.shared_learnable_nets[stage_key] = offset_nets
                        # 位置偏置 - 存储到ParameterDict（直接赋值tensor，不用nn.Parameter包装）
                        self.shared_learnable_params[f"{stage_key}_pos_bias"] = torch.randn(scan_k, 1, 1)
                    elif scan_type == "content_adaptive":
                        # 内容自适应网络
                        content_net = nn.Sequential(
                            nn.Conv2d(stage_dim, stage_dim, 3, 1, 1, groups=stage_dim, bias=False),
                            nn.BatchNorm2d(stage_dim),
                            nn.GELU(),
                            nn.Conv2d(stage_dim, stage_dim, 1, bias=False)
                        )
                        scan_heads = nn.ModuleList([
                            nn.Sequential(
                                nn.Conv2d(stage_dim, 1, 1, bias=True)
                            ) for _ in range(scan_k)
                        ])
                        # 位置偏置 - 存储到ParameterDict（直接赋值tensor，不用nn.Parameter包装）
                        self.shared_learnable_params[f"{stage_key}_pos_bias"] = torch.randn(scan_k, 1, 1)
                        self.shared_learnable_nets[f"{stage_key}_content"] = content_net
                        self.shared_learnable_nets[f"{stage_key}_heads"] = scan_heads

        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                # =================
                # scan configuration
                scan_type=scan_type,
                scan_k=scan_k,
                scan_learnable=scan_learnable,
                scan_diversity_weight=scan_diversity_weight,
                scan_coverage_weight=scan_coverage_weight,
                use_independent_learnable=use_independent_learnable,
                scan_force_torch=scan_force_torch,
                # =================
                # 传递共享网络（stage级别）
                stage_index=i_layer,
                shared_learnable_nets=self.shared_learnable_nets,
                shared_learnable_params=self.shared_learnable_params,
                # =================
                _SS2D=_SS2D,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features),  # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                          channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                             channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            # ===========================
            # scan configuration
            scan_type="cross",
            scan_k=4,
            scan_learnable=False,
            scan_diversity_weight=0.0,
            scan_coverage_weight=0.0,
            use_independent_learnable=False,
            scan_force_torch=False,  # force PyTorch implementation (disable Triton)
            # ===========================
            # 共享网络参数
            stage_index=0,
            shared_learnable_nets=None,
            shared_learnable_params=None,
            # ===========================
            _SS2D=SS2D,
            **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                # scan configuration
                scan_type=scan_type,
                scan_k=scan_k,
                scan_learnable=scan_learnable,
                scan_diversity_weight=scan_diversity_weight,
                scan_coverage_weight=scan_coverage_weight,
                use_independent_learnable=use_independent_learnable,
                scan_force_torch=scan_force_torch,
                # 传递共享网络
                stage_index=stage_index,
                shared_learnable_nets=shared_learnable_nets,
                shared_learnable_params=shared_learnable_params,
                _SS2D=_SS2D,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        # print(x.shape)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed

        # print(x.shape)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4],
                                                             align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

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


# =====================================================
def vanilla_vmamba_tiny():
    return VSSM(
        depths=[2, 2, 9, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v0",
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v1", patchembed_version="v1",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vanilla_vmamba_small():
    return VSSM(
        depths=[2, 2, 27, 2], dims=96, drop_path_rate=0.3,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v0",
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v1", patchembed_version="v1",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vanilla_vmamba_base():
    return VSSM(
        depths=[2, 2, 27, 2], dims=128, drop_path_rate=0.6,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v0",
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v1", patchembed_version="v1",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


# =====================================================
def vmamba_tiny_s2l5(channel_first=True):
    return VSSM(
        depths=[2, 2, 5, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vmamba_small_s2l15(channel_first=True):
    return VSSM(
        depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vmamba_base_s2l15(channel_first=True):
    return VSSM(
        depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


# =====================================================
def vmamba_tiny_s1l8(channel_first=True):
    return VSSM(
        depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vmamba_small_s1l20(channel_first=True):
    return VSSM(
        depths=[2, 2, 20, 2], dims=96, drop_path_rate=0.3,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vmamba_base_s1l20(channel_first=True):
    return VSSM(
        depths=[2, 2, 20, 2], dims=128, drop_path_rate=0.5,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


# mamba2 support =====================================================
# FLOPS count do not work now for mamba2!
def vmamba_tiny_m2():
    return VSSM(
        depths=[2, 2, 4, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v2", forward_type="m0_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vmamba_small_m2():
    return VSSM(
        depths=[2, 2, 12, 2], dims=96, drop_path_rate=0.3,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v2", forward_type="m0_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vmamba_base_m2():
    return VSSM(
        depths=[2, 2, 12, 2], dims=128, drop_path_rate=0.3,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v2", forward_type="m0_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


if __name__ == "__main__":
    model_ref = vmamba_tiny_s1l8()

    model = VSSM(
        depths=[2, 2, 4, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v2", forward_type="m0_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )
    print(parameter_count(model)[""])
    print(model.flops())  # wrong
    model.cuda().train()
    model_ref.cuda().train()


    def bench(model):
        import time
        inp = torch.randn((128, 3, 224, 224)).cuda()
        for _ in range(30):
            model(inp)
        torch.cuda.synchronize()
        tim = time.time()
        for _ in range(30):
            model(inp)
        torch.cuda.synchronize()
        tim1 = time.time() - tim

        for _ in range(30):
            model(inp).sum().backward()
        torch.cuda.synchronize()
        tim = time.time()
        for _ in range(30):
            model(inp).sum().backward()
        torch.cuda.synchronize()
        tim2 = time.time() - tim

        return tim1 / 30, tim2 / 30


    print(bench(model_ref))
    print(bench(model))

    breakpoint()


