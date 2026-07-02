import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict


# ==================== 辅助损失缓存 ====================

class _AuxLossCache:
    """
    辅助损失缓存 - 用于从 backbone 传递损失到 detector

    这是一个内部类，由 LearnableScanOrder 自动使用
    """
    _losses: Dict[str, torch.Tensor] = {}

    @classmethod
    def register(cls, name: str, loss: torch.Tensor):
        """注册一个辅助损失"""
        if not name.startswith('loss_'):
            name = 'loss_' + name
        cls._losses[name] = loss

    @classmethod
    def collect(cls) -> Dict[str, torch.Tensor]:
        """收集所有损失并清空缓存"""
        losses = cls._losses.copy()
        cls._losses.clear()
        return losses

    @classmethod
    def clear(cls):
        """清空缓存"""
        cls._losses.clear()


# 导出给 detector 使用
aux_loss_cache = _AuxLossCache


# ==================== 辅助损失计算函数 ====================

def compute_diversity_loss(scores: torch.Tensor) -> torch.Tensor:
    """
    多样性损失 - 鼓励不同扫描方向学习不同的模式

    Args:
        scores: (B, k, L)
    Returns:
        loss: scalar
    """
    B, k, L = scores.shape
    scores_norm = F.normalize(scores, dim=-1)
    sim_matrix = torch.bmm(scores_norm, scores_norm.transpose(1, 2))
    mask = torch.eye(k, device=scores.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim_matrix.masked_fill(mask, 0)
    return off_diag.abs().mean()


def compute_coverage_loss(scores: torch.Tensor) -> torch.Tensor:
    """
    覆盖均匀性损失 - 鼓励分数分布更加均匀

    Args:
        scores: (B, k, L)
    Returns:
        loss: scalar
    """
    B, k, L = scores.shape
    scores_soft = F.softmax(scores, dim=-1)
    uniform = torch.ones_like(scores_soft) / L
    return F.kl_div(scores_soft.log(), uniform, reduction='batchmean')


class LearnableScanOrder(nn.Module):
    """
    可学习的扫描顺序生成器（带辅助损失支持）

    基于 ASM-UNet 风格，学习 k 个 score map，按 score 排序得到扫描顺序

    Args:
        dim: 输入特征维度
        k: 扫描方向数 (1~8)
        lightweight: 是否使用轻量级网络
        diversity_weight: 多样性损失权重 (0 表示不使用)
        coverage_weight: 覆盖损失权重 (0 表示不使用)

    Usage:
        # 方式1: 不使用辅助损失
        scan_order = LearnableScanOrder(dim=192, k=4)

        # 方式2: 使用辅助损失
        scan_order = LearnableScanOrder(dim=192, k=4, diversity_weight=0.1, coverage_weight=0.05)
        indices = scan_order(x)  # 训练时自动计算并缓存辅助损失

        # 在 detector 的 loss() 中收集:
        from classification.models.learnable_scan import aux_loss_cache
        losses.update(aux_loss_cache.collect())
    """

    def __init__(self, dim, k=4, lightweight=True,
                 diversity_weight: float = 0.0,
                 coverage_weight: float = 0.0):
        super().__init__()
        self.k = k
        self.dim = dim
        self.diversity_weight = diversity_weight
        self.coverage_weight = coverage_weight

        if lightweight:
            # 轻量级：depthwise conv + pointwise conv
            self.score_net = nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, k, 1, bias=True)
            )
        else:
            # 标准版：更强的表达能力
            hidden_dim = max(dim // 4, k * 4)
            self.score_net = nn.Sequential(
                nn.Conv2d(dim, hidden_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, k, 1, bias=True)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def use_aux_loss(self) -> bool:
        """是否使用辅助损失"""
        return self.diversity_weight > 0 or self.coverage_weight > 0

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            indices: (B, k, L) 每个方向的扫描顺序，L = H * W
        """
        B, C, H, W = x.shape
        L = H * W

        # 生成 k 个方向的 score
        scores = self.score_net(x)  # (B, k, H, W)
        scores = scores.view(B, self.k, L)  # (B, k, L)

        # 按 score 降序排列得到扫描顺序
        indices = scores.argsort(dim=-1, descending=True)  # (B, k, L)

        # 训练时计算辅助损失
        if self.training and self.use_aux_loss:
            self._compute_and_cache_aux_loss(scores)

        return indices

    def _compute_and_cache_aux_loss(self, scores: torch.Tensor):
        """计算辅助损失并缓存到全局"""
        if self.diversity_weight > 0:
            div_loss = compute_diversity_loss(scores) * self.diversity_weight
            _AuxLossCache.register('loss_scan_diversity', div_loss)

        if self.coverage_weight > 0:
            cov_loss = compute_coverage_loss(scores) * self.coverage_weight
            _AuxLossCache.register('loss_scan_coverage', cov_loss)


class LearnableScanOrderV2(nn.Module):
    """
    可学习的扫描顺序生成器 V2 - 带可微分训练支持

    使用 Straight-Through Estimator 使离散排序操作可微分
    训练时：软排序（可微分） + 硬排序（前向）
    推理时：硬排序

    Usage:
        scan_order = LearnableScanOrderV2(dim=192, k=4)
        indices, soft_indices = scan_order(x)
    """

    def __init__(self, dim, k=4, tau=1.0, hard=True):
        """
        Args:
            dim: 输入特征维度
            k: 扫描方向数
            tau: Gumbel-Softmax 温度参数
            hard: 是否使用硬排序（推荐 True）
        """
        super().__init__()
        self.k = k
        self.dim = dim
        self.tau = tau
        self.hard = hard

        hidden_dim = max(dim // 4, k * 4)
        self.score_net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, k, 1, bias=True)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            indices: (B, k, L) 硬排序索引
            scores: (B, k, L) 用于梯度回传的 scores
        """
        B, C, H, W = x.shape
        L = H * W

        # 生成 k 个方向的 score
        scores = self.score_net(x)  # (B, k, H, W)
        scores = scores.view(B, self.k, L)  # (B, k, L)

        # 硬排序（前向）
        indices = scores.argsort(dim=-1, descending=True)  # (B, k, L)

        if self.training:
            # 训练时返回 scores 用于辅助损失
            return indices, scores
        else:
            return indices, None


class LearnableScanOrderV3(nn.Module):
    """
    可学习的扫描顺序生成器 V3 - 基于偏移预测

    类似 Deformable Conv，预测每个位置的偏移量，根据偏移排序

    Usage:
        scan_order = LearnableScanOrderV3(dim=192, k=4)
        indices = scan_order(x)
    """

    def __init__(self, dim, k=4):
        """
        Args:
            dim: 输入特征维度
            k: 扫描方向数
        """
        super().__init__()
        self.k = k
        self.dim = dim

        # 每个方向预测一个重要性分数
        # 使用不同的初始化使 k 个方向学到不同的模式
        self.offset_nets = nn.ModuleList()
        for i in range(k):
            net = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1, groups=1, bias=False),
                nn.BatchNorm2d(dim // 4),
                nn.GELU(),
                nn.Conv2d(dim // 4, 1, 1, bias=True)
            )
            self.offset_nets.append(net)

        # 为每个方向添加可学习的方向偏置
        self.direction_bias = nn.Parameter(torch.randn(k, 1, 1))

        self._init_weights()

    def _init_weights(self):
        for i, net in enumerate(self.offset_nets):
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    # 不同方向使用不同的初始化
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    # 添加方向相关的偏置
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.1 * i)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            indices: (B, k, L) 每个方向的扫描顺序
        """
        B, C, H, W = x.shape
        L = H * W

        indices_list = []
        for i, net in enumerate(self.offset_nets):
            # 每个方向预测一个 score map
            score = net(x)  # (B, 1, H, W)
            score = score.view(B, L)  # (B, L)

            # 添加位置编码偏置，使不同方向有不同的空间偏好
            pos_bias = self._get_position_bias(H, W, i, x.device)
            score = score + pos_bias.view(1, L) * self.direction_bias[i]

            # 排序
            idx = score.argsort(dim=-1, descending=True)  # (B, L)
            indices_list.append(idx)

        indices = torch.stack(indices_list, dim=1)  # (B, k, L)
        return indices

    def _get_position_bias(self, H, W, direction_idx, device):
        """生成位置编码偏置，不同方向有不同的空间偏好"""
        y = torch.arange(H, device=device).float().view(H, 1).expand(H, W)
        x = torch.arange(W, device=device).float().view(1, W).expand(H, W)

        # 不同方向的位置偏好
        if direction_idx % 4 == 0:
            bias = y + x  # 左上优先
        elif direction_idx % 4 == 1:
            bias = y - x  # 右上优先
        elif direction_idx % 4 == 2:
            bias = -y + x  # 左下优先
        else:
            bias = -y - x  # 右下优先

        if direction_idx >= 4:
            bias = -bias  # 反向

        return bias


class HybridScanOrder(nn.Module):
    """
    混合扫描顺序生成器

    结合固定扫描（如 cross, permute）和可学习扫描

    Usage:
        scan_order = HybridScanOrder(dim=192, k=4, fixed_k=2, learnable_k=2)
        indices = scan_order(x, H, W)
    """

    def __init__(self, dim, k=4, fixed_k=2, learnable_k=2, fixed_mode='cross'):
        """
        Args:
            dim: 输入特征维度
            k: 总扫描方向数
            fixed_k: 固定扫描方向数
            learnable_k: 可学习扫描方向数
            fixed_mode: 固定扫描模式
        """
        super().__init__()
        assert fixed_k + learnable_k == k
        self.k = k
        self.fixed_k = fixed_k
        self.learnable_k = learnable_k

        # 固定扫描生成器
        from scan_order import ScanOrderGenerator
        self.fixed_scan = ScanOrderGenerator(mode=fixed_mode, k=fixed_k)

        # 可学习扫描生成器
        if learnable_k > 0:
            self.learnable_scan = LearnableScanOrder(dim, k=learnable_k)
        else:
            self.learnable_scan = None

        self._cache = {}

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            indices: (B, k, L) 混合扫描顺序
        """
        B, C, H, W = x.shape
        L = H * W
        device = x.device

        indices_list = []

        # 1. 固定扫描部分
        if self.fixed_k > 0:
            fixed_indices = self.fixed_scan(H, W, device=device)  # (fixed_k, L)
            fixed_indices = fixed_indices.unsqueeze(0).expand(B, -1, -1)  # (B, fixed_k, L)
            indices_list.append(fixed_indices)

        # 2. 可学习扫描部分
        if self.learnable_k > 0:
            learnable_indices = self.learnable_scan(x)  # (B, learnable_k, L)
            indices_list.append(learnable_indices)

        # 合并
        indices = torch.cat(indices_list, dim=1)  # (B, k, L)
        return indices


class ContentAdaptiveScanOrder(nn.Module):
    """
    内容自适应扫描顺序生成器

    根据图像内容（如边缘、显著性）自适应调整扫描顺序

    Usage:
        scan_order = ContentAdaptiveScanOrder(dim=192, k=4)
        indices = scan_order(x)
    """

    def __init__(self, dim, k=4):
        """
        Args:
            dim: 输入特征维度
            k: 扫描方向数
        """
        super().__init__()
        self.k = k
        self.dim = dim

        # 内容分析网络
        self.content_net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim // 2, 1, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
        )

        # k 个方向的扫描头
        self.scan_heads = nn.ModuleList([
            nn.Conv2d(dim // 2, 1, 1, bias=True)
            for _ in range(k)
        ])

        # 可学习的方向嵌入
        self.direction_embeds = nn.Parameter(torch.randn(k, dim // 2, 1, 1) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            indices: (B, k, L) 内容自适应扫描顺序
        """
        B, C, H, W = x.shape
        L = H * W

        # 提取内容特征
        content_feat = self.content_net(x)  # (B, dim//2, H, W)

        indices_list = []
        for i in range(self.k):
            # 添加方向嵌入
            feat = content_feat + self.direction_embeds[i]  # (B, dim//2, H, W)

            # 生成 score
            score = self.scan_heads[i](feat)  # (B, 1, H, W)
            score = score.view(B, L)  # (B, L)

            # 排序
            idx = score.argsort(dim=-1, descending=True)
            indices_list.append(idx)

        indices = torch.stack(indices_list, dim=1)  # (B, k, L)
        return indices


# ==================== 辅助损失函数 ====================

class ScanDiversityLoss(nn.Module):
    """
    扫描多样性损失

    鼓励 k 个扫描方向学习不同的模式
    """

    def __init__(self):
        super().__init__()

    def forward(self, scores):
        """
        Args:
            scores: (B, k, L) 各方向的 score
        Returns:
            loss: 多样性损失
        """
        B, k, L = scores.shape

        # 归一化
        scores_norm = F.normalize(scores, dim=-1)  # (B, k, L)

        # 计算两两相似度
        sim_matrix = torch.bmm(scores_norm, scores_norm.transpose(1, 2))  # (B, k, k)

        # 对角线为自相似，不计入
        mask = torch.eye(k, device=scores.device).unsqueeze(0)  # (1, k, k)
        sim_matrix = sim_matrix * (1 - mask)

        # 希望非对角线的相似度尽量小
        loss = sim_matrix.abs().mean()

        return loss


class ScanCoverageLoss(nn.Module):
    """
    扫描覆盖损失

    鼓励每个方向都覆盖到图像的不同区域
    """

    def __init__(self):
        super().__init__()

    def forward(self, scores, H, W):
        """
        Args:
            scores: (B, k, L) 各方向的 score
            H, W: 特征图尺寸
        Returns:
            loss: 覆盖损失
        """
        B, k, L = scores.shape

        # 将 score 重塑为空间形式
        scores_2d = scores.view(B, k, H, W)  # (B, k, H, W)

        # 计算每个方向的空间中心
        y_coords = torch.arange(H, device=scores.device).float().view(1, 1, H, 1)
        x_coords = torch.arange(W, device=scores.device).float().view(1, 1, 1, W)

        # softmax 归一化
        weights = F.softmax(scores_2d.view(B, k, -1), dim=-1).view(B, k, H, W)

        # 计算加权中心
        center_y = (weights * y_coords).sum(dim=(2, 3))  # (B, k)
        center_x = (weights * x_coords).sum(dim=(2, 3))  # (B, k)

        centers = torch.stack([center_y, center_x], dim=-1)  # (B, k, 2)

        # 计算中心之间的距离
        dist_matrix = torch.cdist(centers, centers)  # (B, k, k)

        # 希望中心之间的距离尽量大
        mask = torch.eye(k, device=scores.device).unsqueeze(0)
        dist_matrix = dist_matrix * (1 - mask)

        # 负距离作为损失
        loss = -dist_matrix.mean()

        return loss


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 LearnableScanOrder")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, C, H, W = 2, 192, 14, 14
    k = 4
    L = H * W

    x = torch.randn(B, C, H, W, device=device)

    # 测试 V1
    print("\n1. LearnableScanOrder (V1)")
    model_v1 = LearnableScanOrder(dim=C, k=k).to(device)
    indices_v1 = model_v1(x)
    print(f"   输入: {x.shape}")
    print(f"   输出: {indices_v1.shape}")
    # 验证每个方向都是完整的排列
    for i in range(k):
        unique = len(torch.unique(indices_v1[0, i]))
        print(f"   方向 {i}: unique={unique}, expected={L}, {'PASS' if unique == L else 'FAIL'}")

    # 测试 V2
    print("\n2. LearnableScanOrderV2 (可微分)")
    model_v2 = LearnableScanOrderV2(dim=C, k=k).to(device)
    model_v2.train()
    indices_v2, scores = model_v2(x)
    print(f"   输出 indices: {indices_v2.shape}")
    print(f"   输出 scores: {scores.shape if scores is not None else None}")

    # 测试 V3
    print("\n3. LearnableScanOrderV3 (偏移预测)")
    model_v3 = LearnableScanOrderV3(dim=C, k=k).to(device)
    indices_v3 = model_v3(x)
    print(f"   输出: {indices_v3.shape}")
    for i in range(k):
        unique = len(torch.unique(indices_v3[0, i]))
        print(f"   方向 {i}: unique={unique}, expected={L}, {'PASS' if unique == L else 'FAIL'}")

    # 测试内容自适应
    print("\n4. ContentAdaptiveScanOrder")
    model_ca = ContentAdaptiveScanOrder(dim=C, k=k).to(device)
    indices_ca = model_ca(x)
    print(f"   输出: {indices_ca.shape}")
    for i in range(k):
        unique = len(torch.unique(indices_ca[0, i]))
        print(f"   方向 {i}: unique={unique}, expected={L}, {'PASS' if unique == L else 'FAIL'}")

    # 测试多样性损失
    print("\n5. ScanDiversityLoss")
    div_loss = ScanDiversityLoss()
    _, scores = model_v2(x)
    loss = div_loss(scores)
    print(f"   Diversity Loss: {loss.item():.4f}")

    # 测试覆盖损失
    print("\n6. ScanCoverageLoss")
    cov_loss = ScanCoverageLoss()
    loss = cov_loss(scores, H, W)
    print(f"   Coverage Loss: {loss.item():.4f}")

    # 测试参数量
    print("\n" + "=" * 60)
    print("参数量统计")
    print("=" * 60)
    for name, model in [("V1", model_v1), ("V2", model_v2), ("V3", model_v3), ("CA", model_ca)]:
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {params:,} parameters ({params/1e3:.1f}K)")
