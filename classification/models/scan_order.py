import torch
import numpy as np


class ScanOrderGenerator:
    """
    统一的扫描顺序生成器 - 懒加载缓存

    支持的扫描模式:
        - cross: 8方向交叉扫描
        - permute: 蛇形扫描 (PlainMamba风格)
        - diagonal: 对角线扫描
        - zigzag: 锯齿形扫描
        - spiral: 螺旋扫描
        - efficent: 先偶后奇扫描
        - hilbert: 希尔伯特曲线扫描 (适用于任意尺寸)
        - blockwise: 分块扫描

    Usage:
        scan_gen = ScanOrderGenerator(mode='permute', k=4)
        indices = scan_gen(H, W, device='cuda')  # (k, H*W)
    """

    SUPPORTED_MODES = ['cross', 'permute', 'diagonal', 'zigzag', 'spiral', 'efficent', 'hilbert', 'blockwise', 'random']

    def __init__(self, mode='cross', k=4):
        """
        Args:
            mode: 扫描模式
            k: 扫描方向数 (1, 2, 4, 8)
        """
        assert mode in self.SUPPORTED_MODES, f"mode must be one of {self.SUPPORTED_MODES}"
        self.mode = mode
        self.k = k
        self._cache = {}

    def __call__(self, H, W, device='cuda'):
        """
        返回扫描顺序索引

        Args:
            H: 特征图高度
            W: 特征图宽度
            device: 目标设备

        Returns:
            torch.LongTensor: (k, H*W) 的扫描顺序索引
        """
        key = (H, W)
        if key not in self._cache:
            orders = self._generate(H, W)
            # 使用 np.stack 确保正确的数组格式
            self._cache[key] = torch.from_numpy(np.stack(orders, axis=0)).long()
        return self._cache[key].to(device)

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()

    def _generate(self, H, W):
        """根据 mode 生成扫描顺序"""
        return getattr(self, f'_{self.mode}')(H, W)

    # ==================== Cross Scan ====================
    def _cross(self, H, W):
        """8方向交叉扫描，取前k个 - 向量化优化版本"""
        L = H * W
        # 基础索引矩阵 (H, W)
        base = np.arange(L, dtype=np.int64).reshape(H, W)

        orders = []
        for idx in range(self.k):
            if idx == 0:    # 行主序: 左上→右下
                order = base.ravel()
            elif idx == 1:  # 列主序: 左上→右下
                order = base.T.ravel()
            elif idx == 2:  # 行主序: 右上→左下
                order = base[:, ::-1].ravel()
            elif idx == 3:  # 列主序: 右上→左下
                order = base[:, ::-1].T.ravel()
            elif idx == 4:  # 行主序: 左下→右上
                order = base[::-1, :].ravel()
            elif idx == 5:  # 列主序: 左下→右上
                order = base[::-1, :].T.ravel()
            elif idx == 6:  # 行主序: 右下→左上
                order = base[::-1, ::-1].ravel()
            elif idx == 7:  # 列主序: 右下→左上
                order = base[::-1, ::-1].T.ravel()
            else:
                order = base.ravel()

            # 注意：需要 copy，因为某些操作返回视图
            orders.append(np.ascontiguousarray(order))

        return orders

    # ==================== Permute Scan (PlainMamba) ====================
    def _permute(self, H, W):
        """蛇形扫描，取前k个 - 向量化优化版本"""
        L = H * W
        base = np.arange(L, dtype=np.int64).reshape(H, W)

        def snake_h(reverse_start=False):
            """水平蛇形扫描"""
            result = base.copy()
            for i in range(H):
                if (i % 2 == 0) != reverse_start:
                    pass  # 保持原序
                else:
                    result[i, :] = result[i, ::-1]
            return result.ravel()

        def snake_v(reverse_start=False):
            """垂直蛇形扫描"""
            result = base.T.copy()
            for j in range(W):
                if (j % 2 == 0) != reverse_start:
                    pass
                else:
                    result[j, :] = result[j, ::-1]
            return result.ravel()

        orders = []
        for idx in range(self.k):
            if idx == 0:    # 水平蛇形，偶数行左→右
                order = snake_h(False)
            elif idx == 1:  # 水平蛇形，偶数行右→左
                order = snake_h(True)
            elif idx == 2:  # 垂直蛇形，偶数列上→下
                order = snake_v(False)
            elif idx == 3:  # 垂直蛇形，偶数列下→上
                order = snake_v(True)
            elif idx >= 4:  # 4-7是0-3的逆序
                order = orders[idx - 4][::-1].copy()
            orders.append(np.ascontiguousarray(order))

        return orders

    # ==================== Diagonal Scan ====================
    def _diagonal(self, H, W):
        """对角线扫描，取前k个 - 向量化优化版本"""
        L = H * W
        base = np.arange(L, dtype=np.int64).reshape(H, W)

        # 创建坐标网格
        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        flat_indices = base.ravel()
        flat_rows = rows.ravel()
        flat_cols = cols.ravel()

        # 主对角线: 按 (row + col) 排序，同一对角线内按 row 排序
        diag_sum = flat_rows + flat_cols
        main_diag_order = np.lexsort((flat_rows, diag_sum))
        main_diag = flat_indices[main_diag_order]

        # 主对角线反向: 同一对角线内按 col 排序
        main_diag_rev_order = np.lexsort((flat_cols, diag_sum))
        main_diag_rev = flat_indices[main_diag_rev_order]

        # 副对角线: 按 (row - col) 排序，同一对角线内按 row 降序
        diag_diff = flat_rows - flat_cols
        anti_diag_order = np.lexsort((-flat_rows, diag_diff))
        anti_diag = flat_indices[anti_diag_order]

        # 副对角线反向: 同一对角线内按 row 升序
        anti_diag_rev_order = np.lexsort((flat_rows, diag_diff))
        anti_diag_rev = flat_indices[anti_diag_rev_order]

        patterns = [main_diag, main_diag_rev, anti_diag, anti_diag_rev]

        orders = []
        for idx in range(self.k):
            if idx < 4:
                order = patterns[idx]
            else:
                order = patterns[idx - 4][::-1].copy()
            orders.append(np.ascontiguousarray(order))

        return orders

    # ==================== Zigzag Scan ====================
    def _zigzag(self, H, W):
        """锯齿形扫描，取前k个 - 向量化优化版本"""
        L = H * W
        base = np.arange(L, dtype=np.int64).reshape(H, W)

        # 创建坐标网格
        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        flat_indices = base.ravel()
        flat_rows = rows.ravel()
        flat_cols = cols.ravel()
        diag_sum = flat_rows + flat_cols

        # zigzag: 偶数对角线从上往下，奇数对角线从下往上
        # rev=False: 偶数对角线按 row 升序
        even_mask = (diag_sum % 2 == 0)
        sort_key = np.where(even_mask, flat_rows, -flat_rows)
        zigzag_order = np.lexsort((sort_key, diag_sum))
        zigzag_0 = flat_indices[zigzag_order]

        # rev=True: 偶数对角线按 row 降序
        sort_key_rev = np.where(even_mask, -flat_rows, flat_rows)
        zigzag_rev_order = np.lexsort((sort_key_rev, diag_sum))
        zigzag_1 = flat_indices[zigzag_rev_order]

        patterns = [zigzag_0, zigzag_1]

        orders = []
        for idx in range(self.k):
            if idx < 2:
                order = patterns[idx]
            elif idx < 4:
                order = patterns[idx - 2][::-1].copy()
            else:
                order = orders[idx % 4]
            orders.append(np.ascontiguousarray(order))

        return orders

    # ==================== Spiral Scan ====================
    def _spiral(self, H, W):
        """螺旋扫描，取前k个 - 优化版本"""
        L = H * W
        base = np.arange(L, dtype=np.int64).reshape(H, W)

        def spiral(start_dir):
            """生成螺旋顺序，使用 numpy 切片加速"""
            result = []
            top, bottom, left, right = 0, H - 1, 0, W - 1
            d = start_dir
            while top <= bottom and left <= right:
                if d % 4 == 0 and top <= bottom:
                    result.append(base[top, left:right+1])
                    top += 1
                elif d % 4 == 1 and left <= right:
                    result.append(base[top:bottom+1, right])
                    right -= 1
                elif d % 4 == 2 and top <= bottom:
                    result.append(base[bottom, left:right+1][::-1])
                    bottom -= 1
                elif d % 4 == 3 and left <= right:
                    result.append(base[top:bottom+1, left][::-1])
                    left += 1
                d += 1
            return np.concatenate(result) if result else np.array([], dtype=np.int64)

        # 预计算 4 种起始方向
        patterns = [spiral(i) for i in range(4)]

        orders = []
        for idx in range(self.k):
            if idx < 4:
                order = patterns[idx]
            else:
                order = patterns[idx - 4][::-1].copy()
            orders.append(np.ascontiguousarray(order))

        return orders

    # ==================== Efficent Scan ====================
    def _efficent(self, H, W):
        """高效扫描（先偶数再奇数），取前k个 - 向量化优化版本"""
        L = H * W
        base = np.arange(L, dtype=np.int64).reshape(H, W)

        # 预计算偶奇列/行索引
        even_cols = np.arange(0, W, 2)
        odd_cols = np.arange(1, W, 2)
        even_rows = np.arange(0, H, 2)
        odd_rows = np.arange(1, H, 2)

        orders = []
        for idx in range(self.k):
            if idx == 0:    # 行优先，先偶数列再奇数列
                order = np.concatenate([base[:, even_cols].ravel(), base[:, odd_cols].ravel()])
            elif idx == 1:  # 列优先，先偶数行再奇数行
                order = np.concatenate([base[even_rows, :].T.ravel(), base[odd_rows, :].T.ravel()])
            elif idx == 2:  # 行优先，反向（先奇数列再偶数列，从右到左）
                order = np.concatenate([base[:, odd_cols[::-1]].ravel(), base[:, even_cols[::-1]].ravel()])
            elif idx == 3:  # 列优先，反向
                order = np.concatenate([base[even_rows, ::-1].T.ravel(), base[odd_rows, ::-1].T.ravel()])
            elif idx >= 4:
                order = orders[idx - 4][::-1].copy()
            orders.append(np.ascontiguousarray(order))

        return orders

    # ==================== Hilbert Curve Scan ====================
    def _hilbert(self, H, W):
        """
        Hilbert 曲线扫描 - 向量化优化版本
        思路: padding 到 2 的幂次，生成完整曲线，再过滤掉超出范围的点
        """
        def d2xy_vectorized(n, d_arr):
            """向量化版本：将一维 Hilbert 索引数组转换为二维坐标数组"""
            d = d_arr.copy()
            x = np.zeros_like(d)
            y = np.zeros_like(d)
            s = 1
            while s < n:
                rx = (d // 2) & 1
                ry = (d ^ rx) & 1
                # 当 ry == 0 时的变换
                mask_ry0 = (ry == 0)
                mask_rx1 = (rx == 1) & mask_ry0
                # if rx == 1: x, y = s-1-x, s-1-y
                x = np.where(mask_rx1, s - 1 - x, x)
                y = np.where(mask_rx1, s - 1 - y, y)
                # swap x, y when ry == 0
                x_new = np.where(mask_ry0, y, x)
                y_new = np.where(mask_ry0, x, y)
                x, y = x_new, y_new
                # 更新坐标
                x = x + s * rx
                y = y + s * ry
                d = d // 4
                s *= 2
            return x, y

        def hilbert_padded(H, W, rotate=0):
            # 找到 >= max(H, W) 的最小2的幂
            padded_size = 1
            while padded_size < max(H, W):
                padded_size *= 2

            # 向量化生成完整的 hilbert 曲线
            total = padded_size * padded_size
            d_arr = np.arange(total, dtype=np.int64)
            x, y = d2xy_vectorized(padded_size, d_arr)

            # 旋转坐标
            for _ in range(rotate):
                x, y = y, padded_size - 1 - x

            row, col = y, x  # (row, col)

            # 过滤：只保留有效范围内的点
            valid_mask = (row < H) & (col < W)
            valid_indices = row[valid_mask] * W + col[valid_mask]

            return valid_indices

        # 预计算 4 种旋转方向
        patterns = [hilbert_padded(H, W, rotate=i) for i in range(4)]

        orders = []
        for idx in range(self.k):
            if idx < 4:
                order = patterns[idx]
            else:
                order = patterns[idx - 4][::-1].copy()
            orders.append(np.ascontiguousarray(order))

        return orders

    # ==================== Blockwise Scan ====================
    def _blockwise(self, H, W):
        """分块扫描，取前k个 - 向量化优化版本"""
        block_h = max(H // 8, 1)
        block_w = max(W // 8, 1)
        L = H * W
        base = np.arange(L, dtype=np.int64).reshape(H, W)

        def blockwise_permute(perm_idx):
            result = []
            for bi in range(0, H, block_h):
                for bj in range(0, W, block_w):
                    h_end = min(bi + block_h, H)
                    w_end = min(bj + block_w, W)
                    block = base[bi:h_end, bj:w_end].copy()

                    # 块内蛇形扫描
                    h_block = h_end - bi
                    for i in range(h_block):
                        if (i + perm_idx) % 2 == 1:
                            block[i, :] = block[i, ::-1]

                    result.append(block.ravel())
            return np.concatenate(result)

        orders = []
        for idx in range(self.k):
            if idx < 4:
                order = blockwise_permute(idx)
            else:
                order = orders[idx - 4][::-1].copy()
            orders.append(np.ascontiguousarray(order))

        return orders

    # ==================== Random Scan ====================
    def _random(self, H, W):
        """
        随机扫描，生成 k 个不同的随机序列 - 使用 numpy 加速

        注意：
        - 默认会缓存，同一个 (H, W) 返回相同的随机序列
        - 如需每个 epoch 使用不同随机序列，调用 clear_cache() 或 reset_random()
        """
        L = H * W

        orders = []
        for i in range(self.k):
            # 使用 numpy 随机生成器，更快且可复现
            rng = np.random.default_rng(seed=hash((H, W, i)) % (2**32))
            order = rng.permutation(L).astype(np.int64)
            orders.append(order)

        return orders

    def reset_random(self, seed=None):
        """
        重置随机扫描的缓存，用于在新 epoch 时生成新的随机序列

        Args:
            seed: 可选的随机种子，用于可复现性
        """
        # 只清除 random 模式的缓存
        if self.mode == 'random':
            self._cache.clear()
            if seed is not None:
                self._random_seed = seed


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 ScanOrderGenerator")
    print("=" * 60)

    # 测试各种尺寸
    test_sizes = [(8, 8), (14, 14), (7, 7), (56, 56), (28, 28), (13, 17)]

    for mode in ScanOrderGenerator.SUPPORTED_MODES:
        print(f"\n{mode.upper()}")
        print("-" * 40)

        for H, W in test_sizes:
            L = H * W
            try:
                gen = ScanOrderGenerator(mode=mode, k=4)
                indices = gen(H, W, device='cpu')

                # 验证
                valid = True
                for i in range(4):
                    unique_count = len(set(indices[i].tolist()))
                    covers_all = set(indices[i].tolist()) == set(range(L))
                    if unique_count != L or not covers_all:
                        valid = False
                        break

                status = "PASS" if valid else "FAIL"
                print(f"  {H:2d}x{W:2d}: shape={tuple(indices.shape)}, {status}")

            except Exception as e:
                print(f"  {H:2d}x{W:2d}: ERROR - {e}")

    # 测试缓存
    print("\n" + "=" * 60)
    print("测试缓存机制")
    print("=" * 60)
    gen = ScanOrderGenerator(mode='cross', k=4)
    _ = gen(56, 56)
    _ = gen(28, 28)
    _ = gen(14, 14)
    _ = gen(7, 7)
    print(f"缓存的尺寸: {list(gen._cache.keys())}")

    # 测试重复调用（应该命中缓存）
    import time
    gen = ScanOrderGenerator(mode='permute', k=4)

    # 首次调用
    start = time.time()
    for _ in range(100):
        gen.clear_cache()
        _ = gen(56, 56, device='cpu')
    first_time = time.time() - start

    # 缓存调用
    _ = gen(56, 56, device='cpu')  # 预热缓存
    start = time.time()
    for _ in range(100):
        _ = gen(56, 56, device='cpu')
    cache_time = time.time() - start

    print(f"\n100次调用 (56x56, permute, k=4):")
    print(f"  无缓存: {first_time*1000:.2f} ms")
    print(f"  有缓存: {cache_time*1000:.2f} ms")
    if cache_time > 0:
        print(f"  加速比: {first_time/cache_time:.1f}x")
    else:
        print(f"  加速比: 缓存命中几乎无开销!")
