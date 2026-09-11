"""评估路径专用的 float64 双线性重采样（benchmark-protocol.md §6.9）。

模型图 → 评估网格：双线性；概率图重采样后必须重新归一化（调用方负责）。
CB 上采样：双线性 + 重归一化。实现独立于 AOI 路径（metrics.probability），
避免两条路径共享防护口径。
"""

from __future__ import annotations

import numpy as np

from .density import as_float64_2d


def resample_bilinear(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """像素中心对齐的双线性重采样（float64，无精度损失路径）。"""
    a = as_float64_2d(arr, "resample 输入")
    th, tw = int(target_shape[0]), int(target_shape[1])
    if th <= 0 or tw <= 0:
        raise ValueError(f"目标尺寸必须为正：{target_shape}")
    sh, sw = a.shape
    if (sh, sw) == (th, tw):
        return a.copy()
    ys = np.clip((np.arange(th, dtype=np.float64) + 0.5) * (sh / th) - 0.5, 0.0, sh - 1.0)
    xs = np.clip((np.arange(tw, dtype=np.float64) + 0.5) * (sw / tw) - 0.5, 0.0, sw - 1.0)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, sh - 1)
    x1 = np.minimum(x0 + 1, sw - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = a[np.ix_(y0, x0)] * (1.0 - wx) + a[np.ix_(y0, x1)] * wx
    bottom = a[np.ix_(y1, x0)] * (1.0 - wx) + a[np.ix_(y1, x1)] * wx
    return top * (1.0 - wy) + bottom * wy


def resample_to_density(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """双线性重采样 + sum=1 重归一化（§6.9：概率图重采样后必须重新归一化）。"""
    out = resample_bilinear(arr, target_shape)
    total = out.sum()
    if total <= 0:
        raise ValueError("重采样结果 sum<=0，无法归一化（应先进入 to_density degenerate 分支）")
    return out / total
