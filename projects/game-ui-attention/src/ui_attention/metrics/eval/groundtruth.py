"""真值构造（benchmark-protocol.md §6.0/§6.9）。

- 注视点 → 像素：``px = floor(x)``（左上原点，与 AOI 坐标约定一致）；
  越界注视丢弃并计数入审计（复现官方规则）。
- δ 累加图 → 高斯模糊 F（σ=1° 换算见 blur.py），F 归一化 sum=1（经 to_density）。
- 离散经验分布 F̂(p) = w_p / Σw（IG/NSS/sAUC/AUC 用离散注视点或 F̂；CC/KL/SIM 用模糊 F）。
- 默认计数加权（每注视点权重 1）；时长加权（权重=FPOGD）作敏感性分析，两者不得混表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .blur import gaussian_blur, sigma_px_for_image
from .config import EvalConfig
from .density import to_density


@dataclass(frozen=True)
class FixationSet:
    """单图单窗口的真值注视点集。

    points: (n, 2) float64，列序 (x, y)，图像像素坐标；
    weights: (n,) float64，计数加权全 1 / 时长加权为注视时长；
    n_dropped_out_of_bounds: 越界丢弃计数（表 D 审计）。
    """

    points: np.ndarray
    weights: np.ndarray
    n_dropped_out_of_bounds: int = 0

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=np.float64)
        w = np.asarray(self.weights, dtype=np.float64)
        if pts.ndim == 0 or (pts.size == 0 and w.size == 0):
            pts = pts.reshape(0, 2)
            w = w.reshape(0)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points 形状必须是 (n, 2)，得到 {pts.shape}")
        if w.shape != (pts.shape[0],):
            raise ValueError(f"weights 长度必须与 points 一致：{w.shape} vs {pts.shape}")
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "weights", w)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def total_weight(self) -> float:
        return float(self.weights.sum())


def build_fixation_set(
    raw_points: Any,
    shape: tuple[int, int],
    *,
    weights: Any = None,
) -> FixationSet:
    """从原始坐标构建注视点集：越界丢弃（计数），默认计数加权。"""
    pts = np.asarray(raw_points, dtype=np.float64).reshape(-1, 2)
    if weights is None:
        w = np.ones(pts.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != pts.shape[0]:
            raise ValueError("weights 数量与注视点数量不一致")
    h, w_img = int(shape[0]), int(shape[1])
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w_img) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    n_dropped = int(np.count_nonzero(~inside))
    return FixationSet(points=pts[inside], weights=w[inside], n_dropped_out_of_bounds=n_dropped)


def pixel_indices(fix: FixationSet) -> tuple[np.ndarray, np.ndarray]:
    """注视点 → 像素索引：px = floor(x)（§6.9）。"""
    xi = np.floor(fix.points[:, 0]).astype(np.int64)
    yi = np.floor(fix.points[:, 1]).astype(np.int64)
    return xi, yi


def delta_map(fix: FixationSet, shape: tuple[int, int]) -> np.ndarray:
    """δ 累加图（权重累加到落点像素；越界注视须已由 build_fixation_set 丢弃）。"""
    h, w = int(shape[0]), int(shape[1])
    out = np.zeros((h, w), dtype=np.float64)
    if len(fix) == 0:
        return out
    xi, yi = pixel_indices(fix)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    np.add.at(out, (yi[ok], xi[ok]), fix.weights[ok])
    return out


def blurred_truth(fix: FixationSet, shape: tuple[int, int], config: EvalConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """模糊真值图 F = G_σ * F_delta，经 to_density 归一化（附录 A 管线）。"""
    h, w = int(shape[0]), int(shape[1])
    sigma = sigma_px_for_image(w, h, config)
    d = delta_map(fix, shape)
    blurred = gaussian_blur(d, sigma, mode=config.blur_mode, truncate=config.blur_truncate)
    F, flags = to_density(blurred, config.lam)
    flags = dict(flags)
    flags["sigma_px"] = sigma
    return F, flags


def discrete_truth(fix: FixationSet, shape: tuple[int, int]) -> np.ndarray:
    """离散经验注视分布 F̂(p) = w_p / Σw（未模糊；IG/NSS 的离散口径）。"""
    d = delta_map(fix, shape)
    total = d.sum()
    if total <= 0:
        return d
    return d / total
