"""σ=1° 视角模糊换算与高斯模糊（benchmark-protocol.md §6.0/§6.9）。

UEyes 换算：``σ = 40 × max(W/1920, H/1200)`` px（复现官方管线，σ≈40px@1920×1200 ≈ 1°）；
FiWI：σ ≈ 26 px（1° ≈ 26px 观看条件，用 fixed_px 规则）。
边界处理 = reflect（与 scipy.ndimage.gaussian_filter 默认一致，官方管线同源）；
高斯核截断半径 ≥ 4σ（truncate=4.0）。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import EvalConfig

# UEyes 官方参考分辨率与 σ（§2.1 官方管线：sigma = 40 / scalar, scalar = min(1920/w, 1200/h)
# ⇔ σ = 40 × max(w/1920, h/1200)）
UEYES_REFERENCE_WIDTH = 1920
UEYES_REFERENCE_HEIGHT = 1200
UEYES_SIGMA_REF_PX = 40.0


def ueyes_sigma_px(width: int, height: int) -> float:
    """UEyes 官方 σ 换算：40 × max(W/1920, H/1200) px。"""
    return UEYES_SIGMA_REF_PX * max(width / UEYES_REFERENCE_WIDTH, height / UEYES_REFERENCE_HEIGHT)


def sigma_px_for_image(width: int, height: int, config: EvalConfig) -> float:
    """按配置规则解析该图的模糊 σ（px）。"""
    if config.sigma_rule == "ueyes_1dva":
        return ueyes_sigma_px(width, height)
    if config.sigma_rule == "fixed_px":
        assert config.sigma_fixed_px is not None  # validate() 已保证
        return float(config.sigma_fixed_px)
    raise ValueError(f"未知 sigma_rule：{config.sigma_rule!r}")


def gaussian_blur(F: np.ndarray, sigma_px: float, *, mode: str = "reflect", truncate: float = 4.0) -> np.ndarray:
    """高斯模糊（float64）；mode 取 reflect|zero（配置冻结二选一）。"""
    if mode not in ("reflect", "zero"):
        raise ValueError(f"blur mode 必须是 reflect|zero，得到 {mode!r}")
    scipy_mode = "reflect" if mode == "reflect" else "constant"
    return gaussian_filter(F.astype(np.float64), sigma=sigma_px, mode=scipy_mode, truncate=truncate)
