"""概率转换与归一化（technical-design.md §7、§4.5）。

口径：``P = exp(L - logsumexp(L))``，有效画面上 ``sum(P) = 1``；
重采样后再次校验有限、非负及归一化（容差固定 1e-6，validation-plan §1）。
本模块只服务 AOI 统计路径（严格归一化概率图）；公开数据评估路径的数值防护
（to_density、λ 混合等）以 benchmark-protocol §6 为唯一口径，实现于 metrics/eval/，
两条路径不得互相混用数值或防护常数。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import ErrorCode, UiAttentionError

# sum=1 校验容差（validation-plan §1：初始单元检查容差 1e-6，实现在精度约束下固定）
DENSITY_SUM_TOLERANCE = 1e-6


def _as_float64_2d(array: Any, what: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise UiAttentionError(ErrorCode.INVALID_DENSITY, f"{what} 必须是 numpy.ndarray", {"got": type(array).__name__})
    if array.ndim != 2:
        raise UiAttentionError(ErrorCode.INVALID_DENSITY, f"{what} 形状必须是 (h, w)", {"ndim": array.ndim})
    if array.size == 0:
        raise UiAttentionError(ErrorCode.INVALID_DENSITY, f"{what} 不得为空")
    return array.astype(np.float64, copy=False)


def validate_probability(P: np.ndarray, tolerance: float = DENSITY_SUM_TOLERANCE) -> None:
    """校验概率图：全图非负、有限、sum 在容差内为 1；否则 INVALID_DENSITY（退出码 5）。"""
    arr = _as_float64_2d(P, "概率图")
    problems: list[str] = []
    if not np.all(np.isfinite(arr)):
        problems.append("含 NaN/Inf")
    if np.any(arr < 0):
        problems.append(f"含负值（min={float(arr.min()):.6g}）")
    total = float(arr.sum())
    if abs(total - 1.0) > tolerance:
        problems.append(f"sum={total!r} 与 1 的偏差超出容差 {tolerance}")
    if problems:
        raise UiAttentionError(
            ErrorCode.INVALID_DENSITY,
            "概率图无效（禁止把无效输出当成功结果）",
            {"problems": problems, "shape": list(arr.shape), "tolerance": tolerance},
        )


def logsumexp(L: np.ndarray) -> float:
    """数值稳定 logsumexp（float64）。"""
    arr = _as_float64_2d(L, "log-density")
    m = float(arr.max())
    if not np.isfinite(m):
        raise UiAttentionError(ErrorCode.INVALID_DENSITY, "log-density 含 NaN/Inf，无法归一化")
    return m + float(np.log(np.sum(np.exp(arr - m))))


def log_density_to_probability(L: np.ndarray) -> np.ndarray:
    """``P = exp(L - logsumexp(L))``，随后校验 sum=1（technical-design §7）。"""
    arr = _as_float64_2d(L, "log-density")
    if not np.all(np.isfinite(arr)):
        raise UiAttentionError(ErrorCode.INVALID_DENSITY, "log-density 含 NaN/Inf")
    P = np.exp(arr - logsumexp(arr))
    validate_probability(P)
    return P


def normalize_probability(P: np.ndarray, tolerance: float = DENSITY_SUM_TOLERANCE) -> np.ndarray:
    """对非负有限图做 sum=1 归一化并校验（适配层重归一化入口）。"""
    arr = _as_float64_2d(P, "概率图")
    if not np.all(np.isfinite(arr)):
        raise UiAttentionError(ErrorCode.INVALID_DENSITY, "概率图含 NaN/Inf")
    if np.any(arr < 0):
        raise UiAttentionError(ErrorCode.INVALID_DENSITY, "概率图含负值，拒绝归一化（不静默截断）")
    total = float(arr.sum())
    if total <= 0 or not np.isfinite(total):
        raise UiAttentionError(ErrorCode.INVALID_DENSITY, f"概率图 sum 无效：{total!r}，无法归一化")
    out = arr / total
    validate_probability(out, tolerance)
    return out


def bilinear_resample(P: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """float64 双线性重采样（像素中心对齐），纯 numpy 实现，保持全精度。"""
    arr = _as_float64_2d(P, "概率图")
    th, tw = int(target_shape[0]), int(target_shape[1])
    if th <= 0 or tw <= 0:
        raise UiAttentionError(ErrorCode.INVALID_DENSITY, f"目标尺寸无效：{target_shape}")
    sh, sw = arr.shape
    if (sh, sw) == (th, tw):
        return arr.copy()
    # 目标像素中心 → 源坐标（半像素对齐）
    ys = (np.arange(th, dtype=np.float64) + 0.5) * (sh / th) - 0.5
    xs = (np.arange(tw, dtype=np.float64) + 0.5) * (sw / tw) - 0.5
    ys = np.clip(ys, 0.0, sh - 1.0)
    xs = np.clip(xs, 0.0, sw - 1.0)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, sh - 1)
    x1 = np.minimum(x0 + 1, sw - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = arr[y0][:, x0] * (1.0 - wx) + arr[y0][:, x1] * wx
    bottom = arr[y1][:, x0] * (1.0 - wx) + arr[y1][:, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def resample_probability(
    P: np.ndarray, target_shape: tuple[int, int], tolerance: float = DENSITY_SUM_TOLERANCE
) -> np.ndarray:
    """重采样到目标尺寸后**再次归一化**并校验（technical-design §4.5：映射后再次归一化）。"""
    resampled = bilinear_resample(P, target_shape)
    return normalize_probability(resampled, tolerance)


def resample_to_original(
    density: np.ndarray, shape_mapping: dict[str, Any], tolerance: float = DENSITY_SUM_TOLERANCE
) -> np.ndarray:
    """按 PredictionResult.shape_mapping 把推理尺寸概率图映射回原图尺寸并重归一化。

    ``method`` 描述后端的前向预处理（如 direct_anisotropic_to_target），不参与逆变换判定；
    逆变换方法读 ``inverse`` 键（缺省 bilinear）。第一版只实现双线性逆变换；
    声明了其他逆变换或非零填充（letterbox）时结构化拒绝，不静默选择替代方法。
    """
    if not isinstance(shape_mapping, dict):
        raise UiAttentionError(ErrorCode.INVALID_DENSITY, "shape_mapping 必须是 dict")
    original = shape_mapping.get("original_shape")
    if not (isinstance(original, (tuple, list)) and len(original) == 2):
        raise UiAttentionError(ErrorCode.INVALID_DENSITY, "shape_mapping.original_shape 缺失或非法")
    inverse = str(shape_mapping.get("inverse", "bilinear")).lower()
    if "bilinear" not in inverse:
        raise UiAttentionError(
            ErrorCode.INVALID_DENSITY,
            f"不支持的逆变换方法 {inverse!r}（第一版只实现双线性；不静默选择其他方法）",
            {"inverse": inverse},
        )
    padding = shape_mapping.get("padding")
    if padding is not None:
        flat = list(padding.values()) if isinstance(padding, dict) else list(padding)
        if any(isinstance(v, (int, float)) and v != 0 for v in flat):
            raise UiAttentionError(
                ErrorCode.INVALID_DENSITY,
                "shape_mapping 声明了非零填充（letterbox）：第一版逆变换只支持纯缩放，拒绝猜测裁剪区域",
                {"padding": padding},
            )
    return resample_probability(density, (int(original[0]), int(original[1])), tolerance)
