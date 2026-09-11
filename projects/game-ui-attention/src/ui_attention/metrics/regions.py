"""AOI 区域指标（technical-design.md §7）。

| 指标 | 定义 |
| --- | --- |
| probability_mass | ``sum(P[R])``，预测注视分布落入区域的质量，范围 0~1 |
| area_fraction | ``area(R) / area(image)``（面积以像素掩码计数为准） |
| relative_density | ``probability_mass / area_fraction`` |
| delta_pp | ``100 * (mass_after - mass_before)``，百分点 |

语义边界：占比 8% 不表示"8% 玩家会看到"；相对密度较高不自动代表设计更好；
区域尺寸变化时同时报告面积变化（本模块返回 area_px 供上层并列展示）。
保留浮点原始值，展示舍入不影响计算。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ..errors import ErrorCode, UiAttentionError
from .probability import validate_probability


def region_statistics(P: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """单区域指标。P 必须是已校验的 sum=1 概率图；mask 为 bool 像素掩码。"""
    validate_probability(P)
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != P.shape:
        raise UiAttentionError(
            ErrorCode.INVALID_AOI,
            "掩码必须是与概率图同形状的 bool 数组",
            {"mask_shape": getattr(mask, "shape", None), "P_shape": list(P.shape)},
        )
    area_px = int(mask.sum())
    if area_px == 0:
        # 零面积按输入错误处理（technical-design §7）。
        raise UiAttentionError(ErrorCode.INVALID_AOI, "区域面积为零（像素掩码为空），拒绝统计")
    total_pixels = int(P.size)
    mass = float(P[mask].sum())
    area_fraction = area_px / total_pixels
    return {
        "area_px": area_px,
        "area_fraction": area_fraction,
        "probability_mass": mass,
        "relative_density": mass / area_fraction,
    }


def delta_pp(mass_after: float, mass_before: float) -> float:
    """A/B 百分点差：``100 * (mass_after - mass_before)``。"""
    return 100.0 * (float(mass_after) - float(mass_before))


def union_statistics(P: np.ndarray, masks: Sequence[np.ndarray]) -> dict[str, Any]:
    """多区域汇总：按像素掩码**并集去重**（嵌套/重叠不重复计数，technical-design §6）。"""
    if not masks:
        raise UiAttentionError(ErrorCode.INVALID_AOI, "并集汇总至少需要一个区域掩码")
    union = np.zeros(P.shape, dtype=np.bool_)
    for m in masks:
        if m.shape != P.shape or m.dtype != np.bool_:
            raise UiAttentionError(ErrorCode.INVALID_AOI, "并集汇总的掩码必须同形状且为 bool")
        union |= m
    stats = region_statistics(P, union)
    overlap_px = int(sum(int(m.sum()) for m in masks) - stats["area_px"])
    stats["overlap_dedup_px"] = max(overlap_px, 0)
    return stats


def uniform_sanity_check(P: np.ndarray, mask: np.ndarray, tolerance: float = 1e-9) -> dict[str, Any]:
    """均匀图 sanity：mass == 面积占比、relative_density == 1（validation-plan §1）。"""
    stats = region_statistics(P, mask)
    uniform = np.full(P.shape, 1.0 / P.size, dtype=np.float64)
    is_uniform = bool(np.allclose(P, uniform, rtol=0, atol=tolerance))
    mass_matches_area = abs(stats["probability_mass"] - stats["area_fraction"]) <= tolerance
    rel_is_one = abs(stats["relative_density"] - 1.0) <= tolerance * max(1.0, stats["relative_density"])
    return {
        "is_uniform_map": is_uniform,
        "mass_equals_area_fraction": mass_matches_area,
        "relative_density_is_one": rel_is_one,
        "statistics": stats,
    }


def compare_region_rows(
    before_regions: Iterable[dict[str, Any]],
    after_regions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """按稳定 ID 配对区域行并计算 delta_pp；新增/移除单列，不编造缺失一侧的 0 值。"""
    before_by_id = {r["id"]: r for r in before_regions}
    after_by_id = {r["id"]: r for r in after_regions}
    matched: list[dict[str, Any]] = []
    for rid in sorted(set(before_by_id) & set(after_by_id)):
        b, a = before_by_id[rid], after_by_id[rid]
        matched.append(
            {
                "id": rid,
                "label": a.get("label") or b.get("label"),
                "mass_before": b["probability_mass"],
                "mass_after": a["probability_mass"],
                "delta_pp": delta_pp(a["probability_mass"], b["probability_mass"]),
                "area_px_before": b["area_px"],
                "area_px_after": a["area_px"],
                "area_px_change": int(a["area_px"]) - int(b["area_px"]),
                "relative_density_before": b["relative_density"],
                "relative_density_after": a["relative_density"],
                "geometry_before": b.get("geometry"),
                "geometry_after": a.get("geometry"),
            }
        )
    return {
        "matched": matched,
        "added": [after_by_id[rid] for rid in sorted(set(after_by_id) - set(before_by_id))],
        "removed": [before_by_id[rid] for rid in sorted(set(before_by_id) - set(after_by_id))],
    }
