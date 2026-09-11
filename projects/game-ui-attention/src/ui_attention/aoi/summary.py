"""AOI 区域掩码汇总（technical-design.md §6）。

允许嵌套和重叠，但指标不可简单求和：父区域使用掩码并集；
汇总多个区域时按像素去重（:func:`aggregate_masks`）。
没有可靠区域时允许全图分析（:func:`full_canvas_mask`），不虚构精确的按钮占比。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..errors import ErrorCode, UiAttentionError
from .mask import geometry_mask, mask_area_px, union_masks

if TYPE_CHECKING:  # 仅类型标注引用，避免 aoi ↔ contracts 运行时循环导入
    from ..contracts.request import RegionSpec


def build_region_masks(
    regions: Sequence[RegionSpec],
    image_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """为每个区域生成 bool 掩码；先做边界校验（越界拒绝，退出码 2 语义）。"""
    masks: dict[str, np.ndarray] = {}
    h, w = int(image_shape[0]), int(image_shape[1])
    for region in regions:
        region.check_bounds(w, h)
        if region.id in masks:
            raise UiAttentionError(
                ErrorCode.INVALID_AOI,
                "区域 ID 重复，拒绝生成掩码",
                {"region_id": region.id},
            )
        masks[region.id] = geometry_mask((h, w), region.geometry)
    return masks


def aggregate_masks(masks: Sequence[np.ndarray]) -> np.ndarray:
    """嵌套/重叠区域的汇总 = 像素掩码并集（去重，无重复计数）。"""
    return union_masks(masks)


def full_canvas_mask(image_shape: tuple[int, int]) -> np.ndarray:
    """全图分析掩码（无可靠区域时的合法退化，不虚构区域占比）。"""
    return np.ones((int(image_shape[0]), int(image_shape[1])), dtype=np.bool_)


def mask_areas(masks: dict[str, np.ndarray]) -> dict[str, int]:
    """各区域像素面积（矩形半开边界下 rect 面积恰为 w*h）。"""
    return {rid: mask_area_px(m) for rid, m in masks.items()}
