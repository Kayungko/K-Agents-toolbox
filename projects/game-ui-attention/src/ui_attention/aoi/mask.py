"""AOI 掩码生成（technical-design.md §6）。

- 矩形：半开边界 ``[x, x+w) × [y, y+h)`` → 恰好 w*h 个像素；
- 多边形：PIL 多边形填充光栅化（含边界像素），面积以像素掩码计数为准；
- 嵌套/重叠汇总：按像素掩码并集去重（:func:`union_masks`），指标不可简单求和。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..errors import ErrorCode, UiAttentionError
from .geometry import (
    Geometry,
    PolygonGeometry,
    RectGeometry,
    polygon_in_bounds,
    rect_in_bounds,
    validate_polygon_points,
    validate_rect,
)


def rect_mask(shape: tuple[int, int], x: int, y: int, width: int, height: int) -> np.ndarray:
    """矩形掩码：半开区间行 y..y+h-1、列 x..x+w-1 为 True。"""
    h, w = int(shape[0]), int(shape[1])
    errors = validate_rect(x, y, width, height)
    errors += rect_in_bounds(x, y, width, height, w, h)
    if errors:
        raise UiAttentionError(ErrorCode.INVALID_AOI, "矩形掩码参数无效", {"errors": errors})
    mask = np.zeros((h, w), dtype=np.bool_)
    mask[y : y + height, x : x + width] = True
    return mask


def polygon_mask(shape: tuple[int, int], points: Sequence[tuple[int, int]]) -> np.ndarray:
    """多边形掩码（PIL 填充光栅化）；先做结构校验与边界校验。"""
    h, w = int(shape[0]), int(shape[1])
    pts = [(int(p[0]), int(p[1])) for p in points]
    errors = validate_polygon_points(pts)
    errors += polygon_in_bounds(pts, w, h)
    if errors:
        raise UiAttentionError(ErrorCode.INVALID_AOI, "多边形掩码参数无效", {"errors": errors})
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(pts, fill=255)
    return np.asarray(img, dtype=np.uint8) > 0


def geometry_mask(shape: tuple[int, int], geometry: Geometry) -> np.ndarray:
    """按 RegionSpec.geometry 生成掩码。"""
    if isinstance(geometry, RectGeometry):
        return rect_mask(shape, geometry.x, geometry.y, geometry.width, geometry.height)
    if isinstance(geometry, PolygonGeometry):
        return polygon_mask(shape, geometry.points)
    raise UiAttentionError(
        ErrorCode.INVALID_AOI,
        f"未知 geometry 类型 {type(geometry).__name__}",
    )


def union_masks(masks: Sequence[np.ndarray]) -> np.ndarray:
    """像素级并集（嵌套/重叠去重的唯一汇总方式）。"""
    if not masks:
        raise UiAttentionError(ErrorCode.INVALID_AOI, "并集至少需要一个掩码")
    shape = masks[0].shape
    union = np.zeros(shape, dtype=np.bool_)
    for m in masks:
        if m.shape != shape or m.dtype != np.bool_:
            raise UiAttentionError(ErrorCode.INVALID_AOI, "并集掩码必须同形状且为 bool")
        union |= m
    return union


def mask_area_px(mask: np.ndarray) -> int:
    """面积 = 掩码 True 像素计数（矩形半开边界下恰为 w*h）。"""
    return int(mask.sum())
