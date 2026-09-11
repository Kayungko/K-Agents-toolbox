"""aoi：矩形/多边形校验、掩码、嵌套重叠去重（C1 持有）。

- geometry.py：几何数学与 geometry 数据类（contracts/ 与 aoi/ 共用单一来源）
- mask.py：矩形半开边界与多边形光栅化掩码、并集
- summary.py：区域掩码构建、像素面积、汇总去重
"""

from .geometry import (
    Geometry,
    PolygonGeometry,
    RectGeometry,
    polygon_in_bounds,
    polygon_pixel_area_upper_bound,
    polygon_self_intersects,
    rect_in_bounds,
    shoelace_area,
    validate_polygon_points,
    validate_rect,
)
from .mask import geometry_mask, mask_area_px, polygon_mask, rect_mask, union_masks
from .summary import aggregate_masks, build_region_masks, full_canvas_mask, mask_areas

__all__ = [
    "Geometry",
    "PolygonGeometry",
    "RectGeometry",
    "aggregate_masks",
    "build_region_masks",
    "full_canvas_mask",
    "geometry_mask",
    "mask_area_px",
    "mask_areas",
    "polygon_in_bounds",
    "polygon_mask",
    "polygon_pixel_area_upper_bound",
    "polygon_self_intersects",
    "rect_in_bounds",
    "rect_mask",
    "shoelace_area",
    "union_masks",
    "validate_polygon_points",
    "validate_rect",
]
