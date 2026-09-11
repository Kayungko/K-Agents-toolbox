"""aoi/ 测试：矩形半开边界、多边形掩码、嵌套/重叠并集去重（validation-plan §1“重叠”行）。"""

from __future__ import annotations

import numpy as np
import pytest

from ui_attention.aoi import (
    PolygonGeometry,
    RectGeometry,
    aggregate_masks,
    build_region_masks,
    full_canvas_mask,
    geometry_mask,
    mask_area_px,
    mask_areas,
    polygon_mask,
    rect_mask,
    shoelace_area,
    union_masks,
)
from ui_attention.contracts.request import RegionSpec
from ui_attention.errors import ErrorCode, UiAttentionError

SHAPE = (48, 64)  # (H, W)


def _region(rid: str, geometry) -> RegionSpec:
    return RegionSpec(id=rid, geometry=geometry, source="manual", status="confirmed")


# ---------------------------------------------------------------------------
# 矩形：半开边界 [x, x+w) × [y, y+h)
# ---------------------------------------------------------------------------


def test_rect_mask_half_open_boundaries():
    m = rect_mask(SHAPE, x=10, y=8, width=32, height=16)
    assert mask_area_px(m) == 32 * 16  # 恰好 w*h 个像素
    # 包含：起点与最后一个内部像素
    assert m[8, 10] and m[8 + 15, 10 + 31]
    # 排除：半开边界外沿
    assert not m[8 + 16, 10]
    assert not m[8, 10 + 32]
    assert not m[7, 10]
    assert not m[8, 9]
    # 掩码外全 False
    assert m.sum() == (m[8:24, 10:42]).sum()


def test_rect_mask_edge_of_image():
    """贴边矩形：右下角抵达画布边界仍合法（x+w == W）。"""
    m = rect_mask(SHAPE, x=64 - 5, y=48 - 4, width=5, height=4)
    assert mask_area_px(m) == 20
    with pytest.raises(UiAttentionError) as exc:
        rect_mask(SHAPE, x=64 - 5, y=48 - 4, width=6, height=4)  # 超出 1 像素即拒绝
    assert exc.value.code is ErrorCode.INVALID_AOI


def test_rect_mask_rejects_zero_area_and_negative():
    with pytest.raises(UiAttentionError):
        rect_mask(SHAPE, x=0, y=0, width=0, height=5)
    with pytest.raises(UiAttentionError):
        rect_mask(SHAPE, x=-1, y=0, width=5, height=5)


# ---------------------------------------------------------------------------
# 多边形掩码
# ---------------------------------------------------------------------------


def test_polygon_mask_triangle_interior():
    tri = [(0, 0), (63, 0), (0, 47)]
    m = polygon_mask(SHAPE, tri)
    assert m[1, 1]  # 内部
    assert not m[47, 63]  # 对角远端外部
    # 像素面积与解析面积同量级（光栅化含边界像素，容差放宽到周长量级）
    analytic = abs(shoelace_area(tri))
    assert abs(mask_area_px(m) - analytic) < 200


def test_polygon_mask_rejects_self_intersection_and_bounds():
    with pytest.raises(UiAttentionError):
        polygon_mask(SHAPE, [(0, 0), (20, 20), (20, 0), (0, 20)])  # 蝴蝶结
    with pytest.raises(UiAttentionError):
        polygon_mask(SHAPE, [(0, 0), (10, 0), (10, 100)])  # 越界


def test_geometry_mask_dispatch():
    r = geometry_mask(SHAPE, RectGeometry(x=1, y=2, width=3, height=4))
    assert mask_area_px(r) == 12
    p = geometry_mask(SHAPE, PolygonGeometry(points=((0, 0), (10, 0), (10, 10), (0, 10))))
    assert mask_area_px(p) >= 100  # 含边界像素的方形


# ---------------------------------------------------------------------------
# 嵌套 / 重叠：并集去重，无重复计数
# ---------------------------------------------------------------------------


def test_union_deduplicates_overlap():
    a = rect_mask(SHAPE, 10, 10, 20, 10)  # 200 px
    b = rect_mask(SHAPE, 20, 10, 20, 10)  # 200 px，与 a 重叠 100 px
    u = union_masks([a, b])
    assert mask_area_px(u) == 300  # 200+200-100，重叠只计一次
    assert mask_area_px(u) < mask_area_px(a) + mask_area_px(b)


def test_union_nested_equals_parent():
    parent = rect_mask(SHAPE, 5, 5, 40, 30)
    child = rect_mask(SHAPE, 10, 10, 10, 10)
    u = union_masks([parent, child])
    assert np.array_equal(u, parent)  # 嵌套子区域并集 == 父区域


def test_aggregate_masks_is_union():
    a = rect_mask(SHAPE, 0, 0, 10, 10)
    b = rect_mask(SHAPE, 5, 5, 10, 10)
    assert np.array_equal(aggregate_masks([a, b]), union_masks([a, b]))
    with pytest.raises(UiAttentionError):
        union_masks([])


def test_union_rejects_mismatched_shapes():
    a = rect_mask(SHAPE, 0, 0, 10, 10)
    b = rect_mask((32, 32), 0, 0, 10, 10)
    with pytest.raises(UiAttentionError):
        union_masks([a, b])


# ---------------------------------------------------------------------------
# RegionSpec → 掩码构建
# ---------------------------------------------------------------------------


def test_build_region_masks_with_bounds_and_duplicate_checks():
    regions = [
        _region("btn", RectGeometry(x=10, y=8, width=32, height=16)),
        _region("banner", PolygonGeometry(points=((2, 2), (30, 4), (28, 20), (4, 18)))),
    ]
    masks = build_region_masks(regions, SHAPE)
    assert set(masks) == {"btn", "banner"}
    assert mask_areas(masks)["btn"] == 32 * 16

    # 越界区域拒绝
    oob = _region("oob", RectGeometry(x=60, y=40, width=32, height=16))
    with pytest.raises(UiAttentionError) as exc:
        build_region_masks([oob], SHAPE)
    assert exc.value.code is ErrorCode.INVALID_AOI

    # 重复 ID 拒绝
    with pytest.raises(UiAttentionError):
        build_region_masks([regions[0], _region("btn", RectGeometry(x=0, y=0, width=2, height=2))], SHAPE)


def test_full_canvas_mask():
    m = full_canvas_mask(SHAPE)
    assert m.all() and m.shape == SHAPE and m.dtype == np.bool_
