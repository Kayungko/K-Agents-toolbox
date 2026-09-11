"""metrics.regions 测试：均匀图、已知局部质量、delta_pp、并集去重（validation-plan §1 三行）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fixtures import synthetic  # noqa: E402
from ui_attention.aoi import polygon_mask, rect_mask, union_masks  # noqa: E402
from ui_attention.errors import ErrorCode, UiAttentionError  # noqa: E402
from ui_attention.metrics import (  # noqa: E402
    compare_region_rows,
    delta_pp,
    region_statistics,
    uniform_sanity_check,
    union_statistics,
)

SHAPE = (48, 64)
TOTAL_PX = 48 * 64


# ---------------------------------------------------------------------------
# 均匀图：mass == 面积占比、relative_density == 1
# ---------------------------------------------------------------------------


def test_uniform_map_rect_region():
    P = synthetic.uniform_probability(SHAPE)
    mask = rect_mask(SHAPE, 10, 8, 32, 16)
    s = region_statistics(P, mask)
    assert s["area_px"] == 512
    assert s["area_fraction"] == pytest.approx(512 / TOTAL_PX)
    assert s["probability_mass"] == pytest.approx(s["area_fraction"], abs=1e-12)
    assert s["relative_density"] == pytest.approx(1.0, abs=1e-9)


def test_uniform_map_polygon_region():
    P = synthetic.uniform_probability(SHAPE)
    mask = polygon_mask(SHAPE, [(2, 2), (30, 4), (28, 20), (4, 18)])
    s = region_statistics(P, mask)
    assert s["probability_mass"] == pytest.approx(s["area_fraction"], abs=1e-12)
    assert s["relative_density"] == pytest.approx(1.0, abs=1e-9)


def test_uniform_sanity_check_flags():
    P = synthetic.uniform_probability(SHAPE)
    mask = rect_mask(SHAPE, 0, 0, 8, 8)
    report = uniform_sanity_check(P, mask)
    assert report["is_uniform_map"] is True
    assert report["mass_equals_area_fraction"] is True
    assert report["relative_density_is_one"] is True


def test_non_uniform_map_detected_by_sanity():
    P = synthetic.block_probability(SHAPE, (0, 0, 8, 8), mass=0.5)
    mask = rect_mask(SHAPE, 0, 0, 8, 8)
    report = uniform_sanity_check(P, mask)
    assert report["is_uniform_map"] is False
    assert report["mass_equals_area_fraction"] is False
    assert report["statistics"]["probability_mass"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 已知局部质量：手工分配概率的区域统计与解析结果一致
# ---------------------------------------------------------------------------


def test_known_local_mass_exact_block():
    """构造：block 内总质量恰 0.3 → 区域统计 == 解析值。"""
    P = synthetic.block_probability(SHAPE, (10, 8, 32, 16), mass=0.3)
    mask = rect_mask(SHAPE, 10, 8, 32, 16)
    s = region_statistics(P, mask)
    area_fraction = 512 / TOTAL_PX
    assert s["probability_mass"] == pytest.approx(0.3, abs=1e-12)
    assert s["area_fraction"] == pytest.approx(area_fraction)
    assert s["relative_density"] == pytest.approx(0.3 / area_fraction, abs=1e-9)


def test_known_local_mass_partial_region():
    """区域只覆盖质量块的一半 → mass 为解析的一半。"""
    P = synthetic.block_probability(SHAPE, (10, 8, 32, 16), mass=0.4)
    mask = rect_mask(SHAPE, 10, 8, 16, 16)  # 块左半
    s = region_statistics(P, mask)
    assert s["probability_mass"] == pytest.approx(0.2, abs=1e-12)


def test_known_local_mass_outside_region():
    """质量块之外的区域：mass == (1-0.25) 按面积均分的解析值。"""
    P = synthetic.block_probability(SHAPE, (0, 0, 8, 8), mass=0.25)
    mask = rect_mask(SHAPE, 40, 40, 8, 8)  # 完全在块外
    rest_total = TOTAL_PX - 64
    expected = (1 - 0.25) * 64 / rest_total
    s = region_statistics(P, mask)
    assert s["probability_mass"] == pytest.approx(expected, abs=1e-12)


def test_region_statistics_preserves_raw_float():
    """保留浮点原始值（展示舍入不影响计算）。"""
    P = synthetic.block_probability(SHAPE, (10, 8, 32, 16), mass=1 / 3)
    s = region_statistics(P, rect_mask(SHAPE, 10, 8, 32, 16))
    assert s["probability_mass"] == pytest.approx(1 / 3, abs=1e-12)
    assert s["probability_mass"] != round(s["probability_mass"], 3)


# ---------------------------------------------------------------------------
# 输入错误：零面积 / 掩码不匹配
# ---------------------------------------------------------------------------


def test_zero_area_mask_rejected():
    P = synthetic.uniform_probability(SHAPE)
    empty = np.zeros(SHAPE, dtype=np.bool_)
    with pytest.raises(UiAttentionError) as exc:
        region_statistics(P, empty)
    assert exc.value.code is ErrorCode.INVALID_AOI


def test_mask_shape_mismatch_rejected():
    P = synthetic.uniform_probability(SHAPE)
    with pytest.raises(UiAttentionError):
        region_statistics(P, rect_mask((32, 32), 0, 0, 4, 4))


def test_invalid_probability_rejected_before_stats():
    bad = np.full(SHAPE, 2.0)  # sum != 1
    with pytest.raises(UiAttentionError) as exc:
        region_statistics(bad, rect_mask(SHAPE, 0, 0, 4, 4))
    assert exc.value.code is ErrorCode.INVALID_DENSITY


# ---------------------------------------------------------------------------
# delta_pp 与 A/B 行配对
# ---------------------------------------------------------------------------


def test_delta_pp_definition():
    assert delta_pp(0.35, 0.20) == pytest.approx(15.0)
    assert delta_pp(0.20, 0.35) == pytest.approx(-15.0)
    assert delta_pp(0.25, 0.25) == 0.0


def test_compare_region_rows_pairing():
    before = [synthetic.sample_region_result(rid="a", mass=0.2), synthetic.sample_region_result(rid="b", mass=0.1)]
    after = [synthetic.sample_region_result(rid="a", mass=0.3), synthetic.sample_region_result(rid="c", mass=0.05)]
    out = compare_region_rows(before, after)
    assert [r["id"] for r in out["matched"]] == ["a"]
    assert out["matched"][0]["delta_pp"] == pytest.approx(10.0)
    assert [r["id"] for r in out["added"]] == ["c"]
    assert [r["id"] for r in out["removed"]] == ["b"]
    # 不匹配区域不计算差值、不补 0
    assert all("delta_pp" not in r for r in out["added"] + out["removed"])


# ---------------------------------------------------------------------------
# 并集去重统计（重叠无重复计数）
# ---------------------------------------------------------------------------


def test_union_statistics_deduplicates_overlap():
    P = synthetic.uniform_probability(SHAPE)
    a = rect_mask(SHAPE, 10, 10, 20, 10)
    b = rect_mask(SHAPE, 20, 10, 20, 10)
    s = union_statistics(P, [a, b])
    assert s["area_px"] == 300
    assert s["probability_mass"] == pytest.approx(300 / TOTAL_PX, abs=1e-12)
    assert s["overlap_dedup_px"] == 100  # 200+200-300
    # 简单求和会重复计数（证明去重生效）
    naive = region_statistics(P, a)["probability_mass"] + region_statistics(P, b)["probability_mass"]
    assert naive > s["probability_mass"]


def test_union_statistics_with_block_probability():
    P = synthetic.block_probability(SHAPE, (10, 10, 20, 10), mass=0.5)
    a = rect_mask(SHAPE, 10, 10, 20, 10)  # 质量块
    b = rect_mask(SHAPE, 15, 12, 4, 4)  # 块内子区域（嵌套）
    s = union_statistics(P, [a, b])
    assert np.array_equal(union_masks([a, b]), a)
    assert s["probability_mass"] == pytest.approx(0.5, abs=1e-12)
