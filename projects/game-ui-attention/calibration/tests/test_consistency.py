"""consistency.py 单测：合成 responses（中心点/边角点/均匀框）+ 基线 win-rate 边界。"""

from __future__ import annotations

import consistency as c
import numpy as np
import pytest

SHAPE = (101, 101)


def _center_model() -> np.ndarray:
    return c.gaussian_density(SHAPE, (50.0, 50.0), 0.25)


# ---------------------------------------------------------------------------
# 密度百分位
# ---------------------------------------------------------------------------


def test_percentile_at_uniform_is_half() -> None:
    uni = c.uniform_density(SHAPE)
    assert c.percentile_at(uni, 50, 50) == pytest.approx(0.5, abs=1e-12)
    assert c.percentile_at(uni, 0, 0) == pytest.approx(0.5, abs=1e-12)


def test_center_point_percentile_high() -> None:
    model = _center_model()
    assert c.percentile_at(model, 50, 50) > 0.9
    cb = c.center_bias_density(SHAPE, 0.25)
    assert c.percentile_at(cb, 50, 50) > 0.9


def test_corner_point_percentile_low() -> None:
    model = _center_model()
    assert c.percentile_at(model, 0, 0) < 0.1
    cb = c.center_bias_density(SHAPE, 0.25)
    assert c.percentile_at(cb, 0, 0) < 0.1


# ---------------------------------------------------------------------------
# 框统计（与技术方案 §7 同口径）
# ---------------------------------------------------------------------------


def test_uniform_box_stats_matches_area_fraction() -> None:
    model = c.uniform_density(SHAPE)
    boxes = [{"x": 10, "y": 10, "width": 20, "height": 30}]
    st = c.box_stats(model, boxes)
    assert st["area_px"] == 20 * 30
    assert st["mass"] == pytest.approx(st["area_fraction"], abs=1e-9)
    assert st["relative_density"] == pytest.approx(1.0, abs=1e-9)


def test_box_union_dedup() -> None:
    model = c.uniform_density((10, 10))
    boxes = [{"x": 0, "y": 0, "width": 5, "height": 5}, {"x": 2, "y": 2, "width": 5, "height": 5}]
    st = c.box_stats(model, boxes)
    union_px = 5 * 5 + 5 * 5 - 3 * 3  # 重叠 3×3 去重 → 41
    assert st["area_px"] == union_px
    assert st["area_fraction"] == pytest.approx(union_px / 100, abs=1e-12)


def test_center_bias_normalized() -> None:
    cb = c.center_bias_density(SHAPE, 0.25)
    assert cb.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(cb >= 0)


# ---------------------------------------------------------------------------
# 基线 win-rate 边界
# ---------------------------------------------------------------------------


def test_win_rate_model_equals_cb_is_zero() -> None:
    model = c.center_bias_density(SHAPE, 0.25)
    cb = c.center_bias_density(SHAPE, 0.25)
    uni = c.uniform_density(SHAPE)
    pair = c.compute_pair({"x": 50, "y": 50}, [{"x": 30, "y": 30, "width": 40, "height": 40}], model, cb, uni)
    # 模型与 CB 完全一致 → 无严格胜出 → 胜率恰为 0
    assert pair["first_look_percentile_model"] == pytest.approx(pair["first_look_percentile_cb"], abs=1e-12)
    assert c._win_rate([pair["first_look_percentile_model"]], [pair["first_look_percentile_cb"]]) == 0.0
    assert c._win_rate([pair["box_union_mass_model"]], [pair["box_union_mass_cb"]]) == 0.0


def test_win_rate_model_equals_uniform_is_zero() -> None:
    model = c.uniform_density(SHAPE)
    cb = c.center_bias_density(SHAPE, 0.25)
    uni = c.uniform_density(SHAPE)
    pair = c.compute_pair({"x": 20, "y": 20}, [{"x": 10, "y": 10, "width": 20, "height": 20}], model, cb, uni)
    assert pair["first_look_percentile_model"] == pytest.approx(0.5, abs=1e-12)
    assert c._win_rate([pair["first_look_percentile_model"]], [pair["first_look_percentile_uniform"]]) == 0.0
    assert (
        c._win_rate([pair["box_union_relative_density_model"]], [pair["box_union_relative_density_uniform"]]) == 0.0
    )


def test_win_rate_model_dominates_uniform_box_mass() -> None:
    shape = (32, 32)
    boxes = [{"x": 8, "y": 8, "width": 16, "height": 16}]
    mask = c.box_mask(shape, boxes)
    model = np.zeros(shape, dtype=np.float64)
    model[mask] = 1.0
    model /= model.sum()
    cb = c.center_bias_density(shape, 0.25)
    uni = c.uniform_density(shape)
    pair = c.compute_pair({"x": 16, "y": 16}, boxes, model, cb, uni)
    assert pair["box_union_mass_model"] == pytest.approx(1.0, abs=1e-9)
    assert pair["box_union_mass_uniform"] == pytest.approx(0.25, abs=1e-9)
    assert c._win_rate([pair["box_union_mass_model"]], [pair["box_union_mass_uniform"]]) == 1.0
    assert c._win_rate([pair["box_union_mass_model"]], [pair["box_union_mass_cb"]]) == 1.0


# ---------------------------------------------------------------------------
# 端到端 evaluate + 排除标准 + 报告渲染
# ---------------------------------------------------------------------------


def _synthetic_config() -> dict:
    return {
        "schema_version": "game-ui-attention-calibration-package/v1",
        "study_id": "stub",
        "duration_seconds": 3.0,
        "shuffle_seed": 1,
        "images": [
            {
                "index": 0, "filename": "a.png", "source_name": "a.png", "sha256": "aaa",
                "width": 32, "height": 32, "screen_type": "x",
            },
            {
                "index": 1, "filename": "b.png", "source_name": "b.png", "sha256": "bbb",
                "width": 32, "height": 32, "screen_type": "x",
            },
        ],
    }


def _response(pid: str, completed: bool, entries: list[dict]) -> dict:
    return {
        "schema_version": "game-ui-attention-calibration-response/v1",
        "study_id": "stub",
        "participant_id": pid,
        "completed": completed,
        "images": entries,
    }


def _entry(sha: str, dx: int = 0) -> dict:
    return {
        "sha256": sha,
        "responded": True,
        "first_look": {"x": 16 + dx, "y": 16 + dx},
        "boxes": [{"x": 8 + dx, "y": 8 + dx, "width": 16, "height": 16}],
    }


def _densities() -> dict:
    d = c.center_bias_density((32, 32), 0.25)
    return {"aaa": {"density": d, "width": 32, "height": 32}, "bbb": {"density": d, "width": 32, "height": 32}}


def test_evaluate_end_to_end() -> None:
    report = c.evaluate(
        _synthetic_config(), [_response("p1", True, [_entry("aaa", 0), _entry("bbb", 2)])], _densities(), 0.25
    )
    assert report["n_pairs"] == 2
    assert report["n_participants_included"] == 1
    assert report["aggregate"]["first_look_percentile"]["model_mean"] > 0.5
    assert report["aggregate"]["n"] == 2
    assert report["by_screen_type"]["x"]["n"] == 2
    assert len(report["limitations"]) >= 5
    md = c.render_markdown(report)
    assert "粗标注≠眼动真值" in md
    assert "中心偏置" in md
    assert "样本量小" in md


def test_evaluate_excludes_incomplete_participant() -> None:
    responses = [
        _response("p_bad", False, [_entry("aaa", 0), _entry("bbb", 2)]),
        _response("p_good", True, [_entry("aaa", 0), _entry("bbb", 2)]),
    ]
    report = c.evaluate(_synthetic_config(), responses, _densities(), 0.25)
    assert report["n_participants_included"] == 1
    assert report["n_pairs"] == 2
    assert any(e["participant_id"] == "p_bad" for e in report["excluded_participants"])


def test_evaluate_excludes_random_click_participant() -> None:
    entries = [
        _entry("aaa"),
        {
            "sha256": "bbb",
            "responded": True,
            "first_look": {"x": 16, "y": 16},
            "boxes": [{"x": 8, "y": 8, "width": 16, "height": 16}],
        },
    ]
    report = c.evaluate(_synthetic_config(), [_response("p_rand", True, entries)], _densities(), 0.25)
    assert any(e["participant_id"] == "p_rand" for e in report["excluded_participants"])
