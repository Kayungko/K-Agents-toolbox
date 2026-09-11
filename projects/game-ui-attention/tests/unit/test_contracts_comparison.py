"""contracts.comparison（comparison v1，data-contract §6）测试。

覆盖 validation-plan §1 行：A/B 自比较差 0、配置不兼容拒绝（模型/权重/预处理/先验/画布）、
稳定 ID 配对、新增/移除单列且不编造 0 值、不匹配区域不计算差值。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fixtures import synthetic  # noqa: E402
from ui_attention.contracts import (  # noqa: E402
    COMPARISON_SCHEMA_VERSION,
    AnalysisRecord,
    build_comparison,
    check_compatibility,
)
from ui_attention.errors import ErrorCode, UiAttentionError  # noqa: E402


def _record(**overrides) -> AnalysisRecord:
    return AnalysisRecord.from_dict(synthetic.sample_analysis_dict(**overrides))


def test_self_comparison_delta_zero():
    """A/B 自比较：相同结果差值为 0（validation-plan §1）。"""
    rec = _record()
    cmp = build_comparison(rec, rec)
    assert cmp.schema_version == COMPARISON_SCHEMA_VERSION
    assert cmp.compatibility.compatible is True
    assert len(cmp.matched) == 1
    row = cmp.matched[0]
    assert row["delta_pp"] == 0.0
    assert row["mass_before"] == row["mass_after"]
    assert row["area_px_change"] == 0
    assert cmp.added == () and cmp.removed == ()
    d = cmp.to_dict()
    assert d["regions"]["matched"][0]["delta_pp"] == 0.0


def test_delta_pp_value():
    before = _record(regions=[synthetic.sample_region_result(mass=0.20)])
    after = _record(analysis_id="analysis-test-0002", regions=[synthetic.sample_region_result(mass=0.35)])
    cmp = build_comparison(before, after)
    row = cmp.matched[0]
    assert row["delta_pp"] == pytest.approx(15.0)  # 100*(0.35-0.20)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda o: o["model"].update(backend_id="deepgaze-iie"),  # 模型不同
        lambda o: o["model"].update(version="9.9.9"),  # 后端版本不同
        lambda o: o["model"].update(weights_sha256=["f" * 64]),  # 权重哈希不同
        lambda o: o["profile"].update(config_hash="e" * 64),  # 预处理/先验/观看配置合成哈希不同
        lambda o: o.update(metrics_version="aoi-metrics/v2"),  # 指标版本不同
        # 注：evidence_type 非 model_prediction 的记录在 analysis v1 校验层即被拒绝，
        # 不可能进入比较（AnalysisRecord.validate 强制 EVIDENCE_TYPE）。
    ],
)
def test_incompatible_config_rejected(mutate):
    base = synthetic.sample_analysis_dict()
    other = copy.deepcopy(base)
    mutate(other)
    with pytest.raises(UiAttentionError) as exc:
        build_comparison(_record_from(base), _record_from(other))
    assert exc.value.code is ErrorCode.COMPARISON_INCOMPATIBLE
    assert exc.value.exit_code == 6
    assert exc.value.details["mismatches"], "不兼容原因必须保留"


def test_canvas_size_mismatch_rejected():
    """第一版要求相同画布尺寸（technical-design §9）。"""
    before = _record()
    after_obj = synthetic.sample_analysis_dict(width=128, height=96)
    # 画布不同还会连带 input 校验通过（只是尺寸不同）
    with pytest.raises(UiAttentionError) as exc:
        build_comparison(before, _record_from(after_obj))
    assert exc.value.code is ErrorCode.COMPARISON_INCOMPATIBLE
    fields = {m["field"] for m in exc.value.details["mismatches"]}
    assert {"input.width", "input.height"} <= fields


def _record_from(obj) -> AnalysisRecord:
    return AnalysisRecord.from_dict(obj)


def test_check_compatibility_report_structure():
    rep = check_compatibility(_record(), _record())
    assert rep.compatible and not rep.mismatches
    assert "画布尺寸相同" in rep.canvas_note
    assert rep.exploratory is False


def test_exploratory_flag_on_different_screen_or_goal():
    base = synthetic.sample_analysis_dict()
    other = copy.deepcopy(base)
    other["input"]["screen_type"] = "shop"
    other["player_goal"] = "比较价格"
    rep = check_compatibility(_record_from(base), _record_from(other))
    assert rep.compatible is True  # 配置仍兼容
    assert rep.exploratory is True
    assert any("画面类型" in r for r in rep.exploratory_reasons)
    assert any("玩家目标" in r for r in rep.exploratory_reasons)


def test_added_removed_regions_listed_separately_no_fabricated_zero():
    """新增/移除区域单列；缺失一侧不编造 0 值、不计算差值。"""
    before = _record(
        regions=[
            synthetic.sample_region_result(rid="kept", mass=0.3),
            synthetic.sample_region_result(rid="gone", mass=0.1),
        ]
    )
    after = _record(
        analysis_id="analysis-test-0002",
        regions=[
            synthetic.sample_region_result(rid="kept", mass=0.4),
            synthetic.sample_region_result(rid="newcomer", mass=0.05),
        ],
    )
    cmp = build_comparison(before, after)
    assert [r["id"] for r in cmp.matched] == ["kept"]
    assert cmp.matched[0]["delta_pp"] == pytest.approx(10.0)
    assert [r["id"] for r in cmp.added] == ["newcomer"]
    assert [r["id"] for r in cmp.removed] == ["gone"]
    # added/removed 行不含 delta_pp（不匹配区域不计算差值）
    assert "delta_pp" not in cmp.added[0]
    assert "delta_pp" not in cmp.removed[0]


def test_position_and_size_may_change_with_stable_id():
    """稳定 ID 配对：位置和大小可以改变，但各自提供有效边界。"""
    before = _record(regions=[synthetic.sample_region_result(rid="btn", mass=0.2, area_px=400)])
    after = _record(
        analysis_id="analysis-test-0002",
        regions=[
            synthetic.sample_region_result(
                rid="btn",
                mass=0.3,
                area_px=800,
                geometry={"type": "rect", "x": 20, "y": 10, "width": 40, "height": 20},
            )
        ],
    )
    cmp = build_comparison(before, after)
    row = cmp.matched[0]
    assert row["geometry_before"] != row["geometry_after"]
    assert row["area_px_change"] == 400  # 面积变化同时报告，防止把面积扩大解释成效率提升
    assert row["delta_pp"] == pytest.approx(10.0)


def test_comparison_limitations_wording():
    cmp = build_comparison(_record(), _record())
    joined = " ".join(cmp.limitations)
    assert "点击率" in joined  # 禁止表述红线：delta 不代表点击率提升
    assert "0 值" in joined
