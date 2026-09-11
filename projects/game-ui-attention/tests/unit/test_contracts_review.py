"""contracts.review 校验测试（data-contract §4 review.json / technical-design §8 证据类型）。"""

from __future__ import annotations

import pytest

from ui_attention.contracts import REVIEW_SCHEMA_VERSION, ReviewRecord, load_review
from ui_attention.errors import ErrorCode, UiAttentionError


def _finding(**overrides):
    f = {
        "id": "f1",
        "region_ids": ["claim-button"],
        "evidence_refs": ["analysis.json#/regions/0/probability_mass"],
        "evidence_type": "computed",
        "observation": "领取按钮 probability_mass 0.30，relative_density 2.9",
        "inference": "模型预测该区域获得高于平均的注视质量",
        "recommendation": "保持当前对比度；如需更强引导可增大与背景的差异",
        "validation_needed": "需真人评审确认引导性（模型预测非真实眼动）",
    }
    f.update(overrides)
    return f


def _review(**overrides):
    obj = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "analysis_id": "analysis-test-0001",
        "findings": [_finding()],
    }
    obj.update(overrides)
    return obj


def test_valid_review_roundtrip():
    rec = ReviewRecord.from_dict(_review())
    assert rec.findings[0]["id"] == "f1"
    assert rec.analysis_id == "analysis-test-0001"
    ReviewRecord.from_dict(rec.to_dict())


@pytest.mark.parametrize("version", [None, "review/v2", "game-ui-attention-review/v9"])
def test_unknown_schema_rejected(version):
    with pytest.raises(UiAttentionError) as exc:
        ReviewRecord.from_dict(_review(schema_version=version))
    assert exc.value.code is ErrorCode.INVALID_SCHEMA_VERSION


@pytest.mark.parametrize("etype", ["measured", "COMPUTED", None, "fact"])
def test_evidence_type_enum(etype):
    with pytest.raises(UiAttentionError):
        ReviewRecord.from_dict(_review(findings=[_finding(evidence_type=etype)]))


def test_all_four_evidence_types_allowed():
    for t in ("computed", "observed", "inferred", "unverified"):
        rec = ReviewRecord.from_dict(_review(findings=[_finding(id=f"f-{t}", evidence_type=t)]))
        assert rec.findings[0]["evidence_type"] == t


def test_duplicate_finding_ids_rejected():
    with pytest.raises(UiAttentionError) as exc:
        ReviewRecord.from_dict(_review(findings=[_finding(), _finding()]))
    assert "重复" in str(exc.value.details)


def test_missing_required_finding_fields():
    for key in ("id", "region_ids", "evidence_refs", "observation", "inference", "recommendation", "validation_needed"):
        f = _finding()
        del f[key]
        with pytest.raises(UiAttentionError):
            ReviewRecord.from_dict(_review(findings=[f]))


def test_unknown_fields_rejected():
    with pytest.raises(UiAttentionError):
        ReviewRecord.from_dict(_review(confidence_percent=97))
    with pytest.raises(UiAttentionError):
        ReviewRecord.from_dict(_review(findings=[_finding(click_rate=0.8)]))


def test_evidence_refs_must_be_str_list():
    with pytest.raises(UiAttentionError):
        ReviewRecord.from_dict(_review(findings=[_finding(evidence_refs="analysis.json")]))
    with pytest.raises(UiAttentionError):
        ReviewRecord.from_dict(_review(findings=[_finding(region_ids="claim-button")]))


def test_load_review_from_disk(tmp_path):
    import json

    p = tmp_path / "review.json"
    p.write_text(json.dumps(_review(), ensure_ascii=False), encoding="utf-8")
    rec = load_review(p)
    assert len(rec.findings) == 1
    with pytest.raises(UiAttentionError):
        load_review(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(UiAttentionError):
        load_review(bad)
