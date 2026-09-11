"""contracts.analysis（analysis v1，data-contract §4）校验测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fixtures import synthetic  # noqa: E402
from ui_attention.contracts import (  # noqa: E402
    ANALYSIS_SCHEMA_VERSION,
    EVIDENCE_TYPE,
    AnalysisRecord,
    ArtifactRecord,
    load_analysis,
)
from ui_attention.errors import ErrorCode, UiAttentionError  # noqa: E402


def test_valid_analysis_roundtrip():
    obj = synthetic.sample_analysis_dict()
    rec = AnalysisRecord.from_dict(obj)
    assert rec.schema_version == ANALYSIS_SCHEMA_VERSION
    assert rec.evidence_type == EVIDENCE_TYPE
    assert rec.input["image_sha256"] == synthetic.IMAGE_SHA
    back = rec.to_dict()
    assert back["regions"][0]["id"] == "claim-button"
    # 浮点原始值保留（展示舍入不影响计算）
    assert back["regions"][0]["probability_mass"] == 0.25
    AnalysisRecord.from_dict(back).validate()


def test_unknown_schema_version_rejected():
    obj = synthetic.sample_analysis_dict(schema_version="game-ui-attention-analysis/v2")
    with pytest.raises(UiAttentionError) as exc:
        AnalysisRecord.from_dict(obj)
    assert exc.value.code is ErrorCode.INVALID_SCHEMA_VERSION


def test_unknown_top_level_field_rejected():
    obj = synthetic.sample_analysis_dict()
    obj["click_rate"] = 0.9
    with pytest.raises(UiAttentionError):
        AnalysisRecord.from_dict(obj)


@pytest.mark.parametrize("status", ["done", "COMPLETE", None])
def test_bad_computation_status(status):
    obj = synthetic.sample_analysis_dict(computation_status=status)
    with pytest.raises(UiAttentionError):
        AnalysisRecord.from_dict(obj)


@pytest.mark.parametrize("review", ["not_requested", "pending", "complete", "failed"])
def test_review_status_enum(review):
    obj = synthetic.sample_analysis_dict(review_status=review)
    assert AnalysisRecord.from_dict(obj).review_status == review


def test_bad_evidence_type_rejected():
    obj = synthetic.sample_analysis_dict(evidence_type="real_eye_tracking")
    with pytest.raises(UiAttentionError):
        AnalysisRecord.from_dict(obj)


def test_input_field_validation():
    obj = synthetic.sample_analysis_dict()
    obj["input"]["image_sha256"] = "short"
    with pytest.raises(UiAttentionError):
        AnalysisRecord.from_dict(obj)
    obj2 = synthetic.sample_analysis_dict()
    obj2["input"]["width"] = 0
    with pytest.raises(UiAttentionError):
        AnalysisRecord.from_dict(obj2)


def test_model_requires_all_weight_hashes():
    obj = synthetic.sample_analysis_dict()
    obj["model"]["weights_sha256"] = ["abc"]
    with pytest.raises(UiAttentionError):
        AnalysisRecord.from_dict(obj)


def test_profile_requires_config_hash():
    obj = synthetic.sample_analysis_dict()
    obj["profile"]["config_hash"] = ""
    with pytest.raises(UiAttentionError):
        AnalysisRecord.from_dict(obj)


def test_region_row_validation():
    bad = synthetic.sample_region_result()
    bad["probability_mass"] = 1.5  # 超出 [0,1]
    obj = synthetic.sample_analysis_dict(regions=[bad])
    with pytest.raises(UiAttentionError):
        AnalysisRecord.from_dict(obj)

    zero_area = synthetic.sample_region_result()
    zero_area["area_px"] = 0  # 零面积拒绝
    obj2 = synthetic.sample_analysis_dict(regions=[zero_area])
    with pytest.raises(UiAttentionError):
        AnalysisRecord.from_dict(obj2)


def test_artifact_record_validation():
    errors: list[str] = []
    ArtifactRecord.from_dict({"path": "", "sha256": "x"}, errors, "artifacts[0]")
    assert errors
    ok = ArtifactRecord.from_dict({"path": "density.npy", "sha256": "d" * 64}, [], "artifacts[0]")
    assert ok.path == "density.npy"


def test_load_analysis_from_disk(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    synthetic.write_json(run / "analysis.json", synthetic.sample_analysis_dict())
    rec = load_analysis(run)
    assert rec.analysis_id == "analysis-test-0001"

    with pytest.raises(UiAttentionError) as exc:
        load_analysis(tmp_path / "no-such-run")
    assert exc.value.code is ErrorCode.ARTIFACT_CHECK_FAILED
