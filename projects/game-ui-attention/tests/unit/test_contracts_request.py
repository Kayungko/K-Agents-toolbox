"""contracts.request / contracts.regions 校验测试（validation-plan §1“区域校验”行的 schema 侧）。

覆盖：合法请求解析、未知 schema_version 拒绝、重复 ID、零面积、越界、自交多边形、
共线多边形、枚举校验、未知字段、相对路径解析、独立 regions 文件 SHA-256 强制。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fixtures import synthetic  # noqa: E402
from ui_attention.contracts import (  # noqa: E402
    AnalyzeRequest,
    RegionsFile,
    load_analyze_request,
    load_regions_file,
)
from ui_attention.errors import ErrorCode, UiAttentionError  # noqa: E402


def _parse(req_dict, base_dir=None):
    return AnalyzeRequest.from_dict(req_dict, base_dir=base_dir)


def test_valid_request_minimal_and_full():
    req = _parse(synthetic.base_request())
    assert req.backend_profile == "foveacast-onnx-3s-v1"
    assert req.regions == ()  # regions 可缺省
    assert req.player_goal == "查看奖励并领取"

    req2 = _parse(synthetic.base_request(regions=[synthetic.VALID_RECT_REGION, synthetic.VALID_POLYGON_REGION]))
    assert len(req2.regions) == 2
    assert req2.regions[0].geometry.type == "rect"
    assert req2.regions[1].geometry.points == ((2, 2), (30, 4), (28, 20), (4, 18))


@pytest.mark.parametrize("version", [None, "v1", "game-ui-attention-request/v2", "GAME-UI-ATTENTION-REQUEST/V1"])
def test_unknown_schema_version_rejected(version):
    with pytest.raises(UiAttentionError) as exc:
        _parse(synthetic.base_request(schema_version=version))
    assert exc.value.code is ErrorCode.INVALID_SCHEMA_VERSION
    assert exc.value.exit_code == 2


def test_duplicate_region_ids_rejected():
    regions = [synthetic.VALID_RECT_REGION, dict(synthetic.VALID_RECT_REGION)]
    with pytest.raises(UiAttentionError) as exc:
        _parse(synthetic.base_request(regions=regions))
    assert exc.value.code is ErrorCode.INVALID_AOI
    assert exc.value.details["duplicate_ids"] == ["claim-button"]


def test_zero_area_rect_rejected():
    with pytest.raises(UiAttentionError) as exc:
        _parse(synthetic.base_request(regions=[synthetic.ZERO_AREA_RECT]))
    assert exc.value.code is ErrorCode.INVALID_AOI
    assert any("零面积" in e or "width" in e for e in exc.value.details.get("errors", [exc.value.message]))


def test_out_of_bounds_detected_against_image():
    req = _parse(synthetic.base_request(regions=[synthetic.OUT_OF_BOUNDS_RECT]))
    with pytest.raises(UiAttentionError) as exc:
        req.check_bounds(image_width=64, image_height=48)
    assert exc.value.code is ErrorCode.INVALID_AOI
    # 足够大的画布下同一区域合法
    req.check_bounds(image_width=128, image_height=64)


def test_self_intersecting_polygon_rejected():
    with pytest.raises(UiAttentionError) as exc:
        _parse(synthetic.base_request(regions=[synthetic.SELF_INTERSECTING_POLYGON]))
    assert exc.value.code is ErrorCode.INVALID_AOI
    assert any("自交" in e for e in exc.value.details["errors"])


def test_collinear_polygon_rejected():
    with pytest.raises(UiAttentionError) as exc:
        _parse(synthetic.base_request(regions=[synthetic.COLLINEAR_POLYGON]))
    assert exc.value.code is ErrorCode.INVALID_AOI
    assert any("共线" in e for e in exc.value.details["errors"])


def test_polygon_needs_three_points():
    region = {
        "id": "p2",
        "geometry": {"type": "polygon", "points": [[0, 0], [1, 1]]},
        "source": "manual",
        "status": "candidate",
    }
    with pytest.raises(UiAttentionError):
        _parse(synthetic.base_request(regions=[region]))


def test_polygon_out_of_bounds_rejected():
    region = {
        "id": "poly-oob",
        "geometry": {"type": "polygon", "points": [[0, 0], [10, 0], [10, 100]]},
        "source": "manual",
        "status": "candidate",
    }
    req = _parse(synthetic.base_request(regions=[region]))
    with pytest.raises(UiAttentionError) as exc:
        req.check_bounds(64, 48)
    assert exc.value.code is ErrorCode.INVALID_AOI


@pytest.mark.parametrize("source", ["auto", "MANUAL", ""])
def test_bad_source_enum_rejected(source):
    region = dict(synthetic.VALID_RECT_REGION, source=source)
    with pytest.raises(UiAttentionError):
        _parse(synthetic.base_request(regions=[region]))


@pytest.mark.parametrize("status", ["approved", "CONFIRMED", None])
def test_bad_status_enum_rejected(status):
    region = dict(synthetic.VALID_RECT_REGION, status=status)
    with pytest.raises(UiAttentionError):
        _parse(synthetic.base_request(regions=[region]))


def test_unknown_keys_rejected():
    with pytest.raises(UiAttentionError) as exc:
        _parse(synthetic.base_request(extra_field=1))
    assert exc.value.code is ErrorCode.INVALID_REQUEST

    region = dict(synthetic.VALID_RECT_REGION, click_heatzone=True)
    with pytest.raises(UiAttentionError):
        _parse(synthetic.base_request(regions=[region]))


def test_imported_region_requires_source_reference():
    region = dict(synthetic.VALID_RECT_REGION, source="imported", source_reference=None)
    with pytest.raises(UiAttentionError):
        _parse(synthetic.base_request(regions=[region]))
    region_ok = dict(
        synthetic.VALID_RECT_REGION,
        source="imported",
        source_reference={"node": "design://panel/btn", "transform": {"scale": 0.5, "offset": [10, 20]}},
    )
    req = _parse(synthetic.base_request(regions=[region_ok]))
    assert req.regions[0].source_reference["node"] == "design://panel/btn"


def test_missing_required_fields():
    with pytest.raises(UiAttentionError):
        _parse({"schema_version": "game-ui-attention-request/v1"})  # 无 image/backend_profile
    with pytest.raises(UiAttentionError):
        _parse("not-a-dict")


def test_float_coordinates_rejected():
    region = dict(
        synthetic.VALID_RECT_REGION,
        geometry={"type": "rect", "x": 1.5, "y": 2, "width": 4, "height": 4},
    )
    with pytest.raises(UiAttentionError):
        _parse(synthetic.base_request(regions=[region]))


def test_relative_image_path_resolves_against_request_dir(tmp_path):
    req_dir = tmp_path / "reqs"
    req_dir.mkdir()
    req_path = synthetic.write_json(req_dir / "request.json", synthetic.base_request(image="./shots/input.png"))
    req = load_analyze_request(req_path)
    assert req.image_path == req_dir / "shots" / "input.png"
    assert req.base_dir == req_dir


def test_load_request_file_errors(tmp_path):
    with pytest.raises(UiAttentionError) as exc:
        load_analyze_request(tmp_path / "missing.json")
    assert exc.value.code is ErrorCode.INVALID_REQUEST
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(UiAttentionError):
        load_analyze_request(bad)


# ---------------------------------------------------------------------------
# 独立 regions 文件（必须携带图片 SHA-256）
# ---------------------------------------------------------------------------

SHA = "a" * 64


def _regions_file_obj(sha=SHA, regions=None):
    return {
        "schema_version": "game-ui-attention-regions/v1",
        "image_sha256": sha,
        "regions": regions if regions is not None else [synthetic.VALID_RECT_REGION],
    }


def test_regions_file_requires_image_sha256():
    obj = _regions_file_obj()
    del obj["image_sha256"]
    with pytest.raises(UiAttentionError) as exc:
        RegionsFile.from_dict(obj)
    assert exc.value.code is ErrorCode.INVALID_REQUEST

    obj2 = _regions_file_obj(sha="tooshort")
    with pytest.raises(UiAttentionError):
        RegionsFile.from_dict(obj2)


def test_regions_file_sha_mismatch_rejected():
    rf = RegionsFile.from_dict(_regions_file_obj())
    rf.check_image_sha256(SHA)  # 一致 → 通过
    with pytest.raises(UiAttentionError) as exc:
        rf.check_image_sha256("b" * 64)
    assert exc.value.code is ErrorCode.REGIONS_IMAGE_MISMATCH
    assert exc.value.exit_code == 2


def test_regions_file_unknown_schema_and_dup_ids():
    with pytest.raises(UiAttentionError):
        RegionsFile.from_dict(_regions_file_obj() | {"schema_version": "regions/v9"})
    with pytest.raises(UiAttentionError):
        RegionsFile.from_dict(
            _regions_file_obj(regions=[synthetic.VALID_RECT_REGION, dict(synthetic.VALID_RECT_REGION)])
        )


def test_regions_file_roundtrip_via_disk(tmp_path):
    p = synthetic.write_json(tmp_path / "regions.json", _regions_file_obj())
    rf = load_regions_file(p)
    assert rf.image_sha256 == SHA
    assert rf.regions[0].id == "claim-button"
    assert rf.to_dict()["regions"][0]["geometry"] == synthetic.VALID_RECT_REGION["geometry"]
