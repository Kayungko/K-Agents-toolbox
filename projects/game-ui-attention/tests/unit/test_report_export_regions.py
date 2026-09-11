"""C3：regions 导出/校验单测（export_regions.py）。

覆盖任务验收项：regions 导出往返一致（export→校验函数→结构不变）、
SHA-256 绑定拒绝错图、越界/零面积/自交/重复 ID 拒绝，以及
validation-plan §1"标注变更"行的报告侧支撑（重新导出只携带标注与
图片绑定，不触碰概率图/统计字段——密度哈希保持不变由 C1 summarize 侧
验证，本侧验证 regions 文件不含任何统计载荷）。

校验口径与 C1 冻结契约一致（contracts/regions.py、aoi/geometry.py），
并用 C1 契约直接复核"通过我校验的 payload 必被 C1 接受"。
"""

from __future__ import annotations

import json

import pytest

from ui_attention.contracts.regions import RegionsFile
from ui_attention.report.export_regions import (
    REGIONS_SCHEMA_VERSION,
    compute_image_sha256,
    export_regions_file,
    load_and_validate_regions_file,
    make_regions_payload,
    normalize_region_for_export,
    validate_regions_payload,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
IMG_SIZE = (64, 48)  # (width, height)


def rect_region(rid: str = "claim-button", **over) -> dict:
    base = {
        "id": rid,
        "label": "领取按钮",
        "role": "primary-action",
        "geometry": {"type": "rect", "x": 10, "y": 8, "width": 20, "height": 6},
        "source": "manual",
        "status": "confirmed",
    }
    base.update(over)
    return base


def poly_region(rid: str = "hero-art", points=None, **over) -> dict:
    base = {
        "id": rid,
        "label": "立绘",
        "role": "decoration",
        "geometry": {
            "type": "polygon",
            "points": points if points is not None else [[4, 4], [30, 6], [18, 40]],
        },
        "source": "agent",
        "status": "candidate",
    }
    base.update(over)
    return base


def codes(result) -> set[str]:
    return set(result.error_codes())


class TestExportRoundtrip:
    def test_roundtrip_structure_unchanged(self, tmp_path):
        regions = [rect_region(), poly_region()]
        out = export_regions_file(regions=regions, image_sha256=SHA_A, out_path=tmp_path / "regions.json")
        payload = json.loads(out.read_text(encoding="utf-8"))
        result = validate_regions_payload(payload, expected_image_sha256=SHA_A, image_size=IMG_SIZE)
        assert result.ok, result.errors
        # 结构不变：regions 逐项等于规范化后的输入（normalize 对齐 RegionSpec.to_dict）
        assert payload["regions"] == [normalize_region_for_export(r) for r in regions]
        assert payload["schema_version"] == REGIONS_SCHEMA_VERSION
        assert payload["image_sha256"] == SHA_A
        assert set(payload) == {"schema_version", "image_sha256", "regions"}

    def test_roundtrip_passes_c1_contract(self, tmp_path):
        """通过我校验的 payload 必被 C1 冻结契约接受（权威一致性）。"""
        regions = [rect_region(), poly_region()]
        out = export_regions_file(regions=regions, image_sha256=SHA_A, out_path=tmp_path / "r.json")
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert validate_regions_payload(payload, expected_image_sha256=SHA_A, image_size=IMG_SIZE).ok
        rf = RegionsFile.from_dict(payload)  # 不抛异常即通过
        rf.check_bounds(*IMG_SIZE)
        rf.check_image_sha256(SHA_A)
        assert len(rf.regions) == 2

    def test_export_deterministic_bytes(self, tmp_path):
        regions = [rect_region(), poly_region()]
        p1 = export_regions_file(regions=regions, image_sha256=SHA_A, out_path=tmp_path / "r1.json")
        p2 = export_regions_file(regions=regions, image_sha256=SHA_A, out_path=tmp_path / "r2.json")
        assert p1.read_bytes() == p2.read_bytes()
        assert p1.read_bytes().endswith(b"\n")
        assert b"\r\n" not in p1.read_bytes()  # LF 换行

    def test_normalize_keeps_source_reference_and_nulls(self):
        region = rect_region(source="imported", source_reference={"node": "btn_claim", "transform": {"scale": 2}})
        region["label"] = ""
        region.pop("role")
        norm = normalize_region_for_export(region)
        assert norm["label"] is None  # 空串 → null（RegionSpec str|None 口径）
        assert norm["role"] is None
        assert norm["source_reference"] == {"node": "btn_claim", "transform": {"scale": 2}}
        assert list(norm) == ["id", "label", "role", "geometry", "source", "status", "source_reference"]


class TestSha256Binding:
    def test_wrong_image_rejected(self, tmp_path):
        """SHA-256 绑定拒绝错图：regions 文件不得应用到其他图片。"""
        payload = make_regions_payload([rect_region()], SHA_A)
        result = validate_regions_payload(payload, expected_image_sha256=SHA_B, image_size=IMG_SIZE)
        assert not result.ok
        assert "IMAGE_SHA256_MISMATCH" in codes(result)

    def test_matching_image_accepted(self):
        payload = make_regions_payload([rect_region()], SHA_A)
        assert validate_regions_payload(payload, expected_image_sha256=SHA_A, image_size=IMG_SIZE).ok

    def test_missing_sha_rejected(self):
        payload = {"schema_version": REGIONS_SCHEMA_VERSION, "regions": []}
        result = validate_regions_payload(payload)
        assert not result.ok
        assert "MISSING_IMAGE_SHA256" in codes(result)
        payload_null = {"schema_version": REGIONS_SCHEMA_VERSION, "image_sha256": None, "regions": []}
        assert "MISSING_IMAGE_SHA256" in codes(validate_regions_payload(payload_null))

    @pytest.mark.parametrize("bad", ["A" * 64, "a" * 63, "a" * 65, "g" * 64, 123, ""])
    def test_bad_sha_format_rejected(self, bad):
        payload = {"schema_version": REGIONS_SCHEMA_VERSION, "image_sha256": bad, "regions": []}
        result = validate_regions_payload(payload)
        assert not result.ok
        assert "BAD_IMAGE_SHA256" in codes(result)

    def test_make_payload_rejects_bad_sha(self):
        with pytest.raises(ValueError, match="SHA-256"):
            make_regions_payload([], "not-a-sha")

    def test_compute_image_sha256_matches_hashlib(self, tmp_path):
        p = tmp_path / "img.bin"
        p.write_bytes(b"synthetic-png-bytes")
        import hashlib

        assert compute_image_sha256(p) == hashlib.sha256(b"synthetic-png-bytes").hexdigest()


class TestRejections:
    def test_unknown_schema_version(self):
        payload = make_regions_payload([rect_region()], SHA_A)
        payload["schema_version"] = "game-ui-attention-regions/v2"
        result = validate_regions_payload(payload)
        assert not result.ok and "UNKNOWN_SCHEMA_VERSION" in codes(result)

    def test_duplicate_ids(self):
        payload = make_regions_payload([rect_region("a"), rect_region("a")], SHA_A)
        result = validate_regions_payload(payload, image_size=IMG_SIZE)
        assert "DUPLICATE_REGION_ID" in codes(result)

    @pytest.mark.parametrize("rid", ["", "   ", None, 7])
    def test_bad_id(self, rid):
        payload = make_regions_payload([rect_region()], SHA_A)
        payload["regions"][0]["id"] = rid
        assert "BAD_REGION_ID" in codes(validate_regions_payload(payload, image_size=IMG_SIZE))

    def test_rect_zero_area(self):
        for geom in (
            {"type": "rect", "x": 1, "y": 1, "width": 0, "height": 5},
            {"type": "rect", "x": 1, "y": 1, "width": 5, "height": 0},
            {"type": "rect", "x": 1, "y": 1, "width": -3, "height": 5},
        ):
            payload = make_regions_payload([rect_region(geometry=geom)], SHA_A)
            result = validate_regions_payload(payload, image_size=IMG_SIZE)
            assert not result.ok
            assert "ZERO_AREA_REGION" in codes(result), geom

    @pytest.mark.parametrize(
        "geom",
        [
            {"type": "rect", "x": 60, "y": 0, "width": 10, "height": 5},  # x+w > W
            {"type": "rect", "x": 0, "y": 45, "width": 5, "height": 10},  # y+h > H
            {"type": "rect", "x": -1, "y": 0, "width": 5, "height": 5},
        ],
    )
    def test_rect_out_of_bounds(self, geom):
        payload = make_regions_payload([rect_region(geometry=geom)], SHA_A)
        result = validate_regions_payload(payload, image_size=IMG_SIZE)
        assert "REGION_OUT_OF_BOUNDS" in codes(result)

    def test_rect_non_integer_coords(self):
        for geom in (
            {"type": "rect", "x": 1.5, "y": 0, "width": 5, "height": 5},
            {"type": "rect", "x": True, "y": 0, "width": 5, "height": 5},
        ):
            payload = make_regions_payload([rect_region(geometry=geom)], SHA_A)
            assert "NON_INTEGER_COORD" in codes(validate_regions_payload(payload, image_size=IMG_SIZE))

    def test_polygon_needs_three_points(self):
        payload = make_regions_payload([poly_region(points=[[1, 1], [5, 5]])], SHA_A)
        assert "BAD_GEOMETRY" in codes(validate_regions_payload(payload, image_size=IMG_SIZE))

    def test_polygon_collinear_zero_area(self):
        payload = make_regions_payload([poly_region(points=[[2, 2], [4, 4], [8, 8]])], SHA_A)
        result = validate_regions_payload(payload, image_size=IMG_SIZE)
        assert "ZERO_AREA_REGION" in codes(result)

    def test_polygon_self_intersecting_bowtie(self):
        bowtie = [[2, 2], [20, 20], [20, 2], [2, 20]]
        payload = make_regions_payload([poly_region(points=bowtie)], SHA_A)
        result = validate_regions_payload(payload, image_size=IMG_SIZE)
        assert "SELF_INTERSECTING_POLYGON" in codes(result)

    def test_polygon_duplicate_vertices_rejected(self):
        # C1 口径：重复顶点拒绝，含首尾闭合写法（不做自动去闭合）
        closed = [[4, 4], [30, 6], [18, 40], [4, 4]]
        payload = make_regions_payload([poly_region(points=closed)], SHA_A)
        result = validate_regions_payload(payload, image_size=IMG_SIZE)
        assert "BAD_GEOMETRY" in codes(result)

    def test_polygon_vertex_strictly_inside_pixels(self):
        # polygon_in_bounds 口径：0 <= x < W、0 <= y < H；x == W 越界
        edge = [[64, 4], [60, 40], [30, 6]]
        payload = make_regions_payload([poly_region(points=edge)], SHA_A)
        result = validate_regions_payload(payload, image_size=IMG_SIZE)
        assert "REGION_OUT_OF_BOUNDS" in codes(result)

    def test_valid_concave_polygon_accepted(self):
        concave = [[5, 5], [40, 5], [40, 40], [22, 25], [5, 40]]
        payload = make_regions_payload([poly_region(points=concave)], SHA_A)
        result = validate_regions_payload(payload, image_size=IMG_SIZE)
        assert result.ok, result.errors

    @pytest.mark.parametrize(
        ("source", "status"),
        [("auto", "confirmed"), ("manual", "approved"), ("", "candidate"), (None, "confirmed")],
    )
    def test_enum_rejections(self, source, status):
        payload = make_regions_payload([rect_region(source=source, status=status)], SHA_A)
        result = validate_regions_payload(payload, image_size=IMG_SIZE)
        assert not result.ok
        assert codes(result) & {"BAD_REGION_SOURCE", "BAD_REGION_STATUS"}

    def test_imported_requires_source_reference(self):
        payload = make_regions_payload([rect_region(source="imported")], SHA_A)
        result = validate_regions_payload(payload, image_size=IMG_SIZE)
        assert "BAD_REGION_FIELD" in codes(result)
        with_ref = make_regions_payload(
            [rect_region(source="imported", source_reference={"node": "n1", "transform": {"dx": 0}})], SHA_A
        )
        assert validate_regions_payload(with_ref, image_size=IMG_SIZE).ok

    def test_label_role_null_allowed(self):
        payload = make_regions_payload([rect_region(label=None, role=None)], SHA_A)
        assert validate_regions_payload(payload, image_size=IMG_SIZE).ok

    def test_unknown_region_key_rejected(self):
        payload = make_regions_payload([rect_region()], SHA_A)
        payload["regions"][0]["score"] = 0.9  # 未知字段（如统计残留）拒绝
        result = validate_regions_payload(payload, image_size=IMG_SIZE)
        assert "BAD_REGION_FIELD" in codes(result)

    def test_regions_not_list(self):
        payload = make_regions_payload([], SHA_A)
        payload["regions"] = {"id": "x"}
        assert "BAD_REGIONS" in codes(validate_regions_payload(payload))

    def test_payload_not_dict(self):
        assert "BAD_REGIONS" in codes(validate_regions_payload([1, 2, 3]))


class TestFileLoading:
    def test_load_and_validate_ok(self, tmp_path):
        out = export_regions_file(regions=[rect_region()], image_sha256=SHA_A, out_path=tmp_path / "r.json")
        payload, result = load_and_validate_regions_file(
            out, expected_image_sha256=SHA_A, image_size=IMG_SIZE
        )
        assert result.ok and payload is not None

    def test_load_missing_file_structured_error(self, tmp_path):
        payload, result = load_and_validate_regions_file(tmp_path / "nope.json")
        assert payload is None and not result.ok
        assert "BAD_REGIONS" in codes(result)

    def test_load_invalid_json_structured_error(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        payload, result = load_and_validate_regions_file(p)
        assert payload is None and not result.ok


class TestAnnotationChangeRow:
    """validation-plan §1"标注变更"行的报告侧支撑。

    模拟 HTML 圈选编辑后的重新导出：regions 文件只携带标注 + 图片绑定，
    image_sha256 不变（同一底层图片），不含任何统计/密度载荷——因此
    summarize 只需重算统计、概率图哈希保持不变（该断言链的统计侧由 C1
    test_metrics_* 覆盖）。
    """

    def test_reexport_preserves_binding_and_carries_no_stats(self, tmp_path):
        original = [rect_region(), poly_region()]
        out1 = export_regions_file(regions=original, image_sha256=SHA_A, out_path=tmp_path / "v1.json")
        payload1 = json.loads(out1.read_text(encoding="utf-8"))

        # HTML 编辑结果：修改 label/status + 新增一个圈选区域，重新导出
        edited = json.loads(json.dumps(payload1["regions"]))
        edited[0]["label"] = "领取按钮（改）"
        edited[1]["status"] = "confirmed"  # 候选边界经人工确认
        edited.append(
            normalize_region_for_export(
                {
                    "id": "price-tag",
                    "label": "价格",
                    "role": "info",
                    "geometry": {"type": "rect", "x": 30, "y": 20, "width": 12, "height": 8},
                    "source": "manual",
                    "status": "confirmed",
                }
            )
        )
        out2 = export_regions_file(regions=edited, image_sha256=SHA_A, out_path=tmp_path / "v2.json")
        payload2, result = load_and_validate_regions_file(
            out2, expected_image_sha256=SHA_A, image_size=IMG_SIZE
        )
        assert result.ok, result.errors
        # 图片绑定不变（同一图片，仅标注变更）
        assert payload2["image_sha256"] == payload1["image_sha256"] == SHA_A
        # 不携带任何统计/密度字段（每条区域键集合封闭）
        allowed = {"id", "label", "role", "geometry", "source", "status", "source_reference"}
        for region in payload2["regions"]:
            assert set(region) <= allowed
        assert set(payload2) == {"schema_version", "image_sha256", "regions"}
        # C1 权威契约同样接受
        RegionsFile.from_dict(payload2)
