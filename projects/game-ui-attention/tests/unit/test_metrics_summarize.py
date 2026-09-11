"""summarize 重算测试（validation-plan §1“标注变更”行：仅重算统计，保留相同底层概率图哈希，不重新推理）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fixtures import synthetic  # noqa: E402
from ui_attention import cli, imaging  # noqa: E402
from ui_attention.contracts.analysis import load_analysis  # noqa: E402

BLOCK = (10, 8, 32, 16)
MASS = 0.3
SHAPE = (48, 64)


def _build_source_run(tmp_path: Path, *, record_density_sha: bool = True, tamper_density: bool = False) -> Path:
    """构造既有运行目录：已知局部质量的 density.npy + 对应 analysis.json。"""
    run = tmp_path / "run-src"
    run.mkdir()
    density = synthetic.block_probability(SHAPE, BLOCK, mass=MASS)
    if tamper_density:
        density = density * 0.5  # sum != 1 → 无效概率图
    np.save(run / "density.npy", density)
    sha = imaging.sha256_file(run / "density.npy")
    analysis = synthetic.sample_analysis_dict(
        artifacts=[{"path": "density.npy", "sha256": sha if record_density_sha else "f" * 64, "size_bytes": 100}],
    )
    synthetic.write_json(run / "analysis.json", analysis)
    return run


def _regions_file(tmp_path: Path, *, sha: str = synthetic.IMAGE_SHA, regions=None) -> Path:
    obj = {
        "schema_version": "game-ui-attention-regions/v1",
        "image_sha256": sha,
        "regions": regions if regions is not None else [synthetic.VALID_RECT_REGION],
    }
    return synthetic.write_json(tmp_path / "regions.json", obj)


def _run(capsys, argv):
    rc = cli.main(argv)
    return rc, json.loads(capsys.readouterr().out)


def test_summarize_recomputes_without_reinference(monkeypatch, capsys, tmp_path):
    """标注变更 → 仅重算统计：相同底层概率图哈希、registry 全程未被导入。"""
    sys.modules.pop(cli.REGISTRY_MODULE, None)
    # 若 summarize 尝试重新推理（导入 registry），让导入立即失败以证明未发生
    monkeypatch.setitem(sys.modules, "ui_attention.backends", None)
    src = _build_source_run(tmp_path)
    src_density_sha = imaging.sha256_file(src / "density.npy")
    regions = _regions_file(tmp_path)
    out = tmp_path / "runs" / "run-sum"

    rc, env = _run(capsys, ["summarize", "--analysis", str(src), "--regions", str(regions), "--out", str(out)])
    assert rc == 0 and env["ok"] is True
    assert cli.REGISTRY_MODULE not in sys.modules  # 未重新推理（未触碰后端）

    # 相同底层概率图哈希
    assert env["result"]["density_sha256"] == src_density_sha
    assert imaging.sha256_file(out / "density.npy") == src_density_sha

    analysis = load_analysis(out)
    row = analysis.regions[0]
    # 已知局部质量：区域与质量块重合 → mass == 0.3（解析值）
    assert row["probability_mass"] == pytest.approx(MASS, abs=1e-12)
    assert row["area_fraction"] == pytest.approx(512 / (48 * 64))
    assert row["relative_density"] == pytest.approx(MASS / (512 / (48 * 64)))
    assert any("未重新推理" in x for x in analysis.limitations)
    assert analysis.analysis_id != "analysis-test-0001"  # 新 analysis_id
    assert (out / "manifest.json").is_file()


def test_summarize_new_regions_change_statistics(monkeypatch, capsys, tmp_path):
    """更换标注（质量块外区域）→ 统计随之变化，概率图不变。"""
    src = _build_source_run(tmp_path)
    outside_region = {
        "id": "corner",
        "label": "角落",
        "role": "decoration",
        "geometry": {"type": "rect", "x": 0, "y": 40, "width": 8, "height": 8},
        "source": "manual",
        "status": "confirmed",
    }
    regions = _regions_file(tmp_path, regions=[outside_region])
    out = tmp_path / "runs" / "run-sum2"
    rc, env = _run(capsys, ["summarize", "--analysis", str(src), "--regions", str(regions), "--out", str(out)])
    assert rc == 0
    analysis = load_analysis(out)
    rest_total = 48 * 64 - 512
    expected = (1 - MASS) * 64 / rest_total
    assert analysis.regions[0]["probability_mass"] == pytest.approx(expected, abs=1e-12)


def test_summarize_sha_mismatch_exit2(capsys, tmp_path):
    """regions 文件图片 SHA-256 不符 → 拒绝（防止标注应用到其他图片）。"""
    src = _build_source_run(tmp_path)
    regions = _regions_file(tmp_path, sha="9" * 64)
    out = tmp_path / "runs" / "run-bad-sha"
    rc, env = _run(capsys, ["summarize", "--analysis", str(src), "--regions", str(regions), "--out", str(out)])
    assert rc == 2
    assert env["error"]["id"] == "REGIONS_IMAGE_MISMATCH"
    assert not out.exists()


def test_summarize_tampered_density_exit5(capsys, tmp_path):
    """density.npy 与 analysis 记录哈希不符 → ARTIFACT_CHECK_FAILED（退出码 5）。"""
    src = _build_source_run(tmp_path)
    (src / "density.npy").write_bytes(b"tampered-bytes-not-npy")  # 改动产物
    regions = _regions_file(tmp_path)
    out = tmp_path / "runs" / "run-tampered"
    rc, env = _run(capsys, ["summarize", "--analysis", str(src), "--regions", str(regions), "--out", str(out)])
    assert rc == 5
    assert env["error"]["id"] == "ARTIFACT_CHECK_FAILED"


def test_summarize_invalid_density_values_exit5(capsys, tmp_path):
    """density 数值无效（sum≠1）→ INVALID_DENSITY，不产生假成功。"""
    src = _build_source_run(tmp_path, record_density_sha=False, tamper_density=True)
    # 记录哈希按“篡改后”文件写回，绕过产物哈希门，专测数值门
    sha = imaging.sha256_file(src / "density.npy")
    analysis = json.loads((src / "analysis.json").read_text(encoding="utf-8"))
    analysis["artifacts"][0]["sha256"] = sha
    synthetic.write_json(src / "analysis.json", analysis)
    regions = _regions_file(tmp_path)
    out = tmp_path / "runs" / "run-invalid"
    rc, env = _run(capsys, ["summarize", "--analysis", str(src), "--regions", str(regions), "--out", str(out)])
    assert rc == 5
    assert env["error"]["id"] == "INVALID_DENSITY"
    assert not out.exists()


def test_summarize_missing_density_exit5(capsys, tmp_path):
    src = _build_source_run(tmp_path)
    (src / "density.npy").unlink()
    regions = _regions_file(tmp_path)
    out = tmp_path / "runs" / "run-nodensity"
    rc, env = _run(capsys, ["summarize", "--analysis", str(src), "--regions", str(regions), "--out", str(out)])
    assert rc == 5
    assert env["error"]["id"] == "ARTIFACT_CHECK_FAILED"


def test_summarize_out_exists_exit7(capsys, tmp_path):
    src = _build_source_run(tmp_path)
    regions = _regions_file(tmp_path)
    out = tmp_path / "runs" / "run-exists"
    out.mkdir(parents=True)
    rc, env = _run(capsys, ["summarize", "--analysis", str(src), "--regions", str(regions), "--out", str(out)])
    assert rc == 7
    assert env["error"]["id"] == "OUTPUT_PATH_EXISTS"


# ---------------------------------------------------------------------------
# C3 接线（总控接线点①④）：兜底校验、base.png/review.json 复用与渲染
# ---------------------------------------------------------------------------

import types  # noqa: E402

VALID_REVIEW = {
    "schema_version": "game-ui-attention-review/v1",
    "analysis_id": "analysis-test-0001",
    "findings": [
        {
            "id": "f1",
            "region_ids": ["claim-button"],
            "evidence_refs": ["analysis.json#/regions/0"],
            "evidence_type": "computed",
            "observation": "领取按钮 relative_density 高于 1",
            "inference": "模型预测其获得高于平均的注视质量",
            "recommendation": "保持当前层级",
            "validation_needed": "需真人评审确认（模型预测非真实眼动）",
        }
    ],
}


def _install_fake_report(monkeypatch, *, regions_ok: bool = True):
    module = types.ModuleType(cli.REPORT_MODULE)
    module.DEFAULT_OVERLAY_ALPHA = 0.55

    class FakeColorScale:
        @classmethod
        def shared(cls, name, da, db, normalization="linear"):
            return cls()

        def to_params(self):
            return {"name": "ember", "vmin": 0.0, "vmax": 1.0, "normalization": "linear"}

    module.ColorScale = FakeColorScale
    module.overlay_record = lambda cs, alpha: {"alpha": alpha, "colorscale": cs.to_params()}

    def render_run_report(
        *,
        out_dir,
        base_image,
        density,
        analysis,
        regions_payload,
        colorscale,
        alpha,
        review=None,
        base_png_path=None,
        title=None,
    ):
        out = Path(out_dir)
        (out / "overlay.png").write_bytes(b"fake")
        (out / "heatmap.png").write_bytes(b"fake")
        (out / "report.html").write_text("<html></html>", encoding="utf-8")
        return {"overlay": out / "overlay.png", "heatmap": out / "heatmap.png", "report": out / "report.html"}

    def load_and_validate_regions_file(path, *, expected_image_sha256=None, image_size=None):
        if not regions_ok:
            return None, types.SimpleNamespace(ok=False, errors=({"code": "BAD_REGIONS", "message": "c3 兜底拒绝"},))
        return None, types.SimpleNamespace(ok=True, errors=())

    module.render_run_report = render_run_report
    module.load_and_validate_regions_file = load_and_validate_regions_file
    monkeypatch.setitem(sys.modules, cli.REPORT_MODULE, module)


def test_summarize_c3_fallback_rejects_regions(monkeypatch, capsys, tmp_path):
    """接线点④：C3 兜底校验拒绝 → INVALID_AOI 退出码 2（权威校验之后第二道防线）。"""
    _install_fake_report(monkeypatch, regions_ok=False)
    src = _build_source_run(tmp_path)
    regions = _regions_file(tmp_path)
    out = tmp_path / "runs" / "run-c3-reject"
    rc, env = _run(capsys, ["summarize", "--analysis", str(src), "--regions", str(regions), "--out", str(out)])
    assert rc == 2
    assert env["error"]["id"] == "INVALID_AOI"
    assert not out.exists()


def test_summarize_carries_base_and_review_and_renders(monkeypatch, capsys, tmp_path):
    """接线点①：base.png/review.json 从源运行目录复用；渲染产物与色阶参数入 manifest。"""
    _install_fake_report(monkeypatch)
    src = _build_source_run(tmp_path)
    base = np.zeros((48, 64, 3), dtype=np.uint8)
    base[:, :, 1] = 90
    imaging.save_rgb_png(base, src / "base.png")
    synthetic.write_json(src / "review.json", VALID_REVIEW)

    regions = _regions_file(tmp_path)
    out = tmp_path / "runs" / "run-render"
    rc, env = _run(capsys, ["summarize", "--analysis", str(src), "--regions", str(regions), "--out", str(out)])
    assert rc == 0 and env["ok"] is True
    assert (out / "base.png").is_file()
    assert (out / "review.json").is_file()  # Agent 评审只校验与呈现，原样携带
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["rendered_report_artifacts"]) == {"overlay.png", "heatmap.png", "report.html"}
    assert manifest["overlay"]["alpha"] == 0.55
    assert manifest["colorscale"]["name"] == "ember"


def test_summarize_without_base_skips_render_with_limitation(monkeypatch, capsys, tmp_path):
    _install_fake_report(monkeypatch)
    src = _build_source_run(tmp_path)  # 无 base.png
    regions = _regions_file(tmp_path)
    out = tmp_path / "runs" / "run-norender"
    rc, _env = _run(capsys, ["summarize", "--analysis", str(src), "--regions", str(regions), "--out", str(out)])
    assert rc == 0
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert any("base.png" in x for x in analysis["limitations"])
    assert not (out / "report.html").exists()


def test_summarize_invalid_review_exit2(capsys, tmp_path):
    """源运行的 review.json 无效 → 退出码 2（CLI 只校验与呈现，不放过坏结构）。"""
    src = _build_source_run(tmp_path)
    bad_review = dict(VALID_REVIEW)
    bad_review["findings"] = [dict(VALID_REVIEW["findings"][0], evidence_type="guessed")]
    synthetic.write_json(src / "review.json", bad_review)
    regions = _regions_file(tmp_path)
    out = tmp_path / "runs" / "run-badreview"
    rc, env = _run(capsys, ["summarize", "--analysis", str(src), "--regions", str(regions), "--out", str(out)])
    assert rc == 2
    assert env["error"]["id"] == "INVALID_REQUEST"
    assert not out.exists()
