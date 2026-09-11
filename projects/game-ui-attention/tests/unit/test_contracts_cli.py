"""cli.py 契约测试：stdout JSON 信封、退出码分支、失败恢复无假成功（validation-plan §1“失败恢复”行）。

后端经 monkeypatch 注入 sys.modules 的假 registry 驱动（不依赖 C2 实现与真实权重）。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import synthetic  # noqa: E402
from ui_attention import cli  # noqa: E402
from ui_attention.contracts import BackendInfo, PredictionResult, ResolvedProfile, WeightRef  # noqa: E402
from ui_attention.errors import ErrorCode, UiAttentionError  # noqa: E402

SHA = "8" * 64
WEIGHT = WeightRef(name="fake.onnx", source_url="https://example.invalid/fake.onnx", sha256=SHA, size_bytes=1024)
INFO = BackendInfo(
    backend_id="foveacast-onnx-3s",
    version="0.1.0",
    capabilities=("spatial_density",),
    native_semantics="probability_density",
    device_requirements={"device": "cpu", "min_free_vram_mb": None},
    license_status={"code": "MIT", "weights": "MIT", "gaps": [], "cleared_for": "internal-eval"},
    weights=(WEIGHT,),
)
PROFILE = ResolvedProfile(
    profile_name="foveacast-onnx-3s-v1",
    backend_id="foveacast-onnx-3s",
    backend_version="0.1.0",
    preprocessing={"target_hw": [240, 320]},
    centerbias=None,
    viewing_conditions={"window": "3s"},
    config_hash="c" * 64,
    weights=(WEIGHT,),
)


class FakeBackend:
    """合成测试后端：按构造参数返回 uniform/log_density/NaN/vendor 输出，或抛结构化错误。"""

    def __init__(self, *, mode: str = "uniform", raise_exc: Exception | None = None) -> None:
        self.mode = mode
        self.raise_exc = raise_exc
        self.predict_calls = 0

    def describe(self) -> BackendInfo:
        return INFO

    def predict(self, image: np.ndarray, resolved_profile: ResolvedProfile) -> PredictionResult:
        self.predict_calls += 1
        if self.raise_exc is not None:
            raise self.raise_exc
        h, w = 240, 320
        mapping = {
            "original_shape": [int(image.shape[0]), int(image.shape[1])],
            "inference_shape": [h, w],
            "method": "direct_anisotropic_to_target",
        }
        runtime = {"device": "cpu", "precision": "fp32", "elapsed_ms": 12.5, "peak_mem_mb": 300.0}
        limits = ("合成测试后端：非真实模型输出", "分辨率上限 240×320", "训练分布不含游戏 UI")
        if self.mode == "uniform":
            arr = np.full((h, w), 1.0 / (h * w), dtype=np.float64)
            return PredictionResult("probability_density", arr, mapping, runtime, None, limits)
        if self.mode == "log_density":
            arr = np.full((h, w), -np.log(h * w), dtype=np.float64)
            return PredictionResult("log_density", arr, mapping, runtime, None, limits)
        if self.mode == "nan":
            arr = np.full((h, w), 1.0 / (h * w), dtype=np.float64)
            arr[5, 5] = np.nan
            return PredictionResult("probability_density", arr, mapping, runtime, None, limits)
        if self.mode == "vendor":
            arr = np.full((h, w), 0.5, dtype=np.float64)
            return PredictionResult("vendor_metrics", arr, mapping, runtime, {"vendor_score": 0.7}, limits)
        raise AssertionError(f"unknown mode {self.mode}")


def install_fake_registry(monkeypatch, *, mode="uniform", raise_exc=None, resolve_raises=None):
    backend = FakeBackend(mode=mode, raise_exc=raise_exc)
    module = types.ModuleType(cli.REGISTRY_MODULE)

    def resolve_profile(name: str) -> ResolvedProfile:
        if resolve_raises is not None:
            raise resolve_raises
        return PROFILE

    module.list_profiles = lambda: (PROFILE.profile_name,)
    module.resolve_profile = resolve_profile
    module.get_backend = lambda profile: backend
    module.doctor_report = lambda: {"status": "ok", "backends": [{"backend_id": INFO.backend_id}]}
    monkeypatch.setitem(sys.modules, cli.REGISTRY_MODULE, module)
    return backend


def run_cli(capsys, argv) -> tuple[int, dict]:
    rc = cli.main(argv)
    out = capsys.readouterr().out
    return rc, json.loads(out)


def make_workspace(tmp_path, *, regions=None) -> tuple[Path, Path, Path]:
    """合成图片 + request.json + 不存在的 out 目录。"""
    img = synthetic.solid_color_png(tmp_path / "input.png")
    request = synthetic.write_json(
        tmp_path / "request.json",
        synthetic.base_request(regions=[synthetic.VALID_RECT_REGION] if regions is None else regions),
    )
    out = tmp_path / "runs" / "run-a"
    return img, request, out


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_doctor_missing_registry_exit3(monkeypatch, capsys):
    """后端 registry 缺失 → 退出码 3 结构化报错，不崩溃。"""
    monkeypatch.setitem(sys.modules, cli.REGISTRY_MODULE, None)  # import 立即 ImportError
    rc, env = run_cli(capsys, ["doctor"])
    assert rc == 3
    assert env["ok"] is False
    assert env["error"]["id"] == "MODEL_NOT_READY"
    assert env["error"]["exit_code"] == 3


def test_doctor_with_registry_exit0(monkeypatch, capsys):
    install_fake_registry(monkeypatch)
    rc, env = run_cli(capsys, ["doctor"])
    assert rc == 0
    assert env["ok"] is True
    assert env["result"]["profiles"] == ["foveacast-onnx-3s-v1"]
    assert env["result"]["doctor"]["status"] == "ok"
    assert "numpy" in env["result"]["dependencies"]


# ---------------------------------------------------------------------------
# analyze 成功路径
# ---------------------------------------------------------------------------


def test_analyze_success_writes_artifacts_manifest_last(monkeypatch, capsys, tmp_path):
    install_fake_registry(monkeypatch, mode="uniform")
    _img, request, out = make_workspace(tmp_path)
    rc, env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 0 and env["ok"] is True

    # 产物齐备；manifest.json 最后写（其内容收录全部其他产物）
    files = {p.name for p in out.iterdir()}
    assert {"manifest.json", "analysis.json", "regions.json", "density.npy"} <= files
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {a["path"] for a in manifest["artifacts"]}
    assert {"analysis.json", "regions.json", "density.npy"} <= manifest_paths

    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["computation_status"] == "complete"
    assert analysis["review_status"] == "not_requested"
    assert analysis["evidence_type"] == "model_prediction"
    # 均匀概率图：mass == 面积占比、relative_density == 1
    row = analysis["regions"][0]
    assert row["probability_mass"] == pytest.approx(row["area_fraction"], abs=1e-9)
    assert row["relative_density"] == pytest.approx(1.0, abs=1e-9)
    # profile 记录实际解析结果（不只保存别名）
    assert analysis["profile"]["profile_name"] == "foveacast-onnx-3s-v1"
    assert analysis["profile"]["config_hash"] == "c" * 64
    assert analysis["model"]["weights_sha256"] == [SHA]
    # 不落盘机器绝对路径（image_path 保留请求原样）
    assert analysis["input"]["image_path"] == "./input.png"
    # density.npy 全精度 float64 且 sum=1
    density = np.load(out / "density.npy")
    assert density.dtype == np.float64
    assert density.shape == (48, 64)
    assert density.sum() == pytest.approx(1.0, abs=1e-6)
    # regions.json 携带图片 SHA-256
    regions_out = json.loads((out / "regions.json").read_text(encoding="utf-8"))
    assert regions_out["image_sha256"] == analysis["input"]["image_sha256"]


def test_analyze_log_density_semantics_converted(monkeypatch, capsys, tmp_path):
    backend = install_fake_registry(monkeypatch, mode="log_density")
    _img, request, out = make_workspace(tmp_path)
    rc, env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 0
    assert backend.predict_calls == 1
    density = np.load(out / "density.npy")
    assert density.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# analyze 失败路径：无假成功、不覆盖历史
# ---------------------------------------------------------------------------


def test_analyze_invalid_density_exit5_and_cleanup(monkeypatch, capsys, tmp_path):
    install_fake_registry(monkeypatch, mode="nan")
    _img, request, out = make_workspace(tmp_path)
    rc, env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 5
    assert env["error"]["id"] == "INVALID_DENSITY"
    assert not out.exists()  # 失败恢复：清理自建目录，无假成功、无半成品


def test_analyze_vendor_metrics_exit5(monkeypatch, capsys, tmp_path):
    install_fake_registry(monkeypatch, mode="vendor")
    _img, request, out = make_workspace(tmp_path)
    rc, env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 5
    assert env["error"]["id"] == "UNSUPPORTED_SEMANTICS"
    assert not out.exists()


def test_analyze_gpu_oom_exit4(monkeypatch, capsys, tmp_path):
    install_fake_registry(monkeypatch, raise_exc=UiAttentionError(ErrorCode.GPU_OOM, "显存不足", {"free_mb": 10}))
    _img, request, out = make_workspace(tmp_path)
    rc, env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 4
    assert env["error"]["id"] == "GPU_OOM"
    assert not out.exists()


def test_analyze_model_not_ready_exit3(monkeypatch, capsys, tmp_path):
    install_fake_registry(
        monkeypatch,
        raise_exc=UiAttentionError(ErrorCode.MODEL_NOT_READY, "权重哈希不符", {"weight": "fake.onnx"}),
    )
    _img, request, out = make_workspace(tmp_path)
    rc, env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 3
    assert env["error"]["id"] == "MODEL_NOT_READY"


def test_analyze_profile_not_registered_exit3(monkeypatch, capsys, tmp_path):
    install_fake_registry(
        monkeypatch,
        resolve_raises=UiAttentionError(
            ErrorCode.PROFILE_NOT_REGISTERED, "profile 未登记", {"requested": "local-static-v1"}
        ),
    )
    _img, request, out = make_workspace(tmp_path)
    rc, env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 3
    assert env["error"]["id"] == "PROFILE_NOT_REGISTERED"


def test_analyze_output_exists_exit7_no_overwrite(monkeypatch, capsys, tmp_path):
    backend = install_fake_registry(monkeypatch)
    _img, request, out = make_workspace(tmp_path)
    out.mkdir(parents=True)
    sentinel = out / "old-manifest.json"
    sentinel.write_text("历史产物", encoding="utf-8")
    rc, env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 7
    assert env["error"]["id"] == "OUTPUT_PATH_EXISTS"
    # 历史目录未被触碰，且未发生推理
    assert sentinel.read_text(encoding="utf-8") == "历史产物"
    assert {p.name for p in out.iterdir()} == {"old-manifest.json"}
    assert backend.predict_calls == 0


def test_analyze_invalid_request_exit2(monkeypatch, capsys, tmp_path):
    install_fake_registry(monkeypatch)
    request = synthetic.write_json(tmp_path / "bad.json", synthetic.base_request(schema_version="request/v9"))
    out = tmp_path / "runs" / "run-bad"
    rc, env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 2
    assert env["error"]["id"] == "INVALID_SCHEMA_VERSION"
    assert not out.exists()  # 请求无效时不创建运行目录


def test_analyze_invalid_aoi_exit2(monkeypatch, capsys, tmp_path):
    install_fake_registry(monkeypatch)
    _img, request, out = make_workspace(tmp_path, regions=[synthetic.OUT_OF_BOUNDS_RECT])
    rc, env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 2
    assert env["error"]["id"] == "INVALID_AOI"
    assert not out.exists()


def test_analyze_transparent_image_exit2_without_background(monkeypatch, capsys, tmp_path):
    install_fake_registry(monkeypatch)
    img = synthetic.rgba_png(tmp_path / "input.png")
    request = synthetic.write_json(tmp_path / "request.json", synthetic.base_request())
    out = tmp_path / "runs" / "run-alpha"
    rc, env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 2
    assert env["error"]["id"] == "INVALID_IMAGE"
    assert "透明" in env["error"]["message"]
    # 显式合成背景则允许
    rc2, env2 = run_cli(
        capsys, ["analyze", "--request", str(request), "--out", str(out), "--composite-background", "0,0,0"]
    )
    assert rc2 == 0 and env2["ok"] is True
    del img


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _write_run(tmp_path: Path, name: str, analysis_dict: dict) -> Path:
    run = tmp_path / name
    run.mkdir()
    synthetic.write_json(run / "analysis.json", analysis_dict)
    return run


def test_compare_self_exit0_delta_zero(capsys, tmp_path):
    run_a = _write_run(tmp_path, "run-a", synthetic.sample_analysis_dict())
    out = tmp_path / "runs" / "cmp"
    rc, env = run_cli(capsys, ["compare", "--before", str(run_a), "--after", str(run_a), "--out", str(out)])
    assert rc == 0 and env["ok"] is True
    assert env["result"]["matched_regions"] == 1
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["regions"]["matched"][0]["delta_pp"] == 0.0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "comparison.json" in {a["path"] for a in manifest["artifacts"]}


def test_compare_incompatible_exit6_with_reasons(capsys, tmp_path):
    before = synthetic.sample_analysis_dict()
    after = synthetic.sample_analysis_dict(analysis_id="analysis-test-0002")
    after["profile"]["config_hash"] = "d" * 64  # 预处理/先验配置变化
    run_a = _write_run(tmp_path, "run-a", before)
    run_b = _write_run(tmp_path, "run-b", after)
    out = tmp_path / "runs" / "cmp-bad"
    rc, env = run_cli(capsys, ["compare", "--before", str(run_a), "--after", str(run_b), "--out", str(out)])
    assert rc == 6
    assert env["error"]["id"] == "COMPARISON_INCOMPATIBLE"
    mismatches = env["error"]["details"]["mismatches"]
    assert any(m["field"] == "profile.config_hash" for m in mismatches)  # 原因保留
    assert not out.exists()  # 拒绝正式比较时不产生部分产物


# ---------------------------------------------------------------------------
# 未预期异常 → 结构化 INTERNAL_ERROR（退出码 1），不产生假成功
# ---------------------------------------------------------------------------


def test_unexpected_exception_structured(monkeypatch, capsys, tmp_path):
    install_fake_registry(monkeypatch)

    def boom(*_a, **_k):
        raise RuntimeError("意外缺陷")

    monkeypatch.setattr(cli, "load_analyze_request", boom)
    _img, request, out = make_workspace(tmp_path)
    rc, env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 1
    assert env["ok"] is False
    assert env["error"]["id"] == "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# C3 report 接线（总控接线点①③：渲染入口、manifest overlay/colorscale 键）
# 用假 report 模块隔离 C3 实现：验证的是 C1 接线逻辑本身
# ---------------------------------------------------------------------------


class FakeColorScale:
    def __init__(self, params):
        self._params = params

    @classmethod
    def shared(cls, name, density_a, density_b, normalization="linear"):
        vmin = float(min(density_a.min(), density_b.min()))
        vmax = float(max(density_a.max(), density_b.max()))
        if vmax <= vmin:  # 与 C3 shared_range 相同的退化规则：确定性扩展 (vmin, vmin+1)
            vmax = vmin + 1.0
        return cls({"name": name, "vmin": vmin, "vmax": vmax, "normalization": normalization})

    def to_params(self):
        return dict(self._params)


def install_fake_report(monkeypatch, *, fail_render: bool = False, regions_ok: bool = True):
    module = types.ModuleType(cli.REPORT_MODULE)
    module.DEFAULT_OVERLAY_ALPHA = 0.55
    module.ColorScale = FakeColorScale
    module.overlay_record = lambda cs, alpha: {
        "alpha": alpha,
        "colorscale": cs.to_params(),
        "output_size_matches_original": True,
        "random_sources": "none",
    }

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
        if fail_render:
            raise RuntimeError("render boom")
        out = Path(out_dir)
        (out / "overlay.png").write_bytes(b"fake-png")
        (out / "heatmap.png").write_bytes(b"fake-png")
        (out / "report.html").write_text("<html></html>", encoding="utf-8")
        return {"overlay": out / "overlay.png", "heatmap": out / "heatmap.png", "report": out / "report.html"}

    def render_compare_report(
        *,
        out_dir,
        comparison,
        base_a,
        density_a,
        regions_a,
        base_b,
        density_b,
        regions_b,
        name="ember",
        normalization="linear",
        alpha=0.55,
        title=None,
    ):
        if fail_render:
            raise RuntimeError("render boom")
        out = Path(out_dir)
        for n in ("overlay_a.png", "heatmap_a.png", "overlay_b.png", "heatmap_b.png"):
            (out / n).write_bytes(b"fake-png")
        (out / "report.html").write_text("<html></html>", encoding="utf-8")
        return {"report": out / "report.html"}

    def load_and_validate_regions_file(path, *, expected_image_sha256=None, image_size=None):
        if not regions_ok:
            return None, types.SimpleNamespace(ok=False, errors=({"code": "BAD_REGIONS", "message": "c3 兜底拒绝"},))
        return None, types.SimpleNamespace(ok=True, errors=())

    module.render_run_report = render_run_report
    module.render_compare_report = render_compare_report
    module.load_and_validate_regions_file = load_and_validate_regions_file
    monkeypatch.setitem(sys.modules, cli.REPORT_MODULE, module)
    return module


def test_analyze_render_wiring_manifest_keys(monkeypatch, capsys, tmp_path):
    install_fake_registry(monkeypatch)
    install_fake_report(monkeypatch)
    _img, request, out = make_workspace(tmp_path)
    rc, _env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rendered_report_artifacts"] == ["overlay.png", "heatmap.png", "report.html"]
    assert manifest["overlay"]["alpha"] == 0.55
    assert manifest["colorscale"]["name"] == "ember"
    assert manifest["colorscale"]["vmin"] < manifest["colorscale"]["vmax"]
    assert (out / "base.png").is_file()  # 展示副本落盘（渲染内嵌 + summarize 复用）


def test_analyze_render_failure_is_partial_completion(monkeypatch, capsys, tmp_path):
    """渲染失败 → 计算结果保留为部分完成（退出码仍 0），limitations 记录原因。"""
    install_fake_registry(monkeypatch)
    install_fake_report(monkeypatch, fail_render=True)
    _img, request, out = make_workspace(tmp_path)
    rc, _env = run_cli(capsys, ["analyze", "--request", str(request), "--out", str(out)])
    assert rc == 0
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert any("report 渲染失败" in x for x in analysis["limitations"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rendered_report_artifacts"] == []
    assert "overlay" not in manifest


def _write_renderable_run(tmp_path: Path, name: str, analysis_dict: dict) -> Path:
    """带渲染素材（base.png + density.npy）的运行目录。"""
    from ui_attention import imaging

    run = _write_run(tmp_path, name, analysis_dict)
    density = synthetic.uniform_probability((48, 64))
    np.save(run / "density.npy", density)
    base = np.zeros((48, 64, 3), dtype=np.uint8)
    base[:, :, 2] = 120
    imaging.save_rgb_png(base, run / "base.png")
    return run


def test_compare_render_wiring_shared_colorscale(monkeypatch, capsys, tmp_path):
    install_fake_report(monkeypatch)
    run_a = _write_renderable_run(tmp_path, "run-a", synthetic.sample_analysis_dict())
    out = tmp_path / "runs" / "cmp-render"
    rc, _env = run_cli(capsys, ["compare", "--before", str(run_a), "--after", str(run_a), "--out", str(out)])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "report.html" in manifest["rendered_report_artifacts"]
    # A/B 共用色阶参数写入 manifest（data-contract §6）
    assert manifest["colorscale"]["name"] == "ember"
    assert manifest["overlay"]["colorscale"] == manifest["colorscale"]
    assert (out / "overlay_a.png").is_file() and (out / "overlay_b.png").is_file()


def test_compare_render_skipped_without_materials(capsys, tmp_path):
    """源运行目录缺 base.png/density.npy → 跳过渲染并记录，comparison.json 仍为正式产物。"""
    run_a = _write_run(tmp_path, "run-a", synthetic.sample_analysis_dict())  # 只有 analysis.json
    out = tmp_path / "runs" / "cmp-skip"
    rc, _env = run_cli(capsys, ["compare", "--before", str(run_a), "--after", str(run_a), "--out", str(out)])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rendered_report_artifacts"] == []
    assert "缺少 base.png/density.npy" in manifest["render_note"]
