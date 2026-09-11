"""Unit tests for backends/onnx_foveacast.py (C2).

适配层逻辑（重归一化、shape_mapping、无效输出拒绝、缺失权重结构化错误、
元数据运行时读取）全部用假 session（monkeypatch）测试，不依赖真权重；
真权重推理测试带 skipif(model-cache 无权重) 守卫。
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from ui_attention.backends import registry
from ui_attention.backends.errors import (
    ImageRejectedError,
    InferenceFailureError,
    InvalidDensityError,
    MemoryExhaustedError,
    ProfileRejectedError,
    WeightNotReadyError,
)
from ui_attention.backends.onnx_foveacast import FoveacastOnnxBackend
from ui_attention.backends.weights import WeightFetchResult, default_cache_dir
from ui_attention.contracts.backend import Backend, PredictionResult
from ui_attention.errors import ErrorCode

PROFILE = registry.resolve_profile(registry.FOVEACAST_3S_PROFILE)
WEIGHT_PATH = default_cache_dir() / registry.FOVEACAST_3S_WEIGHT.name
HAS_WEIGHTS = WEIGHT_PATH.is_file()
needs_weights = pytest.mark.skipif(
    not HAS_WEIGHTS, reason="real weights not cached (run python -m ui_attention.backends)"
)

RELEASE_SHA256 = "842a23f97908d146b8749e05f6b220bdb495eae76c75cc7252550825585ef76e"


# ---------------------------------------------------------------------------
# Fake ORT session infrastructure
# ---------------------------------------------------------------------------


class FakeMeta:
    def __init__(self, name: str, shape: list, type_: str = "tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_


class FakeSession:
    """Mimics the onnxruntime.InferenceSession surface used by the adapter."""

    def __init__(
        self,
        output_fn,
        input_name: str = "input",
        output_name: str = "output",
        input_shape=None,
        output_shape=None,
        input_type: str = "tensor(float)",
        output_type: str = "tensor(float)",
        extra_inputs: int = 0,
    ):
        self._inputs = [FakeMeta(input_name, input_shape or ["batch_size", 3, 240, 320], input_type)]
        self._inputs += [FakeMeta(f"extra_{i}", ["batch_size", 8]) for i in range(extra_inputs)]
        self._outputs = [FakeMeta(output_name, output_shape or ["batch_size", 1, 240, 320], output_type)]
        self._output_fn = output_fn
        self.calls: list[tuple[list[str], dict[str, np.ndarray]]] = []

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):
        self.calls.append((list(output_names), dict(feeds)))
        return self._output_fn(feeds)


def ramp_output(h: int = 240, w: int = 320, dtype=np.float32):
    yy = np.linspace(0.0, 1.0, h, dtype=np.float64)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float64)[None, :]
    plane = ((yy + xx) / 2.0).astype(dtype).reshape(1, 1, h, w)
    return lambda feeds: [plane]


def make_backend(monkeypatch: pytest.MonkeyPatch, session: FakeSession, cache_dir: Path) -> FoveacastOnnxBackend:
    """Backend with weight verification and ORT session creation faked out."""

    def fake_verify(ref, _cache_dir):
        return WeightFetchResult(
            name=ref.name,
            path=cache_dir / ref.name,
            sha256=ref.sha256,
            size_bytes=ref.size_bytes,
            source="cache",
        )

    monkeypatch.setattr("ui_attention.backends.onnx_foveacast.verify_cached_weight", fake_verify)
    monkeypatch.setattr("onnxruntime.InferenceSession", lambda path, providers=None: session)
    return FoveacastOnnxBackend(cache_dir=cache_dir)


def ui_image(h: int = 120, w: int = 200, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def constant_image(h: int = 64, w: int = 96, rgb=(255, 0, 0)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = np.array(rgb, dtype=np.uint8)
    return img


# ---------------------------------------------------------------------------
# Happy path: renormalization, contract fields, metadata-driven I/O
# ---------------------------------------------------------------------------


def test_predict_renormalizes_to_probability_density(monkeypatch, tmp_path):
    session = FakeSession(ramp_output())
    backend = make_backend(monkeypatch, session, tmp_path)
    result = backend.predict(ui_image(120, 200), PROFILE)
    assert isinstance(result, PredictionResult)
    assert result.semantics == "probability_density"
    assert result.array.dtype == np.float64
    assert result.array.shape == (240, 320)
    assert abs(float(result.array.sum()) - 1.0) < 1e-12
    assert float(result.array.min()) >= 0.0
    assert np.all(np.isfinite(result.array))
    assert result.vendor_metrics is None
    result.validate()  # C1 frozen contract validation


def test_runtime_field_is_measured_and_frozen_shaped(monkeypatch, tmp_path):
    backend = make_backend(monkeypatch, FakeSession(ramp_output()), tmp_path)
    runtime = backend.predict(ui_image(), PROFILE).runtime
    assert set(runtime) == {"device", "precision", "elapsed_ms", "peak_mem_mb"}
    assert runtime["device"] == "cpu"
    assert runtime["precision"] == "fp16"
    assert isinstance(runtime["elapsed_ms"], (int, float)) and runtime["elapsed_ms"] > 0
    assert runtime["peak_mem_mb"] is None or isinstance(runtime["peak_mem_mb"], (int, float))


def test_limitations_required_entries(monkeypatch, tmp_path):
    backend = make_backend(monkeypatch, FakeSession(ramp_output()), tmp_path)
    limitations = backend.predict(ui_image(), PROFILE).limitations
    joined = "\n".join(limitations)
    assert "min-max" in joined and "跨图绝对强度不可比" in joined  # 相对量语义
    assert "240×320" in joined and "小元素" in joined  # 分辨率上限
    assert "不含游戏 UI" in joined and "效果未验证" in joined  # 训练分布边界
    assert "未经本项目复现" in joined  # 作者指标声明
    assert limitations == FoveacastOnnxBackend.LIMITATIONS


def test_tensor_names_come_from_session_metadata_not_hardcoded(monkeypatch, tmp_path):
    session = FakeSession(ramp_output(), input_name="x_in", output_name="y_out")
    backend = make_backend(monkeypatch, session, tmp_path)
    result = backend.predict(ui_image(), PROFILE)
    assert abs(float(result.array.sum()) - 1.0) < 1e-12
    output_names, feeds = session.calls[0]
    assert output_names == ["y_out"]
    assert list(feeds) == ["x_in"]
    meta = backend.session_metadata
    assert meta["input"]["name"] == "x_in" and meta["output"]["name"] == "y_out"


def test_input_tensor_preprocessing_contract(monkeypatch, tmp_path):
    session = FakeSession(ramp_output())
    backend = make_backend(monkeypatch, session, tmp_path)
    backend.predict(constant_image(rgb=(255, 0, 0)), PROFILE)
    tensor = session.calls[0][1]["input"]
    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 3, 240, 320)
    assert tensor.flags["C_CONTIGUOUS"]
    # RGB 通道序 + [0,255] 值域 + 均值减除在图内（适配层不得预处理均值）
    assert abs(float(tensor[0, 0].mean()) - 255.0) < 1e-3
    assert abs(float(tensor[0, 1].mean())) < 1e-3
    assert abs(float(tensor[0, 2].mean())) < 1e-3
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 255.0


def test_shape_mapping_records_original_inference_and_inverse(monkeypatch, tmp_path):
    backend = make_backend(monkeypatch, FakeSession(ramp_output()), tmp_path)
    mapping = backend.predict(ui_image(120, 200), PROFILE).shape_mapping
    assert mapping["original_shape"] == [120, 200]
    assert mapping["inference_shape"] == [240, 320]
    assert mapping["method"] == "direct_anisotropic_resize"
    assert mapping["resize_filter"] == "pillow_bicubic"
    assert mapping["padding"] is None
    assert mapping["scale_yx"] == pytest.approx([2.0, 1.6])
    inverse = mapping["inverse"]
    assert inverse["target_shape"] == [120, 200]
    assert inverse["renormalize"] == "sum_to_1"
    assert inverse["method"] == "resize_density_to_original_then_renormalize"


def test_predict_is_deterministic_with_fake_session(monkeypatch, tmp_path):
    backend = make_backend(monkeypatch, FakeSession(ramp_output()), tmp_path)
    image = ui_image()
    a = backend.predict(image, PROFILE).array
    b = backend.predict(image, PROFILE).array
    assert np.array_equal(a, b)


def test_backend_satisfies_frozen_protocol(monkeypatch, tmp_path):
    backend = make_backend(monkeypatch, FakeSession(ramp_output()), tmp_path)
    assert isinstance(backend, Backend)


def test_cold_load_recorded_once(monkeypatch, tmp_path):
    backend = make_backend(monkeypatch, FakeSession(ramp_output()), tmp_path)
    assert backend.cold_load_ms is None
    assert backend.session_metadata is None
    backend.predict(ui_image(), PROFILE)
    assert backend.cold_load_ms is not None and backend.cold_load_ms >= 0.0
    assert backend.weight_path is not None


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------


def test_describe_matches_frozen_contract(monkeypatch, tmp_path):
    backend = make_backend(monkeypatch, FakeSession(ramp_output()), tmp_path)
    info = backend.describe()
    info.validate()  # C1 frozen validation
    assert info.backend_id == "foveacast-onnx-3s"
    assert info.version == registry.FOVEACAST_3S_BACKEND_VERSION
    assert info.capabilities == ("spatial_density",)
    assert info.native_semantics == "probability_density"
    assert info.device_requirements == {"device": "cpu", "min_free_vram_mb": None}
    ls = info.license_status
    assert set(ls) == {"code", "weights", "gaps", "cleared_for"}
    assert ls["gaps"] == ["G3", "G4", "G8"] and ls["cleared_for"] == "internal-eval"
    assert len(info.weights) == 1
    assert info.weights[0].sha256 == RELEASE_SHA256


# ---------------------------------------------------------------------------
# Failure paths: all structured, never a fabricated heatmap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("image", "reason"),
    [
        (None, "not_ndarray"),
        ([1, 2, 3], "not_ndarray"),
        (np.zeros((10, 10, 3), dtype=np.float32), "dtype"),
        (np.zeros((10, 10, 4), dtype=np.uint8), "shape"),
        (np.zeros((10, 10), dtype=np.uint8), "shape"),
        (np.zeros((0, 5, 3), dtype=np.uint8), "empty"),
    ],
)
def test_invalid_image_structured(tmp_path, image, reason):
    backend = FoveacastOnnxBackend(cache_dir=tmp_path)
    with pytest.raises(ImageRejectedError) as ei:
        backend.predict(image, PROFILE)
    assert ei.value.code is ErrorCode.INVALID_IMAGE
    assert ei.value.exit_code == 2
    assert ei.value.details["reason"] == reason


@pytest.mark.parametrize(
    ("output_fn", "reason"),
    [
        (lambda feeds: [np.full((1, 1, 240, 320), np.nan, dtype=np.float32)], "non_finite"),
        (lambda feeds: [np.full((1, 1, 240, 320), -1.0, dtype=np.float32)], "negative_values"),
        (lambda feeds: [np.zeros((1, 1, 240, 320), dtype=np.float32)], "degenerate_sum"),
        (lambda feeds: [np.ones((1, 1, 120, 160), dtype=np.float32)], "shape"),
        (lambda feeds: [np.ones((1, 3, 240, 320), dtype=np.float32)], "shape"),
        (lambda feeds: [np.ones((1, 1, 240, 320), dtype=np.int32)], "dtype"),
        (lambda feeds: [], "missing_output"),
    ],
)
def test_invalid_model_output_structured(monkeypatch, tmp_path, output_fn, reason):
    backend = make_backend(monkeypatch, FakeSession(output_fn), tmp_path)
    with pytest.raises(InvalidDensityError) as ei:
        backend.predict(ui_image(), PROFILE)
    assert ei.value.code is ErrorCode.INVALID_DENSITY
    assert ei.value.exit_code == 5
    assert ei.value.details["reason"] == reason


def test_missing_weight_structured_before_session(tmp_path):
    backend = FoveacastOnnxBackend(cache_dir=tmp_path)  # empty cache, no patching
    with pytest.raises(WeightNotReadyError) as ei:
        backend.predict(ui_image(), PROFILE)
    assert ei.value.code is ErrorCode.MODEL_NOT_READY
    assert ei.value.exit_code == 3
    assert ei.value.details["reason"] == "missing_file"


def test_hash_mismatch_weight_structured(tmp_path):
    (tmp_path / registry.FOVEACAST_3S_WEIGHT.name).write_bytes(b"corrupt-payload")
    backend = FoveacastOnnxBackend(cache_dir=tmp_path)
    with pytest.raises(WeightNotReadyError) as ei:
        backend.predict(ui_image(), PROFILE)
    assert ei.value.details["reason"] == "sha256_mismatch"


@pytest.mark.parametrize(
    "session_kwargs",
    [
        {"input_type": "tensor(float16)"},
        {"output_type": "tensor(float16)"},
        {"input_shape": ["batch_size", 3, 480, 640]},
        {"input_shape": ["batch_size", 4, 240, 320]},
        {"output_shape": ["batch_size", 1, 120, 160]},
        {"extra_inputs": 1},
    ],
)
def test_unsupported_onnx_metadata_structured(monkeypatch, tmp_path, session_kwargs):
    session = FakeSession(ramp_output(), **session_kwargs)
    backend = make_backend(monkeypatch, session, tmp_path)
    with pytest.raises(WeightNotReadyError) as ei:
        backend.predict(ui_image(), PROFILE)
    assert ei.value.code is ErrorCode.MODEL_NOT_READY
    assert ei.value.details["reason"] == "metadata_unsupported"


def test_profile_type_rejected(tmp_path):
    backend = FoveacastOnnxBackend(cache_dir=tmp_path)
    with pytest.raises(ProfileRejectedError) as ei:
        backend.predict(ui_image(), {"profile": "not a ResolvedProfile"})  # type: ignore[arg-type]
    assert ei.value.code is ErrorCode.PROFILE_NOT_REGISTERED
    assert ei.value.exit_code == 3
    assert ei.value.details["reason"] == "profile_type"


def test_profile_tampered_rejected_at_predict(tmp_path):
    tampered = dataclasses.replace(
        PROFILE, preprocessing={**PROFILE.preprocessing, "target_width": 640}
    )
    backend = FoveacastOnnxBackend(cache_dir=tmp_path)
    with pytest.raises(ProfileRejectedError) as ei:
        backend.predict(ui_image(), tampered)
    assert ei.value.details["reason"] == "profile_tampered"


def test_profile_of_other_backend_rejected(tmp_path):
    foreign_hash = registry.compute_config_hash(
        backend_id="deepgaze-iie",
        backend_version=PROFILE.backend_version,
        weights=PROFILE.weights,
        preprocessing=PROFILE.preprocessing,
        centerbias=PROFILE.centerbias,
        viewing_conditions=PROFILE.viewing_conditions,
    )
    foreign = dataclasses.replace(PROFILE, backend_id="deepgaze-iie", config_hash=foreign_hash)
    backend = FoveacastOnnxBackend(cache_dir=tmp_path)
    with pytest.raises(ProfileRejectedError) as ei:
        backend.predict(ui_image(), foreign)
    assert ei.value.details["reason"] == "backend_mismatch"


def test_profile_without_weights_structured(tmp_path):
    bare_hash = registry.compute_config_hash(
        backend_id=PROFILE.backend_id,
        backend_version=PROFILE.backend_version,
        weights=(),
        preprocessing=PROFILE.preprocessing,
        centerbias=PROFILE.centerbias,
        viewing_conditions=PROFILE.viewing_conditions,
    )
    bare = dataclasses.replace(PROFILE, weights=(), config_hash=bare_hash)
    backend = FoveacastOnnxBackend(cache_dir=tmp_path)
    with pytest.raises(WeightNotReadyError) as ei:
        backend.predict(ui_image(), bare)
    assert ei.value.details["reason"] == "no_weights_registered"


def test_session_run_failure_structured(monkeypatch, tmp_path):
    def _boom(feeds):
        raise RuntimeError("ort boom (simulated)")

    backend = make_backend(monkeypatch, FakeSession(_boom), tmp_path)
    with pytest.raises(InferenceFailureError) as ei:
        backend.predict(ui_image(), PROFILE)
    assert ei.value.code is ErrorCode.INFERENCE_FAILED
    assert ei.value.exit_code == 4
    assert ei.value.details["exception_type"] == "RuntimeError"


def test_session_run_oom_maps_to_gpu_oom_code(monkeypatch, tmp_path):
    def _oom(feeds):
        raise MemoryError("oom (simulated)")

    backend = make_backend(monkeypatch, FakeSession(_oom), tmp_path)
    with pytest.raises(MemoryExhaustedError) as ei:
        backend.predict(ui_image(), PROFILE)
    assert ei.value.code is ErrorCode.GPU_OOM
    assert ei.value.exit_code == 4


# ---------------------------------------------------------------------------
# Real-weight tests (skipif-guarded; never fabricate outputs)
# ---------------------------------------------------------------------------


@needs_weights
def test_real_weights_end_to_end_and_metadata():
    backend = registry.get_backend(PROFILE)
    assert isinstance(backend, FoveacastOnnxBackend)
    image = ui_image(60, 80, seed=11)
    result = backend.predict(image, PROFILE)
    assert result.semantics == "probability_density"
    assert result.array.dtype == np.float64
    assert result.array.shape == (240, 320)
    assert abs(float(result.array.sum()) - 1.0) < 1e-9
    assert float(result.array.min()) >= 0.0
    # U8 实测元数据（2026-09-11）：动态 batch + 固定 3×240×320 / 1×240×320。
    # 适配层不依赖这些名称；此处断言仅记录实测证据。
    meta = backend.session_metadata
    assert meta["input"]["name"] == "input"
    assert meta["output"]["name"] == "output"
    assert meta["input"]["shape"] == ["batch_size", 3, 240, 320]
    assert meta["output"]["shape"] == ["batch_size", 1, 240, 320]
    assert meta["input"]["type"] == "tensor(float)"
    # 确定性：同输入两次推理逐位一致
    again = backend.predict(image, PROFILE)
    assert np.array_equal(result.array, again.array)


@needs_weights
def test_real_weights_raw_output_is_minmax_relative():
    backend = registry.get_backend(PROFILE)
    image = ui_image(60, 80, seed=13)
    backend.predict(image, PROFILE)  # ensure session
    tensor = backend._preprocess(image, 240, 320, PROFILE.preprocessing)  # noqa: SLF001
    raw = backend._session.run(  # noqa: SLF001
        [backend._output_meta["name"]], {backend._input_meta["name"]: tensor}  # noqa: SLF001
    )[0]
    arr = np.asarray(raw)
    assert arr.shape == (1, 1, 240, 320)
    assert arr.dtype == np.float32
    # 原生输出为逐图 min-max [0,1] 相对量（非概率）：min≈0, max≈1, sum≫1
    assert float(arr.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(arr.max()) == pytest.approx(1.0, abs=1e-6)
    assert float(arr.sum()) > 1.0


@needs_weights
def test_real_describe_weights_digest_matches_release_page():
    info = FoveacastOnnxBackend().describe()
    assert info.weights[0].sha256 == RELEASE_SHA256
    assert info.weights[0].size_bytes == registry.FOVEACAST_3S_WEIGHT.size_bytes
