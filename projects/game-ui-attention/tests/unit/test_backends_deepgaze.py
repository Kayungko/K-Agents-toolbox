"""Unit tests for backends/deepgaze_iie.py (C2, G1' internal-eval line).

适配层逻辑（log_density 直通、centerbias README 口径、长边 1024 等比降采样、
无效输出拒绝、缺失权重结构化错误、设备/许可门禁）用假模型（monkeypatch）
测试；真权重测试（U1 strict-load 关卡 + 真实推理）带 skipif 守卫。
DeepGaze 代码/权重许可未闭合（G1/G2/G5）——测试与实现仅限内部研究评估。
"""

import dataclasses

import numpy as np
import pytest

from ui_attention.backends import registry
from ui_attention.backends.deepgaze_iie import (
    DeepGazeIIEBackend,
    default_deepgaze_cache_dir,
)
from ui_attention.backends.errors import (
    ImageRejectedError,
    InvalidDensityError,
    ProfileRejectedError,
    WeightNotReadyError,
)
from ui_attention.contracts.backend import PredictionResult
from ui_attention.errors import ErrorCode

torch = pytest.importorskip("torch")

PROFILE = registry.resolve_profile(registry.DEEPGAZE_IIE_PROFILE)
CACHE = default_deepgaze_cache_dir()
HAS_WEIGHTS = all((CACHE / w.name).is_file() for w in PROFILE.weights)
needs_weights = pytest.mark.skipif(
    not HAS_WEIGHTS, reason="deepgaze weights not cached (internal-eval line)"
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTorchModel:
    def __init__(self, output_fn):
        self._fn = output_fn
        self.calls: list[tuple] = []

    def __call__(self, image_tensor, centerbias_tensor):
        self.calls.append((image_tensor, centerbias_tensor))
        return self._fn(image_tensor, centerbias_tensor)


def zeros_like_output(image_tensor, centerbias_tensor):
    b, _, h, w = image_tensor.shape
    return torch.full((b, 1, h, w), -12.0, dtype=torch.float32)


def make_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    output_fn=zeros_like_output,
    template_shape: tuple[int, int] = (64, 64),
) -> tuple[DeepGazeIIEBackend, FakeTorchModel, "registry.ResolvedProfile"]:
    """Backend with model construction and weight verification faked out.

    The registered centerbias template_shape is rewritten to the small fake
    template via a config-hash-consistent profile copy (also exercises the
    integrity recompute path).
    """
    cb_path = tmp_path / "fake_centerbias.npy"
    template = np.log(np.ones(template_shape, dtype=np.float64) / np.prod(template_shape))
    np.save(cb_path, template)

    centerbias = {**PROFILE.centerbias, "template_shape": list(template_shape)}
    profile = dataclasses.replace(
        PROFILE,
        centerbias=centerbias,
        config_hash=registry.compute_config_hash(
            backend_id=PROFILE.backend_id,
            backend_version=PROFILE.backend_version,
            weights=PROFILE.weights,
            preprocessing=PROFILE.preprocessing,
            centerbias=centerbias,
            viewing_conditions=PROFILE.viewing_conditions,
        ),
    )

    model = FakeTorchModel(output_fn)
    backend = DeepGazeIIEBackend(cache_dir=tmp_path)
    monkeypatch.setattr(backend, "_ensure_model", lambda prof: None)
    monkeypatch.setattr(backend, "_model", model)
    monkeypatch.setattr(backend, "_torch", torch)
    monkeypatch.setattr(
        backend,
        "_weight_paths",
        lambda prof: {"model": tmp_path / "fake.pth", "centerbias": cb_path},
    )
    return backend, model, profile


def ui_image(h: int = 120, w: int = 200, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Adapter semantics
# ---------------------------------------------------------------------------


def test_predict_returns_native_log_density_unrenormalized(monkeypatch, tmp_path):
    backend, _, profile = make_fake_backend(monkeypatch, tmp_path)
    result = backend.predict(ui_image(120, 200), profile)
    assert isinstance(result, PredictionResult)
    assert result.semantics == "log_density"  # 原生语义, 适配层不做 exp/归一化
    assert result.array.dtype == np.float64
    assert result.array.shape == (120, 200)
    assert float(result.array.min()) == pytest.approx(-12.0)
    assert float(result.array.max()) == pytest.approx(-12.0)  # 未被重归一化
    assert result.vendor_metrics is None
    result.validate()  # C1 frozen validation (log_density 允许负值, 要求有限)


def test_runtime_and_limitations(monkeypatch, tmp_path):
    backend, _, profile = make_fake_backend(monkeypatch, tmp_path)
    result = backend.predict(ui_image(), profile)
    assert set(result.runtime) == {"device", "precision", "elapsed_ms", "peak_mem_mb"}
    assert result.runtime["device"] == "cpu"
    assert result.runtime["precision"] == "fp32"
    joined = "\n".join(result.limitations)
    assert "MIT1003" in joined and "35 px/dva" in joined  # 观看条件实验假设
    assert "G1/G2/G5" in joined and "内部研究评估" in joined  # 许可门禁
    assert "未经本项目复现" in joined


def test_downscale_long_side_only(monkeypatch, tmp_path):
    backend, model, profile = make_fake_backend(monkeypatch, tmp_path)
    result = backend.predict(ui_image(1080, 1920), profile)
    image_tensor, centerbias_tensor = model.calls[0]
    assert tuple(image_tensor.shape) == (1, 3, 576, 1024)  # 长边 1024 等比
    assert image_tensor.dtype == torch.float32
    assert float(image_tensor.min()) >= 0.0 and float(image_tensor.max()) <= 255.0
    assert result.array.shape == (576, 1024)
    assert result.shape_mapping["original_shape"] == [1080, 1920]
    assert result.shape_mapping["inference_shape"] == [576, 1024]
    assert result.shape_mapping["method"] == "aspect_preserving_downscale_only"
    assert result.shape_mapping["scale"] == pytest.approx(1024 / 1920)
    assert result.shape_mapping["inverse"]["target_shape"] == [1080, 1920]


def test_small_image_not_upscaled(monkeypatch, tmp_path):
    backend, model, profile = make_fake_backend(monkeypatch, tmp_path)
    result = backend.predict(ui_image(120, 200), profile)
    image_tensor, _ = model.calls[0]
    assert tuple(image_tensor.shape) == (1, 3, 120, 200)  # 只降不升
    assert result.shape_mapping["scale"] == 1.0


def test_centerbias_follows_readme_recipe(monkeypatch, tmp_path):
    backend, model, profile = make_fake_backend(monkeypatch, tmp_path)
    backend.predict(ui_image(1080, 1920), profile)
    _, centerbias_tensor = model.calls[0]
    assert tuple(centerbias_tensor.shape) == (1, 576, 1024)
    assert centerbias_tensor.dtype == torch.float32
    # logsumexp 重归一：log density 全图 logsumexp == 0（README 官方口径）
    lse = float(torch.logsumexp(centerbias_tensor.double(), dim=(1, 2))[0])
    assert lse == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("image", "reason"),
    [
        (None, "not_ndarray"),
        (np.zeros((10, 10, 3), dtype=np.float32), "dtype"),
        (np.zeros((10, 10, 4), dtype=np.uint8), "shape"),
        (np.zeros((10, 10), dtype=np.uint8), "shape"),
        (np.zeros((0, 5, 3), dtype=np.uint8), "empty"),
    ],
)
def test_invalid_image_structured(tmp_path, image, reason):
    backend = DeepGazeIIEBackend(cache_dir=tmp_path)
    with pytest.raises(ImageRejectedError) as ei:
        backend.predict(image, PROFILE)
    assert ei.value.code is ErrorCode.INVALID_IMAGE
    assert ei.value.exit_code == 2
    assert ei.value.details["reason"] == reason


def test_profile_type_rejected(tmp_path):
    backend = DeepGazeIIEBackend(cache_dir=tmp_path)
    with pytest.raises(ProfileRejectedError) as ei:
        backend.predict(ui_image(), {"profile": 1})  # type: ignore[arg-type]
    assert ei.value.details["reason"] == "profile_type"
    assert ei.value.code is ErrorCode.PROFILE_NOT_REGISTERED


def test_profile_tampered_rejected(tmp_path):
    tampered = dataclasses.replace(
        PROFILE, preprocessing={**PROFILE.preprocessing, "target_long_side": 512}
    )
    backend = DeepGazeIIEBackend(cache_dir=tmp_path)
    with pytest.raises(ProfileRejectedError) as ei:
        backend.predict(ui_image(), tampered)
    assert ei.value.details["reason"] == "profile_tampered"


def test_foveacast_profile_rejected_by_deepgaze_backend(tmp_path):
    foveacast_profile = registry.resolve_profile(registry.FOVEACAST_3S_PROFILE)
    backend = DeepGazeIIEBackend(cache_dir=tmp_path)
    with pytest.raises(ProfileRejectedError) as ei:
        backend.predict(ui_image(), foveacast_profile)
    assert ei.value.details["reason"] == "backend_mismatch"


def test_profile_without_centerbias_rejected(tmp_path):
    bare_hash = registry.compute_config_hash(
        backend_id=PROFILE.backend_id,
        backend_version=PROFILE.backend_version,
        weights=PROFILE.weights,
        preprocessing=PROFILE.preprocessing,
        centerbias=None,
        viewing_conditions=PROFILE.viewing_conditions,
    )
    bare = dataclasses.replace(PROFILE, centerbias=None, config_hash=bare_hash)
    backend = DeepGazeIIEBackend(cache_dir=tmp_path)
    with pytest.raises(ProfileRejectedError) as ei:
        backend.predict(ui_image(), bare)
    assert ei.value.details["reason"] == "centerbias_missing"


@pytest.mark.parametrize(
    ("output_fn", "reason"),
    [
        (
            lambda img, cb: torch.full(
                (img.shape[0], 1, img.shape[2], img.shape[3]), float("nan")
            ),
            "non_finite",
        ),
        (
            lambda img, cb: torch.zeros((img.shape[0], 1, 7, 9), dtype=torch.float32),
            "shape",
        ),
    ],
)
def test_invalid_output_structured(monkeypatch, tmp_path, output_fn, reason):
    backend, _, profile = make_fake_backend(monkeypatch, tmp_path, output_fn=output_fn)
    with pytest.raises(InvalidDensityError) as ei:
        backend.predict(ui_image(), profile)
    assert ei.value.code is ErrorCode.INVALID_DENSITY
    assert ei.value.exit_code == 5
    assert ei.value.details["reason"] == reason


def test_missing_weights_structured(tmp_path):
    backend = DeepGazeIIEBackend(cache_dir=tmp_path)  # empty cache, no patching
    with pytest.raises(WeightNotReadyError) as ei:
        backend.predict(ui_image(), PROFILE)
    assert ei.value.code is ErrorCode.MODEL_NOT_READY
    assert ei.value.exit_code == 3
    assert ei.value.details["reason"] == "missing_file"


def test_ctor_rejects_non_cpu_device(tmp_path):
    with pytest.raises(ProfileRejectedError) as ei:
        DeepGazeIIEBackend(cache_dir=tmp_path, device="cuda")
    assert ei.value.details["reason"] == "device_unsupported"


def test_describe_matches_frozen_contract():
    info = DeepGazeIIEBackend().describe()
    info.validate()
    assert info.backend_id == "deepgaze-iie"
    assert info.capabilities == ("spatial_density",)
    assert info.native_semantics == "log_density"
    assert info.device_requirements == {"device": "cpu", "min_free_vram_mb": None}
    ls = info.license_status
    assert ls["gaps"] == ["G1", "G2", "G5"]
    assert ls["cleared_for"] == "internal-eval"
    assert "未闭合" in ls["code"] and "未闭合" in ls["weights"]  # 如实登记
    names = {w.name for w in info.weights}
    assert names == {"deepgaze2e.pth", "centerbias_mit1003.npy"}


# ---------------------------------------------------------------------------
# Registry integration (deepgaze-iie-v1)
# ---------------------------------------------------------------------------


def test_registry_deepgaze_profile_fields():
    assert registry.DEEPGAZE_IIE_PROFILE in registry.list_profiles()
    profile = registry.resolve_profile(registry.DEEPGAZE_IIE_PROFILE)
    profile.validate()
    assert profile.backend_id == "deepgaze-iie"
    assert profile.centerbias is not None
    assert profile.centerbias["name"] == "centerbias_mit1003.npy"
    assert profile.centerbias["sha256"] == registry.DEEPGAZE_CENTERBIAS_WEIGHT.sha256
    assert profile.centerbias["template_shape"] == [1024, 1024]
    assert profile.viewing_conditions["pixel_per_degree"] == 35.0
    assert "实验假设" in profile.viewing_conditions["assumption"]
    assert profile.preprocessing["target_long_side"] == 1024
    assert len(profile.weights) == 2
    record = registry.license_record(registry.DEEPGAZE_IIE_PROFILE)
    assert registry.DEEPGAZE_SOURCE_PIN in str(record["evidence"]["source_pin"])
    assert any("2105.12441" in a for a in record["attribution"])


@needs_weights
def test_get_backend_dispatch_and_u1_strict_load():
    backend = registry.get_backend(PROFILE)
    assert isinstance(backend, DeepGazeIIEBackend)
    backend._ensure_model(PROFILE)  # noqa: SLF001 — U1 关卡：离线构建 + strict=True 加载
    stats = backend.model_stats
    assert stats["parameters"] == 104048808
    assert stats["state_dict_entries"] == 5461
    assert stats["torch_cuda_build"] is None  # U18: CPU-only wheel 实证
    assert backend.cold_load_ms is not None and backend.cold_load_ms > 0


@needs_weights
def test_real_predict_small_image():
    backend = registry.get_backend(PROFILE)
    result = backend.predict(ui_image(240, 320, seed=5), PROFILE)
    assert result.semantics == "log_density"
    assert result.array.shape == (240, 320)
    assert result.array.dtype == np.float64
    assert np.all(np.isfinite(result.array))
    assert float(result.array.max()) > float(result.array.min())  # 非退化
    # C1 转换口径 sanity：exp(L - logsumexp(L)) 归一
    from scipy.special import logsumexp

    prob = np.exp(result.array - logsumexp(result.array))
    assert float(prob.sum()) == pytest.approx(1.0, abs=1e-9)
