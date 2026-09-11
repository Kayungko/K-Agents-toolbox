"""contracts.backend 冻结签名测试（data-contract §3 字段逐一对应）。

覆盖：BackendInfo/WeightRef/ResolvedProfile/PredictionResult 合法构造与逐项拒绝、
Backend Protocol 结构（describe/predict）、runtime_checkable 行为。
"""

from __future__ import annotations

import numpy as np
import pytest

from ui_attention.contracts import (
    CAPABILITIES,
    CLEARED_FOR,
    DEVICES,
    NATIVE_SEMANTICS,
    Backend,
    BackendInfo,
    PredictionResult,
    ResolvedProfile,
    WeightRef,
)
from ui_attention.errors import ErrorCode, UiAttentionError

SHA = "842a23f9" + "0" * 48 + "585ef76e"
assert len(SHA) == 64

WEIGHT = WeightRef(
    name="foveacast-v3-3s-fp16.onnx", source_url="https://example.invalid/releases", sha256=SHA, size_bytes=56_500_000
)

INFO = BackendInfo(
    backend_id="foveacast-onnx-3s",
    version="0.1.0",
    capabilities=("spatial_density",),
    native_semantics="probability_density",
    device_requirements={"device": "cpu", "min_free_vram_mb": None},
    license_status={
        "code": "MIT",
        "weights": "MIT(UEyes CC-BY-4.0 署名)",
        "gaps": ["G3", "G4"],
        "cleared_for": "internal-eval",
    },
    weights=(WEIGHT,),
)

PROFILE = ResolvedProfile(
    profile_name="foveacast-onnx-3s-v1",
    backend_id="foveacast-onnx-3s",
    backend_version="0.1.0",
    preprocessing={"target_hw": [240, 320], "value_range": [0, 255], "mean_subtraction": "in-graph"},
    centerbias=None,
    viewing_conditions={"window": "3s", "assumption": "实验假设：静态 3s 观看窗口"},
    config_hash="c" * 64,
    weights=(WEIGHT,),
)


def make_result(semantics="probability_density", shape=(240, 320), dtype=np.float64, **kw):
    arr = np.full(shape, 1.0 / (shape[0] * shape[1]), dtype=dtype)
    defaults = dict(
        semantics=semantics,
        array=arr,
        shape_mapping={"original_shape": [480, 640], "inference_shape": list(shape), "method": "bilinear-aspect"},
        runtime={"device": "cpu", "precision": "fp32", "elapsed_ms": 123.4, "peak_mem_mb": 512.0},
        vendor_metrics=None,
        limitations=("输出为逐图 min-max 相对量，跨图不可比", "输入分辨率上限 240×320", "训练分布不含游戏 UI"),
    )
    defaults.update(kw)
    return PredictionResult(**defaults)


# ---------------------------------------------------------------------------
# 冻结枚举本身
# ---------------------------------------------------------------------------


def test_frozen_enumerations():
    assert CAPABILITIES == ("spatial_density", "vendor_metrics", "scanpath")
    assert NATIVE_SEMANTICS == ("log_density", "probability_density", "vendor_metrics")
    assert DEVICES == ("cpu", "cuda")
    assert CLEARED_FOR == ("internal-eval", "packaged")


# ---------------------------------------------------------------------------
# WeightRef / BackendInfo
# ---------------------------------------------------------------------------


def test_backend_info_valid():
    INFO.validate()
    d = INFO.to_dict()
    assert BackendInfo.from_dict(d).to_dict() == d


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend_id": ""},
        {"version": ""},
        {"capabilities": ("spatial_density", "mind_reading")},
        {"capabilities": ["spatial_density"]},  # 冻结口径要求 tuple
        {"native_semantics": "attention_score"},
        {"device_requirements": {"device": "tpu"}},
        {"device_requirements": {"device": "cpu", "min_free_vram_mb": "lots"}},
        {"license_status": {"code": "MIT", "weights": "MIT", "gaps": [], "cleared_for": "public"}},
        {"license_status": {"code": "", "weights": "MIT", "gaps": [], "cleared_for": "packaged"}},
        {"license_status": {"code": "MIT", "weights": "MIT", "gaps": "G1", "cleared_for": "packaged"}},
        {"weights": ("not-a-weightref",)},
    ],
)
def test_backend_info_rejects_invalid(kwargs):
    bad = BackendInfo(**{**INFO.__dict__, **kwargs})
    with pytest.raises(UiAttentionError) as exc:
        bad.validate()
    assert exc.value.code is ErrorCode.MODEL_NOT_READY


def test_weight_ref_rejects_bad_sha_and_size():
    with pytest.raises(UiAttentionError):
        WeightRef(name="w", source_url="u", sha256="abc", size_bytes=1).validate()
    with pytest.raises(UiAttentionError):
        WeightRef(name="w", source_url="u", sha256=SHA, size_bytes=-1).validate()
    with pytest.raises(UiAttentionError):
        WeightRef.from_dict({"name": "w", "source_url": "u", "sha256": SHA, "size_bytes": 1, "extra": 2})


# ---------------------------------------------------------------------------
# ResolvedProfile（登记名规范 + 合成哈希）
# ---------------------------------------------------------------------------


def test_resolved_profile_valid_roundtrip():
    PROFILE.validate()
    assert PROFILE.centerbias is None
    assert ResolvedProfile.from_dict(PROFILE.to_dict()).config_hash == "c" * 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile_name": "local-static-v1-rc"},  # 不符合 <backend>-<variant>-v<N>
        {"profile_name": "Foveacast-v1"},  # 大写拒绝
        {"profile_name": "foveacast"},  # 缺版本号
        {"config_hash": "short"},
        {"backend_id": ""},
        {"centerbias": "mit1003"},  # 必须是 dict|None
        {"preprocessing": None},
    ],
)
def test_resolved_profile_rejects_invalid(kwargs):
    bad = ResolvedProfile(**{**PROFILE.__dict__, **kwargs})
    with pytest.raises(UiAttentionError) as exc:
        bad.validate()
    assert exc.value.code is ErrorCode.PROFILE_NOT_REGISTERED


# ---------------------------------------------------------------------------
# PredictionResult（冻结字段表）
# ---------------------------------------------------------------------------


def test_prediction_result_valid_probability_density():
    r = make_result()
    r.validate()
    assert r.array.dtype == np.float64
    assert r.vendor_metrics is None
    assert len(r.limitations) >= 3


def test_prediction_result_log_density_allowed():
    r = make_result(semantics="log_density", array=np.full((240, 320), -np.log(240 * 320)))
    r.validate()


def test_prediction_result_rejects_8bit_array():
    with pytest.raises(UiAttentionError) as exc:
        make_result(dtype=np.uint8).validate()
    # 数组本体无效 = 输出概率无效（退出码 5），与执行失败（4）区分
    assert exc.value.code is ErrorCode.INVALID_DENSITY
    assert exc.value.exit_code == 5
    assert any("float64" in e for e in exc.value.details["errors"])


def test_prediction_result_rejects_bad_shape_and_nan():
    with pytest.raises(UiAttentionError):
        make_result(shape=(3, 240, 320)).validate()  # 空间图必须 (h, w)
    nan_arr = np.full((4, 4), 0.0625)
    nan_arr[0, 0] = np.nan
    with pytest.raises(UiAttentionError):
        make_result(array=nan_arr, shape_mapping={"original_shape": [8, 8], "inference_shape": [4, 4]}).validate()


def test_prediction_result_shape_mapping_consistency():
    with pytest.raises(UiAttentionError) as exc:
        make_result(shape_mapping={"original_shape": [480, 640], "inference_shape": [100, 100]}).validate()
    assert any("inference_shape" in e for e in exc.value.details["errors"])
    with pytest.raises(UiAttentionError):
        make_result(shape_mapping={"inference_shape": [240, 320]}).validate()  # 缺 original_shape


def test_prediction_result_runtime_frozen_keys():
    rt = {"device": "cpu", "precision": "fp32", "elapsed_ms": 1.0}  # 缺 peak_mem_mb
    with pytest.raises(UiAttentionError) as exc:
        make_result(runtime=rt).validate()
    assert any("peak_mem_mb" in e for e in exc.value.details["errors"])


def test_prediction_result_vendor_metrics_rules():
    # 非 vendor_metrics 语义时 vendor_metrics 必须为 None
    with pytest.raises(UiAttentionError):
        make_result(vendor_metrics={"attention_score": 0.9}).validate()
    # vendor_metrics 语义时必须是 dict，且 array 仍按冻结口径校验
    r = make_result(semantics="vendor_metrics", vendor_metrics={"vendor_score_name": "ai", "semantics": "vendor"})
    r.validate()
    with pytest.raises(UiAttentionError):
        make_result(semantics="vendor_metrics", vendor_metrics=None).validate()


def test_prediction_result_requires_limitations():
    with pytest.raises(UiAttentionError) as exc:
        make_result(limitations=()).validate()
    assert any("limitations" in e for e in exc.value.details["errors"])


# ---------------------------------------------------------------------------
# Backend Protocol
# ---------------------------------------------------------------------------


def test_backend_protocol_structural_conformance():
    class DummyBackend:
        def describe(self) -> BackendInfo:
            return INFO

        def predict(self, image: np.ndarray, resolved_profile: ResolvedProfile) -> PredictionResult:
            assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
            return make_result()

    backend = DummyBackend()
    assert isinstance(backend, Backend)  # runtime_checkable 方法存在性检查
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    result = backend.predict(img, PROFILE)
    result.validate()
    assert backend.describe().backend_id == "foveacast-onnx-3s"


def test_object_without_predict_is_not_backend():
    class NotABackend:
        def describe(self):
            return None

    assert not isinstance(NotABackend(), Backend)
