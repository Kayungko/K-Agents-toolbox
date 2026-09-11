"""foveacast ONNX backend adapter — UI 微调 MSI-Net（C2 scope）.

Implements the frozen G0 backend interface (``docs/data-contract.md`` §3,
C1 类型定义于 ``ui_attention/contracts/backend.py``)::

    describe() -> BackendInfo
    predict(image, resolved_profile) -> PredictionResult

Model I/O contract (technical-design §3 冻结口径, R1 已验证 msinet.py 源码):

- 输入 ``(N, 3, 240, 320)`` float32，RGB，值域 ``[0, 255]``；均值减除
  （Kroner 特例均值 ``(103.939, 116.779, 123.68)`` 按 RGB 序）**在 ONNX 图内
  完成**，调用方与适配层都不得预处理均值。
- 输出 ``(N, 1, 240, 320)`` float32，**逐图 min-max 归一化到 [0,1] 的相对
  显著量**——不是概率密度也不是 log density。适配层校验（有限、非负、形状
  匹配）后做 **sum=1 重归一化**，以 ``probability_density`` 语义进入统计
  管线（float64，不量化）。
- InferenceSession 初始化时**运行时读取实际输入/输出元数据**（名称、形状、
  dtype），不硬编码张量名（消解 runtime-feasibility U8；上游 README 示例名
  为 ``"input"``/``"output"``，仅作旁证不作依赖）。
- ``runtime`` 字段全部实测：device=cpu；precision="fp16" 指权重工件量化精度
  （FP16；CPU EP 实际计算精度由 ORT 决定，通常升 FP32 计算，见
  runtime-measurements.json 注记）；elapsed_ms 为 predict 端到端耗时
  （预处理+推理+重归一化）；peak_mem_mb 为进程峰值工作集（Windows
  PeakWorkingSetSize，见 runtime_probe.py 语义注记）。
- 失败路径全部结构化（C1 错误码）：权重缺失/哈希不符 → MODEL_NOT_READY
  （退出码 3）；profile 未登记/运行时改写/后端不匹配 → PROFILE_NOT_REGISTERED
  （3）；image 违约 → INVALID_IMAGE（2）；session.run 失败 → INFERENCE_FAILED
  （4）；内存耗尽 → GPU_OOM（4）；输出无效（NaN/负值/形状/退化求和/重归一化
  超差）→ INVALID_DENSITY（5）。**任何失败都不产生热图，绝不合成假输出。**

署名义务（CC-BY-4.0 传递，UEyes 数据集 → foveacast 微调权重 → 本项目下游；
与 registry.ATTRIBUTION / LICENSE_RECORD 同源记录）：

1. Jiang, Y., Leiva, L. A., Rezazadegan Tavakoli, H., Houssel, P. R. B.,
   Kylmälä, J., & Oulasvirta, A. (2023). UEyes: Understanding Visual Saliency
   across User Interface Types. In Proceedings of the 2023 CHI Conference on
   Human Factors in Computing Systems (CHI '23), Article 285, 1-21.
   DOI: 10.1145/3544548.3581096. Dataset: Zenodo record 8010312, CC-BY-4.0.
2. Kroner, A., Senden, M., Driessens, K., & Goebel, R. (2020). Contextual
   Encoder-Decoder Network for Visual Saliency Prediction. Neural Networks,
   129, 261-270. DOI: 10.1016/j.neunet.2020.05.004. (MSI-Net architecture, MIT)

模型代码/权重许可：MIT（foveacast-training, (c) 2026 Ken Hawkins）+ UEyes
署名传递；商用链残留缺口 G3/G4/G8 未关闭 → ``cleared_for="internal-eval"``。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ui_attention.contracts.backend import (
    BackendInfo,
    PredictionResult,
    ResolvedProfile,
)

from . import registry
from .errors import (
    ImageRejectedError,
    InferenceFailureError,
    InvalidDensityError,
    MemoryExhaustedError,
    ProfileRejectedError,
    WeightNotReadyError,
)
from .runtime_probe import peak_working_set_mb
from .weights import default_cache_dir, verify_cached_weight

__all__ = ["FoveacastOnnxBackend"]

_SUM_TOLERANCE = 1e-9


class FoveacastOnnxBackend:
    """Adapter for the foveacast v0.2.0 FP16 ONNX artifacts (CPU only).

    同一适配层服务三个观看窗口变体（1s/3s/7s，同架构同预处理，仅权重与
    profile 观看假设不同）：``backend_id`` 由构造参数指定（默认 3s 主用），
    ``registry.get_backend`` 按 profile 传入 ``foveacast-onnx-{1s,3s,7s}``。
    """

    BACKEND_ID = registry.FOVEACAST_3S_BACKEND_ID  # default (3s 主用窗口)
    VERSION = registry.FOVEACAST_3S_BACKEND_VERSION
    SEMANTICS = "probability_density"

    #: 后端已知限制（冻结必录项：相对量语义 / 分辨率上限 / 训练分布边界，
    #: 外加作者指标未复现声明）。
    LIMITATIONS: tuple[str, ...] = (
        "原始输出为逐图 min-max 相对量，跨图绝对强度不可比",
        "输入分辨率上限 240×320，小元素可能无法解析",
        "训练分布为 2020-2022 西方语言桌面/移动 UI，不含游戏 UI，效果未验证",
        "作者公布指标未经本项目复现",
    )

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        backend_id: str = BACKEND_ID,
    ) -> None:
        if backend_id not in registry.FOVEACAST_BACKEND_IDS:
            raise ProfileRejectedError(
                f"backend_id {backend_id!r} is not a registered foveacast window variant",
                {"backend_id": backend_id, "supported": list(registry.FOVEACAST_BACKEND_IDS)},
                reason="backend_mismatch",
            )
        self.backend_id = backend_id
        self._cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self._providers = list(providers)
        self._session: Any = None
        self._input_meta: dict[str, Any] | None = None
        self._output_meta: dict[str, Any] | None = None
        self._weight_path: Path | None = None
        self._cold_load_ms: float | None = None

    # ------------------------------------------------------------------
    # Frozen interface
    # ------------------------------------------------------------------

    def describe(self) -> BackendInfo:
        """Backend self-description built from the registered profile (read-only)."""
        profile = registry.profile_for_backend(self.backend_id)
        registration = registry.registration_for(profile.profile_name)
        info = BackendInfo(
            backend_id=self.backend_id,
            version=self.VERSION,
            capabilities=("spatial_density",),
            native_semantics=self.SEMANTICS,
            device_requirements={"device": "cpu", "min_free_vram_mb": None},
            license_status=dict(registration.license_status),
            weights=tuple(profile.weights),
        )
        info.validate()
        return info

    def predict(self, image: np.ndarray, resolved_profile: ResolvedProfile) -> PredictionResult:
        """Run the frozen predict() contract on an orientation-processed RGB image.

        ``image``: RGB uint8 ``(H, W, 3)`` 原图（调用方不得预处理）；模型侧
        缩放按 ``resolved_profile`` 内部执行并写入 ``shape_mapping``。
        """
        self._check_profile(resolved_profile)
        _validate_image(image)
        height, width = int(image.shape[0]), int(image.shape[1])

        preprocessing = resolved_profile.preprocessing
        target_h = int(preprocessing["target_height"])
        target_w = int(preprocessing["target_width"])

        self._ensure_session(resolved_profile)
        assert self._session is not None and self._input_meta is not None
        assert self._output_meta is not None

        started = time.perf_counter()
        tensor = self._preprocess(image, target_h, target_w, preprocessing)
        try:
            outputs = self._session.run(
                [self._output_meta["name"]], {self._input_meta["name"]: tensor}
            )
        except MemoryError as exc:
            raise MemoryExhaustedError(
                "memory exhausted during onnxruntime session.run",
                {"device": "cpu", "input_shape": list(tensor.shape)},
                reason="cpu_oom",
            ) from exc
        except Exception as exc:  # onnxruntime.OrtError and friends
            raise InferenceFailureError(
                f"onnxruntime session.run failed: {type(exc).__name__}",
                {
                    "exception_type": type(exc).__name__,
                    "device": "cpu",
                    "input_shape": list(tensor.shape),
                },
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        raw = outputs[0] if outputs else None
        density = _validate_and_renormalize(raw, expected_spatial=(target_h, target_w))

        result = PredictionResult(
            semantics=self.SEMANTICS,
            array=density,
            shape_mapping=_build_shape_mapping(height, width, target_h, target_w, preprocessing),
            runtime={
                "device": "cpu",
                "precision": "fp16",
                "elapsed_ms": round(elapsed_ms, 3),
                "peak_mem_mb": peak_working_set_mb(),
            },
            vendor_metrics=None,
            limitations=self.LIMITATIONS,
        )
        result.validate()  # C1 frozen validation before anything consumes it
        return result

    # ------------------------------------------------------------------
    # Measurement surface (consumed by measure_runtime.py; not part of the
    # frozen interface)
    # ------------------------------------------------------------------

    @property
    def session_metadata(self) -> dict[str, Any] | None:
        """Runtime-read ONNX I/O metadata (U8 evidence); None before first load."""
        if self._input_meta is None or self._output_meta is None:
            return None
        return {
            "input": dict(self._input_meta),
            "output": dict(self._output_meta),
            "providers": list(self._providers),
        }

    @property
    def cold_load_ms(self) -> float | None:
        """Cold InferenceSession creation time in ms (fresh process only once)."""
        return self._cold_load_ms

    @property
    def weight_path(self) -> Path | None:
        return self._weight_path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_profile(self, resolved_profile: ResolvedProfile) -> None:
        if not isinstance(resolved_profile, ResolvedProfile):
            raise ProfileRejectedError(
                "resolved_profile must be the frozen contracts ResolvedProfile "
                f"produced by registry.resolve_profile, got {type(resolved_profile).__name__}",
                {"got_type": type(resolved_profile).__name__},
                reason="profile_type",
            )
        registry.verify_profile_integrity(resolved_profile)
        if resolved_profile.backend_id != self.backend_id:
            raise ProfileRejectedError(
                f"profile {resolved_profile.profile_name!r} targets backend "
                f"{resolved_profile.backend_id!r}, not {self.backend_id!r}",
                {
                    "profile_backend_id": resolved_profile.backend_id,
                    "expected_backend_id": self.backend_id,
                },
                reason="backend_mismatch",
            )

    def _ensure_session(self, profile: ResolvedProfile) -> None:
        """Lazily create the ORT session; read metadata at init (never hardcoded)."""
        if self._session is not None:
            return
        if not profile.weights:
            raise WeightNotReadyError(
                f"profile {profile.profile_name!r} carries no registered weights",
                {"profile_name": profile.profile_name},
                reason="no_weights_registered",
            )
        ref = profile.weights[0]  # foveacast 3s: single-file artifact
        verified = verify_cached_weight(ref, self._cache_dir)

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise WeightNotReadyError(
                f"onnxruntime not importable: {type(exc).__name__}",
                {"hint": "install extras [onnx] (onnxruntime==1.30.0 pinned)"},
                reason="dependency_missing",
            ) from exc

        started = time.perf_counter()
        try:
            session = ort.InferenceSession(str(verified.path), providers=self._providers)
        except MemoryError as exc:
            raise MemoryExhaustedError(
                "memory exhausted while creating InferenceSession",
                {"weight": ref.name},
                reason="cpu_oom_session_load",
            ) from exc
        except Exception as exc:  # corrupt/incompatible model file
            raise WeightNotReadyError(
                f"InferenceSession creation failed for {ref.name}: {type(exc).__name__}",
                {"weight": ref.name, "path": str(verified.path)},
                reason="session_load_failed",
            ) from exc
        self._cold_load_ms = round((time.perf_counter() - started) * 1000.0, 3)

        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise WeightNotReadyError(
                f"unexpected ONNX graph arity: {len(inputs)} input(s), {len(outputs)} output(s)",
                {"weight": ref.name, "inputs": len(inputs), "outputs": len(outputs)},
                reason="metadata_unsupported",
            )
        in0, out0 = inputs[0], outputs[0]
        self._input_meta = {"name": in0.name, "shape": list(in0.shape), "type": in0.type}
        self._output_meta = {"name": out0.name, "shape": list(out0.shape), "type": out0.type}
        self._check_metadata(profile)
        self._session = session
        self._weight_path = verified.path

    def _check_metadata(self, profile: ResolvedProfile) -> None:
        """Validate runtime-read metadata against the registered profile."""
        assert self._input_meta is not None and self._output_meta is not None
        preprocessing = profile.preprocessing
        target_h = int(preprocessing["target_height"])
        target_w = int(preprocessing["target_width"])

        in_shape = self._input_meta["shape"]
        out_shape = self._output_meta["shape"]
        problems: list[str] = []
        if self._input_meta["type"] != "tensor(float)":
            problems.append(f"input dtype {self._input_meta['type']!r} != tensor(float)")
        if self._output_meta["type"] != "tensor(float)":
            problems.append(f"output dtype {self._output_meta['type']!r} != tensor(float)")
        if len(in_shape) != 4:
            problems.append(f"input rank {len(in_shape)} != 4 (NCHW)")
        else:
            channels = in_shape[1]
            if isinstance(channels, int) and channels != 3:
                problems.append(f"input channels {channels} != 3")
            for dim, expected, label in ((in_shape[2], target_h, "height"), (in_shape[3], target_w, "width")):
                if isinstance(dim, int) and dim != expected:
                    problems.append(f"static input {label} {dim} != registered {expected}")
        if len(out_shape) != 4:
            problems.append(f"output rank {len(out_shape)} != 4 (N,1,H,W)")
        else:
            for dim, expected, label in (
                (out_shape[2], target_h, "height"),
                (out_shape[3], target_w, "width"),
            ):
                if isinstance(dim, int) and dim != expected:
                    problems.append(f"static output {label} {dim} != registered {expected}")
        if problems:
            raise WeightNotReadyError(
                "ONNX session metadata does not match the registered profile",
                {
                    "problems": problems,
                    "input": dict(self._input_meta),
                    "output": dict(self._output_meta),
                },
                reason="metadata_unsupported",
            )

    def _preprocess(
        self,
        image: np.ndarray,
        target_h: int,
        target_w: int,
        preprocessing: Any,
    ) -> np.ndarray:
        """RGB uint8 (H,W,3) → float32 (1,3,target_h,target_w) in [0,255].

        Resize follows the registered profile (pillow bicubic, direct
        anisotropic to 240×320, no padding). NO mean subtraction — the graph
        does it (Kroner special mean, RGB order).
        """
        filter_name = str(preprocessing.get("resize_filter", "pillow_bicubic"))
        resample = {
            "pillow_bicubic": Image.Resampling.BICUBIC,
            "pillow_bilinear": Image.Resampling.BILINEAR,
        }.get(filter_name)
        if resample is None:
            raise WeightNotReadyError(
                f"registered resize_filter {filter_name!r} is not supported by this adapter",
                {"resize_filter": filter_name},
                reason="metadata_unsupported",
            )
        pil_image = Image.fromarray(image, mode="RGB").resize((target_w, target_h), resample)
        array = np.asarray(pil_image, dtype=np.float32)  # (h, w, 3) in [0, 255]
        tensor = np.ascontiguousarray(array.transpose(2, 0, 1)[np.newaxis])  # (1, 3, h, w)
        return tensor


def _validate_image(image: Any) -> None:
    """Frozen input contract: orientation-processed original, RGB uint8 (H,W,3)."""
    if not isinstance(image, np.ndarray):
        raise ImageRejectedError(
            f"image must be numpy.ndarray, got {type(image).__name__}",
            {"got_type": type(image).__name__},
            reason="not_ndarray",
        )
    if image.dtype != np.uint8:
        raise ImageRejectedError(
            f"image dtype must be uint8 (方向已处理原图), got {image.dtype}",
            {"dtype": str(image.dtype)},
            reason="dtype",
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageRejectedError(
            f"image shape must be (H, W, 3) RGB, got {image.shape}",
            {"shape": list(image.shape)},
            reason="shape",
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ImageRejectedError(
            f"image must be non-empty, got {image.shape}",
            {"shape": list(image.shape)},
            reason="empty",
        )


def _validate_and_renormalize(raw: Any, expected_spatial: tuple[int, int]) -> np.ndarray:
    """Validate model output, then renormalize to sum=1 probability density.

    Checks (all failures → INVALID_DENSITY, exit 5; never fabricate a map):
    single output, float kind, shape (1,1,h,w) (or squeezable (1,h,w)),
    spatial == expected, all finite, all non-negative, positive total mass;
    then float64 sum=1 renormalization verified within 1e-9.
    """
    if raw is None:
        raise InvalidDensityError(
            "model returned no output tensor", {}, reason="missing_output"
        )
    array = np.asarray(raw)
    if array.dtype.kind != "f":
        raise InvalidDensityError(
            f"model output dtype {array.dtype} is not floating point",
            {"dtype": str(array.dtype)},
            reason="dtype",
        )
    if array.ndim == 4 and array.shape[0] == 1 and array.shape[1] == 1:
        spatial = array[0, 0]
    elif array.ndim == 3 and array.shape[0] == 1:
        spatial = array[0]
    else:
        raise InvalidDensityError(
            f"model output shape {array.shape} does not match expected (1, 1, "
            f"{expected_spatial[0]}, {expected_spatial[1]}) (batch=1, single channel)",
            {"shape": list(array.shape), "expected_spatial": list(expected_spatial)},
            reason="shape",
        )
    if tuple(spatial.shape) != tuple(expected_spatial):
        raise InvalidDensityError(
            f"model output spatial shape {spatial.shape} != expected {expected_spatial}",
            {"shape": list(spatial.shape), "expected_spatial": list(expected_spatial)},
            reason="shape",
        )
    if not np.all(np.isfinite(spatial)):
        raise InvalidDensityError(
            "model output contains NaN/Inf",
            {"shape": list(spatial.shape)},
            reason="non_finite",
        )
    minimum = float(spatial.min())
    if minimum < 0.0:
        raise InvalidDensityError(
            f"model output contains negative values (min={minimum!r})",
            {"min": minimum},
            reason="negative_values",
        )
    density = spatial.astype(np.float64, copy=True)
    total = float(density.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise InvalidDensityError(
            f"model output has degenerate total mass (sum={total!r}); cannot renormalize",
            {"sum": total},
            reason="degenerate_sum",
        )
    density /= total
    deviation = abs(float(density.sum()) - 1.0)
    if deviation > _SUM_TOLERANCE:
        raise InvalidDensityError(
            f"renormalization failed: |sum-1| = {deviation!r} > {_SUM_TOLERANCE}",
            {"sum_deviation": deviation},
            reason="renormalization_failed",
        )
    return density


def _build_shape_mapping(
    height: int,
    width: int,
    target_h: int,
    target_w: int,
    preprocessing: Any,
) -> dict[str, Any]:
    """Record original/inference sizes, scale & padding params, inverse method.

    对齐 technical-design §4.4-4.5：直接各向异性缩放到登记尺寸（无填充）；
    逆变换 = 概率图重采样回原图尺寸后**再次归一化**（由 C1 统计/报告侧按本
    记录执行，本适配层返回推理尺寸概率图，sum=1）。
    """
    return {
        "original_shape": [height, width],
        "inference_shape": [target_h, target_w],
        "method": "direct_anisotropic_resize",
        "resize_filter": str(preprocessing.get("resize_filter", "pillow_bicubic")),
        "padding": None,
        "scale_yx": [target_h / height, target_w / width],
        "aspect_ratio_original": width / height,
        "aspect_ratio_inference": target_w / target_h,
        "coordinate_mapping": {
            "forward": f"x_inf = x_orig * ({target_w}/{width}); y_inf = y_orig * ({target_h}/{height})",
            "note": "半像素中心约定忽略；缩放到 240×320 的量化误差 << 1 推理像素",
        },
        "inverse": {
            "method": "resize_density_to_original_then_renormalize",
            "target_shape": [height, width],
            "filter": "bilinear",
            "renormalize": "sum_to_1",
            "reference": "technical-design §4.5: 需要返回原图尺寸时，对概率图映射后再次归一化",
        },
    }
