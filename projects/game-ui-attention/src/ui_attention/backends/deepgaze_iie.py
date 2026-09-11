"""DeepGaze IIE backend adapter — 通用域对照基线（C2 scope, G1' 批准）.

**许可门禁（先于一切）**：DeepGaze 代码与权重许可未闭合（G1：仓库无
LICENSE、setup.py MIT 字段被注释、Issue #15 open 0 评论；G2：ShapeNetC
骨干权重条款未声明；G5：训练数据集条款未逐一核实）。一级总控 G1' 批准
（2026-09-11）：**仅限内部研究评估，不打包、不分发、不商用**。本模块与
权重仅存在于本地隔离环境（.venv + model-cache，均为 git 忽略目录），代码
来源 pin ``c7db17e2d1d7ea6468ffdee2cfaddf141095dcff``（R1 核查 commit）。

Implements the frozen G0 backend interface (data-contract §3)::

    describe() -> BackendInfo
    predict(image, resolved_profile) -> PredictionResult

模型 I/O（R1 已验证源码 + README 官方用法）：

- 输入 image ``(B,3,H,W)`` float32 [0,255]（``Normalizer`` 在模型内部做
  /255 + ImageNet mean/std，适配层不预处理均值）；显式 centerbias 输入
  ``(B,H,W)`` log density（MIT1003 先验模板 zoom order=0 到推理尺寸后
  ``-= logsumexp`` 重归一，README 官方口径）。
- 输出 ``(B,1,H,W)`` **原生 log density** → 适配层不做 exp/归一化，
  ``semantics="log_density"``（C1 metrics 按 technical-design §7
  ``P = exp(L - logsumexp(L))`` 转换）。
- 观看条件：MIT1003 采集于 35 px/dva、图像长边约 1024。profile
  ``deepgaze-iie-v1`` 冻结「长边>1024 时等比降采样到 1024（只降不升）」
  为**实验假设**（technical-design §5：固定实验配置必须标记为实验假设；
  游戏截图真实观看条件未实测，不得从截图猜测）。

U1 风险处置（老 API × 新 torch，任务书指定策略）：

1. 上游包 ``deepgaze_pytorch/__init__.py`` 急切导入 MSDB → 需要 ``clip``
   （不在批准安装范围）→ 适配层用 ``importlib.util.module_from_spec``
   包桩（不执行 ``__init__``）直接加载 ``deepgaze_pytorch.deepgaze2e``。
2. 四个骨干构造器全部会触发网络下载/老 API（``torch.hub.load
   ('pytorch/vision:v0.6.0', pretrained=True)``、``resnet50(pretrained=)``、
   bitbucket ``model_zoo.load_url``、``EfficientNet.from_pretrained``）→
   适配层在构建前把四个 feature 类替换为**离线等价构造**（当前
   torchvision ``weights=None`` + vendored ``EfficientNet.from_name``，
   保持 nn.Sequential(normalizer, backbone) 结构与 state_dict 键名逐一
   一致），再以 ``DeepGazeIIE(pretrained=False)`` 构建、
   ``load_state_dict(本地 deepgaze2e.pth, strict=True)`` 加载完整权重。
   **strict=True 即 U1 的实证关卡**：任何架构漂移（键/形状不匹配）都会
   结构化报错（``MODEL_NOT_READY``, reason=state_dict_mismatch），绝不
   静默降级为 strict=False。
3. 全部补丁限于本适配层运行时行为，不 fork 上游代码入仓（红线）；不安装
   clip/einops/tensorflow 等 MSDB/训练依赖。

失败路径全部结构化（C1 错误码）：torch/deepgaze 缺失 → MODEL_NOT_READY；
权重缺失/哈希不符 → MODEL_NOT_READY；state_dict 不匹配 → MODEL_NOT_READY；
image 违约 → INVALID_IMAGE；profile 问题 → PROFILE_NOT_REGISTERED；
推理失败/内存耗尽 → INFERENCE_FAILED / GPU_OOM；输出非有限 → INVALID_DENSITY。
不合成假热图。

学术引用（非法律义务——DeepGaze 无许可声明；引用为学术规范）：
Linardos, A., Kümmerer, M., Press, O., & Bethge, M. (2021). Calibrated
prediction in and out-of-domain for state-of-the-art saliency modeling.
arXiv:2105.12441.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from collections import OrderedDict
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
from .weights import cache_root, verify_cached_weight

__all__ = ["DeepGazeIIEBackend"]


def default_deepgaze_cache_dir() -> Path:
    """``<project-root>/model-cache/deepgaze`` (ignored dir, C2-owned)."""
    return cache_root() / "deepgaze"


class DeepGazeIIEBackend:
    """Adapter for DeepGaze IIE (internal research evaluation only)."""

    BACKEND_ID = registry.DEEPGAZE_IIE_BACKEND_ID
    VERSION = registry.DEEPGAZE_IIE_BACKEND_VERSION
    SEMANTICS = "log_density"

    LIMITATIONS: tuple[str, ...] = (
        "训练分布为 MIT1003 自然图像（2010 年前后采集），不含 UI/游戏 UI，效果未验证",
        "观看条件 35 px/dva 为 MIT1003 采集条件的实验假设，游戏截图真实观看条件未实测",
        "显式 centerbias 为 MIT1003 先验模板（版本化登记+哈希），不代表游戏玩家真实注视习惯",
        "代码与权重许可未闭合（G1/G2/G5），仅限内部研究评估，禁止打包分发",
        "作者公布指标未经本项目复现",
        "CPU 上为 4 骨干 × 30 组件集成，延迟显著高于 foveacast ONNX，实测数字为准",
    )

    def __init__(self, cache_dir: Path | str | None = None, device: str = "cpu") -> None:
        if device != "cpu":
            # 路线 A：CPU only（onnxruntime-gpu/CUDA wheel 未批准；torch 为 CPU 构建）
            raise ProfileRejectedError(
                f"deepgaze-iie adapter is CPU-only in this environment, got device={device!r}",
                {"device": device},
                reason="device_unsupported",
            )
        self._cache_dir = Path(cache_dir) if cache_dir is not None else default_deepgaze_cache_dir()
        self._device = device
        self._model: Any = None
        self._torch: Any = None
        self._centerbias_template: np.ndarray | None = None
        self._cold_load_ms: float | None = None
        self._model_stats: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Frozen interface
    # ------------------------------------------------------------------

    def describe(self) -> BackendInfo:
        profile = registry.profile_for_backend(self.BACKEND_ID)
        registration = registry.registration_for(profile.profile_name)
        info = BackendInfo(
            backend_id=self.BACKEND_ID,
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
        self._check_profile(resolved_profile)
        _validate_image(image)
        height, width = int(image.shape[0]), int(image.shape[1])

        preprocessing = resolved_profile.preprocessing
        target_long = int(preprocessing["target_long_side"])
        scale = min(1.0, target_long / max(height, width))  # downscale-only
        inf_h = max(1, int(round(height * scale)))
        inf_w = max(1, int(round(width * scale)))

        self._ensure_model(resolved_profile)
        torch = self._torch

        started = time.perf_counter()
        resized = _resize_image(image, (inf_h, inf_w), str(preprocessing.get("resize_filter", "pillow_bilinear")))
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
        ).unsqueeze(0)  # (1,3,h,w), [0,255]; Normalizer 在模型内部处理
        centerbias_tensor = self._prepare_centerbias(resolved_profile, (inf_h, inf_w), torch)

        try:
            with torch.inference_mode():
                prediction = self._model(image_tensor, centerbias_tensor)
        except MemoryError as exc:
            raise MemoryExhaustedError(
                "memory exhausted during DeepGaze IIE inference",
                {"device": "cpu", "inference_shape": [inf_h, inf_w]},
                reason="cpu_oom",
            ) from exc
        except Exception as exc:  # torch runtime errors
            raise InferenceFailureError(
                f"DeepGaze IIE inference failed: {type(exc).__name__}",
                {"exception_type": type(exc).__name__, "device": "cpu", "inference_shape": [inf_h, inf_w]},
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        array = _validate_log_density(prediction, expected_spatial=(inf_h, inf_w))

        result = PredictionResult(
            semantics=self.SEMANTICS,
            array=array,
            shape_mapping=_build_shape_mapping(height, width, inf_h, inf_w, scale, preprocessing),
            runtime={
                "device": "cpu",
                "precision": "fp32",
                "elapsed_ms": round(elapsed_ms, 3),
                "peak_mem_mb": peak_working_set_mb(),
            },
            vendor_metrics=None,
            limitations=self.LIMITATIONS,
        )
        result.validate()
        return result

    # ------------------------------------------------------------------
    # Measurement surface (not part of the frozen interface)
    # ------------------------------------------------------------------

    @property
    def cold_load_ms(self) -> float | None:
        """Cold model build + state_dict load time in ms (fresh process once)."""
        return self._cold_load_ms

    @property
    def model_stats(self) -> dict[str, Any] | None:
        return self._model_stats

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
        if resolved_profile.backend_id != self.BACKEND_ID:
            raise ProfileRejectedError(
                f"profile {resolved_profile.profile_name!r} targets backend "
                f"{resolved_profile.backend_id!r}, not {self.BACKEND_ID!r}",
                {
                    "profile_backend_id": resolved_profile.backend_id,
                    "expected_backend_id": self.BACKEND_ID,
                },
                reason="backend_mismatch",
            )
        if resolved_profile.centerbias is None:
            # DeepGaze IIE 必须有显式 centerbias 引用（profile 冻结规则）
            raise ProfileRejectedError(
                "deepgaze-iie profile must carry an explicit centerbias reference",
                {"profile_name": resolved_profile.profile_name},
                reason="centerbias_missing",
            )

    def _weight_paths(self, profile: ResolvedProfile) -> dict[str, Path]:
        """Verify all registered weights offline; map role → local path."""
        paths: dict[str, Path] = {}
        for ref in profile.weights:
            verified = verify_cached_weight(ref, self._cache_dir)
            if ref.name.endswith(".pth"):
                paths["model"] = verified.path
            elif ref.name.endswith(".npy"):
                paths["centerbias"] = verified.path
        if "model" not in paths or "centerbias" not in paths:
            raise WeightNotReadyError(
                "deepgaze-iie profile must register exactly one .pth and one centerbias .npy",
                {"weights": [w.name for w in profile.weights]},
                reason="manifest_incomplete",
            )
        return paths

    def _ensure_model(self, profile: ResolvedProfile) -> None:
        if self._model is not None:
            return
        started = time.perf_counter()
        paths = self._weight_paths(profile)

        try:
            import torch
        except ImportError as exc:
            raise WeightNotReadyError(
                f"torch not importable: {type(exc).__name__}",
                {"hint": "install extras [deepgaze] (torch==2.14.0 CPU, torchvision==0.29.0)"},
                reason="dependency_missing",
            ) from exc
        self._torch = torch

        deepgaze2e, features_modules = _import_deepgaze2e_offline()
        patch_record = _install_offline_backbones(features_modules)

        try:
            model = deepgaze2e.DeepGazeIIE(pretrained=False)
        except MemoryError as exc:
            raise MemoryExhaustedError(
                "memory exhausted while building DeepGazeIIE (4 backbones × 30 components)",
                {},
                reason="cpu_oom_build",
            ) from exc

        try:
            state_dict = torch.load(paths["model"], map_location=torch.device("cpu"), weights_only=True)
            model.load_state_dict(state_dict, strict=True)
        except MemoryError as exc:
            raise MemoryExhaustedError(
                "memory exhausted while loading deepgaze2e.pth",
                {"weight": paths["model"].name},
                reason="cpu_oom_load",
            ) from exc
        except RuntimeError as exc:
            # strict=True: any architecture drift (U1) fails loudly here.
            message = str(exc)
            raise WeightNotReadyError(
                "deepgaze2e.pth state_dict does not match the constructed model "
                "(U1 architecture drift); refusing strict=False fallback",
                {
                    "weight": paths["model"].name,
                    "error_excerpt": message[:500],
                },
                reason="state_dict_mismatch",
            ) from exc

        model.eval()
        self._model = model
        self._cold_load_ms = round((time.perf_counter() - started) * 1000.0, 3)
        self._model_stats = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "backbone_patch": patch_record,
            "state_dict_entries": len(state_dict),
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "source_pin": registry.DEEPGAZE_SOURCE_PIN,
        }

    def _prepare_centerbias(self, profile: ResolvedProfile, shape: tuple[int, int], torch: Any) -> Any:
        """MIT1003 template → zoom(order=0, nearest) → subtract logsumexp (README 口径)."""
        if self._centerbias_template is None:
            paths = self._weight_paths(profile)
            template = np.load(paths["centerbias"])
            if template.ndim != 2:
                raise WeightNotReadyError(
                    f"centerbias template must be a 2-D log-density map, got shape {template.shape}",
                    {"shape": list(template.shape)},
                    reason="centerbias_invalid",
                )
            self._centerbias_template = np.asarray(template, dtype=np.float64)
        template = self._centerbias_template
        expected = profile.centerbias or {}
        template_shape = expected.get("template_shape")
        if template_shape is not None and list(template.shape) != list(template_shape):
            raise WeightNotReadyError(
                f"centerbias template shape {template.shape} != registered {template_shape}",
                {"actual": list(template.shape), "registered": list(template_shape)},
                reason="centerbias_shape_mismatch",
            )
        try:
            from scipy.ndimage import zoom
            from scipy.special import logsumexp
        except ImportError as exc:
            raise WeightNotReadyError(
                f"scipy not importable: {type(exc).__name__}",
                {"hint": "scipy is required for centerbias zoom/logsumexp"},
                reason="dependency_missing",
            ) from exc
        zoom_factors = (shape[0] / template.shape[0], shape[1] / template.shape[1])
        centerbias = zoom(template, zoom_factors, order=0, mode="nearest")
        centerbias = centerbias - logsumexp(centerbias)  # renormalized log density
        return torch.from_numpy(np.ascontiguousarray(centerbias, dtype=np.float32)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Offline import & backbone patching (U1 mitigation; adapter-local only)
# ---------------------------------------------------------------------------


def _import_deepgaze2e_offline():
    """Import ``deepgaze_pytorch.deepgaze2e`` without executing the package
    ``__init__`` (which eagerly imports MSDB → requires ``clip``, outside the
    approved install list). Uses ``importlib.util.module_from_spec`` package
    stubbing — no upstream code is modified or vendored into the repo.

    Returns (deepgaze2e_module, features_modules_dict).
    """
    if "deepgaze_pytorch" in sys.modules:
        package = sys.modules["deepgaze_pytorch"]
    else:
        spec = importlib.util.find_spec("deepgaze_pytorch")
        if spec is None:
            raise WeightNotReadyError(
                "deepgaze_pytorch not installed",
                {
                    "hint": "pip install git+https://github.com/matthias-k/DeepGaze"
                    f"@{registry.DEEPGAZE_SOURCE_PIN}",
                },
                reason="dependency_missing",
            )
        package = importlib.util.module_from_spec(spec)  # NOT executed
        sys.modules["deepgaze_pytorch"] = package
    try:
        import deepgaze_pytorch.deepgaze2e as deepgaze2e
        from deepgaze_pytorch.features import densenet, efficientnet, normalizer, resnext, shapenet
        from deepgaze_pytorch.features.efficientnet_pytorch import EfficientNet
    except ImportError as exc:
        raise WeightNotReadyError(
            f"deepgaze_pytorch modules not importable: {type(exc).__name__}",
            {"exception": str(exc)[:300], "source_pin": registry.DEEPGAZE_SOURCE_PIN},
            reason="dependency_missing",
        ) from exc
    modules = {
        "deepgaze2e": deepgaze2e,
        "shapenet": shapenet,
        "efficientnet": efficientnet,
        "densenet": densenet,
        "resnext": resnext,
        "normalizer": normalizer,
        "EfficientNet": EfficientNet,
    }
    return deepgaze2e, modules


def _install_offline_backbones(modules: dict[str, Any]) -> dict[str, str]:
    """Replace the four IIE backbone classes with offline, architecture-only
    equivalents (no torch.hub, no bitbucket/ ImageNet weight downloads).

    Structure is kept key-for-key identical to upstream so that
    ``load_state_dict(deepgaze2e.pth, strict=True)`` is the equivalence proof:

    - RGBShapeNetC:   Sequential(Normalizer, Sequential(OrderedDict(module=resnet50)))
    - RGBEfficientNetB5: Sequential(Normalizer, EfficientNet.from_name('efficientnet-b5'))
    - RGBDenseNet201: Sequential(Normalizer, densenet201)
    - RGBResNext50:   Sequential(Normalizer, resnext50_32x4d)

    Returns a record of replaced classes (provenance for measurements/report).
    """
    import torch.nn as nn
    import torchvision

    Normalizer = modules["normalizer"].Normalizer
    EfficientNet = modules["EfficientNet"]

    class OfflineRGBShapeNetC(nn.Sequential):
        def __init__(self) -> None:
            super().__init__()
            backbone = torchvision.models.resnet50(weights=None)
            wrapped = nn.Sequential(OrderedDict([("module", backbone)]))
            super().__init__(Normalizer(), wrapped)

    class OfflineRGBEfficientNetB5(nn.Sequential):
        def __init__(self) -> None:
            super().__init__()
            super().__init__(Normalizer(), EfficientNet.from_name("efficientnet-b5"))

    class OfflineRGBDenseNet201(nn.Sequential):
        def __init__(self) -> None:
            super().__init__()
            super().__init__(Normalizer(), torchvision.models.densenet201(weights=None))

    class OfflineRGBResNext50(nn.Sequential):
        def __init__(self) -> None:
            super().__init__()
            super().__init__(Normalizer(), torchvision.models.resnext50_32x4d(weights=None))

    modules["shapenet"].RGBShapeNetC = OfflineRGBShapeNetC
    modules["efficientnet"].RGBEfficientNetB5 = OfflineRGBEfficientNetB5
    modules["densenet"].RGBDenseNet201 = OfflineRGBDenseNet201
    modules["resnext"].RGBResNext50 = OfflineRGBResNext50
    return {
        "RGBShapeNetC": "torchvision.models.resnet50(weights=None) + OrderedDict(module=...) wrapper",
        "RGBEfficientNetB5": "vendored EfficientNet.from_name('efficientnet-b5') (no download)",
        "RGBDenseNet201": "torchvision.models.densenet201(weights=None)",
        "RGBResNext50": "torchvision.models.resnext50_32x4d(weights=None)",
    }


# ---------------------------------------------------------------------------
# Shared validation helpers (frozen predict() contract)
# ---------------------------------------------------------------------------


def _validate_image(image: Any) -> None:
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


def _resize_image(image: np.ndarray, target_hw: tuple[int, int], filter_name: str) -> np.ndarray:
    if (image.shape[0], image.shape[1]) == target_hw:
        return image
    resample = {
        "pillow_bilinear": Image.Resampling.BILINEAR,
        "pillow_bicubic": Image.Resampling.BICUBIC,
    }.get(filter_name)
    if resample is None:
        raise WeightNotReadyError(
            f"registered resize_filter {filter_name!r} is not supported by this adapter",
            {"resize_filter": filter_name},
            reason="preprocessing_unsupported",
        )
    pil = Image.fromarray(image, mode="RGB").resize((target_hw[1], target_hw[0]), resample)
    return np.asarray(pil, dtype=np.uint8)


def _validate_log_density(prediction: Any, expected_spatial: tuple[int, int]) -> np.ndarray:
    """Validate native log-density output; NO renormalization (C1 metrics owns
    the exp/logsumexp conversion per technical-design §7)."""
    array = np.asarray(prediction.detach().cpu().numpy(), dtype=np.float64)
    if array.ndim == 4 and array.shape[0] == 1 and array.shape[1] == 1:
        array = array[0, 0]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    else:
        raise InvalidDensityError(
            f"unexpected log-density output shape {array.shape}, expected "
            f"(1, 1, {expected_spatial[0]}, {expected_spatial[1]})",
            {"shape": list(array.shape)},
            reason="shape",
        )
    if tuple(array.shape) != tuple(expected_spatial):
        raise InvalidDensityError(
            f"log-density spatial shape {array.shape} != inference shape {expected_spatial}",
            {"shape": list(array.shape), "expected_spatial": list(expected_spatial)},
            reason="shape",
        )
    if not np.all(np.isfinite(array)):
        raise InvalidDensityError(
            "log-density output contains NaN/Inf",
            {"shape": list(array.shape)},
            reason="non_finite",
        )
    return np.ascontiguousarray(array, dtype=np.float64)


def _build_shape_mapping(
    height: int,
    width: int,
    inf_h: int,
    inf_w: int,
    scale: float,
    preprocessing: Any,
) -> dict[str, Any]:
    return {
        "original_shape": [height, width],
        "inference_shape": [inf_h, inf_w],
        "method": "aspect_preserving_downscale_only",
        "resize_filter": str(preprocessing.get("resize_filter", "pillow_bilinear")),
        "padding": None,
        "scale": scale,
        "scale_yx": [inf_h / height, inf_w / width],
        "target_long_side": int(preprocessing["target_long_side"]),
        "coordinate_mapping": {
            "forward": f"x_inf = x_orig * {scale!r}; y_inf = y_orig * {scale!r} (等比)",
            "note": "只降不升；短边取整误差 <1 推理像素",
        },
        "inverse": {
            "method": "resize_density_to_original_then_renormalize",
            "target_shape": [height, width],
            "filter": "bilinear",
            "renormalize": "log_density_renormalize_after_mapping",
            "reference": "technical-design §4.5：对概率图/密度图映射后再次归一化（C1 侧执行）",
        },
    }
