"""后端冻结接口（data-contract.md §3，G0 冻结 2026-09-11）。

本节为 C1（contracts/ 定义）与 C2（backends/ 实现）之间的冻结接口；
任何签名变更必须递增 ``schema_version`` 并经二级总控重新冻结，实现线不得单方面改动。

冻结签名::

    describe() -> BackendInfo
    predict(image, resolved_profile) -> PredictionResult

- ``predict`` 的 ``image`` 为方向已处理的原图 ``numpy.ndarray``，RGB、``uint8``、形状 ``(H, W, 3)``；
  缩放、均值减除等模型侧预处理由后端按 ``resolved_profile`` 内部执行并写入 ``shape_mapping``，
  调用方不得预处理。
- ``resolved_profile`` 为登记 profile 的解析结果（:class:`ResolvedProfile`），含预处理参数、
  可选 centerbias 引用、观看条件与全部配置哈希；后端拒绝未登记或运行时改写的 profile（退出码 3）。

另附 C1↔C2 协作所需的 ``backends/registry.py`` 期望接口（:class:`BackendRegistry` Protocol）；
该接口不属于 §3 冻结字段表，但为 cli.py 动态导入的依赖面，变更需经二级总控。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from ..errors import ErrorCode, UiAttentionError

# 冻结枚举（data-contract §3 字段表）
CAPABILITIES: tuple[str, ...] = ("spatial_density", "vendor_metrics", "scanpath")
NATIVE_SEMANTICS: tuple[str, ...] = ("log_density", "probability_density", "vendor_metrics")
DEVICES: tuple[str, ...] = ("cpu", "cuda")
CLEARED_FOR: tuple[str, ...] = ("internal-eval", "packaged")

# profile 登记名规范 <backend>-<variant>-v<N>（data-contract §3 profile 登记冻结规则）
PROFILE_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-v[0-9]+$")

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _check_sha256(value: object, field_name: str, errors: list[str]) -> None:
    if not (isinstance(value, str) and _SHA256_PATTERN.match(value)):
        errors.append(f"{field_name} 必须是 64 位小写十六进制 sha256，得到 {value!r}")


@dataclass(frozen=True)
class WeightRef:
    """权重登记项（data-contract §3：``{name, source_url, sha256, size_bytes}``）。

    doctor 校验实际文件哈希 == 登记值，不匹配退出码 3（MODEL_NOT_READY）。
    """

    name: str
    source_url: str
    sha256: str
    size_bytes: int

    def validate(self) -> None:
        errors: list[str] = []
        if not (isinstance(self.name, str) and self.name):
            errors.append("WeightRef.name 必须是非空字符串")
        if not isinstance(self.source_url, str):
            errors.append("WeightRef.source_url 必须是字符串")
        _check_sha256(self.sha256, "WeightRef.sha256", errors)
        if not (isinstance(self.size_bytes, int) and not isinstance(self.size_bytes, bool) and self.size_bytes >= 0):
            errors.append(f"WeightRef.size_bytes 必须是非负整数，得到 {self.size_bytes!r}")
        if errors:
            raise UiAttentionError(ErrorCode.MODEL_NOT_READY, "WeightRef 校验失败", {"errors": errors})

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "source_url": self.source_url, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, obj: Any) -> WeightRef:
        if not isinstance(obj, dict):
            raise UiAttentionError(ErrorCode.MODEL_NOT_READY, "WeightRef 必须是对象", {"got": type(obj).__name__})
        unknown = set(obj) - {"name", "source_url", "sha256", "size_bytes"}
        if unknown:
            raise UiAttentionError(ErrorCode.MODEL_NOT_READY, "WeightRef 含未知字段", {"unknown": sorted(unknown)})
        ref = cls(
            name=obj.get("name"),  # type: ignore[arg-type]
            source_url=obj.get("source_url", ""),
            sha256=obj.get("sha256"),  # type: ignore[arg-type]
            size_bytes=obj.get("size_bytes"),  # type: ignore[arg-type]
        )
        ref.validate()
        return ref


@dataclass(frozen=True)
class BackendInfo:
    """后端自描述（data-contract §3 冻结字段表，逐字段对应）。

    - backend_id: 如 ``foveacast-onnx-3s``、``deepgaze-iie``
    - version: 后端实现版本（独立于模型权重版本）
    - capabilities: ``spatial_density`` / ``vendor_metrics`` / ``scanpath``；第一版只消费 ``spatial_density``
    - native_semantics: ``log_density`` | ``probability_density`` | ``vendor_metrics``
    - device_requirements: ``{"device": "cpu"|"cuda", "min_free_vram_mb": int|None}``
    - license_status: ``{"code", "weights", "gaps": [...], "cleared_for": "internal-eval"|"packaged"}``
    - weights: tuple[WeightRef]
    """

    backend_id: str
    version: str
    capabilities: tuple[str, ...]
    native_semantics: str
    device_requirements: dict[str, Any] = field(default_factory=dict)
    license_status: dict[str, Any] = field(default_factory=dict)
    weights: tuple[WeightRef, ...] = ()

    def validate(self) -> None:
        errors: list[str] = []
        if not (isinstance(self.backend_id, str) and self.backend_id):
            errors.append("backend_id 必须是非空字符串")
        if not (isinstance(self.version, str) and self.version):
            errors.append("version 必须是非空字符串")
        if not isinstance(self.capabilities, tuple):
            errors.append("capabilities 必须是 tuple[str]（冻结口径）")
        else:
            for cap in self.capabilities:
                if cap not in CAPABILITIES:
                    errors.append(f"capability {cap!r} 不在冻结枚举 {CAPABILITIES}")
        if self.native_semantics not in NATIVE_SEMANTICS:
            errors.append(f"native_semantics {self.native_semantics!r} 不在冻结枚举 {NATIVE_SEMANTICS}")
        self._validate_device_requirements(errors)
        self._validate_license_status(errors)
        if not isinstance(self.weights, tuple):
            errors.append("weights 必须是 tuple[WeightRef]（冻结口径）")
        else:
            for i, w in enumerate(self.weights):
                if not isinstance(w, WeightRef):
                    errors.append(f"weights[{i}] 必须是 WeightRef")
                else:
                    try:
                        w.validate()
                    except UiAttentionError as exc:
                        errors.extend(exc.details.get("errors", [exc.message]))
        if errors:
            raise UiAttentionError(ErrorCode.MODEL_NOT_READY, "BackendInfo 校验失败", {"errors": errors})

    def _validate_device_requirements(self, errors: list[str]) -> None:
        dr = self.device_requirements
        if not isinstance(dr, dict):
            errors.append("device_requirements 必须是 dict")
            return
        if dr.get("device") not in DEVICES:
            errors.append(f"device_requirements.device 必须是 {DEVICES} 之一，得到 {dr.get('device')!r}")
        vram = dr.get("min_free_vram_mb")
        if vram is not None and not (isinstance(vram, int) and not isinstance(vram, bool) and vram >= 0):
            errors.append(f"device_requirements.min_free_vram_mb 必须是 int|None，得到 {vram!r}")

    def _validate_license_status(self, errors: list[str]) -> None:
        ls = self.license_status
        if not isinstance(ls, dict):
            errors.append("license_status 必须是 dict")
            return
        for key in ("code", "weights"):
            if not isinstance(ls.get(key), str) or not ls.get(key):
                errors.append(f"license_status.{key} 必须是非空字符串")
        gaps = ls.get("gaps")
        if not isinstance(gaps, (list, tuple)) or not all(isinstance(g, str) for g in gaps):
            errors.append("license_status.gaps 必须是 list[str]（缺口编号沿用 sources-and-decisions）")
        if ls.get("cleared_for") not in CLEARED_FOR:
            errors.append(f"license_status.cleared_for 必须是 {CLEARED_FOR} 之一，得到 {ls.get('cleared_for')!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "native_semantics": self.native_semantics,
            "device_requirements": dict(self.device_requirements),
            "license_status": dict(self.license_status),
            "weights": [w.to_dict() for w in self.weights],
        }

    @classmethod
    def from_dict(cls, obj: Any) -> BackendInfo:
        if not isinstance(obj, dict):
            raise UiAttentionError(ErrorCode.MODEL_NOT_READY, "BackendInfo 必须是对象")
        known = {
            "backend_id",
            "version",
            "capabilities",
            "native_semantics",
            "device_requirements",
            "license_status",
            "weights",
        }
        unknown = set(obj) - known
        if unknown:
            raise UiAttentionError(ErrorCode.MODEL_NOT_READY, "BackendInfo 含未知字段", {"unknown": sorted(unknown)})
        info = cls(
            backend_id=obj.get("backend_id"),  # type: ignore[arg-type]
            version=obj.get("version"),  # type: ignore[arg-type]
            capabilities=tuple(obj.get("capabilities", ())),
            native_semantics=obj.get("native_semantics"),  # type: ignore[arg-type]
            device_requirements=dict(obj.get("device_requirements", {})),
            license_status=dict(obj.get("license_status", {})),
            weights=tuple(WeightRef.from_dict(w) for w in obj.get("weights", ())),
        )
        info.validate()
        return info


@dataclass(frozen=True)
class ResolvedProfile:
    """登记 profile 的解析结果（data-contract §3 profile 登记冻结规则）。

    分析记录必须包含实际执行的后端配置解析结果，不能只保存请求中的别名。
    ``config_hash`` 为全部配置（backend_id+version、权重清单与哈希、预处理参数、
    centerbias 引用、观看条件假设）的合成哈希。
    """

    profile_name: str
    backend_id: str
    backend_version: str
    preprocessing: dict[str, Any]
    centerbias: dict[str, Any] | None
    viewing_conditions: dict[str, Any]
    config_hash: str
    weights: tuple[WeightRef, ...] = ()

    def validate(self) -> None:
        errors: list[str] = []
        if not (isinstance(self.profile_name, str) and PROFILE_NAME_PATTERN.match(self.profile_name)):
            errors.append(f"profile_name {self.profile_name!r} 不符合登记名规范 <backend>-<variant>-v<N>")
        if not (isinstance(self.backend_id, str) and self.backend_id):
            errors.append("ResolvedProfile.backend_id 必须是非空字符串")
        if not (isinstance(self.backend_version, str) and self.backend_version):
            errors.append("ResolvedProfile.backend_version 必须是非空字符串")
        if not isinstance(self.preprocessing, dict):
            errors.append("ResolvedProfile.preprocessing 必须是 dict（目标尺寸、值域、均值处理）")
        if self.centerbias is not None and not isinstance(self.centerbias, dict):
            errors.append("ResolvedProfile.centerbias 必须是 dict|None（引用含来源与哈希）")
        if not isinstance(self.viewing_conditions, dict):
            errors.append("ResolvedProfile.viewing_conditions 必须是 dict（实验假设须标记）")
        _check_sha256(self.config_hash, "ResolvedProfile.config_hash", errors)
        for i, w in enumerate(self.weights):
            if not isinstance(w, WeightRef):
                errors.append(f"ResolvedProfile.weights[{i}] 必须是 WeightRef")
        if errors:
            raise UiAttentionError(ErrorCode.PROFILE_NOT_REGISTERED, "ResolvedProfile 校验失败", {"errors": errors})

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "preprocessing": dict(self.preprocessing),
            "centerbias": dict(self.centerbias) if self.centerbias is not None else None,
            "viewing_conditions": dict(self.viewing_conditions),
            "config_hash": self.config_hash,
            "weights": [w.to_dict() for w in self.weights],
        }

    @classmethod
    def from_dict(cls, obj: Any) -> ResolvedProfile:
        if not isinstance(obj, dict):
            raise UiAttentionError(ErrorCode.PROFILE_NOT_REGISTERED, "ResolvedProfile 必须是对象")
        known = {
            "profile_name",
            "backend_id",
            "backend_version",
            "preprocessing",
            "centerbias",
            "viewing_conditions",
            "config_hash",
            "weights",
        }
        unknown = set(obj) - known
        if unknown:
            raise UiAttentionError(
                ErrorCode.PROFILE_NOT_REGISTERED, "ResolvedProfile 含未知字段", {"unknown": sorted(unknown)}
            )
        cb = obj.get("centerbias")
        profile = cls(
            profile_name=obj.get("profile_name"),  # type: ignore[arg-type]
            backend_id=obj.get("backend_id"),  # type: ignore[arg-type]
            backend_version=obj.get("backend_version"),  # type: ignore[arg-type]
            preprocessing=dict(obj.get("preprocessing", {})),
            centerbias=dict(cb) if cb is not None else None,
            viewing_conditions=dict(obj.get("viewing_conditions", {})),
            config_hash=obj.get("config_hash"),  # type: ignore[arg-type]
            weights=tuple(WeightRef.from_dict(w) for w in obj.get("weights", ())),
        )
        profile.validate()
        return profile


@dataclass(frozen=True)
class PredictionResult:
    """后端推理结果（data-contract §3 冻结字段表，逐字段对应）。

    - semantics: 显式声明 ``log_density`` | ``probability_density`` | ``vendor_metrics``；
      foveacast 适配层完成 sum=1 重归一化后声明 ``probability_density``，DeepGaze IIE 原生 ``log_density``
    - array: float64 空间图，形状 ``(h, w)``；不得为 8-bit 或量化存储
    - shape_mapping: 原图尺寸 ``(H, W)``、推理尺寸 ``(h, w)``、缩放/填充参数与逆变换方法
      （对齐 technical-design §4.4-4.5）
    - runtime: ``{"device", "precision", "elapsed_ms", "peak_mem_mb"}`` 实测值
    - vendor_metrics: 仅 ``vendor_metrics`` 语义时保存原名称与语义；其他语义必须为 None
    - limitations: 必须包含相对量语义、分辨率上限、训练分布边界等后端已知限制
    """

    semantics: str
    array: np.ndarray
    shape_mapping: dict[str, Any]
    runtime: dict[str, Any]
    vendor_metrics: dict[str, Any] | None = None
    limitations: tuple[str, ...] = ()

    RUNTIME_KEYS: ClassVar[tuple[str, ...]] = ("device", "precision", "elapsed_ms", "peak_mem_mb")

    def validate(self) -> None:
        errors: list[str] = []
        array_errors: list[str] = []
        if self.semantics not in NATIVE_SEMANTICS:
            errors.append(f"semantics {self.semantics!r} 不在冻结枚举 {NATIVE_SEMANTICS}")
        self._validate_array(array_errors)
        self._validate_shape_mapping(errors)
        self._validate_runtime(errors)
        if self.semantics != "vendor_metrics" and self.vendor_metrics is not None:
            errors.append("vendor_metrics 仅允许在 semantics=vendor_metrics 时非 None")
        if self.semantics == "vendor_metrics" and not isinstance(self.vendor_metrics, dict):
            errors.append("semantics=vendor_metrics 时 vendor_metrics 必须是 dict（保存原名称与语义）")
        if not isinstance(self.limitations, tuple) or not all(isinstance(x, str) and x for x in self.limitations):
            errors.append("limitations 必须是非空字符串 tuple（相对量语义/分辨率上限/训练分布边界等）")
        elif len(self.limitations) == 0:
            errors.append("limitations 不得为空：必须记录后端已知限制")
        if array_errors:
            # 数组本体无效 = "输出概率无效"（data-contract §7 退出码 5），与执行失败（4）区分
            raise UiAttentionError(
                ErrorCode.INVALID_DENSITY,
                "PredictionResult.array 无效（输出概率无效）",
                {"errors": array_errors + errors},
            )
        if errors:
            raise UiAttentionError(ErrorCode.INFERENCE_FAILED, "PredictionResult 校验失败", {"errors": errors})

    def _validate_array(self, errors: list[str]) -> None:
        arr = self.array
        if not isinstance(arr, np.ndarray):
            errors.append("array 必须是 numpy.ndarray")
            return
        if arr.dtype != np.float64:
            errors.append(f"array.dtype 必须是 float64（不得 8-bit 或量化存储），得到 {arr.dtype}")
        if arr.ndim != 2:
            errors.append(f"空间图 array 形状必须是 (h, w)，得到 ndim={arr.ndim}")
            return
        if arr.size == 0:
            errors.append("array 不得为空")
        elif not np.all(np.isfinite(arr)):
            errors.append("array 含 NaN/Inf")

    def _validate_shape_mapping(self, errors: list[str]) -> None:
        sm = self.shape_mapping
        if not isinstance(sm, dict):
            errors.append("shape_mapping 必须是 dict")
            return
        for key in ("original_shape", "inference_shape"):
            value = sm.get(key)
            ok = (
                isinstance(value, (tuple, list))
                and len(value) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value)
            )
            if not ok:
                errors.append(f"shape_mapping.{key} 必须是正整数 (H, W)，得到 {value!r}")
        inf = sm.get("inference_shape")
        if (
            isinstance(self.array, np.ndarray)
            and self.array.ndim == 2
            and isinstance(inf, (tuple, list))
            and len(inf) == 2
            and tuple(inf) != tuple(self.array.shape)
        ):
            errors.append(f"shape_mapping.inference_shape {tuple(inf)} 与 array.shape {self.array.shape} 不一致")

    def _validate_runtime(self, errors: list[str]) -> None:
        rt = self.runtime
        if not isinstance(rt, dict):
            errors.append("runtime 必须是 dict")
            return
        for key in self.RUNTIME_KEYS:
            if key not in rt:
                errors.append(f"runtime 缺少冻结键 {key!r}")
        if rt.get("device") not in DEVICES:
            errors.append(f"runtime.device 必须是 {DEVICES} 之一，得到 {rt.get('device')!r}")
        elapsed = rt.get("elapsed_ms")
        if elapsed is not None and not isinstance(elapsed, (int, float)):
            errors.append(f"runtime.elapsed_ms 必须是数值（实测），得到 {elapsed!r}")
        peak = rt.get("peak_mem_mb")
        if peak is not None and not isinstance(peak, (int, float)):
            errors.append(f"runtime.peak_mem_mb 必须是数值（实测），得到 {peak!r}")

    def to_dict(self) -> dict[str, Any]:
        """序列化（array 不入 JSON，仅元信息；数组本体走 density.npy）。"""
        return {
            "semantics": self.semantics,
            "array_shape": list(self.array.shape) if isinstance(self.array, np.ndarray) else None,
            "shape_mapping": dict(self.shape_mapping),
            "runtime": dict(self.runtime),
            "vendor_metrics": dict(self.vendor_metrics) if self.vendor_metrics is not None else None,
            "limitations": list(self.limitations),
        }


@runtime_checkable
class Backend(Protocol):
    """后端实现 Protocol（G0 冻结签名，C2 实现于 backends/）。"""

    def describe(self) -> BackendInfo:
        """返回后端自描述；doctor 与 analysis.json 的 model 字段消费。"""
        ...

    def predict(self, image: np.ndarray, resolved_profile: ResolvedProfile) -> PredictionResult:
        """对方向已处理的原图 RGB uint8 (H, W, 3) 推理。

        模型侧预处理由后端内部执行并写入 shape_mapping；调用方不得预处理。
        失败时抛 :class:`UiAttentionError`（MODEL_NOT_READY/GPU_OOM/INFERENCE_FAILED，退出码 3/4）。
        """
        ...


@runtime_checkable
class BackendRegistry(Protocol):
    """``backends/registry.py``（C2 持有）的期望模块接口。

    cli.py 通过 ``importlib.import_module("ui_attention.backends.registry")`` 动态导入；
    模块缺失 → MODEL_NOT_READY（退出码 3），不得崩溃。
    此 Protocol 不在 §3 冻结字段表内，属 C1↔C2 协作面，变更需经二级总控。
    """

    def list_profiles(self) -> tuple[str, ...]:
        """全部已登记 profile 名（doctor 展示用）。"""
        ...

    def resolve_profile(self, profile_name: str) -> ResolvedProfile:
        """解析登记 profile；未登记或运行时改写 → PROFILE_NOT_REGISTERED（退出码 3）。"""
        ...

    def get_backend(self, profile: ResolvedProfile) -> Backend:
        """按解析结果实例化后端；权重缺失/哈希不符 → MODEL_NOT_READY（退出码 3）。"""
        ...

    def doctor_report(self) -> dict[str, Any]:
        """依赖版本、设备与空闲显存、权重存在性与 sha256、许可核查记录；不打印凭据。"""
        ...
