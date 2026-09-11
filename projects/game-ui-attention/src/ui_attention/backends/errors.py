"""Backend-specific exception types (C2 scope).

按二级总控接口对齐纠偏（2026-09-11）：本模块**不定义任何错误码**，全部
复用 C1 唯一持有的 :mod:`ui_attention.errors` 错误码目录（``ErrorCode``）与
结构化错误基类（``UiAttentionError``，含退出码映射与 JSON 信封 ``to_dict``）。
这里只保留后端专属的异常类型（便于 isinstance 分支与测试），错误码与退出码
语义完全由 C1 目录决定：

- :class:`WeightNotReadyError`   → ``MODEL_NOT_READY``（退出码 3；权重缺失/哈希不符/元数据不支持）
- :class:`DownloadFailureError`  → ``MODEL_NOT_READY``（退出码 3；
  details.reason=download_failed/network_error/http_error）
- :class:`ProfileRejectedError`  → ``PROFILE_NOT_REGISTERED``（退出码 3；未登记/运行时改写/后端不匹配，
  对齐 contracts/backend.py ``BackendRegistry.resolve_profile`` 文档口径）
- :class:`ImageRejectedError`    → ``INVALID_IMAGE``（退出码 2；image 违反冻结 predict() 输入约定）
- :class:`InferenceFailureError` → ``INFERENCE_FAILED``（退出码 4）
- :class:`MemoryExhaustedError`  → ``GPU_OOM``（退出码 4；C1 目录注释：显存/内存不足，CPU OOM 同用此码）
- :class:`InvalidDensityError`   → ``INVALID_DENSITY``（退出码 5；输出无效，绝不合成假热图）

细节分类一律放 ``details["reason"]``，不新增顶层错误码。
"""

from __future__ import annotations

from typing import Any

from ui_attention.errors import ErrorCode, UiAttentionError

__all__ = [
    "BackendError",
    "DownloadFailureError",
    "ImageRejectedError",
    "InferenceFailureError",
    "InvalidDensityError",
    "MemoryExhaustedError",
    "ProfileRejectedError",
    "WeightNotReadyError",
]


class BackendError(UiAttentionError):
    """Base class for backend structured failures; inherits C1 envelope logic."""

    default_code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
    ) -> None:
        merged: dict[str, Any] = dict(details or {})
        if reason is not None:
            merged.setdefault("reason", reason)
        super().__init__(self.default_code, message, merged)


class WeightNotReadyError(BackendError):
    """Weights missing / sha256 mismatch / size mismatch / bad ONNX metadata."""

    default_code = ErrorCode.MODEL_NOT_READY


class DownloadFailureError(BackendError):
    """Weight download failed (network/HTTP). No fallback source, no retry-elsewhere."""

    default_code = ErrorCode.MODEL_NOT_READY


class ProfileRejectedError(BackendError):
    """Profile not registered, runtime-tampered, or bound to another backend."""

    default_code = ErrorCode.PROFILE_NOT_REGISTERED


class ImageRejectedError(BackendError):
    """Input image violates the frozen predict() contract (RGB uint8 (H, W, 3))."""

    default_code = ErrorCode.INVALID_IMAGE


class InferenceFailureError(BackendError):
    """ONNX Runtime session run failed."""

    default_code = ErrorCode.INFERENCE_FAILED


class MemoryExhaustedError(BackendError):
    """Memory exhaustion during inference (CPU or GPU)."""

    default_code = ErrorCode.GPU_OOM


class InvalidDensityError(BackendError):
    """Model output failed validity checks (finite/non-negative/shape/sum)."""

    default_code = ErrorCode.INVALID_DENSITY
