"""结构化错误目录、stdout JSON 信封与退出码常量（data-contract.md §7）。

约定：
- stdout 只输出一个 JSON 信封：``{"ok": true, "result": {...}}`` 或 ``{"ok": false, "error": {...}}``；
- 进度信息一律走 stderr（:func:`progress`）；
- 调用方以退出码分支，不以文字匹配代替（data-contract §7）；
- 错误 details 不得携带凭据、Cookie、连接串或原始第三方敏感响应（data-contract §4 errors 行）。
"""

from __future__ import annotations

import json
import sys
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# 退出码（data-contract.md §7 冻结表）
# ---------------------------------------------------------------------------
EXIT_OK = 0  # 请求的计算步骤成功
EXIT_INVALID_INPUT = 2  # 参数、图片或 AOI 无效
EXIT_NOT_READY = 3  # 后端、依赖、权重或许可配置未就绪
EXIT_RUNTIME = 4  # 设备资源不足或推理失败
EXIT_INVALID_OUTPUT = 5  # 输出概率无效或产物校验失败
EXIT_INCOMPATIBLE = 6  # A/B 配置不兼容
EXIT_IO = 7  # 输出路径已存在、不可写或文件操作失败

# 未预期内部错误。冻结表（0/2-7）不覆盖此情形；使用非零 1 保证不产生假成功，
# 语义为"工具缺陷"，与输入/后端/输出错误区分。（偏差已在 C1 完成报中注明。）
EXIT_INTERNAL = 1


class ErrorCode(StrEnum):
    """结构化错误 ID 目录（含 data-contract §7 示例错误 ID）。"""

    # 输入侧（退出码 2）
    INVALID_REQUEST = "INVALID_REQUEST"  # 请求结构/字段/枚举无效
    INVALID_SCHEMA_VERSION = "INVALID_SCHEMA_VERSION"  # 未知 schema_version（拒绝，不猜测迁移）
    INVALID_IMAGE = "INVALID_IMAGE"  # 图片不可解码/格式不支持/带透明且未给合成背景
    INVALID_AOI = "INVALID_AOI"  # 重复 ID、零面积、越界、自交多边形、共线等
    REGIONS_IMAGE_MISMATCH = "REGIONS_IMAGE_MISMATCH"  # regions 文件 SHA-256 与图片不符

    # 就绪侧（退出码 3）
    MODEL_NOT_READY = "MODEL_NOT_READY"  # 权重缺失/哈希不符/后端模块缺失
    PROFILE_NOT_REGISTERED = "PROFILE_NOT_REGISTERED"  # backend_profile 未登记（不得静默改写）
    LICENSE_NOT_CLEARED = "LICENSE_NOT_CLEARED"  # 许可缺口未闭合

    # 运行侧（退出码 4）
    GPU_OOM = "GPU_OOM"  # 显存/内存不足
    INFERENCE_FAILED = "INFERENCE_FAILED"  # 推理执行失败

    # 输出侧（退出码 5）
    INVALID_DENSITY = "INVALID_DENSITY"  # 概率图含 NaN/负值/sum≠1 超差
    UNSUPPORTED_SEMANTICS = "UNSUPPORTED_SEMANTICS"  # 后端输出语义不被当前管线消费
    ARTIFACT_CHECK_FAILED = "ARTIFACT_CHECK_FAILED"  # 产物缺失或哈希校验失败

    # 比较侧（退出码 6）
    COMPARISON_INCOMPATIBLE = "COMPARISON_INCOMPATIBLE"  # A/B 配置不兼容（保留原因）

    # IO 侧（退出码 7）
    OUTPUT_PATH_EXISTS = "OUTPUT_PATH_EXISTS"  # 输出路径已存在（不覆盖历史）
    IO_ERROR = "IO_ERROR"  # 不可写或其他文件操作失败

    # 工具缺陷（退出码 1）
    INTERNAL_ERROR = "INTERNAL_ERROR"


ERROR_EXIT_CODES: dict[ErrorCode, int] = {
    ErrorCode.INVALID_REQUEST: EXIT_INVALID_INPUT,
    ErrorCode.INVALID_SCHEMA_VERSION: EXIT_INVALID_INPUT,
    ErrorCode.INVALID_IMAGE: EXIT_INVALID_INPUT,
    ErrorCode.INVALID_AOI: EXIT_INVALID_INPUT,
    ErrorCode.REGIONS_IMAGE_MISMATCH: EXIT_INVALID_INPUT,
    ErrorCode.MODEL_NOT_READY: EXIT_NOT_READY,
    ErrorCode.PROFILE_NOT_REGISTERED: EXIT_NOT_READY,
    ErrorCode.LICENSE_NOT_CLEARED: EXIT_NOT_READY,
    ErrorCode.GPU_OOM: EXIT_RUNTIME,
    ErrorCode.INFERENCE_FAILED: EXIT_RUNTIME,
    ErrorCode.INVALID_DENSITY: EXIT_INVALID_OUTPUT,
    ErrorCode.UNSUPPORTED_SEMANTICS: EXIT_INVALID_OUTPUT,
    ErrorCode.ARTIFACT_CHECK_FAILED: EXIT_INVALID_OUTPUT,
    ErrorCode.COMPARISON_INCOMPATIBLE: EXIT_INCOMPATIBLE,
    ErrorCode.OUTPUT_PATH_EXISTS: EXIT_IO,
    ErrorCode.IO_ERROR: EXIT_IO,
    ErrorCode.INTERNAL_ERROR: EXIT_INTERNAL,
}


class UiAttentionError(Exception):
    """携带错误 ID、退出码与可序列化 details 的结构化错误。

    ``details`` 必须可 JSON 序列化，且不得包含凭据或第三方原始敏感响应。
    """

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        details: dict[str, Any] | None = None,
        exit_code: int | None = None,
    ) -> None:
        self.code = ErrorCode(code)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})
        default_exit = ERROR_EXIT_CODES[self.code]
        if exit_code is not None and exit_code != default_exit:
            # 允许显式覆盖（例如组合场景），但默认严格遵循目录映射。
            self.exit_code = exit_code
        else:
            self.exit_code = default_exit
        super().__init__(f"[{self.code.value}] {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.code.value,
            "message": self.message,
            "details": self.details,
            "exit_code": self.exit_code,
        }


# ---------------------------------------------------------------------------
# stdout JSON 信封 / stderr 进度
# ---------------------------------------------------------------------------


def success_envelope(result: dict[str, Any] | None = None) -> dict[str, Any]:
    """成功信封：``{"ok": true, "result": {...}}``。"""
    return {"ok": True, "result": result if result is not None else {}}


def error_envelope(error: UiAttentionError) -> dict[str, Any]:
    """失败信封：``{"ok": false, "error": {id, message, details, exit_code}}``。"""
    return {"ok": False, "error": error.to_dict()}


def ensure_utf8_streams() -> None:
    """尽力把 stdout/stderr 切到 UTF-8（Windows GBK 控制台兜底）。

    pytest 捕获流等无 reconfigure 的场景静默跳过；此时 emit/progress
    仍各自有 UnicodeEncodeError 兜底，保证信封永远可输出。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def emit(payload: dict[str, Any]) -> None:
    """把信封写到 stdout（唯一 JSON 输出点）。

    控制台编码不支持非 ASCII 时退化为 ensure_ascii 转义输出：
    信封必须是可解析 JSON，宁可损失可读性也不能截断或崩溃。
    """
    ensure_utf8_streams()
    try:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    except UnicodeEncodeError:
        sys.stdout.write(json.dumps(payload, ensure_ascii=True, indent=2, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


def emit_success(result: dict[str, Any] | None = None) -> None:
    emit(success_envelope(result))


def emit_error(error: UiAttentionError) -> None:
    emit(error_envelope(error))


def progress(message: str) -> None:
    """进度信息走 stderr，不污染 stdout 信封。"""
    try:
        sys.stderr.write(f"[ui-attention] {message}\n")
    except UnicodeEncodeError:
        sys.stderr.write(f"[ui-attention] {message}\n".encode("ascii", "replace").decode("ascii"))
    sys.stderr.flush()
