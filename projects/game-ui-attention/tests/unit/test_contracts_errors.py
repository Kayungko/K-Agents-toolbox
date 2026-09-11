"""errors.py 测试：错误目录、退出码映射（data-contract §7 冻结表）、stdout JSON 信封。"""

from __future__ import annotations

import json

import pytest

from ui_attention import errors
from ui_attention.errors import (
    ERROR_EXIT_CODES,
    EXIT_INCOMPATIBLE,
    EXIT_INTERNAL,
    EXIT_INVALID_INPUT,
    EXIT_INVALID_OUTPUT,
    EXIT_IO,
    EXIT_NOT_READY,
    EXIT_OK,
    EXIT_RUNTIME,
    ErrorCode,
    UiAttentionError,
    error_envelope,
    success_envelope,
)


def test_exit_code_table_matches_contract():
    """data-contract §7 冻结退出码：0/2/3/4/5/6/7。"""
    assert (
        EXIT_OK,
        EXIT_INVALID_INPUT,
        EXIT_NOT_READY,
        EXIT_RUNTIME,
        EXIT_INVALID_OUTPUT,
        EXIT_INCOMPATIBLE,
        EXIT_IO,
    ) == (0, 2, 3, 4, 5, 6, 7)
    assert EXIT_INTERNAL == 1  # 工具缺陷扩展码（非冻结表成员，保证非零不假成功）


def test_error_catalog_contains_contract_examples():
    """§7 示例错误 ID 全部在目录中。"""
    for name in ("MODEL_NOT_READY", "INVALID_AOI", "GPU_OOM", "INVALID_DENSITY", "COMPARISON_INCOMPATIBLE"):
        assert name in ErrorCode.__members__


@pytest.mark.parametrize(
    ("code", "expected_exit"),
    [
        (ErrorCode.INVALID_REQUEST, 2),
        (ErrorCode.INVALID_SCHEMA_VERSION, 2),
        (ErrorCode.INVALID_IMAGE, 2),
        (ErrorCode.INVALID_AOI, 2),
        (ErrorCode.REGIONS_IMAGE_MISMATCH, 2),
        (ErrorCode.MODEL_NOT_READY, 3),
        (ErrorCode.PROFILE_NOT_REGISTERED, 3),
        (ErrorCode.LICENSE_NOT_CLEARED, 3),
        (ErrorCode.GPU_OOM, 4),
        (ErrorCode.INFERENCE_FAILED, 4),
        (ErrorCode.INVALID_DENSITY, 5),
        (ErrorCode.UNSUPPORTED_SEMANTICS, 5),
        (ErrorCode.ARTIFACT_CHECK_FAILED, 5),
        (ErrorCode.COMPARISON_INCOMPATIBLE, 6),
        (ErrorCode.OUTPUT_PATH_EXISTS, 7),
        (ErrorCode.IO_ERROR, 7),
        (ErrorCode.INTERNAL_ERROR, 1),
    ],
)
def test_error_exit_mapping(code, expected_exit):
    assert ERROR_EXIT_CODES[code] == expected_exit
    err = UiAttentionError(code, "msg")
    assert err.exit_code == expected_exit


def test_every_error_code_has_exit_mapping():
    assert set(ERROR_EXIT_CODES) == set(ErrorCode)


def test_success_envelope_shape():
    env = success_envelope({"run_directory": "runs/x"})
    assert env["ok"] is True
    assert env["result"] == {"run_directory": "runs/x"}
    assert "error" not in env
    json.dumps(env)  # 必须可序列化


def test_error_envelope_shape():
    err = UiAttentionError(ErrorCode.MODEL_NOT_READY, "权重缺失", {"weights": ["a.onnx"]})
    env = error_envelope(err)
    assert env["ok"] is False
    assert env["error"]["id"] == "MODEL_NOT_READY"
    assert env["error"]["message"] == "权重缺失"
    assert env["error"]["details"] == {"weights": ["a.onnx"]}
    assert env["error"]["exit_code"] == 3
    assert "result" not in env
    json.dumps(env)


def test_emit_writes_single_json_to_stdout_only(capsys):
    errors.progress("处理中…")
    errors.emit_success({"a": 1})
    captured = capsys.readouterr()
    # stdout 恰好一个 JSON 信封
    payload = json.loads(captured.out)
    assert payload == {"ok": True, "result": {"a": 1}}
    # 进度只出现在 stderr
    assert "处理中" in captured.err
    assert captured.err.startswith("[ui-attention]")


def test_emit_error_to_stdout(capsys):
    errors.emit_error(UiAttentionError(ErrorCode.INVALID_AOI, "自交"))
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["id"] == "INVALID_AOI"


def test_error_accepts_string_code():
    err = UiAttentionError("GPU_OOM", "显存不足")
    assert err.code is ErrorCode.GPU_OOM
    assert err.exit_code == 4
    with pytest.raises(ValueError):
        UiAttentionError("NOT_A_REAL_CODE", "x")
