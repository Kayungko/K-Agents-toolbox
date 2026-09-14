"""哈希与 config_hash 工具（calibration/web 共用；仅 stdlib）。

``config_hash = sha256(canonical_json(七参与项))``（A4 建议书 §5.5 / 任务书 §6）：

1. 图集合版本：active 图库排序后的 ``sha256`` 列表（内容指纹）；
2. 协议版本 + schema_version（protocol.md 版本 + response schema 版本）；
3. 客户端版本：扩展版 ``index.html`` 代码内容哈希；
4. 服务端版本：``calibration/web/`` 应用代码哈希（排序 .py 的 (名, sha256)）；
5. 观看时长 ``duration_seconds``；
6. 图序种子策略（``"per-session"`` 或 ``"global:<seed>"``；每会话实际种子写入 session/response）；
7. （预留）``device`` / ``device_params``（未来眼动，默认 None）。

任一参与项不同 → ``config_hash`` 不同 → 结果分表并列（对齐 benchmark-protocol §8.1）。

本模块零第三方依赖（无 fastapi / PIL），可被 store / app / export_responses / tests 复用。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20

# 协议与 schema 版本常量（与 A1 protocol.md / index.html / make_study_package.py 对齐）
PROTOCOL_VERSION = "v1"
RESPONSE_SCHEMA_VERSION = "game-ui-attention-calibration-response/v1"
SESSION_SCHEMA_VERSION = "game-ui-attention-calibration-session/v1"

# 种子策略标识（写入 config_hash 第 6 参与项；全局种子时形如 "global:<seed>"）
SEED_STRATEGY_PER_SESSION = "per-session"


def sha256_file(path: str | Path) -> str:
    """文件字节 SHA-256（与主项目 imaging.sha256_file / make_study_package.sha256_file 同口径）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """字节 SHA-256。"""
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    """确定性 JSON：排序键 + 紧凑分隔符 + 保留 unicode（config_hash 的稳定输入）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_config_hash(
    *,
    image_sha256s: list[str],
    protocol_version: str,
    schema_version: str,
    client_code_hash: str,
    server_code_hash: str,
    duration_seconds: float,
    seed_strategy: str,
    device: Any = None,
    device_params: Any = None,
) -> str:
    """七参与项 config_hash（§5.5）。``device`` / ``device_params`` 为预留项，默认 None。"""
    obj: dict[str, Any] = {
        "image_sha256s": sorted(image_sha256s),
        "protocol_version": protocol_version,
        "schema_version": schema_version,
        "client_code_hash": client_code_hash,
        "server_code_hash": server_code_hash,
        "duration_seconds": duration_seconds,
        "seed_strategy": seed_strategy,
        "device": device,
        "device_params": device_params,
    }
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def compute_study_id(metas: list[dict[str, Any]], duration_seconds: float) -> str:
    """study_id = sha256(排序(sha256, source_name) 集合 + 时长)[:16]。

    与 ``make_study_package.compute_study_id`` 完全同口径（图集合身份 + 时长，与图序无关）。
    """
    key = json.dumps(
        sorted((m["sha256"], m["source_name"]) for m in metas), ensure_ascii=False, separators=(",", ":")
    )
    key += "|" + repr(duration_seconds)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def compute_client_code_hash(web_dir: Path) -> str:
    """客户端代码哈希 = 扩展版 ``index.html`` 的文件字节 SHA-256（§5.5 第 3 参与项）。"""
    return sha256_file(web_dir / "index.html")


def compute_server_code_hash(web_dir: Path) -> str:
    """服务端代码哈希 = 对 ``calibration/web/`` 下全部 .py 的 (文件名, sha256) 排序后取哈希。

    （§5.5 第 4 参与项；任一服务端文件改动都会改变该哈希。）
    """
    entries = sorted((p.name, sha256_file(p)) for p in sorted(web_dir.glob("*.py")))
    return sha256_bytes(canonical_json(entries).encode("utf-8"))
