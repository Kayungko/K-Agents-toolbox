"""存储层（calibration/web 共用；仅 stdlib：sqlite3/json/secrets/random/hashlib/pathlib/threading）。

- 图库 / 会话 / 种子 / added_at → SQLite（WAL 模式，多读单写、崩溃可恢复）；
- responses → ``responses.jsonl`` append-only（单行追加 + fsync，进程内 ``threading.Lock`` 串行化追加）；
- 图片本体 → ``images_dir``（上传时按 sha256 命名，内容寻址去重）。

零 fastapi / PIL 依赖，可独立单测（test_web_store.py）。

关键语义（A4 §5.3）：

- **快照式发号**：会话创建时冻结图清单（``images_json``），进行中会话不受后续加稿/下架影响；
- **下架 = soft-delete**：``active=false``，不物理删图，保证历史 response 的 sha256 引用始终可解析；
- **删除权**：按 participant_id 删 JSONL 行（重写文件原子替换），重导出即自动剔除该人。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import secrets
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hashing

_CHUNK = 1 << 20

# 图库默认排序键（确定性：按 sha256 升序），保证「同一图集合 + 同一种子 → 同一图序」可复现。
_TASKS_ORDER_BY = "sha256 ASC"


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_response(
    body: Any,
    session_config: dict[str, Any],
    known_sha256s: set[str],
    dims_by_sha: dict[str, tuple[int, int]],
) -> tuple[bool, str | None]:
    """最小回传校验（A4 §5.3）：返回 ``(accepted, reason)``；非法不丢弃，仅标记。

    校验项：schema_version 匹配 / 每图 sha256 ∈ 当前或历史图库 / 坐标钳制（越界兜底）/
    ``completed`` 标志一致。``session_config`` 用于核对图数与 completed 一致性。
    """
    if not isinstance(body, dict):
        return False, "body 非 JSON 对象"
    if body.get("schema_version") != hashing.RESPONSE_SCHEMA_VERSION:
        return False, f"schema_version 不匹配：{body.get('schema_version')!r}"
    images = body.get("images")
    if not isinstance(images, list):
        return False, "images 非列表"

    session_images = {im["sha256"]: im for im in session_config.get("images", [])}
    completed = bool(body.get("completed", False))

    for entry in images:
        if not isinstance(entry, dict):
            return False, "images 含非对象条目"
        sha = entry.get("sha256")
        if not isinstance(sha, str) or sha not in known_sha256s:
            return False, f"sha256 不在图库：{sha!r}"
        dims = dims_by_sha.get(sha)
        if entry.get("responded") and dims is not None:
            if not _coords_in_bounds(entry, dims[0], dims[1]):
                return False, f"坐标越界：{sha}"

    if completed:
        if len(images) != len(session_images):
            return False, f"completed=true 但图数 {len(images)} != 会话图数 {len(session_images)}"
        for entry in images:
            if not entry.get("responded"):
                return False, "completed=true 但存在未作答图"
            if not entry.get("first_look") or not entry.get("boxes"):
                return False, "completed=true 但缺第一眼落点或主看框"
    return True, None


def _coords_in_bounds(entry: dict[str, Any], w: int, h: int) -> bool:
    point = entry.get("first_look")
    if not isinstance(point, dict):
        return False
    x, y = point.get("x"), point.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return False
    if not (0.0 <= float(x) < w and 0.0 <= float(y) < h):
        return False
    for b in entry.get("boxes", []):
        if not isinstance(b, dict):
            return False
        bx, by, bw, bh = b.get("x"), b.get("y"), b.get("width"), b.get("height")
        if not all(isinstance(v, (int, float)) for v in (bx, by, bw, bh)):
            return False
        if float(bw) < 0 or float(bh) < 0:
            return False
        x_ok = 0.0 <= float(bx) <= w and float(bx) + float(bw) <= w
        y_ok = 0.0 <= float(by) <= h and float(by) + float(bh) <= h
        if not (x_ok and y_ok):
            return False
    return True


class Storage:
    """图库 + 会话 + responses 的持久化门面（SQLite WAL + JSONL append-only）。"""

    def __init__(
        self,
        data_dir: str | Path,
        images_dir: str | Path,
        *,
        duration_seconds: float = 3.0,
        seed_strategy: str = hashing.SEED_STRATEGY_PER_SESSION,
        global_seed: int | None = None,
        client_code_hash: str = "",
        server_code_hash: str = "",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.images_dir = Path(images_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "calibration.db"
        self.responses_path = self.data_dir / "responses.jsonl"

        self.duration_seconds = float(duration_seconds)
        self.seed_strategy = seed_strategy
        self.global_seed = global_seed
        self.client_code_hash = client_code_hash
        self.server_code_hash = server_code_hash

        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------ SQLite

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    sha256        TEXT PRIMARY KEY,
                    filename      TEXT NOT NULL,
                    source_name   TEXT NOT NULL,
                    width         INTEGER NOT NULL,
                    height        INTEGER NOT NULL,
                    screen_type   TEXT NOT NULL DEFAULT 'unknown',
                    active        INTEGER NOT NULL DEFAULT 1,
                    added_at_utc  TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id       TEXT PRIMARY KEY,
                    study_id         TEXT NOT NULL,
                    config_hash      TEXT NOT NULL,
                    shuffle_seed     INTEGER NOT NULL,
                    seed_strategy    TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    images_json      TEXT NOT NULL,
                    created_at_utc   TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ tasks

    def add_task(
        self, sha256: str, filename: str, source_name: str, width: int, height: int, screen_type: str
    ) -> dict[str, Any]:
        """登记图库（上传后调用）。幂等：同 sha256 已存在时仅更新 screen_type 并恢复 active，不覆盖首次 source_name。"""
        now = _utcnow()
        conn = self._connect()
        try:
            row = conn.execute("SELECT source_name FROM tasks WHERE sha256 = ?", (sha256,)).fetchone()
            if row is not None:
                conn.execute("UPDATE tasks SET screen_type = ?, active = 1 WHERE sha256 = ?", (screen_type, sha256))
                conn.commit()
                return {"sha256": sha256, "already_existed": True, "source_name": row["source_name"]}
            conn.execute(
                "INSERT INTO tasks (sha256, filename, source_name, width, height, screen_type, active, added_at_utc) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                (sha256, filename, source_name, int(width), int(height), screen_type, now),
            )
            conn.commit()
            return {"sha256": sha256, "already_existed": False, "source_name": source_name}
        finally:
            conn.close()

    def list_tasks(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT sha256, filename, source_name, width, height, screen_type, active, added_at_utc "
                "FROM tasks ORDER BY added_at_utc ASC, sha256 ASC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_task(self, sha256: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT sha256, filename, source_name, width, height, screen_type, active, added_at_utc "
                "FROM tasks WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def active_images(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT sha256, filename, source_name, width, height, screen_type FROM tasks "
                f"WHERE active = 1 ORDER BY {_TASKS_ORDER_BY}"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def retire_task(self, sha256: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute("UPDATE tasks SET active = 0 WHERE sha256 = ? AND active = 1", (sha256,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def all_sha256s(self) -> set[str]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT sha256 FROM tasks").fetchall()
            return {r["sha256"] for r in rows}
        finally:
            conn.close()

    def dims_by_sha(self) -> dict[str, tuple[int, int]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT sha256, width, height FROM tasks").fetchall()
            return {r["sha256"]: (int(r["width"]), int(r["height"])) for r in rows}
        finally:
            conn.close()

    # ------------------------------------------------------------------ sessions

    def create_session(self) -> dict[str, Any]:
        """快照式发号：冻结当前 active 图清单，按会话种子 shuffle，写入 session 记录。

        - 进行中/已发会话不受后续加稿、下架影响（图序快照）；
        - shuffle_seed：``per-session`` 策略为随机整数，``global:<seed>`` 策略为固定值；
        - config_hash 依据七参与项计算（图集合 + 协议/schema + 客户端/服务端代码 + 时长 + 种子策略 + device=None）。
        """
        images = self.active_images()
        if not images:
            raise ValueError("图库为空：请先通过 POST /admin/tasks 上传至少一张图片后再发号")

        shuffle_seed = self._resolve_seed()
        order = list(range(len(images)))
        random.Random(shuffle_seed).shuffle(order)

        ordered: list[dict[str, Any]] = []
        for presentation_index, src_idx in enumerate(order):
            src = images[src_idx]
            ordered.append(
                {
                    "index": presentation_index,
                    "filename": src["filename"],
                    "source_name": src["source_name"],
                    "sha256": src["sha256"],
                    "width": int(src["width"]),
                    "height": int(src["height"]),
                    "screen_type": src["screen_type"],
                }
            )

        metas = [{"sha256": im["sha256"], "source_name": im["source_name"]} for im in images]
        study_id = hashing.compute_study_id(metas, self.duration_seconds)
        config_hash = hashing.compute_config_hash(
            image_sha256s=[im["sha256"] for im in images],
            protocol_version=hashing.PROTOCOL_VERSION,
            schema_version=hashing.RESPONSE_SCHEMA_VERSION,
            client_code_hash=self.client_code_hash,
            server_code_hash=self.server_code_hash,
            duration_seconds=self.duration_seconds,
            seed_strategy=self.seed_strategy,
        )

        session_id = secrets.token_urlsafe(32)
        now = _utcnow()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO sessions "
                "(session_id, study_id, config_hash, shuffle_seed, seed_strategy, "
                "duration_seconds, images_json, created_at_utc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    study_id,
                    config_hash,
                    shuffle_seed,
                    self.seed_strategy,
                    self.duration_seconds,
                    json.dumps(ordered, ensure_ascii=False, separators=(",", ":")),
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return self._session_config(
            session_id, study_id, config_hash, shuffle_seed, ordered, now, self.duration_seconds
        )

    def _resolve_seed(self) -> int:
        if self.global_seed is not None:
            return int(self.global_seed)
        return secrets.randbelow(2**31)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        images = json.loads(row["images_json"])
        return self._session_config(
            row["session_id"],
            row["study_id"],
            row["config_hash"],
            row["shuffle_seed"],
            images,
            row["created_at_utc"],
            row["duration_seconds"],
        )

    @staticmethod
    def _session_config(
        session_id: str,
        study_id: str,
        config_hash: str,
        shuffle_seed: int,
        images: list[dict[str, Any]],
        created_at: str,
        duration_seconds: float,
    ) -> dict[str, Any]:
        return {
            "schema_version": hashing.SESSION_SCHEMA_VERSION,
            "session_id": session_id,
            "study_id": study_id,
            "config_hash": config_hash,
            "protocol_version": hashing.PROTOCOL_VERSION,
            "shuffle_seed": shuffle_seed,
            "duration_seconds": duration_seconds,
            "images": images,
            "created_at_utc": created_at,
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM sessions ORDER BY created_at_utc ASC").fetchall()
            out: list[dict[str, Any]] = []
            for row in rows:
                out.append(
                    {
                        "session_id": row["session_id"],
                        "study_id": row["study_id"],
                        "config_hash": row["config_hash"],
                        "shuffle_seed": row["shuffle_seed"],
                        "seed_strategy": row["seed_strategy"],
                        "duration_seconds": row["duration_seconds"],
                        "created_at_utc": row["created_at_utc"],
                        "n_images": len(json.loads(row["images_json"])),
                    }
                )
            return out
        finally:
            conn.close()

    # ------------------------------------------------------------------ responses

    def append_response(self, row: dict[str, Any]) -> str:
        """单行追加 + fsync；返回回执 id（该行字节 SHA-256，幂等标识）。"""
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with self.responses_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        return hashlib.sha256(line.encode("utf-8")).hexdigest()

    def read_responses(self) -> list[dict[str, Any]]:
        if not self.responses_path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self.responses_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            out.append(json.loads(line))
        return out

    def delete_participant(self, participant_id: str) -> int:
        """删除权：重写 JSONL 剔除该 participant 的全部行（原子替换 + fsync）。"""
        rows = self.read_responses()
        kept = [r for r in rows if r.get("participant_id") != participant_id]
        removed = len(rows) - len(kept)
        if removed:
            with self._lock:
                tmp = self.responses_path.with_suffix(".jsonl.tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    for r in kept:
                        f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.responses_path)
        return removed
