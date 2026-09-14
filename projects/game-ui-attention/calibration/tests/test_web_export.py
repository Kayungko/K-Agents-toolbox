"""export_responses.py 单测（仅 stdlib）：过滤 accepted / 按 participant 拆分 / 剥 _server。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_WEB_DIR = Path(__file__).resolve().parents[1] / "web"
if str(_WEB_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_DIR))

import export_responses as ex  # noqa: E402


def _row(pid: str, accepted: bool) -> dict:
    return {
        "schema_version": "game-ui-attention-calibration-response/v1",
        "participant_id": pid,
        "completed": True,
        "images": [
            {
                "sha256": "a" * 64,
                "responded": True,
                "first_look": {"x": 1, "y": 1},
                "boxes": [{"x": 0, "y": 0, "width": 1, "height": 1}],
            }
        ],
        "_server": {"session_id": "s", "config_hash": "c" * 64, "accepted": accepted},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_export_filters_and_splits(tmp_path: Path) -> None:
    jl = tmp_path / "responses.jsonl"
    _write_jsonl(
        jl,
        [
            _row("p1", True),
            _row("p1", False),  # rejected → 剔除
            _row("p2", True),
            {"participant_id": "p3"},  # 无 _server → 剔除
        ],
    )
    summary = ex.export_from_jsonl(jl, tmp_path / "export")
    assert summary["n_rows_total"] == 4
    assert summary["n_rows_exported"] == 2
    assert summary["n_participants"] == 2
    assert sorted(summary["files"]) == ["response-p1.json", "response-p2.json"]

    p1 = json.loads((tmp_path / "export" / "response-p1.json").read_text(encoding="utf-8"))
    assert "_server" not in p1  # 剥 _server
    assert p1["participant_id"] == "p1"
    assert p1["schema_version"] == "game-ui-attention-calibration-response/v1"  # A1 同构


def test_export_no_accepted_creates_no_files(tmp_path: Path) -> None:
    jl = tmp_path / "responses.jsonl"
    _write_jsonl(jl, [_row("p1", False)])
    summary = ex.export_from_jsonl(jl, tmp_path / "export")
    assert summary["n_rows_exported"] == 0
    assert summary["files"] == []


def test_export_include_rejected(tmp_path: Path) -> None:
    jl = tmp_path / "responses.jsonl"
    _write_jsonl(jl, [_row("p1", True), _row("p1", False)])
    summary = ex.export_from_jsonl(jl, tmp_path / "export", only_accepted=False)
    assert summary["n_rows_exported"] == 2


def test_export_same_participant_last_wins(tmp_path: Path) -> None:
    jl = tmp_path / "responses.jsonl"
    _write_jsonl(jl, [_row("p1", True), {**_row("p1", True), "completed": False}])
    summary = ex.export_from_jsonl(jl, tmp_path / "export")
    assert summary["n_rows_exported"] == 2
    assert summary["n_participants"] == 1
    p1 = json.loads((tmp_path / "export" / "response-p1.json").read_text(encoding="utf-8"))
    assert p1["completed"] is False  # 取最后一份 accepted


def test_safe_filename(tmp_path: Path) -> None:
    assert ex._safe_participant_filename("p/01 x") == "p_01_x"
    assert ex._safe_participant_filename("###") == "participant"
