"""responses 导出 CLI（calibration/web）。

读 ``responses.jsonl`` → 过滤 ``_server.accepted=true`` → 按 participant_id 拆分写 A1 同构 JSON
（剥除 ``_server`` 信封）到导出目录 → ``consistency.py --responses <导出目录>`` 直接消费。

删除权重跑：``DELETE /admin/participants/{id}`` 删行后重新运行本脚本，即自动剔除该人，
``consistency.py`` 重跑即得删后报告（protocol §8 删除权条款的 web 化实现，A4 §5.6）。

自包含：仅 stdlib（零 fastapi / PIL / numpy），可独立运行与测试。

用法::

    python calibration/web/export_responses.py \
        --responses local-data/web-calibration/responses.jsonl \
        --out local-data/web-calibration/export
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESPONSES = PROJECT_ROOT / "local-data" / "web-calibration" / "responses.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "local-data" / "web-calibration" / "export"


def _safe_participant_filename(pid: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_-]", "_", str(pid)).strip("_") or "participant"


def export_from_jsonl(responses_path: str | Path, out_dir: str | Path, *, only_accepted: bool = True) -> dict:
    """读 JSONL → 过滤 accepted → 按 participant 拆分写 A1 同构 JSON（剥 _server）。"""
    responses_path = Path(responses_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if responses_path.is_file():
        for line in responses_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    total = len(rows)
    if only_accepted:
        rows = [r for r in rows if (r.get("_server") or {}).get("accepted") is True]

    by_pid: dict[str, list[dict]] = {}
    for r in rows:
        pid = r.get("participant_id") or "unknown"
        by_pid.setdefault(pid, []).append(r)

    files: list[str] = []
    for pid, items in sorted(by_pid.items()):
        safe = _safe_participant_filename(pid)
        target = out / f"response-{safe}.json"
        # A1 形态 = 每 participant 单份对象（consistency.load_responses 按 dict 逐文件消费）；
        # 同一 participant 多次 accepted 提交时按「最后一份」导出（append 顺序最新在后）。
        payload = items[-1]
        stripped = {k: v for k, v in payload.items() if k != "_server"}
        target.write_text(json.dumps(stripped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        files.append(target.name)

    return {
        "n_rows_total": total,
        "n_rows_exported": len(rows),
        "n_participants": len(by_pid),
        "files": sorted(files),
        "out_dir": str(out),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_responses.py", description="导出 responses 为 A1 同构 JSON（按 participant 拆分）"
    )
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES, help="responses.jsonl 路径")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="导出目录（默认 local-data/web-calibration/export）"
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="同时导出 _server.accepted=false 的行（默认仅导出 accepted）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = export_from_jsonl(args.responses, args.out, only_accepted=not args.include_rejected)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
