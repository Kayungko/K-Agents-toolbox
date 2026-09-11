"""L2 串行整合测试：进程级全链路（真实子进程 + 真实 foveacast 后端）。

与 C1 的进程内单测（test_contracts_cli.py）互补：本文件通过 subprocess 调用
`python -m ui_attention.cli`，验证真实进程边界上的退出码、stdout JSON 信封、
产物落盘与跨线集成（C1 契约/统计 × C2 后端 × C3 渲染）。

所有权：二级总控（tests/integration/ 由 L2 串行维护，见 implementation-plan.md）。
运行：项目根 cwd 下 `.venv\\Scripts\\python.exe -m pytest tests/integration -q`。
权重缺失时整体跳过（下载属 C2 显式安装步骤）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
WEIGHTS = PROJECT_ROOT / "model-cache" / "foveacast" / "foveacast-v3-3s-fp16.onnx"

pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(),
    reason="foveacast 权重未安装（python -m ui_attention.backends 显式安装后运行）",
)


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_cli(tmp_path: Path, argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ui_attention.cli", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env=_env(),
        timeout=180,
    )


def _make_input_png(path: Path, w: int = 640, h: int = 480) -> None:
    from PIL import Image

    xs = np.linspace(0.0, 255.0, w, dtype=np.float64)
    ys = np.linspace(0.0, 255.0, h, dtype=np.float64)
    r = np.tile(xs, (h, 1))
    g = np.tile(ys[:, None], (1, w))
    b = np.full((h, w), 96.0)
    arr = np.dstack([r, g, b]).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def _make_request(dirpath: Path, *, profile: str = "foveacast-onnx-3s-v1", zero_area: bool = False) -> Path:
    _make_input_png(dirpath / "input.png")
    width = 0 if zero_area else 120
    req = {
        "schema_version": "game-ui-attention-request/v1",
        "image": "./input.png",
        "player_goal": "整合测试：查看主按钮",
        "screen_type": "reward-summary",
        "backend_profile": profile,
        "regions": [
            {
                "id": "main-button",
                "label": "主按钮",
                "role": "primary-action",
                "geometry": {"type": "rect", "x": 100, "y": 80, "width": width, "height": 60},
                "source": "manual",
                "status": "confirmed",
            }
        ],
    }
    p = dirpath / "request.json"
    p.write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _envelope(proc: subprocess.CompletedProcess) -> dict:
    assert proc.stdout.strip(), f"stdout 应为 JSON 信封, stderr={proc.stderr[-500:]}"
    return json.loads(proc.stdout)


def test_analyze_full_chain(tmp_path: Path) -> None:
    """analyze 真实后端全链路：exit 0、信封 ok、产物齐全、manifest 哈希互验、density sum=1。"""
    req = _make_request(tmp_path)
    out = tmp_path / "run-a"
    proc = _run_cli(tmp_path, ["analyze", "--request", str(req), "--out", str(out)])
    assert proc.returncode == 0, proc.stderr[-800:]
    env = _envelope(proc)
    assert env["ok"] is True

    for name in ("manifest.json", "analysis.json", "regions.json", "density.npy",
                 "overlay.png", "heatmap.png", "report.html"):
        assert (out / name).exists(), f"缺少产物 {name}"

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["computation_status"] == "complete"
    listed = {a["path"] for a in manifest["artifacts"]}
    assert {"analysis.json", "density.npy", "report.html"} <= listed

    import hashlib

    for art in manifest["artifacts"]:
        data = (out / art["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == art["sha256"], f"manifest 哈希不符: {art['path']}"
        assert len(data) == art["size_bytes"]

    density = np.load(out / "density.npy")
    assert density.dtype == np.float64
    assert np.isfinite(density).all() and (density >= 0).all()
    assert abs(float(density.sum()) - 1.0) < 1e-6

    # manifest 最后落盘（完成语义）：mtime 不早于任何产物
    m_mtime = (out / "manifest.json").stat().st_mtime_ns
    for art in manifest["artifacts"]:
        assert (out / art["path"]).stat().st_mtime_ns <= m_mtime

    # C3 接线：manifest 含 overlay 与 colorscale 共用参数
    assert manifest["overlay"]["colorscale"]["name"] == manifest["colorscale"]["name"]


def test_summarize_no_reinference(tmp_path: Path) -> None:
    """summarize 只重算区域：density.npy 哈希与 analyze 完全一致，区域统计更新。"""
    req = _make_request(tmp_path)
    run_a = tmp_path / "run-a"
    assert _run_cli(tmp_path, ["analyze", "--request", str(req), "--out", str(run_a)]).returncode == 0

    regions = json.loads((run_a / "regions.json").read_text(encoding="utf-8"))
    regions["regions"][0]["geometry"]["width"] = 240  # 扩大区域
    regions2 = tmp_path / "regions2.json"
    regions2.write_text(json.dumps(regions, ensure_ascii=False), encoding="utf-8")

    out_s = tmp_path / "run-s"
    proc = _run_cli(tmp_path, ["summarize", "--analysis", str(run_a),
                               "--regions", str(regions2), "--out", str(out_s)])
    assert proc.returncode == 0, proc.stderr[-800:]
    assert _envelope(proc)["ok"] is True

    import hashlib

    h1 = hashlib.sha256((run_a / "density.npy").read_bytes()).hexdigest()
    h2 = hashlib.sha256((out_s / "density.npy").read_bytes()).hexdigest()
    assert h1 == h2, "summarize 不得重新推理（density 哈希必须不变）"

    a1 = json.loads((run_a / "analysis.json").read_text(encoding="utf-8"))
    a2 = json.loads((out_s / "analysis.json").read_text(encoding="utf-8"))
    m1 = {r["id"]: r for r in a1["regions"]}["main-button"]
    m2 = {r["id"]: r for r in a2["regions"]}["main-button"]
    assert m2["area_px"] > m1["area_px"], "扩大区域后面积应增大"


def test_compare_chain(tmp_path: Path) -> None:
    """compare 全链：exit 0、comparison.json 配对与 delta_pp、A/B 共用色阶参数入 manifest。"""
    req = _make_request(tmp_path)
    run_a = tmp_path / "run-a"
    assert _run_cli(tmp_path, ["analyze", "--request", str(req), "--out", str(run_a)]).returncode == 0

    req_b = json.loads((tmp_path / "request.json").read_text(encoding="utf-8"))
    req_b["regions"][0]["geometry"].update({"x": 300, "y": 200})
    req_b_path = tmp_path / "request-b.json"
    req_b_path.write_text(json.dumps(req_b, ensure_ascii=False), encoding="utf-8")
    run_b = tmp_path / "run-b"
    assert _run_cli(tmp_path, ["analyze", "--request", str(req_b_path), "--out", str(run_b)]).returncode == 0

    out_c = tmp_path / "cmp"
    proc = _run_cli(tmp_path, ["compare", "--before", str(run_a), "--after", str(run_b), "--out", str(out_c)])
    assert proc.returncode == 0, proc.stderr[-800:]
    cmp_json = json.loads((out_c / "comparison.json").read_text(encoding="utf-8"))
    matched = cmp_json["regions"]["matched"]
    assert any(r["id"] == "main-button" for r in matched)
    mb = next(r for r in matched if r["id"] == "main-button")
    assert isinstance(mb["delta_pp"], float)
    man = json.loads((out_c / "manifest.json").read_text(encoding="utf-8"))
    assert man["colorscale"]["name"], "compare manifest 必须记录共用色阶参数"


def test_invalid_aoi_exit2(tmp_path: Path) -> None:
    """零面积 AOI → 退出码 2 + 结构化错误 + 不产生输出目录。"""
    req = _make_request(tmp_path, zero_area=True)
    out = tmp_path / "run-bad"
    proc = _run_cli(tmp_path, ["analyze", "--request", str(req), "--out", str(out)])
    assert proc.returncode == 2
    env = _envelope(proc)
    assert env["ok"] is False
    assert env["error"]["id"] == "INVALID_AOI"
    assert env["error"]["exit_code"] == 2
    assert not out.exists() or not any(out.iterdir()), "失败不得留下部分产物冒充成功"


def test_unregistered_profile_exit3(tmp_path: Path) -> None:
    """未登记 backend_profile → 退出码 3（后端未就绪语义）。"""
    req = _make_request(tmp_path, profile="no-such-profile-v1")
    out = tmp_path / "run-noprofile"
    proc = _run_cli(tmp_path, ["analyze", "--request", str(req), "--out", str(out)])
    assert proc.returncode == 3
    assert _envelope(proc)["ok"] is False


def test_existing_out_dir_exit7(tmp_path: Path) -> None:
    """输出目录已存在 → 退出码 7，不覆盖历史。"""
    req = _make_request(tmp_path)
    out = tmp_path / "run-a"
    assert _run_cli(tmp_path, ["analyze", "--request", str(req), "--out", str(out)]).returncode == 0
    before = (out / "manifest.json").read_bytes()
    proc = _run_cli(tmp_path, ["analyze", "--request", str(req), "--out", str(out)])
    assert proc.returncode == 7
    assert _envelope(proc)["ok"] is False
    assert (out / "manifest.json").read_bytes() == before, "历史产物不得被覆盖"


def test_doctor_exit0(tmp_path: Path) -> None:
    """doctor 真实后端就绪 → exit 0 + ok 信封 + 许可记录在案。"""
    proc = _run_cli(tmp_path, ["doctor"])
    assert proc.returncode == 0, proc.stderr[-800:]
    env = _envelope(proc)
    assert env["ok"] is True
