"""app.py TestClient 集成测试：协议流程 E2E / 并发写 / 删除权 / retire / 非法回传 / config_hash / 零外链 / 种子快照。

依赖 fastapi/httpx（由 C2 线装入共享 venv）；httpx 未到位时整组跳过（importorskip）。
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# 本环境 TestClient 的传输层由 httpx2 提供（无 httpx 模块名）；以 TestClient 是否可导入为准决定跳过。
try:
    from starlette.testclient import TestClient
except ImportError:  # pragma: no cover - 依赖未到位时跳过整组
    pytest.skip("starlette TestClient 不可用（httpx 传输层缺失）", allow_module_level=True)

_WEB_DIR = Path(__file__).resolve().parents[1] / "web"
_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
for _d in (_WEB_DIR, _TOOLS_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import app as web_app  # noqa: E402
import consistency  # noqa: E402
import hashing  # noqa: E402

TOKEN = "test-admin-token"
ADMIN = {"X-Admin-Token": TOKEN}


def _make_png(path: Path, w: int = 64, h: int = 48, color: tuple[int, int, int] = (200, 30, 30)) -> Path:
    Image.new("RGB", (w, h), color).save(path, format="PNG")
    return path


def _make_client(
    tmp_path: Path, *, duration: float = 3.0, seed: int | None = None
) -> tuple[TestClient, web_app.FastAPI]:
    strategy = hashing.SEED_STRATEGY_PER_SESSION if seed is None else f"global:{seed}"
    application = web_app.create_app(
        data_dir=tmp_path / "wc",
        images_dir=tmp_path / "wi",
        admin_token=TOKEN,
        duration_seconds=duration,
        seed_strategy=strategy,
        global_seed=seed,
    )
    return TestClient(application), application


def _upload(client: TestClient, path: Path, screen_type: str = "unknown") -> dict:
    with path.open("rb") as f:
        r = client.post(
            "/admin/tasks",
            files={"file": (path.name, f, "image/png")},
            data={"screen_type": screen_type},
            headers=ADMIN,
        )
    assert r.status_code == 200, r.text
    return r.json()


def _create_session(client: TestClient) -> dict:
    r = client.post("/admin/sessions", headers=ADMIN)
    assert r.status_code == 200, r.text
    return r.json()


def _completed_body(cfg: dict, pid: str) -> dict:
    """由会话 config 构造 A1 同构 completed 回传体（坐标落在图内，逐图错开避免乱点自检误判）。"""
    images = []
    for idx, im in enumerate(cfg["images"]):
        w, h = im["width"], im["height"]
        dx = (idx % 3) * 5
        dy = (idx % 2) * 4
        images.append(
            {
                "index": im["index"],
                "filename": im["filename"],
                "sha256": im["sha256"],
                "width": w,
                "height": h,
                "screen_type": im["screen_type"],
                "display_started_at_utc": "2026-09-13T00:00:00Z",
                "display_ended_at_utc": "2026-09-13T00:00:03Z",
                "display_ms": 3000,
                "responded": True,
                "first_look": {"x": round(w / 2 + dx, 2), "y": round(h / 2 + dy, 2)},
                "boxes": [
                    {
                        "x": round(w * 0.1 + dx, 2),
                        "y": round(h * 0.1 + dy, 2),
                        "width": round(w * 0.3, 2),
                        "height": round(h * 0.3, 2),
                    }
                ],
            }
        )
    return {
        "schema_version": hashing.RESPONSE_SCHEMA_VERSION,
        "study_id": cfg["study_id"],
        "participant_id": pid,
        "started_at_utc": "2026-09-13T00:00:00Z",
        "finished_at_utc": "2026-09-13T00:00:10Z",
        "duration_seconds": cfg["duration_seconds"],
        "shuffle_seed": cfg["shuffle_seed"],
        "user_agent": "test",
        "completed": True,
        "images": images,
    }


def _write_runs(runs_dir: Path, shas: list[str], w: int = 64, h: int = 48) -> None:
    for sha in shas:
        run_dir = runs_dir / f"run-{sha[:12]}"
        run_dir.mkdir(parents=True)
        np.save(run_dir / "density.npy", consistency.center_bias_density((h, w), 0.25))
        analysis = {
            "schema_version": "game-ui-attention-analysis/v1",
            "input": {"image_sha256": sha, "width": w, "height": h},
        }
        (run_dir / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")


# --------------------------------------------------------------------------- E2E


def test_e2e_protocol_flow_and_consistency(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)

    # 1) 上传 3 张图
    shas = []
    files: list[Path] = []
    for i, color in enumerate([(200, 30, 30), (30, 200, 30), (30, 30, 200)]):
        p = tmp_path / f"img{i}.png"
        _make_png(p, color=color)
        files.append(p)
        shas.append(_upload(client, p)["sha256"])
    assert len(set(shas)) == 3

    # 2) 发号
    cfg = _create_session(client)
    sid = cfg["session_id"]
    assert len(cfg["images"]) == 3
    assert len(cfg["config_hash"]) == 64

    # 3) 渲染扩展客户端页 + 会话 config
    page = client.get(f"/s/{sid}")
    assert page.status_code == 200
    assert "注意力校准标注" in page.text

    api = client.get(f"/api/session/{sid}")
    assert api.status_code == 200
    assert api.json()["session_id"] == sid
    assert [im["sha256"] for im in api.json()["images"]] == [im["sha256"] for im in cfg["images"]]

    # 4) 图服务（字节等同由 test_image_bytes_match_upload 单独覆盖）
    for im in cfg["images"]:
        img = client.get(f"/images/{im['filename']}")
        assert img.status_code == 200

    # 5) 回传
    body = _completed_body(cfg, "p1")
    r = client.post("/api/responses", params={"session_id": sid}, json=body)
    assert r.status_code == 200
    receipt = r.json()
    assert receipt["accepted"] is True
    assert receipt["receipt_id"]

    # 6) 导出 → consistency 合成 run 消费出数字
    import export_responses as ex

    summary = ex.export_from_jsonl(tmp_path / "wc" / "responses.jsonl", tmp_path / "export")
    assert summary["n_participants"] == 1

    _write_runs(tmp_path / "runs", shas)
    responses = consistency.load_responses(tmp_path / "export")
    models = consistency.load_model_densities(tmp_path / "runs")
    report = consistency.evaluate(cfg, responses, models, 0.25)
    assert report["n_pairs"] == 3
    assert report["n_participants_included"] == 1
    assert report["aggregate"]["first_look_percentile"]["model_mean"] > 0.5


def test_image_bytes_match_upload(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    p = tmp_path / "img.png"
    _make_png(p)
    up = _upload(client, p)
    r = client.get(f"/images/{up['filename']}")
    assert r.status_code == 200
    assert r.content == p.read_bytes()


# --------------------------------------------------------------------------- 并发写


def test_concurrent_posts_no_loss(tmp_path: Path) -> None:
    client, application = _make_client(tmp_path)
    p = tmp_path / "img.png"
    _make_png(p)
    _upload(client, p)
    cfg = _create_session(client)
    sid = cfg["session_id"]

    n = 8
    barrier = threading.Barrier(n)

    def worker(pid: str) -> int:
        barrier.wait()
        c = TestClient(application)  # 每线程独立 TestClient，共享同一 app（storage）
        r = c.post("/api/responses", params={"session_id": sid}, json=_completed_body(cfg, pid))
        return r.status_code

    with ThreadPoolExecutor(max_workers=n) as ex:
        codes = list(ex.map(worker, [f"p{i}" for i in range(n)]))
    assert codes == [200] * n

    r = client.get("/admin/responses", headers=ADMIN)
    assert r.json()["count"] == n  # 不丢行


# --------------------------------------------------------------------------- 删除权


def test_delete_participant_then_reexport(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    p = tmp_path / "img.png"
    _make_png(p)
    _upload(client, p)
    cfg = _create_session(client)
    sid = cfg["session_id"]

    for pid in ("p1", "p2"):
        r = client.post("/api/responses", params={"session_id": sid}, json=_completed_body(cfg, pid))
        assert r.json()["accepted"] is True

    d = client.delete("/admin/participants/p1", headers=ADMIN)
    assert d.json()["removed_rows"] == 1

    import export_responses as ex

    summary = ex.export_from_jsonl(tmp_path / "wc" / "responses.jsonl", tmp_path / "export")
    assert summary["n_participants"] == 1
    assert summary["files"] == ["response-p2.json"]  # p1 已剔除


# --------------------------------------------------------------------------- retire


def test_retire_keeps_history_resolvable(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    _make_png(p1, color=(200, 30, 30))
    _make_png(p2, color=(30, 200, 30))
    sha1 = _upload(client, p1)["sha256"]
    _upload(client, p2)

    cfg = _create_session(client)  # 2 张都在快照里
    sid = cfg["session_id"]

    r = client.post(f"/admin/tasks/{sha1}/retire", headers=ADMIN)
    assert r.status_code == 200

    # 历史 sha256 仍可解析：文件仍服务、已发会话 config 仍含该图
    im1 = next(im for im in cfg["images"] if im["sha256"] == sha1)
    assert client.get(f"/images/{im1['filename']}").status_code == 200
    assert sha1 in [im["sha256"] for im in client.get(f"/api/session/{sid}").json()["images"]]

    # 新会话不再含已下架图
    cfg2 = _create_session(client)
    assert sha1 not in [im["sha256"] for im in cfg2["images"]]


# --------------------------------------------------------------------------- 非法回传


def test_invalid_response_marked_rejected_but_appended(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    p = tmp_path / "img.png"
    _make_png(p)
    _upload(client, p)
    cfg = _create_session(client)
    sid = cfg["session_id"]

    # 未知 sha256 → accepted=false 仍落盘
    body = _completed_body(cfg, "p1")
    body["images"][0]["sha256"] = "f" * 64
    r = client.post("/api/responses", params={"session_id": sid}, json=body)
    assert r.status_code == 200
    assert r.json()["accepted"] is False

    rows = client.get("/admin/responses", headers=ADMIN).json()["responses"]
    assert len(rows) == 1
    assert rows[0]["_server"]["accepted"] is False
    assert rows[0]["_server"]["config_hash"] == cfg["config_hash"]  # 信封仍记录 config_hash


def test_unknown_session_404(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    assert client.get("/s/nonexistent").status_code == 404
    assert client.get("/api/session/nonexistent").status_code == 404
    assert client.post("/api/responses", params={"session_id": "nonexistent"}, json={}).status_code == 404


# --------------------------------------------------------------------------- config_hash / 快照 / 安全


def test_config_hash_reproducible_across_sessions(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path, seed=42)
    p = tmp_path / "img.png"
    _make_png(p)
    _upload(client, p)
    c1 = _create_session(client)
    c2 = _create_session(client)
    assert c1["config_hash"] == c2["config_hash"]
    assert c1["shuffle_seed"] == c2["shuffle_seed"] == 42


def test_seed_snapshot_semantics_add_image(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    p1 = tmp_path / "a.png"
    _make_png(p1)
    _upload(client, p1)
    sa = _create_session(client)
    assert len(sa["images"]) == 1

    p2 = tmp_path / "b.png"
    _make_png(p2, color=(30, 200, 30))
    _upload(client, p2)
    sb = _create_session(client)
    assert len(sb["images"]) == 2

    # 已发会话不受加稿影响
    assert len(client.get(f"/api/session/{sa['session_id']}").json()["images"]) == 1


def test_admin_token_required(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    assert client.get("/admin/tasks").status_code == 401
    assert client.post("/admin/sessions").status_code == 401
    assert client.get("/admin/responses").status_code == 401


def test_image_path_traversal_guarded(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    assert client.get("/images/../../etc/passwd").status_code == 404
    assert client.get("/images/not-a-sha.png").status_code == 404


_BANNED = (
    "http://",
    "https://",
    "data:image",
    "data:text",
    "sendBeacon",
    "XMLHttpRequest",
    "<script src=",
    "<link rel=",
    "@import",
)


def test_serve_html_zero_external(tmp_path: Path) -> None:
    client, _ = _make_client(tmp_path)
    p = tmp_path / "img.png"
    _make_png(p)
    _upload(client, p)
    cfg = _create_session(client)

    for url in (f"/s/{cfg['session_id']}", "/admin"):
        resp = client.get(url)
        assert resp.status_code == 200
        html = resp.text
        for banned in _BANNED:
            assert banned not in html, f"{url} 含外链/外发特征：{banned}"
        # 所有 fetch 目标必须为同源相对路径
        for call in _fetch_calls(html):
            assert call.startswith("/"), f"{url} 存在非相对 fetch：{call}"


def _fetch_calls(html: str) -> list[str]:
    """提取 fetch( 的字面量首参 URL（首参为变量时跳过）。"""
    out = []
    idx = 0
    marker = "fetch("
    while True:
        i = html.find(marker, idx)
        if i == -1:
            return out
        p = i + len(marker)
        while p < len(html) and html[p] in " \t":
            p += 1
        if p < len(html) and html[p] in "\"'":
            q = html[p]
            end = html.find(q, p + 1)
            if end != -1:
                out.append(html[p + 1 : end])
        idx = i + len(marker)
