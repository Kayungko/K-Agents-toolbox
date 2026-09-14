"""Web 协作校准平台 FastAPI 应用（calibration/web）。

端点全集对齐 A4 建议书 §3.1/§5.3 与任务书 §1：

- GET  /s/{session_id}                    参与者标注页（扩展版 index.html）
- GET  /api/session/{session_id}          会话 config（图清单+sha256+按会话种子 shuffle 的图序+时长+config_hash）
- GET  /images/{filename}                 只读图服务（local-data/web-images）
- POST /api/responses                     校验后 JSONL append+fsync，返回回执 id（非法行 accepted=false 仍落盘）
- admin（X-Admin-Token）：
  - POST   /admin/tasks                    multipart 上传 → sha256/尺寸/screen_type 登记 active=true（运行期热加）
  - GET    /admin/tasks                    图库列表
  - POST   /admin/tasks/{sha256}/retire    soft-delete（active=false，不物理删除）
  - GET    /admin/responses                导出 responses（JSON 数组）
  - DELETE /admin/participants/{pid}       删除该人全部 JSONL 行（删除权）
  - GET    /admin/sessions                 会话列表
  - POST   /admin/sessions                 新建会话（发号；A4 端点表未列，为管理页发号所需的最小补齐）
  - GET    /admin                          管理页（admin.html）
  - GET    /admin                          管理页（admin.html）

安全：bind 默认 127.0.0.1（--host 可配内网接口）；admin token 启动生成打印或读环境变量
CALIB_ADMIN_TOKEN；访问日志脱敏（不落 IP 原文 / 会话 token / admin token）；serve 的 HTML 零外链。

运行：``python app.py --host 127.0.0.1 --port 8000``（或 ``uvicorn app:create_app --factory``）。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import secrets
import sys
import threading
from pathlib import Path
from typing import Annotated, Any

import hashing
import store as store_mod
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "local-data" / "web-calibration"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "local-data" / "web-images"
WEB_DIR = Path(__file__).resolve().parent

_ADMIN_TOKEN_ENV = "CALIB_ADMIN_TOKEN"
_UPLOAD_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
_FILENAME_RE = re.compile(r"^[0-9a-f]{64}\.[a-zA-Z0-9]{1,10}$")


def _redact_path(path: str) -> str:
    """访问日志脱敏：遮蔽 session token / admin token 类路径段，不落 IP 原文。"""
    return re.sub(r"/(s|api/session)/[^/]+", r"/\1/[redacted]", path)


class _SanitizedAccessLog:
    """脱敏访问日志：仅记 {time, method, path, status}，绝不记录 remote_addr / header / token 原文。"""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "access.log"
        self._lock = threading.Lock()

    def write(self, method: str, path: str, status: int) -> None:
        line = json.dumps(
            {"time": store_mod._utcnow(), "method": method, "path": _redact_path(path), "status": status},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError:  # 日志写失败不影响业务（内网只读磁盘等异常）
            return


def create_app(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    images_dir: str | Path = DEFAULT_IMAGES_DIR,
    admin_token: str | None = None,
    duration_seconds: float = 3.0,
    seed_strategy: str = hashing.SEED_STRATEGY_PER_SESSION,
    global_seed: int | None = None,
) -> FastAPI:
    """应用工厂：显式参数便于测试注入 tmp 目录与固定 token；无模块级副作用。"""
    data_dir = Path(data_dir)
    images_dir = Path(images_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    client_code_hash = hashing.compute_client_code_hash(WEB_DIR)
    server_code_hash = hashing.compute_server_code_hash(WEB_DIR)

    storage = store_mod.Storage(
        data_dir=data_dir,
        images_dir=images_dir,
        duration_seconds=duration_seconds,
        seed_strategy=seed_strategy,
        global_seed=global_seed,
        client_code_hash=client_code_hash,
        server_code_hash=server_code_hash,
    )

    if admin_token is None:
        admin_token = os.environ.get(_ADMIN_TOKEN_ENV) or secrets.token_urlsafe(32)
    access_log = _SanitizedAccessLog(data_dir)

    app = FastAPI(title="Game UI Attention Web Calibration", docs_url=None, redoc_url=None, openapi_url=None)

    # ------------------------------------------------------------------ auth

    def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
        if x_admin_token is None or not secrets.compare_digest(x_admin_token, admin_token):
            raise HTTPException(status_code=401, detail="unauthorized")

    # ------------------------------------------------------------------ middleware

    @app.middleware("http")
    async def _access_log_middleware(request: Request, call_next):
        response = await call_next(request)
        access_log.write(request.method, request.url.path, response.status_code)
        return response

    # ------------------------------------------------------------------ participant

    @app.get("/s/{session_id}")
    def session_page(session_id: str):
        if storage.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return FileResponse(WEB_DIR / "index.html", media_type="text/html")

    @app.get("/api/session/{session_id}")
    def api_session(session_id: str):
        cfg = storage.get_session(session_id)
        if cfg is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return JSONResponse(cfg)

    @app.get("/images/{filename}")
    def get_image(filename: str):
        if not _FILENAME_RE.match(filename):
            raise HTTPException(status_code=404, detail="not found")
        root = images_dir.resolve()
        path = (images_dir / filename).resolve()
        if path.parent != root or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)

    @app.post("/api/responses")
    async def post_responses(request: Request):
        session_id = request.query_params.get("session_id", "")
        session = storage.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - 损坏 body 也要落盘排查
            body = None

        accepted, reason = store_mod.validate_response(
            body, session, storage.all_sha256s(), storage.dims_by_sha()
        )
        envelope: dict[str, Any] = {
            "session_id": session_id,
            "received_at_utc": store_mod._utcnow(),
            "config_hash": session["config_hash"],
            "client_version": client_code_hash,
            "server_version": server_code_hash,
            "device": None,
            "device_params": None,
            "remote_addr_redacted": True,
            "accepted": accepted,
            "validation_reason": reason,
        }
        row: dict[str, Any] = {"_server": envelope}
        if isinstance(body, dict):
            row = {**body, "_server": envelope}
        receipt = storage.append_response(row)
        return JSONResponse({"receipt_id": receipt, "accepted": accepted, "validation_reason": reason})

    # ------------------------------------------------------------------ admin

    @app.get("/admin")
    def admin_page():
        return FileResponse(WEB_DIR / "admin.html", media_type="text/html")

    @app.post("/admin/tasks")
    def admin_upload(
        file: Annotated[UploadFile, File()],
        screen_type: Annotated[str, Form()] = "unknown",
        source_name: Annotated[str | None, Form()] = None,
        _admin: None = Depends(require_admin),
    ):
        data = file.file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")
        sha = hashing.sha256_bytes(data)
        ext = Path(file.filename or "upload.png").suffix.lower()
        if ext not in _UPLOAD_EXTENSIONS:
            ext = ".png"
        stored_name = sha + ext
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                if img.width <= 0 or img.height <= 0:
                    raise HTTPException(status_code=400, detail="invalid image size")
                width, height = int(img.width), int(img.height)
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail=f"undecodable image: {exc}") from exc

        (images_dir / stored_name).write_bytes(data)
        src = source_name or file.filename or stored_name
        result = storage.add_task(sha, stored_name, src, width, height, screen_type)
        return JSONResponse(
            {
                "sha256": sha,
                "filename": stored_name,
                "width": width,
                "height": height,
                "screen_type": screen_type,
                "already_existed": result["already_existed"],
            }
        )

    @app.get("/admin/tasks")
    def admin_list_tasks(_admin: None = Depends(require_admin)):
        return JSONResponse({"tasks": storage.list_tasks()})

    @app.post("/admin/tasks/{sha256}/retire")
    def admin_retire(sha256: str, _admin: None = Depends(require_admin)):
        if storage.retire_task(sha256):
            return JSONResponse({"ok": True, "sha256": sha256, "active": False})
        raise HTTPException(status_code=404, detail="unknown sha256")

    @app.get("/admin/responses")
    def admin_responses(_admin: None = Depends(require_admin)):
        rows = storage.read_responses()
        return JSONResponse({"count": len(rows), "responses": rows})

    @app.delete("/admin/participants/{participant_id}")
    def admin_delete_participant(participant_id: str, _admin: None = Depends(require_admin)):
        removed = storage.delete_participant(participant_id)
        return JSONResponse({"ok": True, "participant_id": participant_id, "removed_rows": removed})

    @app.get("/admin/sessions")
    def admin_sessions(_admin: None = Depends(require_admin)):
        return JSONResponse({"sessions": storage.list_sessions()})

    @app.post("/admin/sessions")
    def admin_create_session(_admin: None = Depends(require_admin)):
        return JSONResponse(storage.create_session())

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.py", description="内网 Web 协作校准平台（FastAPI）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1；内网可配本机局域网 IP）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="SQLite/JSONL 数据目录")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="图片库目录")
    parser.add_argument("--duration", type=float, default=3.0, help="每图观看秒数（默认 3，可配 5）")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="全局图序种子（给定则所有会话同序；默认每会话独立随机种子）",
    )
    parser.add_argument(
        "--admin-token",
        default=None,
        help=f"admin token（默认读环境变量 {_ADMIN_TOKEN_ENV}；均缺省则启动时随机生成并打印）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    token = args.admin_token or os.environ.get(_ADMIN_TOKEN_ENV)
    generated = token is None
    if token is None:
        token = secrets.token_urlsafe(32)

    seed_strategy = hashing.SEED_STRATEGY_PER_SESSION if args.seed is None else f"global:{args.seed}"

    app = create_app(
        data_dir=args.data_dir,
        images_dir=args.images_dir,
        admin_token=token,
        duration_seconds=args.duration,
        seed_strategy=seed_strategy,
        global_seed=args.seed,
    )

    # 启动提示（token 仅打印到控制台，禁入日志文件原文）
    print(f"[web-calibration] bind={args.host}:{args.port} data_dir={args.data_dir} images_dir={args.images_dir}")
    if generated:
        print(f"[web-calibration] generated admin token: {token}")
    else:
        print("[web-calibration] admin token: 来自命令行/环境变量（不打印原文）")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
