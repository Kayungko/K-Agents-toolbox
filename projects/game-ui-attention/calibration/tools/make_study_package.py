"""研究包生成器（calibration/protocol.md §3/§9）。

输入：图片目录 + 配置（观看时长 / 种子 / 图列表）→ 生成研究包目录：

::

    <out>/
      images/            图片副本（字节等同源文件，sha256 与 analyze 的 image_sha256 一致）
      index.html         实例化后的纯客户端标注页（相对路径引用图片，禁止 data:URI 内嵌大图）
      config.json        图清单（展示顺序）+ sha256 + 顺序种子 + 观看时长

默认输出到 ``local-data/calibration-package/``（仓库忽略目录，截图副本不入 Git）；
本工具自身只放代码，不复制截图到 ``calibration/`` 下。

自包含实现（仅 stdlib + Pillow）：不 import ``ui_attention``，避免运行时依赖 src。
sha256 口径与主项目 imaging.sha256_file 一致（文件字节 SHA-256）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

PACKAGE_SCHEMA_VERSION = "game-ui-attention-calibration-package/v1"
TEMPLATE_PLACEHOLDER = "__STUDY_CONFIG__"
ALLOWED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "local-data" / "calibration-package"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "local-data" / "screenshots"
DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "index.html"

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """文件字节 SHA-256（与主项目 imaging.sha256_file 同口径）。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def image_dimensions(path: Path) -> tuple[int, int]:
    """返回 (width, height)；不可解码/非正尺寸 → ValueError（不静默跳过）。"""
    try:
        with Image.open(path) as img:
            img.load()
            if img.width <= 0 or img.height <= 0:
                raise ValueError(f"图片尺寸无效：{img.width}×{img.height}")
            return int(img.width), int(img.height)
    except UnidentifiedImageError as exc:
        raise ValueError(f"图片不可解码：{path.name}") from exc


def _utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_source_images(images_dir: Path, image_list: list[str] | None) -> list[Path]:
    """按确定性顺序收集图片源文件（目录扫描按名称排序；显式列表保持给定顺序）。"""
    if image_list is not None:
        paths = [images_dir / name for name in image_list]
    else:
        paths = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError("图片列表中存在不存在的文件：" + "；".join(missing))
    return paths


def compute_study_id(metas: list[dict[str, Any]], duration_seconds: float) -> str:
    """study_id = 图片集合（排序后）+ 时长的合成哈希前 16 位；与图序无关。"""
    key = json.dumps(
        sorted((m["sha256"], m["source_name"]) for m in metas), ensure_ascii=False, separators=(",", ":")
    )
    key += "|" + repr(duration_seconds)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def load_screen_types(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in obj.items()):
        raise ValueError("--screen-types 必须是 {源文件名: screen_type} 的 JSON 对象")
    return obj


def instantiate_template(template_text: str, config: dict[str, Any]) -> str:
    """把 config 以紧凑 JSON 注入模板占位符；转义 ``</`` 防止跳出 script 标签。"""
    if TEMPLATE_PLACEHOLDER not in template_text:
        raise ValueError(f"模板缺少占位符 {TEMPLATE_PLACEHOLDER}")
    payload = json.dumps(config, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return template_text.replace(TEMPLATE_PLACEHOLDER, payload)


def build_package(
    *,
    images_dir: Path,
    out: Path,
    template_path: Path,
    duration_seconds: float,
    seed: int,
    image_list: list[str] | None,
    screen_types: dict[str, str],
    default_screen_type: str,
    force: bool,
) -> dict[str, Any]:
    """构建研究包；返回摘要 dict（供 stdout 信封与测试断言）。"""
    if duration_seconds <= 0:
        raise ValueError(f"观看时长必须为正，得到 {duration_seconds}")
    if not template_path.is_file():
        raise ValueError(f"模板文件不存在：{template_path}")

    sources = collect_source_images(images_dir, image_list)
    if not sources:
        raise ValueError("未找到任何图片（目录为空或列表为空）")

    # 先读元数据（含哈希与尺寸），再复制，保证 config 的 sha256 = 源文件字节哈希
    metas: list[dict[str, Any]] = []
    for src in sources:
        width, height = image_dimensions(src)
        metas.append(
            {
                "source_name": src.name,
                "sha256": sha256_file(src),
                "width": width,
                "height": height,
                "screen_type": screen_types.get(src.name, default_screen_type),
            }
        )

    study_id = compute_study_id(metas, duration_seconds)

    # 图序随机化（种子记录入 config）
    order = list(range(len(metas)))
    random.Random(seed).shuffle(order)

    if out.exists():
        if not force:
            raise FileExistsError(f"输出目录已存在，拒绝覆盖（--force 可清空重建）：{out}")
        shutil.rmtree(out)
    images_out = out / "images"
    images_out.mkdir(parents=True)

    images: list[dict[str, Any]] = []
    for presentation_index, src_idx in enumerate(order):
        src = sources[src_idx]
        meta = metas[src_idx]
        ext = src.suffix.lower() or ".png"
        copied_name = f"img_{presentation_index:03d}{ext}"
        copied_path = images_out / copied_name
        shutil.copyfile(src, copied_path)
        copied_sha = sha256_file(copied_path)
        if copied_sha != meta["sha256"]:
            raise OSError(f"复制后哈希不一致（IO 损坏）：{src.name} {copied_sha} != {meta['sha256']}")
        images.append(
            {
                "index": presentation_index,
                "filename": copied_name,
                "source_name": meta["source_name"],
                "sha256": meta["sha256"],
                "width": meta["width"],
                "height": meta["height"],
                "screen_type": meta["screen_type"],
            }
        )

    config: dict[str, Any] = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "study_id": study_id,
        "created_at_utc": _utcnow(),
        "duration_seconds": duration_seconds,
        "shuffle_seed": seed,
        "screen_type_default": default_screen_type,
        "images": images,
    }

    template_text = template_path.read_text(encoding="utf-8")
    (out / "index.html").write_text(instantiate_template(template_text, config), encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "out_dir": str(out),
        "n_images": len(images),
        "study_id": study_id,
        "shuffle_seed": seed,
        "duration_seconds": duration_seconds,
        "images": [{"index": im["index"], "filename": im["filename"], "sha256": im["sha256"]} for im in images],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_study_package.py", description="生成内部粗标注一致性检查的研究包（纯静态、无上传）"
    )
    parser.add_argument(
        "--images", type=Path, default=DEFAULT_IMAGES_DIR, help="图片目录（默认 local-data/screenshots）"
    )
    parser.add_argument("--image-list", type=Path, default=None, help="可选：逐行图片文件名的列表文件")
    parser.add_argument("--duration", type=float, default=3.0, help="每图观看秒数（默认 3，可配 5）")
    parser.add_argument("--seed", type=int, default=20260913, help="图序随机化种子（记录入 config.json）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出目录（默认 local-data/calibration-package）")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="index.html 模板路径")
    parser.add_argument("--screen-types", type=Path, default=None, help="可选 {源文件名: screen_type} JSON")
    parser.add_argument("--default-screen-type", default="unknown", help="未映射图片的 screen_type（默认 unknown）")
    parser.add_argument("--force", action="store_true", help="输出目录已存在时清空重建")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        image_list = None
        if args.image_list is not None:
            image_list = [
                line.strip() for line in args.image_list.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
        screen_types = load_screen_types(args.screen_types)
        summary = build_package(
            images_dir=args.images,
            out=args.out,
            template_path=args.template,
            duration_seconds=args.duration,
            seed=args.seed,
            image_list=image_list,
            screen_types=screen_types,
            default_screen_type=args.default_screen_type,
            force=args.force,
        )
    except FileExistsError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 7
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
