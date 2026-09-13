"""make_study_package.py 自测：3 张合成截图出包 + config sha256 校验 + index.html 无外链断言。"""

from __future__ import annotations

import json
from pathlib import Path

import make_study_package as mp
import numpy as np
import pytest
from PIL import Image


def _make_png(path: Path, size: tuple[int, int] = (16, 16), color: tuple[int, int, int] = (200, 30, 30)) -> None:
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path, format="PNG")


def test_build_package_three_images(tmp_path: Path) -> None:
    imgdir = tmp_path / "src"
    imgdir.mkdir()
    for name in ("1.png", "2.png", "3.png"):
        _make_png(imgdir / name)
    out = tmp_path / "pkg"
    summary = mp.build_package(
        images_dir=imgdir,
        out=out,
        template_path=mp.DEFAULT_TEMPLATE,
        duration_seconds=3.0,
        seed=42,
        image_list=None,
        screen_types={},
        default_screen_type="unknown",
        force=False,
    )
    assert summary["n_images"] == 3
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["duration_seconds"] == 3.0
    assert config["shuffle_seed"] == 42
    assert len(config["images"]) == 3
    assert [im["index"] for im in config["images"]] == [0, 1, 2]
    # config 的 sha256 == 复制后文件字节哈希 == 源文件字节哈希
    for im in config["images"]:
        copied = out / "images" / im["filename"]
        assert copied.is_file()
        assert mp.sha256_file(copied) == im["sha256"]
        assert mp.sha256_file(imgdir / im["source_name"]) == im["sha256"]


def test_build_package_is_deterministic_given_seed(tmp_path: Path) -> None:
    imgdir = tmp_path / "src"
    imgdir.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        _make_png(imgdir / name)

    def order(out: Path) -> list[str]:
        mp.build_package(
            images_dir=imgdir, out=out, template_path=mp.DEFAULT_TEMPLATE, duration_seconds=3.0,
            seed=7, image_list=None, screen_types={}, default_screen_type="unknown", force=False,
        )
        cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        return [im["sha256"] for im in cfg["images"]]

    assert order(tmp_path / "pkg1") == order(tmp_path / "pkg2")


def test_build_package_refuses_existing_without_force(tmp_path: Path) -> None:
    imgdir = tmp_path / "src"
    imgdir.mkdir()
    _make_png(imgdir / "1.png")
    out = tmp_path / "pkg"
    out.mkdir()

    with pytest.raises(FileExistsError):
        mp.build_package(
            images_dir=imgdir, out=out, template_path=mp.DEFAULT_TEMPLATE, duration_seconds=3.0,
            seed=1, image_list=None, screen_types={}, default_screen_type="unknown", force=False,
        )


def test_real_template_no_external_and_relative_images(tmp_path: Path) -> None:
    imgdir = tmp_path / "src"
    imgdir.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        _make_png(imgdir / name)
    out = tmp_path / "pkg"
    mp.build_package(
        images_dir=imgdir, out=out, template_path=mp.DEFAULT_TEMPLATE, duration_seconds=3.0,
        seed=7, image_list=None, screen_types={}, default_screen_type="unknown", force=False,
    )
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "__STUDY_CONFIG__" not in html  # 占位符已实例化
    assert "study_id" in html  # config 已注入
    assert "images/" in html  # 相对路径引用图片
    assert "data:image" not in html  # 禁止 data:URI 内嵌大图
    assert "data:text" not in html  # 下载用 Blob，不用 data URI
    assert "http://" not in html and "https://" not in html  # 零外链
    for banned in ("fetch(", "XMLHttpRequest", "sendBeacon", "<script src=", "<link rel=", "@import", "FormData"):
        assert banned not in html


def test_screen_type_mapping(tmp_path: Path) -> None:
    imgdir = tmp_path / "src"
    imgdir.mkdir()
    _make_png(imgdir / "1.png")
    _make_png(imgdir / "2.png")
    out = tmp_path / "pkg"
    mp.build_package(
        images_dir=imgdir, out=out, template_path=mp.DEFAULT_TEMPLATE, duration_seconds=3.0,
        seed=1, image_list=None, screen_types={"1.png": "reward", "2.png": "shop"},
        default_screen_type="unknown", force=False,
    )
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    by_source = {im["source_name"]: im["screen_type"] for im in config["images"]}
    assert by_source["1.png"] == "reward"
    assert by_source["2.png"] == "shop"
