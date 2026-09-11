"""imaging.py 测试（validation-plan §1“坐标映射”行：横屏、竖屏、缩放和边界 AOI 正确落到原图）。

另覆盖：格式/尺寸/方向/可解码校验、透明通道拒绝（不擅自填黑/白）、
保宽高比缩放与变换记录、尺寸超限明确失败、图片哈希绑定。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fixtures import synthetic  # noqa: E402
from ui_attention import imaging  # noqa: E402
from ui_attention.errors import ErrorCode, UiAttentionError  # noqa: E402

# ---------------------------------------------------------------------------
# 加载与校验
# ---------------------------------------------------------------------------


def test_load_solid_png(tmp_path):
    p = synthetic.solid_color_png(tmp_path / "solid.png")
    img = imaging.load_image(p)
    assert (img.width, img.height) == (64, 48)
    assert img.array.shape == (48, 64, 3)
    assert img.array.dtype == np.uint8
    assert img.format == "PNG"
    assert img.sha256 == imaging.sha256_file(p)
    assert img.orientation_applied is False
    assert img.alpha_composited is False
    # 纯色图：全部像素等于给定颜色
    assert tuple(img.array[0, 0]) == (30, 60, 90)


def test_load_portrait_png(tmp_path):
    p = synthetic.portrait_png(tmp_path / "portrait.png")
    img = imaging.load_image(p)
    assert img.shape == (64, 48)  # (H, W) 竖屏
    assert img.width == 48 and img.height == 64


def test_load_large_gradient(tmp_path):
    p = synthetic.gradient_png(tmp_path / "grad.png", width=640, height=360)
    img = imaging.load_image(p)
    assert img.shape == (360, 640)
    # 水平渐变：左暗右亮
    assert img.array[:, 0, 0].mean() < img.array[:, -1, 0].mean()


def test_missing_file_rejected(tmp_path):
    with pytest.raises(UiAttentionError) as exc:
        imaging.load_image(tmp_path / "nope.png")
    assert exc.value.code is ErrorCode.INVALID_IMAGE
    assert exc.value.exit_code == 2


def test_corrupt_file_rejected(tmp_path):
    p = synthetic.corrupt_file(tmp_path / "corrupt.png")
    with pytest.raises(UiAttentionError) as exc:
        imaging.load_image(p)
    assert exc.value.code is ErrorCode.INVALID_IMAGE


def test_unsupported_format_rejected(tmp_path):
    from PIL import Image

    p = tmp_path / "anim.gif"
    Image.new("P", (16, 16)).save(p)  # GIF 不在允许格式内
    with pytest.raises(UiAttentionError) as exc:
        imaging.load_image(p)
    assert exc.value.code is ErrorCode.INVALID_IMAGE
    assert "格式" in exc.value.message


# ---------------------------------------------------------------------------
# 透明度规则：未给合成背景一律拒绝，不擅自填黑/白
# ---------------------------------------------------------------------------


def test_rgba_rejected_without_background(tmp_path):
    p = synthetic.rgba_png(tmp_path / "alpha.png")
    with pytest.raises(UiAttentionError) as exc:
        imaging.load_image(p)
    assert exc.value.code is ErrorCode.INVALID_IMAGE
    assert "透明" in exc.value.message


def test_opaque_alpha_channel_also_rejected(tmp_path):
    """alpha 全 255 也拒绝：保守口径，不猜测合成结果。"""
    p = synthetic.opaque_alpha_png(tmp_path / "opaque_alpha.png")
    with pytest.raises(UiAttentionError):
        imaging.load_image(p)


def test_rgba_with_explicit_background_composited(tmp_path):
    p = synthetic.rgba_png(tmp_path / "alpha.png")  # R=200, A=128
    img = imaging.load_image(p, composite_background=(0, 0, 0))
    assert img.alpha_composited is True
    assert img.composite_background == (0, 0, 0)
    # 合成计算：200 * (128/255) ≈ 100.4
    r = int(img.array[0, 0, 0])
    assert 95 <= r <= 106
    assert int(img.array[0, 0, 1]) == 0

    white = imaging.load_image(p, composite_background=(255, 255, 255))
    rw = int(white.array[0, 0, 0])
    assert rw > r  # 白底合成结果更亮（合成背景确实生效，非固定填色）


def test_invalid_background_value(tmp_path):
    p = synthetic.rgba_png(tmp_path / "alpha.png")
    with pytest.raises(UiAttentionError):
        imaging.load_image(p, composite_background=(300, 0, 0))


def test_grayscale_converted_to_rgb(tmp_path):
    from PIL import Image

    p = tmp_path / "gray.png"
    Image.new("L", (8, 6), 128).save(p)
    img = imaging.load_image(p)
    assert img.array.shape == (6, 8, 3)
    assert img.original_mode == "L"


# ---------------------------------------------------------------------------
# EXIF 方向
# ---------------------------------------------------------------------------


def test_exif_orientation_applied(tmp_path):
    p = synthetic.oriented_jpeg(tmp_path / "rot.jpg", width=40, height=20, orientation=6)
    img = imaging.load_image(p)
    assert img.orientation_applied is True
    # 存储 40×20，Orientation=6（顺时针 90°）→ 显示尺寸 20×40
    assert (img.width, img.height) == (20, 40)
    # 原左半红色旋转后应位于上半部（JPEG 有损，用宽松阈值）
    top = img.array[:10, :, 0].mean()
    bottom = img.array[-10:, :, 0].mean()
    assert top > 150 > bottom


# ---------------------------------------------------------------------------
# 哈希绑定
# ---------------------------------------------------------------------------


def test_sha256_binds_content(tmp_path):
    p1 = synthetic.solid_color_png(tmp_path / "a.png", color=(1, 2, 3))
    p2 = synthetic.solid_color_png(tmp_path / "b.png", color=(1, 2, 3))
    p3 = synthetic.solid_color_png(tmp_path / "c.png", color=(9, 9, 9))
    h1, h2, h3 = (imaging.sha256_file(p) for p in (p1, p2, p3))
    assert h1 == h2  # 相同内容 → 相同哈希（Pillow PNG 输出确定性）
    assert h1 != h3  # 内容不同 → 哈希不同（哈希改变时必须重新检查区域）
    assert len(h1) == 64


# ---------------------------------------------------------------------------
# 保宽高比缩放与变换记录
# ---------------------------------------------------------------------------


def test_resize_keep_aspect_landscape():
    arr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    r = imaging.resize_keep_aspect(arr, max_height=240, max_width=320)
    # scale = min(240/1080, 320/1920) = 1/6 → 320×180
    assert r.inference_shape == (180, 320)
    assert r.array.shape == (180, 320, 3)
    assert r.scale == pytest.approx(1 / 6)
    mapping = r.shape_mapping()
    assert mapping["original_shape"] == [1080, 1920]
    assert mapping["inference_shape"] == [180, 320]
    assert "inverse" in mapping


def test_resize_keep_aspect_portrait():
    arr = np.zeros((1600, 900, 3), dtype=np.uint8)  # 竖屏 (H, W)
    r = imaging.resize_keep_aspect(arr, max_height=320, max_width=240)
    # scale = min(320/1600, 240/900) = 0.2 → 320×180
    assert r.inference_shape == (320, 180)
    assert r.array.shape == (320, 180, 3)


def test_resize_never_upscales():
    arr = np.zeros((50, 80, 3), dtype=np.uint8)
    r = imaging.resize_keep_aspect(arr, max_height=240, max_width=320)
    assert r.inference_shape == (50, 80)
    assert r.scale == 1.0
    assert r.array is arr  # 不复制、不改动


# ---------------------------------------------------------------------------
# 坐标映射：横屏/竖屏/缩放/边界
# ---------------------------------------------------------------------------

LANDSCAPE_MAPPING = {"original_shape": [1080, 1920], "inference_shape": [240, 320]}
PORTRAIT_MAPPING = {"original_shape": [1600, 900], "inference_shape": [320, 180]}


@pytest.mark.parametrize("mapping", [LANDSCAPE_MAPPING, PORTRAIT_MAPPING])
def test_point_mapping_roundtrip(mapping):
    oh, ow = mapping["original_shape"]
    for x, y in [(0, 0), (ow / 2, oh / 2), (ow - 1, oh - 1), (123.5, 456.25)]:
        ix, iy = imaging.original_to_inference_point(x, y, mapping)
        bx, by = imaging.inference_to_original_point(ix, iy, mapping)
        assert bx == pytest.approx(x, abs=1e-6)
        assert by == pytest.approx(y, abs=1e-6)


def test_boundary_maps_into_inference_range():
    """边界 AOI 坐标（含最大像素）必须落到推理图画布范围内。"""
    oh, ow = LANDSCAPE_MAPPING["original_shape"]
    ih, iw = LANDSCAPE_MAPPING["inference_shape"]
    ix, iy = imaging.original_to_inference_point(ow - 1, oh - 1, LANDSCAPE_MAPPING)
    assert 0 <= ix < iw
    assert 0 <= iy < ih
    # 逆变换（推理→原图）带截断：边界与越界输入都落回原图像素范围
    bx, by = imaging.inference_to_original_point(iw, ih, LANDSCAPE_MAPPING)
    assert bx <= ow - 1 and by <= oh - 1


def test_inverse_clips_to_image_range():
    """逆变换越界输入被截断回原图范围（不产生图外坐标）。"""
    oh, ow = LANDSCAPE_MAPPING["original_shape"]
    x, y = imaging.inference_to_original_point(-5, 1e6, LANDSCAPE_MAPPING)
    assert 0 <= x <= ow - 1
    assert 0 <= y <= oh - 1


def test_shape_mapping_invalid_rejected():
    with pytest.raises(UiAttentionError):
        imaging.original_to_inference_point(1, 1, {"original_shape": [10]})
    with pytest.raises(UiAttentionError):
        imaging.inference_to_original_point(1, 1, {"original_shape": [0, 5], "inference_shape": [1, 1]})


# ---------------------------------------------------------------------------
# 尺寸超限：明确失败，不静默缩小
# ---------------------------------------------------------------------------


def test_enforce_size_limits():
    imaging.enforce_size_limits(640, 480, max_width=1920, max_height=1080)  # 合规不抛
    with pytest.raises(UiAttentionError) as exc:
        imaging.enforce_size_limits(4000, 480, max_width=1920, max_height=1080)
    assert exc.value.code is ErrorCode.INVALID_IMAGE
    assert "不静默缩小" in exc.value.message
    imaging.enforce_size_limits(4000, 480, max_width=None, max_height=None)  # 无上限配置不抛
