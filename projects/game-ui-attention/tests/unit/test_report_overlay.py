"""C3：overlay 渲染单测（overlay.py）。

覆盖：输出尺寸与原图一致、字节级确定性、固定透明度参数可记录、
alpha 边界行为、无效输入拒绝、展示专用重采样的确定性，以及
validation-plan §1"色阶"行的"浮点数据不受颜色和透明度影响"
（渲染不改变输入密度数组——统计只读浮点概率图）。

合成图全部代码生成，不含真实游戏截图。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from ui_attention.report.colorscale import ColorScale
from ui_attention.report.overlay import (
    DEFAULT_OVERLAY_ALPHA,
    overlay_record,
    overlay_rgba,
    render_heatmap_png,
    render_overlay_png,
    resize_density_nearest,
)


def make_base(h: int = 48, w: int = 64) -> np.ndarray:
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:, :, 0] = np.linspace(30, 200, w, dtype=np.uint8)[None, :]
    base[:, :, 2] = np.linspace(200, 30, h, dtype=np.uint8)[:, None]
    base[h // 4 : h // 2, w // 4 : w // 2] = (240, 220, 40)
    return base


def make_density(h: int = 48, w: int = 64, peak: float = 1.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.exp(-(((xx - w * 0.6) ** 2 + (yy - h * 0.5) ** 2) / (2 * (w * 0.12) ** 2)))
    return d / d.max() * peak


class TestOverlayRender:
    def test_size_matches_original(self, tmp_path):
        base = make_base(37, 53)  # 非对齐尺寸也要一致
        d = make_density(37, 53)
        out = render_overlay_png(
            base_image=base, density=d, colorscale=ColorScale("ember", 0.0, 1.0),
            alpha=0.5, out_path=tmp_path / "overlay.png",
        )
        img = Image.open(out)
        assert img.size == (53, 37)  # (W, H) 与原图一致
        assert img.mode == "RGB"

    def test_deterministic_bytes(self, tmp_path):
        base = make_base()
        d = make_density()
        cs = ColorScale("ember", 0.0, 1.0)
        p1 = render_overlay_png(
            base_image=base, density=d, colorscale=cs, alpha=0.55,
            out_path=tmp_path / "a.png",
        )
        p2 = render_overlay_png(
            base_image=base, density=d, colorscale=cs, alpha=0.55,
            out_path=tmp_path / "b.png",
        )
        assert p1.read_bytes() == p2.read_bytes(), "同输入必须字节级一致（随机源为零）"

    def test_deterministic_across_call_order(self, tmp_path):
        # 中间插入其他渲染调用也不影响（无隐藏状态）
        base, d = make_base(), make_density()
        cs = ColorScale("lagoon", 0.0, 1.0)
        first = render_overlay_png(
            base_image=base, density=d, colorscale=cs, alpha=0.4, out_path=tmp_path / "1.png"
        ).read_bytes()
        render_heatmap_png(density=d * 0.3, colorscale=cs, out_path=tmp_path / "h.png")
        second = render_overlay_png(
            base_image=base, density=d, colorscale=cs, alpha=0.4, out_path=tmp_path / "2.png"
        ).read_bytes()
        assert first == second

    def test_alpha_zero_returns_original_pixels(self, tmp_path):
        base = make_base()
        d = make_density()
        out = render_overlay_png(
            base_image=base, density=d, colorscale=ColorScale("ember", 0.0, 1.0),
            alpha=0.0, out_path=tmp_path / "zero.png",
        )
        assert np.array_equal(np.asarray(Image.open(out)), base)

    def test_alpha_one_peak_pixel_is_lut_top(self, tmp_path):
        # t=1 处 LUT alpha=255 → w=alpha=1 → 输出即 LUT 顶端 RGB
        base = make_base(16, 16)
        d = np.zeros((16, 16))
        d[4, 4] = 1.0
        cs = ColorScale("ember", 0.0, 1.0)
        out = render_overlay_png(
            base_image=base, density=d, colorscale=cs, alpha=1.0, out_path=tmp_path / "one.png"
        )
        arr = np.asarray(Image.open(out))
        assert np.array_equal(arr[4, 4], cs.lut()[-1, :3])
        # 零密度处 LUT alpha=0 → w=0 → 保留原图像素
        assert np.array_equal(arr[0, 0], base[0, 0])

    def test_fixed_alpha_recordable_params(self):
        cs = ColorScale("ember", 1e-6, 0.02)
        rec = overlay_record(cs, DEFAULT_OVERLAY_ALPHA)
        text = json.dumps(rec)  # 可写入 manifest
        assert json.loads(text) == rec
        assert rec["alpha"] == DEFAULT_OVERLAY_ALPHA
        assert rec["colorscale"] == cs.to_params()
        assert rec["output_size_matches_original"] is True
        assert rec["random_sources"] == "none"

    def test_density_not_mutated_by_rendering(self, tmp_path):
        """浮点数据不受颜色和透明度影响：渲染不得改动输入密度数组。"""
        base = make_base()
        d = make_density()
        snapshot = d.copy()
        for alpha in (0.0, 0.35, 1.0):
            for name in ("ember", "lagoon", "mono"):
                render_overlay_png(
                    base_image=base, density=d, colorscale=ColorScale(name, 0.0, 1.0),
                    alpha=alpha, out_path=tmp_path / f"o-{name}-{alpha}.png",
                )
                render_heatmap_png(
                    density=d, colorscale=ColorScale(name, 0.0, 1.0),
                    out_path=tmp_path / f"h-{name}-{alpha}.png",
                )
        assert np.array_equal(d, snapshot)

    def test_rejects_shape_mismatch(self, tmp_path):
        base = make_base(48, 64)
        small = make_density(24, 32)
        with pytest.raises(ValueError, match="resample density"):
            render_overlay_png(
                base_image=base, density=small, colorscale=ColorScale("ember", 0, 1),
                alpha=0.5, out_path=tmp_path / "x.png",
            )

    def test_rejects_nonfinite_density(self, tmp_path):
        base = make_base(8, 8)
        d = make_density(8, 8)
        d[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            render_overlay_png(
                base_image=base, density=d, colorscale=ColorScale("ember", 0, 1),
                alpha=0.5, out_path=tmp_path / "x.png",
            )

    def test_rejects_bad_alpha(self, tmp_path):
        base, d = make_base(8, 8), make_density(8, 8)
        for bad in (-0.1, 1.5, float("nan")):
            with pytest.raises(ValueError):
                overlay_rgba(
                    base_image=base, density=d, colorscale=ColorScale("ember", 0, 1), alpha=bad
                )

    def test_rejects_bad_base(self, tmp_path):
        d = make_density(8, 8)
        with pytest.raises(ValueError, match="uint8"):
            overlay_rgba(
                base_image=np.zeros((8, 8, 3), dtype=np.float32),
                density=d, colorscale=ColorScale("ember", 0, 1), alpha=0.5,
            )
        with pytest.raises(ValueError, match="RGB"):
            overlay_rgba(
                base_image=np.zeros((8, 8, 4), dtype=np.uint8),
                density=d, colorscale=ColorScale("ember", 0, 1), alpha=0.5,
            )


class TestHeatmapRender:
    def test_heatmap_rgba_alpha_from_lut(self, tmp_path):
        d = np.zeros((12, 12))
        d[6, 6] = 1.0
        out = render_heatmap_png(
            density=d, colorscale=ColorScale("ember", 0.0, 1.0), out_path=tmp_path / "h.png"
        )
        img = Image.open(out)
        assert img.mode == "RGBA"
        arr = np.asarray(img)
        assert arr[0, 0, 3] == 0  # 零密度全透明
        assert arr[6, 6, 3] == 255  # 峰值不透明

    def test_heatmap_deterministic_bytes(self, tmp_path):
        d = make_density(20, 20)
        cs = ColorScale("lagoon", 0.0, float(d.max()))
        b1 = render_heatmap_png(density=d, colorscale=cs, out_path=tmp_path / "h1.png").read_bytes()
        b2 = render_heatmap_png(density=d, colorscale=cs, out_path=tmp_path / "h2.png").read_bytes()
        assert b1 == b2


class TestResizeDisplayOnly:
    def test_resize_shape_and_determinism(self):
        d = make_density(24, 32)
        r1 = resize_density_nearest(d, (48, 64))
        r2 = resize_density_nearest(d, (48, 64))
        assert r1.shape == (48, 64)
        assert np.array_equal(r1, r2)
        # 最近邻只复制原值，不引入新数值
        assert set(np.unique(r1)) <= set(np.unique(d))

    def test_resize_rejects_bad_input(self):
        d = make_density(8, 8)
        with pytest.raises(ValueError):
            resize_density_nearest(d, (0, 4))
        with pytest.raises(ValueError):
            resize_density_nearest(np.zeros((2, 2, 2)), (4, 4))
