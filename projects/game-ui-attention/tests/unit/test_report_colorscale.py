"""C3：色阶模块单测（colorscale.py）。

覆盖 validation-plan §1"色阶"行（A/B 展示使用相同范围、浮点数据不受颜色
和透明度影响——后者另见 test_report_overlay.py）与任务验收项：
A/B 共用色阶一致性（同参数断言、不同密度范围不各自拉满）、
色阶参数可序列化并还原、确定性映射。

合成数据全部代码生成（validation-plan §1：合成图只能证明计算链正确）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ui_attention.report.colorscale import (
    LUT_SIZE,
    ColorScale,
    build_lut,
    color_table_names,
    shared_range,
)


def make_density(h: int, w: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((h, w)) * scale


class TestLut:
    def test_names_nonempty(self):
        names = color_table_names()
        assert len(names) >= 2
        assert all(isinstance(n, str) for n in names)

    def test_lut_shape_dtype_deterministic(self):
        for name in color_table_names():
            a = build_lut(name)
            b = build_lut(name)
            assert a.shape == (LUT_SIZE, 4)
            assert a.dtype == np.uint8
            assert np.array_equal(a, b), f"LUT for {name} must be deterministic"

    def test_lut_alpha_ramp(self):
        lut = build_lut("ember")
        assert lut[0, 3] == 0  # t=0 全透明（低密度不遮挡原图）
        assert lut[-1, 3] == 255  # t=1 不透明
        assert np.all(np.diff(lut[:, 3].astype(int)) >= 0)  # 单调不减

    def test_lut_unknown_name_rejected(self):
        with pytest.raises(ValueError, match="unknown color table"):
            build_lut("not-a-scale")

    def test_lut_bad_size_rejected(self):
        with pytest.raises(ValueError):
            build_lut("ember", size=1)


class TestMap:
    def test_map_rgba_deterministic(self):
        d = make_density(16, 24, 0.5, seed=1)
        cs = ColorScale("lagoon", 0.0, 1.0)
        assert np.array_equal(cs.map_rgba(d), cs.map_rgba(d))

    def test_map_shape_dtype(self):
        d = make_density(9, 11, 1.0, seed=2)
        rgba = ColorScale("ember", 0.0, 1.0).map_rgba(d)
        assert rgba.shape == (9, 11, 4)
        assert rgba.dtype == np.uint8

    def test_map_clamps_out_of_range(self):
        cs = ColorScale("ember", vmin=0.25, vmax=0.75)
        lut = cs.lut()
        d = np.array([[0.0, 0.1], [0.9, 1.0]])
        rgba = cs.map_rgba(d)
        assert np.array_equal(rgba[0, 0], lut[0])  # 低于 vmin → 最低端颜色
        assert np.array_equal(rgba[0, 1], lut[0])
        assert np.array_equal(rgba[1, 0], lut[-1])  # 高于 vmax → 最高端颜色
        assert np.array_equal(rgba[1, 1], lut[-1])

    def test_map_index_rule_recorded(self):
        # 量化规则 index = rint(t * (LUT_SIZE-1)) 的自证（确定性、可复现）
        cs = ColorScale("mono", vmin=0.0, vmax=2.0)
        d = np.array([[0.5, 1.0, 1.5]])
        expected = cs.lut()[np.rint((d / 2.0) * (LUT_SIZE - 1)).astype(int)]
        assert np.array_equal(cs.map_rgba(d), expected)

    def test_map_rejects_nonfinite(self):
        cs = ColorScale("ember", 0.0, 1.0)
        with pytest.raises(ValueError, match="non-finite"):
            cs.map_rgba(np.array([[0.1, np.nan]]))
        with pytest.raises(ValueError, match="non-finite"):
            cs.map_rgba(np.array([[0.1, np.inf]]))

    def test_map_rejects_non_2d(self):
        cs = ColorScale("ember", 0.0, 1.0)
        with pytest.raises(ValueError, match="2-D"):
            cs.map_rgba(np.zeros((3, 4, 5)))

    def test_map_rgb_drops_alpha(self):
        cs = ColorScale("ember", 0.0, 1.0)
        d = make_density(5, 7, 1.0, seed=3)
        assert np.array_equal(cs.map_rgb(d), cs.map_rgba(d)[:, :, :3])

    def test_sqrt_normalization_differs_and_deterministic(self):
        d = np.array([[0.25]])
        lin = ColorScale("ember", 0.0, 1.0, "linear").map_rgba(d)
        sq = ColorScale("ember", 0.0, 1.0, "sqrt").map_rgba(d)
        assert not np.array_equal(lin, sq)  # sqrt(0.25)=0.5 > 0.25
        assert np.array_equal(sq, ColorScale("ember", 0.0, 1.0, "sqrt").map_rgba(d))


class TestConstructor:
    def test_degenerate_range_rejected(self):
        with pytest.raises(ValueError, match="degenerate"):
            ColorScale("ember", 1.0, 1.0)
        with pytest.raises(ValueError, match="degenerate"):
            ColorScale("ember", 2.0, 1.0)

    def test_unknown_name_and_normalization_rejected(self):
        with pytest.raises(ValueError, match="unknown color table"):
            ColorScale("plasma-copy", 0.0, 1.0)
        with pytest.raises(ValueError, match="unknown normalization"):
            ColorScale("ember", 0.0, 1.0, "log")

    def test_nonfinite_bounds_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            ColorScale("ember", 0.0, float("inf"))


class TestSerialization:
    def test_params_json_serializable_roundtrip(self):
        cs = ColorScale("lagoon", vmin=1e-6, vmax=0.02, normalization="sqrt")
        params = cs.to_params()
        text = json.dumps(params)  # 可写入 manifest（JSON 可序列化）
        restored = ColorScale.from_params(json.loads(text))
        assert restored == cs
        assert set(params) == {"name", "vmin", "vmax", "normalization"}
        assert params["name"] == "lagoon"
        assert params["normalization"] == "sqrt"

    def test_from_params_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown colorscale param"):
            ColorScale.from_params(
                {"name": "ember", "vmin": 0, "vmax": 1, "normalization": "linear", "extra": 1}
            )

    def test_from_params_rejects_missing_keys(self):
        with pytest.raises(ValueError, match="missing colorscale param"):
            ColorScale.from_params({"name": "ember", "vmin": 0, "vmax": 1})

    def test_frozen_and_hashable(self):
        import dataclasses

        cs = ColorScale("ember", 0.0, 1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cs.name = "lagoon"  # type: ignore[misc]
        assert hash(cs) == hash(ColorScale("ember", 0.0, 1.0))


class TestSharedRange:
    def test_joint_range(self):
        a = make_density(8, 8, 0.5, seed=10)
        b = make_density(8, 8, 2.0, seed=11) + 0.25
        vmin, vmax = shared_range(a, b)
        assert vmin == pytest.approx(min(a.min(), b.min()))
        assert vmax == pytest.approx(max(a.max(), b.max()))

    def test_degenerate_constant_pair_expanded(self):
        a = np.full((4, 4), 0.7)
        b = np.full((4, 4), 0.7)
        vmin, vmax = shared_range(a, b)
        assert vmin == pytest.approx(0.7)
        assert vmax == pytest.approx(1.7)  # 固定规则 (vmin, vmin+1)，确定性
        cs = ColorScale.shared("ember", a, b)  # 可构造非退化实例
        assert cs.vmax > cs.vmin

    def test_rejects_nonfinite_and_empty(self):
        good = np.ones((2, 2))
        with pytest.raises(ValueError, match="non-finite"):
            shared_range(good, np.array([[np.nan, 0.0]]))
        with pytest.raises(ValueError, match="non-empty"):
            shared_range(good, np.zeros((0, 3)))

    def test_shared_scale_uses_identical_params_for_both_images(self):
        """A/B 共用色阶 = 同一份参数作用于两图（同参数断言）。"""
        a = make_density(8, 10, 0.5, seed=20)
        b = make_density(8, 10, 2.0, seed=21)
        shared = ColorScale.shared("ember", a, b)
        # 渲染两侧时使用的是同一实例/同一份可序列化参数
        params_for_a = shared.to_params()
        params_for_b = shared.to_params()
        assert json.dumps(params_for_a) == json.dumps(params_for_b)
        vmin, vmax = shared_range(a, b)
        assert params_for_a["vmin"] == pytest.approx(vmin)
        assert params_for_a["vmax"] == pytest.approx(vmax)
        # 两侧映射均在该共用值域下 clip：a 的最大值不再映射到 LUT 顶端
        rgba_a = shared.map_rgba(a)
        rgba_b = shared.map_rgba(b)
        top = shared.lut()[-1]
        assert not np.any(np.all(rgba_a == top, axis=-1)), "A 侧不得被拉满到顶端色"
        assert np.any(np.all(rgba_b == top, axis=-1)), "B 侧最大值应到达共用值域顶端"

    def test_shared_scale_not_per_image_autoscale(self):
        """不同密度范围不各自拉满：对比逐图 auto-range 的行为差异。"""
        a = make_density(8, 10, 0.5, seed=30)
        b = make_density(8, 10, 2.0, seed=31)
        shared = ColorScale.shared("ember", a, b)
        auto_a = ColorScale("ember", float(a.min()), float(a.max()) if a.max() > a.min() else a.min() + 1)
        auto_b = ColorScale("ember", float(b.min()), float(b.max()) if b.max() > b.min() else b.min() + 1)

        top = shared.lut()[-1]

        def touches_top(rgba: np.ndarray) -> bool:
            return bool(np.any(np.all(rgba == top, axis=-1)))

        # 各自拉满时两侧都会出现顶端色（"各自拉满红色"的病态行为）
        assert touches_top(auto_a.map_rgba(a))
        assert touches_top(auto_b.map_rgba(b))
        # 共用色阶下只有到达联合最大值的一侧出现顶端色
        assert not touches_top(shared.map_rgba(a))
        assert touches_top(shared.map_rgba(b))
        # 同一密度值在共用色阶下映射唯一（与出现在哪一侧无关）
        v = float(a.max())
        one = shared.map_rgba(np.array([[v]]))[0, 0]
        assert np.array_equal(one, shared.map_rgba(np.array([[v]]))[0, 0])
