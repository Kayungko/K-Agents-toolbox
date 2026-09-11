"""色阶定义：确定性浮点密度 → RGBA 映射，A/B 共用色阶，参数可序列化进 manifest。

设计口径（对齐 technical-design §9、data-contract §5/§6、validation-plan §1"色阶"行）：

- 色表为原创锚点渐变的查找表（LUT）实现，不引入 matplotlib 依赖，不复制第三方色表数据。
- ``ColorScale`` 参数（色表名、值域 [vmin, vmax]、归一化方式）完全可序列化：
  ``to_params()`` 产出的 dict 可直接 ``json.dumps`` 并由 C1 写入 manifest.json
  （data-contract §6："共享色阶参数写入 manifest"）。
- A/B 共用色阶 = 同一个 ``ColorScale`` 实例的参数作用于两张密度图；
  ``shared_range(density_a, density_b)`` 计算共用值域（联合 min/max），
  ``ColorScale.shared(...)`` 直接产出作用于两图的同一实例——不各自拉满。
- 映射确定性：同输入（密度数组 + 参数）恒得逐字节相同的 uint8 RGBA；
  无随机源。非有限值（NaN/inf）密度直接拒绝（ValueError），
  因为统计只读浮点概率图，渲染层不得吞掉无效密度。
- 本模块只做展示映射；任何统计数值不得从本模块输出的颜色反推
  （technical-design §9："统计只读取浮点概率图"）。

主要公开接口：

- ``color_table_names() -> tuple[str, ...]``
- ``build_lut(name, size=256) -> np.ndarray``，形状 ``(size, 4)`` uint8 RGBA
- ``shared_range(density_a, density_b) -> tuple[float, float]``
- ``class ColorScale(name, vmin, vmax, normalization)``：
  ``lut()`` / ``normalize(density)`` / ``map_rgba(density)`` /
  ``to_params()`` / ``from_params(params)`` / ``shared(name, a, b, ...)``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "LUT_SIZE",
    "NORMALIZATIONS",
    "ColorScale",
    "build_lut",
    "color_table_names",
    "shared_range",
]

LUT_SIZE = 256
NORMALIZATIONS: tuple[str, ...] = ("linear", "sqrt")

# 原创锚点色表：每条为 (t, R, G, B) 控制点序列，t ∈ [0,1] 严格递增，
# 段间线性插值生成 LUT。锚点数值为本项目自行选取，非第三方色表复制。
# alpha 通道按 t 线性上升（t=0 全透明 → t=1 不透明），属于色表定义的一部分，
# 使热图低密度区不遮挡原图；overlay 的固定透明度参数另见 overlay.py。
_ANCHOR_TABLES: dict[str, tuple[tuple[int, int, int], ...]] = {
    # 深空蓝紫 → 品红 → 橙 → 亮黄白（高温端暖色，类"热力"语义）
    "ember": (
        (10, 6, 28),
        (62, 12, 102),
        (140, 30, 110),
        (204, 62, 62),
        (240, 138, 36),
        (252, 224, 120),
        (255, 252, 240),
    ),
    # 深海蓝 → 青 → 绿 → 黄绿（低温端冷色，类"海洋"语义）
    "lagoon": (
        (14, 10, 42),
        (32, 56, 124),
        (20, 110, 138),
        (26, 158, 104),
        (128, 198, 66),
        (226, 232, 120),
        (250, 250, 226),
    ),
    # 中性灰阶：黑白展示/打印友好
    "mono": (
        (18, 18, 22),
        (238, 238, 242),
    ),
}


def color_table_names() -> tuple[str, ...]:
    """返回可用色表名（可序列化参数 ``name`` 的合法取值）。"""
    return tuple(_ANCHOR_TABLES)


def build_lut(name: str, size: int = LUT_SIZE) -> np.ndarray:
    """由锚点线性插值生成 ``(size, 4)`` uint8 RGBA 查找表。

    确定性：同名同 size 恒得逐元素相同结果。alpha = round(255 * t)。
    """
    if name not in _ANCHOR_TABLES:
        raise ValueError(
            f"unknown color table {name!r}; available: {color_table_names()}"
        )
    if not isinstance(size, int) or isinstance(size, bool) or size < 2:
        raise ValueError(f"lut size must be an int >= 2, got {size!r}")
    anchors = np.asarray(_ANCHOR_TABLES[name], dtype=np.float64)  # (k, 3)
    k = anchors.shape[0]
    # 锚点等距分布在 t ∈ [0,1]
    anchor_t = np.linspace(0.0, 1.0, k)
    t = np.linspace(0.0, 1.0, size)
    rgb = np.empty((size, 3), dtype=np.float64)
    for ch in range(3):
        rgb[:, ch] = np.interp(t, anchor_t, anchors[:, ch])
    alpha = np.rint(255.0 * t)
    lut = np.empty((size, 4), dtype=np.uint8)
    lut[:, :3] = np.rint(rgb).astype(np.uint8)
    lut[:, 3] = alpha.astype(np.uint8)
    return lut


def shared_range(
    density_a: np.ndarray, density_b: np.ndarray
) -> tuple[float, float]:
    """计算 A/B 两图的共用值域 ``(vmin, vmax)``：联合 min/max。

    - 两图必须为非空、全有限值的浮点密度数组，否则 ValueError。
    - 退化情形（联合 vmax <= vmin，如两图均为常数）：固定规则扩展为
      ``(vmin, vmin + 1.0)``，保证可构造非退化 ColorScale；该规则确定性且
      会随参数一起写入 manifest，不做任何数据相关猜测。
    """
    a = np.asarray(density_a, dtype=np.float64)
    b = np.asarray(density_b, dtype=np.float64)
    for tag, arr in (("density_a", a), ("density_b", b)):
        if arr.size == 0:
            raise ValueError(f"{tag} must be non-empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{tag} contains non-finite values (NaN/inf)")
    vmin = float(min(a.min(), b.min()))
    vmax = float(max(a.max(), b.max()))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return (vmin, vmax)


def _normalize_t(
    density: np.ndarray, vmin: float, vmax: float, normalization: str
) -> np.ndarray:
    """密度 → [0,1] 归一化 t 值（float64，clip 后返回）。"""
    arr = np.asarray(density, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("density must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("density contains non-finite values (NaN/inf)")
    t = (arr - vmin) / (vmax - vmin)
    t = np.clip(t, 0.0, 1.0)
    if normalization == "sqrt":
        t = np.sqrt(t)
    return t


@dataclass(frozen=True)
class ColorScale:
    """可序列化的确定性色阶：色表名 + 值域 + 归一化方式。

    A/B 共用色阶即把同一个实例（同一份参数）分别作用于两图，
    不做逐图 auto-range。冻结 dataclass，可哈希、可比较。
    """

    name: str = "ember"
    vmin: float = 0.0
    vmax: float = 1.0
    normalization: str = "linear"

    def __post_init__(self) -> None:
        if self.name not in _ANCHOR_TABLES:
            raise ValueError(
                f"unknown color table {self.name!r}; "
                f"available: {color_table_names()}"
            )
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(
                f"unknown normalization {self.normalization!r}; "
                f"allowed: {NORMALIZATIONS}"
            )
        vmin = float(self.vmin)
        vmax = float(self.vmax)
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            raise ValueError("vmin/vmax must be finite")
        if vmax <= vmin:
            raise ValueError(
                f"degenerate range: vmax ({vmax!r}) must be > vmin ({vmin!r})"
            )
        # frozen dataclass：用 object.__setattr__ 固化 float 类型
        object.__setattr__(self, "vmin", vmin)
        object.__setattr__(self, "vmax", vmax)

    # ------------------------------------------------------------------ LUT
    def lut(self) -> np.ndarray:
        """``(256, 4)`` uint8 RGBA 查找表（缓存无副作用，逐次生成亦确定）。"""
        return build_lut(self.name, LUT_SIZE)

    # ------------------------------------------------------------ normalize
    def normalize(self, density: np.ndarray) -> np.ndarray:
        """密度 → [0,1] float64 t 值（clip + normalization）。"""
        return _normalize_t(density, self.vmin, self.vmax, self.normalization)

    # ------------------------------------------------------------------ map
    def map_rgba(self, density: np.ndarray) -> np.ndarray:
        """浮点密度 ``(H, W)`` → uint8 RGBA ``(H, W, 4)``，确定性映射。

        量化规则（记录在案，保证可复现）：
        ``index = rint(clip(t, 0, 1) * (LUT_SIZE - 1))``，查 LUT 得 RGBA。
        """
        arr = np.asarray(density, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"density must be 2-D (H, W), got shape {arr.shape}")
        t = self.normalize(arr)
        idx = np.rint(t * (LUT_SIZE - 1)).astype(np.intp)
        return self.lut()[idx]

    def map_rgb(self, density: np.ndarray) -> np.ndarray:
        """同 :meth:`map_rgba` 但丢弃 alpha，返回 ``(H, W, 3)`` uint8。"""
        return self.map_rgba(density)[:, :, :3]

    # ----------------------------------------------------------- serialize
    def to_params(self) -> dict[str, Any]:
        """产出 JSON 可序列化参数 dict（供 manifest 记录）。

        结构（写入 manifest 的建议键名为 ``"colorscale"``）::

            {"name": str, "vmin": float, "vmax": float, "normalization": str}
        """
        return {
            "name": self.name,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "normalization": self.normalization,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> ColorScale:
        """由 :meth:`to_params` 的 dict 还原实例；未知键或非法值拒绝。"""
        if not isinstance(params, dict):
            raise ValueError(f"colorscale params must be a dict, got {type(params)}")
        allowed = {"name", "vmin", "vmax", "normalization"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"unknown colorscale param keys: {sorted(unknown)}")
        missing = allowed - set(params)
        if missing:
            raise ValueError(f"missing colorscale param keys: {sorted(missing)}")
        return cls(
            name=str(params["name"]),
            vmin=float(params["vmin"]),
            vmax=float(params["vmax"]),
            normalization=str(params["normalization"]),
        )

    # --------------------------------------------------------------- shared
    @classmethod
    def shared(
        cls,
        name: str,
        density_a: np.ndarray,
        density_b: np.ndarray,
        normalization: str = "linear",
    ) -> ColorScale:
        """构造 A/B 共用色阶：值域取 ``shared_range(density_a, density_b)``。

        返回的同一实例作用于两图即"共用颜色标尺，不各自拉满"。
        """
        vmin, vmax = shared_range(density_a, density_b)
        return cls(name=name, vmin=vmin, vmax=vmax, normalization=normalization)
