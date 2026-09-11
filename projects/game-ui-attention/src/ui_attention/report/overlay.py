"""热图叠加 PNG 渲染：原图 + 密度图 → 固定透明度、可记录、确定性输出。

设计口径（对齐 technical-design §9、data-contract §5、validation-plan §1"色阶"行）：

- 合成公式（float64 中间量，最终 ``rint`` 量化为 uint8）::

      w(pixel) = alpha * (lut_alpha(t(pixel)) / 255)
      out_rgb  = rint(clip(base * (1 - w) + heat_rgb * w, 0, 255))

  其中 ``alpha`` 是**固定透明度参数**（标量，全部像素一致，可记录进
  manifest），``lut_alpha`` 是色阶 LUT 自带的 alpha 斜坡（低密度更透明，
  属于色表定义，见 colorscale.py）。不使用任何随机源。
- 输出尺寸与原图一致：密度数组形状必须与原图 ``(H, W)`` 完全相同，
  否则 ValueError——推理分辨率 → 原图分辨率的重采样属于 C1 imaging/
  shape_mapping 职责，本模块只做展示合成。另提供 *展示专用* 的
  :func:`resize_density_nearest`（确定性最近邻），其输出不得用于统计
  （"统计只读取浮点概率图，不能从 PNG 颜色或额外显示模糊后的图计算"）。
- 确定性：同输入（原图数组、密度数组、ColorScale 参数、alpha、Pillow
  版本）产出逐字节一致的 PNG。:func:`render_overlay_png` 两次渲染
  字节级一致由 test_report_overlay.py 验证。
- ``overlay_record(...)`` 产出可写入 manifest 的展示参数记录
  （色阶参数 + alpha + 尺寸一致性声明）。

主要公开接口（纯函数式：输入数据 + 配置 → 输出文件路径）：

- ``render_overlay_png(*, base_image, density, colorscale, alpha, out_path) -> Path``
- ``render_heatmap_png(*, density, colorscale, out_path) -> Path``
- ``overlay_rgba(*, base_image, density, colorscale, alpha) -> np.ndarray``
- ``resize_density_nearest(density, target_hw) -> np.ndarray``（仅展示）
- ``overlay_record(colorscale, alpha) -> dict``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .colorscale import ColorScale

__all__ = [
    "DEFAULT_OVERLAY_ALPHA",
    "overlay_record",
    "overlay_rgba",
    "render_heatmap_png",
    "render_overlay_png",
    "resize_density_nearest",
]

# 默认固定透明度：记录进 manifest，可被 CLI 配置覆盖（C1 接线）。
DEFAULT_OVERLAY_ALPHA = 0.55


def _check_base(base_image: np.ndarray) -> np.ndarray:
    base = np.asarray(base_image)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError(
            f"base_image must be RGB (H, W, 3), got shape {base.shape}"
        )
    if base.dtype != np.uint8:
        raise ValueError(f"base_image must be uint8, got {base.dtype}")
    return base


def _check_density_for(density: np.ndarray, base: np.ndarray) -> np.ndarray:
    d = np.asarray(density, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"density must be 2-D (H, W), got shape {d.shape}")
    if d.shape != base.shape[:2]:
        raise ValueError(
            f"density shape {d.shape} != base image shape {base.shape[:2]}; "
            "resample density to original size first (C1 imaging/shape_mapping)"
        )
    if d.size == 0:
        raise ValueError("density must be non-empty")
    if not np.all(np.isfinite(d)):
        raise ValueError("density contains non-finite values (NaN/inf)")
    return d


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not (0.0 <= a <= 1.0):
        raise ValueError(f"alpha must be a finite float in [0, 1], got {alpha!r}")
    return a


def overlay_rgba(
    *,
    base_image: np.ndarray,
    density: np.ndarray,
    colorscale: ColorScale,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> np.ndarray:
    """原图 + 热图合成，返回 ``(H, W, 3)`` uint8 RGB 数组（不写盘）。

    合成权重 ``w = alpha * lut_alpha/255``；alpha 为固定透明度参数。
    确定性：无随机源，float64 中间量 + ``rint`` 量化。
    """
    base = _check_base(base_image)
    d = _check_density_for(density, base)
    a = _check_alpha(alpha)
    heat = colorscale.map_rgba(d)  # (H, W, 4) uint8
    w = (heat[:, :, 3].astype(np.float64) / 255.0) * a  # (H, W)
    wf = w[:, :, None]
    out = base.astype(np.float64) * (1.0 - wf) + heat[:, :, :3].astype(np.float64) * wf
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def render_overlay_png(
    *,
    base_image: np.ndarray,
    density: np.ndarray,
    colorscale: ColorScale,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
    out_path: Path | str,
) -> Path:
    """渲染叠加图并写出 PNG；返回输出路径。输出尺寸与原图一致。

    PNG 编码不附加任何时间戳/文本块（Pillow 默认行为），同输入字节级一致。
    """
    rgb = overlay_rgba(
        base_image=base_image, density=density, colorscale=colorscale, alpha=alpha
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out, format="PNG")
    return out


def render_heatmap_png(
    *,
    density: np.ndarray,
    colorscale: ColorScale,
    out_path: Path | str,
) -> Path:
    """渲染纯热图（RGBA，alpha 来自色表斜坡）并写出 PNG；返回输出路径。

    HTML 报告的"热图"视图把该 RGBA 层直接叠在原图层之上展示
    （色阶全强度，无附加透明度），与烘焙固定 alpha 的 overlay.png 区分。
    """
    d = np.asarray(density, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"density must be 2-D (H, W), got shape {d.shape}")
    if d.size == 0:
        raise ValueError("density must be non-empty")
    if not np.all(np.isfinite(d)):
        raise ValueError("density contains non-finite values (NaN/inf)")
    rgba = colorscale.map_rgba(d)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out, format="PNG")
    return out


def resize_density_nearest(
    density: np.ndarray, target_hw: tuple[int, int]
) -> np.ndarray:
    """确定性最近邻重采样，**仅供展示**（把推理分辨率密度图放大到原图尺寸）。

    统计计算不得使用本函数输出（validation-plan §1：统计只读原始浮点
    概率图）；C1 的正式逆变换以 imaging/shape_mapping 为准。
    """
    d = np.asarray(density, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"density must be 2-D (H, W), got shape {d.shape}")
    th, tw = target_hw
    if not (isinstance(th, int) and isinstance(tw, int)) or th < 1 or tw < 1:
        raise ValueError(f"target_hw must be positive ints, got {target_hw!r}")
    src_h, src_w = d.shape
    rows = np.clip(np.rint(np.arange(th) * (src_h - 1) / max(th - 1, 1)).astype(int), 0, src_h - 1)
    cols = np.clip(np.rint(np.arange(tw) * (src_w - 1) / max(tw - 1, 1)).astype(int), 0, src_w - 1)
    return d[rows][:, cols]


def overlay_record(
    colorscale: ColorScale, alpha: float = DEFAULT_OVERLAY_ALPHA
) -> dict[str, Any]:
    """产出可写入 manifest 的展示参数记录（JSON 可序列化）。

    建议键名 ``"overlay"``；内含色阶参数（data-contract §6 要求的共享
    色阶参数）与固定透明度。A/B 对比时对两侧写入同一份记录。
    """
    a = _check_alpha(alpha)
    return {
        "alpha": a,
        "colorscale": colorscale.to_params(),
        "output_size_matches_original": True,
        "random_sources": "none",
    }
