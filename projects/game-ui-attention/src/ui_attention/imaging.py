"""图像处理（technical-design.md §4）。

规则：
1. 校验文件格式、尺寸、方向和可解码性；应用明确的方向与颜色处理规则。
2. 默认要求完整、已合成的 RGB 游戏截图。存在透明度且未给定真实合成背景时**拒绝分析**，
   不擅自填黑或白背景。
3. 保留宽高比缩放；记录原图尺寸、推理尺寸和坐标变换。尺寸超限时明确失败，不静默缩小。
4. 图片哈希与标注绑定（sha256 为文件字节哈希；哈希改变时重新检查区域，不沿用过期位置）。

冻结接口口径（data-contract §3）：``predict(image, resolved_profile)`` 的 image 为
**方向已处理的原图** RGB uint8 (H, W, 3)；模型侧缩放/均值减除由后端内部执行。
本模块的 :func:`resize_keep_aspect` / :func:`make_shape_mapping` 为后端适配层与
坐标映射记录提供的共用工具，analyze 主管线不对送入后端的原图做缩放。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ErrorCode, UiAttentionError

# 允许解码的静态截图格式（第一版：完整、已合成的位图截图）
ALLOWED_FORMATS: frozenset[str] = frozenset({"PNG", "JPEG", "BMP", "TIFF", "WEBP"})

# 明确的颜色处理规则：以下 PIL mode 按记录在案的规则转 RGB；其余模式拒绝（不猜测语义）
_CONVERTIBLE_MODES: frozenset[str] = frozenset({"RGB", "L", "P", "RGBA", "LA", "PA", "CMYK"})

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """文件字节 SHA-256（图片哈希与标注绑定的唯一口径）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class LoadedImage:
    """方向已处理的原图 + 证据元数据。"""

    array: np.ndarray  # (H, W, 3) uint8 RGB
    width: int
    height: int
    format: str
    sha256: str
    source_path: str
    original_mode: str
    orientation_applied: bool
    alpha_composited: bool = False
    composite_background: tuple[int, int, int] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """(H, W)，与 numpy 数组一致。"""
        return (self.height, self.width)


def _exif_orientation(img: Image.Image) -> int | None:
    try:
        exif = img.getexif()
    except Exception:  # noqa: BLE001 - 损坏 EXIF 不得导致崩溃，按“无方向标签”处理
        return None
    value = exif.get(274) if exif else None
    return value if isinstance(value, int) else None


def load_image(path: str | Path, *, composite_background: tuple[int, int, int] | None = None) -> LoadedImage:
    """读取并校验图片：格式/尺寸/方向/可解码性/透明度规则 → RGB uint8 原图。

    ``composite_background`` 仅在图片存在透明度时使用：给定真实合成背景才允许合成；
    缺省时拒绝（INVALID_IMAGE，退出码 2），不擅自填黑或白。
    """
    p = Path(path)
    if not p.is_file():
        raise UiAttentionError(ErrorCode.INVALID_IMAGE, f"图片文件不存在：{p.name}")
    digest = sha256_file(p)
    try:
        with Image.open(p) as img:
            img.load()
            fmt = img.format or ""
            if fmt.upper() not in ALLOWED_FORMATS:
                raise UiAttentionError(
                    ErrorCode.INVALID_IMAGE,
                    f"不支持的图片格式 {fmt or '未知'!r}（允许：{sorted(ALLOWED_FORMATS)}）",
                    {"format": fmt},
                )
            if img.width <= 0 or img.height <= 0:
                raise UiAttentionError(ErrorCode.INVALID_IMAGE, f"图片尺寸无效：{img.width}×{img.height}")
            original_mode = img.mode
            orientation = _exif_orientation(img)
            orientation_applied = orientation is not None and orientation != 1
            work = ImageOps.exif_transpose(img) if orientation_applied else img
            work = _to_rgb(work, original_mode, composite_background)
            alpha_composited = _has_alpha(img) and composite_background is not None
            array = np.asarray(work, dtype=np.uint8)
            if array.ndim != 3 or array.shape[2] != 3:
                raise UiAttentionError(
                    ErrorCode.INVALID_IMAGE,
                    f"颜色处理后必须是 (H, W, 3) RGB，得到 {array.shape}",
                )
            return LoadedImage(
                array=np.ascontiguousarray(array),
                width=int(array.shape[1]),
                height=int(array.shape[0]),
                format=fmt.upper(),
                sha256=digest,
                source_path=str(p),
                original_mode=original_mode,
                orientation_applied=orientation_applied,
                alpha_composited=alpha_composited,
                composite_background=tuple(composite_background) if alpha_composited else None,
            )
    except UiAttentionError:
        raise
    except UnidentifiedImageError as exc:
        raise UiAttentionError(ErrorCode.INVALID_IMAGE, f"图片不可解码：{p.name}", {"reason": str(exc)}) from exc
    except OSError as exc:
        raise UiAttentionError(ErrorCode.INVALID_IMAGE, f"图片读取失败：{p.name}", {"reason": str(exc)}) from exc


def work_mode_bands(img: Image.Image) -> tuple[str, ...]:
    try:
        return img.getbands()
    except Exception:  # noqa: BLE001
        return ()


def _has_alpha(img: Image.Image) -> bool:
    if "A" in work_mode_bands(img):
        return True
    # P 模式带 transparency 表同样视为存在透明度
    return img.mode == "P" and "transparency" in img.info


def _to_rgb(img: Image.Image, original_mode: str, composite_background: tuple[int, int, int] | None) -> Image.Image:
    """明确的颜色处理规则（不猜测、不静默合成）。"""
    if _has_alpha(img):
        if composite_background is None:
            # technical-design §4.2：存在透明度且未给定真实合成背景时拒绝分析，
            # 不擅自填黑或白背景（即使 alpha 全不透明也不猜测合成结果）。
            raise UiAttentionError(
                ErrorCode.INVALID_IMAGE,
                "图片存在透明通道且未提供合成背景：拒绝分析（不擅自填黑/白背景）；"
                "请提供已合成的 RGB 截图，或显式给定 --composite-background",
                {"mode": original_mode},
            )
        bg = tuple(int(c) for c in composite_background)
        if len(bg) != 3 or not all(0 <= c <= 255 for c in bg):
            raise UiAttentionError(
                ErrorCode.INVALID_IMAGE, f"合成背景必须是 RGB 三元组 0-255，得到 {composite_background!r}"
            )
        rgba = img.convert("RGBA")
        base = Image.new("RGB", rgba.size, bg)  # type: ignore[arg-type]
        base.paste(rgba, mask=rgba.split()[-1])
        return base
    if img.mode == "RGB":
        return img
    if img.mode in _CONVERTIBLE_MODES:
        # L/P/CMYK → RGB：明确的记录在案转换（无透明度分支）
        return img.convert("RGB")
    raise UiAttentionError(
        ErrorCode.INVALID_IMAGE,
        f"不支持的颜色模式 {img.mode!r}（不猜测高位深/特殊模式语义）",
        {"mode": img.mode},
    )


# ---------------------------------------------------------------------------
# 缩放与坐标变换记录（后端适配层与 summarize/compare 共用）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResizeResult:
    """保宽高比缩放结果 + 完整变换记录。"""

    array: np.ndarray  # uint8 RGB
    original_shape: tuple[int, int]  # (H, W)
    inference_shape: tuple[int, int]  # (h, w)
    scale: float  # 统一缩放系数（保宽高比）
    method: str = "aspect_preserving_bilinear"

    def shape_mapping(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return make_shape_mapping(self.original_shape, self.inference_shape, self.method, extra)


def make_shape_mapping(
    original_shape: tuple[int, int],
    inference_shape: tuple[int, int],
    method: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 PredictionResult.shape_mapping（原图尺寸、推理尺寸、缩放参数与逆变换方法）。"""
    mapping: dict[str, Any] = {
        "original_shape": [int(original_shape[0]), int(original_shape[1])],
        "inference_shape": [int(inference_shape[0]), int(inference_shape[1])],
        "method": method,
        "inverse": "bilinear_resample_then_renormalize",  # metrics.probability.resample_to_original 消费
    }
    if extra:
        mapping.update(extra)
    return mapping


def resize_keep_aspect(array: np.ndarray, *, max_height: int, max_width: int) -> ResizeResult:
    """保留宽高比缩放到 (max_height, max_width) 界限内；使用完整画面，不切块。

    仅缩小不放大（scale ≤ 1）：小于界限的图保持原尺寸（放大不产生信息，
    且会改变模型观看条件）。变换完整记录于返回值。
    """
    if array.ndim != 3 or array.shape[2] != 3:
        raise UiAttentionError(ErrorCode.INVALID_IMAGE, f"resize 输入必须是 (H, W, 3) RGB，得到 {array.shape}")
    h, w = array.shape[:2]
    if max_height <= 0 or max_width <= 0:
        raise UiAttentionError(ErrorCode.INVALID_IMAGE, f"缩放界限必须为正：{max_width}×{max_height}")
    scale = min(max_height / h, max_width / w, 1.0)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    if (new_h, new_w) == (h, w):
        return ResizeResult(array=array, original_shape=(h, w), inference_shape=(h, w), scale=1.0)
    img = Image.fromarray(array, "RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)
    return ResizeResult(
        array=np.asarray(img, dtype=np.uint8),
        original_shape=(h, w),
        inference_shape=(new_h, new_w),
        scale=scale,
    )


def enforce_size_limits(width: int, height: int, *, max_width: int | None, max_height: int | None) -> None:
    """尺寸超限时明确失败，不静默缩小（technical-design §4.4）。"""
    problems = []
    if max_width is not None and width > max_width:
        problems.append(f"宽度 {width} 超过上限 {max_width}")
    if max_height is not None and height > max_height:
        problems.append(f"高度 {height} 超过上限 {max_height}")
    if problems:
        raise UiAttentionError(
            ErrorCode.INVALID_IMAGE,
            "图片尺寸超过模型输入上限：明确失败（不静默缩小；缩放必须由登记 profile 显式定义）",
            {"problems": problems, "width": width, "height": height},
        )


def original_to_inference_point(x: float, y: float, shape_mapping: dict[str, Any]) -> tuple[float, float]:
    """原图连续坐标 → 推理连续坐标（横竖屏通用；仅缩放，无填充时）。

    坐标为连续画布口径：原图最后一个像素中心 (ow-1) 映射为 iw-iw/ow，
    结果恒在 [0, iw) 画布范围内；往返变换与 :func:`inference_to_original_point` 精确互逆。
    """
    (oh, ow), (ih, iw) = _shapes(shape_mapping)
    return (x * iw / ow, y * ih / oh)


def inference_to_original_point(x: float, y: float, shape_mapping: dict[str, Any]) -> tuple[float, float]:
    """推理坐标 → 原图坐标（逆变换；结果截断在原图范围内）。"""
    (oh, ow), (ih, iw) = _shapes(shape_mapping)
    ox = min(max(x * ow / iw, 0.0), ow - 1.0)
    oy = min(max(y * oh / ih, 0.0), oh - 1.0)
    return (ox, oy)


def _shapes(shape_mapping: dict[str, Any]) -> tuple[tuple[int, int], tuple[int, int]]:
    try:
        oh, ow = (int(v) for v in shape_mapping["original_shape"])
        ih, iw = (int(v) for v in shape_mapping["inference_shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UiAttentionError(
            ErrorCode.INVALID_DENSITY, "shape_mapping 缺少合法尺寸字段", {"reason": str(exc)}
        ) from exc
    if min(oh, ow, ih, iw) <= 0:
        raise UiAttentionError(ErrorCode.INVALID_DENSITY, "shape_mapping 尺寸必须为正")
    return (oh, ow), (ih, iw)


# ---------------------------------------------------------------------------
# 运行目录内的 base.png 存取（渲染用展示副本；非哈希绑定原文件）
# ---------------------------------------------------------------------------


def save_rgb_png(array: np.ndarray, path: str | Path) -> Path:
    """把方向已处理的 RGB uint8 数组确定性编码为 PNG（无时间戳块）。

    落盘于运行目录（runs/ 被 gitignore 覆盖）供 C3 渲染内嵌；
    它是**展示副本**，字节不等于原始上传文件，图片哈希绑定仍以
    :func:`sha256_file`（原文件字节）为准。
    """
    arr = np.asarray(array)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise UiAttentionError(
            ErrorCode.INVALID_IMAGE, f"save_rgb_png 需要 (H, W, 3) uint8，得到 {arr.shape} {arr.dtype}"
        )
    p = Path(path)
    try:
        Image.fromarray(arr, "RGB").save(p, format="PNG")
    except OSError as exc:
        raise UiAttentionError(ErrorCode.IO_ERROR, f"base.png 写入失败：{p.name}", {"reason": str(exc)}) from exc
    return p


def load_rgb_png(path: str | Path) -> np.ndarray:
    """读回运行目录内的 base.png（渲染链路复用；校验为 RGB uint8）。"""
    p = Path(path)
    try:
        with Image.open(p) as img:
            img.load()
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise UiAttentionError(
            ErrorCode.ARTIFACT_CHECK_FAILED, f"base.png 读取失败：{p.name}", {"reason": str(exc)}
        ) from exc
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise UiAttentionError(ErrorCode.ARTIFACT_CHECK_FAILED, f"base.png 必须是 RGB，得到 {arr.shape}")
    return np.ascontiguousarray(arr)
