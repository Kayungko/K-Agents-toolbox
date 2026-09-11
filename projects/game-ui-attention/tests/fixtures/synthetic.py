"""合成夹具生成器：图片、概率图、AOI/请求 JSON（全部代码生成，无真实截图）。

使用方式（测试文件内）::

    import sys
    from pathlib import Path
    _TESTS = Path(__file__).resolve().parents[1]
    if str(_TESTS) not in sys.path:
        sys.path.insert(0, str(_TESTS))
    from fixtures import synthetic
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 合成图片
# ---------------------------------------------------------------------------


def solid_color_png(path: Path, width: int = 64, height: int = 48, color: tuple[int, int, int] = (30, 60, 90)) -> Path:
    """纯色合成 PNG（横屏 64×48 默认）。"""
    Image.new("RGB", (width, height), color).save(path)
    return path


def portrait_png(path: Path, width: int = 48, height: int = 64) -> Path:
    """竖屏纯色合成 PNG（坐标映射竖屏用例）。"""
    return solid_color_png(path, width=width, height=height, color=(10, 200, 120))


def gradient_png(path: Path, width: int = 1920, height: int = 1080) -> Path:
    """大尺寸水平渐变合成 PNG（1920×1080 默认；纯 numpy 生成）。"""
    xs = np.linspace(0, 255, width, dtype=np.float64)
    row = np.stack([xs, 255 - xs, np.full(width, 128.0)], axis=-1)
    arr = np.repeat(row[None, :, :], height, axis=0).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path)
    return path


def rgba_png(path: Path, width: int = 64, height: int = 48) -> Path:
    """带透明通道的合成 PNG（未给合成背景时必须拒绝，不擅自填黑/白）。"""
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :, 0] = 200
    arr[:, :, 3] = 128  # 半透明
    Image.fromarray(arr, "RGBA").save(path)
    return path


def opaque_alpha_png(path: Path, width: int = 32, height: int = 32) -> Path:
    """带 alpha 通道但 alpha 全 255（完全不透明）的 PNG：仍按“存在透明度通道”处理，
    未给合成背景时拒绝（保守口径：不猜测合成结果）。"""
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :, 1] = 90
    arr[:, :, 3] = 255
    Image.fromarray(arr, "RGBA").save(path)
    return path


def oriented_jpeg(path: Path, width: int = 40, height: int = 20, orientation: int = 6) -> Path:
    """带 EXIF Orientation 的合成 JPEG（默认 6=顺时针 90°，显示尺寸应为 20×40）。"""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, : width // 2] = (250, 0, 0)  # 左半红，用于断言方向处理后的像素位置
    img = Image.fromarray(arr, "RGB")
    exif = Image.Exif()
    exif[274] = orientation  # Orientation tag
    img.save(path, format="JPEG", exif=exif)
    return path


def corrupt_file(path: Path) -> Path:
    """不可解码文件（可解码性校验用例）。"""
    path.write_bytes(b"not-an-image\x00\xff\xfe")
    return path


# ---------------------------------------------------------------------------
# 合成概率图（已知解析性质）
# ---------------------------------------------------------------------------


def uniform_probability(shape: tuple[int, int]) -> np.ndarray:
    """均匀概率图：任一区域 mass == 面积占比、relative_density == 1。"""
    n = int(shape[0]) * int(shape[1])
    return np.full(shape, 1.0 / n, dtype=np.float64)


def block_probability(shape: tuple[int, int], block: tuple[int, int, int, int], mass: float) -> np.ndarray:
    """已知局部质量的构造概率图。

    block=(x, y, w, h) 半开区间内总质量恰为 ``mass``（均匀分布），
    其余像素均分 ``1 - mass``；全图 sum=1。区域==block 时解析期望：
    probability_mass == mass、area_fraction == w*h/(H*W)、relative_density == mass/area_fraction。
    """
    h, w = shape
    x, y, bw, bh = block
    arr = np.zeros(shape, dtype=np.float64)
    n_block = bw * bh
    n_rest = h * w - n_block
    arr[y : y + bh, x : x + bw] = mass / n_block
    rest_value = (1.0 - mass) / n_rest if n_rest > 0 else 0.0
    outside = np.ones(shape, dtype=np.bool_)
    outside[y : y + bh, x : x + bw] = False
    arr[outside] = rest_value
    return arr


def log_density_from_probability(P: np.ndarray) -> np.ndarray:
    """由严格概率图构造 log_density（log 域），供 P=exp(L−logsumexp(L)) 往返测试。"""
    with np.errstate(divide="ignore"):
        return np.log(P, dtype=np.float64)


# ---------------------------------------------------------------------------
# AOI / 请求 JSON 夹具
# ---------------------------------------------------------------------------

VALID_RECT_REGION: dict[str, Any] = {
    "id": "claim-button",
    "label": "领取按钮",
    "role": "primary-action",
    "geometry": {"type": "rect", "x": 10, "y": 8, "width": 32, "height": 16},
    "source": "manual",
    "status": "confirmed",
}

VALID_POLYGON_REGION: dict[str, Any] = {
    "id": "banner",
    "label": "标题横幅",
    "role": "title",
    "geometry": {"type": "polygon", "points": [[2, 2], [30, 4], [28, 20], [4, 18]]},
    "source": "manual",
    "status": "candidate",
}

SELF_INTERSECTING_POLYGON: dict[str, Any] = {
    "id": "bowtie",
    "geometry": {"type": "polygon", "points": [[0, 0], [20, 20], [20, 0], [0, 20]]},  # 蝴蝶结自交
    "source": "manual",
    "status": "candidate",
}

COLLINEAR_POLYGON: dict[str, Any] = {
    "id": "line",
    "geometry": {"type": "polygon", "points": [[1, 1], [5, 5], [9, 9]]},  # 全共线
    "source": "manual",
    "status": "candidate",
}

ZERO_AREA_RECT: dict[str, Any] = {
    "id": "zero",
    "geometry": {"type": "rect", "x": 4, "y": 4, "width": 0, "height": 8},
    "source": "manual",
    "status": "candidate",
}

OUT_OF_BOUNDS_RECT: dict[str, Any] = {
    "id": "oob",
    "geometry": {"type": "rect", "x": 60, "y": 40, "width": 32, "height": 16},  # 超出 64×48
    "source": "manual",
    "status": "candidate",
}


def base_request(regions: list[dict[str, Any]] | None = None, **overrides: Any) -> dict[str, Any]:
    """构造合法 AnalyzeRequest v1 dict（可覆盖字段）。"""
    req: dict[str, Any] = {
        "schema_version": "game-ui-attention-request/v1",
        "image": "./input.png",
        "player_goal": "查看奖励并领取",
        "screen_type": "reward-summary",
        "backend_profile": "foveacast-onnx-3s-v1",
    }
    if regions is not None:
        req["regions"] = regions
    req.update(overrides)
    return req


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# analysis v1 记录夹具（供 contracts/cli 测试复用）
# ---------------------------------------------------------------------------

IMAGE_SHA = "1" * 64
CONFIG_HASH = "c" * 64
WEIGHT_SHA = "8" * 64


def sample_region_result(
    rid: str = "claim-button",
    mass: float = 0.25,
    area_px: int = 512,
    total_px: int = 64 * 48,
    geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    area_fraction = area_px / total_px
    return {
        "id": rid,
        "label": f"标签-{rid}",
        "role": "primary-action",
        "source": "manual",
        "status": "confirmed",
        "geometry": geometry or {"type": "rect", "x": 10, "y": 8, "width": 32, "height": 16},
        "area_px": area_px,
        "area_fraction": area_fraction,
        "probability_mass": mass,
        "relative_density": mass / area_fraction,
    }


def sample_analysis_dict(
    analysis_id: str = "analysis-test-0001",
    regions: list[dict[str, Any]] | None = None,
    width: int = 64,
    height: int = 48,
    **overrides: Any,
) -> dict[str, Any]:
    """构造合法 analysis v1 dict（字段与 data-contract §4 对应）。"""
    record: dict[str, Any] = {
        "schema_version": "game-ui-attention-analysis/v1",
        "analysis_id": analysis_id,
        "computation_status": "complete",
        "review_status": "not_requested",
        "evidence_type": "model_prediction",
        "created_at_utc": "2026-09-11T00:00:00Z",
        "metrics_version": "aoi-metrics/v1",
        "player_goal": "查看奖励并领取",
        "input": {
            "image_path": "input.png",
            "image_sha256": IMAGE_SHA,
            "width": width,
            "height": height,
            "inference_width": 320,
            "inference_height": 240,
            "screen_type": "reward-summary",
        },
        "model": {
            "backend_id": "foveacast-onnx-3s",
            "version": "0.1.0",
            "weights_sha256": [WEIGHT_SHA],
            "code_version": "0.1.0",
            "capabilities": ["spatial_density"],
        },
        "profile": {
            "profile_name": "foveacast-onnx-3s-v1",
            "config_hash": CONFIG_HASH,
            "preprocessing": {"target_hw": [240, 320], "value_range": [0, 255]},
            "centerbias": None,
            "viewing_conditions": {"window": "3s", "assumption": "实验假设"},
        },
        "runtime": {
            "dependencies": {"numpy": "2.5.3", "onnxruntime": "1.30.0"},
            "device": "cpu",
            "precision": "fp32",
            "elapsed_ms": 100.0,
            "peak_mem_mb": 256.0,
        },
        "regions": regions if regions is not None else [sample_region_result()],
        "limitations": ["合成夹具：非真实模型输出"],
        "artifacts": [],
        "errors": [],
    }
    record.update(overrides)
    return record
