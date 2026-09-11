"""生成 game-ui-attention Skill 示例用合成图（禁止真实游戏截图/内部资产）。

产出（与本脚本同目录）：
- synthetic-reward-summary.png    图 A：抽象"奖励结算"界面基线版（960×540）
- synthetic-reward-summary-v2.png 图 B：变体版（领取按钮放大提亮、装饰光效减弱）

纯 PIL + numpy 代码生成的几何抽象界面：不含任何真实游戏素材、字体资源或
内部资产；所有"文字"以色块抽象表示。两图画布尺寸一致（compare 要求同画布）。

运行（项目根 cwd）：
    $env:PYTHONPATH = "$PWD\\src"
    & .\\.venv\\Scripts\\python.exe skill\\examples\\make_synthetic_images.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
W, H = 960, 540


def _vertical_gradient(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    arr = np.zeros((H, W, 3), dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, H)[:, None]
    for c in range(3):
        arr[:, :, c] = top[c] + (bottom[c] - top[c]) * ramp
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def _radial_glow(
    base: Image.Image,
    center: tuple[float, float],
    radius: float,
    color: tuple[int, int, int],
    strength: float,
) -> Image.Image:
    """在 base 上叠加径向光晕（strength 0-1；装饰强度在两图间变化的实现点）。"""
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    falloff = np.clip(1.0 - d / radius, 0.0, 1.0) ** 2 * strength
    arr = np.asarray(base, dtype=np.float64)
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1.0 - falloff) + color[c] * falloff
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _abstract_text_line(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    """用断续小色块抽象表示一行文字（不使用任何字体资源）。"""
    x0, y0, x1, y1 = box
    cy = (y0 + y1) // 2
    x = x0
    rng = np.random.default_rng(7)  # 固定种子：两图/多次生成一致
    while x < x1 - 8:
        w = int(rng.integers(10, 26))
        draw.rectangle([x, cy - 4, min(x + w, x1), cy + 4], fill=color)
        x += w + 8


def build(variant: bool) -> Image.Image:
    """variant=False → 图 A 基线；variant=True → 图 B（按钮强化、装饰减弱）。"""
    img = _vertical_gradient((26, 16, 48), (42, 26, 74))
    # 背景庆祝光晕：变体减弱（装饰竞争注意力 ↓ 的设计改动之一）
    img = _radial_glow(img, (480, 240), 420, (255, 214, 140), 0.16 if variant else 0.34)
    # 标题周围光晕：变体减弱
    img = _radial_glow(img, (480, 122), 190, (255, 236, 190), 0.18 if variant else 0.40)
    draw = ImageDraw.Draw(img)

    # 中央面板
    draw.rounded_rectangle([300, 70, 659, 469], radius=18, fill=(36, 26, 62), outline=(106, 90, 154), width=3)

    # 标题条（两图相同；id=title）
    draw.rectangle([360, 100, 599, 143], fill=(232, 179, 74), outline=(122, 90, 16), width=2)
    _abstract_text_line(draw, (382, 112, 578, 132), (58, 40, 10))

    # 奖励槽 ×3（两图相同；合并标注 id=reward-row）
    for i, x in enumerate((340, 450, 560)):
        jewel = ((200, 70, 70), (70, 160, 90), (80, 110, 200))[i]
        draw.rectangle([x, 190, x + 89, 289], fill=(52, 40, 84), outline=(150, 130, 200), width=2)
        draw.rectangle([x + 14, 204, x + 75, 265], fill=jewel, outline=(230, 220, 250), width=2)
        draw.rectangle([x + 24, 272, x + 65, 280], fill=(190, 180, 220))

    # 装饰光效带（id=deco-band）：变体减弱
    band = (120, 60, 150) if variant else (214, 110, 240)
    draw.rectangle([320, 330, 639, 359], fill=band, outline=(240, 200, 255) if not variant else (150, 120, 170), width=1)

    # 领取按钮（id=claim-button）：变体放大 + 提亮（核心设计改动）
    if variant:
        bx0, by0, bx1, by1 = 360, 368, 599, 439
        fill, edge = (255, 215, 94), (122, 90, 16)
        draw.rounded_rectangle([bx0 - 6, by0 - 6, bx1 + 6, by1 + 6], radius=16, fill=(90, 66, 20))
    else:
        bx0, by0, bx1, by1 = 380, 380, 579, 439
        fill, edge = (242, 193, 78), (122, 90, 16)
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=12, fill=fill, outline=edge, width=3)
    _abstract_text_line(draw, (bx0 + 24, by0 + 14, bx1 - 24, by1 - 14), (58, 40, 10))

    # 关闭按钮（两图相同；id=close-button）
    draw.rectangle([614, 84, 645, 115], fill=(138, 122, 176), outline=(220, 210, 240), width=2)
    draw.line([622, 92, 638, 108], fill=(40, 30, 60), width=3)
    draw.line([638, 92, 622, 108], fill=(40, 30, 60), width=3)

    return img


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    outputs = {
        "synthetic-reward-summary.png": build(variant=False),
        "synthetic-reward-summary-v2.png": build(variant=True),
    }
    for name, img in outputs.items():
        p = HERE / name
        img.save(p, "PNG")
        print(f"{name}: {img.width}x{img.height} sha256={sha256_of(p)}")


if __name__ == "__main__":
    main()
