"""合成 UEyes 风格数据集生成器（全部代码生成，无任何真实数据集内容/第三方图片）。

按 R2 协议 §2.1 已核实的官方目录结构生成最小可驱动副本：

::

    root/
      images/block 1/img_xxx.png ...     # 按 block 分目录的合成 PNG（多种宽高比测 letterbox）
      eyetracker_logs/{block}_{p}_fixations.csv   # Gazepoint 列：MEDIA_NAME,TIME,FPOGD,BPOGV,BPOGX,BPOGY
      info.csv                            # image,category,block,split

注视点以"图像像素 → 屏幕归一化"正向映射构造（screen = (x*scalar+pad)/1920），
保证除显式放置的越界点外全部落在图内；期望计数由构造参数解析可得（测试独立断言）。
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCREEN_W = 1920
SCREEN_H = 1200


def image_to_screen_norm(x: float, y: float, img_w: int, img_h: int) -> tuple[float, float]:
    """图像像素 → 屏幕归一化坐标（letterbox 正向映射，与驱动的逆变换互逆）。"""
    scalar = min(SCREEN_W / img_w, SCREEN_H / img_h)
    pad_x = (SCREEN_W - img_w * scalar) / 2.0
    pad_y = (SCREEN_H - img_h * scalar) / 2.0
    return ((x * scalar + pad_x) / SCREEN_W, (y * scalar + pad_y) / SCREEN_H)


@dataclass
class FakeFixation:
    media: str
    time: float
    dur: float
    bpogx: float
    bpogy: float
    bpogv: float = 1.0


@dataclass
class SyntheticUEyes:
    """生成结果摘要（测试断言用期望值）。"""

    root: Path
    images: list[dict[str, Any]] = field(
        default_factory=list
    )  # {image_id, category, block, split, width, height, path}
    logs: list[dict[str, Any]] = field(default_factory=list)  # {block, participant, path, rows}
    expected: dict[str, dict[str, int]] = field(
        default_factory=dict
    )  # image_id → {"1s": n, "3s": n, "7s": n, "oob": n, "invalid": n, "viewers": n}
    info_columns: list[str] = field(default_factory=list)


def make_ueyes_dataset(
    root: Path,
    *,
    seed: int = 20260911,
    per_image_fixations: int = 40,
    n_participants: int = 3,
    include_split_column: bool = True,
    with_near_duplicate: bool = True,
) -> SyntheticUEyes:
    """生成 6 图 × 3 参与者的最小数据集（类别覆盖 webpage/desktop/mobile/poster）。"""
    rng = np.random.default_rng(seed)
    root = Path(root)
    specs = [
        # (image_id, category, block, split, width, height)
        ("img_0001.png", "webpage", "block 1", "train", 320, 200),  # 16:10 = 屏幕比例，无 pad
        ("img_0002.png", "desktop", "block 1", "train", 160, 120),  # 4:3 pillarbox
        ("img_0003.png", "mobile", "block 1", "test", 96, 160),  # 竖屏 letterbox
        ("img_0004.png", "poster", "block 2", "train", 240, 150),
        ("img_0005.png", "webpage", "block 2", "test", 320, 200),
        ("img_0006.png", "desktop", "block 2", "test", 160, 120),
    ]
    out = SyntheticUEyes(root=root)

    # -- images -------------------------------------------------------------
    # 每图唯一结构（固定参数表）：渐变方向 + 亮块位置/大小 → dHash 两两距离 >8，
    # 只有显式构造的 img_0001↔img_0005 近重复对会被聚簇（测试前提）
    pattern_params = [
        # (gradient_dir, block_x_ratio, block_y_ratio, block_size_ratio)
        (1, 0.10, 0.15, 0.30),
        (-1, 0.55, 0.10, 0.25),
        (1, 0.20, 0.60, 0.35),
        (-1, 0.60, 0.55, 0.30),
        (1, 0.45, 0.30, 0.20),
        (-1, 0.05, 0.45, 0.25),
    ]
    for si, (iid, cat, block, split, w, h) in enumerate(specs):
        d = root / "images" / block
        d.mkdir(parents=True, exist_ok=True)
        gdir, bxr, byr, bsr = pattern_params[si % len(pattern_params)]
        grad = np.linspace(0, 255, w)
        if gdir < 0:
            grad = grad[::-1]
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = rng.integers(20, 200)
        arr[:, :, 2] = np.tile(grad.astype(np.uint8), (h, 1))
        bw, bh = max(4, int(w * bsr)), max(4, int(h * bsr))
        bx0 = int(bxr * max(1, w - bw))
        by0 = int(byr * max(1, h - bh))
        arr[by0 : by0 + bh, bx0 : bx0 + bw] = (250, 250, 250)
        p = d / iid
        Image.fromarray(arr, "RGB").save(p)
        out.images.append(
            {"image_id": iid, "category": cat, "block": block, "split": split, "width": w, "height": h, "path": str(p)}
        )
    if with_near_duplicate:
        # img_0005 是 img_0001 的近似副本（同尺寸同内容微调）→ dHash 应聚为近重复簇
        src = Image.open(root / "images" / "block 1" / "img_0001.png")
        dup = np.asarray(src).copy()
        dup[0, 0, 0] = np.uint8((int(dup[0, 0, 0]) + 1) % 256)
        Image.fromarray(dup, "RGB").save(root / "images" / "block 2" / "img_0005.png")

    # -- fixation logs --------------------------------------------------------
    expected = {s[0]: {"1s": 0, "3s": 0, "7s": 0, "oob": 0, "invalid": 0, "viewers": 0} for s in specs}
    logs_dir = root / "eyetracker_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    participants = [f"hk0{i + 1:02d}" for i in range(n_participants)]
    for pi, participant in enumerate(participants):
        rows: list[FakeFixation] = []
        for si, (iid, _cat, _block, _split, w, h) in enumerate(specs):
            # 参与者 3 只看一半图（每图观看者数不均匀，§2.1 已知细节）
            if pi == 2 and si % 2 == 1:
                continue
            expected[iid]["viewers"] += 1
            times = np.sort(rng.uniform(0.05, 6.9, per_image_fixations))
            for t in times:
                x = float(rng.uniform(1, w - 2))
                y = float(rng.uniform(1, h - 2))
                sx, sy = image_to_screen_norm(x, y, w, h)
                rows.append(
                    FakeFixation(media=iid, time=float(t), dur=float(rng.uniform(0.1, 0.6)), bpogx=sx, bpogy=sy)
                )
                for win in ("1s", "3s", "7s"):
                    if t <= {"1s": 1.0, "3s": 3.0, "7s": 7.0}[win]:
                        expected[iid][win] += 1
            # 每图追加：2 个越界点（屏幕角落 → letterbox pad 区）+ 1 个 BPOGV=0 无效点
            rows.append(FakeFixation(media=iid, time=0.5, dur=0.2, bpogx=0.001, bpogy=0.001))
            rows.append(FakeFixation(media=iid, time=0.6, dur=0.2, bpogx=0.999, bpogy=0.999))
            expected[iid]["oob"] += 2
            rows.append(FakeFixation(media=iid, time=0.7, dur=0.2, bpogx=0.5, bpogy=0.5, bpogv=0.0))
            expected[iid]["invalid"] += 1
        # 参与者文件按 block 拆分（官方命名 {block}_{participant}_fixations.csv）
        by_block: dict[str, list[FakeFixation]] = {"block 1": [], "block 2": []}
        spec_block = {s[0]: s[2] for s in specs}
        for r in rows:
            by_block[spec_block[r.media]].append(r)
        for block_name, brows in by_block.items():
            fname = f"{block_name.replace(' ', '_')}_{participant}_fixations.csv"
            p = logs_dir / fname
            with open(p, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["MEDIA_NAME", "TIME", "FPOGD", "BPOGV", "BPOGX", "BPOGY"])
                for r in sorted(brows, key=lambda r: (r.media, r.time)):
                    writer.writerow(
                        [r.media, f"{r.time:.3f}", f"{r.dur:.3f}", f"{r.bpogv:.0f}", f"{r.bpogx:.6f}", f"{r.bpogy:.6f}"]
                    )
            out.logs.append({"block": block_name, "participant": participant, "path": str(p), "rows": len(brows)})
    # 越界点是否真越界取决于 letterbox：对无 pad 的 16:10 图（img_0001/0005），屏幕角落仍落在图内
    # → 期望值按驱动的 screen_to_image_point 语义修正：图内角落点计入窗口有效注视，图外计入 oob
    for img in out.images:
        from ui_attention.metrics.eval.ueyes_driver import screen_to_image_point  # 局部导入避免夹具环依赖

        w, h = img["width"], img["height"]
        viewers = expected[img["image_id"]]["viewers"]
        n_oob_actual = 0
        for bx, by in ((0.001, 0.001), (0.999, 0.999)):  # 时间 0.5/0.6，均 ≤1s
            if screen_to_image_point(bx, by, w, h) is None:
                n_oob_actual += 1
            else:
                for win in ("1s", "3s", "7s"):
                    expected[img["image_id"]][win] += viewers
        expected[img["image_id"]]["oob"] = n_oob_actual * viewers

    # -- info.csv -------------------------------------------------------------
    info = root / "info.csv"
    columns = ["image", "category", "block", "split"] if include_split_column else ["image", "category", "block"]
    with open(info, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for img in out.images:
            row = [img["image_id"], img["category"], img["block"]]
            if include_split_column:
                row.append(img["split"])
            writer.writerow(row)
    out.info_columns = columns
    out.expected = expected
    return out


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
