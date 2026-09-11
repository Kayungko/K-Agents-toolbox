"""UEyes 数据集适配与评估驱动（benchmark-protocol.md R2 协议实跑入口，L1 批准 2026-09-11）。

职责链（L2 任务包②）：
1. 数据集布局发现与 zip md5 核对（Zenodo 8010312 记录值）；
2. info.csv 官方 train/test 划分冻结 → splits.v1.json（含文件哈希；官方标志优先，
   缺失时协议 §3.3 确定性哈希备选划分）；
3. Gazepoint 原始 CSV → 注视点重建（复现官方管线：1920×1200 letterbox 居中、
   BPOGV=1 有效、TIME≤窗口前缀截断、越界丢弃计数、跨参与者按图聚合）；
4. 泄漏审计自动化（§3.4.5：划分哈希一致、近重复 dHash 簇不跨划分、CB 仅 train、
   排除清单分侧记录）；
5. CB train-only 构造 → npz 版本化 + sha256；
6. 评估编排：registry 推理接入（foveacast-onnx-{1s,3s,7s}-v1，未登记 profile 结构化跳过）、
   逐图 CSV、bootstrap/配对差值/Holm、表 A-E、S0-S3 判定。

红线：文献值/作者报告值只进独立参考表；md5 不符/S0 不过/NaN → 结构化退出，
不产出部分成功；产物写 local-data/eval-output/（忽略目录）。

命令行入口::

    python -m ui_attention.metrics.eval.ueyes_driver --dataset local-data/ueyes/extracted \
        --out local-data/eval-output [--windows 3s] [--profiles foveacast-onnx-3s-v1]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ...errors import ErrorCode, UiAttentionError
from .baselines import CenterBiasBaseline, CenterBiasFitInput
from .config import EvalConfig
from .groundtruth import FixationSet, build_fixation_set
from .pipeline import EvalGateError

# ---------------------------------------------------------------------------
# 常量（来源：benchmark-protocol §2.1，Zenodo 8010312 页面已核实值）
# ---------------------------------------------------------------------------

#: UEyes_dataset.zip 的 md5（Zenodo 记录；下载后必须校验，不符 → 结构化失败）
UEYES_ZIP_MD5 = "c2d53e6af0a47e1f459416d6839ec2c1"

#: 官方真值管线的屏幕假设（§2.1：1920×1200 letterbox 居中）
UEYES_SCREEN_WIDTH = 1920
UEYES_SCREEN_HEIGHT = 1200

#: 观看窗口（§5：同一 trial 注视序列的前缀截断）
WINDOW_SECONDS: dict[str, float] = {"1s": 1.0, "3s": 3.0, "7s": 7.0}

SPLITS_VERSION = "splits.v1.json"

_LOG_NAME_RE = re.compile(r"^(?P<block>.+)_(?P<participant>[A-Za-z0-9]+)_fixations\.csv$")


class DatasetError(UiAttentionError):
    """数据集缺失/结构不符/校验失败（评估侧结构化错误，退出码 5=产物校验语义）。"""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.ARTIFACT_CHECK_FAILED, message, dict(details or {}))
        self.details.setdefault("reason", "dataset_error")


# ---------------------------------------------------------------------------
# Part 1：布局发现与 md5 核对
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetLayout:
    """UEyes 解压目录布局（仓库 README 已核实结构；不符 → DatasetError）。"""

    root: Path
    images_dir: Path
    logs_dir: Path
    info_csv: Path

    def describe(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "images_dir": str(self.images_dir),
            "logs_dir": str(self.logs_dir),
            "info_csv": str(self.info_csv),
        }


def discover_layout(root: str | Path) -> DatasetLayout:
    """发现数据集目录结构；缺关键路径 → DatasetError（不猜测、不静默降级）。

    划分权威文件名按实际数据勘误：zip 内为 ``image_types.csv``（L2 于 2026-09-11
    登记勘误：R2 协议 §3.2 所写 info.csv 在 zip 内不存在）；兼容 info.csv 命名。
    """
    r = Path(root)
    if not r.is_dir():
        raise DatasetError(f"数据集根目录不存在：{r}（UEyes 下载/解压由 C2 负责，就绪后重试）")
    images = r / "images"
    logs = r / "eyetracker_logs"
    info: Path | None = None
    for name in ("image_types.csv", "info.csv"):
        candidate = r / name
        if candidate.is_file():
            info = candidate
            break
    missing = [str(p) for p in (images, logs) if not p.exists()]
    if info is None:
        missing.append(str(r / "image_types.csv|info.csv"))
    if missing:
        raise DatasetError(
            "数据集结构不完整（与仓库 README 已核实结构不符）",
            {"missing": missing, "root": str(r)},
        )
    assert info is not None
    return DatasetLayout(root=r, images_dir=images, logs_dir=logs, info_csv=info)


def md5_file(path: str | Path, chunk: int = 1 << 22) -> str:
    """大文件 md5（流式，12.9GB zip 可安全计算）。"""
    h = hashlib.md5()  # noqa: S324 - 数据集完整性核对口径（Zenodo 发布值即 md5），非安全用途
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_zip_md5(zip_path: str | Path, expected: str = UEYES_ZIP_MD5) -> dict[str, Any]:
    """校验 zip md5 == Zenodo 记录值；不符 → DatasetError（红线：md5 不符结构化退出）。"""
    actual = md5_file(zip_path)
    ok = actual == expected
    if not ok:
        raise DatasetError(
            "UEyes_dataset.zip md5 与 Zenodo 8010312 记录值不符：拒绝使用（下载损坏或版本不符）",
            {"expected": expected, "actual": actual},
        )
    return {"expected": expected, "actual": actual, "ok": True}


def load_eval_image(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """评估复现路径的图像加载（与分析 CLI 的"透明拒绝"口径不同，属两条路径）。

    官方口径对齐：UEyes 官方管线为 cv2 基（``imread`` 默认丢弃 alpha、保留存储 RGB 值，
    不做合成）。本加载器采用确定性等价规则：EXIF 方向处理 → ``convert("RGB")``
    （RGBA 丢 alpha 不合成、不填黑/白；P/L/CMYK 按 PIL 确定转换）。
    返回 (uint8 RGB 数组, 模式审计信息)；全部事实（模式、alpha 极值、真透明清单）
    进 audit.image_mode_audit 与 limitations，不静默。

    实测（2026-09-11 全量扫描）：1980 图 = RGB 1664 / RGBA 304（其中 6 张真透明）/ P 10 / CMYK 1 / L 1。
    """
    from PIL import Image, ImageOps

    p = Path(path)
    try:
        with Image.open(p) as img:
            img.load()
            info: dict[str, Any] = {"mode": img.mode, "has_alpha": "A" in img.getbands()}
            if info["has_alpha"]:
                lo, hi = img.getchannel("A").getextrema()
                info["alpha_extrema"] = [int(lo), int(hi)]
                info["non_opaque_alpha"] = bool(lo < 255)
            work = ImageOps.exif_transpose(img)
            arr = np.asarray(work.convert("RGB"), dtype=np.uint8)
    except OSError as exc:
        raise DatasetError(f"评估图像读取失败：{p.name}", {"reason": str(exc)}) from exc
    return np.ascontiguousarray(arr), info


# ---------------------------------------------------------------------------
# Part 2：info.csv 与划分冻结
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageMeta:
    """每图元数据（info.csv 行 + 解析出的图像尺寸）。"""

    image_id: str  # 图片名（含扩展名，与 MEDIA_NAME 对应）
    category: str  # webpage / desktop / mobile / poster
    block: str
    official_split: str | None  # info.csv 官方标志（train/test；None=缺失）
    rel_path: str | None = None  # images/ 下相对路径（发现后回填）
    width: int = 0
    height: int = 0


#: 划分权威文件列名候选（实际 UEyes zip 内为 image_types.csv：'Image Name;Category;Block;Train/Test'，
#: 分隔符 ';'——L2 勘误登记 2026-09-11；防御性发现覆盖两种命名与分隔符）
_INFO_COLUMN_CANDIDATES = {
    "image_id": ("image name", "image", "image_name", "imagename", "media_name", "filename", "file", "name"),
    "category": ("category", "type", "ui_type", "class"),
    "block": ("block", "block_id", "blockid", "folder"),
    "split": ("train/test", "split", "train_test", "set", "partition", "istrain", "is_train", "train"),
}

_SPLIT_VALUE_MAP = {
    "train": "train",
    "test": "test",
    "1": "train",
    "0": "test",
    "true": "train",
    "false": "test",
}


def _resolve_column(fieldnames: Sequence[str], key: str) -> str | None:
    lowered = {f.strip().lower(): f for f in fieldnames}
    for cand in _INFO_COLUMN_CANDIDATES[key]:
        if cand in lowered:
            return lowered[cand]
    return None


def _normalize_split_value(raw: str) -> str | None:
    v = raw.strip().lower()
    if v in _SPLIT_VALUE_MAP:
        return _SPLIT_VALUE_MAP[v]
    if v.startswith("train"):
        return "train"
    if v.startswith("test") or v.startswith("val"):
        return "test" if v.startswith("test") else "val"
    return None


@dataclass(frozen=True)
class InfoTable:
    """划分权威文件解析结果（含来源结构，如实写入 splits.v1.json notes）。"""

    metas: tuple[ImageMeta, ...]
    source_file: str  # 实际文件名（image_types.csv 或 info.csv）
    delimiter: str
    columns: tuple[str, ...]
    n_rows: int
    split_counts: dict[str, int]


def _sniff_delimiter(first_line: str) -> str:
    """分隔符嗅探：实际 image_types.csv 用 ';'（R2 协议记录的 info.csv 为 ','）。"""
    if first_line.count(";") > first_line.count(","):
        return ";"
    return ","


def load_info_table(path: str | Path) -> InfoTable:
    """解析划分权威文件（image_types.csv / info.csv）；关键列缺失 → DatasetError。"""
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"划分权威文件不存在：{p}")
    with open(p, newline="", encoding="utf-8-sig") as f:
        first_line = f.readline()
    delimiter = _sniff_delimiter(first_line)
    with open(p, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = [fn for fn in (reader.fieldnames or []) if fn.strip()]
        col_image = _resolve_column(fieldnames, "image_id")
        if col_image is None:
            raise DatasetError(
                f"{p.name} 无法识别图片名列（预期候选之一），拒绝猜测",
                {"fieldnames": list(fieldnames), "candidates": list(_INFO_COLUMN_CANDIDATES["image_id"])},
            )
        col_cat = _resolve_column(fieldnames, "category")
        col_block = _resolve_column(fieldnames, "block")
        col_split = _resolve_column(fieldnames, "split")
        metas: list[ImageMeta] = []
        split_counts: dict[str, int] = {}
        for row in reader:
            image_id = (row.get(col_image) or "").strip()
            if not image_id:
                continue
            split_raw = (row.get(col_split) or "").strip() if col_split else ""
            official = _normalize_split_value(split_raw) if split_raw else None
            if official:
                split_counts[official] = split_counts.get(official, 0) + 1
            metas.append(
                ImageMeta(
                    image_id=image_id,
                    category=(row.get(col_cat) or "unknown").strip() if col_cat else "unknown",
                    block=(row.get(col_block) or "").strip() if col_block else "",
                    official_split=official,
                )
            )
    if not metas:
        raise DatasetError(f"{p.name} 无有效数据行")
    return InfoTable(
        metas=tuple(metas),
        source_file=p.name,
        delimiter=delimiter,
        columns=tuple(fieldnames),
        n_rows=len(metas),
        split_counts=split_counts,
    )


def load_info_csv(path: str | Path) -> list[ImageMeta]:
    """兼容入口：只返回每图元数据列表（来源结构用 :func:`load_info_table`）。"""
    return list(load_info_table(path).metas)


def discover_image_paths(layout: DatasetLayout, metas: Sequence[ImageMeta]) -> list[ImageMeta]:
    """在 images/（按 block 分目录）下定位每图路径并读取尺寸；回填新 ImageMeta 列表。"""
    from PIL import Image

    index: dict[str, Path] = {}
    for p in layout.images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            index.setdefault(p.name, p)
    out: list[ImageMeta] = []
    missing: list[str] = []
    for m in metas:
        p = index.get(m.image_id) or index.get(Path(m.image_id).name)
        if p is None:
            missing.append(m.image_id)
            continue
        with Image.open(p) as img:
            w, h = img.size
        rel = str(p.relative_to(layout.root)).replace("\\", "/")
        out.append(ImageMeta(m.image_id, m.category, m.block, m.official_split, rel, int(w), int(h)))
    if missing:
        raise DatasetError(
            f"{len(missing)} 张 info.csv 图片在 images/ 下未找到（示例前 5：{missing[:5]}）",
            {"n_missing": len(missing)},
        )
    return out


def fallback_hash_split(image_ids: Iterable[str], protocol_version: str) -> dict[str, str]:
    """备选自产划分（§3.3，完全确定、可冻结）：h = SHA256(protocol_version|image_id)，
    h mod 100：0-69 → train，70-84 → val，85-99 → test。仅官方划分不可用时启用。"""
    out: dict[str, str] = {}
    for iid in sorted(image_ids):
        h = hashlib.sha256(f"{protocol_version}|{iid}".encode()).hexdigest()
        bucket = int(h[:8], 16) % 100
        out[iid] = "train" if bucket < 70 else ("val" if bucket < 85 else "test")
    return out


def fallback_cluster_constrained_split(
    image_ids: Iterable[str],
    hashes: dict[str, int],
    *,
    protocol_version: str,
    max_hamming: int = 8,
) -> tuple[dict[str, str], dict[str, Any]]:
    """簇约束备选划分（L2 对照补跑指令①：先应用 §3.4.1 近重复簇约束，再 §3.3 分桶）。

    规则（完全确定、可复现）：
    1. dHash 聚簇（Hamming ≤ max_hamming，冻结阈值 8）；
    2. 每簇整体同侧：簇的 split = 簇内**排序后首个 image_id**（代表）的 §3.3 哈希桶值；
    3. 非簇内单图按自身哈希桶正常分配（0-69 train / 70-84 val / 85-99 test）。

    返回 (assign, params)；params 完整记录生成参数（入 splits 文件 notes 与表 E）。
    """
    ids = sorted(image_ids)
    base = fallback_hash_split(ids, protocol_version)
    clusters = duplicate_clusters({i: hashes[i] for i in ids if i in hashes}, max_hamming)
    assign = dict(base)
    cluster_records = []
    for cluster in clusters:
        representative = cluster[0]  # duplicate_clusters 返回排序后列表
        target = base[representative]
        moved = [iid for iid in cluster if assign[iid] != target]
        for iid in cluster:
            assign[iid] = target
        cluster_records.append(
            {"representative": representative, "split": target, "size": len(cluster), "moved": len(moved)}
        )
    counts = {"train": 0, "val": 0, "test": 0}
    for s in assign.values():
        counts[s] += 1
    params = {
        "algorithm": "SHA256(protocol_version|image_id) mod 100; 0-69 train / 70-84 val / 85-99 test",
        "protocol_version": protocol_version,
        "salt": None,
        "cluster_constraint": f"dHash64 Hamming<={max_hamming} 聚簇；整簇随排序后首个 image_id 的桶值同侧（§3.4.1）",
        "dup_hamming_max": max_hamming,
        "clusters_total": len(clusters),
        "images_moved_by_cluster_rule": sum(c["moved"] for c in cluster_records),
        "split_counts": counts,
        "cluster_assignments": cluster_records,
    }
    return assign, params


@dataclass(frozen=True)
class SplitsFile:
    """splits.v1.json 的内存形态（每图 {image_id, category, block, split} + 生成参数 + 文件 SHA256）。"""

    entries: tuple[dict[str, str], ...]
    protocol_version: str
    method: str  # official_info_csv | fallback_hash
    generated_at_utc: str
    file_sha256: str | None = None  # 落盘后回填
    notes: tuple[str, ...] = ()

    def split_of(self, image_id: str) -> str | None:
        for e in self.entries:
            if e["image_id"] == image_id:
                return e["split"]
        return None

    def image_ids(self, split: str) -> list[str]:
        return [e["image_id"] for e in self.entries if e["split"] == split]

    def to_dict(self) -> dict[str, Any]:
        """序列化（**确定性**：不含时间戳——同内容跨重跑文件字节一致，sha256 稳定，
        冻结哈希才可审计/可比对；生成时刻记录于 summary.json 而非冻结文件内）。"""
        return {
            "schema": "game-ui-attention-splits/v1",
            "protocol_version": self.protocol_version,
            "method": self.method,
            "notes": list(self.notes),
            "images": [dict(e) for e in self.entries],
        }

    @classmethod
    def build(
        cls,
        metas: Sequence[ImageMeta],
        *,
        protocol_version: str,
        allow_fallback: bool = True,
        extra_notes: Sequence[str] = (),
    ) -> SplitsFile:
        """官方标志优先冻结（§3.2 不重新发明划分）；缺失/不明 → §3.3 备选（须 allow_fallback）。

        ``extra_notes`` 如实记录划分权威文件的实际来源结构（文件名/分隔符/列/行数），
        随 splits.v1.json 冻结（L2 勘误登记要求，2026-09-11）。
        """
        notes: list[str] = list(extra_notes)
        officials = {m.image_id: m.official_split for m in metas if m.official_split in ("train", "test")}
        n_missing = len(metas) - len(officials)
        if officials and n_missing == 0:
            method = "official_info_csv"
            assign = dict(officials)
        elif officials and allow_fallback:
            # 部分缺失：官方标志图沿用官方，缺失图按备选哈希补（记录混用原因）
            method = "official_info_csv+fallback_hash"
            assign = dict(officials)
            assign.update(
                {
                    k: v
                    for k, v in fallback_hash_split(
                        [m.image_id for m in metas if m.image_id not in officials], protocol_version
                    ).items()
                }
            )
            notes.append(f"info.csv 官方划分缺失 {n_missing} 图，缺失侧用 §3.3 确定性哈希补齐")
        elif allow_fallback:
            method = "fallback_hash"
            assign = fallback_hash_split([m.image_id for m in metas], protocol_version)
            notes.append("info.csv 无可用官方划分标志，整体启用 §3.3 备选自产划分")
        else:
            raise DatasetError("官方划分缺失且未允许备选划分", {"n_missing": n_missing})
        entries = tuple(
            {"image_id": m.image_id, "category": m.category, "block": m.block, "split": assign[m.image_id]}
            for m in sorted(metas, key=lambda x: (x.category, x.block, x.image_id))
        )
        return cls(
            entries=entries,
            protocol_version=protocol_version,
            method=method,
            generated_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            notes=tuple(notes),
        )

    @classmethod
    def build_fallback_cluster(
        cls,
        metas: Sequence[ImageMeta],
        hashes: dict[str, int],
        *,
        protocol_version: str,
        max_hamming: int = 8,
        extra_notes: Sequence[str] = (),
    ) -> SplitsFile:
        """备选划分 v2（L2 对照补跑指令①）：dHash 簇约束 + §3.3 确定性哈希分桶。

        生成参数（算法/盐/阈值/簇数/整簇同侧规则/split 计数/逐簇分配）全部写入 notes，
        随 splits.v2-fallback.json 冻结（修改必须递增版本号并重跑全部对比，§3.3.4）。
        """
        assign, params = fallback_cluster_constrained_split(
            [m.image_id for m in metas], hashes, protocol_version=protocol_version, max_hamming=max_hamming
        )
        params_json = json.dumps(
            {k: v for k, v in params.items() if k != "cluster_assignments"}, ensure_ascii=False, sort_keys=True
        )
        notes = [
            "备选划分 v2（对照补跑专用，与官方划分严格分表不混，§8.1 划分文件不同）",
            f"生成参数：{params_json}",
            f"簇分配明细：{json.dumps(params['cluster_assignments'], ensure_ascii=False, sort_keys=True)}",
            *extra_notes,
        ]
        entries = tuple(
            {"image_id": m.image_id, "category": m.category, "block": m.block, "split": assign[m.image_id]}
            for m in sorted(metas, key=lambda x: (x.category, x.block, x.image_id))
        )
        return cls(
            entries=entries,
            protocol_version=protocol_version,
            method="fallback_hash_cluster_constrained",
            generated_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            notes=tuple(notes),
        )

    def write(self, path: str | Path) -> str:
        """落盘并返回文件 SHA256（冻结：此后只读；修改必须递增版本号）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=False) + "\n"
        p.write_text(text, encoding="utf-8", newline="\n")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path, *, verify_sha256: str | None = None) -> SplitsFile:
        """读取已冻结划分；给定哈希时复核（泄漏审计清单项：划分文件哈希一致）。"""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if verify_sha256 is not None and actual != verify_sha256:
            raise DatasetError(
                "splits 文件哈希与记录不符（划分被改动 → 结果不可与旧表比较，须重跑全部对比）",
                {"expected": verify_sha256, "actual": actual},
            )
        obj = json.loads(text)
        return cls(
            entries=tuple(dict(e) for e in obj.get("images", ())),
            protocol_version=obj.get("protocol_version", ""),
            method=obj.get("method", ""),
            generated_at_utc=obj.get("generated_at_utc", ""),
            file_sha256=actual,
            notes=tuple(obj.get("notes", ())),
        )


# ---------------------------------------------------------------------------
# Part 3：Gazepoint 原始日志 → 注视点重建（复现官方管线，§2.1/§6.0）
# ---------------------------------------------------------------------------


def screen_to_image_point(bpogx: float, bpogy: float, img_w: int, img_h: int) -> tuple[float, float] | None:
    """官方坐标映射：屏幕归一化 (0-1) → 1920×1200 屏幕像素 → letterbox 逆变换 → 图像像素。

    letterbox：图像按宽高比缩放居中放置（scalar = min(1920/w, 1200/h)，pad 居中）。
    落在图像边界外 → None（调用方丢弃并计数，复现官方规则）。
    """
    sx = float(bpogx) * UEYES_SCREEN_WIDTH
    sy = float(bpogy) * UEYES_SCREEN_HEIGHT
    scalar = min(UEYES_SCREEN_WIDTH / img_w, UEYES_SCREEN_HEIGHT / img_h)
    pad_x = (UEYES_SCREEN_WIDTH - img_w * scalar) / 2.0
    pad_y = (UEYES_SCREEN_HEIGHT - img_h * scalar) / 2.0
    x = (sx - pad_x) / scalar
    y = (sy - pad_y) / scalar
    if not (0.0 <= x < img_w and 0.0 <= y < img_h):
        return None
    return (x, y)


@dataclass(frozen=True)
class LogFile:
    """eyetracker_logs/ 下参与者日志文件（{block}_{participant}_fixations.csv）。"""

    path: Path
    block: str
    participant: str


def discover_logs(logs_dir: str | Path) -> list[LogFile]:
    """发现全部参与者日志（文件名正则；容忍子目录）。"""
    out: list[LogFile] = []
    for p in sorted(Path(logs_dir).rglob("*_fixations.csv")):
        m = _LOG_NAME_RE.match(p.name)
        if m:
            out.append(LogFile(path=p, block=m.group("block"), participant=m.group("participant")))
        else:
            out.append(LogFile(path=p, block="", participant=p.stem))
    if not out:
        raise DatasetError(f"未在 {logs_dir} 发现任何 *_fixations.csv 日志")
    return out


_GAZEPOINT_COLUMNS = ("MEDIA_NAME", "TIME", "FPOGD", "BPOGV", "BPOGX", "BPOGY")


def _resolve_gazepoint_columns(fieldnames: Sequence[str]) -> dict[str, str]:
    """解析 Gazepoint CSV 列名。

    实际数据的 TIME 列名带会话起始时间戳后缀（如 ``TIME(2022/03/28 14:28:03.787)``，
    每个日志文件不同）：先精确匹配 ``TIME``，再匹配 ``TIME(…)`` 前缀；
    ``TIMETICK`` 不是时间秒列，显式排除。
    """
    lowered = {f.strip().upper(): f for f in fieldnames if f.strip()}
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for col in _GAZEPOINT_COLUMNS:
        if col in lowered:
            resolved[col] = lowered[col]
            continue
        if col == "TIME":
            match = next(
                (
                    orig
                    for upper, orig in lowered.items()
                    if upper.startswith("TIME(") and not upper.startswith("TIMETICK")
                ),
                None,
            )
            if match is not None:
                resolved[col] = match
                continue
        missing.append(col)
    if missing:
        raise DatasetError(
            f"Gazepoint CSV 缺少关键列 {missing}（列定义见 Gazepoint API v2.0 / 仓库 README）",
            {"fieldnames": list(fieldnames)},
        )
    return resolved


def iter_log_rows(path: str | Path, window_s: float) -> Iterator[dict[str, Any]]:
    """流式读取单参与者日志：BPOGV==1 有效注视、TIME ≤ 窗口秒数（前缀截断，§5.1）。"""
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        cols = _resolve_gazepoint_columns(reader.fieldnames or [])
        for row in reader:
            try:
                valid = float(row[cols["BPOGV"]])
                t = float(row[cols["TIME"]])
            except (TypeError, ValueError):
                continue  # 脏行跳过（计数在审计侧以 dropped 体现）
            if valid != 1.0 or t > window_s:
                continue
            try:
                yield {
                    "media": (row[cols["MEDIA_NAME"]] or "").strip(),
                    "x": float(row[cols["BPOGX"]]),
                    "y": float(row[cols["BPOGY"]]),
                    "dur": float(row[cols["FPOGD"]] or 0.0),
                    "time": t,
                }
            except (TypeError, ValueError):
                continue


@dataclass(frozen=True)
class GtCase:
    """单图单窗口的聚合真值（跨参与者合并；计数或时长加权）。"""

    image_id: str
    fixations: FixationSet  # 图像像素坐标（越界已丢弃）
    n_viewers: int
    n_dropped_oob: int
    n_fix_valid: int  # 窗口内有效注视总数（丢弃前）

    @property
    def n_fix(self) -> int:
        return len(self.fixations)


def build_groundtruth(
    layout: DatasetLayout,
    metas: Sequence[ImageMeta],
    window: str,
    *,
    weighting: str = "count",
    progress_every: int = 0,
    progress_fn: Any = None,
) -> dict[str, GtCase]:
    """按窗口重建全部图像真值（窗口绝不混合：调用方逐窗口独立调用，§5.2）。"""
    if window not in WINDOW_SECONDS:
        raise DatasetError(f"未知窗口 {window!r}（合法：{sorted(WINDOW_SECONDS)}）")
    if weighting not in ("count", "duration"):
        raise DatasetError(f"weighting 必须是 count|duration，得到 {weighting!r}")
    window_s = WINDOW_SECONDS[window]

    # MEDIA_NAME 索引：全名/去扩展名/小写，三键兼容
    meta_index: dict[str, ImageMeta] = {}
    for m in metas:
        meta_index[m.image_id] = m
        meta_index[Path(m.image_id).stem] = m
        meta_index[m.image_id.lower()] = m
        meta_index[Path(m.image_id).stem.lower()] = m

    points: dict[str, list[list[float]]] = {}
    weights: dict[str, list[float]] = {}
    viewers: dict[str, set[str]] = {}
    dropped: dict[str, int] = {}
    valid_count: dict[str, int] = {}

    logs = discover_logs(layout.logs_dir)
    for i, log in enumerate(logs):
        if progress_every and progress_fn and i % progress_every == 0:
            progress_fn(f"真值重建 {window}：{i}/{len(logs)} 日志（{log.path.name}）")
        for row in iter_log_rows(log.path, window_s):
            meta = (
                meta_index.get(row["media"])
                or meta_index.get(Path(row["media"]).stem)
                or meta_index.get(row["media"].lower())
                or meta_index.get(Path(row["media"]).stem.lower())
            )
            if meta is None:
                continue  # 日志中出现的未知媒体名：不计入（审计侧可对照 info.csv 覆盖率）
            iid = meta.image_id
            valid_count[iid] = valid_count.get(iid, 0) + 1
            viewers.setdefault(iid, set()).add(log.participant)
            pt = screen_to_image_point(row["x"], row["y"], meta.width, meta.height)
            if pt is None:
                dropped[iid] = dropped.get(iid, 0) + 1
                continue
            points.setdefault(iid, []).append([pt[0], pt[1]])
            weights.setdefault(iid, []).append(row["dur"] if weighting == "duration" else 1.0)

    out: dict[str, GtCase] = {}
    for m in metas:
        iid = m.image_id
        pts = points.get(iid, [])
        ws = weights.get(iid, [])
        fix = build_fixation_set(
            np.asarray(pts, dtype=np.float64).reshape(-1, 2),
            (m.height, m.width),
            weights=np.asarray(ws, dtype=np.float64) if ws else None,
        )
        out[iid] = GtCase(
            image_id=iid,
            fixations=fix,
            n_viewers=len(viewers.get(iid, ())),
            n_dropped_oob=dropped.get(iid, 0),
            n_fix_valid=valid_count.get(iid, 0),
        )
    return out


# ---------------------------------------------------------------------------
# Part 4：近重复感知哈希与泄漏审计（§3.4）
# ---------------------------------------------------------------------------


def dhash64(image: np.ndarray) -> int:
    """64-bit 差值哈希（dHash）：9×8 灰度 → 水平梯度位。用于近重复聚簇（非安全用途）。"""
    from PIL import Image

    arr = np.asarray(image)
    if arr.ndim == 3:
        gray = arr[:, :, :3].mean(axis=2).astype(np.uint8)
    else:
        gray = arr.astype(np.uint8)
    small = np.asarray(Image.fromarray(gray).resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if small[row, col] < small[row, col + 1] else 0)
    return bits


def hamming64(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def duplicate_clusters(hashes: dict[str, int], max_hamming: int = 8) -> list[list[str]]:
    """近重复聚簇（union-find）：任意两图 Hamming ≤ max_hamming（冻结默认 8）归同簇。"""
    ids = sorted(hashes)
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            if hamming64(hashes[ids[i]], hashes[ids[j]]) <= max_hamming:
                union(ids[i], ids[j])
    clusters: dict[str, list[str]] = {}
    for iid in ids:
        clusters.setdefault(find(iid), []).append(iid)
    return [sorted(c) for c in clusters.values() if len(c) > 1]


@dataclass(frozen=True)
class LeakageAudit:
    """泄漏审计清单结果（§3.4.5，每次评估运行必查；进表 D）。"""

    splits_sha256: str
    splits_sha256_ok: bool
    clusters_total: int
    clusters_cross_split: tuple[tuple[str, ...], ...] = ()
    cb_source_split: str = ""
    cb_only_train: bool = True
    cb_built_before_test_metrics: bool | None = None
    excluded_train: tuple[str, ...] = ()
    excluded_test: tuple[str, ...] = ()
    known_limitation: str = "来源级泄漏无法完全排除（UEyes 官方未提供逐图来源字段；以 block+感知哈希簇为近似）"

    @property
    def cross_split_violations(self) -> int:
        return len(self.clusters_cross_split)

    def to_dict(self) -> dict[str, Any]:
        return {
            "splits_sha256": self.splits_sha256,
            "splits_sha256_ok": self.splits_sha256_ok,
            "clusters_total": self.clusters_total,
            "clusters_cross_split": [list(c) for c in self.clusters_cross_split],
            "cross_split_violations": self.cross_split_violations,
            "cb_source_split": self.cb_source_split,
            "cb_only_train": self.cb_only_train,
            "cb_built_before_test_metrics": self.cb_built_before_test_metrics,
            "excluded_train": list(self.excluded_train),
            "excluded_test": list(self.excluded_test),
            "known_limitation": self.known_limitation,
        }


def run_leakage_audit(
    *,
    splits: SplitsFile,
    recorded_splits_sha256: str | None,
    hashes: dict[str, int],
    max_hamming: int,
    cb: CenterBiasBaseline | None,
    cb_built_at_utc: str | None = None,
    test_metrics_started_utc: str | None = None,
    excluded_train: Sequence[str] = (),
    excluded_test: Sequence[str] = (),
) -> LeakageAudit:
    """执行审计清单五查：哈希一致 / 簇不跨划分 / CB 仅 train / CB 时序 / 排除清单分侧。

    官方划分冻结不改（§3.2）：近重复簇跨划分违规如实记录（表 D 显著标注 + 完成报），
    不静默重排划分。
    """
    sha_ok = recorded_splits_sha256 is None or splits.file_sha256 == recorded_splits_sha256
    clusters = duplicate_clusters(hashes, max_hamming)
    cross: list[tuple[str, ...]] = []
    for cluster in clusters:
        splits_seen = {splits.split_of(iid) for iid in cluster}
        if len(splits_seen - {None}) > 1:
            cross.append(tuple(cluster))
    cb_only_train = cb is None or cb.source_split == "train"
    cb_before: bool | None = None
    if cb is not None and cb_built_at_utc and test_metrics_started_utc:
        cb_before = cb_built_at_utc <= test_metrics_started_utc
    return LeakageAudit(
        splits_sha256=splits.file_sha256 or "",
        splits_sha256_ok=sha_ok,
        clusters_total=len(clusters),
        clusters_cross_split=tuple(cross),
        cb_source_split=cb.source_split if cb is not None else "",
        cb_only_train=cb_only_train,
        cb_built_before_test_metrics=cb_before,
        excluded_train=tuple(excluded_train),
        excluded_test=tuple(excluded_test),
    )


# ---------------------------------------------------------------------------
# Part 5：评估编排（CB 构造 / 推理接入 / 逐图运行 / 聚合 / 表 A-E / S0-S3 判定）
# ---------------------------------------------------------------------------

Predictor = Any  # Callable[[ImageMeta, np.ndarray], tuple[np.ndarray, dict]] | None


def build_cb_for_window(
    gt: dict[str, GtCase],
    metas: Sequence[ImageMeta],
    splits: SplitsFile,
    window: str,
    config: EvalConfig,
    out_dir: Path,
    *,
    dataset_name: str = "ueyes",
    version_tag: str = "v1",
) -> tuple[CenterBiasBaseline, str]:
    """CB train-only 构造（§4.2 + 反泄漏 §3.4.3）；npz 版本化落盘，返回 (baseline, 文件sha256)。

    ``version_tag`` 区分划分版本（官方划分 v1 / 备选划分对照 v2-fallback），
    新版本 = 新文件 + 新哈希，不与旧版混表（§8.1）。
    """
    train_ids = set(splits.image_ids("train"))
    meta_by_id = {m.image_id: m for m in metas}
    inputs: list[CenterBiasFitInput] = []
    for iid in sorted(train_ids):
        case = gt.get(iid)
        meta = meta_by_id.get(iid)
        if case is None or meta is None or case.n_fix == 0:
            continue
        inputs.append(CenterBiasFitInput(shape=(meta.height, meta.width), fixations=case.fixations))
    if not inputs:
        raise EvalGateError(f"窗口 {window}：train 划分无有效注视，无法构造 CB（禁止用 test 数据兜底）")
    cb = CenterBiasBaseline.fit(
        inputs,
        bins=config.cb_bins,
        sigma_bin=config.cb_sigma_bin,
        weighting=config.weighting,
        source_split="train",
        source_split_hash=splits.file_sha256 or "",
        version=f"cb.{version_tag}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cb_{dataset_name}_{window}_train.{version_tag}.npz"
    sha = cb.save(path)
    return cb, sha


def make_registry_predictor(profile_name: str) -> Any:
    """经 backends/registry（C2）构造预测函数：图像 → 原图分辨率 float64 概率图。

    推理与窗口无关（§5.7）：一次推理输出可被三个窗口分别评估。
    registry/profile/权重缺失 → 结构化错误（退出码 3 语义），由调用方决定跳过或失败。
    """
    import importlib

    from ..probability import log_density_to_probability, resample_to_original, validate_probability

    try:
        registry = importlib.import_module("ui_attention.backends.registry")
    except ImportError as exc:
        raise UiAttentionError(
            ErrorCode.MODEL_NOT_READY,
            f"后端 registry 不可用，profile {profile_name!r} 无法解析（模型侧未就绪）",
            {"profile": profile_name, "reason": str(exc)},
        ) from exc
    profile = registry.resolve_profile(profile_name)
    backend = registry.get_backend(profile)
    info = backend.describe()

    def predict(image_array: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        result = backend.predict(image_array, profile)
        result.validate()
        if "spatial_density" not in tuple(info.capabilities):
            raise EvalGateError(f"后端 {info.backend_id} 未声明 spatial_density，不可进入评估管线")
        if result.semantics == "log_density":
            p_inf = log_density_to_probability(result.array)
        elif result.semantics == "probability_density":
            p_inf = np.asarray(result.array, dtype=np.float64)
            validate_probability(p_inf)
        else:
            raise EvalGateError(f"语义 {result.semantics!r} 不进入公开数据评估管线（厂商分数不反推像素概率）")
        density = resample_to_original(p_inf, result.shape_mapping)
        meta = {
            "profile": profile.profile_name,
            "backend_id": info.backend_id,
            "backend_version": info.version,
            "weights_sha256": [w.sha256 for w in info.weights],
            "runtime": dict(result.runtime),
            "semantics": result.semantics,
            # 表 E 如实标注：centerbias 引用（如 DeepGaze MIT1003）与观看条件实验假设
            "centerbias": dict(profile.centerbias) if profile.centerbias is not None else None,
            "viewing_conditions": dict(profile.viewing_conditions),
            "config_hash": profile.config_hash,
            "limitations": list(result.limitations),
        }
        return density, meta

    return predict


def _image_case(meta: ImageMeta, gt: GtCase, split: str) -> Any:
    from .pipeline import ImageCase

    return ImageCase(
        image_id=meta.image_id,
        shape=(meta.height, meta.width),
        fixations=gt.fixations,
        category=meta.category,
        block=meta.block,
        n_viewers=gt.n_viewers,
        split=split,
    )


def _rows_for_prediction(
    *,
    dataset_name: str,
    window: str,
    model: str,
    model_version: str,
    S: np.ndarray,
    case: Any,  # pipeline.ImageCase
    meta: ImageMeta,
    gt_case: GtCase,
    cb: CenterBiasBaseline | None,
    neg: FixationSet,
    config: EvalConfig,
    split: str = "test",
    F_precomputed: np.ndarray | None = None,
) -> list[Any]:
    """单图 × 单模型 × 单窗口的全部指标行（流式编排与批量编排共用同一实现）。

    ``F_precomputed``：模糊真值图与模型无关（每 图×窗口 只需计算一次），
    流式路径传入缓存值避免对每模型重复高斯模糊。
    """
    from .baselines import uniform_baseline
    from .groundtruth import blurred_truth
    from .metrics import MetricResult, auc_judd, cc, information_gain, kl_divergence, nss, sauc, similarity
    from .pipeline import PerImageRow, _validate_prediction, exclusion_flag

    S = _validate_prediction(S, case.image_id)
    if S.shape != (meta.height, meta.width):
        raise EvalGateError(f"预测图与评估网格不一致：{case.image_id} {S.shape} vs {(meta.height, meta.width)}")
    flag = exclusion_flag(case, min_fixations=config.min_fixations, min_viewers=config.min_viewers)
    F = F_precomputed if F_precomputed is not None else blurred_truth(case.fixations, case.shape, config)[0]
    U = uniform_baseline(case.shape)
    results: list[MetricResult] = []
    if cb is not None:
        r = information_gain(S, cb.evaluate(case.shape), case.fixations, config)
        results.append(MetricResult("IG_CB", r.value, r.flags))
    r = information_gain(S, U, case.fixations, config)
    results.append(MetricResult("IG_U", r.value, r.flags))
    results.append(nss(S, case.fixations, config))
    results.append(cc(F, S, config))
    results.append(sauc(S, case.fixations, neg, config))
    results.append(auc_judd(S, case.fixations, config))
    results.append(kl_divergence(F, S, config))
    if config.include_sim:
        results.append(similarity(F, S, config))
    rows: list[Any] = []
    for res in results:
        rows.append(
            PerImageRow(
                dataset=dataset_name,
                split=split,
                window=window,
                model=model,
                model_version=model_version,
                image_id=case.image_id,
                category=meta.category,
                block=meta.block,
                n_viewers=gt_case.n_viewers,
                n_fix=gt_case.n_fix,
                excluded_flag=flag or ("empty_samples" if res.excluded else ""),
                metric=res.name,
                value=float(res.value),
            )
        )
    return rows


def evaluate_model_rows(
    *,
    dataset_name: str,
    window: str,
    model: str,
    model_version: str,
    predictions: dict[str, np.ndarray],
    gt: dict[str, GtCase],
    metas: Sequence[ImageMeta],
    splits: SplitsFile,
    cb: CenterBiasBaseline | None,
    config: EvalConfig,
    split: str = "test",
) -> list[Any]:
    """单模型 × 单窗口 × split 内全部图像的逐图行（批量入口；评估网格=原图分辨率；窗口绝不混合）。"""
    from .baselines import pooled_other_fixations

    meta_by_id = {m.image_id: m for m in metas}
    ids = [iid for iid in splits.image_ids(split) if iid in predictions]
    pool = [(iid, gt[iid].fixations) for iid in ids if iid in gt]
    rows: list[Any] = []
    for iid in ids:
        meta = meta_by_id[iid]
        case = _image_case(meta, gt[iid], split)
        rows += _rows_for_prediction(
            dataset_name=dataset_name,
            window=window,
            model=model,
            model_version=model_version,
            S=predictions[iid],
            case=case,
            meta=meta,
            gt_case=gt[iid],
            cb=cb,
            neg=pooled_other_fixations(pool, iid),
            config=config,
            split=split,
        )
    return rows


# ---------------------------------------------------------------------------
# Part 6：聚合（bootstrap CI / 配对差值 / Holm）与表 A-E、S0-S3 判定
# ---------------------------------------------------------------------------


def _pivot_rows(rows: Sequence[Any]) -> dict[tuple[str, str, str, str], dict[str, float]]:
    """(model, window, metric, split) → {image_id: value}（只收未排除且有限的值）。"""
    out: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for r in rows:
        if r.excluded_flag or not np.isfinite(r.value):
            continue
        out.setdefault((r.model, r.window, r.metric, r.split), {})[r.image_id] = r.value
    return out


def _categories_of(rows: Sequence[Any], image_ids: Sequence[str]) -> list[str]:
    cat_by_id = {r.image_id: r.category for r in rows}
    return [cat_by_id.get(iid, "unknown") for iid in image_ids]


def aggregate_table_a(
    rows: Sequence[Any], config: EvalConfig, split: str = "test"
) -> tuple[list[dict[str, Any]], dict]:
    """表 A：模型 × 划分 × 窗口 × 指标，单元格 = mean [95% CI]（图像级分层 bootstrap）。"""
    from .bootstrap import image_bootstrap_ci

    pivot = _pivot_rows(rows)
    cells: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    used: dict[tuple[str, str, str], dict[str, float]] = {}
    for (model, window, metric, ssplit), values in pivot.items():
        if ssplit != split:
            continue
        ids = sorted(values)
        ci = image_bootstrap_ci(
            [values[i] for i in ids],
            categories=_categories_of(rows, ids) if config.bootstrap_stratify_by_category else None,
            b=config.bootstrap_b,
            seed=config.bootstrap_seed,
            ci=config.ci_level,
        )
        cells.setdefault((model, window), {})[metric] = {
            "mean": ci["mean"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "n_images": ci["n_images"],
        }
        used[(model, window, metric)] = values
    table_rows = [
        {"model": model, "split": split, "window": window, "metrics": metrics}
        for (model, window), metrics in sorted(cells.items())
    ]
    return table_rows, used


def aggregate_table_b(
    rows: Sequence[Any],
    config: EvalConfig,
    pairs: Sequence[tuple[str, str, tuple[str, ...]]],
    split: str = "test",
) -> list[dict[str, Any]]:
    """表 B：配对差值（candidate−baseline 同图配对）+ 差值 CI + 胜率 + Holm 校正。

    pairs: (model, baseline, metrics) 三元组序列；Holm 校正范围 = 本表全部 (model×metric×window) 检验，
    范围在报告注明（§7.2）。两模型 CI 各自不重叠 ≠ 差值显著——判定只看配对差值 CI。
    """
    from .bootstrap import holm_correction, paired_bootstrap

    pivot = _pivot_rows(rows)
    entries: list[dict[str, Any]] = []
    for model, baseline, metrics in pairs:
        for window in config.windows:
            for metric in metrics:
                a = pivot.get((model, window, metric, split), {})
                b = pivot.get((baseline, window, metric, split), {})
                common = sorted(set(a) & set(b))
                if len(common) < 2:
                    continue
                cats = _categories_of(rows, common) if config.bootstrap_stratify_by_category else None
                res = paired_bootstrap(
                    [a[i] for i in common],
                    [b[i] for i in common],
                    categories=cats,
                    b=config.bootstrap_b,
                    seed=config.bootstrap_seed,
                    ci=config.ci_level,
                )
                entries.append(
                    {
                        "model": model,
                        "baseline": baseline,
                        "window": window,
                        "metric": metric,
                        "mean_diff": res["mean_diff"],
                        "ci_low": res["ci_low"],
                        "ci_high": res["ci_high"],
                        "win_rate": res["win_rate"],
                        "n_paired": res["n_paired"],
                        "p_value": res["p_value"],
                    }
                )
    if entries:
        holm = holm_correction([e["p_value"] for e in entries], alpha=0.05)
        for e, adj, sig in zip(entries, holm["adjusted"], holm["significant"], strict=True):
            e["p_adjusted"] = adj
            e["holm_significant"] = sig
        holm_scope = f"Holm 校正范围：本表全部 {len(entries)} 项 (model×window×metric) 检验，alpha=0.05"
        for e in entries:
            e["holm_scope"] = holm_scope
    return entries


def aggregate_table_c(rows: Sequence[Any], config: EvalConfig, split: str = "test") -> list[dict[str, Any]]:
    """表 C：category × model × window × 指标（分层口径：类别内部重采样）。"""
    from .bootstrap import image_bootstrap_ci

    pivot = _pivot_rows(rows)
    cat_by_id = {r.image_id: r.category for r in rows}
    cells: dict[tuple[str, str, str], dict[str, dict[str, float]]] = {}
    for (model, window, metric, ssplit), values in pivot.items():
        if ssplit != split:
            continue
        by_cat: dict[str, list[float]] = {}
        for iid, v in values.items():
            by_cat.setdefault(cat_by_id.get(iid, "unknown"), []).append(v)
        for cat, vals in by_cat.items():
            if len(vals) < 2:  # 单图类别无抽样变异可言：均值 + 退化标注
                cells.setdefault((cat, model, window), {})[metric] = {
                    "mean": float(np.mean(vals)),
                    "ci_low": None,
                    "ci_high": None,
                    "n_images": len(vals),
                }
                continue
            ci = image_bootstrap_ci(vals, b=config.bootstrap_b, seed=config.bootstrap_seed, ci=config.ci_level)
            cells.setdefault((cat, model, window), {})[metric] = {
                "mean": ci["mean"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "n_images": ci["n_images"],
            }
    return [
        {"category": cat, "model": model, "window": window, "metrics": metrics}
        for (cat, model, window), metrics in sorted(cells.items())
    ]


def build_audit_rows(
    *,
    gt_by_window: dict[str, dict[str, GtCase]],
    splits: SplitsFile,
    audit: LeakageAudit,
    config: EvalConfig,
    zip_md5_ok: bool | None,
) -> list[dict[str, Any]]:
    """表 D：每 (split × window) 图像数/排除数/参与者覆盖/注视点/重复簇/md5。"""
    rows: list[dict[str, Any]] = []
    for window, gt in sorted(gt_by_window.items()):
        for split in ("train", "test"):
            ids = splits.image_ids(split)
            cases = [gt[i] for i in ids if i in gt]
            n_low_fix = sum(1 for c in cases if c.n_fix < config.min_fixations)
            n_low_view = sum(1 for c in cases if c.n_viewers and c.n_viewers < config.min_viewers)
            viewers = sorted(c.n_viewers for c in cases if c.n_viewers)
            rows.append(
                {
                    "split": split,
                    "window": window,
                    "n_images": len(cases),
                    "excluded_low_fixation": n_low_fix,
                    "excluded_low_viewers": n_low_view,
                    "excluded_empty_samples": sum(1 for c in cases if c.n_fix == 0),
                    "viewers_min": viewers[0] if viewers else None,
                    "viewers_median": float(np.median(viewers)) if viewers else None,
                    "viewers_max": viewers[-1] if viewers else None,
                    "n_fixations_total": sum(c.n_fix for c in cases),
                    "n_dropped_oob_total": sum(c.n_dropped_oob for c in cases),
                    "dup_clusters": audit.clusters_total,
                    "dup_clusters_cross_split": audit.cross_split_violations,
                    "md5_ok": zip_md5_ok,
                }
            )
    return rows


def _git_commit() -> str:
    """评估代码版本（表 E）；只读 git 查询，失败 → unknown（不阻塞评估）。"""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def build_fingerprint(
    *,
    config: EvalConfig,
    splits_sha256: str,
    cb_versions: dict[str, str],
    models: list[dict[str, Any]],
    dataset_name: str,
    dataset_md5: str | None,
) -> dict[str, Any]:
    """表 E 配置指纹：任一项不同 → 结果不可与旧表同表比较（§8.1）。"""
    return {
        "protocol_version": config.protocol_version,
        "dataset": {"name": dataset_name, "zip_md5": dataset_md5},
        "splits_file_hash": splits_sha256,
        "models": models,
        "preprocessing": {
            "interp": "bilinear_resample_to_original_then_renormalize",
            "sigma_blur": config.sigma_rule,
            "blur_mode": config.blur_mode,
            "weighting": config.weighting,
        },
        "numeric_guards": {"lambda": config.lam, "eps_rel": config.eps_rel},
        "log_base": config.log_base,
        "baseline_versions": cb_versions,
        "eval_code_version": _git_commit(),
        "bootstrap": {
            "B": config.bootstrap_b,
            "seed": config.bootstrap_seed,
            "stratified": config.bootstrap_stratify_by_category,
            "ci": config.ci_level,
        },
        "frozen_thresholds": {
            "dup_hamming_max": config.dup_hamming_max,
            "min_fixations": config.min_fixations,
            "min_viewers": config.min_viewers,
        },
        "config_hash": config.config_hash(),
        "l1_approval": config.l1_approval,
    }


def judge_criteria(
    *,
    config: EvalConfig,
    table_b: Sequence[dict[str, Any]],
    candidates: Sequence[str],
    s0_report: dict[str, Any],
) -> dict[str, Any]:
    """S0-S3 判定（S4 本轮不设，L1 批准口径）。

    S2 为后端选型硬门槛：任一窗口 IG_CB/NSS 配对差值（vs CB）CI 下界 ≤ 0 →
    该后端只能表述为"在公开 UI 数据上未显著优于中心偏置基线"（§8.3 失败动作）。
    """
    crit = config.criteria
    verdicts: dict[str, Any] = {}
    verdicts["S0"] = {
        "status": "skipped" if s0_report.get("skipped") else ("pass" if not s0_report.get("failed") else "fail"),
        "detail": "管线自检（均匀基线 NSS≈0/AUC≈0.5/IG_U≈0；CB IG_CB≈0）",
    }

    def _look(model: str, baseline: str, window: str, metric: str) -> dict[str, Any] | None:
        for e in table_b:
            if e["model"] == model and e["baseline"] == baseline and e["window"] == window and e["metric"] == metric:
                return e
        return None

    if crit.s1_enabled:
        detail = {}
        ok_all = True
        for window in config.windows:
            for metric in crit.s1_metrics:
                e = _look("center_bias", "uniform", window, metric)
                if e is None:
                    ok_all = False
                    detail[f"{window}/{metric}"] = "missing"
                    continue
                ok = e["ci_low"] > crit.s1_ci_lower_min
                ok_all &= ok
                detail[f"{window}/{metric}"] = {"ci_low": e["ci_low"], "pass": ok}
        verdicts["S1"] = {
            "status": "pass" if ok_all else "fail",
            "detail": detail,
            "note": "CB 显著优于均匀基线（配对差值 CI 下界>0）",
        }

    if crit.s2_enabled:
        per_model: dict[str, Any] = {}
        for cand in candidates:
            detail = {}
            ok_all = True
            for window in config.windows:
                for metric in crit.s2_metrics:
                    e = _look(cand, "center_bias", window, metric)
                    if e is None:
                        ok_all = False
                        detail[f"{window}/{metric}"] = "missing"
                        continue
                    ok = e["ci_low"] > crit.s2_ci_lower_min
                    ok_all &= ok
                    detail[f"{window}/{metric}"] = {
                        "mean_diff": e["mean_diff"],
                        "ci_low": e["ci_low"],
                        "win_rate": e["win_rate"],
                        "pass": ok,
                    }
            per_model[cand] = {
                "status": "pass" if ok_all else "fail",
                "detail": detail,
                "failure_wording": None
                if ok_all
                else "在公开 UI 数据上未显著优于中心偏置基线（不得宣称可用于游戏 UI 评审）",
            }
        verdicts["S2"] = {"hard_gate": True, "models": per_model}

    if crit.s3_enabled:
        verdicts["S3"] = {
            "status": "not_applicable",
            "note": (
                "FiWI 不纳入（许可未核实，L1 批准 2026-09-11）：缺跨数据集对照面，定性判据本轮不可执行，不判 pass/fail"
            ),
        }
    verdicts["S4"] = {"status": "disabled", "note": "本轮不设（UMSI++ 不跑；作者报告值仅进独立文献参考表，不混表）"}
    return verdicts


# ---------------------------------------------------------------------------
# Part 7：全流程编排与 CLI 入口
# ---------------------------------------------------------------------------


@dataclass
class EvaluationOutputs:
    """一次完整评估运行的产物索引（全部在 out_root 内；可公开摘要另由完成报承载）。"""

    out_root: Path
    splits_path: Path
    splits_sha256: str
    cb_paths: dict[str, str] = field(default_factory=dict)  # window → npz 路径
    cb_hashes: dict[str, str] = field(default_factory=dict)  # window → sha256
    csv_paths: dict[str, str] = field(default_factory=dict)  # window → per-image CSV
    tables: dict[str, Any] = field(default_factory=dict)  # A-E dict（含 markdown）
    audit: dict[str, Any] = field(default_factory=dict)
    verdicts: dict[str, Any] = field(default_factory=dict)
    fingerprint: dict[str, Any] = field(default_factory=dict)
    model_meta: dict[str, Any] = field(default_factory=dict)
    skipped_profiles: list[str] = field(default_factory=list)


def run_full_evaluation(
    *,
    dataset_root: str | Path,
    out_root: str | Path,
    config: EvalConfig | None = None,
    profiles: Sequence[str] = (),
    predictor_factory: Any = None,
    zip_path: str | Path | None = None,
    dataset_name: str = "ueyes",
    windows: Sequence[str] | None = None,
    progress_fn: Any = None,
    splits_mode: str = "official",
    splits_filename: str = SPLITS_VERSION,
    cb_version_tag: str = "v1",
    dataset_md5_verified: str | None = None,
    variant_label: str = "",
    verdicts_descriptive: bool = False,
) -> EvaluationOutputs:
    """R2 协议全流程（附录 A 伪代码的可运行编排）。

    步骤：布局发现 →（可选 zip md5 红线校验）→ 划分权威文件 → 图像路径/尺寸 →
    dHash 近重复聚簇 → 划分冻结（official=官方标志 / fallback_cluster=§3.3 哈希分桶
    +§3.4.1 簇约束，L2 对照补跑指令）→ 逐窗口真值重建 → CB train-only 构造 →
    S0 自检门 → **流式**推理+指标（图外层/窗口中层/模型内层：每图只加载与推理一次、
    density 跨窗口复用后即弃、模糊真值每图×窗口一次，内存 O(单图×模型数)）→
    逐图 CSV（每窗口落盘）→ bootstrap/配对/Holm → 表 A-E → S0-S3 判定 → summary。
    失败一律结构化抛出。

    对照运行参数（L2 指令④⑥⑦）：``variant_label`` 标注全部表（与官方划分严格分表
    不混，§8.1 划分文件不同）；``verdicts_descriptive=True`` 时 S1/S2 判定仅作描述性
    记录；``dataset_md5_verified`` 注入主运行已验证的同一 zip md5（不重复计算，来源注明）。

    predictor_factory: ``profile_name -> predict(image_array) -> (density, meta)``；
    缺省用 :func:`make_registry_predictor`（经 C2 registry）。未登记 profile → 记录跳过。
    """
    from .baselines import pooled_other_fixations, uniform_baseline
    from .groundtruth import blurred_truth
    from .pipeline import s0_self_check, write_per_image_csv
    from .tables import build_table_a, build_table_b, build_table_c, build_table_d, build_table_e

    config = config or EvalConfig()
    config.validate()
    windows = tuple(windows or config.windows)
    if splits_mode not in ("official", "fallback_cluster"):
        raise DatasetError(f"未知 splits_mode {splits_mode!r}（official|fallback_cluster）")
    prog = progress_fn or (lambda msg: None)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    outputs = EvaluationOutputs(out_root=out, splits_path=out / splits_filename, splits_sha256="")

    dataset_md5 = None
    md5_source = ""
    if zip_path is not None:
        prog("校验 zip md5（红线：不符 → 结构化失败）…")
        dataset_md5 = verify_zip_md5(zip_path)["actual"]
        md5_source = "verified_this_run"
    elif dataset_md5_verified is not None:
        dataset_md5 = dataset_md5_verified
        md5_source = "verified_in_official_run(same_zip)"

    layout = discover_layout(dataset_root)
    prog(f"解析划分权威文件 {layout.info_csv.name} 并发现图像路径…")
    info_table = load_info_table(layout.info_csv)
    metas = discover_image_paths(layout, list(info_table.metas))
    base_note = (
        f"划分权威文件：{info_table.source_file}（分隔符 {info_table.delimiter!r}；"
        f"列 {list(info_table.columns)}；{info_table.n_rows} 行；官方划分计数 {info_table.split_counts}）"
    )
    if info_table.source_file == "image_types.csv":
        split_note = base_note + (
            "。勘误：R2 协议 §3.2 所写 info.csv 在实际 zip 内不存在，权威文件为 image_types.csv"
            "（L2 于 2026-09-11 独立复核并登记勘误）"
        )
    else:
        split_note = base_note

    prog("dHash 近重复聚簇（1980 图）…")
    hashes: dict[str, int] = {}
    mode_audit: dict[str, Any] = {"modes": {}, "n_alpha": 0, "n_non_opaque_alpha": 0, "non_opaque_alpha_ids": []}
    for m in metas:
        arr, minfo = load_eval_image(layout.root / m.rel_path)
        hashes[m.image_id] = dhash64(arr)
        mode_audit["modes"][minfo["mode"]] = mode_audit["modes"].get(minfo["mode"], 0) + 1
        if minfo["has_alpha"]:
            mode_audit["n_alpha"] += 1
            if minfo.get("non_opaque_alpha"):
                mode_audit["n_non_opaque_alpha"] += 1
                mode_audit["non_opaque_alpha_ids"].append(m.image_id)

    if splits_mode == "official":
        prog(f"冻结官方划分 → {splits_filename}…")
        splits = SplitsFile.build(metas, protocol_version=config.protocol_version, extra_notes=[split_note])
    else:
        prog(f"构造备选划分（dHash 簇约束 + §3.3 哈希分桶）→ {splits_filename}…")
        splits = SplitsFile.build_fallback_cluster(
            metas,
            hashes,
            protocol_version=config.protocol_version,
            max_hamming=config.dup_hamming_max,
            extra_notes=[split_note],
        )
    outputs.splits_sha256 = splits.write(outputs.splits_path)
    splits = SplitsFile.load(outputs.splits_path, verify_sha256=outputs.splits_sha256)

    test_metrics_started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    gt_by_window: dict[str, dict[str, GtCase]] = {}
    cbs: dict[str, CenterBiasBaseline] = {}
    for window in windows:
        prog(f"重建真值（窗口 {window}，计数/时长口径={config.weighting}）…")
        gt = build_groundtruth(
            layout, metas, window, weighting=config.weighting, progress_every=20, progress_fn=progress_fn
        )
        gt_by_window[window] = gt
        prog(f"构造 CB 基线（train-only，窗口 {window}，版本 {cb_version_tag}）…")
        cb, cb_sha = build_cb_for_window(
            gt, metas, splits, window, config, out, dataset_name=dataset_name, version_tag=cb_version_tag
        )
        cbs[window] = cb
        outputs.cb_paths[window] = str(out / f"cb_{dataset_name}_{window}_train.{cb_version_tag}.npz")
        outputs.cb_hashes[window] = cb_sha

    excluded = {
        w: {
            s: sorted(
                m.image_id
                for m in metas
                if splits.split_of(m.image_id) == s
                and (
                    gt_by_window[w][m.image_id].n_fix < config.min_fixations
                    or (
                        gt_by_window[w][m.image_id].n_viewers
                        and gt_by_window[w][m.image_id].n_viewers < config.min_viewers
                    )
                )
            )
            for s in ("train", "test")
        }
        for w in windows
    }
    audit = run_leakage_audit(
        splits=splits,
        recorded_splits_sha256=outputs.splits_sha256,
        hashes=hashes,
        max_hamming=config.dup_hamming_max,
        cb=cbs[windows[0]] if cbs else None,
        cb_built_at_utc=test_metrics_started,  # CB 在同一次运行内先于 test 指标构造
        test_metrics_started_utc=test_metrics_started,
        excluded_train=excluded[windows[0]]["train"],
        excluded_test=excluded[windows[0]]["test"],
    )
    outputs.audit = audit.to_dict()
    outputs.audit["image_mode_audit"] = mode_audit  # 图像模式与 alpha 事实（load_eval_image 确定性规则）

    prog("S0 管线自检门…")
    s0_report = s0_self_check(config)  # 不过 → EvalGateError（禁止出报告）

    # 预测器创建（未登记 profile → 结构化跳过，不崩溃）
    factory = predictor_factory or make_registry_predictor
    predictors: dict[str, Any] = {}
    model_meta: dict[str, dict[str, Any]] = {}
    for profile_name in profiles:
        try:
            predictors[profile_name] = factory(profile_name)
        except UiAttentionError as exc:
            prog(f"profile {profile_name} 不可用（{exc.code.value}）：记录跳过，待登记后重跑")
            outputs.skipped_profiles.append(profile_name)
            outputs.model_meta.setdefault("skipped", {})[profile_name] = {"reason": exc.message, "code": exc.code.value}

    # 流式评估：图外层 / 窗口中层 / 模型内层——每图只加载与推理一次（推理与窗口无关
    # §5.7，density 三窗口复用后即弃）、模糊真值每 图×窗口 只算一次，内存 O(单图×模型数)
    test_ids = splits.image_ids("test")
    meta_by_id = {m.image_id: m for m in metas}
    pools = {w: [(iid, gt_by_window[w][iid].fixations) for iid in test_ids if iid in gt_by_window[w]] for w in windows}
    all_rows: list[Any] = []
    for i, iid in enumerate(test_ids):
        meta = meta_by_id[iid]
        img_arr: np.ndarray | None = None
        densities: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        for pname, predict in predictors.items():
            if img_arr is None:
                img_arr, _minfo = load_eval_image(layout.root / meta.rel_path)
            densities[pname] = predict(img_arr)
            model_meta.setdefault(pname, densities[pname][1])
        for window in windows:
            gt = gt_by_window[window]
            cb = cbs[window]
            case = _image_case(meta, gt[iid], "test")
            neg = pooled_other_fixations(pools[window], iid)
            F = blurred_truth(case.fixations, case.shape, config)[0]
            shape = (meta.height, meta.width)
            # 基线必须过同一管线（§4.4）：uniform 与 center_bias 逐图现生成
            all_rows += _rows_for_prediction(
                dataset_name=dataset_name,
                window=window,
                model="uniform",
                model_version=config.protocol_version,
                S=uniform_baseline(shape),
                case=case,
                meta=meta,
                gt_case=gt[iid],
                cb=cb,
                neg=neg,
                config=config,
                F_precomputed=F,
            )
            all_rows += _rows_for_prediction(
                dataset_name=dataset_name,
                window=window,
                model="center_bias",
                model_version=cb.version,
                S=cb.evaluate(shape),
                case=case,
                meta=meta,
                gt_case=gt[iid],
                cb=cb,
                neg=neg,
                config=config,
                F_precomputed=F,
            )
            for pname, (density, rmeta) in densities.items():
                all_rows += _rows_for_prediction(
                    dataset_name=dataset_name,
                    window=window,
                    model=pname,
                    model_version=str(rmeta.get("backend_version", "unknown")),
                    S=density,
                    case=case,
                    meta=meta,
                    gt_case=gt[iid],
                    cb=cb,
                    neg=neg,
                    config=config,
                    F_precomputed=F,
                )
        if progress_fn and (i + 1) % 10 == 0:
            progress_fn(f"流式评估：{i + 1}/{len(test_ids)} 图（{len(windows)} 窗口 × {2 + len(predictors)} 模型）")
    for window in windows:
        csv_path = out / f"per_image_{dataset_name}_{window}.csv"
        n = write_per_image_csv([r for r in all_rows if r.window == window], csv_path)
        outputs.csv_paths[window] = f"{csv_path} ({n} rows)"
        prog(f"窗口 {window}：逐图 CSV 落盘 {n} 行")
    outputs.model_meta = {**outputs.model_meta, **model_meta}

    prog("聚合：bootstrap CI / 配对差值 / Holm / 表 A-E…")
    candidates = [p for p in profiles if p in model_meta]
    table_a_rows, _used = aggregate_table_a(all_rows, config)
    pairs: list[tuple[str, str, tuple[str, ...]]] = [("center_bias", "uniform", config.criteria.s1_metrics)]
    pairs += [(c, "center_bias", config.criteria.s2_metrics) for c in candidates]
    pairs += [(c, "uniform", ("IG_U", "NSS")) for c in candidates]
    table_b_rows = aggregate_table_b(all_rows, config, pairs)
    table_c_rows = aggregate_table_c(all_rows, config)
    table_d_rows = build_audit_rows(
        gt_by_window=gt_by_window, splits=splits, audit=audit, config=config, zip_md5_ok=(dataset_md5 is not None)
    )
    fingerprint = build_fingerprint(
        config=config,
        splits_sha256=outputs.splits_sha256,
        cb_versions={w: {"version": cbs[w].version, "sha256": outputs.cb_hashes[w]} for w in windows},
        models=[
            {"profile": p, **{k: v for k, v in model_meta.get(p, {}).items() if k != "runtime"}} for p in candidates
        ],
        dataset_name=dataset_name,
        dataset_md5=dataset_md5,
    )
    fingerprint["variant"] = variant_label or "official"
    fingerprint["splits_mode"] = splits_mode
    fingerprint["dataset_zip_md5_source"] = md5_source
    outputs.fingerprint = fingerprint

    from .tables import REPORT_DISCLAIMER

    # 限制声明如实标注（L2 指令：观看条件实验假设 / centerbias 引用 / 混表禁令）
    limitation_notes: list[str] = [
        "参与者间变异未单独建模，CI 仅反映图像抽样变异（§7.2 最小方案）",
        "公开数据（通用 UI：webpage/desktop/mobile/poster）评估，非游戏 UI 验证；"
        "结果不得外推为游戏 UI 结论（协议 §0.2）",
        "本表分数不得与 UEyes 论文表格或官方 eval 输出混表（§6.8 口径差异 + 论文 Table 2 校准勘误）；"
        "文献值仅进独立参考表",
    ]
    if mode_audit.get("n_alpha"):
        limitation_notes.append(
            f"图像加载口径（评估复现路径）：{mode_audit['n_alpha']} 张 RGBA 按官方 cv2 口径丢弃 alpha 通道"
            f"（不合成、不填黑/白，保留存储 RGB 值），其中 {mode_audit['n_non_opaque_alpha']} 张含真实透明像素"
            f"（ID 与 alpha 极值见 audit.image_mode_audit）；模式分布 {mode_audit['modes']}。"
            "分析 CLI 路径（imaging.load_image）保持更严格的透明拒绝口径，两条路径互不混用"
        )
    if variant_label:
        limitation_notes.append(
            f"本运行为【{variant_label}】：划分文件（{splits_filename}，sha256={outputs.splits_sha256[:16]}…）"
            "与 CB 版本均不同于官方划分主运行——依 §8.1 与主运行严格分表并列，不得同表/同轴柱状图"
        )
    for pname in candidates:
        mm = model_meta.get(pname, {})
        vc = mm.get("viewing_conditions") or {}
        if vc.get("assumption"):
            limitation_notes.append(f"{pname}：观看条件为实验假设——{vc['assumption']}")
        if mm.get("centerbias"):
            limitation_notes.append(
                f"{pname}：显式 centerbias 先验（来源与哈希见表 E 配置指纹）；不得宣称其为游戏玩家的真实观看习惯"
            )
        if mm.get("semantics") == "log_density":
            limitation_notes.append(f"{pname}：原生 log_density 语义，经 P=exp(L−logsumexp(L)) 转换后进入指标管线")

    outputs.tables = {
        "A": build_table_a(table_a_rows),
        "B": build_table_b(table_b_rows),
        "C": build_table_c(table_c_rows),
        "D": build_table_d(table_d_rows),
        "E": build_table_e(fingerprint),
        "disclaimer": REPORT_DISCLAIMER,
        "participant_limitation": "参与者间变异未单独建模，CI 仅反映图像抽样变异（§7.2 最小方案）",
        "limitations": limitation_notes,
        "variant": variant_label or "official",
    }
    outputs.verdicts = judge_criteria(config=config, table_b=table_b_rows, candidates=candidates, s0_report=s0_report)
    if verdicts_descriptive:
        outputs.verdicts["descriptive_note"] = (
            f"本运行（{variant_label or '对照'}）的 S1/S2 判定仅作描述性记录：划分文件与 CB 版本不同于官方划分主运行"
            "（§8.1），正式门槛判定以官方划分运行为准；口径裁决权在 L1，不下选型结论"
        )
        if isinstance(outputs.verdicts.get("S2"), dict):
            outputs.verdicts["S2"]["descriptive_only"] = True

    variant_header = (
        f"> **{variant_label}**：与官方划分主运行严格分表、不混表（§8.1 划分文件不同；CB 版本不同）。\n\n"
        if variant_label
        else ""
    )
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name in ("A", "B", "C", "D", "E"):
        (tables_dir / f"table_{name}.md").write_text(
            variant_header + outputs.tables[name]["markdown"] + "\n", encoding="utf-8", newline="\n"
        )
    (tables_dir / "tables.json").write_text(
        json.dumps(
            {"tables": outputs.tables, "verdicts": outputs.verdicts, "audit": outputs.audit},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (out / "audit.json").write_text(
        json.dumps(outputs.audit, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "variant": variant_label or "official",
                "splits_mode": splits_mode,
                "splits_file": splits_filename,
                "dataset_zip_md5_source": md5_source,
                "windows": list(windows),
                "candidates": candidates,
                "skipped_profiles": outputs.skipped_profiles,
                "splits_sha256": outputs.splits_sha256,
                "cb_hashes": outputs.cb_hashes,
                "verdicts": outputs.verdicts,
                "config_hash": config.config_hash(),
                "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "disclaimer": REPORT_DISCLAIMER,
                "limitations": limitation_notes,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
        newline="\n",
    )
    prog("评估完成：产物在 " + str(out))
    return outputs


def build_split_comparison(
    official_tables_json: str | Path,
    fallback_tables_json: str | Path,
    out_dir: str | Path,
    *,
    metrics: Sequence[str] = ("IG_CB", "NSS"),
) -> dict[str, Any]:
    """对照报告（L2 指令⑤）：逐模型×窗口关键指标（IG_CB/NSS）官方划分 vs 备选划分的差值。

    两次运行的 test 集合不同（划分文件不同，§8.1 禁止同表）→ 本差值为**两次独立评估
    的组间描述性差异**（非同图配对差值 CI），用途 = 量化官方划分"test 分数偏乐观"幅度
    （近重复簇跨划分泄漏的影响）。diff = official_mean − fallback_mean；diff>0 表示
    官方划分下该指标更高（乐观方向）。
    """
    off = json.loads(Path(official_tables_json).read_text(encoding="utf-8"))
    fbk = json.loads(Path(fallback_tables_json).read_text(encoding="utf-8"))

    # tables.json 的 A.rows 为 {model, split, window, "IG_CB ↑": "mean [lo, hi]", ...} 的展示单元格；
    # 对照表从单元格文本解析 mean（4 位小数，描述性对照足够；全精度值在各自 per-image CSV）。
    def _parse_cell(text: str) -> tuple[float | None, str]:
        text = text.strip()
        if text in ("—", ""):
            return None, text
        mean_part = text.split("[")[0].strip()
        try:
            return float(mean_part), text
        except ValueError:
            return None, text

    rows: list[dict[str, Any]] = []
    fbk_rows = {(r["model"], r["window"]): r for r in fbk["tables"]["A"]["rows"]}
    for off_row in off["tables"]["A"]["rows"]:
        key = (off_row["model"], off_row["window"])
        fbk_row = fbk_rows.get(key)
        if fbk_row is None:
            continue
        for metric in metrics:
            col = next((c for c in off_row if c.startswith(metric + " ")), None)
            if col is None:
                continue
            off_mean, off_text = _parse_cell(str(off_row[col]))
            fbk_mean, fbk_text = _parse_cell(str(fbk_row[col]))
            if off_mean is None or fbk_mean is None:
                continue
            rows.append(
                {
                    "model": off_row["model"],
                    "window": off_row["window"],
                    "metric": metric,
                    "official_mean": off_mean,
                    "fallback_mean": fbk_mean,
                    "diff_official_minus_fallback": round(off_mean - fbk_mean, 6),
                    "official_cell": off_text,
                    "fallback_cell": fbk_text,
                }
            )
    header = ["model", "window", "metric", "official_mean", "fallback_mean", "diff(O−F)"]
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['window']} | {r['metric']} | {r['official_mean']:.4f} "
            f"| {r['fallback_mean']:.4f} | {r['diff_official_minus_fallback']:+.4f} |"
        )
    note = (
        "对照说明：官方划分与备选划分（dHash 簇约束 + §3.3 哈希分桶）的 test 集合不同，"
        "本表为两次独立评估的组间描述性差异（非同图配对差值 CI），仅用于量化官方划分近重复泄漏的"
        "乐观偏置幅度；两运行严格分表不混（§8.1）。diff>0 = 官方划分下更高（乐观方向）。"
        "备选划分上的 S2 判定仅描述性（口径裁决在 L1）。"
    )
    result = {
        "rows": rows,
        "markdown": "\n".join(md_lines),
        "note": note,
        "official_splits_sha256": off.get("audit", {}).get("splits_sha256", ""),
        "fallback_splits_sha256": fbk.get("audit", {}).get("splits_sha256", ""),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison_official_vs_fallback.md").write_text(
        f"# 官方划分 vs 备选划分对照（描述性）\n\n{note}\n\n{result['markdown']}\n", encoding="utf-8", newline="\n"
    )
    (out / "comparison_official_vs_fallback.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：``python -m ui_attention.metrics.eval.ueyes_driver --dataset … --out …``。"""
    parser = argparse.ArgumentParser(
        prog="ueyes_driver", description="R2 公开数据评估驱动（UEyes；合成数据自测经 pytest）"
    )
    parser.add_argument("--dataset", required=True, help="UEyes 解压目录（含 images/ eyetracker_logs/ info.csv）")
    parser.add_argument("--out", required=True, help="输出目录（建议 local-data/eval-output/，忽略目录）")
    parser.add_argument("--zip", default=None, help="可选：UEyes_dataset.zip 路径（给定则先做 md5 红线校验）")
    parser.add_argument("--windows", nargs="*", default=None, help="窗口子集（默认 1s 3s 7s；绝不混合）")
    parser.add_argument(
        "--profiles", nargs="*", default=["foveacast-onnx-3s-v1"], help="候选后端 profile（经 C2 registry 解析）"
    )
    parser.add_argument("--dataset-name", default="ueyes")
    parser.add_argument(
        "--splits-mode",
        default="official",
        choices=["official", "fallback_cluster"],
        help="official=官方 image_types.csv 划分；fallback_cluster=§3.3 哈希分桶+§3.4.1 簇约束（对照补跑）",
    )
    parser.add_argument(
        "--splits-file", default=SPLITS_VERSION, help="划分文件名（对照运行用 splits.v2-fallback.json）"
    )
    parser.add_argument("--cb-version-tag", default="v1", help="CB npz 版本标签（对照运行用 v2-fallback）")
    parser.add_argument(
        "--md5-verified", default=None, help="注入已验证 zip md5（同一 zip 在主运行已校验时免于重算，来源如实记录）"
    )
    parser.add_argument("--variant-label", default="", help="表标注（如：备选划分对照；与官方划分严格分表不混）")
    parser.add_argument("--descriptive-verdicts", action="store_true", help="S1/S2 判定仅作描述性记录（对照运行）")
    args = parser.parse_args(argv)

    def prog(msg: str) -> None:
        sys.stderr.write(f"[eval] {msg}\n")
        sys.stderr.flush()

    config = EvalConfig(dataset=args.dataset_name)
    try:
        outputs = run_full_evaluation(
            dataset_root=args.dataset,
            out_root=args.out,
            config=config,
            profiles=tuple(args.profiles),
            zip_path=args.zip,
            dataset_name=args.dataset_name,
            windows=tuple(args.windows) if args.windows else None,
            progress_fn=prog,
            splits_mode=args.splits_mode,
            splits_filename=args.splits_file,
            cb_version_tag=args.cb_version_tag,
            dataset_md5_verified=args.md5_verified,
            variant_label=args.variant_label,
            verdicts_descriptive=args.descriptive_verdicts,
        )
    except UiAttentionError as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": exc.to_dict()}, ensure_ascii=False, indent=2) + "\n")
        return exc.exit_code
    summary = {
        "ok": True,
        "result": {
            "out_root": str(outputs.out_root),
            "variant": args.variant_label or "official",
            "splits_mode": args.splits_mode,
            "splits_sha256": outputs.splits_sha256,
            "cb_hashes": outputs.cb_hashes,
            "skipped_profiles": outputs.skipped_profiles,
            "verdicts": outputs.verdicts,
            "disclaimer": outputs.tables.get("disclaimer", ""),
        },
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
