"""评估基线（benchmark-protocol.md §4）：均匀分布 U 与数据驱动中心偏置 CB。

规则：
- CB 按数据集 × 窗口 × 划分分别估计；**只能由 train 划分构造**（§3.4.3 反泄漏，
  fit() 对 test 划分直接拒绝）；
- CB 自身也作为"模型"跑完整指标管线（表 A 占一行 model=center_bias）；
- CB 是版本化配置：文件 + 生成参数 + 哈希（save/load/fingerprint）；
- 不得把 UEyes/FiWI 估计的 CB 宣称为"游戏玩家的中心偏置"（technical-design §5）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .groundtruth import FixationSet
from .interp import resample_to_density

CB_VERSION = "cb.v1"
CB_BINS_DEFAULT = 64
CB_SIGMA_BIN_DEFAULT = 1.0


def uniform_baseline(shape: tuple[int, int]) -> np.ndarray:
    """均匀分布基线 U：U(p) = 1/N，每图相同（§4.1）。"""
    n = int(shape[0]) * int(shape[1])
    return np.full((int(shape[0]), int(shape[1])), 1.0 / n, dtype=np.float64)


@dataclass(frozen=True)
class CenterBiasFitInput:
    """单图 train 划分注视点（构造 CB 的输入单位）。"""

    shape: tuple[int, int]  # (H, W)
    fixations: FixationSet


@dataclass(frozen=True)
class CenterBiasBaseline:
    """构造方法 A（默认，非参数经验分布，§4.2）。

    步骤：归一化坐标 u=(x+0.5)/W, v=(y+0.5)/H（像素中心约定）→ 2D 直方图 64×64
    → 计数归一化 sum=1 → σ_bin=1 高斯平滑（消除量化伪影）→ 按图双线性上采样 + 重归一化。
    """

    histogram: np.ndarray  # (bins, bins) float64, sum=1（平滑后）
    bins: int = CB_BINS_DEFAULT
    sigma_bin: float = CB_SIGMA_BIN_DEFAULT
    weighting: str = "count"
    source_split: str = "train"
    source_split_hash: str = ""
    n_fixations: int = 0
    n_images: int = 0
    version: str = CB_VERSION

    # -- 构造 ----------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        per_image: Iterable[CenterBiasFitInput],
        *,
        bins: int = CB_BINS_DEFAULT,
        sigma_bin: float = CB_SIGMA_BIN_DEFAULT,
        weighting: str = "count",
        source_split: str = "train",
        source_split_hash: str = "",
        version: str = CB_VERSION,
    ) -> CenterBiasBaseline:
        if source_split != "train":
            # 反泄漏规则 §3.4.3：基线/先验/超参只能用 train 划分估计。
            raise ValueError(f"CB 只能由 train 划分构造（反泄漏），拒绝 source_split={source_split!r}")
        if bins <= 1:
            raise ValueError(f"bins 必须 >1，得到 {bins}")
        hist = np.zeros((bins, bins), dtype=np.float64)
        n_fix = 0
        n_img = 0
        for case in per_image:
            fix = case.fixations
            n_img += 1
            if len(fix) == 0:
                continue
            h, w = int(case.shape[0]), int(case.shape[1])
            x = fix.points[:, 0]
            y = fix.points[:, 1]
            u = np.clip((x + 0.5) / w, 0.0, 1.0 - 1e-12)
            v = np.clip((y + 0.5) / h, 0.0, 1.0 - 1e-12)
            weights = fix.weights if weighting == "duration" else np.ones_like(fix.weights)
            h_bin, _, _ = np.histogram2d(v, u, bins=bins, range=[[0.0, 1.0], [0.0, 1.0]], weights=weights)
            hist += h_bin
            n_fix += int(len(fix))
        total = hist.sum()
        if total <= 0:
            raise ValueError("train 划分注视点为空，无法构造 CB（禁止用 test 数据兜底）")
        hist /= total
        smoothed = gaussian_filter(hist, sigma=sigma_bin, mode="reflect")
        smoothed = np.maximum(smoothed, 0.0)
        s_total = smoothed.sum()
        if s_total <= 0:
            raise ValueError("CB 平滑后 sum<=0（数值异常）")
        smoothed /= s_total
        return cls(
            histogram=smoothed,
            bins=bins,
            sigma_bin=sigma_bin,
            weighting=weighting,
            source_split=source_split,
            source_split_hash=source_split_hash,
            n_fixations=n_fix,
            n_images=n_img,
            version=version,
        )

    # -- 求值 ----------------------------------------------------------------

    def evaluate(self, shape: tuple[int, int]) -> np.ndarray:
        """按图双线性上采样到评估网格并重归一化 sum=1（§4.2 步骤 4）。"""
        return resample_to_density(self.histogram, (int(shape[0]), int(shape[1])))

    # -- 版本化 + 哈希 ----------------------------------------------------------

    def params_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "bins": self.bins,
            "sigma_bin": self.sigma_bin,
            "weighting": self.weighting,
            "source_split": self.source_split,
            "source_split_hash": self.source_split_hash,
            "n_fixations": self.n_fixations,
            "n_images": self.n_images,
        }

    def save(self, path: str | Path) -> str:
        """落盘 npz（直方图 + 参数 + 源划分哈希 + 注视计数）；返回文件 sha256。

        **确定性字节**：手工构造 zip 并使用固定条目时间戳（1980-01-01）——
        np.savez 默认把当前时间写入 zip 条目导致同内容重跑哈希漂移，
        破坏"版本化+哈希"的可审计性（2026-09-11 UEyes 实跑发现）。
        产物仍是标准 npz（np.load 直接可读）。
        """
        import io
        import zipfile

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        params = json.dumps(self.params_dict(), sort_keys=True, ensure_ascii=False)

        def _npy_bytes(arr: np.ndarray) -> bytes:
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            return buf.getvalue()

        blob = io.BytesIO()
        with zipfile.ZipFile(blob, "w", zipfile.ZIP_STORED) as zf:
            # params 以 unicode 字符串数组存储（避免 object dtype 强制 allow_pickle）
            for name, arr in (("histogram.npy", self.histogram), ("params.npy", np.array(params))):
                zf.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), _npy_bytes(arr))
        data = blob.getvalue()
        p.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> tuple[CenterBiasBaseline, str]:
        """读取 npz；返回 (baseline, 文件 sha256)。"""
        p = Path(path)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p, allow_pickle=False) as data:
            hist = np.asarray(data["histogram"], dtype=np.float64)
            params = json.loads(str(data["params"]))
        baseline = cls(
            histogram=hist,
            bins=int(params["bins"]),
            sigma_bin=float(params["sigma_bin"]),
            weighting=str(params["weighting"]),
            source_split=str(params["source_split"]),
            source_split_hash=str(params["source_split_hash"]),
            n_fixations=int(params["n_fixations"]),
            n_images=int(params["n_images"]),
            version=str(params["version"]),
        )
        return baseline, digest

    def fingerprint(self, file_sha256: str | None = None) -> dict[str, Any]:
        """表 E 配置指纹消费：版本 + 参数 + 直方图内容哈希（+ 文件哈希）。"""
        content_hash = hashlib.sha256(np.ascontiguousarray(self.histogram).tobytes()).hexdigest()
        fp: dict[str, Any] = {**self.params_dict(), "histogram_sha256": content_hash}
        if file_sha256 is not None:
            fp["file_sha256"] = file_sha256
        return fp


def leakage_audit_check(cb: CenterBiasBaseline, splits_file_hash: str | None = None) -> list[str]:
    """泄漏审计清单（§3.4.5）CB 相关项：返回违规列表（空 = 通过）。"""
    problems: list[str] = []
    if cb.source_split != "train":
        problems.append(f"CB 源划分必须是 train，得到 {cb.source_split!r}")
    if splits_file_hash is not None and cb.source_split_hash and cb.source_split_hash != splits_file_hash:
        problems.append("CB 记录的源划分哈希与当前划分文件不一致（划分已变更 → 必须重建 CB）")
    return problems


def pooled_other_fixations(
    cases: Sequence[tuple[str, FixationSet]],
    self_image_id: str,
) -> FixationSet:
    """sAUC 负样本池：同划分、同窗口内其他图像的注视点合并（§6.4）。"""
    pts = [c[1].points for c in cases if c[0] != self_image_id and len(c[1])]
    ws = [c[1].weights for c in cases if c[0] != self_image_id and len(c[1])]
    if not pts:
        return FixationSet(points=np.empty((0, 2)), weights=np.empty(0))
    return FixationSet(points=np.concatenate(pts), weights=np.concatenate(ws))
