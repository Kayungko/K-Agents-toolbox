"""评估指标（benchmark-protocol.md §6.1-§6.7，规范性口径的唯一实现）。

- IG（bits/fixation，log2；默认基线 CB，另报 IG_U）
- NSS（离散注视点，常数图 → 记 0 + constant_map flag）
- CC（模糊真值 F vs 模型图；任一侧 std=0 → 记 0 + flag）
- sAUC（shuffled-fixations 口径；全阈值梯形积分——以平均秩 Mann-Whitney 等价实现）
- AUC-Judd（正样本 = 注视存在像素，负样本 = 同图其余像素）
- KL(F‖S)（真值在前；方向与 UEyes 官方实现相反，§6.8 防混表）
- SIM（可选，默认关闭；to_density λ=0）

全管线 float64；数值防护统一走 to_density（λ=1e-8），禁止 8-bit 中间存储参与计算。
UEyes 官方 eval 口径与本实现**数值不可比**（§6.8），复现分数不得与论文表格混表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import rankdata

from .config import EvalConfig
from .density import to_density
from .groundtruth import FixationSet, delta_map, pixel_indices


@dataclass(frozen=True)
class MetricResult:
    """单指标结果：数值 + 防护 flags（NaN 表示"该图排除，计入审计"）。"""

    name: str
    value: float
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def excluded(self) -> bool:
        return not np.isfinite(self.value)


def _weighted_mean_at_fixations(values: np.ndarray, fix: FixationSet) -> float:
    return float(np.sum(values * fix.weights) / fix.total_weight)


def _fixation_samples(S: np.ndarray, fix: FixationSet) -> np.ndarray:
    """S 在离散注视点像素上的取值（重复落点按次数/权重计入，§6.2）。"""
    xi, yi = pixel_indices(fix)
    return S[yi, xi]


# ---------------------------------------------------------------------------
# IG（§6.1）
# ---------------------------------------------------------------------------


def information_gain(S_raw: np.ndarray, B_raw: np.ndarray, fix: FixationSet, config: EvalConfig) -> MetricResult:
    """IG(S | B) = (1/|Φ|) Σ [log2 S(x,y) − log2 B(x,y)]，S、B 均经 to_density(λ)。

    基线 B 由调用方选择：CB（默认，IG_CB）或均匀 U（IG_U）。
    """
    if len(fix) == 0:
        return MetricResult("IG", float("nan"), {"empty_fixations": True})
    S, s_flags = to_density(S_raw, config.lam)
    B, b_flags = to_density(B_raw, config.lam)
    s_vals = _fixation_samples(S, fix)
    b_vals = _fixation_samples(B, fix)
    value = _weighted_mean_at_fixations(np.log2(s_vals) - np.log2(b_vals), fix)
    flags = {
        "baseline_degenerate": b_flags["degenerate"],
        "s_clipped": s_flags["n_clipped"],
        "b_clipped": b_flags["n_clipped"],
    }
    return MetricResult("IG", value, flags)


# ---------------------------------------------------------------------------
# NSS（§6.2）
# ---------------------------------------------------------------------------


def nss(S_raw: np.ndarray, fix: FixationSet, config: EvalConfig) -> MetricResult:
    """Ŝ = (S − mean)/std（全图像素统计，S 为原始显著图）；注视点上加权平均。

    std=0（常数图，含均匀基线 U）→ 固定记 0 并打 constant_map flag
    （U 基线 NSS≈0 是预期 sanity 结果，由 flag 机制而非除零保护产生）。
    """
    if len(fix) == 0:
        return MetricResult("NSS", float("nan"), {"empty_fixations": True})
    S = np.asarray(S_raw, dtype=np.float64)
    std = float(S.std())
    if std == 0.0:
        return MetricResult("NSS", 0.0, {"constant_map": True})
    normalized = (S - float(S.mean())) / std
    value = _weighted_mean_at_fixations(_fixation_samples(normalized, fix), fix)
    return MetricResult("NSS", value, {"constant_map": False})


# ---------------------------------------------------------------------------
# CC（§6.3）
# ---------------------------------------------------------------------------


def cc(F_density: np.ndarray, S_raw: np.ndarray, config: EvalConfig) -> MetricResult:
    """线性相关系数：F=模糊真值图（to_density 后），S=模型图（仿射不变）。

    任一侧 std=0 → 固定记 0 并打 flag（pysaliency 约定；双常数 NaN 覆写为 0）。
    """
    F = np.asarray(F_density, dtype=np.float64)
    S = np.asarray(S_raw, dtype=np.float64)
    if F.shape != S.shape:
        raise ValueError(f"CC 输入形状不一致：{F.shape} vs {S.shape}")
    f_std, s_std = float(F.std()), float(S.std())
    if f_std == 0.0 or s_std == 0.0:
        return MetricResult("CC", 0.0, {"constant_map": True, "constant_side": "F" if f_std == 0.0 else "S"})
    f = (F - float(F.mean())) / f_std
    s = (S - float(S.mean())) / s_std
    value = float(np.mean(f * s))
    return MetricResult("CC", value, {"constant_map": False})


# ---------------------------------------------------------------------------
# AUC 公共原语：全阈值扫描 + 梯形积分（平均秩 Mann-Whitney 为精确等价实现）
# ---------------------------------------------------------------------------


def _auc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    """ROC-AUC（全阈值梯形积分口径）。

    梯形法对 ties 的自然处理 ≡ 平均秩 Mann-Whitney U：
    ``AUC = (Σ rank(pos) − n₊(n₊+1)/2) / (n₊·n₋)``。S 全常数 → 0.5（ties 全并列的解析结果）。
    正或负样本为空 → NaN + 排除 flag（§6.4 数值防护）。
    """
    n_pos, n_neg = pos_scores.size, neg_scores.size
    if n_pos == 0 or n_neg == 0:
        return float("nan"), {"empty_samples": True, "n_pos": n_pos, "n_neg": n_neg}
    combined = np.concatenate([pos_scores, neg_scores])
    ranks = rankdata(combined, method="average")
    rank_sum_pos = float(ranks[:n_pos].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    constant = bool(np.all(combined == combined[0]))
    return float(auc), {"empty_samples": False, "constant_map": constant, "n_pos": n_pos, "n_neg": n_neg}


def _subsample(values: np.ndarray, m: int, seed: int) -> np.ndarray:
    if values.size <= m:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(values.size, size=m, replace=False)]


# ---------------------------------------------------------------------------
# sAUC（§6.4，shuffled-fixations 口径）
# ---------------------------------------------------------------------------


def sauc(
    S_raw: np.ndarray,
    fix_pos: FixationSet,
    fix_neg: FixationSet,
    config: EvalConfig,
) -> MetricResult:
    """sAUC：正样本 = 本图真值注视点；负样本 = 同划分同窗口**其他图像**的注视点。

    负样本默认取全部；``sauc_negative_multiplier`` 设置时按固定种子抽样
    M = multiplier × |Φ|（记录种子，入配置哈希）。
    """
    S = np.asarray(S_raw, dtype=np.float64)
    pos_scores = _fixation_samples(S, fix_pos) if len(fix_pos) else np.empty(0)
    neg_scores = _fixation_samples(S, fix_neg) if len(fix_neg) else np.empty(0)
    if config.sauc_negative_multiplier is not None and len(fix_pos):
        m = int(config.sauc_negative_multiplier * len(fix_pos))
        neg_scores = _subsample(neg_scores, m, config.sampling_seed)
    auc, flags = _auc_from_scores(pos_scores, neg_scores)
    return MetricResult("sAUC", auc, flags)


# ---------------------------------------------------------------------------
# AUC-Judd（§6.5）
# ---------------------------------------------------------------------------


def auc_judd(S_raw: np.ndarray, fix: FixationSet, config: EvalConfig) -> MetricResult:
    """AUC-Judd：正样本 = 注视像素（F_delta > 0），负样本 = 同图所有其他像素。

    大图像允许对负样本按固定种子均匀抽样（M = multiplier × 正样本数）；不抽样为默认。
    """
    S = np.asarray(S_raw, dtype=np.float64)
    d = delta_map(fix, S.shape)
    pos_mask = d > 0
    pos_scores = S[pos_mask]
    neg_scores = S[~pos_mask]
    if config.aucjudd_negative_multiplier is not None and pos_scores.size:
        m = int(config.aucjudd_negative_multiplier * pos_scores.size)
        neg_scores = _subsample(neg_scores, m, config.sampling_seed)
    auc, flags = _auc_from_scores(pos_scores, neg_scores)
    return MetricResult("AUC-Judd", auc, flags)


# ---------------------------------------------------------------------------
# KL（§6.6，方向固定 KL(F‖S)）
# ---------------------------------------------------------------------------


def kl_divergence(F_raw: np.ndarray, S_raw: np.ndarray, config: EvalConfig) -> MetricResult:
    """KL(F‖S) = Σ F·[log2 F − log2 S]（真值在前，越低越好）。

    F、S 均经 to_density(λ=1e-8) 保证 log 有限；F(p)=0 的项按 0·log0=0 跳过。
    警示：UEyes 官方 kldiv 为 KL(S‖F)，方向相反且数值不同（§6.8），不得混表。
    """
    F, f_flags = to_density(F_raw, config.lam)
    S, s_flags = to_density(S_raw, config.lam)
    mask = F > 0
    value = float(np.sum(F[mask] * (np.log2(F[mask]) - np.log2(S[mask]))))
    return MetricResult(
        "KL",
        value,
        {"direction": "F||S", "f_degenerate": f_flags["degenerate"], "s_clipped": s_flags["n_clipped"]},
    )


# ---------------------------------------------------------------------------
# SIM（§6.7，可选，默认关闭）
# ---------------------------------------------------------------------------


def similarity(F_raw: np.ndarray, S_raw: np.ndarray, config: EvalConfig) -> MetricResult:
    """SIM = Σ min(F̂_d, S_d)，双方 to_density(λ=0)（纯归一化，无混合）。"""
    F, f_flags = to_density(F_raw, lam=0.0)
    S, s_flags = to_density(S_raw, lam=0.0)
    value = float(np.sum(np.minimum(F, S)))
    return MetricResult("SIM", value, {"f_degenerate": f_flags["degenerate"], "s_degenerate": s_flags["degenerate"]})
