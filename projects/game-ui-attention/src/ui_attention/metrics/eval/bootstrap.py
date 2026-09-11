"""图像级不确定性（benchmark-protocol.md §7.2）。

原则：像素非独立——显著图相邻像素高度相关，任何以像素为样本单位的方差/CI 均无效；
**全部不确定性以图像为重采样单位**。

- 图像级 bootstrap（默认）：有放回重采样 B=2000 次（固定种子 20260911），
  每次重算指标均值 → 2.5/97.5 百分位为 95% CI；
- 分层 bootstrap（按类别）：webpage/desktop/mobile/poster 各自内部重采样后合并，
  避免类别构成波动主导 CI（表 C 默认口径）；
- 配对比较：同图配对差值 d_i 的 bootstrap CI + 胜率 #{d_i>0}/n；
  两模型 CI 各自不重叠 ≠ 差值显著，必须用配对差值 CI 判断；
- 多重比较：同一报告内两两检验用 Holm 校正，校正范围在报告中写明；
- 参与者维度：最小方案不做（限制声明："参与者间变异未单独建模，CI 仅反映图像抽样变异"）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import BOOTSTRAP_B_DEFAULT, BOOTSTRAP_SEED_DEFAULT

#: 参与者维度限制声明（§7.2，报告必须携带）
PARTICIPANT_LIMITATION = "参与者间变异未单独建模，CI 仅反映图像抽样变异"


def _clean(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    return v[np.isfinite(v)]


def image_bootstrap_ci(
    values: Sequence[float],
    *,
    categories: Sequence[str] | None = None,
    b: int = BOOTSTRAP_B_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
    ci: float = 0.95,
) -> dict[str, Any]:
    """图像级（可选按类别分层）bootstrap 均值 CI。

    values 为逐图指标值（NaN 图已排除）；categories 等长，给定时分层重采样。
    """
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    v = v[finite]
    if categories is not None:
        cats = np.asarray(list(categories), dtype=object)[finite]
    else:
        cats = None
    n = v.size
    if n == 0:
        raise ValueError("bootstrap 输入为空（全部图像被排除？）")
    if n == 1:
        # 单图无抽样变异可言：CI 退化为点值并显式标注
        return {
            "mean": float(v[0]),
            "ci_low": float(v[0]),
            "ci_high": float(v[0]),
            "b": b,
            "seed": seed,
            "ci_level": ci,
            "stratified": cats is not None,
            "n_images": 1,
            "degenerate_single_image": True,
        }
    rng = np.random.default_rng(seed)
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    means = np.empty(b, dtype=np.float64)
    if cats is None:
        idx_all = np.arange(n)
        for i in range(b):
            means[i] = v[rng.choice(idx_all, size=n, replace=True)].mean()
    else:
        groups = [np.flatnonzero(cats == c) for c in dict.fromkeys(cats.tolist())]
        for i in range(b):
            parts = [v[g[rng.choice(len(g), size=len(g), replace=True)]] for g in groups]
            means[i] = np.concatenate(parts).mean()
    return {
        "mean": float(v.mean()),
        "ci_low": float(np.quantile(means, lo_q)),
        "ci_high": float(np.quantile(means, hi_q)),
        "b": b,
        "seed": seed,
        "ci_level": ci,
        "stratified": cats is not None,
        "n_images": n,
        "degenerate_single_image": False,
    }


def paired_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    categories: Sequence[str] | None = None,
    b: int = BOOTSTRAP_B_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
    ci: float = 0.95,
) -> dict[str, Any]:
    """同图配对差值 d_i = A_i − B_i 的图像级 bootstrap：差值 CI + 胜率 + p 值。

    p 值为双侧 bootstrap 百分位法：``2·min(P(mean_d≤0), P(mean_d≥0))``，
    下限截断为 ``1/(B+1)``（重采样分布的分辨率极限），需在报告注明方法。
    """
    a = np.asarray(values_a, dtype=np.float64)
    bb = np.asarray(values_b, dtype=np.float64)
    if a.shape != bb.shape:
        raise ValueError(f"配对差值要求同图对齐：{a.shape} vs {bb.shape}")
    finite = np.isfinite(a) & np.isfinite(bb)
    d = (a - bb)[finite]
    cats = np.asarray(list(categories), dtype=object)[finite] if categories is not None else None
    n = d.size
    if n == 0:
        raise ValueError("配对差值输入为空")
    win_rate = float(np.count_nonzero(d > 0) / n)
    result: dict[str, Any] = {
        "mean_diff": float(d.mean()),
        "win_rate": win_rate,
        "n_paired": n,
        "b": b,
        "seed": seed,
        "ci_level": ci,
        "p_method": "two-sided bootstrap percentile, floor 1/(B+1)",
    }
    if n == 1:
        result.update({"ci_low": float(d[0]), "ci_high": float(d[0]), "p_value": 1.0, "degenerate_single_image": True})
        return result
    rng = np.random.default_rng(seed)
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    means = np.empty(b, dtype=np.float64)
    if cats is None:
        for i in range(b):
            means[i] = d[rng.integers(0, n, size=n)].mean()
    else:
        groups = [np.flatnonzero(cats == c) for c in dict.fromkeys(cats.tolist())]
        for i in range(b):
            parts = [d[g[rng.integers(0, len(g), size=len(g))]] for g in groups]
            means[i] = np.concatenate(parts).mean()
    frac_le0 = float(np.count_nonzero(means <= 0) / b)
    frac_ge0 = float(np.count_nonzero(means >= 0) / b)
    p = min(1.0, 2.0 * min(frac_le0, frac_ge0))
    p = max(p, 1.0 / (b + 1))
    result.update(
        {
            "ci_low": float(np.quantile(means, lo_q)),
            "ci_high": float(np.quantile(means, hi_q)),
            "p_value": p,
            "degenerate_single_image": False,
        }
    )
    return result


def holm_correction(p_values: Sequence[float], *, alpha: float = 0.05) -> dict[str, Any]:
    """Holm 逐步校正（多重比较；校正范围须由调用方在报告中写明）。"""
    p = np.asarray(list(p_values), dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running_max = max(running_max, adj)
        adjusted_sorted[idx] = min(1.0, running_max)
    return {
        "method": "holm",
        "alpha": alpha,
        "p_values": [float(x) for x in p],
        "adjusted": [float(adjusted_sorted[i]) for i in range(m)],
        "significant": [bool(adjusted_sorted[i] < alpha) for i in range(m)],
    }
