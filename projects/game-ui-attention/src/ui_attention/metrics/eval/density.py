"""统一防护原语 to_density（benchmark-protocol.md §6.0）。

::

    to_density(M, λ):                     # 概率图化
        M ← max(M, 0)                     # 负值截断为 0（记录被截断像素数）
        if sum(M) == 0: M ← 1/N（均匀）    # 全零 → 均匀，flag=degenerate
        M ← M / sum(M)
        M ← (1−λ)·M + λ·U                 # 严格正性混合（U=均匀），保证 log 有限
        return M                          # sum 仍为 1

评估路径专用；AOI 统计路径（metrics.probability）使用严格归一化，
两条路径不得互相混用数值或防护常数（technical-design §7 G0 锁定）。
全管线 float64，禁止 8-bit 中间存储参与计算（§6.0）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import LAMBDA_DEFAULT


def as_float64_2d(M: Any, what: str = "输入图") -> np.ndarray:
    if not isinstance(M, np.ndarray):
        raise TypeError(f"{what} 必须是 numpy.ndarray，得到 {type(M).__name__}")
    if M.ndim != 2:
        raise TypeError(f"{what} 必须是 (H, W) 二维数组，得到 ndim={M.ndim}")
    if M.dtype != np.float64:
        M = M.astype(np.float64)
    return M


def to_density(M: np.ndarray, lam: float = LAMBDA_DEFAULT) -> tuple[np.ndarray, dict[str, Any]]:
    """概率图化原语；返回 (density, flags)。

    flags：
    - ``n_clipped``: 被截断的负值像素数
    - ``degenerate``: 原图全零 → 已替换为均匀分布
    - ``constant``: 归一化后为常数图（含均匀基线；NSS/CC 的常数防护消费此 flag）

    输出保证：非负、有限、严格正（λ>0 时）、sum=1。
    """
    if not (0.0 <= lam < 1.0):
        raise ValueError(f"λ 必须在 [0,1)，得到 {lam}")
    arr = as_float64_2d(M, "to_density 输入")
    if not np.all(np.isfinite(arr)):
        raise ValueError("to_density 输入含 NaN/Inf（评估路径不做静默修补）")
    n = arr.size
    n_clipped = int(np.count_nonzero(arr < 0))
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    degenerate = total == 0.0
    if degenerate:
        arr = np.full_like(arr, 1.0 / n)
    else:
        arr = arr / total
    constant = bool(arr.max() == arr.min())
    if lam > 0.0:
        arr = (1.0 - lam) * arr + lam / n
    flags = {"n_clipped": n_clipped, "degenerate": degenerate, "constant": constant}
    return arr, flags
