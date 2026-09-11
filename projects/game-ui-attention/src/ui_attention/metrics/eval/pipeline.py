"""评估管线（benchmark-protocol.md 附录 A 最小伪代码的可运行实现）。

- 逐图计算（§7.1）：每 (dataset, split, window, model, image) 一行一指标落 CSV；
  聚合与 CI 只从逐图值计算，禁止把多图真值/预测拼接后计算"数据集级"指标；
- 窗口绝不混合（§5）：window 是每行主键字段；
- 基线必须过同一管线（§4.4）：uniform 与 center_bias 作为"模型"各占一行集合；
- 排除门槛（§3.5 建议默认，冻结值待一级总控确认）：窗口内有效注视 <10 → low_fixation；
  观看者 <3 → low_viewers；空正/负样本 → empty_samples（NaN + 审计）；
- S0 管线自检门（§8.3）：均匀基线 NSS≈0 / AUC≈0.5 / IG_U≈0、CB 的 IG_CB≈0；
  不过 → 脚本缺陷，结构化错误退出，不产出假成功；
- S1-S4 判据可配置且默认关闭（config.SuccessCriteria），待一级总控确认。

失败行为：S 含 NaN/负值、归一化超差、S0 不过 → 结构化错误退出（EvalGateError），
不产出部分成功报告（与技术方案 §10 一致）。本管线当前仅以合成数据自测——
UEyes 下载待一级总控批准，禁止下载任何数据集。
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...errors import ErrorCode, UiAttentionError
from .baselines import CenterBiasBaseline, CenterBiasFitInput, pooled_other_fixations, uniform_baseline
from .config import EvalConfig
from .groundtruth import FixationSet, blurred_truth
from .metrics import MetricResult, auc_judd, cc, information_gain, kl_divergence, nss, sauc, similarity

#: 逐图 CSV 列（§7.1 冻结列序）
CSV_COLUMNS: tuple[str, ...] = (
    "dataset",
    "split",
    "window",
    "model",
    "model_version",
    "image_id",
    "category",
    "block",
    "n_viewers",
    "n_fix",
    "excluded_flag",
    "metric",
    "value",
)

#: §3.5 图像有效性门槛（建议默认，冻结时确认——可配置）
MIN_FIXATIONS_DEFAULT = 10
MIN_VIEWERS_DEFAULT = 3


class EvalGateError(UiAttentionError):
    """评估自检门失败（S0 不过 / 输入无效）：脚本缺陷或数据缺陷，禁止出报告。"""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.INTERNAL_ERROR, message, dict(details or {}))
        self.details.setdefault("reason", "eval_gate_failed")


@dataclass(frozen=True)
class ImageCase:
    """单图评估输入：真值注视点 + 元数据（预测由 predictions 提供）。"""

    image_id: str
    shape: tuple[int, int]  # (H, W)
    fixations: FixationSet
    category: str = "synthetic"
    block: str = ""
    n_viewers: int = 0
    split: str = "test"


@dataclass(frozen=True)
class PerImageRow:
    """逐图 CSV 行（§7.1）。"""

    dataset: str
    split: str
    window: str
    model: str
    model_version: str
    image_id: str
    category: str
    block: str
    n_viewers: int
    n_fix: int
    excluded_flag: str
    metric: str
    value: float

    def to_csv_row(self) -> list[str]:
        # 保留浮点原始值（repr 全精度）；NaN → 空值 + excluded_flag 已在行内
        value_str = "" if not np.isfinite(self.value) else repr(float(self.value))
        return [
            self.dataset,
            self.split,
            self.window,
            self.model,
            self.model_version,
            self.image_id,
            self.category,
            str(self.block),
            str(self.n_viewers),
            str(self.n_fix),
            self.excluded_flag,
            self.metric,
            value_str,
        ]


def write_per_image_csv(rows: Iterable[PerImageRow], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow(row.to_csv_row())
            count += 1
    return count


def exclusion_flag(
    case: ImageCase,
    *,
    min_fixations: int = MIN_FIXATIONS_DEFAULT,
    min_viewers: int = MIN_VIEWERS_DEFAULT,
) -> str:
    """§3.5 图像有效性门槛；排除规则一经冻结不得中途更改（更改即新 protocol_version）。"""
    if len(case.fixations) < min_fixations:
        return "low_fixation"
    if case.n_viewers and case.n_viewers < min_viewers:
        return "low_viewers"
    return ""


def _validate_prediction(S_raw: np.ndarray, image_id: str) -> np.ndarray:
    """S 含 NaN/负值 → 结构化错误退出（附录 A 失败行为；不静默修补）。"""
    S = np.asarray(S_raw, dtype=np.float64)
    if S.ndim != 2:
        raise EvalGateError(f"预测图必须是 (H, W)：{image_id}", {"shape": list(S.shape)})
    if not np.all(np.isfinite(S)):
        raise EvalGateError(f"预测图含 NaN/Inf：{image_id}", {"image_id": image_id})
    if np.any(S < 0):
        raise EvalGateError(f"预测图含负值：{image_id}", {"image_id": image_id, "min": float(S.min())})
    return S


def evaluate_case(
    case: ImageCase,
    S_raw: np.ndarray,
    *,
    cb: CenterBiasBaseline | None,
    neg_pool: FixationSet,
    config: EvalConfig,
    dataset: str,
    window: str,
    model: str,
    model_version: str,
    min_fixations: int = MIN_FIXATIONS_DEFAULT,
    min_viewers: int = MIN_VIEWERS_DEFAULT,
) -> list[PerImageRow]:
    """单图全指标（IG_CB/IG_U/NSS/CC/sAUC/AUC-Judd/KL[, SIM]）。"""
    S = _validate_prediction(S_raw, case.image_id)
    if S.shape != tuple(case.shape):
        raise EvalGateError(
            f"预测图与评估网格不一致（应先在模型侧双线性重采样到原图分辨率）：{case.image_id}",
            {"prediction": list(S.shape), "grid": list(case.shape)},
        )
    flag = exclusion_flag(case, min_fixations=min_fixations, min_viewers=min_viewers)
    F, _f_flags = blurred_truth(case.fixations, case.shape, config)
    U = uniform_baseline(case.shape)

    results: list[MetricResult] = []
    cb_density = cb.evaluate(case.shape) if cb is not None else None
    if cb_density is not None:
        # 默认双报：IG_CB（Tuebingen 惯例主基线）+ IG_U（§6.1）
        ig_cb = information_gain(S, cb_density, case.fixations, config)
        results.append(MetricResult("IG_CB", ig_cb.value, ig_cb.flags))
    ig_u = information_gain(S, U, case.fixations, config)
    results.append(MetricResult("IG_U", ig_u.value, ig_u.flags))
    results.append(nss(S, case.fixations, config))
    results.append(cc(F, S, config))
    results.append(sauc(S, case.fixations, neg_pool, config))
    results.append(auc_judd(S, case.fixations, config))
    results.append(kl_divergence(F, S, config))
    if config.include_sim:
        results.append(similarity(F, S, config))

    rows: list[PerImageRow] = []
    for res in results:
        excluded_flag_row = flag or ("empty_samples" if res.excluded else "")
        rows.append(
            PerImageRow(
                dataset=dataset,
                split=case.split,
                window=window,
                model=model,
                model_version=model_version,
                image_id=case.image_id,
                category=case.category,
                block=case.block,
                n_viewers=case.n_viewers,
                n_fix=len(case.fixations),
                excluded_flag=excluded_flag_row,
                metric=res.name,
                value=float(res.value),
            )
        )
    return rows


def run_window_evaluation(
    cases: Sequence[ImageCase],
    predictions: Mapping[str, np.ndarray] | Callable[[ImageCase], np.ndarray],
    *,
    cb: CenterBiasBaseline | None,
    config: EvalConfig,
    dataset: str,
    window: str,
    model: str,
    model_version: str,
) -> list[PerImageRow]:
    """按窗口独立评估（§5：绝不混合窗口）；同一 model 的全部逐图行。"""
    fixation_pool = [(c.image_id, c.fixations) for c in cases]
    rows: list[PerImageRow] = []
    for case in cases:
        S = predictions(case) if callable(predictions) else predictions[case.image_id]
        neg_pool = pooled_other_fixations(fixation_pool, case.image_id)
        rows.extend(
            evaluate_case(
                case,
                S,
                cb=cb,
                neg_pool=neg_pool,
                config=config,
                dataset=dataset,
                window=window,
                model=model,
                model_version=model_version,
            )
        )
    return rows


def run_baseline_rows(
    cases: Sequence[ImageCase],
    cb: CenterBiasBaseline | None,
    *,
    config: EvalConfig,
    dataset: str,
    window: str,
) -> list[PerImageRow]:
    """基线必须过同一管线（§4.4）：uniform 与 center_bias 各作为"模型"跑全指标。"""
    rows: list[PerImageRow] = []
    uniform_pred = {c.image_id: uniform_baseline(c.shape) for c in cases}
    rows += run_window_evaluation(
        cases,
        uniform_pred,
        cb=cb,
        config=config,
        dataset=dataset,
        window=window,
        model="uniform",
        model_version=config.protocol_version,
    )
    if cb is not None:
        cb_pred = {c.image_id: cb.evaluate(c.shape) for c in cases}
        rows += run_window_evaluation(
            cases,
            cb_pred,
            cb=cb,
            config=config,
            dataset=dataset,
            window=window,
            model="center_bias",
            model_version=cb.version,
        )
    return rows


# ---------------------------------------------------------------------------
# S0 管线自检门（§8.3；不过 → EvalGateError 结构化失败，禁止出报告）
# ---------------------------------------------------------------------------


def _synthetic_fixations(
    shape: tuple[int, int], seed: int, n: int = 60, center_bias_strength: float = 0.7
) -> FixationSet:
    """确定性合成注视点（中心聚集 + 均匀背景），仅用于 S0 自检。"""
    rng = np.random.default_rng(seed)
    h, w = shape
    n_center = int(n * center_bias_strength)
    cx = rng.normal(w * 0.5, w * 0.08, n_center)
    cy = rng.normal(h * 0.5, h * 0.08, n_center)
    ux = rng.uniform(0, w - 1, n - n_center)
    uy = rng.uniform(0, h - 1, n - n_center)
    pts = np.stack([np.concatenate([cx, ux]), np.concatenate([cy, uy])], axis=1)
    pts = np.clip(pts, 0, [w - 1.0, h - 1.0])
    return FixationSet(points=pts, weights=np.ones(n))


def s0_self_check(config: EvalConfig, shape: tuple[int, int] = (120, 160)) -> dict[str, Any]:
    """S0：均匀基线 NSS≈0 / AUC≈0.5 / IG_U≈0；CB 的 IG_CB≈0（定义自洽）。

    返回逐项检查报告；任一不过 → EvalGateError（脚本缺陷，禁止出报告）。
    """
    crit = config.criteria
    if not crit.s0_enabled:
        return {"skipped": True}
    fix_a = _synthetic_fixations(shape, seed=config.sampling_seed)
    fix_b = _synthetic_fixations(shape, seed=config.sampling_seed + 1)
    U = uniform_baseline(shape)
    cb = CenterBiasBaseline.fit(
        [CenterBiasFitInput(shape=shape, fixations=fix_b)],
        bins=min(16, config.cb_bins),
        sigma_bin=config.cb_sigma_bin,
        source_split="train",
        source_split_hash="s0-synthetic",
    )
    cb_density = cb.evaluate(shape)

    checks: dict[str, dict[str, Any]] = {}

    def _check(name: str, result: MetricResult, expected: float, tol: float) -> None:
        ok = np.isfinite(result.value) and abs(result.value - expected) <= tol
        checks[name] = {
            "value": result.value,
            "expected": expected,
            "tol": tol,
            "ok": bool(ok),
            "flags": dict(result.flags),
        }

    _check("uniform.NSS==0", nss(U, fix_a, config), 0.0, crit.s0_nss_tol)
    _check("uniform.AUC-Judd==0.5", auc_judd(U, fix_a, config), 0.5, crit.s0_auc_tol)
    _check("uniform.sAUC==0.5", sauc(U, fix_a, fix_b, config), 0.5, crit.s0_auc_tol)
    _check("uniform.IG_U==0", information_gain(U, U, fix_a, config), 0.0, crit.s0_ig_tol)
    F, _ = blurred_truth(fix_a, shape, config)
    _check("uniform.CC==0", cc(F, U, config), 0.0, crit.s0_nss_tol)
    _check("center_bias.IG_CB==0", information_gain(cb_density, cb_density, fix_a, config), 0.0, crit.s0_ig_tol)

    # 大网格常数图数值稳健性（防回归）：667×1110 均匀图的 np.full 均值求和存在 ulp 级
    # 舍入 → std() 非零；常数图判定必须走逐元素 max==min 路径（2026-09-11 UEyes 实跑缺陷）
    big_shape = (667, 1110)
    U_big = uniform_baseline(big_shape)
    fix_big = _synthetic_fixations(big_shape, seed=config.sampling_seed + 7, n=40)
    _check("uniform_big.NSS==0", nss(U_big, fix_big, config), 0.0, crit.s0_nss_tol)
    _check("uniform_big.AUC-Judd==0.5", auc_judd(U_big, fix_big, config), 0.5, crit.s0_auc_tol)
    _check("uniform_big.IG_U==0", information_gain(U_big, U_big, fix_big, config), 0.0, crit.s0_ig_tol)

    failed = [name for name, c in checks.items() if not c["ok"]]
    report = {"skipped": False, "checks": checks, "failed": failed}
    if failed:
        raise EvalGateError(
            f"S0 管线自检未通过（脚本缺陷，禁止出报告）：{failed}",
            {"failed": failed, "checks": checks},
        )
    return report
