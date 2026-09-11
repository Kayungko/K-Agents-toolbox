"""metrics.eval 合成数据自测（benchmark-protocol R2 口径；validation-plan §4.1“获批前评估代码以合成数据自测”）。

合成数据只能证明计算链正确，不能证明人会看哪里；UEyes 未下载（待一级总控批准）。
覆盖：to_density 原语、σ=1° 换算、真值构造、IG/NSS/CC/sAUC/AUC-Judd/KL/SIM、
CB 基线（train-only 反泄漏、版本化+哈希）、bootstrap（确定性/分层/配对/Holm）、
逐图 CSV、S0 管线自检门（含失败路径）。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fixtures import synthetic  # noqa: E402
from ui_attention.metrics.eval import (  # noqa: E402
    LAMBDA_DEFAULT,
    CenterBiasBaseline,
    CenterBiasFitInput,
    EvalConfig,
    EvalGateError,
    FixationSet,
    ImageCase,
    auc_judd,
    blurred_truth,
    build_fixation_set,
    build_table_a,
    build_table_e,
    cc,
    delta_map,
    discrete_truth,
    evaluate_case,
    exclusion_flag,
    gaussian_blur,
    holm_correction,
    image_bootstrap_ci,
    information_gain,
    kl_divergence,
    leakage_audit_check,
    nss,
    paired_bootstrap,
    resample_to_density,
    run_baseline_rows,
    run_window_evaluation,
    s0_self_check,
    sauc,
    similarity,
    to_density,
    ueyes_sigma_px,
    uniform_baseline,
    write_per_image_csv,
)
from ui_attention.metrics.eval.config import SuccessCriteria  # noqa: E402

CONFIG = EvalConfig(dataset="synthetic")
SHAPE = (60, 80)  # (H, W)


def _fix(points, weights=None) -> FixationSet:
    return build_fixation_set(np.asarray(points, dtype=np.float64), SHAPE, weights=weights)


def _center_fixations(n=40, seed=7) -> FixationSet:
    rng = np.random.default_rng(seed)
    pts = np.stack(
        [rng.normal(40, 5, n), rng.normal(30, 4, n)],
        axis=1,
    )
    return build_fixation_set(np.clip(pts, 0, [SHAPE[1] - 1, SHAPE[0] - 1]), SHAPE)


def _gaussian_map(shape, cx, cy, sigma=6.0) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))


# ---------------------------------------------------------------------------
# to_density 原语（λ=1e-8、全零→均匀+degenerate、常数图防护）
# ---------------------------------------------------------------------------


def test_lambda_frozen():
    assert LAMBDA_DEFAULT == 1e-8
    assert CONFIG.lam == 1e-8
    assert CONFIG.log_base == 2


def test_to_density_all_zero_becomes_uniform_degenerate():
    P, flags = to_density(np.zeros((4, 5)), lam=1e-8)
    assert flags["degenerate"] is True
    assert P.sum() == pytest.approx(1.0)
    assert P.min() == P.max()  # 均匀


def test_to_density_clips_negatives_and_records_count():
    M = np.array([[1.0, -2.0], [3.0, -0.5]])
    P, flags = to_density(M, lam=0.0)
    assert flags["n_clipped"] == 2
    assert P.sum() == pytest.approx(1.0)
    assert P[0, 1] == 0.0 and P[1, 1] == 0.0


def test_to_density_lambda_mix_keeps_sum_one_and_strict_positive():
    M = np.zeros((8, 8))
    M[0, 0] = 1.0
    P, flags = to_density(M, lam=1e-8)
    assert flags["degenerate"] is False
    assert P.sum() == pytest.approx(1.0, abs=1e-12)
    assert P.min() > 0  # 严格正性：log 不会遇到 0
    assert P[0, 0] == pytest.approx((1 - 1e-8) * 1.0 + 1e-8 / 64)


def test_to_density_constant_flag():
    P, flags = to_density(np.full((4, 4), 2.5))
    assert flags["constant"] is True
    assert P.sum() == pytest.approx(1.0)


def test_to_density_rejects_nan():
    M = np.full((3, 3), 1.0)
    M[1, 1] = np.nan
    with pytest.raises(ValueError):
        to_density(M)


# ---------------------------------------------------------------------------
# σ=1° 换算与模糊
# ---------------------------------------------------------------------------


def test_ueyes_sigma_conversion():
    assert ueyes_sigma_px(1920, 1200) == pytest.approx(40.0)
    assert ueyes_sigma_px(960, 600) == pytest.approx(20.0)
    assert ueyes_sigma_px(3840, 1200) == pytest.approx(80.0)  # max(W/1920, H/1200)
    assert ueyes_sigma_px(1920, 2400) == pytest.approx(80.0)


def test_gaussian_blur_sum_preserved_and_constant_unchanged():
    d = delta_map(_fix([[10.0, 10.0], [10.2, 10.7]]), SHAPE)
    blurred = gaussian_blur(d, 5.0, mode="reflect")
    assert blurred.sum() == pytest.approx(d.sum(), rel=1e-9)  # reflect 边界保质量
    const = np.full(SHAPE, 3.0)
    assert gaussian_blur(const, 4.0) == pytest.approx(const)


# ---------------------------------------------------------------------------
# 真值构造：floor 取整、越界丢弃计数、离散分布
# ---------------------------------------------------------------------------


def test_fixation_floor_convention_and_oob_drop():
    fix = build_fixation_set([[2.7, 3.2], [-1.0, 5.0], [80.0, 5.0], [5.0, 60.0]], SHAPE)
    d = delta_map(fix, SHAPE)
    assert d[3, 2] == 1.0  # floor(2.7)=2, floor(3.2)=3
    assert fix.n_dropped_out_of_bounds == 3  # x<0、x==W、y==H 全部越界丢弃
    assert len(fix) == 1


def test_discrete_truth_normalized():
    fix = _fix([[5.0, 5.0], [5.5, 5.5], [20.0, 30.0]], weights=[1.0, 3.0, 1.0])
    F = discrete_truth(fix, SHAPE)
    assert F.sum() == pytest.approx(1.0)
    assert F[5, 5] == pytest.approx(4.0 / 5.0)  # 同像素重复落点权重合并


def test_blurred_truth_sum_one_with_sigma_recorded():
    fix = _center_fixations()
    F, flags = blurred_truth(fix, SHAPE, CONFIG)
    assert F.sum() == pytest.approx(1.0, abs=1e-9)
    assert flags["sigma_px"] == pytest.approx(ueyes_sigma_px(SHAPE[1], SHAPE[0]))


# ---------------------------------------------------------------------------
# 指标：解析性质
# ---------------------------------------------------------------------------


def test_ig_zero_when_equal_baseline():
    fix = _center_fixations()
    S = _gaussian_map(SHAPE, 40, 30)
    r = information_gain(S, S, fix, CONFIG)
    assert r.value == pytest.approx(0.0, abs=1e-12)


def test_ig_positive_for_concentrated_prediction_and_analytic_delta_case():
    fix = _fix([[10.0, 10.0]])
    S = np.zeros(SHAPE)
    S[10, 10] = 1.0
    U = uniform_baseline(SHAPE)
    r = information_gain(S, U, fix, CONFIG)
    n = SHAPE[0] * SHAPE[1]
    lam = 1e-8
    expected = np.log2((1 - lam) * 1.0 + lam / n) - np.log2(1.0 / n)
    assert r.value == pytest.approx(expected, rel=1e-6)
    assert r.value > 0


def test_ig_empty_fixations_nan():
    r = information_gain(np.ones(SHAPE), np.ones(SHAPE), _fix([]), CONFIG)
    assert r.excluded and r.flags.get("empty_fixations")


def test_nss_constant_map_zero_with_flag():
    r = nss(uniform_baseline(SHAPE), _center_fixations(), CONFIG)
    assert r.value == 0.0
    assert r.flags["constant_map"] is True


def test_nss_positive_for_matching_map():
    fix = _center_fixations(seed=3)
    S = _gaussian_map(SHAPE, 40, 30)
    r = nss(S, fix, CONFIG)
    assert r.value > 1.0  # 注视点集中在高显著区
    assert r.flags["constant_map"] is False


def test_cc_identical_one_opposite_minus_one_constant_zero():
    S = _gaussian_map(SHAPE, 40, 30)
    F, _flags = to_density(S, 0.0)
    assert cc(F, S, CONFIG).value == pytest.approx(1.0)
    assert cc(F, -S, CONFIG).value == pytest.approx(-1.0)
    r = cc(F, np.full(SHAPE, 2.0), CONFIG)
    assert r.value == 0.0 and r.flags["constant_map"] is True


def test_sauc_perfect_separation_and_constant():
    fix_pos = _fix([[10.0, 10.0], [12.0, 11.0]])
    fix_neg = _fix([[60.0, 50.0], [70.0, 55.0]])
    S = np.zeros(SHAPE)
    S[10, 10] = 1.0
    S[11, 12] = 1.0
    r = sauc(S, fix_pos, fix_neg, CONFIG)
    assert r.value == pytest.approx(1.0)
    r_const = sauc(np.full(SHAPE, 0.5), fix_pos, fix_neg, CONFIG)
    assert r_const.value == pytest.approx(0.5)
    assert r_const.flags["constant_map"] is True


def test_sauc_empty_negatives_nan_excluded():
    r = sauc(np.ones(SHAPE), _fix([[5.0, 5.0]]), _fix([]), CONFIG)
    assert r.excluded and r.flags["empty_samples"]


def test_sauc_subsample_deterministic():
    cfg = EvalConfig(sauc_negative_multiplier=2, sampling_seed=42)
    rng = np.random.default_rng(0)
    fix_pos = build_fixation_set(rng.uniform([0, 0], [79, 59], (20, 2)), SHAPE)
    fix_neg = build_fixation_set(rng.uniform([0, 0], [79, 59], (5000, 2)), SHAPE)
    S = _gaussian_map(SHAPE, 40, 30)
    a = sauc(S, fix_pos, fix_neg, cfg)
    b = sauc(S, fix_pos, fix_neg, cfg)
    assert a.value == b.value
    assert a.flags["n_neg"] == 40  # M = 2 × |Φ|


def test_auc_judd_perfect_and_constant_and_subsample():
    fix = _fix([[10.0, 10.0], [20.0, 20.0]])
    S = np.zeros(SHAPE)
    S[10, 10] = 2.0
    S[20, 20] = 1.0
    r = auc_judd(S, fix, CONFIG)
    assert r.value == pytest.approx(1.0)
    r_const = auc_judd(np.full(SHAPE, 7.0), fix, CONFIG)
    assert r_const.value == pytest.approx(0.5)  # ties 全并列的解析结果
    cfg_sub = EvalConfig(aucjudd_negative_multiplier=10)
    r_sub = auc_judd(S, fix, cfg_sub)
    assert r_sub.value == pytest.approx(1.0)
    assert r_sub.flags["n_neg"] == 20  # M = 10 × 正样本数


def test_kl_direction_and_zero_for_identical():
    fix = _center_fixations()
    F, _flags = blurred_truth(fix, SHAPE, CONFIG)
    r_same = kl_divergence(F, F, CONFIG)
    assert r_same.value == pytest.approx(0.0, abs=1e-12)
    assert r_same.flags["direction"] == "F||S"
    S = _gaussian_map(SHAPE, 5, 5)  # 与真值错位
    r_diff = kl_divergence(F, S, CONFIG)
    assert r_diff.value > 0  # KL 非负；方向真值在前


def test_kl_asymmetry_not_mixed_with_official():
    """KL(F‖S) ≠ KL(S‖F)（UEyes 官方方向相反，数值不同——防混表）。"""
    fix = _center_fixations(seed=11)
    F, _ = blurred_truth(fix, SHAPE, CONFIG)
    S = _gaussian_map(SHAPE, 60, 45, sigma=10)
    forward = kl_divergence(F, S, CONFIG).value
    reverse = kl_divergence(S, F, CONFIG).value
    assert forward != pytest.approx(reverse, rel=1e-3)


def test_sim_identical_and_disjoint():
    M = np.zeros(SHAPE)
    M[10:20, 10:20] = 1.0
    r_same = similarity(M, M.copy(), CONFIG)
    assert r_same.value == pytest.approx(1.0)
    N = np.zeros(SHAPE)
    N[40:50, 60:70] = 1.0
    r_dis = similarity(M, N, CONFIG)
    assert r_dis.value == pytest.approx(0.0, abs=1e-12)


def test_uniform_baseline():
    U = uniform_baseline(SHAPE)
    assert U.sum() == pytest.approx(1.0)
    assert U[0, 0] == pytest.approx(1.0 / (60 * 80))


# ---------------------------------------------------------------------------
# CB 基线：构造、求值、版本化+哈希、反泄漏
# ---------------------------------------------------------------------------


def _train_inputs(seed_base=100, n_images=4):
    out = []
    for i in range(n_images):
        rng = np.random.default_rng(seed_base + i)
        pts = np.stack([rng.normal(40, 8, 50), rng.normal(30, 6, 50)], axis=1)
        pts = np.clip(pts, 0, [SHAPE[1] - 1, SHAPE[0] - 1])
        out.append(CenterBiasFitInput(shape=SHAPE, fixations=build_fixation_set(pts, SHAPE)))
    return out


def test_cb_fit_evaluate_and_center_concentration():
    cb = CenterBiasBaseline.fit(_train_inputs(), source_split_hash="h" * 64)
    assert cb.histogram.sum() == pytest.approx(1.0)
    density = cb.evaluate(SHAPE)
    assert density.shape == SHAPE
    assert density.sum() == pytest.approx(1.0)
    # 中心聚集的 train 注视 → CB 中心高于边角
    assert density[30, 40] > density[0, 0]
    assert cb.n_images == 4 and cb.n_fixations == 4 * 50


def test_cb_save_load_roundtrip(tmp_path):
    cb = CenterBiasBaseline.fit(_train_inputs(), source_split_hash="a" * 64)
    p = tmp_path / "cb_synthetic_3s_train.v1.npz"
    digest = cb.save(p)
    assert len(digest) == 64
    loaded, digest2 = CenterBiasBaseline.load(p)
    assert digest2 == digest  # 版本化 + 哈希
    assert np.array_equal(loaded.histogram, cb.histogram)
    assert loaded.source_split_hash == "a" * 64
    assert loaded.version == cb.version
    fp = loaded.fingerprint(digest)
    assert fp["file_sha256"] == digest
    assert fp["bins"] == 64 and fp["sigma_bin"] == 1.0


def test_cb_fit_rejects_test_split():
    """反泄漏 §3.4.3：CB 只能由 train 划分构造。"""
    with pytest.raises(ValueError, match="train"):
        CenterBiasBaseline.fit(_train_inputs(), source_split="test")


def test_cb_fit_rejects_empty_train():
    empty = FixationSet(points=np.empty((0, 2)), weights=np.empty(0))
    with pytest.raises(ValueError):
        CenterBiasBaseline.fit([CenterBiasFitInput(shape=SHAPE, fixations=empty)])


def test_leakage_audit_check():
    cb = CenterBiasBaseline.fit(_train_inputs(n_images=1), source_split_hash="x" * 64)
    assert leakage_audit_check(cb, "x" * 64) == []
    problems = leakage_audit_check(cb, "y" * 64)
    assert problems and "划分" in problems[0]


# ---------------------------------------------------------------------------
# bootstrap：确定性、分层、配对、Holm
# ---------------------------------------------------------------------------


def test_bootstrap_defaults_frozen():
    cfg = EvalConfig()
    assert cfg.bootstrap_b == 2000
    assert cfg.bootstrap_seed == 20260911
    assert cfg.bootstrap_stratify_by_category is True


def test_bootstrap_ci_deterministic_and_contains_mean():
    rng = np.random.default_rng(1)
    values = rng.normal(0.5, 0.1, 50)
    a = image_bootstrap_ci(values, b=500, seed=20260911)
    b = image_bootstrap_ci(values, b=500, seed=20260911)
    assert a == b  # 固定种子 → 完全可复现
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    other = image_bootstrap_ci(values, b=500, seed=1)
    assert other["ci_low"] != a["ci_low"] or other["ci_high"] != a["ci_high"]


def test_bootstrap_stratified_by_category():
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.normal(0.2, 0.05, 30), rng.normal(0.8, 0.05, 30)])
    cats = ["webpage"] * 30 + ["poster"] * 30
    strat = image_bootstrap_ci(values, categories=cats, b=500)
    plain = image_bootstrap_ci(values, b=500)
    assert strat["stratified"] is True and plain["stratified"] is False
    # 两类均值差异大：非分层 CI 受类别构成波动影响更宽
    assert (strat["ci_high"] - strat["ci_low"]) < (plain["ci_high"] - plain["ci_low"])


def test_paired_bootstrap_win_rate_and_ci():
    rng = np.random.default_rng(3)
    n = 60
    b_vals = rng.normal(0.0, 0.1, n)
    a_vals = b_vals + 0.3 + rng.normal(0.0, 0.02, n)  # A 稳定优于 B
    res = paired_bootstrap(a_vals, b_vals, b=500)
    assert res["mean_diff"] == pytest.approx(0.3, abs=0.05)
    assert res["ci_low"] > 0  # 差值 CI 下界 > 0（S2 判据口径）
    assert res["win_rate"] == 1.0
    assert res["p_value"] < 0.05
    # 反向差值
    rev = paired_bootstrap(b_vals, a_vals, b=500)
    assert rev["ci_high"] < 0 and rev["win_rate"] == 0.0


def test_holm_correction_known_values():
    out = holm_correction([0.01, 0.02, 0.03, 0.04], alpha=0.05)
    assert out["adjusted"] == pytest.approx([0.04, 0.06, 0.06, 0.06])
    assert out["significant"] == [True, False, False, False]
    assert out["method"] == "holm"


# ---------------------------------------------------------------------------
# 管线：逐图行、排除门槛、窗口字段、CSV、基线同管线
# ---------------------------------------------------------------------------


def _cases(n=4, n_fix=40, n_viewers=5):
    cases = []
    for i in range(n):
        rng = np.random.default_rng(10 + i)
        pts = np.clip(np.stack([rng.normal(40, 10, n_fix), rng.normal(30, 8, n_fix)], axis=1), 0, [79, 59])
        cases.append(
            ImageCase(
                image_id=f"img-{i}",
                shape=SHAPE,
                fixations=build_fixation_set(pts, SHAPE),
                category=["webpage", "desktop", "mobile", "poster"][i % 4],
                block=str(i),
                n_viewers=n_viewers,
            )
        )
    return cases


def test_exclusion_flags():
    few_fix = ImageCase(image_id="f", shape=SHAPE, fixations=_fix([[5.0, 5.0]] * 4), n_viewers=5)
    assert exclusion_flag(few_fix) == "low_fixation"
    few_viewers = ImageCase(image_id="v", shape=SHAPE, fixations=_center_fixations(), n_viewers=2)
    assert exclusion_flag(few_viewers) == "low_viewers"
    ok = ImageCase(image_id="o", shape=SHAPE, fixations=_center_fixations(), n_viewers=5)
    assert exclusion_flag(ok) == ""


def test_run_window_evaluation_rows_and_csv(tmp_path):
    cases = _cases()
    preds = {c.image_id: _gaussian_map(SHAPE, 40, 30) for c in cases}
    cb = CenterBiasBaseline.fit(_train_inputs())
    rows = run_window_evaluation(
        cases, preds, cb=cb, config=CONFIG, dataset="synthetic", window="3s", model="candidate_x", model_version="v1.2"
    )
    # 7 指标/图（IG_CB, IG_U, NSS, CC, sAUC, AUC-Judd, KL；SIM 默认关闭）
    assert len(rows) == len(cases) * 7
    assert {r.metric for r in rows} == {"IG_CB", "IG_U", "NSS", "CC", "sAUC", "AUC-Judd", "KL"}
    assert all(r.window == "3s" for r in rows)  # 窗口是行主键，绝不混合
    assert all(r.dataset == "synthetic" for r in rows)

    csv_path = tmp_path / "per_image.csv"
    n = write_per_image_csv(rows, csv_path)
    assert n == len(rows)
    with open(csv_path, newline="", encoding="utf-8") as f:
        table = list(csv.reader(f))
    assert table[0] == list(
        (
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
    )
    assert len(table) == len(rows) + 1


def test_sim_included_only_when_enabled():
    case = _cases(1)[0]
    S = _gaussian_map(SHAPE, 40, 30)
    cb = CenterBiasBaseline.fit(_train_inputs())
    rows_off = evaluate_case(
        case,
        S,
        cb=cb,
        neg_pool=case.fixations,
        config=EvalConfig(),
        dataset="d",
        window="1s",
        model="m",
        model_version="v",
    )
    rows_on = evaluate_case(
        case,
        S,
        cb=cb,
        neg_pool=case.fixations,
        config=EvalConfig(include_sim=True),
        dataset="d",
        window="1s",
        model="m",
        model_version="v",
    )
    assert "SIM" not in {r.metric for r in rows_off}
    assert "SIM" in {r.metric for r in rows_on}


def test_baseline_rows_include_uniform_and_cb():
    cases = _cases(2)
    cb = CenterBiasBaseline.fit(_train_inputs())
    rows = run_baseline_rows(cases, cb, config=CONFIG, dataset="synthetic", window="1s")
    models = {r.model for r in rows}
    assert models == {"uniform", "center_bias"}  # 基线过同一管线（§4.4）


def test_prediction_validation_rejects_nan_and_negative():
    case = _cases(1)[0]
    bad = _gaussian_map(SHAPE, 40, 30)
    bad[0, 0] = np.nan
    with pytest.raises(EvalGateError):
        evaluate_case(
            case,
            bad,
            cb=None,
            neg_pool=case.fixations,
            config=CONFIG,
            dataset="d",
            window="1s",
            model="m",
            model_version="v",
        )
    neg = _gaussian_map(SHAPE, 40, 30)
    neg[5, 5] = -1.0
    with pytest.raises(EvalGateError):
        evaluate_case(
            case,
            neg,
            cb=None,
            neg_pool=case.fixations,
            config=CONFIG,
            dataset="d",
            window="1s",
            model="m",
            model_version="v",
        )


# ---------------------------------------------------------------------------
# S0 管线自检门
# ---------------------------------------------------------------------------


def test_s0_self_check_passes_with_synthetic_data():
    report = s0_self_check(EvalConfig())
    assert report["skipped"] is False
    assert report["failed"] == []
    checks = report["checks"]
    assert checks["uniform.NSS==0"]["value"] == 0.0
    assert checks["uniform.AUC-Judd==0.5"]["value"] == pytest.approx(0.5)
    assert checks["uniform.sAUC==0.5"]["value"] == pytest.approx(0.5)
    assert checks["uniform.IG_U==0"]["value"] == pytest.approx(0.0, abs=1e-12)
    assert checks["center_bias.IG_CB==0"]["value"] == pytest.approx(0.0, abs=1e-12)


def test_s0_self_check_fails_closed_on_broken_tolerance():
    """S0 不过 → 结构化错误退出（EvalGateError），不产出假成功。"""
    broken = EvalConfig(criteria=SuccessCriteria(s0_nss_tol=-1.0))  # 人为破坏容差模拟脚本缺陷
    with pytest.raises(EvalGateError) as exc:
        s0_self_check(broken)
    assert exc.value.details["reason"] == "eval_gate_failed"
    assert exc.value.exit_code != 0
    assert "uniform.NSS==0" in exc.value.details["failed"]


def test_s0_can_be_disabled_explicitly():
    cfg = EvalConfig(criteria=SuccessCriteria(s0_enabled=False))
    assert s0_self_check(cfg) == {"skipped": True}


def test_s1_to_s4_default_disabled_pending_confirmation():
    crit = EvalConfig().criteria
    assert crit.s0_enabled is True
    assert (crit.s1_enabled, crit.s2_enabled, crit.s3_enabled, crit.s4_enabled) == (False, False, False, False)
    assert "待一级总控确认" in crit.pending_confirmation_note


# ---------------------------------------------------------------------------
# 表骨架与配置指纹
# ---------------------------------------------------------------------------


def test_table_a_requires_baseline_rows():
    cells = [
        {
            "model": "candidate_x",
            "split": "test",
            "window": "3s",
            "metrics": {"IG_CB": {"mean": 0.2, "ci_low": 0.1, "ci_high": 0.3}},
        },
    ]
    table = build_table_a(cells)
    assert table["missing_baseline_rows"] == ["uniform", "center_bias"]
    assert "不完整" in table["note"]
    assert "模型预测" in table["disclaimer"]


def test_table_e_fingerprint_and_config_hash():
    cfg = EvalConfig()
    fp = {
        "protocol_version": cfg.protocol_version,
        "splits_file_hash": "s" * 64,
        "models": [{"name": "candidate_x", "version": "v1.2", "weights_sha256": ["w" * 64]}],
        "preprocessing": {"interp": "bilinear", "sigma_blur": "ueyes_1dva", "weighting": cfg.weighting},
        "numeric_guards": {"lambda": cfg.lam, "eps_rel": cfg.eps_rel},
        "log_base": cfg.log_base,
        "baseline_versions": {"cb": "cb.v1", "cb_hash": "b" * 64},
        "eval_code_version": "test",
        "bootstrap": {"B": cfg.bootstrap_b, "seed": cfg.bootstrap_seed},
        "config_hash": cfg.config_hash(),
    }
    table = build_table_e(fp)
    assert table["missing_required"] == []
    assert "不可与旧表同表比较" in table["note"]
    # config_hash 对参数敏感
    assert cfg.config_hash() != EvalConfig(lam=1e-9).config_hash()
    assert cfg.config_hash() == EvalConfig().config_hash()


# ---------------------------------------------------------------------------
# 重采样（评估网格）
# ---------------------------------------------------------------------------


def test_resample_to_density_renormalizes():
    P = synthetic.block_probability((30, 40), (5, 5, 10, 10), 0.7)
    R = resample_to_density(P, (60, 80))
    assert R.shape == (60, 80)
    assert R.sum() == pytest.approx(1.0, abs=1e-12)
