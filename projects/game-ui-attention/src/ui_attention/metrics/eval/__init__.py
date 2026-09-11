"""metrics.eval：公开眼动数据评估脚本（R2 benchmark-protocol 的唯一实现）。

边界（benchmark-protocol §0）：
- 本模块只在公开 UI 眼动数据口径下评估显著性后端；**游戏 UI 预测有效性保持未验证**；
- 公开数据结果不得外推为游戏 UI 结论；任何单一指标数值不得称为"预测准确率"；
- 当前仅以合成数据自测（S0 管线自检门）——UEyes 下载待一级总控批准，禁止下载数据集；
- S1-S4 成功判据为建议值（可配置、默认关闭），待一级总控确认，不硬编码为通过标准；
- 与 UEyes 官方 eval 实现的口径差异见协议 §6.8：复现分数不得与论文表格/官方输出混表。

模块地图：
- config.py：口径参数 + SuccessCriteria（S0-S4）+ config_hash（表 E）
- density.py：to_density 防护原语（λ=1e-8、全零→均匀+degenerate、常数图 flag）
- blur.py：σ=1° 换算（UEyes σ=40×max(W/1920,H/1200)px）、gaussian reflect
- groundtruth.py：注视点→δ 图/模糊 F/离散 F̂（floor 取整、越界丢弃计数）
- metrics.py：IG/NSS/CC/sAUC/AUC-Judd/KL(F‖S)/SIM
- baselines.py：均匀 U + 中心偏置 CB（train 划分 2D 直方图构造，版本化+哈希）
- bootstrap.py：图像级 bootstrap（B=2000、种子 20260911、按类别分层）、配对差值 CI+胜率、Holm
- tables.py：表 A-E 生成骨架（Markdown/JSON 双格式）
- pipeline.py：逐图 CSV、窗口隔离、基线同管线、S0 自检门
"""

from .baselines import (
    CB_VERSION,
    CenterBiasBaseline,
    CenterBiasFitInput,
    leakage_audit_check,
    pooled_other_fixations,
    uniform_baseline,
)
from .blur import gaussian_blur, sigma_px_for_image, ueyes_sigma_px
from .bootstrap import PARTICIPANT_LIMITATION, holm_correction, image_bootstrap_ci, paired_bootstrap
from .config import (
    BOOTSTRAP_B_DEFAULT,
    BOOTSTRAP_SEED_DEFAULT,
    LAMBDA_DEFAULT,
    PROTOCOL_VERSION,
    EvalConfig,
    SuccessCriteria,
)
from .density import to_density
from .groundtruth import FixationSet, blurred_truth, build_fixation_set, delta_map, discrete_truth, pixel_indices
from .interp import resample_bilinear, resample_to_density
from .metrics import MetricResult, auc_judd, cc, information_gain, kl_divergence, nss, sauc, similarity
from .pipeline import (
    CSV_COLUMNS,
    EvalGateError,
    ImageCase,
    PerImageRow,
    evaluate_case,
    exclusion_flag,
    run_baseline_rows,
    run_window_evaluation,
    s0_self_check,
    write_per_image_csv,
)
from .tables import (
    REPORT_DISCLAIMER,
    build_table_a,
    build_table_b,
    build_table_c,
    build_table_d,
    build_table_e,
)

__all__ = [
    "BOOTSTRAP_B_DEFAULT",
    "BOOTSTRAP_SEED_DEFAULT",
    "CB_VERSION",
    "CSV_COLUMNS",
    "LAMBDA_DEFAULT",
    "PARTICIPANT_LIMITATION",
    "PROTOCOL_VERSION",
    "REPORT_DISCLAIMER",
    "CenterBiasBaseline",
    "CenterBiasFitInput",
    "EvalConfig",
    "EvalGateError",
    "FixationSet",
    "ImageCase",
    "MetricResult",
    "PerImageRow",
    "SuccessCriteria",
    "auc_judd",
    "blurred_truth",
    "build_fixation_set",
    "build_table_a",
    "build_table_b",
    "build_table_c",
    "build_table_d",
    "build_table_e",
    "cc",
    "delta_map",
    "discrete_truth",
    "evaluate_case",
    "exclusion_flag",
    "gaussian_blur",
    "holm_correction",
    "image_bootstrap_ci",
    "information_gain",
    "kl_divergence",
    "leakage_audit_check",
    "nss",
    "paired_bootstrap",
    "pixel_indices",
    "pooled_other_fixations",
    "resample_bilinear",
    "resample_to_density",
    "run_baseline_rows",
    "run_window_evaluation",
    "s0_self_check",
    "sauc",
    "sigma_px_for_image",
    "similarity",
    "to_density",
    "ueyes_sigma_px",
    "uniform_baseline",
    "write_per_image_csv",
]
