"""评估协议配置（benchmark-protocol.md §6.0/§7.2/§8.3）。

全部口径参数集中于此并进入 config_hash（表 E 配置指纹）：σ 换算、λ/ε 防护常数、
log2 约定、加权方案、bootstrap 参数、S0-S4 判据开关。

S1-S4 成功判据为 R2 研究**建议值，待一级总控确认**——实现为可配置项且默认关闭，
不硬编码为通过标准；S0 管线自检默认开启（不过则失败退出，不产出假成功）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

#: 协议版本（§7.3 表 E 主键之一；口径任何冻结项变更必须递增）
PROTOCOL_VERSION = "r2-v0.1"

#: bootstrap 默认参数（§7.2：B=2000、固定种子 20260911、按类别分层）
BOOTSTRAP_B_DEFAULT = 2000
BOOTSTRAP_SEED_DEFAULT = 20260911

#: 数值防护常数（§6.0 冻结）
LAMBDA_DEFAULT = 1e-8
EPS_REL_DEFAULT = 1e-12
LOG_BASE = 2  # bits/fixation，Tuebingen 基准惯例

WINDOWS_DEFAULT: tuple[str, ...] = ("1s", "3s", "7s")


@dataclass(frozen=True)
class SuccessCriteria:
    """阶段性成功判据（§8.3：全部为建议值，待一级总控确认）。

    S0 管线自检默认开启（脚本自检门，不属于"模型是否合格"判断）；
    S1-S4 默认关闭——**不得硬编码为通过标准**，一级总控确认后经配置开启。
    """

    # S0：均匀基线 NSS≈0 / AUC≈0.5 / IG_U≈0；CB 的 IG_CB≈0（定义自洽）。不过 → 脚本缺陷，禁止出报告。
    s0_enabled: bool = True
    s0_nss_tol: float = 1e-9
    s0_auc_tol: float = 1e-9
    s0_ig_tol: float = 1e-9

    # S1 基线有效性：CB 的 NSS/sAUC 配对差值 CI 下界 > 0（vs 均匀）。待一级总控确认，默认关闭。
    s1_enabled: bool = False
    s1_ci_lower_min: float = 0.0

    # S2 候选及格线：每窗口 IG_CB 与 NSS 配对差值（vs CB）95% CI 下界 > 0。待一级总控确认，默认关闭。
    s2_enabled: bool = False
    s2_ci_lower_min: float = 0.0

    # S3 跨域一致性（定性）。待一级总控确认，默认关闭。
    s3_enabled: bool = False

    # S4 相对 UMSI++ 水平（暂缓定量）。留待一级总控裁决，默认关闭。
    s4_enabled: bool = False
    s4_ig_ratio_min: float | None = None  # 例如 0.9 = "IG 达到其 90%"，未裁决前不设数

    pending_confirmation_note: str = (
        "S1-S4 阈值来自 R2 研究建议（benchmark-protocol §8.3），待一级总控确认后启用；默认关闭，不作为通过标准"
    )


@dataclass(frozen=True)
class EvalConfig:
    """评估口径配置（进入表 E 配置指纹与 config_hash）。"""

    dataset: str = "synthetic"
    protocol_version: str = PROTOCOL_VERSION
    windows: tuple[str, ...] = WINDOWS_DEFAULT
    split: str = "test"

    # 真值模糊（§6.0：σ=1° 视角；UEyes σ = 40 × max(W/1920, H/1200) px）
    sigma_rule: str = "ueyes_1dva"  # ueyes_1dva | fixed_px
    sigma_fixed_px: float | None = None  # sigma_rule=fixed_px 时使用（如 FiWI ≈26）
    blur_mode: str = "reflect"  # §6.9：reflect 或零填充后归一化，二选一冻结（默认 reflect）
    blur_truncate: float = 4.0  # 高斯核截断半径 ≥ 4σ

    # 数值防护（§6.0 冻结；防护方案一经冻结写入配置哈希）
    lam: float = LAMBDA_DEFAULT
    eps_rel: float = EPS_REL_DEFAULT
    log_base: int = LOG_BASE

    # 真值加权（§6.0：默认计数加权；时长加权作敏感性分析，两者不得混表）
    weighting: str = "count"  # count | duration

    # sAUC 负样本（§6.4：默认全部；大划分可抽样 M=10×|Φ|，记录种子）
    sauc_negative_multiplier: int | None = None  # None=全部；10=M=10×|Φ| 抽样
    sampling_seed: int = BOOTSTRAP_SEED_DEFAULT

    # AUC-Judd 负样本抽样（§6.5：不抽样为默认）
    aucjudd_negative_multiplier: int | None = None

    # SIM（§6.7：可选，默认关闭——与 IG/CC 信息重叠）
    include_sim: bool = False

    # CB 基线构造（§4.2：bin 64×64、σ_bin=1）
    cb_bins: int = 64
    cb_sigma_bin: float = 1.0

    # bootstrap（§7.2）
    bootstrap_b: int = BOOTSTRAP_B_DEFAULT
    bootstrap_seed: int = BOOTSTRAP_SEED_DEFAULT
    bootstrap_stratify_by_category: bool = True
    ci_level: float = 0.95

    criteria: SuccessCriteria = field(default_factory=SuccessCriteria)

    def validate(self) -> None:
        problems: list[str] = []
        if self.weighting not in ("count", "duration"):
            problems.append(f"weighting 必须是 count|duration，得到 {self.weighting!r}")
        if self.sigma_rule not in ("ueyes_1dva", "fixed_px"):
            problems.append(f"sigma_rule 必须是 ueyes_1dva|fixed_px，得到 {self.sigma_rule!r}")
        if self.sigma_rule == "fixed_px" and not (self.sigma_fixed_px and self.sigma_fixed_px > 0):
            problems.append("sigma_rule=fixed_px 时 sigma_fixed_px 必须为正数")
        if self.blur_mode not in ("reflect", "zero"):
            problems.append(f"blur_mode 必须是 reflect|zero（二选一冻结），得到 {self.blur_mode!r}")
        if self.log_base != 2:
            problems.append("log_base 冻结为 2（bits/fixation）；若用 ln 必须换算或明确标注 nats")
        if not (0.0 <= self.lam < 1.0):
            problems.append(f"λ 必须在 [0,1)，得到 {self.lam}")
        if self.bootstrap_b <= 0:
            problems.append("bootstrap_b 必须为正")
        if not (0.0 < self.ci_level < 1.0):
            problems.append("ci_level 必须在 (0,1)")
        if problems:
            raise ValueError("EvalConfig 无效：" + "；".join(problems))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["windows"] = list(self.windows)
        return d

    def config_hash(self) -> str:
        """全部口径参数的合成哈希（表 E；任一项不同 → 结果不可与旧表同表比较）。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
