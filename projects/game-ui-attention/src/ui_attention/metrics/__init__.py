"""metrics：AOI 统计路径的概率归一化与区域指标（technical-design §7）。

eval/ 子模块为公开数据评估路径（benchmark-protocol R2 口径），
两条路径的数值防护常数与口径不得互相混用。
"""

from .probability import (
    DENSITY_SUM_TOLERANCE,
    bilinear_resample,
    log_density_to_probability,
    logsumexp,
    normalize_probability,
    resample_probability,
    resample_to_original,
    validate_probability,
)
from .regions import (
    compare_region_rows,
    delta_pp,
    region_statistics,
    uniform_sanity_check,
    union_statistics,
)

__all__ = [
    "DENSITY_SUM_TOLERANCE",
    "bilinear_resample",
    "compare_region_rows",
    "delta_pp",
    "log_density_to_probability",
    "logsumexp",
    "normalize_probability",
    "region_statistics",
    "resample_probability",
    "resample_to_original",
    "union_statistics",
    "uniform_sanity_check",
    "validate_probability",
]
