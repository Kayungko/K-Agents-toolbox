"""contracts：schema 与校验（C1 持有；跨线协作唯一接口面）。

- backend.py：G0 冻结签名（BackendInfo/WeightRef/PredictionResult/ResolvedProfile/Backend Protocol）
- request.py：AnalyzeRequest v1（含 RegionSpec 与几何校验）
- regions.py：独立 regions 文件（必须携带图片 SHA-256）
- analysis.py：analysis v1（§4 计算结果与证据）
- comparison.py：comparison v1（§6 比较结果）
- review.py：review.json（Agent 语义评审，CLI 只校验与呈现）
"""

from .analysis import (
    ANALYSIS_SCHEMA_VERSION,
    EVIDENCE_TYPE,
    METRICS_VERSION,
    AnalysisRecord,
    ArtifactRecord,
    load_analysis,
)
from .backend import (
    CAPABILITIES,
    CLEARED_FOR,
    DEVICES,
    NATIVE_SEMANTICS,
    Backend,
    BackendInfo,
    BackendRegistry,
    PredictionResult,
    ResolvedProfile,
    WeightRef,
)
from .comparison import (
    COMPARISON_SCHEMA_VERSION,
    ComparisonRecord,
    CompatibilityReport,
    build_comparison,
    check_compatibility,
)
from .regions import REGIONS_SCHEMA_VERSION, RegionsFile, load_regions_file
from .request import (
    REQUEST_SCHEMA_VERSION,
    AnalyzeRequest,
    PolygonGeometry,
    RectGeometry,
    RegionSpec,
    load_analyze_request,
)
from .review import EVIDENCE_TYPES, REVIEW_SCHEMA_VERSION, ReviewRecord, load_review

__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "CAPABILITIES",
    "CLEARED_FOR",
    "COMPARISON_SCHEMA_VERSION",
    "DEVICES",
    "EVIDENCE_TYPE",
    "EVIDENCE_TYPES",
    "METRICS_VERSION",
    "NATIVE_SEMANTICS",
    "REGIONS_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "REVIEW_SCHEMA_VERSION",
    "AnalysisRecord",
    "AnalyzeRequest",
    "ArtifactRecord",
    "Backend",
    "BackendInfo",
    "BackendRegistry",
    "CompatibilityReport",
    "ComparisonRecord",
    "PolygonGeometry",
    "PredictionResult",
    "RectGeometry",
    "RegionSpec",
    "RegionsFile",
    "ResolvedProfile",
    "ReviewRecord",
    "WeightRef",
    "build_comparison",
    "check_compatibility",
    "load_analysis",
    "load_analyze_request",
    "load_regions_file",
    "load_review",
]
