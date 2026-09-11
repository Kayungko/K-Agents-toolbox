"""backends：显著性模型适配层（C2 独占；经 contracts/backend.py 冻结接口协作）.

模块清单（implementation-plan.md 所有权表 C2 行）：

- :mod:`registry` — profile 登记与解析（``BackendRegistry`` Protocol：
  ``list_profiles`` / ``resolve_profile`` / ``get_backend`` / ``doctor_report``）；
  首个登记 ``foveacast-onnx-3s-v1``；许可状态与 CC-BY-4.0 署名元数据。
- :mod:`doctor` — 依赖/设备/权重哈希/许可探测，结构化返回（cli 消费）。
- :mod:`weights` — GitHub Release 下载 + sha256 校验 + PROVENANCE 记录；
  显式安装步骤 ``python -m ui_attention.backends``。
- :mod:`onnx_foveacast` — foveacast ONNX 适配层（describe()/predict()）。
- :mod:`runtime_probe` — 进程峰值内存实测助手。
- :mod:`measure_runtime` — 运行实测驱动（冷启动/热推理/峰值内存/ONNX 元数据
  → ``model-cache/foveacast/runtime-measurements.json``）。
- :mod:`errors` — 后端专属异常类型（错误码全部复用 C1 ``ui_attention.errors``）。
"""

from __future__ import annotations

from .deepgaze_iie import DeepGazeIIEBackend
from .doctor import DoctorCheck, DoctorReport, run_doctor
from .errors import (
    BackendError,
    DownloadFailureError,
    ImageRejectedError,
    InferenceFailureError,
    InvalidDensityError,
    MemoryExhaustedError,
    ProfileRejectedError,
    WeightNotReadyError,
)
from .onnx_foveacast import FoveacastOnnxBackend
from .registry import (
    ATTRIBUTION,
    DEEPGAZE_IIE_PROFILE,
    FOVEACAST_3S_PROFILE,
    LICENSE_RECORD,
    ProfileRegistration,
    doctor_report,
    get_backend,
    license_record,
    list_profiles,
    resolve_profile,
    verify_profile_integrity,
)
from .weights import (
    WeightFetchResult,
    default_cache_dir,
    fetch_weight,
    sha256_of_file,
    verify_cached_weight,
    write_provenance,
)

__all__ = [
    "ATTRIBUTION",
    "DEEPGAZE_IIE_PROFILE",
    "FOVEACAST_3S_PROFILE",
    "LICENSE_RECORD",
    "BackendError",
    "DeepGazeIIEBackend",
    "DoctorCheck",
    "DoctorReport",
    "DownloadFailureError",
    "FoveacastOnnxBackend",
    "ImageRejectedError",
    "InferenceFailureError",
    "InvalidDensityError",
    "MemoryExhaustedError",
    "ProfileRegistration",
    "ProfileRejectedError",
    "WeightFetchResult",
    "WeightNotReadyError",
    "default_cache_dir",
    "doctor_report",
    "fetch_weight",
    "get_backend",
    "license_record",
    "list_profiles",
    "resolve_profile",
    "run_doctor",
    "sha256_of_file",
    "verify_cached_weight",
    "verify_profile_integrity",
    "write_provenance",
]
