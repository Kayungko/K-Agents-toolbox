"""Environment doctor probes (C2 scope).

Pure functions/classes returning structured results — the CLI shell, JSON
envelope and exit-code branching belong to C1 (``cli.py`` consumes
``registry.doctor_report()`` / :func:`run_doctor`; every ``fail`` check carries
an ``error_code`` from the C1 catalog mapping to exit 3 semantics:
「后端、依赖、权重或许可配置未就绪」).

Probes (data-contract §1 doctor 行；runtime-feasibility §3 doctor 映射):

1. 依赖版本：python（pyproject 口径 >=3.12,<3.13）、numpy、pillow、
   onnxruntime（foveacast 要求 >=1.17，R1 已验证）+ ORT provider 列表。
2. 设备：device=cpu（profile 冻结要求）；若 ``nvidia-smi`` 存在则实测空闲
   显存并记录，但**不启用 GPU**（路线 A CPU 先行，technical-design §3；
   onnxruntime-gpu 不装）。``min_free_vram_mb`` 门槛仅对 cuda 设备要求生效。
3. 权重存在性 + sha256 校验（== 登记值；不匹配 → MODEL_NOT_READY）。
4. 许可记录输出（license_status + 证据 + G3/G4/G8 缺口 + CC-BY-4.0 署名），
   不打印任何凭据。

模型下载是独立的显式安装步骤（``python -m ui_attention.backends``），
doctor 只检查、不下载。
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import registry
from .errors import BackendError, WeightNotReadyError
from .weights import cache_root, verify_cached_weight

__all__ = [
    "DoctorCheck",
    "DoctorReport",
    "probe_dependencies",
    "probe_device",
    "probe_license",
    "probe_weights",
    "run_doctor",
]

STATUS_OK = "ok"
STATUS_FAIL = "fail"
STATUS_WARN = "warn"
STATUS_SKIP = "skip"

#: pyproject.toml requires-python ">=3.12,<3.13" (C1-owned pin).
PYTHON_MIN = (3, 12)
PYTHON_MAX_EXCLUSIVE = (3, 13)
#: foveacast-training requirement: onnxruntime >= 1.17 (R1 model-candidates §C2).
ONNXRUNTIME_MIN = (1, 17)
_NVIDIA_SMI_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class DoctorCheck:
    """One structured probe result."""

    name: str
    status: str  # ok | fail | warn | skip
    details: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None  # C1 ErrorCode value when status == fail

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "details": dict(self.details),
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class DoctorReport:
    """Aggregated structured result consumed by C1 cli (exit 3 on any fail)."""

    ok: bool
    profile_name: str
    checks: tuple[DoctorCheck, ...]
    error_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "profile_name": self.profile_name,
            "checks": [c.to_dict() for c in self.checks],
            "error_codes": list(self.error_codes),
        }


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in version.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def probe_dependencies() -> DoctorCheck:
    """Dependency versions + ORT providers. Failure → MODEL_NOT_READY (exit 3)."""
    details: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    problems: list[str] = []

    py = sys.version_info[:2]
    if not (PYTHON_MIN <= py < PYTHON_MAX_EXCLUSIVE):
        problems.append(
            f"python {details['python']} 不在 pyproject 约束 >=3.12,<3.13 内"
        )

    for dist in ("numpy", "pillow", "onnxruntime"):
        try:
            details[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            details[dist] = None
            problems.append(f"依赖缺失: {dist}")

    ort_version = details.get("onnxruntime")
    if ort_version and _version_tuple(ort_version) < ONNXRUNTIME_MIN:
        problems.append(
            f"onnxruntime {ort_version} < 最低要求 {'.'.join(map(str, ONNXRUNTIME_MIN))}"
        )

    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
        details["onnxruntime_providers"] = providers
        if "CPUExecutionProvider" not in providers:
            problems.append("onnxruntime 无 CPUExecutionProvider")
    except ImportError as exc:
        details["onnxruntime_providers"] = None
        problems.append(f"onnxruntime 不可导入: {type(exc).__name__}")

    # scipy is C1's eval-path dependency; informational only (not a backend gate).
    try:
        details["scipy"] = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        details["scipy"] = None

    if problems:
        return DoctorCheck(
            name="dependencies",
            status=STATUS_FAIL,
            details={**details, "problems": problems},
            error_code="MODEL_NOT_READY",
        )
    return DoctorCheck(name="dependencies", status=STATUS_OK, details=details)


def probe_device(min_free_vram_mb: int | None = None) -> DoctorCheck:
    """Device probe: CPU execution; GPU is measured (if visible) but never enabled.

    ``min_free_vram_mb`` gates only cuda-device profiles (data-contract §3
    device_requirements); the foveacast profile requires cpu with
    ``min_free_vram_mb=None``, so GPU findings here are informational.
    """
    details: dict[str, Any] = {
        "device": "cpu",
        "gpu_enabled": False,
        "cpu_logical_processors": os.cpu_count(),
    }
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        details["nvidia_smi"] = None
        details["gpus"] = []
        return DoctorCheck(name="device", status=STATUS_OK, details=details)

    details["nvidia_smi"] = nvidia_smi
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        details["gpus"] = []
        details["gpu_query_error"] = type(exc).__name__
        return DoctorCheck(
            name="device",
            status=STATUS_WARN,
            details={**details, "note": "nvidia-smi 存在但查询失败; CPU 路线不受影响"},
        )

    if completed.returncode != 0:
        details["gpus"] = []
        details["gpu_query_error"] = f"exit {completed.returncode}"
        return DoctorCheck(
            name="device",
            status=STATUS_WARN,
            details={**details, "note": "nvidia-smi 查询非零退出; CPU 路线不受影响"},
        )

    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "free_vram_mb": int(parts[2]),
                    "total_vram_mb": int(parts[3]),
                }
            )
        except ValueError:
            continue
    details["gpus"] = gpus
    status = STATUS_OK
    if min_free_vram_mb is not None and not any(
        g["free_vram_mb"] >= min_free_vram_mb for g in gpus
    ):
        # Only meaningful for cuda profiles; foveacast (cpu, None) never gates here.
        status = STATUS_WARN
        details["vram_warning"] = (
            f"无 GPU 满足 min_free_vram_mb={min_free_vram_mb}（当前 profile 为 CPU，"
            "仅记录）"
        )
    return DoctorCheck(name="device", status=status, details=details)


def probe_weights(
    profile_name: str = registry.FOVEACAST_3S_PROFILE,
    cache_dir: Path | str | None = None,
) -> DoctorCheck:
    """Weight existence + sha256 == registered value (offline, no download).

    Failure → ``MODEL_NOT_READY`` (exit 3), details.reason per weight.
    cache_dir 缺省时按登记项的 ``cache_subdir`` 解析（foveacast/deepgaze）。
    """
    registration = registry.registration_for(profile_name)
    if cache_dir is None:
        cache_dir = cache_root() / registration.cache_subdir
    else:
        cache_dir = Path(cache_dir)
    entries: list[dict[str, Any]] = []
    failure: WeightNotReadyError | None = None
    for ref in registration.profile.weights:
        try:
            result = verify_cached_weight(ref, cache_dir)
            entries.append(
                {
                    "name": ref.name,
                    "status": STATUS_OK,
                    "path": str(result.path),
                    "sha256": result.sha256,
                    "size_bytes": result.size_bytes,
                    "expected_sha256": ref.sha256,
                }
            )
        except WeightNotReadyError as exc:
            failure = exc
            entries.append(
                {
                    "name": ref.name,
                    "status": STATUS_FAIL,
                    "reason": exc.details.get("reason"),
                    "expected_sha256": ref.sha256,
                    "actual_sha256": exc.details.get("actual_sha256"),
                    "expected_path": str(cache_dir / ref.name),
                }
            )
    details: dict[str, Any] = {"cache_dir": str(cache_dir), "weights": entries}
    provenance = cache_dir / "PROVENANCE.json"
    details["provenance_file"] = str(provenance) if provenance.is_file() else None
    if failure is not None:
        return DoctorCheck(
            name="weights",
            status=STATUS_FAIL,
            details=details,
            error_code=failure.code.value,
        )
    return DoctorCheck(name="weights", status=STATUS_OK, details=details)


def probe_license(
    profile_name: str = registry.FOVEACAST_3S_PROFILE,
    required_cleared_for: str = "internal-eval",
) -> DoctorCheck:
    """License verification record output (no credentials, ever).

    ``ok`` when the registration's ``cleared_for`` covers the requested usage
    scope; G3/G4/G8 商用链残留缺口如实记录在 details（内部评估已放行，打包/
    商用前必须关闭）。缺口未放行 → ``LICENSE_NOT_CLEARED`` (exit 3).
    """
    registration = registry.registration_for(profile_name)
    record = registration.license_record
    status_value = record["status"].get("cleared_for")
    order = {"internal-eval": 0, "packaged": 1}
    cleared = order.get(status_value, -1) >= order.get(required_cleared_for, 99)
    details = {
        "license_status": dict(record["status"]),
        "evidence": record.get("evidence"),
        "gaps_detail": record.get("gaps_detail"),
        "obligation": record.get("obligation"),
        "attribution": record.get("attribution"),
        "required_cleared_for": required_cleared_for,
    }
    if not cleared:
        return DoctorCheck(
            name="license",
            status=STATUS_FAIL,
            details=details,
            error_code="LICENSE_NOT_CLEARED",
        )
    return DoctorCheck(name="license", status=STATUS_OK, details=details)


def run_doctor(
    profile_name: str = registry.FOVEACAST_3S_PROFILE,
    cache_dir: Path | str | None = None,
) -> DoctorReport:
    """Run all probes for a registered profile; structured, side-effect free.

    An unregistered profile yields a fail report (``PROFILE_NOT_REGISTERED``,
    exit 3) instead of raising, so the CLI can always emit an envelope.
    """
    try:
        registry.registration_for(profile_name)
    except BackendError as exc:
        check = DoctorCheck(
            name="profile",
            status=STATUS_FAIL,
            details={"requested": profile_name, **exc.details},
            error_code=exc.code.value,
        )
        return DoctorReport(
            ok=False,
            profile_name=str(profile_name),
            checks=(check,),
            error_codes=(exc.code.value,),
        )

    checks = (
        probe_dependencies(),
        probe_device(),
        probe_weights(profile_name, cache_dir),
        probe_license(profile_name),
    )
    error_codes = tuple(c.error_code for c in checks if c.status == STATUS_FAIL and c.error_code)
    return DoctorReport(
        ok=not error_codes,
        profile_name=profile_name,
        checks=checks,
        error_codes=error_codes,
    )
