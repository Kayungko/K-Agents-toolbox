"""Unit tests for backends/doctor.py (C2).

Fake nvidia-smi via monkeypatch; real-weight probes are skipif-guarded so the
suite runs green before the explicit download step.
"""

import json
from pathlib import Path

import pytest

from ui_attention.backends import registry
from ui_attention.backends.doctor import (
    probe_dependencies,
    probe_device,
    probe_license,
    probe_weights,
    run_doctor,
)
from ui_attention.backends.weights import default_cache_dir

WEIGHT_PATH = default_cache_dir() / registry.FOVEACAST_3S_WEIGHT.name
HAS_WEIGHTS = WEIGHT_PATH.is_file()
needs_weights = pytest.mark.skipif(
    not HAS_WEIGHTS, reason="real weights not cached (run python -m ui_attention.backends)"
)


class _FakeCompleted:
    returncode = 0
    stdout = "0, NVIDIA GeForce RTX 4070, 7207, 12282\n"
    stderr = ""


def _patch_nvidia(monkeypatch: pytest.MonkeyPatch, run_impl=None) -> None:
    from ui_attention.backends import doctor

    monkeypatch.setattr(
        doctor.shutil,
        "which",
        lambda name: "C:/fake/nvidia-smi" if name == "nvidia-smi" else None,
    )
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        run_impl or (lambda *args, **kwargs: _FakeCompleted()),
    )


# ---------------------------------------------------------------------------
# dependencies
# ---------------------------------------------------------------------------


def test_probe_dependencies_ok_in_bootstrapped_venv():
    check = probe_dependencies()
    assert check.status == "ok", check.details
    assert check.error_code is None
    details = check.details
    assert details["python"].startswith("3.12.")
    for dist in ("numpy", "pillow", "onnxruntime"):
        assert details[dist], f"{dist} version missing"
    assert "CPUExecutionProvider" in details["onnxruntime_providers"]


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------


def test_probe_device_cpu_only(monkeypatch: pytest.MonkeyPatch):
    from ui_attention.backends import doctor

    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    check = probe_device()
    assert check.status == "ok"
    assert check.details["device"] == "cpu"
    assert check.details["gpu_enabled"] is False
    assert check.details["gpus"] == []
    assert check.details["nvidia_smi"] is None


def test_probe_device_records_gpu_but_never_enables_it(monkeypatch: pytest.MonkeyPatch):
    _patch_nvidia(monkeypatch)
    check = probe_device()
    assert check.status == "ok"
    assert check.details["device"] == "cpu"
    assert check.details["gpu_enabled"] is False
    gpu = check.details["gpus"][0]
    assert gpu["name"] == "NVIDIA GeForce RTX 4070"
    assert gpu["free_vram_mb"] == 7207
    assert gpu["total_vram_mb"] == 12282


def test_probe_device_nvidia_query_failure_is_warn(monkeypatch: pytest.MonkeyPatch):
    def _raise(*args: object, **kwargs: object) -> None:
        raise OSError("nvidia-smi exploded (simulated)")

    _patch_nvidia(monkeypatch, run_impl=_raise)
    check = probe_device()
    assert check.status == "warn"
    assert check.error_code is None
    assert check.details["gpu_query_error"] == "OSError"


def test_probe_device_nvidia_nonzero_exit_is_warn(monkeypatch: pytest.MonkeyPatch):
    class _Failed:
        returncode = 6
        stdout = ""
        stderr = "boom"

    _patch_nvidia(monkeypatch, run_impl=lambda *a, **k: _Failed())
    check = probe_device()
    assert check.status == "warn"
    assert check.details["gpu_query_error"] == "exit 6"


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def test_probe_weights_missing_structured(tmp_path: Path):
    check = probe_weights(cache_dir=tmp_path)
    assert check.status == "fail"
    assert check.error_code == "MODEL_NOT_READY"
    entry = check.details["weights"][0]
    assert entry["reason"] == "missing_file"
    assert entry["expected_sha256"] == registry.FOVEACAST_3S_WEIGHT.sha256


def test_probe_weights_hash_mismatch_structured(tmp_path: Path):
    (tmp_path / registry.FOVEACAST_3S_WEIGHT.name).write_bytes(b"corrupt-payload")
    check = probe_weights(cache_dir=tmp_path)
    assert check.status == "fail"
    assert check.error_code == "MODEL_NOT_READY"
    assert check.details["weights"][0]["reason"] == "sha256_mismatch"


@needs_weights
def test_probe_weights_ok_on_real_cache():
    check = probe_weights()
    assert check.status == "ok"
    entry = check.details["weights"][0]
    assert entry["sha256"] == entry["expected_sha256"]
    assert entry["size_bytes"] == registry.FOVEACAST_3S_WEIGHT.size_bytes
    assert check.details["provenance_file"] is not None


# ---------------------------------------------------------------------------
# license
# ---------------------------------------------------------------------------


def test_probe_license_internal_eval_ok():
    check = probe_license()
    assert check.status == "ok"
    assert check.error_code is None
    status = check.details["license_status"]
    assert status["gaps"] == ["G3", "G4", "G8"]
    assert status["cleared_for"] == "internal-eval"
    attribution = check.details["attribution"]
    assert any("10.1145/3544548.3581096" in a for a in attribution)
    assert any("10.1016/j.neunet.2020.05.004" in a for a in attribution)


def test_probe_license_packaged_scope_not_cleared():
    check = probe_license(required_cleared_for="packaged")
    assert check.status == "fail"
    assert check.error_code == "LICENSE_NOT_CLEARED"


# ---------------------------------------------------------------------------
# aggregated report
# ---------------------------------------------------------------------------


def test_run_doctor_unregistered_profile_reports_not_raises():
    report = run_doctor("local-static-v1")
    assert report.ok is False
    assert "PROFILE_NOT_REGISTERED" in report.error_codes
    assert report.checks[0].name == "profile"


def test_run_doctor_weights_missing_gates_report(tmp_path: Path):
    report = run_doctor(cache_dir=tmp_path)
    assert report.ok is False
    assert "MODEL_NOT_READY" in report.error_codes
    statuses = {c.name: c.status for c in report.checks}
    assert statuses["weights"] == "fail"
    assert statuses["dependencies"] == "ok"
    assert statuses["license"] == "ok"


@needs_weights
def test_run_doctor_ok_on_real_environment():
    report = run_doctor()
    assert report.ok is True, report.to_dict()
    assert report.error_codes == ()
    assert {c.name for c in report.checks} == {"dependencies", "device", "weights", "license"}
    payload = report.to_dict()
    json.dumps(payload)  # must be JSON-serializable for the C1 envelope
    assert payload["profile_name"] == registry.FOVEACAST_3S_PROFILE


@needs_weights
def test_registry_doctor_report_entry_point():
    payload = registry.doctor_report()
    assert payload["ok"] is True
    assert isinstance(payload["checks"], list) and len(payload["checks"]) == 4


@pytest.mark.parametrize(
    "profile_name",
    ["foveacast-onnx-1s-v1", "foveacast-onnx-7s-v1"],
)
def test_run_doctor_window_variants(profile_name):
    weight_name = registry.registration_for(profile_name).profile.weights[0].name
    if not (default_cache_dir() / weight_name).is_file():
        pytest.skip(f"{weight_name} not cached (run python -m ui_attention.backends)")
    report = run_doctor(profile_name)
    assert report.ok is True, report.to_dict()
    assert report.profile_name == profile_name


def test_probe_license_deepgaze_internal_eval_ok_but_packaged_blocked():
    check = probe_license(registry.DEEPGAZE_IIE_PROFILE)
    assert check.status == "ok"  # G1' 批准仅限 internal-eval
    assert check.details["license_status"]["gaps"] == ["G1", "G2", "G5"]
    blocked = probe_license(registry.DEEPGAZE_IIE_PROFILE, required_cleared_for="packaged")
    assert blocked.status == "fail"
    assert blocked.error_code == "LICENSE_NOT_CLEARED"


def test_run_doctor_deepgaze_profile():
    from ui_attention.backends.weights import cache_root

    registration = registry.registration_for(registry.DEEPGAZE_IIE_PROFILE)
    cache = cache_root() / registration.cache_subdir
    if not all((cache / w.name).is_file() for w in registration.profile.weights):
        pytest.skip("deepgaze weights not cached (internal-eval line)")
    report = run_doctor(registry.DEEPGAZE_IIE_PROFILE)
    assert report.ok is True, report.to_dict()
    weights_check = next(c for c in report.checks if c.name == "weights")
    assert weights_check.details["cache_dir"] == str(cache)
