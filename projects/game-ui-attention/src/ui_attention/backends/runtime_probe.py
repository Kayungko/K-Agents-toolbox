"""Process memory probing helpers (C2 scope).

``peak_mem_mb`` in the frozen ``PredictionResult.runtime`` dict must be a
measured value. On Windows the true process peak is ``PeakWorkingSetSize``
via ``K32GetProcessMemoryInfo`` (stdlib ctypes — no third-party dependency,
no global environment change). On unsupported platforms the helpers return
``None``; C1 ``PredictionResult.validate()`` accepts ``peak_mem_mb=None``.

Semantics (recorded for analysis.json consumers): the value is the cumulative
peak working set of the whole CLI process at measurement time (includes the
ONNX session, weight mapping and all Python allocations), not an isolated
per-call allocation delta. Complementary tracemalloc-based Python-side peaks
are recorded by :mod:`ui_attention.backends.measure_runtime`.
"""

from __future__ import annotations

import ctypes
import os

__all__ = ["current_working_set_mb", "peak_working_set_mb"]

if os.name == "nt":
    from ctypes import wintypes

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

else:  # pragma: no cover - project target platform is Windows

    class _PROCESS_MEMORY_COUNTERS:  # type: ignore[no-redef]
        """Placeholder so module import succeeds on non-Windows platforms."""


def _query_counters() -> _PROCESS_MEMORY_COUNTERS | None:
    if os.name != "nt":
        return None
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    # Explicit prototypes: the pseudo-handle from GetCurrentProcess is 64-bit;
    # without restype/argtypes ctypes truncates it to a 32-bit int and the
    # info call fails with ERROR_BAD_HANDLE.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = kernel32.GetCurrentProcess()
    try:
        fn = kernel32.K32GetProcessMemoryInfo
    except AttributeError:  # pragma: no cover - very old Windows
        fn = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        ctypes.c_uint32,
    ]
    fn.restype = ctypes.c_int
    if fn(handle, ctypes.byref(counters), counters.cb):
        return counters
    return None


def peak_working_set_mb() -> float | None:
    """Cumulative process peak working set in MiB (None on unsupported OS)."""
    counters = _query_counters()
    if counters is None:
        return None
    return round(counters.PeakWorkingSetSize / (1024 * 1024), 2)


def current_working_set_mb() -> float | None:
    """Current process working set in MiB (None on unsupported OS)."""
    counters = _query_counters()
    if counters is None:
        return None
    return round(counters.WorkingSetSize / (1024 * 1024), 2)
