"""Runtime measurement driver (C2 scope) — 消解 U4/U5/U6/U8 的真实数字来源.

Usage (fresh process per cold-start sample)::

    .venv\\Scripts\\python.exe -m ui_attention.backends.measure_runtime [--fresh]
        [--hot-runs 30] [--warmup 3]

每次调用都是一个「冷启动样本」：在新进程内首次创建 InferenceSession
（``cold_load_ms``）、首次 predict（``first_predict_ms``），随后 warmup +
N 次热推理统计（1920×1080 合成图，确定性种子），并记录：

- ONNX 实际输入/输出元数据（名称/形状/dtype，运行时读取 → U8 证据）
- 原始模型输出值域（逐图 min-max [0,1] 口径核对 → U6 证据：FP16 权重在
  CPU EP 上的实际行为）
- 进程峰值工作集（Windows PeakWorkingSetSize → U4 证据）与热推理循环
  tracemalloc Python 侧峰值（互补口径，ORT 原生分配不计入 tracemalloc）
- 热推理 elapsed_ms 统计（mean/std/min/p50/p95/max → U5 证据）
- 确定性：同一输入两次 predict 的最大绝对差

结果写入 ``model-cache/foveacast/runtime-measurements.json``（忽略目录；
默认追加 ``runs`` 并重算 ``aggregate``，``--fresh`` 重置）。合成图仅作测量
输入，不是眼动真值，也不构成任何有效性证据。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import registry
from .onnx_foveacast import FoveacastOnnxBackend
from .runtime_probe import current_working_set_mb, peak_working_set_mb
from .weights import default_cache_dir, verify_cached_weight

__all__ = ["main", "measure_once", "synthetic_ui_image"]

_SCHEMA = "runtime-measurements/v1"


def synthetic_ui_image(height: int = 1080, width: int = 1920, seed: int = 20260911) -> np.ndarray:
    """Deterministic synthetic UI-like RGB uint8 image (measurement input only).

    Gradient background + soft noise + a few button/panel rectangles. Not eye
    tracking ground truth, not a validity claim — just a stable 1920×1080 load.
    """
    rng = np.random.default_rng(seed)
    yy = np.linspace(40.0, 96.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(20.0, 60.0, width, dtype=np.float32)[None, :]
    base = np.broadcast_to(yy + xx, (height, width)).astype(np.float32)
    noise = rng.normal(0.0, 6.0, size=(height, width)).astype(np.float32)
    gray = np.clip(base + noise, 0.0, 255.0)
    image = np.dstack([gray, gray * 0.98, gray * 1.02]).astype(np.float32)
    # panel + primary button + secondary button (UI-like structure)
    image[140:940, 460:1460, :] = np.clip(image[140:940, 460:1460, :] + 46.0, 0, 255)
    image[760:860, 780:1140, 0] = 224.0
    image[760:860, 780:1140, 1] = 168.0
    image[760:860, 780:1140, 2] = 48.0
    image[620:680, 860:1060, :] = 200.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    n = len(ordered)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(math.ceil(p / 100.0 * n)) - 1))
        return round(ordered[idx], 3)

    return {
        "count": n,
        "mean_ms": round(statistics.fmean(ordered), 3),
        "std_ms": round(statistics.stdev(ordered), 3) if n > 1 else 0.0,
        "min_ms": round(ordered[0], 3),
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "max_ms": round(ordered[-1], 3),
    }


def measure_once(hot_runs: int = 30, warmup: int = 3, height: int = 1080, width: int = 1920) -> dict[str, Any]:
    """One cold-process measurement run; returns a JSON-serializable dict."""
    profile = registry.resolve_profile(registry.FOVEACAST_3S_PROFILE)
    backend = registry.get_backend(profile)  # offline weight verification (no session yet)
    assert isinstance(backend, FoveacastOnnxBackend)
    weight = verify_cached_weight(profile.weights[0], default_cache_dir())

    image = synthetic_ui_image(height, width)

    # --- cold path: first predict creates the InferenceSession -------------
    t0 = time.perf_counter()
    first_result = backend.predict(image, profile)
    first_predict_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    cold_load_ms = backend.cold_load_ms
    metadata = backend.session_metadata

    # --- determinism --------------------------------------------------------
    det_result = backend.predict(image, profile)
    determinism_max_abs_diff = float(np.max(np.abs(det_result.array - first_result.array)))

    # --- raw model output range (pre-renormalization, U6 evidence) ----------
    tensor = backend._preprocess(image, 240, 320, profile.preprocessing)  # noqa: SLF001
    assert backend._session is not None and backend._input_meta is not None  # noqa: SLF001
    assert backend._output_meta is not None  # noqa: SLF001
    raw = backend._session.run(  # noqa: SLF001
        [backend._output_meta["name"]], {backend._input_meta["name"]: tensor}  # noqa: SLF001
    )[0]
    raw_arr = np.asarray(raw, dtype=np.float64)
    raw_stats = {
        "shape": list(raw_arr.shape),
        "dtype": str(np.asarray(raw).dtype),
        "min": float(raw_arr.min()),
        "max": float(raw_arr.max()),
        "sum": float(raw_arr.sum()),
    }

    # --- hot loop ------------------------------------------------------------
    for _ in range(warmup):
        backend.predict(image, profile)
    tracemalloc.start()
    samples: list[float] = []
    for _ in range(hot_runs):
        result = backend.predict(image, profile)
        samples.append(float(result.runtime["elapsed_ms"]))
    _, tracemalloc_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    adapted_sum = float(first_result.array.sum())
    return {
        "measured_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": {
            "kind": "synthetic_ui_image",
            "height": height,
            "width": width,
            "seed": 20260911,
            "note": "确定性合成图，仅测量负载；非眼动真值、非有效性证据",
        },
        "weight": {"name": weight.name, "sha256": weight.sha256, "size_bytes": weight.size_bytes},
        "cold_start_session_load_ms": cold_load_ms,
        "first_predict_ms": first_predict_ms,
        "hot_predict_elapsed_ms": _stats(samples),
        "determinism_max_abs_diff": determinism_max_abs_diff,
        "raw_model_output": raw_stats,
        "adapted_output": {
            "semantics": first_result.semantics,
            "dtype": str(first_result.array.dtype),
            "shape": list(first_result.array.shape),
            "sum": adapted_sum,
            "min": float(first_result.array.min()),
            "max": float(first_result.array.max()),
        },
        "onnx_metadata": metadata,
        "runtime_field_sample": dict(first_result.runtime),
        "memory": {
            "process_peak_working_set_mb": peak_working_set_mb(),
            "process_current_working_set_mb": current_working_set_mb(),
            "hot_loop_tracemalloc_peak_mb": round(tracemalloc_peak_bytes / (1024 * 1024), 2),
            "note": "peak working set 为进程累计峰值（含 ORT 会话与权重映射）；"
            "tracemalloc 仅计 Python 侧分配，ORT 原生分配不计入",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_logical_processors": os.cpu_count(),
            "providers": metadata["providers"] if metadata else None,
            "hot_runs": hot_runs,
            "warmup": warmup,
        },
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def samples(key: str) -> list[float]:
        return [float(r[key]) for r in runs if isinstance(r.get(key), (int, float))]

    def hot_means() -> list[float]:
        return [
            float(r["hot_predict_elapsed_ms"]["mean_ms"])
            for r in runs
            if isinstance(r.get("hot_predict_elapsed_ms"), dict)
        ]

    def peaks() -> list[float]:
        out = []
        for r in runs:
            mem = r.get("memory") or {}
            value = mem.get("process_peak_working_set_mb")
            if isinstance(value, (int, float)):
                out.append(float(value))
        return out

    def summarize(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "samples": len(values),
            "mean": round(statistics.fmean(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    return {
        "runs": len(runs),
        "cold_start_session_load_ms": summarize(samples("cold_start_session_load_ms")),
        "first_predict_ms": summarize(samples("first_predict_ms")),
        "hot_predict_mean_ms": summarize(hot_means()),
        "process_peak_working_set_mb": summarize(peaks()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fresh", action="store_true", help="reset the runs list before appending")
    parser.add_argument("--hot-runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    run_entry = measure_once(args.hot_runs, args.warmup, args.height, args.width)

    out_path = args.out or (default_cache_dir() / "runtime-measurements.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {}
    if out_path.is_file() and not args.fresh:
        try:
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                document = loaded
        except (OSError, json.JSONDecodeError):
            document = {}
    runs: list[dict[str, Any]] = list(document.get("runs", []))
    runs.append(run_entry)
    document = {
        "schema": _SCHEMA,
        "generated_at_utc": run_entry["measured_at_utc"],
        "purpose": "阶段0运行实测（implementation-plan 阶段0：冷启动/热推理/峰值内存/元数据；"
        "消解 runtime-feasibility U4/U5/U6/U8）",
        "backend": {
            "backend_id": registry.FOVEACAST_3S_BACKEND_ID,
            "backend_version": registry.FOVEACAST_3S_BACKEND_VERSION,
            "profile": registry.FOVEACAST_3S_PROFILE,
            "config_hash": registry.resolve_profile(registry.FOVEACAST_3S_PROFILE).config_hash,
        },
        "aggregate": _aggregate(runs),
        "runs": runs,
    }
    out_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out_path), "aggregate": document["aggregate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
