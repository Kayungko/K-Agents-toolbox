"""DeepGaze IIE runtime measurement driver (C2, G1' internal-eval line).

Usage (fresh process per cold-start sample)::

    .venv\\Scripts\\python.exe -m ui_attention.backends.measure_deepgaze [--fresh]
        [--hot-runs 5] [--warmup 1]

记录（真实执行数字；写入 ``model-cache/deepgaze/runtime-measurements.json``，
忽略目录）：

- U1 关卡证据：离线构建（无 torch.hub/bitbucket/ImageNet 下载）+
  ``load_state_dict(strict=True)`` 成功、参数量、state_dict 条目数、骨干补丁记录
- U18 证据：torch.version.cuda is None（CPU wheel）
- 冷启动（构建+加载）ms、首次 predict ms、热推理统计（1920×1080 → 1024×576
  与 240×320 原尺寸两档）、进程峰值工作集 + tracemalloc、确定性、
  log-density 值域与 C1 转换 sanity（exp 归一 sum=1）
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import registry
from .deepgaze_iie import DeepGazeIIEBackend, default_deepgaze_cache_dir
from .measure_runtime import _stats, synthetic_ui_image
from .runtime_probe import current_working_set_mb, peak_working_set_mb
from .weights import verify_cached_weight

__all__ = ["main", "measure_once"]

_SCHEMA = "deepgaze-runtime-measurements/v1"


def measure_once(hot_runs: int = 5, warmup: int = 1) -> dict[str, Any]:
    profile = registry.resolve_profile(registry.DEEPGAZE_IIE_PROFILE)
    weights_verified = [
        {"name": w.name, "sha256": verify_cached_weight(w, default_deepgaze_cache_dir()).sha256}
        for w in profile.weights
    ]

    import torch  # deferred: measurement line requires [deepgaze] extras

    backend = registry.get_backend(profile)
    assert isinstance(backend, DeepGazeIIEBackend)

    big = synthetic_ui_image(1080, 1920, seed=20260911)
    small = synthetic_ui_image(240, 320, seed=5)

    t0 = time.perf_counter()
    first_big = backend.predict(big, profile)  # includes cold build + strict load
    first_predict_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    cold_load_ms = backend.cold_load_ms
    model_stats = dict(backend.model_stats or {})

    det = backend.predict(big, profile)
    determinism_max_abs_diff = float(np.max(np.abs(det.array - first_big.array)))

    from scipy.special import logsumexp

    prob = np.exp(first_big.array - logsumexp(first_big.array))
    conversion_sanity = {
        "probability_sum": float(prob.sum()),
        "log_density_min": float(first_big.array.min()),
        "log_density_max": float(first_big.array.max()),
    }

    for _ in range(warmup):
        backend.predict(big, profile)
    tracemalloc.start()
    big_samples: list[float] = []
    for _ in range(hot_runs):
        big_samples.append(float(backend.predict(big, profile).runtime["elapsed_ms"]))
    small_samples: list[float] = []
    for _ in range(hot_runs):
        small_samples.append(float(backend.predict(small, profile).runtime["elapsed_ms"]))
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "measured_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weights": weights_verified,
        "u1_gate": {
            "offline_build": True,
            "strict_state_dict_load": True,
            **model_stats,
        },
        "cold_start_build_and_load_ms": cold_load_ms,
        "first_predict_1920x1080_ms": first_predict_ms,
        "hot_predict_1024x576_ms": _stats(big_samples),
        "hot_predict_240x320_native_ms": _stats(small_samples),
        "determinism_max_abs_diff": determinism_max_abs_diff,
        "c1_conversion_sanity": conversion_sanity,
        "runtime_field_sample": dict(first_big.runtime),
        "memory": {
            "process_peak_working_set_mb": peak_working_set_mb(),
            "process_current_working_set_mb": current_working_set_mb(),
            "hot_loop_tracemalloc_peak_mb": round(tracemalloc_peak / (1024 * 1024), 2),
            "note": "peak working set 为进程累计峰值（含 torch 运行时与 4 骨干集成权重）",
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torchvision": __import__("torchvision").__version__,
            "hot_runs": hot_runs,
            "warmup": warmup,
        },
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(key: str) -> dict[str, float] | None:
        values = [float(r[key]) for r in runs if isinstance(r.get(key), (int, float))]
        if not values:
            return None
        return {
            "samples": len(values),
            "mean": round(statistics.fmean(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    def hot_means(field: str) -> dict[str, float] | None:
        values = [
            float(r[field]["mean_ms"]) for r in runs if isinstance(r.get(field), dict)
        ]
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
        "cold_start_build_and_load_ms": summarize("cold_start_build_and_load_ms"),
        "first_predict_1920x1080_ms": summarize("first_predict_1920x1080_ms"),
        "hot_predict_1024x576_mean_ms": hot_means("hot_predict_1024x576_ms"),
        "hot_predict_240x320_native_mean_ms": hot_means("hot_predict_240x320_native_ms"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--hot-runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    run_entry = measure_once(args.hot_runs, args.warmup)
    out_path = args.out or (default_deepgaze_cache_dir() / "runtime-measurements.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    document: dict[str, Any] = {}
    if out_path.is_file() and not args.fresh:
        try:
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                document = loaded
        except (OSError, json.JSONDecodeError):
            document = {}
    runs = list(document.get("runs", []))
    runs.append(run_entry)
    document = {
        "schema": _SCHEMA,
        "generated_at_utc": run_entry["measured_at_utc"],
        "purpose": "G2 DeepGaze IIE 内部评估对照线运行实测（U1/U5-IIE/U18 消解证据；"
        "仅限内部研究评估，不打包不分发）",
        "backend": {
            "backend_id": registry.DEEPGAZE_IIE_BACKEND_ID,
            "backend_version": registry.DEEPGAZE_IIE_BACKEND_VERSION,
            "profile": registry.DEEPGAZE_IIE_PROFILE,
            "config_hash": registry.resolve_profile(registry.DEEPGAZE_IIE_PROFILE).config_hash,
            "source_pin": registry.DEEPGAZE_SOURCE_PIN,
            "license_gate": "G1/G2/G5 未闭合; G1' 批准 internal-eval (2026-09-11)",
        },
        "aggregate": _aggregate(runs),
        "runs": runs,
    }
    out_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"ok": True, "out": str(out_path), "aggregate": document["aggregate"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
