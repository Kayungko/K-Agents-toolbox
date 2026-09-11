"""Weight download, cache verification and provenance (C2 scope).

Downloads the registered foveacast ONNX artifact from the official GitHub
Release into ``model-cache/foveacast/`` (an ignored directory; runtime write
scope of C2), verifying SHA-256 against the digest published on the release
page. Integrity rules (data-contract §3 weights row; technical-design §10):

- doctor/推理只接受「实际文件哈希 == 登记哈希」的权重；不符 → 结构化
  ``MODEL_NOT_READY``（退出码 3），绝不静默使用坏文件。
- 下载失败（网络/HTTP/URL）→ 结构化 ``MODEL_NOT_READY``
  （details.reason=network_error/http_error），不降级、不寻找替代来源。
- 下载写入 ``<name>.part`` 临时文件，流式计算 sha256，全部通过后才原子
  ``os.replace`` 落位；任何校验失败都会删除临时/坏文件。

Provenance of the expected digest (recorded 2026-09-11, C2):
GitHub release page ``khawkins98/foveacast-training`` tag ``v0.2.0``,
``releases/expanded_assets/v0.2.0`` fragment, asset ``foveacast-v3-3s-fp16.onnx``
digest ``sha256:842a23f97908d146b8749e05f6b220bdb495eae76c75cc7252550825585ef76e``
(53.9 MB, uploaded 2026-04-18T14:43:11Z). Cross-checked against
docs/research/model-candidates.md §C2 (prefix ``842a23f97908d146``) and
docs/research/runtime-feasibility.md §4 #2 (``842a23f9…585ef76e``).

Only stdlib networking is used (urllib); no third-party download dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ui_attention.contracts.backend import WeightRef

from .errors import BackendError, DownloadFailureError, WeightNotReadyError

__all__ = [
    "WeightFetchResult",
    "default_cache_dir",
    "fetch_weight",
    "sha256_of_file",
    "verify_cached_weight",
    "write_provenance",
]

_CHUNK_BYTES = 1024 * 1024
_USER_AGENT = "game-ui-attention/C2 (weights.py; internal-eval)"


def default_cache_dir() -> Path:
    """``<project-root>/model-cache/foveacast`` (ignored dir, C2-owned).

    Resolved from this file's location: ``src/ui_attention/backends/weights.py``
    → parents[3] is the project root ``projects/game-ui-attention``.
    """
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "model-cache" / "foveacast"


def sha256_of_file(path: Path | str, chunk_bytes: int = _CHUNK_BYTES) -> str:
    """Streaming SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class WeightFetchResult:
    """Outcome of :func:`fetch_weight` / :func:`verify_cached_weight`."""

    name: str
    path: Path
    sha256: str
    size_bytes: int
    source: str  # "cache" | "download"
    elapsed_s: float | None = None  # download wall time, None for cache hits


def verify_cached_weight(ref: WeightRef, cache_dir: Path | str) -> WeightFetchResult:
    """Offline check: file exists, sha256 == registered, size consistent.

    Raises:
        WeightNotReadyError: missing file, hash mismatch or size mismatch
            (details.reason distinguishes; exit code 3 via C1 catalog).
            No network access.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / ref.name
    if not path.is_file():
        raise WeightNotReadyError(
            f"weight file not found: {path}",
            {
                "weight": ref.name,
                "expected_path": str(path),
                "source_url": ref.source_url,
                "hint": "run the explicit weight download step "
                "(`python -m ui_attention.backends`)",
            },
            reason="missing_file",
        )
    actual = sha256_of_file(path)
    if actual != ref.sha256:
        raise WeightNotReadyError(
            f"weight sha256 mismatch for {ref.name}",
            {
                "weight": ref.name,
                "expected_sha256": ref.sha256,
                "actual_sha256": actual,
                "path": str(path),
            },
            reason="sha256_mismatch",
        )
    size = path.stat().st_size
    if ref.size_bytes > 0 and size != ref.size_bytes:
        raise WeightNotReadyError(
            f"weight size mismatch for {ref.name}",
            {
                "weight": ref.name,
                "expected_size_bytes": ref.size_bytes,
                "actual_size_bytes": size,
                "path": str(path),
            },
            reason="size_mismatch",
        )
    return WeightFetchResult(
        name=ref.name, path=path, sha256=actual, size_bytes=size, source="cache"
    )


def fetch_weight(
    ref: WeightRef,
    cache_dir: Path | str | None = None,
    *,
    timeout_s: float = 300.0,
    progress: Callable[[int, int | None], None] | None = None,
    reverify_cache: bool = True,
) -> WeightFetchResult:
    """Download ``ref`` into ``cache_dir`` unless a verified copy exists.

    Flow: cache hit (sha256 verified) → return ``source="cache"``; otherwise
    stream-download to ``<name>.part`` while hashing, then verify sha256 (and
    size when pinned, ``size_bytes > 0``) BEFORE promoting into place. Any
    integrity failure deletes the bad file and raises a structured error — a
    corrupt cached file is likewise deleted and re-downloaded once; if the
    fresh download still mismatches, the bad file is deleted and
    ``WeightNotReadyError`` is raised (绝不静默使用).

    Raises:
        WeightNotReadyError: post-download sha256/size mismatch (exit 3).
        DownloadFailureError: network/HTTP/OS failure (exit 3); no fallback
            source is ever tried.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / ref.name
    part = cache_dir / (ref.name + ".part")

    if dest.is_file() and reverify_cache:
        try:
            return verify_cached_weight(ref, cache_dir)
        except WeightNotReadyError as exc:
            # Corrupt/tampered cache entry: remove it and fall through to a
            # single re-download attempt. Never use the bad file.
            _unlink_quietly(dest)
            exc.details["cache_entry_removed"] = True

    started = time.perf_counter()
    hasher = hashlib.sha256()
    written = 0
    try:
        request = urllib.request.Request(
            ref.source_url, headers={"User-Agent": _USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:  # noqa: S310
            content_length = resp.headers.get("Content-Length")
            total = int(content_length) if content_length else None
            with open(part, "wb") as fh:
                while True:
                    chunk = resp.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    fh.write(chunk)
                    hasher.update(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(written, total)
    except urllib.error.HTTPError as exc:
        _unlink_quietly(part)
        raise DownloadFailureError(
            f"HTTP {exc.code} downloading {ref.name}",
            {
                "http_status": exc.code,
                "url": ref.source_url,
                "bytes_before_failure": written,
            },
            reason="http_error",
        ) from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        _unlink_quietly(part)
        raise DownloadFailureError(
            f"network failure downloading {ref.name}: {type(exc).__name__}",
            {
                "url": ref.source_url,
                "exception_type": type(exc).__name__,
                "bytes_before_failure": written,
            },
            reason="network_error",
        ) from exc

    actual_sha = hasher.hexdigest()
    elapsed = time.perf_counter() - started
    if actual_sha != ref.sha256:
        _unlink_quietly(part)
        raise WeightNotReadyError(
            f"downloaded {ref.name} failed sha256 verification; bad file deleted",
            {
                "weight": ref.name,
                "expected_sha256": ref.sha256,
                "actual_sha256": actual_sha,
                "url": ref.source_url,
                "size_bytes_downloaded": written,
                "bad_file_deleted": True,
            },
            reason="sha256_mismatch",
        )
    if ref.size_bytes > 0 and written != ref.size_bytes:
        _unlink_quietly(part)
        raise WeightNotReadyError(
            f"downloaded {ref.name} failed size verification; bad file deleted",
            {
                "weight": ref.name,
                "expected_size_bytes": ref.size_bytes,
                "actual_size_bytes": written,
                "bad_file_deleted": True,
            },
            reason="size_mismatch",
        )

    os.replace(part, dest)  # atomic promotion only after full verification
    return WeightFetchResult(
        name=ref.name,
        path=dest,
        sha256=actual_sha,
        size_bytes=written,
        source="download",
        elapsed_s=round(elapsed, 3),
    )


def write_provenance(
    result: WeightFetchResult,
    ref: WeightRef,
    cache_dir: Path | str | None = None,
    *,
    extra: dict | None = None,
) -> Path:
    """Append/refresh ``PROVENANCE.json`` records in the cache dir.

    Records where the file came from, the expected digest source (release
    page), the locally measured digest/size and the fetch timestamp. The
    cache dir is ignored by git; this is local evidence, not a repo artifact.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    prov_path = cache_dir / "PROVENANCE.json"
    records: list[dict] = []
    if prov_path.is_file():
        try:
            loaded = json.loads(prov_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records = [
                    r
                    for r in loaded
                    if isinstance(r, dict) and r.get("name") != ref.name
                ]
        except (OSError, json.JSONDecodeError):
            records = []
    record = {
        "name": ref.name,
        "source_url": ref.source_url,
        "expected_sha256": ref.sha256,
        "expected_sha256_source": (
            "GitHub release khawkins98/foveacast-training v0.2.0, "
            "releases/expanded_assets digest, fetched 2026-09-11"
        ),
        "actual_sha256": result.sha256,
        "size_bytes": result.size_bytes,
        "fetch_source": result.source,
        "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        record.update(extra)
    records.append(record)
    prov_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return prov_path


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _main() -> int:
    """Explicit install step (canonical entry: ``python -m ui_attention.backends``).

    Downloads ONLY the registered 3s FP16 artifact (single small file,
    approved by 二级总控). 1s/7s window-sensitivity weights are NOT fetched
    (medium volume, pending approval). Exit 0 on success, 3 on structured
    failure (mirrors CLI exit semantics for weight-not-ready).
    """
    from .registry import FOVEACAST_3S_PROFILE, resolve_profile

    profile = resolve_profile(FOVEACAST_3S_PROFILE)
    cache_dir = default_cache_dir()
    failures = 0
    for ref in profile.weights:
        try:
            result = fetch_weight(ref, cache_dir)
        except BackendError as exc:
            failures += 1
            print(json.dumps(exc.to_dict(), ensure_ascii=False))
            continue
        write_provenance(result, ref, cache_dir)
        print(
            json.dumps(
                {
                    "ok": True,
                    "name": result.name,
                    "path": str(result.path),
                    "sha256": result.sha256,
                    "size_bytes": result.size_bytes,
                    "source": result.source,
                    "elapsed_s": result.elapsed_s,
                },
                ensure_ascii=False,
            )
        )
    return 0 if failures == 0 else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
