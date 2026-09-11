"""Unit tests for backends/weights.py (C2). Fake network via monkeypatch; no real downloads."""

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest

from ui_attention.backends import weights as wmod
from ui_attention.backends.errors import DownloadFailureError, WeightNotReadyError
from ui_attention.backends.weights import (
    WeightFetchResult,
    default_cache_dir,
    fetch_weight,
    sha256_of_file,
    verify_cached_weight,
    write_provenance,
)
from ui_attention.contracts.backend import WeightRef
from ui_attention.errors import ErrorCode

CONTENT = b"fake-onnx-bytes-for-test" * 100
SHA = hashlib.sha256(CONTENT).hexdigest()


def make_ref(
    name: str = "w.onnx",
    sha: str = SHA,
    size: int = len(CONTENT),
    url: str = "https://example.invalid/w.onnx",
) -> WeightRef:
    return WeightRef(name=name, source_url=url, sha256=sha, size_bytes=size)


class FakeResponse:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def patch_urlopen(monkeypatch, payload: bytes) -> None:
    monkeypatch.setattr(
        wmod.urllib.request,
        "urlopen",
        lambda request, timeout=None: FakeResponse(payload),
    )


def forbid_network(monkeypatch) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted in a cache-only test")

    monkeypatch.setattr(wmod.urllib.request, "urlopen", _boom)


# ---------------------------------------------------------------------------
# hashing / offline verification
# ---------------------------------------------------------------------------


def test_sha256_of_file(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"abc")
    assert sha256_of_file(p) == hashlib.sha256(b"abc").hexdigest()


def test_verify_cached_ok(tmp_path: Path):
    (tmp_path / "w.onnx").write_bytes(CONTENT)
    result = verify_cached_weight(make_ref(), tmp_path)
    assert result.source == "cache"
    assert result.sha256 == SHA
    assert result.size_bytes == len(CONTENT)


def test_verify_cached_missing_structured(tmp_path: Path):
    with pytest.raises(WeightNotReadyError) as ei:
        verify_cached_weight(make_ref(), tmp_path)
    exc = ei.value
    assert exc.code is ErrorCode.MODEL_NOT_READY
    assert exc.exit_code == 3
    assert exc.details["reason"] == "missing_file"
    assert "download" in exc.details["hint"]


def test_verify_cached_hash_mismatch_structured(tmp_path: Path):
    (tmp_path / "w.onnx").write_bytes(b"corrupt-payload")
    with pytest.raises(WeightNotReadyError) as ei:
        verify_cached_weight(make_ref(), tmp_path)
    exc = ei.value
    assert exc.code is ErrorCode.MODEL_NOT_READY
    assert exc.details["reason"] == "sha256_mismatch"
    assert exc.details["expected_sha256"] == SHA
    assert exc.details["actual_sha256"] == hashlib.sha256(b"corrupt-payload").hexdigest()


def test_verify_cached_size_mismatch_structured(tmp_path: Path):
    (tmp_path / "w.onnx").write_bytes(CONTENT)
    with pytest.raises(WeightNotReadyError) as ei:
        verify_cached_weight(make_ref(size=len(CONTENT) + 1), tmp_path)
    assert ei.value.details["reason"] == "size_mismatch"


# ---------------------------------------------------------------------------
# download flow (fake network)
# ---------------------------------------------------------------------------


def test_fetch_cache_hit_never_touches_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "w.onnx").write_bytes(CONTENT)
    forbid_network(monkeypatch)
    result = fetch_weight(make_ref(), tmp_path)
    assert result.source == "cache"
    assert result.elapsed_s is None


def test_fetch_downloads_verifies_and_promotes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    patch_urlopen(monkeypatch, CONTENT)
    seen: list[tuple[int, int | None]] = []
    result = fetch_weight(make_ref(), tmp_path, progress=lambda d, t: seen.append((d, t)))
    assert result.source == "download"
    assert result.sha256 == SHA
    assert result.size_bytes == len(CONTENT)
    assert (tmp_path / "w.onnx").read_bytes() == CONTENT
    assert not (tmp_path / "w.onnx.part").exists()
    assert result.elapsed_s is not None and result.elapsed_s >= 0.0
    assert seen and seen[-1][0] == len(CONTENT)


def test_fetch_corrupt_cache_entry_is_removed_and_redownloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "w.onnx").write_bytes(b"corrupt-payload")
    patch_urlopen(monkeypatch, CONTENT)
    result = fetch_weight(make_ref(), tmp_path)
    assert result.source == "download"
    assert (tmp_path / "w.onnx").read_bytes() == CONTENT


def test_fetch_hash_mismatch_deletes_bad_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    patch_urlopen(monkeypatch, b"wrong-bytes" * 50)
    with pytest.raises(WeightNotReadyError) as ei:
        fetch_weight(make_ref(), tmp_path)
    exc = ei.value
    assert exc.code is ErrorCode.MODEL_NOT_READY
    assert exc.details["reason"] == "sha256_mismatch"
    assert exc.details["bad_file_deleted"] is True
    # 绝不静默使用：目标位与临时位都不得残留坏文件
    assert not (tmp_path / "w.onnx").exists()
    assert not (tmp_path / "w.onnx.part").exists()


def test_fetch_size_mismatch_after_hash_deletes_bad_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    patch_urlopen(monkeypatch, CONTENT)  # hash would match…
    with pytest.raises(WeightNotReadyError) as ei:
        fetch_weight(make_ref(size=len(CONTENT) + 7), tmp_path)  # …but pinned size doesn't
    assert ei.value.details["reason"] == "size_mismatch"
    assert not (tmp_path / "w.onnx").exists()
    assert not (tmp_path / "w.onnx.part").exists()


def test_fetch_network_error_structured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _raise(*args: object, **kwargs: object) -> None:
        raise urllib.error.URLError("dns failure (simulated)")

    monkeypatch.setattr(wmod.urllib.request, "urlopen", _raise)
    with pytest.raises(DownloadFailureError) as ei:
        fetch_weight(make_ref(), tmp_path)
    exc = ei.value
    assert exc.code is ErrorCode.MODEL_NOT_READY  # C1 catalog: 权重未就绪 → exit 3
    assert exc.exit_code == 3
    assert exc.details["reason"] == "network_error"
    assert not (tmp_path / "w.onnx.part").exists()


def test_fetch_http_error_structured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _raise(*args: object, **kwargs: object) -> None:
        raise urllib.error.HTTPError(
            "https://example.invalid/w.onnx", 404, "Not Found", {}, None  # type: ignore[arg-type]
        )

    monkeypatch.setattr(wmod.urllib.request, "urlopen", _raise)
    with pytest.raises(DownloadFailureError) as ei:
        fetch_weight(make_ref(), tmp_path)
    assert ei.value.details["reason"] == "http_error"
    assert ei.value.details["http_status"] == 404


# ---------------------------------------------------------------------------
# provenance / cache dir
# ---------------------------------------------------------------------------


def test_write_provenance_records_source_and_digest(tmp_path: Path):
    ref = make_ref()
    result = WeightFetchResult(
        name=ref.name,
        path=tmp_path / ref.name,
        sha256=SHA,
        size_bytes=len(CONTENT),
        source="download",
        elapsed_s=1.5,
    )
    path = write_provenance(result, ref, tmp_path)
    records = json.loads(path.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["expected_sha256"] == SHA
    assert records[0]["actual_sha256"] == SHA
    assert "expanded_assets" in records[0]["expected_sha256_source"]
    # second write for the same name replaces, other names append
    write_provenance(result, ref, tmp_path)
    write_provenance(result, make_ref(name="other.onnx"), tmp_path)
    records = json.loads(path.read_text(encoding="utf-8"))
    assert [r["name"] for r in records] == ["w.onnx", "other.onnx"]


def test_default_cache_dir_inside_ignored_model_cache():
    cache = default_cache_dir()
    assert cache.name == "foveacast"
    assert cache.parent.name == "model-cache"
    assert cache.parent.parent.name == "game-ui-attention"
