"""hashing.py 单测：config_hash 七参与项复现性 / study_id 同口径 / sha256 口径 / 代码哈希。"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

_WEB_DIR = Path(__file__).resolve().parents[1] / "web"
if str(_WEB_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_DIR))

import hashing  # noqa: E402
import make_study_package as mp  # noqa: E402  （conftest 已把 tools 加入 sys.path）


def _base_kwargs() -> dict:
    return {
        "image_sha256s": ["b" * 64, "a" * 64],
        "protocol_version": "v1",
        "schema_version": hashing.RESPONSE_SCHEMA_VERSION,
        "client_code_hash": "c" * 64,
        "server_code_hash": "d" * 64,
        "duration_seconds": 3.0,
        "seed_strategy": "per-session",
    }


def test_canonical_json_sorts_keys() -> None:
    assert hashing.canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_sha256_file_matches_tools_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 4096)
    assert hashing.sha256_file(p) == mp.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_compute_study_id_matches_make_study_package() -> None:
    metas = [{"sha256": "a" * 64, "source_name": "1.png"}, {"sha256": "b" * 64, "source_name": "2.png"}]
    assert hashing.compute_study_id(metas, 3.0) == mp.compute_study_id(metas, 3.0)
    assert len(hashing.compute_study_id(metas, 3.0)) == 16


def test_config_hash_reproducible() -> None:
    assert hashing.compute_config_hash(**_base_kwargs()) == hashing.compute_config_hash(**_base_kwargs())


def test_config_hash_is_64_hex() -> None:
    h = hashing.compute_config_hash(**_base_kwargs())
    assert len(h) == 64
    int(h, 16)


def test_config_hash_ignores_image_order() -> None:
    a = dict(_base_kwargs(), image_sha256s=["1" * 64, "2" * 64, "3" * 64])
    b = dict(_base_kwargs(), image_sha256s=["3" * 64, "1" * 64, "2" * 64])
    assert hashing.compute_config_hash(**a) == hashing.compute_config_hash(**b)


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_sha256s", ["c" * 64]),
        ("protocol_version", "v2"),
        ("schema_version", "game-ui-attention-calibration-response/v2"),
        ("client_code_hash", "e" * 64),
        ("server_code_hash", "f" * 64),
        ("duration_seconds", 5.0),
        ("seed_strategy", "global:42"),
    ],
)
def test_config_hash_changes_per_participant_item(field: str, value: object) -> None:
    base = _base_kwargs()
    changed = dict(base, **{field: value})
    assert hashing.compute_config_hash(**base) != hashing.compute_config_hash(**changed)


def test_config_hash_changes_when_device_set() -> None:
    base = _base_kwargs()
    with_device = dict(base, device="tobii-fusion-250", device_params={"rate": 250})
    assert hashing.compute_config_hash(**base) != hashing.compute_config_hash(**with_device)


def test_client_and_server_code_hashes() -> None:
    assert hashing.compute_client_code_hash(_WEB_DIR) == hashing.sha256_file(_WEB_DIR / "index.html")
    sh = hashing.compute_server_code_hash(_WEB_DIR)
    assert len(sh) == 64
    int(sh, 16)
