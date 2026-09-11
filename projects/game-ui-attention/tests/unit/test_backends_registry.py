"""Unit tests for backends/registry.py (C2). No network, no real weights needed."""

import dataclasses
import re

import pytest

from ui_attention.backends import registry
from ui_attention.backends.errors import ProfileRejectedError
from ui_attention.backends.registry import (
    ATTRIBUTION,
    FOVEACAST_3S_PROFILE,
    compute_config_hash,
    license_record,
    list_profiles,
    profile_for_backend,
    registration_for,
    resolve_profile,
    verify_profile_integrity,
)
from ui_attention.contracts.backend import ResolvedProfile, WeightRef
from ui_attention.errors import ErrorCode

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RELEASE_SHA256 = "842a23f97908d146b8749e05f6b220bdb495eae76c75cc7252550825585ef76e"

WINDOW_CASES = [
    (
        "foveacast-onnx-1s-v1",
        "foveacast-onnx-1s",
        1,
        "4b9fdc2734e36c612a120ab7b0050ae276723160ccc625c6f554e906dd6345d5",
    ),
    (
        "foveacast-onnx-3s-v1",
        "foveacast-onnx-3s",
        3,
        "842a23f97908d146b8749e05f6b220bdb495eae76c75cc7252550825585ef76e",
    ),
    (
        "foveacast-onnx-7s-v1",
        "foveacast-onnx-7s",
        7,
        "cf66388dc6fe5db4712c77cf3929d04380ed73ac963677b8c9de2b6380fb51e0",
    ),
]


def test_all_three_window_profiles_registered():
    names = list_profiles()
    for profile_name, _, _, _ in WINDOW_CASES:
        assert profile_name in names


@pytest.mark.parametrize(("profile_name", "backend_id", "window_s", "sha"), WINDOW_CASES)
def test_window_profile_fields(profile_name, backend_id, window_s, sha):
    profile = resolve_profile(profile_name)
    profile.validate()
    assert profile.backend_id == backend_id
    assert SHA256_RE.match(profile.config_hash)
    assert profile.weights[0].sha256 == sha  # release 页 digest
    assert profile.weights[0].size_bytes == 56549242  # 本地实测精确值
    assert profile.viewing_conditions["viewing_window_s"] == window_s
    assert profile.viewing_conditions["display_conditions_required"] is False
    assert profile.preprocessing["target_height"] == 240
    assert profile.preprocessing["target_width"] == 320
    assert profile.centerbias is None
    reg = registration_for(profile_name)
    # 许可字段三窗口一致（G2 指令：同 3s）
    assert reg.license_status["gaps"] == ["G3", "G4", "G8"]
    assert reg.license_status["cleared_for"] == "internal-eval"
    assert reg.attribution == ATTRIBUTION


def test_window_config_hashes_distinct():
    hashes = {resolve_profile(name).config_hash for name, _, _, _ in WINDOW_CASES}
    assert len(hashes) == 3  # 权重与观看窗口不同 → 合成哈希必然不同


def test_profile_for_backend_windows():
    assert profile_for_backend("foveacast-onnx-1s").profile_name == "foveacast-onnx-1s-v1"
    assert profile_for_backend("foveacast-onnx-3s").profile_name == "foveacast-onnx-3s-v1"
    assert profile_for_backend("foveacast-onnx-7s").profile_name == "foveacast-onnx-7s-v1"


def test_first_profile_registered():
    assert FOVEACAST_3S_PROFILE == "foveacast-onnx-3s-v1"
    assert FOVEACAST_3S_PROFILE in list_profiles()


def test_resolve_returns_contract_type_and_validates():
    profile = resolve_profile(FOVEACAST_3S_PROFILE)
    assert isinstance(profile, ResolvedProfile)
    profile.validate()  # C1 contract validation must pass at registration
    assert profile.backend_id == "foveacast-onnx-3s"
    assert profile.backend_version
    assert SHA256_RE.match(profile.config_hash), "config_hash 必须是 64 位裸十六进制"
    assert profile.centerbias is None  # foveacast 无显式先验输入 → null


def test_registration_items_complete():
    profile = resolve_profile(FOVEACAST_3S_PROFILE)
    assert len(profile.weights) == 1
    w = profile.weights[0]
    assert isinstance(w, WeightRef)
    assert w.name == "foveacast-v3-3s-fp16.onnx"
    assert "khawkins98/foveacast-training/releases/download/v0.2.0/" in w.source_url
    assert w.sha256 == RELEASE_SHA256
    assert w.size_bytes > 0
    pre = profile.preprocessing
    assert (pre["target_height"], pre["target_width"]) == (240, 320)
    assert pre["value_range"] == [0.0, 255.0]
    assert pre["mean_subtraction"] == "in_graph"
    assert pre["adapter_renormalization"] == "sum_to_1_probability_density"
    vc = profile.viewing_conditions
    assert vc["viewing_window_s"] == 3
    assert vc["display_conditions_required"] is False
    assert vc["assumption"]


def test_resolved_profile_is_readonly():
    profile = resolve_profile(FOVEACAST_3S_PROFILE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.profile_name = "other-v1"  # type: ignore[misc]


def test_resolve_unknown_profile_structured():
    with pytest.raises(ProfileRejectedError) as ei:
        resolve_profile("local-static-v1")
    exc = ei.value
    assert exc.code is ErrorCode.PROFILE_NOT_REGISTERED
    assert exc.exit_code == 3
    assert exc.details["reason"] == "not_registered"
    assert FOVEACAST_3S_PROFILE in exc.details["registered"]


@pytest.mark.parametrize("bad", ["", None, 123])
def test_resolve_bad_name_structured(bad):
    with pytest.raises(ProfileRejectedError) as ei:
        resolve_profile(bad)
    assert ei.value.code is ErrorCode.PROFILE_NOT_REGISTERED


def test_tampered_profile_rejected_by_config_hash():
    profile = resolve_profile(FOVEACAST_3S_PROFILE)
    tampered = dataclasses.replace(
        profile, preprocessing={**profile.preprocessing, "target_height": 480}
    )
    with pytest.raises(ProfileRejectedError) as ei:
        verify_profile_integrity(tampered)
    assert ei.value.code is ErrorCode.PROFILE_NOT_REGISTERED
    assert ei.value.details["reason"] == "profile_tampered"
    assert ei.value.details["registered_config_hash"] == profile.config_hash
    # the registered instance itself still passes
    verify_profile_integrity(profile)


def test_config_hash_recipe_stable_and_sensitive():
    kwargs = {
        "backend_id": "b",
        "backend_version": "1",
        "weights": (registry.FOVEACAST_3S_WEIGHT,),
        "preprocessing": {"a": 1},
        "centerbias": None,
        "viewing_conditions": {"v": 3},
    }
    h1 = compute_config_hash(**kwargs)
    assert h1 == compute_config_hash(**kwargs)
    assert SHA256_RE.match(h1)
    assert compute_config_hash(**{**kwargs, "backend_version": "2"}) != h1
    assert compute_config_hash(**{**kwargs, "preprocessing": {"a": 2}}) != h1
    assert compute_config_hash(**{**kwargs, "viewing_conditions": {"v": 7}}) != h1
    assert compute_config_hash(**{**kwargs, "weights": ()}) != h1
    cb = {"source": "mit1003", "sha256": "0" * 64}
    assert compute_config_hash(**{**kwargs, "centerbias": cb}) != h1


def test_license_status_frozen_shape():
    ls = registration_for(FOVEACAST_3S_PROFILE).license_status
    assert set(ls) == {"code", "weights", "gaps", "cleared_for"}
    assert ls["gaps"] == ["G3", "G4", "G8"]
    assert ls["cleared_for"] == "internal-eval"
    assert "MIT" in ls["code"]
    assert "UEyes" in ls["weights"] and "署名" in ls["weights"]


def test_attribution_citations_recorded():
    joined = "\n".join(ATTRIBUTION)
    # CC-BY-4.0 传递义务：两条完整引用必须落地
    assert "10.1145/3544548.3581096" in joined
    assert "10.1016/j.neunet.2020.05.004" in joined
    assert "UEyes: Understanding Visual Saliency across User Interface Types" in joined
    assert "Neural Networks" in joined and "261-270" in joined
    record = license_record()
    assert record["attribution"] == list(ATTRIBUTION)
    assert set(record["gaps_detail"]) == {"G3", "G4", "G8"}


def test_profile_for_backend():
    assert profile_for_backend("foveacast-onnx-3s").profile_name == "foveacast-onnx-3s-v1"
    assert profile_for_backend("foveacast-onnx-1s").profile_name == "foveacast-onnx-1s-v1"
    assert profile_for_backend("foveacast-onnx-7s").profile_name == "foveacast-onnx-7s-v1"
    assert profile_for_backend("deepgaze-iie").profile_name == registry.DEEPGAZE_IIE_PROFILE
    with pytest.raises(ProfileRejectedError):
        profile_for_backend("umsi-plus-plus")  # 真正未登记的 backend


def test_get_backend_rejects_foreign_backend_before_touching_disk():
    profile = resolve_profile(FOVEACAST_3S_PROFILE)
    foreign_hash = compute_config_hash(
        backend_id="umsi-plus-plus",
        backend_version=profile.backend_version,
        weights=profile.weights,
        preprocessing=profile.preprocessing,
        centerbias=profile.centerbias,
        viewing_conditions=profile.viewing_conditions,
    )
    foreign = dataclasses.replace(
        profile, backend_id="umsi-plus-plus", config_hash=foreign_hash
    )
    with pytest.raises(ProfileRejectedError) as ei:
        registry.get_backend(foreign)
    assert ei.value.details["reason"] == "backend_mismatch"
