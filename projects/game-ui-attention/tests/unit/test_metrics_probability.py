"""metrics.probability 测试：归一化、log_density→概率、重采样后再归一化（validation-plan §1“概率归一化”行）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS = Path(__file__).resolve().parents[1]
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fixtures import synthetic  # noqa: E402
from ui_attention.errors import ErrorCode, UiAttentionError  # noqa: E402
from ui_attention.metrics import (  # noqa: E402
    DENSITY_SUM_TOLERANCE,
    bilinear_resample,
    log_density_to_probability,
    logsumexp,
    normalize_probability,
    resample_probability,
    resample_to_original,
    validate_probability,
)


def test_tolerance_is_frozen_1e6():
    assert DENSITY_SUM_TOLERANCE == 1e-6


def test_log_density_to_probability_known_values():
    L = np.log(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))
    P = log_density_to_probability(L)
    assert P == pytest.approx(np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert P.sum() == pytest.approx(1.0, abs=1e-12)


def test_log_density_probability_roundtrip_block_map():
    """构造概率图 → log 域 → P=exp(L−logsumexp(L)) 往返一致。"""
    P0 = synthetic.block_probability((48, 64), (10, 8, 32, 16), mass=0.3)
    L = synthetic.log_density_from_probability(P0)
    P1 = log_density_to_probability(L)
    assert P1 == pytest.approx(P0, abs=1e-12)


def test_logsumexp_stable_with_large_offset():
    L = np.full((4, 4), 700.0)  # exp(700) 溢出域：logsumexp 仍应稳定
    assert logsumexp(L) == pytest.approx(700.0 + np.log(16))
    P = log_density_to_probability(L)
    assert P == pytest.approx(np.full((4, 4), 1 / 16))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_log_density_rejects_non_finite(bad):
    L = np.full((4, 4), -np.log(16.0))
    L[0, 0] = bad
    with pytest.raises(UiAttentionError) as exc:
        log_density_to_probability(L)
    assert exc.value.code is ErrorCode.INVALID_DENSITY
    assert exc.value.exit_code == 5


def test_validate_probability_sum_tolerance_boundary():
    P = synthetic.uniform_probability((10, 10))
    validate_probability(P)  # sum == 1 精确
    drifted = P * (1 + 5e-7)  # 偏差在 1e-6 容差内
    validate_probability(drifted)
    broken = P * (1 + 2e-6)  # 超差
    with pytest.raises(UiAttentionError) as exc:
        validate_probability(broken)
    assert "sum" in str(exc.value.details["problems"])


def test_validate_probability_rejects_negative_and_nan():
    P = synthetic.uniform_probability((8, 8))
    P[0, 0] = -1e-9
    with pytest.raises(UiAttentionError):
        validate_probability(P)
    P2 = synthetic.uniform_probability((8, 8))
    P2[3, 3] = np.nan
    with pytest.raises(UiAttentionError):
        validate_probability(P2)


def test_validate_probability_rejects_wrong_shape_and_type():
    with pytest.raises(UiAttentionError):
        validate_probability(np.zeros((2, 2, 3)))
    with pytest.raises(UiAttentionError):
        validate_probability([0.5, 0.5])  # 非 ndarray


def test_normalize_probability_sums_to_one():
    raw = np.array([[1.0, 3.0], [0.0, 4.0]])
    P = normalize_probability(raw)
    assert P.sum() == pytest.approx(1.0, abs=1e-12)
    assert P[0, 1] == pytest.approx(3 / 8)


def test_normalize_rejects_negative_and_zero_sum():
    with pytest.raises(UiAttentionError):
        normalize_probability(np.array([[1.0, -0.5], [0.5, 0.0]]))  # 不静默截断
    with pytest.raises(UiAttentionError):
        normalize_probability(np.zeros((3, 3)))


def test_bilinear_resample_constant_map_preserved():
    P = np.full((20, 30), 1 / 600.0)
    up = bilinear_resample(P, (40, 60))
    assert up.shape == (40, 60)
    assert up == pytest.approx(1 / 600.0, abs=1e-12)  # 常数图重采样仍为常数
    down = bilinear_resample(up, (20, 30))
    assert down == pytest.approx(P, abs=1e-12)


def test_bilinear_resample_identity_same_shape():
    P = synthetic.block_probability((16, 16), (4, 4, 8, 8), 0.5)
    assert np.array_equal(bilinear_resample(P, (16, 16)), P)


def test_resample_probability_renormalizes():
    """重采样后 sum 漂移 → 再归一化回 sum=1（technical-design §4.5）。"""
    P = synthetic.block_probability((48, 64), (10, 8, 32, 16), mass=0.6)
    R = resample_probability(P, (24, 32))
    assert R.shape == (24, 32)
    assert R.sum() == pytest.approx(1.0, abs=1e-9)
    # 质量集中区域重采样后仍持有主要质量（双线性不搬运远距离质量）
    assert R[4:12, 5:21].sum() > 0.5


def test_resample_to_original_uses_shape_mapping():
    P = synthetic.block_probability((240, 320), (100, 80, 60, 40), mass=0.4)
    mapping = {"original_shape": [480, 640], "inference_shape": [240, 320], "method": "aspect_preserving_bilinear"}
    out = resample_to_original(P, mapping)
    assert out.shape == (480, 640)
    assert out.sum() == pytest.approx(1.0, abs=1e-9)
    validate_probability(out)


def test_resample_to_original_rejects_unknown_inverse():
    P = synthetic.uniform_probability((10, 10))
    mapping = {
        "original_shape": [20, 20],
        "inference_shape": [10, 10],
        "method": "direct_anisotropic_to_target",
        "inverse": "fourier_phase_correlation",
    }
    with pytest.raises(UiAttentionError) as exc:
        resample_to_original(P, mapping)
    assert exc.value.code is ErrorCode.INVALID_DENSITY
    assert "不支持的逆变换方法" in exc.value.message


def test_resample_to_original_accepts_forward_method_names():
    """method 描述前向预处理（C2 登记名），不影响逆变换；缺省 inverse=bilinear。"""
    P = synthetic.uniform_probability((10, 10))
    mapping = {"original_shape": [20, 20], "inference_shape": [10, 10], "method": "direct_anisotropic_to_target"}
    out = resample_to_original(P, mapping)
    assert out.shape == (20, 20)
    assert out.sum() == pytest.approx(1.0, abs=1e-9)


def test_resample_to_original_rejects_nonzero_padding():
    P = synthetic.uniform_probability((10, 10))
    mapping = {
        "original_shape": [20, 20],
        "inference_shape": [10, 10],
        "padding": {"top": 2, "bottom": 2, "left": 0, "right": 0},
    }
    with pytest.raises(UiAttentionError) as exc:
        resample_to_original(P, mapping)
    assert "填充" in exc.value.message


def test_float32_input_upcast_to_float64():
    L = np.log(np.full((4, 4), 1 / 16.0, dtype=np.float32))
    P = log_density_to_probability(L)
    assert P.dtype == np.float64
