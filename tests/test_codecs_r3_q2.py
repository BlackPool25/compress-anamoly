"""Tests for R3c Q2 extreme quantization (4 levels, 2-bit codes). TDD: written first."""

import struct

import numpy as np
import pytest

from harness.codecs.quantization import Q2Codec

N = 1200
# Deterministic fixture: no RNG anywhere.
X = (np.sin(0.05 * np.arange(N, dtype=np.float64)) + 0.05 * np.cos(0.11 * np.arange(N))).astype(np.float32)

UCR_N = 79795
UCR_X = (np.sin(0.001 * np.arange(UCR_N, dtype=np.float64)) + 0.05 * np.cos(0.011 * np.arange(UCR_N))).astype(np.float32)


def test_q2_ratio_band_synth_n1200():
    """Q2 ratio ~15.58: N=1200 float32 -> raw 4800 B, payload 8 + 300 = 308 B."""
    c = Q2Codec()
    payload = c.encode(X)
    assert len(payload) == 8 + (N * 2 + 7) // 8
    assert len(payload) == 8 + 300
    assert c.get_ratio(X, payload) == pytest.approx(4800 / 308, rel=1e-9)
    assert 14.0 <= c.get_ratio(X, payload) <= 17.0


def test_q2_ratio_band_ucr_scale():
    """Band holds at UCR scale: N=79795 -> payload 8 + 19949 = 19957 B."""
    c = Q2Codec()
    payload = c.encode(UCR_X)
    assert len(payload) == 8 + (UCR_N * 2 + 7) // 8
    assert c.get_ratio(UCR_X, payload) == pytest.approx(UCR_X.nbytes / len(payload), rel=1e-9)
    assert 14.0 <= c.get_ratio(UCR_X, payload) <= 17.0


def test_q2_header_little_endian():
    """Q2 header is [min, max] as float32 little-endian."""
    payload = Q2Codec().encode(X)
    mn, mx = struct.unpack("<ff", payload[:8])
    assert mn == pytest.approx(float(X.min())) and mx == pytest.approx(float(X.max()))


def test_q2_four_level_vectors_high_first():
    """2-bit codes pack high-first: [0,1,2,3] -> 0x1B; two groups concatenate."""
    x = np.array([0.0, 1.0 / 3, 2.0 / 3, 1.0], dtype=np.float32)
    body = Q2Codec().encode(x)[8:]
    assert body == bytes([(0 << 6) | (1 << 4) | (2 << 2) | 3])
    x8 = np.array([0.0, 1.0 / 3, 2.0 / 3, 1.0, 0.0, 1.0 / 3, 2.0 / 3, 1.0], dtype=np.float32)
    assert Q2Codec().encode(x8)[8:] == bytes([0x1B, 0x1B])


def test_q2_odd_n_pad_and_align():
    """Odd N pads final 2-bit slots with zero; decode returns padded len; _align_len trims."""
    from harness.run_harness import _align_len

    odd = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    payload = Q2Codec().encode(odd)
    assert len(payload) == 8 + 1  # ceil(3*2/8) = 1 byte
    # codes [0, rint(1.5)=2, 3] + zero pad -> (0<<6)|(2<<4)|(3<<2)|0 = 44
    assert payload[8] == (0 << 6) | (2 << 4) | (3 << 2) | 0
    back = Q2Codec().decode(payload)
    assert back.shape == (4,)  # no length field: pad code decodes too
    assert np.allclose(back[:3], [0.0, 0.5, 1.0], atol=1 / 3 + 1e-6)
    aligned = _align_len(back, odd.shape[0])
    assert aligned.shape == odd.shape


def test_q2_decode_shapes_dtypes_and_accuracy():
    """Even-N round-trip matches shape/dtype; error within half a 2-bit step."""
    c = Q2Codec()
    back = c.decode(c.encode(X))
    assert back.dtype == np.float32 and back.shape == X.shape
    span = float(X.max()) - float(X.min())
    assert np.max(np.abs(back.astype(np.float64) - X.astype(np.float64))) <= span * (1 / 3) / 2 + 1e-6


def test_q2_encode_determinism():
    """Encoding twice yields identical bytes."""
    c = Q2Codec()
    assert c.encode(X) == c.encode(X)


def test_q2_truncated_header_raises():
    """Payloads shorter than the 8-byte header raise on decode."""
    with pytest.raises(ValueError):
        Q2Codec().decode(b"\x00" * 7)


def test_q2_constant_input_returns_constant():
    """min == max edge: all-zero body, decode returns the constant, ratio reported."""
    x = np.full(16, 3.25, dtype=np.float32)
    payload = Q2Codec().encode(x)
    assert set(payload[8:]) == {0}
    back = Q2Codec().decode(payload)
    assert back.dtype == np.float32
    assert np.all(back == np.float32(3.25))
    assert Q2Codec().get_ratio(x, payload) > 1.0


def test_q2_empty_input_raises():
    """Empty input raises (same contract as Q8/Q4: min/max of empty is undefined)."""
    with pytest.raises(ValueError):
        Q2Codec().encode(np.array([], dtype=np.float32))
