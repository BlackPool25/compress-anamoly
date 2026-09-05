"""Tests for codecs R0 (lossless) and R3 (Q8/Q4 scalar quantization)."""

import struct

import numpy as np
import pytest

from harness.codecs.lossless import LosslessCodec
from harness.codecs.quantization import Q4Codec, Q8Codec

N = 2000
# Deterministic fixture: no RNG anywhere (codecs must be deterministic).
X = (np.sin(0.05 * np.arange(N, dtype=np.float64)) + 0.05 * np.cos(0.11 * np.arange(N))).astype(np.float32)


def test_r0_roundtrip_bit_exact_and_ratio_one():
    """R0 is a bit-exact passthrough: bytes equal float32 dump, ratio == 1.0."""
    c = LosslessCodec()
    payload = c.encode(X)
    assert payload == X.astype(np.float32).tobytes()
    back = c.decode(payload)
    assert back.dtype == np.float32 and back.shape == X.shape
    assert np.array_equal(back, X.astype(np.float32))
    assert c.get_ratio(X.astype(np.float32), payload) == 1.0


def test_r0_ragged_payload_raises():
    """R0 decode rejects payloads whose length is not a multiple of 4."""
    with pytest.raises(ValueError):
        LosslessCodec().decode(b"\x00" * 7)


def test_q8_ratio_band():
    """Q8 ratio ~3.98: N=2000 float32 -> raw 8000 B, payload 8 + 2000 = 2008 B."""
    c = Q8Codec()
    payload = c.encode(X)
    assert len(payload) == 8 + N
    assert c.get_ratio(X, payload) == pytest.approx(8000 / 2008, rel=1e-9)
    assert 3.75 <= c.get_ratio(X, payload) <= 4.15


def test_q4_ratio_band():
    """Q4 ratio ~7.94: N=2000 float32 -> raw 8000 B, payload 8 + 1000 = 1008 B."""
    c = Q4Codec()
    payload = c.encode(X)
    assert len(payload) == 8 + N // 2
    assert c.get_ratio(X, payload) == pytest.approx(8000 / 1008, rel=1e-9)
    assert 7.7 <= c.get_ratio(X, payload) <= 8.1


def test_q8_header_little_endian():
    """Q8 header is [min, max] as float32 little-endian."""
    payload = Q8Codec().encode(X)
    mn, mx = struct.unpack("<ff", payload[:8])
    assert mn == pytest.approx(float(X.min())) and mx == pytest.approx(float(X.max()))


def test_q4_nibble_order_and_odd_pad():
    """Q4 packs high-nibble-first; odd N zero-pads the final low nibble."""
    x = np.array([0.0, 1.0 / 3, 2.0 / 3, 1.0], dtype=np.float32)
    body = Q4Codec().encode(x)[8:]
    assert body == bytes([(0 << 4) | 5, (10 << 4) | 15])
    odd = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    odd_body = Q4Codec().encode(odd)[8:]
    assert len(odd_body) == 2 and odd_body[1] & 0x0F == 0
    back = Q4Codec().decode(Q4Codec().encode(odd))
    assert back.shape == (4,)  # no length field: pad nibble decodes too
    assert np.allclose(back[:3], [0.0, 0.5, 1.0], atol=1 / 15 + 1e-6)


def test_decode_shapes_dtypes_and_accuracy():
    """Decodes match shape/dtype; Q8 within half an 8-bit step, Q4 half a nibble step."""
    for codec, step in ((Q8Codec(), 1 / 255), (Q4Codec(), 1 / 15)):
        back = codec.decode(codec.encode(X))
        assert back.dtype == np.float32 and back.shape == X.shape
        span = float(X.max()) - float(X.min())
        assert np.max(np.abs(back.astype(np.float64) - X.astype(np.float64))) <= span * step / 2 + 1e-6


def test_encode_determinism():
    """Encoding twice yields identical bytes (no randomness)."""
    for codec in (LosslessCodec(), Q8Codec(), Q4Codec()):
        assert codec.encode(X) == codec.encode(X)


def test_truncated_header_raises():
    """Payloads shorter than the 8-byte header raise on decode."""
    for codec in (Q8Codec(), Q4Codec()):
        with pytest.raises(ValueError):
            codec.decode(b"\x00" * 7)


def test_constant_input_returns_constant():
    """min == max edge: all-zero codes, decode returns the constant, ratio reported."""
    x = np.full(16, 3.25, dtype=np.float32)
    for codec in (Q8Codec(), Q4Codec()):
        payload = codec.encode(x)
        assert set(payload[8:]) == {0}
        back = codec.decode(payload)
        assert back.dtype == np.float32
        assert np.all(back == np.float32(3.25))
        assert codec.get_ratio(x, payload) > 1.0
