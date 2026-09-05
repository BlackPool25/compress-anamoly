"""Tests for R1 Gorilla exact XOR bit-packing codec. TDD: written first (red)."""

import struct

import numpy as np
import pytest

from harness.codecs.gorilla import GORILLA_RATIO_BAND, GorillaCodec

N = 1200
# Deterministic pilot fixtures: no unseeded RNG in module scope.
X = (np.sin(0.05 * np.arange(N, dtype=np.float64)) + 0.05 * np.cos(0.11 * np.arange(N))).astype(np.float32)

UCR_N = 79795
UCR_X = (np.sin(0.001 * np.arange(UCR_N, dtype=np.float64)) + 0.05 * np.cos(0.011 * np.arange(UCR_N))).astype(np.float32)


def _assert_bit_identical(a: np.ndarray, b: np.ndarray):
    """Byte-identical incl. NaN payloads: compare uint32 views, not ==."""
    assert a.dtype == np.float32 and b.dtype == np.float32
    assert a.shape == b.shape
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_gorilla_header_little_endian():
    """Header is [count u32LE][first f32LE]: 8 bytes, LE throughout."""
    x = np.array([1.5, 2.5, 2.5], dtype=np.float32)
    payload = GorillaCodec().encode(x)
    assert len(payload) >= 8
    (count,) = struct.unpack("<I", payload[:4])
    (first,) = struct.unpack("<f", payload[4:8])
    assert count == 3
    assert first == pytest.approx(1.5)
    assert struct.pack("<I", count) == payload[:4]
    assert struct.pack("<f", np.float32(1.5)) == payload[4:8]


def test_gorilla_constant_series_control_bits_all_zero():
    """Constant series: all XORs zero -> body is zero bytes only."""
    x = np.full(16, 3.25, dtype=np.float32)
    payload = GorillaCodec().encode(x)
    (count,) = struct.unpack("<I", payload[:4])
    assert count == 16
    # 15 zero control bits -> ceil(15/8) = 2 zero bytes
    assert payload[8:] == b"\x00\x00"
    _assert_bit_identical(GorillaCodec().decode(payload), x)


def test_gorilla_100_random_roundtrips_bit_identical():
    """100 seeded random series incl. constants/NaN: byte-identical round-trip."""
    rng = np.random.default_rng(42)
    c = GorillaCodec()
    for i in range(100):
        kind = i % 5
        if kind == 0:
            x = np.full(64, float(i), dtype=np.float32)
        elif kind == 1:
            x = np.zeros(64, dtype=np.float32)
            x[::7] = np.float32("nan")
            x[3] = np.float32("inf")
            x[5] = np.float32("-inf")
        elif kind == 2:
            raw = rng.integers(0, 2**32, size=64, dtype=np.uint32)
            x = raw.view(np.float32)
        elif kind == 3:
            x = (rng.normal(0, 1, size=64)).astype(np.float32)
        else:
            # Smooth series compresses well; random walk stresses leading/trail reuse.
            x = np.cumsum(rng.normal(0, 0.01, size=64)).astype(np.float32)
        _assert_bit_identical(c.decode(c.encode(x)), x)


def test_gorilla_nan_payloads_preserved():
    """Distinct NaN payloads survive bit-exactly (struct uint32 compare)."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    payloads = set()
    for bits in (0x7FC00001, 0x7FC00002, 0x7F800001):
        y = x.copy()
        y.view(np.uint32)[2] = np.uint32(bits)  # via view: float64 interp canonicalizes NaN payloads
        back = GorillaCodec().decode(GorillaCodec().encode(y))
        assert back.view(np.uint32)[2] == np.uint32(bits)
        payloads.add(GorillaCodec().encode(y))
    assert len(payloads) == 3  # distinct payloads stay distinct


def test_gorilla_ratio_band_pilot():
    """Frozen band holds on synth pilot + UCR-scale series."""
    c = GorillaCodec()
    lo, hi = GORILLA_RATIO_BAND
    for x in (X, UCR_X):
        r = c.get_ratio(np.ascontiguousarray(x, dtype=np.float32), c.encode(x))
        assert lo <= r <= hi, f"ratio {r} outside frozen band {(lo, hi)}"
    # Ratio definition frozen: x_raw.nbytes / len(payload).
    p = c.encode(X)
    assert c.get_ratio(X, p) == pytest.approx(X.nbytes / len(p), rel=1e-12)


def test_gorilla_empty_input_raises():
    """Empty input raises ValueError."""
    with pytest.raises(ValueError):
        GorillaCodec().encode(np.array([], dtype=np.float32))


def test_gorilla_truncated_payload_raises():
    """Truncated header/body raises ValueError (never silent)."""
    c = GorillaCodec()
    with pytest.raises(ValueError):
        c.decode(b"\x00" * 7)
    payload = c.encode(X)
    with pytest.raises(ValueError):
        c.decode(payload[:8])
    with pytest.raises(ValueError):
        c.decode(payload[:-1])


def test_gorilla_encode_determinism():
    """Encoding twice yields identical bytes."""
    c = GorillaCodec()
    assert c.encode(X) == c.encode(X)


def test_gorilla_bitflip_mismatch_loud():
    """Flipping one body bit never decodes silently identical."""
    c = GorillaCodec()
    x = (np.sin(0.05 * np.arange(64, dtype=np.float64))).astype(np.float32)
    payload = bytearray(c.encode(x))
    payload[8] ^= 0x01
    try:
        back = c.decode(bytes(payload))
    except ValueError:
        return
    assert not np.array_equal(back.view(np.uint32), x.view(np.uint32))
