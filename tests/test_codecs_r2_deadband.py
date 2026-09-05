"""Tests for Rung R2: error-bounded deadband codec (eps 0.01 / 0.02 of range).

TDD: written BEFORE harness/codecs/deadband.py exists; must fail on import.
"""

import struct

import numpy as np
import pytest

from harness.codecs.deadband import DeadbandCodec
from harness.codecs.quantization import Q4Codec
from harness.datasets import generator

N = 2000
# Deterministic fixture: no RNG anywhere (codecs must be deterministic).
X = (np.sin(0.05 * np.arange(N, dtype=np.float64)) + 0.05 * np.cos(0.11 * np.arange(N))).astype(np.float32)


def _pairs(payload: bytes) -> tuple[int, float, list[tuple[int, float]]]:
    """Unpack [n u32LE][first f32LE][(idx u32LE, val f32LE)*]."""
    n, first = struct.unpack("<If", payload[:8])
    body = payload[8:]
    assert len(body) % 8 == 0, "pairs must be 8-byte (u32LE, f32LE) records"
    pairs = [struct.unpack("<If", body[i:i + 8]) for i in range(0, len(body), 8)]
    return n, first, pairs


def test_ramp_interpolation_exactness_both_eps():
    """Linear ramp decodes to (near-)exact ramp via linear interpolation."""
    ramp = np.linspace(0.0, 10.0, 500, dtype=np.float32)
    for eps in (0.01, 0.02):
        back = DeadbandCodec(eps).decode(DeadbandCodec(eps).encode(ramp))
        assert back.dtype == np.float32 and back.shape == ramp.shape
        assert float(np.max(np.abs(back.astype(np.float64) - ramp.astype(np.float64)))) < 1e-5


def test_constant_series_decodes_to_exact_constant():
    """Constant series: range 0 => keep first point only, decode exact constant."""
    x = np.full(64, -2.5, dtype=np.float32)
    for eps in (0.01, 0.02):
        c = DeadbandCodec(eps)
        payload = c.encode(x)
        assert len(payload) == 8, "constant series stores header only"
        back = c.decode(payload)
        assert back.dtype == np.float32 and back.shape == x.shape
        assert np.all(back == np.float32(-2.5))


def test_payload_header_le_and_idx_sorted():
    """Header is [n u32LE][first f32LE]; pair indices strictly increasing."""
    for eps in (0.01, 0.02):
        payload = DeadbandCodec(eps).encode(X)
        n, first = struct.unpack("<If", payload[:8])
        assert n == N and first == pytest.approx(float(X[0]))
        _, _, pairs = _pairs(payload)
        idxs = [i for i, _ in pairs]
        assert idxs == sorted(idxs) and len(set(idxs)) == len(idxs)
        assert all(0 < i < N for i in idxs)


def test_decode_length_always_n():
    """Decode output length always equals header N (never breaks _align_len)."""
    rng = np.random.default_rng(7)
    for eps in (0.01, 0.02):
        for x in (X, np.full(32, 1.0, dtype=np.float32),
                  rng.normal(0, 1, 300).astype(np.float32),
                  np.array([42.0], dtype=np.float32)):
            c = DeadbandCodec(eps)
            assert c.decode(c.encode(x)).shape == (x.shape[0],)


def test_empty_input_raises():
    """Empty input raises ValueError (never a silent empty payload)."""
    with pytest.raises(ValueError):
        DeadbandCodec(0.01).encode(np.array([], dtype=np.float32))


def test_bad_eps_raises():
    """Only eps in {0.01, 0.02} are valid codec-rows R2a/R2b."""
    with pytest.raises(ValueError):
        DeadbandCodec(0.05)


def test_truncated_payload_raises():
    """Truncated payloads raise loudly on decode."""
    c = DeadbandCodec(0.01)
    with pytest.raises(ValueError):
        c.decode(b"\x00" * 7)
    payload = c.encode(X)
    with pytest.raises(ValueError):
        c.decode(payload[:-3])


def test_encode_determinism():
    """Encoding twice yields identical bytes (no randomness)."""
    for eps in (0.01, 0.02):
        c = DeadbandCodec(eps)
        assert c.encode(X) == c.encode(X)


def test_error_bounded_pointwise():
    """Error-bounded guarantee: max|decode - x| <= eps * range on every pilot series."""
    makers = {"Spike": generator.make_spike, "Rhythm": generator.make_rhythm,
              "Drift": generator.make_drift, "Chaos": generator.make_chaos}
    for eps in (0.01, 0.02):
        c = DeadbandCodec(eps)
        for x in [X] + [np.ascontiguousarray(mk(s).x, dtype=np.float32)
                        for mk in makers.values() for s in (42, 43)]:
            back = c.decode(c.encode(x))
            span = float(x.max()) - float(x.min())
            assert float(np.max(np.abs(back.astype(np.float64) - x.astype(np.float64)))) <= eps * span + 1e-6


def test_ratio_bands_per_eps():
    """Frozen bands from pilot evidence: R2a [0.6, 5.5], R2b [0.8, 7.5].

    Deadband expands (ratio < 1) on noisy series since each kept point
    costs 8 B vs 4 B raw; bands honestly span expansion through smooth-gain.
    Evidence: .omo/evidence/compress-phase2-ladder/task-4-pass.log.
    """
    makers = {"Spike": generator.make_spike, "Rhythm": generator.make_rhythm,
              "Drift": generator.make_drift, "Chaos": generator.make_chaos}
    for eps, lo, hi in ((0.01, 0.6, 5.5), (0.02, 0.8, 7.5)):
        c = DeadbandCodec(eps)
        series = [X] + [np.ascontiguousarray(mk(s).x, dtype=np.float32)
                        for mk in makers.values() for s in (42, 43, 44, 45, 46)]
        for x in series:
            payload = c.encode(x)
            ratio = c.get_ratio(x, payload)
            assert ratio == pytest.approx(x.nbytes / len(payload), rel=1e-9)
            assert lo <= ratio <= hi, f"eps={eps} ratio {ratio:.3f} outside [{lo}, {hi}]"


def test_gradient_preservation_drift_ramp():
    """Drift ramp: RMSE(R2a) < RMSE(Q4) on the same seed (gradient preservation)."""
    x = np.ascontiguousarray(generator.make_drift(42).x, dtype=np.float32)
    rmse = lambda a, b: float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))
    r2a = DeadbandCodec(0.01).decode(DeadbandCodec(0.01).encode(x))
    q4 = Q4Codec().decode(Q4Codec().encode(x))
    assert rmse(r2a, x) < rmse(q4, x)
