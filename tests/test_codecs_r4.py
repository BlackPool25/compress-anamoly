"""Acceptance tests for Rung R4: SAX symbolic codec (PAA window=8, A=8)."""

import numpy as np
import pytest
from scipy.stats import norm

from harness.codecs.symbolic import (
    ALPHABET_SIZE,
    BIN_MIDPOINTS,
    BREAKPOINTS,
    PAA_WINDOW,
    SAXCodec,
)

N = 2000


def _codec(seed: int = 7) -> tuple[SAXCodec, np.ndarray]:
    """Codec fitted on a seeded nominal train slice + float32 test series."""
    rng = np.random.default_rng(seed)
    train = (np.sin(0.05 * np.arange(800)) + rng.normal(0.0, 0.08, 800)).astype(np.float32)
    test = (np.sin(0.05 * np.arange(N)) + rng.normal(0.0, 0.08, N)).astype(np.float32)
    c = SAXCodec()
    c.fit(train)
    return c, test


def test_breakpoints_match_gaussian_quantiles():
    """Breakpoints are exactly norm.ppf(linspace(1/8, 7/8, 7))."""
    assert np.allclose(BREAKPOINTS, norm.ppf(np.linspace(1 / 8, 7 / 8, 7)))
    assert BREAKPOINTS.shape == (7,) and np.all(np.diff(BREAKPOINTS) > 0)


def test_token_alphabet_range():
    """All tokens in [0, 7]; one byte per token."""
    c, test = _codec()
    payload = c.encode(test)
    toks = np.frombuffer(payload, dtype=np.uint8)
    assert toks.min() >= 0 and toks.max() <= ALPHABET_SIZE - 1
    assert len(payload) == N // PAA_WINDOW


def test_decode_length_matches_input():
    """Path A reconstruction length == input length."""
    c, test = _codec()
    assert c.decode(c.encode(test)).shape == test.shape


def test_path_a_piecewise_constant_over_windows():
    """Path A output is constant within each window of 8."""
    c, test = _codec()
    rec = c.decode(c.encode(test))
    blocks = rec.reshape(-1, PAA_WINDOW)
    assert np.all(blocks == blocks[:, :1])


def test_path_b_length_is_n_over_8():
    """Path B int token array has length N/8."""
    c, test = _codec()
    toks = c.decode_tokens(c.encode(test))
    assert toks.shape == (N // PAA_WINDOW,)
    assert toks.dtype.kind == "i"


def test_ratio_in_frozen_band():
    """float32 input realizes ~32x (8000B/250B) inside [14.0, 34.0]."""
    c, test = _codec()
    ratio = c.get_ratio(test, c.encode(test))
    assert 14.0 <= ratio <= 34.0
    assert ratio == pytest.approx(32.0, rel=0.05)


def test_ratio_violation_raises_with_message():
    """Out-of-band ratios raise ValueError with the ratio in the message."""
    c, test = _codec()
    with pytest.raises(ValueError, match="outside frozen band"):
        c.get_ratio(test[:8], bytes(8))  # 32B/8B = 4x, too low
    with pytest.raises(ValueError, match="outside frozen band"):
        c.get_ratio(test, bytes(1))  # 8000B/1B = 8000x, too high


def test_determinism():
    """Encoding twice gives byte-identical payloads (no RNG in module)."""
    c, test = _codec()
    assert c.encode(test) == c.encode(test)


def test_decode_decode_tokens_consistency():
    """Path A equals Path B tokens mapped through bin midpoints + train stats."""
    c, test = _codec()
    payload = c.encode(test)
    toks = c.decode_tokens(payload)
    expected = np.repeat(BIN_MIDPOINTS[toks] * c.std + c.mean, PAA_WINDOW)
    assert np.allclose(c.decode(payload), expected)


def test_train_test_norm_separation():
    """Test data is encoded with TRAIN stats; different train stats change tokens."""
    _, test = _codec()
    c1, c2 = SAXCodec(), SAXCodec()
    c1.fit(np.zeros(800, dtype=np.float32))
    c2.fit(np.full(800, 10.0, dtype=np.float32))
    assert c1.mean != c2.mean
    assert c1.encode(test) != c2.encode(test)
    # Same codec, same input -> same tokens (stats frozen, not recomputed).
    assert c1.encode(test) == c1.encode(test)


def test_encode_before_fit_raises():
    """Encoding/decoding without fit raises (no silent test-stat leakage)."""
    c = SAXCodec()
    x = np.zeros(16, dtype=np.float32)
    with pytest.raises(RuntimeError):
        c.encode(x)
    with pytest.raises(RuntimeError):
        c.decode_tokens(b"\x00\x00")
