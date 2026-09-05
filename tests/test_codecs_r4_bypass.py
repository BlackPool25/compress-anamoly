"""Acceptance tests for Rung R4-bypass: energy-bypass SAX + D4-hybrid (TDD).

Contract under test (see ``harness/codecs/bypass_sax.py``):
- per-chunk RMS energy gate, theta = 99.5th pct of nominal-train energies
- nominal chunk -> 1-byte SAX symbol; flagged -> [0xFF][sym][peak f16LE][off]
- ``decode`` = piecewise-constant + peak impulses; ``decode_tokens`` = tokens+mask
- D4 stays 8x8 (untouched); hybrid = max-train-surprisal + BYPASS_MARGIN (0.5)
"""

import numpy as np
import pytest

from harness.codecs.bypass_sax import (
    BYPASS_MARGIN,
    BypassSAXCodec,
    hybrid_token_surprisals,
)
from harness.codecs.symbolic import PAA_WINDOW, SAXCodec
from harness.datasets import generator
from harness.detectors.direct_symbolic import DirectSymbolicDetector
from harness.metrics.frozen_evaluator import (
    calibrate_frozen_threshold,
    evaluate_event_f1,
)

SEEDS = [42, 43, 44, 45, 46]


def _spike_slices(seed):
    """(train, test) float32 slices for Spike, R4-truncated like the runner."""
    s = generator.make_spike(seed)
    x = np.ascontiguousarray(s.x, dtype=np.float32).ravel()
    y = np.asarray(s.y, dtype=int).ravel()
    xtr, xte = x[: generator.TRAIN_END], x[generator.TRAIN_END :]
    yte = y[generator.TRAIN_END :]
    n = lambda a: (a.shape[0] // PAA_WINDOW) * PAA_WINDOW
    return xtr[: n(xtr)], xte[: n(xte)], yte[: n(yte)]


def _hybrid_run(seed):
    """Full pilot wiring: bypass fit/encode + D4-on-nominal + hybrid scores."""
    xtr, xte, yte = _spike_slices(seed)
    c = BypassSAXCodec()
    c.fit(xtr)
    toks_tr, mask_tr = c.decode_tokens(c.encode(xtr))
    d4 = DirectSymbolicDetector()
    d4.fit(toks_tr[~mask_tr])
    toks_te, mask_te = c.decode_tokens(c.encode(xte))
    tok_tr = hybrid_token_surprisals(d4.transitions, toks_tr, mask_tr)
    tok_te = hybrid_token_surprisals(d4.transitions, toks_te, mask_te)
    s_tr = np.repeat(tok_tr, PAA_WINDOW)
    s_te = np.repeat(tok_te, PAA_WINDOW)
    # Contract tau: MAX (100.0) of train hybrid scores -- the margin puts
    # flagged chunks 0.5 above the novelty ceiling, so max-tau keeps every
    # nominal score below tau while flagged chunks stay detectable.
    tau = calibrate_frozen_threshold(s_tr, 100.0)
    return c, d4, xte, yte, s_te, tau, toks_tr, mask_tr, toks_te, mask_te


def test_bypass_margin_frozen_at_half_nat():
    """BYPASS_MARGIN is exactly 0.5 (Todo 9 runner contract)."""
    assert BYPASS_MARGIN == 0.5


def test_spike_ratio_ge_20x_min_bytes():
    """Spike test slice (1200 f32 = 4800B) compresses to <= 240B on all seeds."""
    for seed in SEEDS:
        xtr, xte, _ = _spike_slices(seed)
        c = BypassSAXCodec()
        c.fit(xtr)
        payload = c.encode(xte)
        ratio = c.get_ratio(xte, payload)
        assert ratio >= 20.0, f"seed {seed}: ratio {ratio:.2f}x < 20x ({len(payload)}B)"
        assert len(payload) <= xte.nbytes / 20.0


def test_spike_event_f1_restoration_signal():
    """Hybrid restores Spike Event-F1 (mean >= 0.75) where plain D4 collapses."""
    hybrids, plains = [], []
    for seed in SEEDS:
        c, d4, xte, yte, s_te, tau, toks_tr, mask_tr, toks_te, mask_te = _hybrid_run(seed)
        hybrids.append(evaluate_event_f1(yte, s_te, tau))
        # Plain baseline: SAX-equivalent D4 on ALL train tokens, no bypass.
        d0 = DirectSymbolicDetector()
        d0.fit(toks_tr)
        s0 = d0.score(toks_te)
        tau0 = calibrate_frozen_threshold(d0.score(toks_tr))
        plains.append(evaluate_event_f1(yte, s0, tau0))
    assert float(np.mean(hybrids)) >= 0.75, f"mean hybrid Event-F1 {np.mean(hybrids):.3f}"
    assert all(h > p for h, p in zip(hybrids, plains)), f"{hybrids} vs {plains}"


def test_nominal_chunks_use_plain_sax_symbols():
    """Nominal payload bytes are SAX symbols in [0, 7]; 0xFF never a symbol."""
    rng = np.random.default_rng(0)
    train = (np.sin(0.05 * np.arange(800)) + rng.normal(0, 0.08, 800)).astype(np.float32)
    c = BypassSAXCodec()
    c.fit(train)
    assert c.theta is not None and np.isfinite(c.theta)
    payload = c.encode(train[:800])
    toks, mask = c.decode_tokens(payload)
    assert toks.min() >= 0 and toks.max() <= 7
    assert mask.dtype == bool and toks.shape == mask.shape == (100,)
    # Framing is positional (parsed sequentially), never scanned for markers,
    # so payload bytes are only asserted structurally here; 0xFF-collision
    # parsing is pinned by test_ff_collision_peak_round_trips.


def test_ff_collision_peak_round_trips():
    """A peak whose f16LE bytes contain 0xFF parses positionally, losslessly."""
    import struct

    peak_val = 31.984375  # f16 bits 0x4FFF, LE bytes FF 4F, exactly representable
    assert struct.pack("<e", peak_val) == b"\xff\x4f"
    rng = np.random.default_rng(1)
    train = (np.sin(0.05 * np.arange(800)) + rng.normal(0, 0.08, 800)).astype(np.float32)
    c = BypassSAXCodec()
    c.fit(train)
    x = np.zeros(64, dtype=np.float32)
    x[7] = peak_val  # chunk 0, offset 7: RMS = peak/sqrt(8) >> theta -> flagged
    payload = c.encode(x)
    toks, mask = c.decode_tokens(payload)
    assert toks.shape == (8,) and mask.shape == (8,)
    assert bool(mask[0]) and not mask[1:].any()
    rec = c.decode(payload)
    assert rec.shape == (64,)
    assert rec[7] == pytest.approx(peak_val, rel=1e-3)
    # A resync bug at the embedded 0xFF byte would corrupt every later chunk.
    assert toks[1:].min() >= 0 and toks[1:].max() <= 7


def test_all_nominal_series_zero_markers_sax_parity():
    """Constant series: energies tie at theta -> zero markers, byte-exact SAX."""
    x = np.full(800, 0.25, dtype=np.float32)
    c = BypassSAXCodec()
    c.fit(x)
    payload = c.encode(x)
    toks, mask = c.decode_tokens(payload)
    assert not mask.any(), "tied energies must stay nominal (strict > gate)"
    assert 0xFF not in payload
    sax = SAXCodec()
    sax.fit(x)
    assert payload == sax.encode(x), "all-nominal payload must equal SAX byte-exact"
    assert np.array_equal(c.decode(payload), sax.decode(payload))


def test_nominal_marker_rate_stays_sparse():
    """Fresh nominal sine: flagged rate < 2% (design ~0.5% above p99.5)."""
    rng = np.random.default_rng(7)
    train = (np.sin(0.05 * np.arange(800)) + rng.normal(0, 0.08, 800)).astype(np.float32)
    test = (np.sin(0.05 * np.arange(800, 2000)) + rng.normal(0, 0.08, 1200)).astype(np.float32)
    c = BypassSAXCodec()
    c.fit(train)
    _, mask = c.decode_tokens(c.encode(test))
    assert mask.mean() < 0.02, f"marker rate {mask.mean():.3f} too high on nominal"


def test_hybrid_nominal_matches_d4_exactly():
    """All-clear mask: helper token scores == D4.score resampled (drop-in)."""
    _, _, _, _, _, _, toks_tr, mask_tr, toks_te, _ = _hybrid_run(42)
    d4 = DirectSymbolicDetector()
    d4.fit(toks_tr[~mask_tr])
    clear = np.zeros_like(toks_te, dtype=bool)
    tok = hybrid_token_surprisals(d4.transitions, toks_te, clear)
    assert np.array_equal(np.repeat(tok, PAA_WINDOW), d4.score(toks_te))


def test_flagged_scores_model_max_plus_margin():
    """Flagged chunks score novelty-ceiling + 0.5, dominating every nominal score."""
    _, d4, _, _, _, _, toks_tr, mask_tr, toks_te, mask_te = _hybrid_run(42)
    assert mask_te.any(), "Spike test must flag at least one chunk"
    tok = hybrid_token_surprisals(d4.transitions, toks_te, mask_te)
    expected = float(-np.log(d4.transitions.min() + 1e-5)) + BYPASS_MARGIN
    assert np.all(tok[mask_te] == pytest.approx(expected))
    assert bool((tok[~mask_te].max() < tok[mask_te].min()))


def test_hybrid_determinism():
    """No RNG anywhere: re-encode + re-score are byte-identical."""
    c, d4, xte, _, _, _, toks_tr, mask_tr, toks_te, mask_te = _hybrid_run(43)
    assert c.encode(xte) == c.encode(xte)
    a = hybrid_token_surprisals(d4.transitions, toks_te, mask_te)
    b = hybrid_token_surprisals(d4.transitions, toks_te, mask_te)
    assert np.array_equal(a, b)


def test_decode_piecewise_constant_plus_impulse():
    """Nominal blocks constant; flagged blocks constant except offset impulse."""
    c, _, xte, _, _, _, _, _, toks_te, mask_te = _hybrid_run(42)
    rec = c.decode(c.encode(xte))
    assert rec.shape == xte.shape
    blocks = rec.reshape(-1, PAA_WINDOW)
    for i, b in enumerate(blocks):
        if not mask_te[i]:
            assert np.all(b == b[0]), f"nominal block {i} not constant"
        else:
            assert len(np.unique(b)) == 2, f"flagged block {i} must be base+impulse"


def test_empty_and_all_flagged_handled_loudly():
    """Empty payload decodes empty; all-flagged series round-trips, ratio unbanded."""
    c = BypassSAXCodec()
    rng = np.random.default_rng(3)
    train = (np.sin(0.05 * np.arange(800)) + rng.normal(0, 0.08, 800)).astype(np.float32)
    c.fit(train)
    assert c.decode(b"").shape == (0,)
    t, m = c.decode_tokens(b"")
    assert t.shape == (0,) and m.shape == (0,)
    x = np.full(64, 50.0, dtype=np.float32)  # every chunk flagged
    p = c.encode(x)
    _, mask = c.decode_tokens(p)
    assert bool(mask.all())
    assert c.decode(p).shape == (64,)
    assert np.isfinite(c.get_ratio(x, p))  # no frozen band: never raises on count
    with pytest.raises(ValueError):
        c.decode(b"\xff")  # truncated bypass record raises loudly
    with pytest.raises(ValueError):
        c.decode(b"\xff\x03")  # still truncated


def test_encode_before_fit_raises():
    """Encode/decode_tokens without fit raise (no silent theta leakage)."""
    c = BypassSAXCodec()
    with pytest.raises(RuntimeError):
        c.encode(np.zeros(16, dtype=np.float32))
    with pytest.raises(RuntimeError):
        c.decode_tokens(b"\x03\x04")


def test_odd_length_raises_like_sax():
    """Length % 8 != 0 raises (runner truncates first, same rule as SAX)."""
    c = BypassSAXCodec()
    c.fit(np.zeros(800, dtype=np.float32))
    with pytest.raises(ValueError, match="divisible"):
        c.encode(np.zeros(100, dtype=np.float32))


def test_raw_floats_only_no_q2_stacking():
    """Integer token arrays are rejected: bypass consumes raw floats only."""
    c = BypassSAXCodec()
    c.fit(np.zeros(800, dtype=np.float32))
    with pytest.raises(ValueError, match="raw float"):
        c.encode(np.arange(64, dtype=np.int64) % 8)
    with pytest.raises(ValueError):
        c.fit(np.arange(800, dtype=np.int64) % 8)
