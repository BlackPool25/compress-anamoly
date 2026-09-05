"""Acceptance tests for the seeded 4-morphology generator."""

import numpy as np
import pytest

from harness.datasets.generator import (
    N,
    TRAIN_END,
    make_chaos,
    make_drift,
    make_rhythm,
    make_spike,
)

MAKERS = {
    "spike": (make_spike, (1400, 1420)),
    "rhythm": (make_rhythm, (1300, 1450)),
    "drift": (make_drift, (1200, 1700)),
    "chaos": (make_chaos, (1350, 1500)),
}


def _zero_crossings(v: np.ndarray) -> int:
    """Count sign changes (zero crossings) in a 1-D signal."""
    s = np.sign(v)
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1]))


def test_length_split_labels_per_morphology():
    """N=2000, train slice all-clean, labels exactly on the anomaly window."""
    for name, (maker, (start, end)) in MAKERS.items():
        s = maker(7)
        assert s.x.shape == (N,) and s.x.dtype == float, name
        assert s.y.shape == (N,) and set(np.unique(s.y)) <= {0, 1}, name
        assert s.y[:TRAIN_END].sum() == 0, name
        assert s.y[start:end].sum() == end - start, name
        assert s.y.sum() == end - start, name
        assert s.meta["morphology"] == name and s.meta["seed"] == 7


def test_seed_reproducibility():
    """Seed 7 vs 7 byte-identical; 7 vs 8 differs, for every morphology."""
    for name, (maker, _) in MAKERS.items():
        a, b, c = maker(7), maker(7), maker(8)
        assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y), name
        assert not np.array_equal(a.x, c.x), name


def test_spike_amplitude_is_3_5_times_train_std():
    """Spike uplift equals 3.5 * train_std, recomputed independently here."""
    seed = 7
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    nominal = np.sin(0.05 * t) + rng.normal(0.0, 0.08, N)
    train_std = float(np.std(nominal[:TRAIN_END]))
    s = make_spike(seed)
    uplift = s.x[1400:1420] - nominal[1400:1420]
    assert np.allclose(uplift, 3.5 * train_std, rtol=1e-9)


def test_rhythm_period_doubles():
    """Zero-crossing period in the anomaly window is ~2x the nominal period."""
    s = make_rhythm(7)
    nom = s.x[800:950]  # clean, same length (150) as anomaly window
    ano = s.x[1300:1450]
    period_nom = 2 * len(nom) / _zero_crossings(nom)
    period_ano = 2 * len(ano) / _zero_crossings(ano)
    assert period_nom == pytest.approx(50, abs=12)
    assert period_ano / period_nom == pytest.approx(2.0, abs=0.6)


def test_drift_endpoint_delta():
    """Ramp endpoint delta is +1.5 +/- 0.15 vs same-seed nominal."""
    seed = 7
    rng = np.random.default_rng(seed)
    nominal = np.sin(0.05 * np.arange(N)) + rng.normal(0.0, 0.05, N)
    s = make_drift(seed)
    delta = float(np.mean(s.x[1690:1700] - nominal[1690:1700]))
    assert delta == pytest.approx(1.5, abs=0.15)


def test_chaos_variance_ratio():
    """Detrended variance ratio anomaly/clean lies in [40, 90] (expect ~64)."""
    s = make_chaos(7)
    resid = s.x - np.sin(0.05 * np.arange(N))
    ratio = float(np.var(resid[1350:1500]) / np.var(resid[:800]))
    assert 40.0 <= ratio <= 90.0


def test_rhythm_amplitude_bounds():
    """Rhythm stays within [-1.3, 1.3] after noise (impl asserts this too)."""
    for seed in (7, 8, 42):
        assert float(np.max(np.abs(make_rhythm(seed).x))) <= 1.3
