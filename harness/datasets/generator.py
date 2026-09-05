"""Seeded 4-morphology time-series generator (PLAN-FREEZE: formulas frozen).

Every morphology produces ``N = 2000`` points. The first 800 points are the
nominal training slice (always label 0); the remaining 1200 points are the
test slice containing exactly one anomaly segment.

All randomness flows through ``np.random.default_rng(seed)`` threaded through
every function. There are no bare ``np.random.*`` calls in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

N: int = 2000
TRAIN_END: int = 800

SPIKE_RANGE: tuple[int, int] = (1400, 1420)
RHYTHM_RANGE: tuple[int, int] = (1300, 1450)
DRIFT_RANGE: tuple[int, int] = (1200, 1700)
CHAOS_RANGE: tuple[int, int] = (1350, 1500)

#: Frozen rhythm base frequency: T0 = 50 samples.
RHYTHM_F0: float = 0.02


@dataclass
class TimeSeriesSample:
    """One generated series: signal, binary labels, and provenance meta."""

    x: np.ndarray
    y: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def _labels(start: int, end: int) -> np.ndarray:
    """Binary label vector with ones on ``[start:end)``."""
    y = np.zeros(N, dtype=int)
    y[start:end] = 1
    return y


def _meta(morphology: str, seed: int, start: int, end: int) -> dict[str, Any]:
    """Provenance meta shared by all morphologies."""
    return {
        "morphology": morphology,
        "seed": seed,
        "n": N,
        "train_range": (0, TRAIN_END),
        "test_range": (TRAIN_END, N),
        "anomaly_range": (start, end),
    }


def make_spike(seed: int) -> TimeSeriesSample:
    """Impulsive spike: ``sin(0.05t) + N(0, 0.08)``, ``[1400:1420] += 3.5*sigma``.

    Order frozen: inject noise first, then compute ``sigma`` as the std of
    the nominal train slice (first 800 pts), then inject the anomaly.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    x = np.sin(0.05 * t) + rng.normal(0.0, 0.08, N)
    sigma = float(np.std(x[:TRAIN_END]))
    start, end = SPIKE_RANGE
    x[start:end] += 3.5 * sigma
    return TimeSeriesSample(x=x, y=_labels(start, end),
                            meta={**_meta("spike", seed, start, end),
                                  "sigma": sigma, "noise_std": 0.08})


def make_rhythm(seed: int) -> TimeSeriesSample:
    """Rhythmic arrhythmia: base ``sin(2*pi*f0*t)``, ``f0 = 0.02`` frozen.

    Anomaly on ``[1300:1450]`` halves the frequency (period 50 -> 100).
    No phase-jitter term: the TASK_01 title mentions phase jitter but the
    formula has none, so the formula is implemented exactly (see
    docs/ARCHITECTURE.md; reversible by adding an explicit jitter term).

    Build order frozen: construct the anomaly on the noiseless base first
    and assert ``max|y| <= 1.0``, then add ``N(0, 0.05)`` observation noise
    and assert ``max|y| <= 1.3``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    start, end = RHYTHM_RANGE
    freq = np.full(N, RHYTHM_F0)
    freq[start:end] = 0.5 * RHYTHM_F0
    base = np.sin(2.0 * np.pi * freq * t)
    assert float(np.max(np.abs(base))) <= 1.0, "noiseless rhythm base escaped [-1, 1]"
    x = base + rng.normal(0.0, 0.05, N)
    assert float(np.max(np.abs(x))) <= 1.3, "noisy rhythm signal escaped [-1.3, 1.3]"
    return TimeSeriesSample(x=x, y=_labels(start, end),
                            meta={**_meta("rhythm", seed, start, end),
                                  "f0": RHYTHM_F0, "noise_std": 0.05})


def make_drift(seed: int) -> TimeSeriesSample:
    """Subtle trend drift: ``sin(0.05t) + N(0, 0.05)``, ramp on ``[1200:1700]``.

    Ramp is ``0.003 * (t - 1200)``: max delta +1.5 at the endpoint.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    x = np.sin(0.05 * t) + rng.normal(0.0, 0.05, N)
    start, end = DRIFT_RANGE
    x[start:end] += 0.003 * (t[start:end] - start)
    return TimeSeriesSample(x=x, y=_labels(start, end),
                            meta={**_meta("drift", seed, start, end),
                                  "noise_std": 0.05, "ramp_rate": 0.003})


def make_chaos(seed: int) -> TimeSeriesSample:
    """Variance burst: ``sin(0.05t)`` with noise std ``0.05 -> 0.40`` on ``[1350:1500]``."""
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    base = np.sin(0.05 * t)
    noise = rng.normal(0.0, 0.05, N)
    start, end = CHAOS_RANGE
    noise[start:end] = rng.normal(0.0, 0.40, end - start)
    x = base + noise
    return TimeSeriesSample(x=x, y=_labels(start, end),
                            meta={**_meta("chaos", seed, start, end),
                                  "noise_std_nominal": 0.05,
                                  "noise_std_anomaly": 0.40})
