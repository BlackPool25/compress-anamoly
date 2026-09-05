"""TCN frozen-tau integration (Todo 8, TDD-red).

- tau_TCN = p99 of TCN.score(R0-train) per (seed, dataset), reused across
  ALL codecs (mirrors PCA/IF; NOT the D4 max-tau rule).
- TCN trains once per (dataset, seed) on R0-train; never on compressed data.
- Q4 staircase inflates normal residuals vs R0 on the Drift pilot
  (directional, documents the H2 mechanism).
- Train windows capped by the FROZEN subsample formula so every worker
  produces identical trains.
"""

import math

import numpy as np

from harness.datasets.generator import TRAIN_END, make_drift, make_spike
from harness.codecs.quantization import Q4Codec
from harness.detectors.tcn_autoencoder import (
    MAX_TRAIN_WINDOWS,
    TCNAutoencoder,
    frozen_subsample_indices,
)
from harness.metrics.frozen_evaluator import calibrate_frozen_threshold


def _f32(x: np.ndarray) -> np.ndarray:
    """Cast a series to canonical float32 (runner rule)."""
    return np.ascontiguousarray(x, dtype=np.float32).ravel()


def test_tcn_tau_reused_across_codecs_object_identity():
    """Runner path reuses the SAME tau object for TCN across codecs."""
    from harness import run_harness

    rows = run_harness.run_cell("Spike", 42, ["R0", "Q4"])
    tcn = [r for r in rows if r["detector"] == "TCN"]
    assert len(tcn) == 2  # one per codec: R0 + Q4
    assert {r["codec"] for r in tcn} == {"R0", "Q4"}
    assert tcn[0]["tau"] == tcn[1]["tau"]
    assert tcn[0]["tau"] is tcn[1]["tau"]


def test_tcn_trained_once_per_dataset_seed():
    """TCN.fit runs once per (dataset, seed) even with two codecs."""
    from harness import run_harness

    calls: list = []
    orig_fit = TCNAutoencoder.fit

    def counting_fit(self, x_train: np.ndarray) -> None:
        """Count fits then delegate to the real fit."""
        calls.append(1)
        return orig_fit(self, x_train)

    TCNAutoencoder.fit = counting_fit  # type: ignore[method-assign]
    try:
        run_harness.run_cell("Spike", 42, ["R0", "Q4"])
    finally:
        TCNAutoencoder.fit = orig_fit  # type: ignore[method-assign]
    assert len(calls) == 1


def test_per_codec_retune_would_differ():
    """Retuning tau on Q4-train changes tau: per-codec retune is detectable."""
    s = make_drift(42)
    x = _f32(s.x)
    xtr = x[:TRAIN_END]
    det = TCNAutoencoder(seed=42)
    det.fit(xtr)
    tau_r0 = calibrate_frozen_threshold(det.score(xtr))
    q4tr = np.asarray(Q4Codec().decode(Q4Codec().encode(xtr))).ravel()[:TRAIN_END]
    tau_q4 = calibrate_frozen_threshold(det.score(q4tr))
    assert tau_r0 != tau_q4  # retune attempt must be visible, never silent


def test_q4_inflates_normal_residuals_on_drift():
    """Q4 staircase inflates nominal TCN residuals vs R0 (H2 mechanism)."""
    s = make_drift(42)
    x = _f32(s.x)
    y = np.asarray(s.y, dtype=int).ravel()
    xtr, xte, yte = x[:TRAIN_END], x[TRAIN_END:], y[TRAIN_END:]
    det = TCNAutoencoder(seed=42)
    det.fit(xtr)
    r0 = det.score(xte)
    dec = np.asarray(Q4Codec().decode(Q4Codec().encode(xte))).ravel()[: xte.size]
    q4 = det.score(dec)
    nom = yte == 0
    assert bool(np.mean(q4[nom]) > np.mean(r0[nom]))


def test_frozen_subsample_formula_exactness():
    """step/offset/every-step-th windows; exact len; <= cap; seed-stable."""
    assert MAX_TRAIN_WINDOWS == 4000
    cases = (
        (737, 42),  # synth train: no subsample
        (4000, 42),  # at cap: identity
        (4001, 42),  # just over cap
        (31849, 42),  # UCR-scale train
        (31849, 43),  # seed offset shifts
        (8000, 46),
    )
    for n_win, seed in cases:
        idx = frozen_subsample_indices(n_win, seed)
        step = math.ceil(n_win / MAX_TRAIN_WINDOWS)
        offset = seed % step
        assert step >= 1
        assert np.array_equal(idx, np.arange(offset, n_win, step))
        assert len(idx) == math.ceil((n_win - offset) / step)
        assert len(idx) <= MAX_TRAIN_WINDOWS
        if n_win <= MAX_TRAIN_WINDOWS:
            assert len(idx) == n_win
    # Seed determinism: same seed identical; offset-shifting seed differs.
    a = frozen_subsample_indices(31849, 42)
    b = frozen_subsample_indices(31849, 42)
    assert np.array_equal(a, b)
    c = frozen_subsample_indices(31849, 43)  # offset 2 -> 3
    assert not np.array_equal(a, c)


def test_tcn_never_fit_on_compressed_data():
    """Runner TCN fit sees only the raw R0 train slice (spy the input)."""
    from harness import run_harness

    seen: list[np.ndarray] = []
    orig_fit = TCNAutoencoder.fit

    def spying_fit(self, x_train: np.ndarray) -> None:
        """Record the fit input then delegate."""
        seen.append(np.asarray(x_train).ravel().copy())
        return orig_fit(self, x_train)

    TCNAutoencoder.fit = spying_fit  # type: ignore[method-assign]
    try:
        s = make_spike(42)
        x_raw = _f32(s.x)
        run_harness.run_cell("Spike", 42, ["R0", "Q4"])
    finally:
        TCNAutoencoder.fit = orig_fit  # type: ignore[method-assign]
    assert len(seen) == 1
    assert np.array_equal(seen[0], x_raw[:TRAIN_END])
