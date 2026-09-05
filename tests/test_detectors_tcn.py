"""TCN autoencoder pins: params < 12k, < 15s train, determinism, NaN loud.

TDD-red file for Todo 7: written BEFORE harness/detectors/tcn_autoencoder.py.
"""

import time

import numpy as np
import pytest

from harness.datasets.generator import TRAIN_END, make_spike
from harness.detectors.tcn_autoencoder import TCN_WINDOW, TCNAutoencoder

SEED = 0


@pytest.fixture()
def spike_split():
    s = make_spike(SEED)
    return s.x[:TRAIN_END], s.x[TRAIN_END:], s.y[TRAIN_END:]


def test_param_count_below_12k():
    d = TCNAutoencoder(seed=SEED)
    assert d.param_count() < 12_000


def test_default_epochs_capped_at_10():
    assert TCNAutoencoder(seed=SEED).epochs <= 10
    with pytest.raises(ValueError):
        TCNAutoencoder(seed=SEED, epochs=11)


def test_train_under_15s_on_synth_1200(spike_split):
    tr, _, _ = spike_split
    d = TCNAutoencoder(seed=SEED)
    t0 = time.perf_counter()
    d.fit(tr)
    dt = time.perf_counter() - t0
    assert dt < 15.0


def test_score_length_matches_input(spike_split):
    tr, te, _ = spike_split
    d = TCNAutoencoder(seed=SEED)
    d.fit(tr)
    assert d.score(te).shape == te.shape


def test_exact_reproducibility_same_seed(spike_split):
    tr, te, _ = spike_split
    d1, d2 = TCNAutoencoder(seed=SEED), TCNAutoencoder(seed=SEED)
    d1.fit(tr)
    d2.fit(tr)
    assert np.array_equal(d1.score(te), d2.score(te))


def test_different_seeds_differ(spike_split):
    tr, te, _ = spike_split
    d1, d2 = TCNAutoencoder(seed=SEED), TCNAutoencoder(seed=SEED + 1)
    d1.fit(tr)
    d2.fit(tr)
    assert not np.array_equal(d1.score(te), d2.score(te))


def test_scores_finite_including_all_zero(spike_split):
    tr, te, _ = spike_split
    d = TCNAutoencoder(seed=SEED)
    d.fit(tr)
    assert bool(np.all(np.isfinite(d.score(te))))
    z = TCNAutoencoder(seed=SEED)
    z.fit(np.zeros(800))
    got = z.score(np.zeros(1200))
    assert got.shape == (1200,)
    assert bool(np.all(np.isfinite(got)))
    short = z.score(np.zeros(10))
    assert short.shape == (10,)
    assert bool(np.all(np.isfinite(short)))


def test_nan_input_raises_value_error(spike_split):
    tr, te, _ = spike_split
    bad_tr = tr.copy()
    bad_tr[0] = np.nan
    with pytest.raises(ValueError):
        TCNAutoencoder(seed=SEED).fit(bad_tr)
    d = TCNAutoencoder(seed=SEED)
    d.fit(tr)
    bad_te = te.copy()
    bad_te[0] = np.nan
    with pytest.raises(ValueError):
        d.score(bad_te)


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        TCNAutoencoder(seed=SEED).score(np.zeros(100))


def test_cpu_only_no_cuda():
    d = TCNAutoencoder(seed=SEED)
    assert d.device == "cpu"
    import torch

    assert torch.cuda.is_available() is False or d.device == "cpu"


def test_window_constant_is_64():
    assert TCN_WINDOW == 64
