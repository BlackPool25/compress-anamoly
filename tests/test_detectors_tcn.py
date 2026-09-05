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


def _long_seeded_series(seed: int, n: int = 60000) -> np.ndarray:
    """Deterministic smooth+noise series; n=60000 -> 59937 windows (>50k)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return (np.sin(t / 50.0) + 0.5 * np.sin(t / 13.0)
            + 0.1 * rng.standard_normal(n)).astype(float)


def _full_forward_reference(d: TCNAutoencoder, arr: np.ndarray) -> np.ndarray:
    """Pre-batch reference: ONE forward over the full window tensor.

    Batch-1 loops take a different oneDNN kernel path (diff ~3e-05), so the
    bit-exact reference for chunked scoring is the single full forward.
    """
    import torch

    from harness.detectors.base import sliding_windows
    from harness.detectors.tcn_autoencoder import _pad_to_mult4

    w = sliding_windows(arr, TCN_WINDOW)
    t = torch.from_numpy(
        (w - d._mean) / d._std).float()[:, None, :].to(d.device)
    d._net.eval()
    with torch.no_grad():
        recon, pad = _pad_to_mult4(d._net(_pad_to_mult4(t)[0]))
        if pad:
            recon = recon[..., : t.shape[-1]]
        return torch.sum((t - recon) ** 2, dim=(1, 2)).cpu().numpy()


def test_score_batch_chunk_size_constant():
    """Batched score chunks windows into blocks of <= 1024 (Todo 10)."""
    from harness.detectors.tcn_autoencoder import SCORE_BATCH_WINDOWS

    assert SCORE_BATCH_WINDOWS == 1024


@pytest.mark.parametrize("seed", [42, 43])
def test_score_batched_bit_identical_to_serial(seed: int):
    """Batched score == single full forward, bit-identical (diff == 0.0)."""
    from harness.metrics.alignment import window_to_point_trailing

    arr = _long_seeded_series(seed)
    d = TCNAutoencoder(seed=seed)
    d.fit(arr[:800])
    got = d.score(arr)
    ref_ws = _full_forward_reference(d, arr)
    ref = window_to_point_trailing(ref_ws, arr.size, TCN_WINDOW,
                                   d._train_median)
    assert got.shape == ref.shape
    assert float(np.max(np.abs(got - ref))) == 0.0
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("seed", [42, 43])
def test_score_issues_chunked_forwards(seed: int):
    """Long inputs run one forward per <=1024-window chunk (Todo 10)."""
    import math

    arr = _long_seeded_series(seed)
    d = TCNAutoencoder(seed=seed)
    d.fit(arr[:800])
    calls: list[int] = []
    orig_forward = d._net.forward

    def counting(x):
        calls.append(int(x.shape[0]))
        return orig_forward(x)

    d._net.forward = counting  # type: ignore[method-assign]
    d.score(arr)
    n_win = arr.size - TCN_WINDOW + 1
    assert calls, "score issued no forwards"
    assert all(c <= 1024 for c in calls)
    assert len(calls) == math.ceil(n_win / 1024)
