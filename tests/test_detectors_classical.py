"""Classical detector pins: PCA + IsolationForest, w=32 trailing alignment."""

import numpy as np
import pytest

from harness.datasets.generator import TRAIN_END, make_spike
from harness.detectors.base import WINDOW
from harness.detectors.isolation_forest import IsolationForestDetector
from harness.detectors.pca_detector import PCADetector

SEED = 0
DETECTORS = [PCADetector, IsolationForestDetector]


@pytest.fixture()
def spike_split():
    s = make_spike(SEED)
    return s.x[:TRAIN_END], s.x[TRAIN_END:], s.y[TRAIN_END:]


@pytest.mark.parametrize("cls", DETECTORS)
def test_score_length_matches_input(cls, spike_split):
    tr, te, _ = spike_split
    d = cls(seed=SEED)
    d.fit(tr)
    assert d.score(te).shape == te.shape


@pytest.mark.parametrize("cls", DETECTORS)
def test_spike_peak_at_anomaly(cls, spike_split):
    """Peak window fully contains the 20-pt spike, so its TRAILING point
    lands up to ``w - 1`` points past the anomaly end (frozen alignment
    physics: a strict ``argmax in labels`` is unachievable — the
    max-SSE window trailing edge sits at ``[1420, 1431]`` while labels
    end at 1420 exclusive). Pin peak inside anomaly dilated by the lag,
    plus mean-inside > mean-outside as the detection proof.
    """
    tr, te, lab = spike_split
    d = cls(seed=SEED)
    d.fit(tr)
    sc = d.score(te)
    idx = np.where(lab == 1)[0]
    peak = int(np.argmax(sc))
    assert idx[0] <= peak <= idx[-1] + (WINDOW - 1)
    assert float(sc[lab == 1].mean()) > float(sc[lab == 0].mean())


@pytest.mark.parametrize("cls", DETECTORS)
def test_exact_reproducibility_same_seed(cls, spike_split):
    tr, te, _ = spike_split
    d1, d2 = cls(seed=SEED), cls(seed=SEED)
    d1.fit(tr)
    d2.fit(tr)
    assert np.array_equal(d1.score(te), d2.score(te))


@pytest.mark.parametrize("cls", DETECTORS)
def test_scores_finite_including_all_zero(cls, spike_split):
    tr, te, _ = spike_split
    d = cls(seed=SEED)
    d.fit(tr)
    assert bool(np.all(np.isfinite(d.score(te))))
    z = cls(seed=SEED)
    z.fit(np.zeros(800))
    got = z.score(np.zeros(1200))
    assert got.shape == (1200,)
    assert bool(np.all(np.isfinite(got)))
    short = z.score(np.zeros(10))
    assert short.shape == (10,)
    assert bool(np.all(np.isfinite(short)))


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        PCADetector(seed=SEED).score(np.zeros(100))
    with pytest.raises(RuntimeError):
        IsolationForestDetector(seed=SEED).score(np.zeros(100))
