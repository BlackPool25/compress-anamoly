"""Acceptance tests for D4: direct-on-symbols Markov (R4 token streams only)."""

import numpy as np
import pytest

from harness.codecs.symbolic import ALPHABET_SIZE, PAA_WINDOW
from harness.detectors.direct_symbolic import EPS, DirectSymbolicDetector


def test_uniform_stream_near_constant_scores():
    """A stream matching train dynamics scores ~flat.

    Epsilon justification: scores live on [0, -log(1e-5)] ~= [0, 11.5]; a
    planted rare transition spikes by several nats (see next test), so
    std < 0.1 sits more than an order of magnitude below any real spike.
    Position 0 is excluded: it scores under the uniform prior (-log(1/8)
    ~= 2.08) by design, while on-dynamics tokens score ~= 0.
    """
    train = np.tile(np.arange(ALPHABET_SIZE), 40).astype(np.int64)
    d = DirectSymbolicDetector()
    d.fit(train)
    scores = d.score(train)
    assert scores.shape == (train.size * PAA_WINDOW,)
    assert float(np.std(scores[PAA_WINDOW:])) < 0.1


def test_planted_rare_transition_spikes_at_that_token():
    """A 0->7 transition unseen in all-zero train spikes exactly at its block."""
    d = DirectSymbolicDetector()
    d.fit(np.zeros(300, dtype=np.int64))
    test = np.zeros(300, dtype=np.int64)
    rare = 150
    test[rare] = ALPHABET_SIZE - 1
    scores = d.score(test)
    block = scores[rare * PAA_WINDOW : (rare + 1) * PAA_WINDOW]
    rest = np.concatenate([scores[: rare * PAA_WINDOW], scores[(rare + 1) * PAA_WINDOW :]])
    assert block.min() > rest.max() + 1.0


def test_resampled_length_alignment():
    """Resampled length == PAA_WINDOW * n_tokens (PAA=8 inverse, frozen)."""
    d = DirectSymbolicDetector()
    d.fit(np.arange(ALPHABET_SIZE, dtype=np.int64))
    n = 37
    scores = d.score(np.arange(n, dtype=np.int64) % ALPHABET_SIZE)
    assert scores.shape == (n * PAA_WINDOW,)


def test_first_token_uses_uniform_prior():
    """Position 0 scores -log(1/8 + EPS) regardless of train dynamics."""
    d = DirectSymbolicDetector()
    d.fit(np.zeros(200, dtype=np.int64))
    scores = d.score(np.full(50, 3, dtype=np.int64))
    assert scores[0] == pytest.approx(-np.log(1.0 / ALPHABET_SIZE + EPS))


def test_byte_identical_reruns():
    """No seed, no stochastic op: refit + rescore is bit-identical."""
    train = np.tile(np.arange(ALPHABET_SIZE), 25).astype(np.int64)
    test = np.arange(64, dtype=np.int64) % ALPHABET_SIZE
    d1, d2 = DirectSymbolicDetector(), DirectSymbolicDetector()
    d1.fit(train)
    d2.fit(train)
    assert np.array_equal(d1.score(test), d2.score(test))
    assert np.array_equal(d1.transitions, d2.transitions)


def test_float_input_rejected():
    """Raw float series are rejected by both fit and score (R4-only gate)."""
    d = DirectSymbolicDetector()
    with pytest.raises(ValueError):
        d.fit(np.zeros(64, dtype=np.float64))
    d.fit(np.zeros(64, dtype=np.int64))
    with pytest.raises(ValueError):
        d.score(np.linspace(0.0, 1.0, 64))
    with pytest.raises(ValueError):
        d.score([0.0, 1.0, 2.0])


def test_constructor_takes_no_seed():
    """Deterministic by construction: no seed parameter exists."""
    import inspect

    assert list(inspect.signature(DirectSymbolicDetector.__init__).parameters) == ["self"]
