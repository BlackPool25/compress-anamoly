"""Detector D4: direct-on-symbols Markov (R4 token streams only).

Fits a 1st-order Markov transition matrix P(token_t | token_{t-1}) on the
nominal R4 int-token stream from ``SAXCodec.decode_tokens`` and scores each
test token as ``-log(P + 1e-5)``. Token scores are resampled to sample length
via ``np.repeat(..., PAA_WINDOW)`` (PAA=8 inverse, frozen).

NOTE on spec label mismatch (PLAN-FREEZE): TASK_01 L176 calls this a
"2nd-order" matrix, but its own formula conditions on exactly ONE prior token
``P(token_t | token_{t-1})``. The formula as written IS a 1st-order chain, so
this module implements 1st-order (8x8 matrix) and treats the "2nd-order" word
as a label error, not a requirement for an 8x8x8 tensor.

R4-ONLY: fit/score accept int token arrays (0-7) and REJECT float arrays with
ValueError. The runner enforces R4-only routing; the dtype gate here is the
proof. NEVER decode tokens to floats in this module. No seed, no stochastic
op — fully deterministic.
"""

import numpy as np

from harness.codecs.symbolic import ALPHABET_SIZE, PAA_WINDOW

EPS = 1e-5


def _require_int_tokens(tokens: np.ndarray) -> np.ndarray:
    """Validate R4 int-token contract; raise ValueError on float/non-int input."""
    arr = np.asarray(tokens)
    if arr.dtype.kind not in ("i", "u"):
        raise ValueError(
            f"D4 accepts R4 int token streams only (dtype kind 'i'/'u'), "
            f"got dtype {arr.dtype}. Raw float series belong to Classes 0/1, not D4."
        )
    if arr.size and (arr.min() < 0 or arr.max() >= ALPHABET_SIZE):
        raise ValueError(f"D4 tokens must be in [0, {ALPHABET_SIZE - 1}].")
    return arr.astype(np.int64)


class DirectSymbolicDetector:
    """Zero-decompression Markov anomaly detector on R4 token streams."""

    def __init__(self) -> None:
        """Create an unfitted detector (NO seed: fully deterministic)."""
        self.transitions: np.ndarray | None = None

    def fit(self, token_train: np.ndarray) -> None:
        """Count t-1 -> t transitions over the 8x8 matrix, +1e-5 smoothing, row-normalize."""
        toks = _require_int_tokens(token_train)
        counts = np.full((ALPHABET_SIZE, ALPHABET_SIZE), EPS, dtype=np.float64)
        if toks.size >= 2:
            np.add.at(counts, (toks[:-1], toks[1:]), 1.0)
        # Rows with no observed outgoing transition stay uniform via smoothing.
        self.transitions = counts / counts.sum(axis=1, keepdims=True)

    def score(self, token_test: np.ndarray) -> np.ndarray:
        """Per-token -log(P+1e-5), resampled to sample length via xPAA_WINDOW repeat."""
        if self.transitions is None:
            raise RuntimeError("DirectSymbolicDetector.score called before fit.")
        toks = _require_int_tokens(token_test)
        if toks.size == 0:
            return np.empty(0, dtype=np.float64)
        # First token has no prior: score under the uniform prior 1/8.
        # (Stationary prior would couple train dynamics into position 0; uniform
        # keeps position 0 deterministic and independent of fit — pinned by test.)
        token_scores = np.empty(toks.size, dtype=np.float64)
        token_scores[0] = -np.log(1.0 / ALPHABET_SIZE + EPS)
        if toks.size >= 2:
            probs = self.transitions[toks[:-1], toks[1:]]
            token_scores[1:] = -np.log(probs + EPS)
        return np.repeat(token_scores, PAA_WINDOW)
