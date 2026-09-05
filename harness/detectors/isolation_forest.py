"""Windowed IsolationForest detector (Class 1, w=32 stride 1)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from harness.detectors.base import WINDOW, BaseDetector, sliding_windows
from harness.metrics.alignment import window_to_point_trailing


class IsolationForestDetector(BaseDetector):
    """IF(100, 0.01) on nominal windows; score = ``-score_samples``.

    Window scores map to points via trailing alignment; the first
    ``w - 1`` points use the median of the TRAIN window scores, cached
    at ``fit`` time (so ``score`` is side-effect free).
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._if = IsolationForest(
            n_estimators=100, contamination=0.01, random_state=seed
        )
        self._train_median: float = 0.0
        self._fitted = False

    def fit(self, x_train: np.ndarray) -> None:
        w = sliding_windows(x_train, WINDOW)
        self._if.fit(w)
        self._train_median = float(np.median(-self._if.score_samples(w)))
        self._fitted = True

    def score(self, x_test: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsolationForestDetector.score called before fit")
        n = int(np.asarray(x_test).size)
        w = sliding_windows(x_test, WINDOW)
        if w.shape[0] == 0:
            return np.full(n, self._train_median, dtype=float)
        ws = -self._if.score_samples(w)
        return window_to_point_trailing(ws, n, WINDOW, self._train_median)
