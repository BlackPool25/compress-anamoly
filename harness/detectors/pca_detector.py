"""Linear PCA reconstruction-error detector (Class 0, w=32 stride 1)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from harness.detectors.base import WINDOW, BaseDetector, sliding_windows
from harness.metrics.alignment import window_to_point_trailing


class PCADetector(BaseDetector):
    """PCA(2) on nominal windows; score = per-window reconstruction SSE.

    Window scores map to points via trailing alignment; the first
    ``w - 1`` points use the median of the TRAIN window scores, cached
    at ``fit`` time (so ``score`` is side-effect free).
    """

    def __init__(self, seed: int) -> None:
        """Seed the PCA(2) model and mark it unfitted."""
        self.seed = seed
        self._pca = PCA(n_components=2, random_state=seed)
        self._train_median: float = 0.0
        self._fitted = False

    def fit(self, x_train: np.ndarray) -> None:
        """Fit PCA(2) on nominal windows; cache the train SSE median fill."""
        w = sliding_windows(x_train, WINDOW)
        self._pca.fit(w)
        recon = self._pca.inverse_transform(self._pca.transform(w))
        self._train_median = float(np.median(np.sum((w - recon) ** 2, axis=1)))
        self._fitted = True

    def score(self, x_test: np.ndarray) -> np.ndarray:
        """SSE scores mapped to points via trailing alignment (train-median fill)."""
        if not self._fitted:
            raise RuntimeError("PCADetector.score called before fit")
        n = int(np.asarray(x_test).size)
        w = sliding_windows(x_test, WINDOW)
        if w.shape[0] == 0:
            return np.full(n, self._train_median, dtype=float)
        recon = self._pca.inverse_transform(self._pca.transform(w))
        ws = np.sum((w - recon) ** 2, axis=1)
        return window_to_point_trailing(ws, n, WINDOW, self._train_median)
