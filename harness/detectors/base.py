"""Abstract detector interface + shared stride-1 windowing helper."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

#: Frozen window length shared by all classical detectors (TASK_01 spec).
WINDOW: int = 32


def sliding_windows(x: np.ndarray, w: int = WINDOW) -> np.ndarray:
    """Stride-1 windows of length ``w`` over a 1-D series.

    Returns shape ``(n - w + 1, w)``; empty ``(0, w)`` if ``n < w``.
    """
    arr = np.asarray(x, dtype=float).ravel()
    n = arr.shape[0]
    if n < w:
        return np.empty((0, w), dtype=float)
    return np.lib.stride_tricks.sliding_window_view(arr, window_shape=w)


class BaseDetector(ABC):
    """Calibrate on nominal train data, then emit point-level scores."""

    @abstractmethod
    def fit(self, x_train: np.ndarray) -> None:
        """Calibrate the baseline model on nominal training data."""
        ...

    @abstractmethod
    def score(self, x_test: np.ndarray) -> np.ndarray:
        """Return point-level anomaly scores, ``len == len(x_test)``."""
        ...
