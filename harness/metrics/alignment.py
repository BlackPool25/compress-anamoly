"""Trailing window-to-point score alignment.

PLAN-FREEZE home for the window-to-point helper: detectors (todo 7) import
:func:`window_to_point_trailing` from this module, so the import path
``harness.metrics.alignment`` is frozen. The spec defines no other alignment.
"""

import numpy as np


def window_to_point_trailing(
    window_scores: np.ndarray,
    n_points: int,
    window: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Map window scores to point scores with TRAILING assignment.

    Window ``i`` covers points ``[i, i + window)`` (stride 1); its score is
    assigned to its last index ``i + window - 1``. The first ``window - 1``
    points have no trailing window and are filled with ``fill_value``.

    Per the frozen-evaluator spec, callers pass the median of that
    detector's R0 train scores (same seed/dataset) as ``fill_value``.

    Args:
        window_scores: 1-D array of ``n_points - window + 1`` window scores.
        n_points: Length of the point-level series to reconstruct.
        window: Window length (detector ``w``).
        fill_value: Fill for the first ``window - 1`` points.

    Returns:
        1-D point-score array of length ``n_points``.

    Raises:
        ValueError: If sizes are inconsistent or ``window < 1``.
    """
    ws = np.asarray(window_scores, dtype=float).ravel()
    n_points = int(n_points)
    window = int(window)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if n_points < window:
        raise ValueError(f"n_points={n_points} < window={window}")
    if ws.shape[0] != n_points - window + 1:
        raise ValueError(
            f"expected {n_points - window + 1} window scores, got {ws.shape[0]}"
        )
    points = np.full(n_points, float(fill_value), dtype=float)
    points[window - 1 :] = ws
    return points
