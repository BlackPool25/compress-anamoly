"""Frozen-threshold evaluator.

TAU KEY FROZEN: ``tau`` is calibrated per (seed, dataset, detector) on
R0-train-scores only via :func:`calibrate_frozen_threshold` and the same
``tau`` object is reused across codecs — it is never recomputed on test
data or per codec. Window scores reach the event metric only after mapping
through ``harness.metrics.alignment.window_to_point_trailing`` (TRAILING
assignment, first ``w - 1`` points filled with the median of that
detector's R0 train scores for that seed/dataset).
"""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def calibrate_frozen_threshold(
    train_scores: np.ndarray, percentile: float = 99.0
) -> float:
    """Calibrate the frozen threshold ``tau`` on R0 train scores.

    Args:
        train_scores: 1-D R0 train scores for one (seed, dataset, detector).
        percentile: Percentile of the train scores used as ``tau``.

    Returns:
        ``tau`` as a float; reuse it across codecs, never recompute it.

    Raises:
        ValueError: If ``train_scores`` is empty.
    """
    scores = np.asarray(train_scores, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("train_scores must be non-empty")
    return float(np.percentile(scores, float(percentile)))


def evaluate_point_f1(
    y_true: np.ndarray, y_score: np.ndarray, tau: float
) -> float:
    """Point-F1 of ``y_score >= tau`` against binary ``y_true``.

    Args:
        y_true: 1-D binary labels.
        y_score: 1-D point-level scores aligned with ``y_true``.
        tau: Frozen threshold from :func:`calibrate_frozen_threshold`.

    Returns:
        Binary F1 in ``[0, 1]`` (``0.0`` when no predicted positive).
    """
    yt = np.asarray(y_true).ravel()
    pred = (np.asarray(y_score, dtype=float).ravel() >= float(tau)).astype(int)
    return float(f1_score(yt, pred, zero_division=0))


def _contiguous_runs(mask: np.ndarray) -> list:
    """Return ``[(start, end)]`` inclusive runs where ``mask`` is True."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


def evaluate_event_f1(
    y_true: np.ndarray, y_score: np.ndarray, tau: float
) -> float:
    """Event-F1 over labeled regions vs detected segments.

    A labeled region (contiguous ``y_true == 1`` run) is a hit if at least
    one point inside it scores ``>= tau``. A detected segment (contiguous
    ``y_score >= tau`` run) with zero overlap with any labeled point is a
    false alarm. Recall = hits / true events; precision = non-false-alarm
    detected segments / detected segments; F1 is their harmonic mean.

    Args:
        y_true: 1-D binary labels.
        y_score: 1-D point-level scores (map window scores via
            ``window_to_point_trailing`` first).
        tau: Frozen threshold from :func:`calibrate_frozen_threshold`.

    Returns:
        Event F1 in ``[0, 1]``.
    """
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    true_events = _contiguous_runs(yt == 1)
    detected = _contiguous_runs(ys >= float(tau))
    if not true_events:
        return 1.0 if not detected else 0.0
    overlap = lambda a, b: a[0] <= b[1] and b[0] <= a[1]
    hits = sum(any(overlap(t, d) for d in detected) for t in true_events)
    false_alarms = sum(
        not any(overlap(d, t) for t in true_events) for d in detected
    )
    precision = (len(detected) - false_alarms) / len(detected) if detected else 0.0
    recall = hits / len(true_events)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def evaluate_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """PR-AUC (average precision) via sklearn.

    Args:
        y_true: 1-D binary labels.
        y_score: 1-D continuous scores aligned with ``y_true``.

    Returns:
        Average precision as a float.
    """
    return float(
        average_precision_score(
            np.asarray(y_true).ravel(),
            np.asarray(y_score, dtype=float).ravel(),
        )
    )
