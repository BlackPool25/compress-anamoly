"""VUS-PR wrapper (VUS-PR FROZEN).

Primary implementation: PyPI ``vus==0.0.6`` author reference (TheDatumOrg,
Apache-2.0). Context7 has no ``vus`` package docs, so the wrapper was
matched against the installed ``vus==0.0.6`` API by introspection:
``generate_curve(label, score, slidingWindow, version='opt', thre=250)``
returns ``(Y, Z, X, X_ap, W, Z_ap, avg_auc_3d, avg_ap_3d)`` where the last
element ``avg_ap_3d`` is VUS-PR (confirmed via ``vus.metrics.get_metrics``
source, which unpacks the same tuple positionally). The window is
passed positionally so the wrapper is exact regardless of the parameter
name (``slidingWindow`` in the installed package).
"""

import numpy as np

# VUS window, DECOUPLED from detector w=32. Scale rationale: anomaly lengths
# span 20-500 pts, so 64 sits at mid-scale; PLAN-FREEZE, do not retune.
VUS_WINDOW = 64


def vus_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Volume Under the Surface of the PR curve (VUS-PR).

    Args:
        y_true: 1-D binary labels.
        y_score: 1-D continuous scores aligned with ``y_true``.

    Returns:
        VUS-PR as a float (1.0 for perfect scores).

    Raises:
        ValueError: If ``y_true`` and ``y_score`` lengths differ.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if yt.shape[0] != ys.shape[0]:
        raise ValueError(
            f"length mismatch: y_true={yt.shape[0]} vs y_score={ys.shape[0]}"
        )
    from vus.metrics import generate_curve

    _, _, _, _, _, _, _, vus_pr_value = generate_curve(yt, ys, VUS_WINDOW)
    return float(vus_pr_value)
