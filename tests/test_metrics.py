"""Frozen-evaluator + VUS-PR conformance (plan lines 81-82)."""

import pathlib

import numpy as np
import pytest

from harness.metrics.alignment import window_to_point_trailing
from harness.metrics.frozen_evaluator import (
    calibrate_frozen_threshold,
    evaluate_event_f1,
    evaluate_point_f1,
    evaluate_pr_auc,
)
from harness.metrics.vus import VUS_WINDOW, vus_pr


def test_tau_is_99th_percentile():
    """Default calibration equals the 99th percentile of R0 train scores."""
    rng = np.random.default_rng(11)
    train = rng.normal(0, 1, 2000)
    assert calibrate_frozen_threshold(train) == float(np.percentile(train, 99.0))


def test_tau_key_frozen_reused_across_codecs():
    """One R0-tau object serves every codec; never recomputed on test data."""
    rng = np.random.default_rng(0)
    r0_train = rng.normal(0, 1, 5000)
    tau = calibrate_frozen_threshold(r0_train)  # single calibration
    y = np.zeros(200, dtype=int)
    y[50:60] = 1
    base_a = rng.uniform(0, 1, 200)
    base_b = rng.uniform(0, 1, 200)
    s_a, s_b = base_a.copy(), base_b.copy()
    s_a[50:60] = 3.0  # codec A scores
    s_b[50:60] = 3.0  # codec B scores
    assert evaluate_point_f1(y, s_a, tau) == 1.0
    assert evaluate_point_f1(y, s_b, tau) == 1.0
    assert tau == float(np.percentile(r0_train, 99.0))
    assert tau != float(np.percentile(s_a, 99.0))  # frozen, not recomputed


def test_point_f1_hand_computed():
    """TP=1, FP=0, FN=1 -> P=1, R=0.5, F1=2/3."""
    y = np.array([0, 1, 1, 0])
    s = np.array([0.1, 0.9, 0.2, 0.1])
    assert evaluate_point_f1(y, s, 0.5) == pytest.approx(2 / 3)


def _event_fixture():
    y = np.zeros(7, dtype=int)
    y[2:5] = 1
    return y


def test_event_hit_no_false_alarm():
    """One labeled region hit, no outside segment -> F1 == 1.0."""
    s = np.array([0.0, 0.1, 0.9, 0.8, 0.9, 0.1, 0.1])
    assert evaluate_event_f1(_event_fixture(), s, 0.5) == 1.0


def test_event_hit_plus_false_alarm():
    """Extra detected segment outside labels -> P=1/2, R=1, F1=2/3."""
    s = np.array([0.0, 0.1, 0.9, 0.8, 0.9, 0.1, 0.9])
    assert evaluate_event_f1(_event_fixture(), s, 0.5) == pytest.approx(2 / 3)


def test_event_miss():
    """No point >= tau inside the region -> recall 0 -> F1 == 0.0."""
    s = np.zeros(7)
    assert evaluate_event_f1(_event_fixture(), s, 0.5) == 0.0


def test_event_uses_trailing_alignment_rule():
    """Window scores flow through window_to_point_trailing (median fill)."""
    train = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    tau = calibrate_frozen_threshold(train)  # 99th pct ~= 0.496
    assert 0.4 < tau < 0.9
    y = np.zeros(7, dtype=int)
    y[4:6] = 1
    ws = np.array([0.1, 0.2, 0.95, 0.95, 0.1])  # 5 windows, w=3
    points = window_to_point_trailing(
        ws, n_points=7, window=3, fill_value=float(np.median(train))
    )
    assert evaluate_event_f1(y, points, tau) == 1.0


def test_event_frozen_tau_default_pins_freeze():
    """Default-calibrated tau (99th pct) freezes the event outcome.

    Freeze-sensitivity pin: the 99th pct of this train slice sits above the
    fixture scores (miss -> F1 0.0), while the 95th pct falls below them
    (hit -> F1 1.0), so mutating the default to 95.0 must FAIL this test.
    """
    train = np.array([0.1] * 19 + [0.9])
    tau = calibrate_frozen_threshold(train)  # 99th pct ~= 0.748
    y = _event_fixture()
    s = np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0])
    assert tau > 0.5
    assert evaluate_event_f1(y, s, tau) == 0.0


def test_pr_auc_sanity():
    """Perfect ranking -> 1.0; imperfect ranking strictly below."""
    y = np.array([0, 0, 1, 1])
    assert evaluate_pr_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    imperfect = evaluate_pr_auc(y, [0.1, 0.9, 0.2, 0.8])
    assert 0.0 < imperfect < 1.0


def _vus_two_region_fixture():
    n = 200
    y = np.zeros(n)
    y[80:120] = 1
    return y


def test_vus_window_is_64():
    """VUS window decoupled from detector w=32 (PLAN-FREEZE)."""
    assert VUS_WINDOW == 64


def test_vus_perfect_scores_is_one():
    """Perfect scores -> VUS-PR 1.0 ±1e-6."""
    y = _vus_two_region_fixture()
    assert vus_pr(y, y.astype(float)) == pytest.approx(1.0, abs=1e-6)


def test_vus_all_zero_strictly_lower():
    """All-zero scores score strictly lower than perfect scores."""
    y = _vus_two_region_fixture()
    assert vus_pr(y, np.zeros_like(y, dtype=float)) < vus_pr(
        y, y.astype(float)
    )


def test_vus_mismatched_lengths_raise():
    """Mismatched label/score lengths raise instead of truncating."""
    with pytest.raises(ValueError):
        vus_pr(np.zeros(10), np.zeros(9))


def test_vus_pip_vs_vendored_agreement():
    """Pip-vs-vendored agreement ±1e-6 when both importable."""
    vendored = pytest.importorskip(
        "harness.metrics._vendored_vus",
        reason="pip vus available; no vendored copy needed",
    )
    y = _vus_two_region_fixture()
    s = y.astype(float)
    assert abs(vus_pr(y, s) - float(vendored.vus_pr(y, s))) <= 1e-6


def test_vus_window_sensitivity_informational():
    """64-vs-32 differ < 0.10 on spike fixture; robustness note, never a gate."""
    n = 300
    y = np.zeros(n)
    y[100:105] = 1
    s = np.zeros(n)
    s[100:105] = 1.0
    s[102] = 0.4
    s[200] = 0.9
    from vus.metrics import generate_curve

    *_, v32 = generate_curve(y, s, 32)
    assert abs(vus_pr(y, s) - float(v32)) < 0.10


def test_no_point_adjust_symbols():
    """PA-F1 ban (METIS-F18): no point_adjust/point_adjustment anywhere."""
    root = pathlib.Path(__file__).resolve().parents[1] / "harness" / "metrics"
    hits = [
        p.name
        for p in root.glob("*.py")
        if "point_adjust" in p.read_text() or "point-adjust" in p.read_text()
    ]
    assert hits == []
