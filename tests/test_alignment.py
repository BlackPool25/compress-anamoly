"""Trailing window-to-point alignment pins (PLAN-FREEZE)."""

import numpy as np
import pytest

from harness.metrics.alignment import window_to_point_trailing


def test_trailing_mapping_exact():
    """Each window score lands on its LAST index; head filled."""
    ws = np.array([10.0, 20.0, 30.0, 40.0])
    got = window_to_point_trailing(ws, n_points=6, window=3, fill_value=-1.0)
    np.testing.assert_array_equal(got, [-1.0, -1.0, 10.0, 20.0, 30.0, 40.0])


def test_leading_mapping_does_not_match():
    """A leading assignment ([w0..w3, fill, fill]) must NOT match."""
    ws = np.array([10.0, 20.0, 30.0, 40.0])
    got = window_to_point_trailing(ws, n_points=6, window=3, fill_value=-1.0)
    leading = np.array([10.0, 20.0, 30.0, 40.0, -1.0, -1.0])
    assert not np.array_equal(got, leading)


def test_window_one_is_identity():
    """w=1 maps every window score onto its own point."""
    ws = np.array([1.0, 2.0, 3.0])
    got = window_to_point_trailing(ws, n_points=3, window=1, fill_value=-9.0)
    np.testing.assert_array_equal(got, ws)


def test_length_mismatch_raises():
    """Wrong window-score count raises instead of silently shifting."""
    with pytest.raises(ValueError):
        window_to_point_trailing(np.array([1.0, 2.0]), n_points=6, window=3)
