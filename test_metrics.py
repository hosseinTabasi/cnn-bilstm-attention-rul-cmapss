"""Unit tests for C-MAPSS asymmetric score and related metrics."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import cmapss_score, mae, rmse


def test_rmse_mae_basic():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([10.0, 20.0, 30.0])
    assert rmse(y, p) == pytest.approx(0.0)
    assert mae(y, p) == pytest.approx(0.0)


def test_cmapss_score_perfect():
    y = np.array([50.0, 60.0, 70.0])
    assert cmapss_score(y, y) == pytest.approx(0.0)


def test_cmapss_score_asymmetric():
    """Early prediction (over-estimate RUL) uses /10; late uses /13."""
    y_true = np.array([100.0])
    # early: pred > true → d = +10 → exp(10/10)-1 = e-1
    early = cmapss_score(y_true, np.array([110.0]))
    # late: pred < true → d = -10 → exp(10/13)-1
    late = cmapss_score(y_true, np.array([90.0]))
    assert early == pytest.approx(np.exp(1.0) - 1.0)
    assert late == pytest.approx(np.exp(10.0 / 13.0) - 1.0)
    # With |d|=10, early penalty > late penalty
    assert early > late


def test_cmapss_score_sum():
    y = np.array([100.0, 100.0])
    p = np.array([110.0, 90.0])
    expected = (np.exp(1.0) - 1.0) + (np.exp(10.0 / 13.0) - 1.0)
    assert cmapss_score(y, p) == pytest.approx(expected)
