"""Evaluation metrics for RUL prediction, including the C-MAPSS asymmetric score."""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA C-MAPSS asymmetric scoring function.

    For each residual d = y_pred - y_true:
      - late prediction (d < 0):  exp(-d / 13) - 1
      - early prediction (d >= 0): exp(d / 10) - 1

    Total score is the sum over all samples. Lower is better.
    Late predictions (underestimating RUL / predicting failure too late)
    are penalized more gently than early ones in the classic formulation;
    the asymmetric constants (13 vs 10) reflect the PHM08 / C-MAPSS convention.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return RMSE, MAE, and C-MAPSS score."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "score": cmapss_score(y_true, y_pred),
    }
