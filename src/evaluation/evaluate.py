"""Evaluation helpers producing metric dicts and optional plots."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_index: int = 0,
    label: str = "",
) -> dict[str, float]:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim == 2:
        yt = yt[:, horizon_index]
    if yp.ndim == 2:
        yp = yp[:, horizon_index]
    metrics = compute_all_metrics(yt, yp)
    metrics["label"] = label  # type: ignore[assignment]
    return metrics


def save_metrics_row(
    path: str | Path,
    row: dict[str, Any],
    append: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if append and path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(path, index=False)


def plot_rul_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str = "RUL: Predicted vs True",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, alpha=0.6, s=20, edgecolors="none")
    lim = [0, max(yt.max(), yp.max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1, label="ideal")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attention_heatmap(
    weights: np.ndarray,
    out_path: str | Path,
    title: str = "Temporal Attention Weights",
) -> None:
    """weights: (N, T) or (T,)"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = np.asarray(weights)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    fig, ax = plt.subplots(figsize=(8, max(2, w.shape[0] * 0.15 + 1)))
    im = ax.imshow(w, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Time step in window")
    ax.set_ylabel("Sample")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
