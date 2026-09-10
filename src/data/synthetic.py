"""Synthetic multi-engine degradation data mimicking C-MAPSS structure.

Used ONLY when real C-MAPSS cannot be downloaded. All experiment outputs
must be labeled as SYNTHETIC when this path is used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_NAMES = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)


def _simulate_engine(
    rng: np.random.Generator,
    unit_id: int,
    n_cycles: int,
    n_sensors: int = 21,
) -> np.ndarray:
    """Simulate one engine run-to-failure trajectory."""
    op = rng.normal(0, 0.3, size=(n_cycles, 3))
    t = np.linspace(0, 1, n_cycles)
    # Slow healthy phase then accelerating degradation
    health = 1.0 - 0.15 * t - 0.85 * (t ** 2.5)
    base = rng.normal(0.5, 0.1, size=n_sensors)
    trend = rng.uniform(-1.0, 1.0, size=n_sensors)
    noise = rng.normal(0, 0.05, size=(n_cycles, n_sensors))
    sensors = base + np.outer(1.0 - health, trend) + noise
    unit = np.full((n_cycles, 1), unit_id, dtype=np.float64)
    cycle = np.arange(1, n_cycles + 1, dtype=np.float64).reshape(-1, 1)
    return np.hstack([unit, cycle, op, sensors])


def generate_synthetic_subset(
    subset: str,
    out_dir: str | Path,
    n_train_engines: int = 40,
    n_test_engines: int = 20,
    seed: int = 0,
) -> Path:
    """Write train_*.txt, test_*.txt, RUL_*.txt for one subset.

    Returns the output directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + hash(subset) % 10_000)

    train_rows = []
    for i in range(1, n_train_engines + 1):
        n_cyc = int(rng.integers(128, 280))
        train_rows.append(_simulate_engine(rng, i, n_cyc))
    train = np.vstack(train_rows)
    pd.DataFrame(train).to_csv(
        out_dir / f"train_{subset}.txt", sep=" ", header=False, index=False
    )

    test_rows = []
    rul_list = []
    for i in range(1, n_test_engines + 1):
        full = int(rng.integers(140, 300))
        trunc = int(rng.integers(60, full - 20))
        remaining = full - trunc
        test_rows.append(_simulate_engine(rng, i, trunc))
        rul_list.append(remaining)
    test = np.vstack(test_rows)
    pd.DataFrame(test).to_csv(
        out_dir / f"test_{subset}.txt", sep=" ", header=False, index=False
    )
    pd.DataFrame(rul_list).to_csv(
        out_dir / f"RUL_{subset}.txt", sep=" ", header=False, index=False
    )

    # Marker so loaders know this is synthetic
    (out_dir / "SYNTHETIC_DATA.txt").write_text(
        "This directory contains SYNTHETIC C-MAPSS-like data.\n"
        "Do not treat results as comparable to published C-MAPSS benchmarks.\n",
        encoding="utf-8",
    )
    return out_dir


def ensure_synthetic_cmapss(
    data_dir: str | Path = "data/raw",
    subsets: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004"),
) -> Path:
    """Generate all four subsets if missing."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for i, sub in enumerate(subsets):
        if not (data_dir / f"train_{sub}.txt").exists():
            generate_synthetic_subset(sub, data_dir, seed=42 + i)
    return data_dir
