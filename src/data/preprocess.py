"""C-MAPSS loading, piecewise RUL, scaling (train-only), windowing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors commonly retained for FD001/FD003 (constant sensors dropped)
USEFUL_SENSORS_FD001 = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]
USEFUL_SENSORS_FD002 = USEFUL_SENSORS_FD001  # keep same feature set for simplicity
USEFUL_SENSORS_FD003 = USEFUL_SENSORS_FD001
USEFUL_SENSORS_FD004 = USEFUL_SENSORS_FD001

SENSOR_MAP = {
    "FD001": USEFUL_SENSORS_FD001,
    "FD002": USEFUL_SENSORS_FD002,
    "FD003": USEFUL_SENSORS_FD003,
    "FD004": USEFUL_SENSORS_FD004,
}


def is_synthetic(data_dir: str | Path) -> bool:
    return (Path(data_dir) / "SYNTHETIC_DATA.txt").exists()


def load_cmapss_txt(path: str | Path) -> pd.DataFrame:
    """Load a space-separated C-MAPSS file into a DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    # Drop trailing empty columns if present
    df = df.dropna(axis=1, how="all")
    return df


def add_piecewise_rul(df: pd.DataFrame, r_early: int = 125, use_piecewise: bool = True) -> pd.DataFrame:
    """Add RUL column; optionally clip at R_early (piecewise linear target)."""
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    rul = max_cycle - out["cycle"]
    if use_piecewise:
        rul = rul.clip(upper=r_early)
    out["RUL"] = rul.astype(np.float64)
    return out


def load_train(
    data_dir: str | Path,
    subset: str = "FD001",
    r_early: int = 125,
    use_piecewise: bool = True,
) -> pd.DataFrame:
    path = Path(data_dir) / f"train_{subset}.txt"
    df = load_cmapss_txt(path)
    return add_piecewise_rul(df, r_early=r_early, use_piecewise=use_piecewise)


def load_test(
    data_dir: str | Path,
    subset: str = "FD001",
    r_early: int = 125,
    use_piecewise: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load test trajectories and true RUL at end of each truncated run."""
    data_dir = Path(data_dir)
    test_df = load_cmapss_txt(data_dir / f"test_{subset}.txt")
    true_rul = pd.read_csv(
        data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None
    ).values.ravel().astype(np.float64)

    # Build per-cycle RUL for windowing (for training-style metrics on windows)
    rows = []
    for unit_id, g in test_df.groupby("unit"):
        uid = int(unit_id) - 1
        remaining = float(true_rul[uid])
        max_c = g["cycle"].max()
        gg = g.copy()
        # RUL at cycle c = remaining + (max_c - c)
        gg["RUL"] = remaining + (max_c - gg["cycle"])
        if use_piecewise:
            gg["RUL"] = gg["RUL"].clip(upper=r_early)
        rows.append(gg)
    test_with_rul = pd.concat(rows, ignore_index=True)
    return test_with_rul, true_rul


def get_feature_columns(subset: str, sensors: list[str] | None = None) -> list[str]:
    if sensors is not None:
        return list(sensors)
    return list(SENSOR_MAP.get(subset, USEFUL_SENSORS_FD001))


def fit_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    """Fit StandardScaler on training engines only (no leakage)."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler


def transform(df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(df[feature_cols].values).astype(np.float32)


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    sequence_length: int,
    horizons: list[int],
    last_window_only: bool = False,
) -> dict[str, Any]:
    """Create sliding windows without engine leakage (per-unit).

    Returns arrays:
      X: (N, L, F)
      y: (N, H) multi-horizon RUL targets at t+h-1 relative to window end
         For horizon h, target is RUL at the last time step of the window,
         shifted conceptually as RUL_end - (h-1) clipped at 0 — for RUL tasks
         the standard multi-horizon setup predicts RUL at the window end for h=1
         and earlier-in-life targets are not always used; here we define:
         y[..., k] = max(RUL_at_window_end - (horizons[k] - 1), 0)
         so larger horizons correspond to "RUL further ahead" compressed targets.
      Also returns unit ids and end-cycle for each window.
    """
    X_list, y_list, units, end_cycles = [], [], [], []
    feats = transform(df, feature_cols, scaler)
    # attach scaled features temporarily
    work = df[["unit", "cycle", "RUL"]].copy()
    for j, c in enumerate(feature_cols):
        work[f"_f{j}"] = feats[:, j]
    feat_keys = [f"_f{j}" for j in range(len(feature_cols))]
    L = sequence_length
    H = horizons

    for unit_id, g in work.groupby("unit"):
        g = g.sort_values("cycle")
        arr = g[feat_keys].values.astype(np.float32)
        rul = g["RUL"].values.astype(np.float32)
        cycles = g["cycle"].values
        n = len(g)
        # Left-pad with first timestep if trajectory shorter than L (keeps unit alignment)
        if n < L:
            pad = L - n
            arr = np.concatenate([np.repeat(arr[:1], pad, axis=0), arr], axis=0)
            rul = np.concatenate([np.repeat(rul[:1], pad, axis=0), rul], axis=0)
            cycles = np.concatenate([np.repeat(cycles[:1], pad, axis=0), cycles], axis=0)
            n = L
        starts = range(0, n - L + 1)
        if last_window_only:
            starts = [n - L]
        for s in starts:
            e = s + L
            window = arr[s:e]
            rul_end = float(rul[e - 1])
            y_vec = [max(rul_end - (h - 1), 0.0) for h in H]
            X_list.append(window)
            y_list.append(y_vec)
            units.append(int(unit_id))
            end_cycles.append(int(cycles[e - 1]))

    return {
        "X": np.stack(X_list, axis=0) if X_list else np.zeros((0, L, len(feature_cols)), np.float32),
        "y": np.asarray(y_list, dtype=np.float32),
        "units": np.asarray(units, dtype=np.int64),
        "end_cycles": np.asarray(end_cycles, dtype=np.int64),
        "feature_cols": feature_cols,
        "horizons": list(H),
    }


def split_engines(
    train_df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by engine id to avoid leakage."""
    units = np.array(sorted(train_df["unit"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(round(len(units) * val_ratio)))
    val_units = set(units[:n_val].tolist())
    tr = train_df[~train_df["unit"].isin(val_units)].copy()
    va = train_df[train_df["unit"].isin(val_units)].copy()
    return tr, va


def prepare_datasets(
    data_dir: str | Path,
    subset: str = "FD001",
    r_early: int = 125,
    use_piecewise: bool = True,
    sequence_length: int = 30,
    horizons: list[int] | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    sensors: list[str] | None = None,
) -> dict[str, Any]:
    """Full prepare pipeline: load → split engines → scale train-only → windows."""
    if horizons is None:
        horizons = [1, 5, 10]
    data_dir = Path(data_dir)
    train_full = load_train(data_dir, subset, r_early, use_piecewise)
    test_df, true_rul = load_test(data_dir, subset, r_early, use_piecewise)
    feature_cols = get_feature_columns(subset, sensors)

    tr_df, va_df = split_engines(train_full, val_ratio=val_ratio, seed=seed)
    scaler = fit_scaler(tr_df, feature_cols)

    train_w = make_windows(tr_df, feature_cols, scaler, sequence_length, horizons, False)
    val_w = make_windows(va_df, feature_cols, scaler, sequence_length, horizons, False)
    # For test evaluation we use last window per engine (standard C-MAPSS protocol)
    test_w = make_windows(test_df, feature_cols, scaler, sequence_length, horizons, True)

    # Align last-window RUL targets with official RUL file for horizon=1
    # Official metric uses piecewise-clipped true RUL at truncation point.
    clipped_true = true_rul.copy()
    if use_piecewise:
        clipped_true = np.minimum(clipped_true, r_early)

    return {
        "train": train_w,
        "val": val_w,
        "test": test_w,
        "true_rul": clipped_true.astype(np.float32),
        "raw_true_rul": true_rul.astype(np.float32),
        "scaler": scaler,
        "feature_cols": feature_cols,
        "subset": subset,
        "synthetic": is_synthetic(data_dir),
        "sequence_length": sequence_length,
        "horizons": horizons,
        "r_early": r_early,
    }
