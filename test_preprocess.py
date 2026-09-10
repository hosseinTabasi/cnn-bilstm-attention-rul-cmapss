"""Basic preprocess / windowing smoke tests on synthetic sample."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.preprocess import prepare_datasets
from src.data.synthetic import generate_synthetic_subset

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_no_engine_leakage(tmp_path):
    generate_synthetic_subset("FD001", tmp_path, n_train_engines=10, n_test_engines=4, seed=0)
    prepared = prepare_datasets(tmp_path, subset="FD001", sequence_length=20, horizons=[1, 5], seed=0)
    assert prepared["train"]["X"].ndim == 3
    assert prepared["test"]["X"].shape[0] == 4  # last window per test engine
    assert prepared["synthetic"] is True
    # val and train units disjoint
    assert len(set(prepared["train"]["units"]) & set(prepared["val"]["units"])) == 0
