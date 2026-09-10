#!/usr/bin/env python3
"""Create a tiny sample excerpt under data/sample/ for CI / quick demos."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthetic import COLUMN_NAMES, generate_synthetic_subset


def main() -> None:
    out = ROOT / "data" / "sample"
    generate_synthetic_subset(
        "FD001", out, n_train_engines=3, n_test_engines=2, seed=1
    )
    # Truncate to very small for repo
    train = pd.read_csv(out / "train_FD001.txt", sep=r"\s+", header=None)
    train = train.groupby(0).head(40)
    train.to_csv(out / "train_FD001.txt", sep=" ", header=False, index=False)
    (out / "README.md").write_text(
        "# Sample data\n\nTiny synthetic excerpt for pipeline smoke tests.\n"
        "Not real C-MAPSS. Full data: `python scripts/download_data.py`.\n",
        encoding="utf-8",
    )
    print("Wrote", out)


if __name__ == "__main__":
    main()
