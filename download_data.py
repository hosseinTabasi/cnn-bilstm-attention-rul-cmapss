#!/usr/bin/env python3
"""Download NASA C-MAPSS or generate clearly labeled synthetic fallback."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.download import download_cmapss
from src.data.synthetic import ensure_synthetic_cmapss


def main() -> None:
    parser = argparse.ArgumentParser(description="Download C-MAPSS or synthetic fallback")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--force-synthetic", action="store_true")
    args = parser.parse_args()
    data_dir = ROOT / args.data_dir

    if args.force_synthetic:
        ensure_synthetic_cmapss(data_dir)
        print(f"SYNTHETIC data written to {data_dir}")
        return

    ok = download_cmapss(data_dir)
    if not ok:
        print("Falling back to SYNTHETIC multi-engine degradation data...")
        ensure_synthetic_cmapss(data_dir)
        print(f"SYNTHETIC data written to {data_dir} — label all experiment tables.")
    else:
        print(f"Real C-MAPSS data ready at {data_dir}")


if __name__ == "__main__":
    main()
