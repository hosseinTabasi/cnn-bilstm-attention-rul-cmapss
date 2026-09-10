#!/usr/bin/env python3
"""Run Experiments 1–6 with modest CPU-friendly defaults.

Exp1: Model ladder on FD001 (Persistence → MLP → VanillaLSTM → BiLSTM → CNN_BiLSTM → CNN_BiLSTM_Attention)
Exp2: Multi-horizon evaluation {1,5,10} for main model
Exp3: Sequence length ablation L ∈ {20,30,50}
Exp4: ManualLSTM vs nn.LSTM (VanillaLSTM) on FD001
Exp5: Multi-seed stability (seeds 42, 43, 44) for CNN_BiLSTM_Attention
Exp6: FD003 main-model transfer / confirmation run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--quick", action="store_true", help="Fewer models / epochs for smoke test")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Subset of exp ids to run, e.g. 1 4 6",
    )
    args = parser.parse_args()
    epochs = 15 if args.quick else args.epochs
    only = set(int(x) for x in args.only) if args.only else {1, 2, 3, 4, 5, 6}

    # Ensure data
    run([sys.executable, "scripts/download_data.py"])

    py = sys.executable
    csv = "experiments/results/metrics.csv"
    # Fresh metrics file for this campaign
    metrics_path = ROOT / csv
    if metrics_path.exists():
        metrics_path.unlink()

    if 1 in only:
        models = (
            ["Persistence", "MLP", "VanillaLSTM", "BiLSTM", "CNN_BiLSTM", "CNN_BiLSTM_Attention"]
            if not args.quick
            else ["Persistence", "VanillaLSTM", "CNN_BiLSTM_Attention"]
        )
        for m in models:
            run(
                [
                    py,
                    "scripts/train.py",
                    "--model",
                    m,
                    "--subset",
                    "FD001",
                    "--epochs",
                    str(epochs),
                    "--seed",
                    "42",
                    "--results-csv",
                    csv,
                    "--tag",
                    "exp1_ladder",
                ]
            )

    if 2 in only:
        # Multi-horizon already trained in models with horizons [1,5,10]; re-run main model tagged
        run(
            [
                py,
                "scripts/train.py",
                "--model",
                "CNN_BiLSTM_Attention",
                "--subset",
                "FD001",
                "--epochs",
                str(epochs),
                "--seed",
                "42",
                "--results-csv",
                csv,
                "--tag",
                "exp2_multihorizon",
            ]
        )

    if 3 in only:
        # Sequence length ablation via temporary config edits through env-less CLI:
        # We call prepare via a small inline runner
        for L in ([30] if args.quick else [20, 30, 50]):
            run(
                [
                    py,
                    "scripts/train_seqlen.py",
                    "--subset",
                    "FD001",
                    "--seq-len",
                    str(L),
                    "--epochs",
                    str(max(10, epochs - 5)),
                    "--seed",
                    "42",
                    "--results-csv",
                    csv,
                ]
            )

    if 4 in only:
        for m in ["VanillaLSTM", "ManualLSTM"]:
            run(
                [
                    py,
                    "scripts/train.py",
                    "--model",
                    m,
                    "--subset",
                    "FD001",
                    "--epochs",
                    str(max(10, epochs - 5)),
                    "--seed",
                    "42",
                    "--results-csv",
                    csv,
                    "--tag",
                    "exp4_manual_cell",
                ]
            )

    if 5 in only:
        seeds = [42] if args.quick else [42, 43, 44]
        for s in seeds:
            run(
                [
                    py,
                    "scripts/train.py",
                    "--model",
                    "CNN_BiLSTM_Attention",
                    "--subset",
                    "FD001",
                    "--epochs",
                    str(max(10, epochs - 5)),
                    "--seed",
                    str(s),
                    "--results-csv",
                    csv,
                    "--tag",
                    "exp5_seeds",
                ]
            )

    if 6 in only:
        run(
            [
                py,
                "scripts/train.py",
                "--model",
                "CNN_BiLSTM_Attention",
                "--subset",
                "FD003",
                "--epochs",
                str(epochs),
                "--seed",
                "42",
                "--results-csv",
                csv,
                "--tag",
                "exp6_fd003",
            ]
        )

    print("\nAll requested experiments finished. See", csv)


if __name__ == "__main__":
    main()
