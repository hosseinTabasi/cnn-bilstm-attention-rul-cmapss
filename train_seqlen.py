#!/usr/bin/env python3
"""Train CNN_BiLSTM_Attention with a custom sequence length (ablation)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import make_loaders
from src.data.download import download_cmapss
from src.data.preprocess import prepare_datasets
from src.data.synthetic import ensure_synthetic_cmapss
from src.evaluation.evaluate import evaluate_predictions, save_metrics_row
from src.models import build_model
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="FD001")
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-csv", default="experiments/results/metrics.csv")
    p.add_argument("--hidden", type=int, default=64)
    args = p.parse_args()
    set_seed(args.seed)

    data_dir = ROOT / "data" / "raw"
    if not (data_dir / "train_FD001.txt").exists():
        if not download_cmapss(data_dir):
            ensure_synthetic_cmapss(data_dir)
    synthetic = (data_dir / "SYNTHETIC_DATA.txt").exists()

    horizons = [1, 5, 10]
    prepared = prepare_datasets(
        data_dir,
        subset=args.subset,
        sequence_length=args.seq_len,
        horizons=horizons,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = make_loaders(prepared, batch_size=64)
    input_dim = prepared["train"]["X"].shape[-1]
    model = build_model(
        "CNN_BiLSTM_Attention",
        input_dim=input_dim,
        sequence_length=args.seq_len,
        hidden_dim=args.hidden,
        n_horizons=len(horizons),
    )
    trainer = Trainer(model, device="cpu", patience=8)
    trainer.fit(train_loader, val_loader, epochs=args.epochs, verbose=True)
    pred, _ = trainer.predict(test_loader)
    true_rul = prepared["true_rul"]
    n = min(len(pred), len(true_rul))
    metrics = evaluate_predictions(true_rul[:n], pred[:n, 0])
    row = {
        "subset": args.subset,
        "model": "CNN_BiLSTM_Attention",
        "seed": args.seed,
        "epochs": args.epochs,
        "synthetic": int(synthetic),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "score": metrics["score"],
        "tag": f"exp3_L{args.seq_len}",
        "seq_len": args.seq_len,
    }
    save_metrics_row(ROOT / args.results_csv, row, append=True)
    print(row)


if __name__ == "__main__":
    main()
