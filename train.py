#!/usr/bin/env python3
"""Train a single named model on one C-MAPSS subset."""
from __future__ import annotations

import argparse
import json
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
from src.evaluation.evaluate import evaluate_predictions, plot_rul_scatter, save_metrics_row
from src.models import build_model
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed


def ensure_data(data_dir: Path, allow_synthetic: bool = True) -> bool:
    """Returns True if synthetic."""
    if (data_dir / "train_FD001.txt").exists():
        return (data_dir / "SYNTHETIC_DATA.txt").exists()
    ok = download_cmapss(data_dir)
    if ok:
        return False
    if allow_synthetic:
        ensure_synthetic_cmapss(data_dir)
        return True
    raise FileNotFoundError("C-MAPSS not available and synthetic disabled")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="CNN_BiLSTM_Attention")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--results-csv", default="experiments/results/metrics.csv")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    ds_cfg = cfg.get("dataset", {})
    tr_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    subset = args.subset or ds_cfg.get("subset", "FD001")
    epochs = args.epochs or tr_cfg.get("epochs", 40)
    seed = args.seed if args.seed is not None else tr_cfg.get("seed", 42)
    set_seed(seed)

    data_dir = ROOT / ds_cfg.get("data_dir", "data/raw")
    synthetic = ensure_data(data_dir, allow_synthetic=ds_cfg.get("synthetic_fallback", True))

    horizons = ds_cfg.get("horizons", [1, 5, 10])
    prepared = prepare_datasets(
        data_dir,
        subset=subset,
        r_early=ds_cfg.get("r_early", 125),
        use_piecewise=ds_cfg.get("use_piecewise", True),
        sequence_length=ds_cfg.get("sequence_length", 30),
        horizons=horizons,
        val_ratio=tr_cfg.get("val_ratio", 0.2),
        seed=seed,
        sensors=ds_cfg.get("sensors"),
    )
    train_loader, val_loader, test_loader = make_loaders(
        prepared, batch_size=tr_cfg.get("batch_size", 64), num_workers=tr_cfg.get("num_workers", 0)
    )
    input_dim = prepared["train"]["X"].shape[-1]
    seq_len = prepared["sequence_length"]
    n_horizons = len(horizons)

    model = build_model(
        args.model,
        input_dim=input_dim,
        sequence_length=seq_len,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
        n_horizons=n_horizons,
        cnn_channels=model_cfg.get("cnn_channels", [32, 64]),
        cnn_kernel_size=model_cfg.get("cnn_kernel_size", 3),
        fail_in_30=ds_cfg.get("fail_in_30", False),
    )

    device = tr_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    trainer = Trainer(
        model,
        device=device,
        lr=tr_cfg.get("lr", 1e-3),
        weight_decay=tr_cfg.get("weight_decay", 1e-5),
        patience=tr_cfg.get("early_stopping_patience", 10),
    )
    trainer.fit(train_loader, val_loader, epochs=epochs, verbose=True)

    # Test: last-window protocol — compare pred[:,0] to official clipped RUL
    pred, y_win = trainer.predict(test_loader)
    true_rul = prepared["true_rul"]
    # Align lengths
    n = min(len(pred), len(true_rul))
    metrics = evaluate_predictions(true_rul[:n], pred[:n, 0], horizon_index=0, label=args.model)
    print("TEST metrics (horizon=1, last window):", metrics)

    # Multi-horizon window metrics on test windows
    mh = {}
    for i, h in enumerate(horizons):
        if i < pred.shape[1]:
            mh[f"rmse_h{h}"] = float(
                np.sqrt(np.mean((y_win[:n, i] - pred[:n, i]) ** 2))
            )

    row = {
        "subset": subset,
        "model": args.model,
        "seed": seed,
        "epochs": epochs,
        "synthetic": int(synthetic or prepared["synthetic"]),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "score": metrics["score"],
        "tag": args.tag,
        **mh,
    }
    save_metrics_row(ROOT / args.results_csv, row, append=True)

    fig_dir = ROOT / "figures"
    plot_rul_scatter(
        true_rul[:n],
        pred[:n, 0],
        fig_dir / f"{subset}_{args.model}_seed{seed}_scatter.png",
        title=f"{subset} {args.model} (synth={row['synthetic']})",
    )
    ckpt = ROOT / "experiments" / "results" / f"{subset}_{args.model}_seed{seed}.pt"
    trainer.save(ckpt)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
