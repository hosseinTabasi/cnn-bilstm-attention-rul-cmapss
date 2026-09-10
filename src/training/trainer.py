"""Training loop with early stopping."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_all_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        horizon_index: int = 0,
    ):
        self.model = model.to(device)
        self.device = device
        self.horizon_index = horizon_index
        self.criterion = nn.MSELoss()
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = (
            torch.optim.Adam(params, lr=lr, weight_decay=weight_decay) if params else None
        )
        self.patience = patience
        self.best_state: dict[str, Any] | None = None
        self.best_val = float("inf")
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_rmse": []}

    def _predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _step(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total, n = 0.0, 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            if train:
                self.optimizer.zero_grad()
            pred = self._predict_batch(x)
            # Match horizons
            h = min(pred.size(1), y.size(1))
            loss = self.criterion(pred[:, :h], y[:, :h])
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        preds, trues = [], []
        for x, y in loader:
            x = x.to(self.device)
            pred = self._predict_batch(x)
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())
        return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 40,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        wait = 0
        # Persistence: set constant from training targets and skip SGD if no params
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable) == 0 and hasattr(self.model, "set_constant"):
            ys = []
            for _, y in train_loader:
                ys.append(y.numpy())
            mean_y = np.concatenate(ys, axis=0).mean(axis=0)
            self.model.set_constant(torch.tensor(mean_y, dtype=torch.float32))
            val_pred, val_true = self.predict(val_loader)
            hi = min(self.horizon_index, val_pred.shape[1] - 1)
            m = compute_all_metrics(val_true[:, hi], val_pred[:, hi])
            self.history["train_loss"].append(0.0)
            self.history["val_loss"].append(m["rmse"] ** 2)
            self.history["val_rmse"].append(m["rmse"])
            self.best_val = m["rmse"]
            self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            return self.history

        iterator = range(1, epochs + 1)
        for epoch in iterator:
            tr_loss = self._step(train_loader, train=True)
            va_loss = self._step(val_loader, train=False)
            val_pred, val_true = self.predict(val_loader)
            hi = min(self.horizon_index, val_pred.shape[1] - 1)
            m = compute_all_metrics(val_true[:, hi], val_pred[:, hi])
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["val_rmse"].append(m["rmse"])
            if verbose:
                print(
                    f"Epoch {epoch:03d}/{epochs}  train_loss={tr_loss:.4f}  "
                    f"val_loss={va_loss:.4f}  val_rmse={m['rmse']:.4f}"
                )
            if m["rmse"] < self.best_val - 1e-6:
                self.best_val = m["rmse"]
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "history": self.history}, path)
