"""PyTorch Dataset wrappers for windowed C-MAPSS tensors."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class RULWindowDataset(Dataset):
    """(X, y) where X is (L, F) and y is (H,) multi-horizon RUL."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_loaders(
    prepared: dict[str, Any],
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = RULWindowDataset(prepared["train"]["X"], prepared["train"]["y"])
    val_ds = RULWindowDataset(prepared["val"]["X"], prepared["val"]["y"])
    test_ds = RULWindowDataset(prepared["test"]["X"], prepared["test"]["y"])
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
