"""Baseline models: Persistence, MLP, VanillaLSTM, BiLSTM."""
from __future__ import annotations

import torch
import torch.nn as nn


class Persistence(nn.Module):
    """Naive baseline: predict a constant RUL equal to the mean training RUL.

    At inference, returns a learned (or set) constant for every sample.
    """

    def __init__(self, n_horizons: int = 1, init_value: float = 70.0):
        super().__init__()
        self.n_horizons = n_horizons
        # Store as buffer so it moves with .to(device); can be set from train stats
        self.register_buffer("constant", torch.full((n_horizons,), float(init_value)))

    def set_constant(self, values: torch.Tensor | float) -> None:
        if isinstance(values, (int, float)):
            self.constant.fill_(float(values))
        else:
            self.constant.copy_(values.reshape(-1)[: self.n_horizons])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return self.constant.unsqueeze(0).expand(B, -1).contiguous()


class MLP(nn.Module):
    """Flattened-window MLP regressor."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        n_horizons: int = 1,
    ):
        super().__init__()
        flat = input_dim * sequence_length
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VanillaLSTM(nn.Module):
    """Unidirectional LSTM RUL regressor."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_horizons: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class BiLSTM(nn.Module):
    """Bidirectional LSTM RUL regressor."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_horizons: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
