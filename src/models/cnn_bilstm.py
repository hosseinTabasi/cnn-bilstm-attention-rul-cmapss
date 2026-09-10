"""CNN–BiLSTM hybrid without attention."""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN_BiLSTM(nn.Module):
    """1D CNN feature extractor + Bidirectional LSTM + linear head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        cnn_channels: list[int] | None = None,
        cnn_kernel_size: int = 3,
        dropout: float = 0.3,
        n_horizons: int = 1,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64]
        layers: list[nn.Module] = []
        in_ch = input_dim
        for out_ch in cnn_channels:
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                ]
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            cnn_channels[-1],
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
        # x: (B, L, F) → Conv1d expects (B, F, L)
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # (B, L, C)
        out, _ = self.lstm(z)
        return self.head(out[:, -1, :])
