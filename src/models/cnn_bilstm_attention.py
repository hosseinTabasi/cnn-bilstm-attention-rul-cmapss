"""CNN–BiLSTM with Bahdanau (additive) temporal attention."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive attention over time steps.

    score(h_t) = v^T tanh(W h_t)
    α = softmax(score)
    context = Σ α_t h_t
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, T, H)
        Returns:
            context: (B, H)
            weights: (B, T)
        """
        energy = torch.tanh(self.W(h))          # (B, T, H)
        scores = self.v(energy).squeeze(-1)     # (B, T)
        weights = F.softmax(scores, dim=1)      # (B, T)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, H)
        return context, weights


class CNN_BiLSTM_Attention(nn.Module):
    """Hybrid CNN–BiLSTM with temporal Bahdanau attention for multi-horizon RUL.

    Returns predictions and (optionally) attention weights for interpretability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        cnn_channels: list[int] | None = None,
        cnn_kernel_size: int = 3,
        dropout: float = 0.3,
        n_horizons: int = 1,
        fail_in_30: bool = False,
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
        attn_dim = hidden_dim * 2
        self.attention = BahdanauAttention(attn_dim)
        self.dropout = nn.Dropout(dropout)
        self.rul_head = nn.Linear(attn_dim, n_horizons)
        self.fail_in_30 = fail_in_30
        self.fail_head = nn.Linear(attn_dim, 1) if fail_in_30 else None

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(z)  # (B, T, 2H)
        context, weights = self.attention(out)
        context = self.dropout(context)
        rul = self.rul_head(context)
        if self.fail_head is not None:
            fail_logit = self.fail_head(context)
            if return_attention:
                return rul, fail_logit, weights
            return rul, fail_logit
        if return_attention:
            return rul, weights
        return rul
