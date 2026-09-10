"""Manual LSTM cell and ManualLSTM for educational comparison vs nn.LSTM.

Implements the standard LSTM equations:

  i_t = σ(W_ii x_t + b_ii + W_hi h_{t-1} + b_hi)
  f_t = σ(W_if x_t + b_if + W_hf h_{t-1} + b_hf)
  g_t = tanh(W_ig x_t + b_ig + W_hg h_{t-1} + b_hg)
  o_t = σ(W_io x_t + b_io + W_ho h_{t-1} + b_ho)
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
  h_t = o_t ⊙ tanh(c_t)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class ManualLSTMCell(nn.Module):
    """Single-step LSTM cell with explicit gates (no nn.LSTMCell)."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_size)
            state: (h, c) each (B, hidden_size)
        Returns:
            h_new, c_new
        """
        if state is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        gates = x @ self.weight_ih.t() + h @ self.weight_hh.t()
        if self.bias_ih is not None:
            gates = gates + self.bias_ih + self.bias_hh
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ManualLSTM(nn.Module):
    """Multi-layer (optionally stacked) unidirectional LSTM using ManualLSTMCell.

    API mirrors a subset of nn.LSTM: input (B, T, F) → output (B, T, H), (h_n, c_n).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        super().__init__()
        if not batch_first:
            raise ValueError("ManualLSTM only supports batch_first=True")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        cells = []
        for layer in range(num_layers):
            in_sz = input_size if layer == 0 else hidden_size
            cells.append(ManualLSTMCell(in_sz, hidden_size, bias=bias))
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (B, T, input_size)
            hx: optional (h_0, c_0) each (num_layers, B, hidden_size)
        Returns:
            output: (B, T, hidden_size)
            (h_n, c_n): each (num_layers, B, hidden_size)
        """
        B, T, _ = x.shape
        if hx is None:
            h_states = [
                torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
                for _ in range(self.num_layers)
            ]
            c_states = [
                torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
                for _ in range(self.num_layers)
            ]
        else:
            h_states = [hx[0][i] for i in range(self.num_layers)]
            c_states = [hx[1][i] for i in range(self.num_layers)]

        layer_input = x
        for layer, cell in enumerate(self.cells):
            h, c = h_states[layer], c_states[layer]
            outs = []
            for t in range(T):
                h, c = cell(layer_input[:, t, :], (h, c))
                outs.append(h)
            seq = torch.stack(outs, dim=1)  # (B, T, H)
            h_states[layer] = h
            c_states[layer] = c
            if layer < self.num_layers - 1:
                seq = self.dropout(seq)
            layer_input = seq

        h_n = torch.stack(h_states, dim=0)
        c_n = torch.stack(c_states, dim=0)
        return layer_input, (h_n, c_n)


class ManualLSTMRegressor(nn.Module):
    """RUL regressor using ManualLSTM (for comparison experiments)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_horizons: int = 1,
    ):
        super().__init__()
        self.lstm = ManualLSTM(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
