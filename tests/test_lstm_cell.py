"""Tests for ManualLSTMCell / ManualLSTM vs nn.LSTM numerical agreement."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.lstm_cell import ManualLSTM, ManualLSTMCell


def test_manual_cell_shapes():
    cell = ManualLSTMCell(8, 16)
    x = torch.randn(4, 8)
    h, c = cell(x)
    assert h.shape == (4, 16)
    assert c.shape == (4, 16)


def test_manual_lstm_shapes():
    lstm = ManualLSTM(8, 16, num_layers=2, dropout=0.0)
    x = torch.randn(3, 10, 8)
    out, (hn, cn) = lstm(x)
    assert out.shape == (3, 10, 16)
    assert hn.shape == (2, 3, 16)
    assert cn.shape == (2, 3, 16)


def _copy_lstm_weights(manual: ManualLSTM, ref: nn.LSTM) -> None:
    """Copy nn.LSTM weights into ManualLSTM for a fair comparison."""
    for layer in range(manual.num_layers):
        cell = manual.cells[layer]
        # nn.LSTM stores weight_ih_l{k}, weight_hh_l{k}, bias_ih_l{k}, bias_hh_l{k}
        wh_ih = getattr(ref, f"weight_ih_l{layer}").detach().clone()
        wh_hh = getattr(ref, f"weight_hh_l{layer}").detach().clone()
        cell.weight_ih.data.copy_(wh_ih)
        cell.weight_hh.data.copy_(wh_hh)
        if cell.bias_ih is not None:
            cell.bias_ih.data.copy_(getattr(ref, f"bias_ih_l{layer}").detach())
            cell.bias_hh.data.copy_(getattr(ref, f"bias_hh_l{layer}").detach())


def test_manual_lstm_matches_nn_lstm():
    torch.manual_seed(0)
    B, T, F, H, L = 2, 7, 5, 11, 2
    ref = nn.LSTM(F, H, num_layers=L, batch_first=True, dropout=0.0)
    manual = ManualLSTM(F, H, num_layers=L, dropout=0.0)
    _copy_lstm_weights(manual, ref)
    x = torch.randn(B, T, F)
    with torch.no_grad():
        out_m, (hn_m, cn_m) = manual(x)
        out_r, (hn_r, cn_r) = ref(x)
    assert torch.allclose(out_m, out_r, atol=1e-5, rtol=1e-4)
    assert torch.allclose(hn_m, hn_r, atol=1e-5, rtol=1e-4)
    assert torch.allclose(cn_m, cn_r, atol=1e-5, rtol=1e-4)


def test_manual_lstm_backward():
    lstm = ManualLSTM(4, 8, num_layers=1)
    x = torch.randn(2, 5, 4, requires_grad=True)
    out, _ = lstm(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert lstm.cells[0].weight_ih.grad is not None
