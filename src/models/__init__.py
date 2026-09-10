"""Model zoo for C-MAPSS RUL experiments."""
from __future__ import annotations

from src.models.baselines import BiLSTM, MLP, Persistence, VanillaLSTM
from src.models.cnn_bilstm import CNN_BiLSTM
from src.models.cnn_bilstm_attention import BahdanauAttention, CNN_BiLSTM_Attention
from src.models.lstm_cell import ManualLSTM, ManualLSTMCell, ManualLSTMRegressor


def build_model(
    name: str,
    input_dim: int,
    sequence_length: int = 30,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    n_horizons: int = 1,
    cnn_channels: list[int] | None = None,
    cnn_kernel_size: int = 3,
    fail_in_30: bool = False,
    **kwargs,
):
    """Factory for named models."""
    name = name.strip()
    if name == "Persistence":
        return Persistence(n_horizons=n_horizons)
    if name == "MLP":
        return MLP(input_dim, sequence_length, hidden_dim, dropout, n_horizons)
    if name == "VanillaLSTM":
        return VanillaLSTM(input_dim, hidden_dim, num_layers, dropout, n_horizons)
    if name == "BiLSTM":
        return BiLSTM(input_dim, hidden_dim, num_layers, dropout, n_horizons)
    if name == "CNN_BiLSTM":
        return CNN_BiLSTM(
            input_dim, hidden_dim, num_layers, cnn_channels, cnn_kernel_size, dropout, n_horizons
        )
    if name == "CNN_BiLSTM_Attention":
        return CNN_BiLSTM_Attention(
            input_dim,
            hidden_dim,
            num_layers,
            cnn_channels,
            cnn_kernel_size,
            dropout,
            n_horizons,
            fail_in_30=fail_in_30,
        )
    if name == "ManualLSTM":
        return ManualLSTMRegressor(input_dim, hidden_dim, num_layers, dropout, n_horizons)
    raise ValueError(f"Unknown model: {name}")


__all__ = [
    "Persistence",
    "MLP",
    "VanillaLSTM",
    "BiLSTM",
    "CNN_BiLSTM",
    "CNN_BiLSTM_Attention",
    "BahdanauAttention",
    "ManualLSTM",
    "ManualLSTMCell",
    "ManualLSTMRegressor",
    "build_model",
]
