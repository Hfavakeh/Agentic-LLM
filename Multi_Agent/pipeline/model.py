"""The LSTM localizer network."""

from typing import Tuple

import torch
import torch.nn as nn


class LSTM_Localizer(nn.Module):
    """Pure LSTM network for 2-D position regression from time-series sensor data."""

    def __init__(
        self,
        input_features: int,
        target_dim: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_features, lstm_hidden, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])

    def count_parameters(self) -> Tuple[int, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
