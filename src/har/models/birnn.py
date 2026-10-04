"""Model 3: bidirectional recurrent classifier on raw inertial windows, LSTM or GRU (BUILD_SPEC 7.3)."""

from __future__ import annotations

import torch
from torch import nn

from har.data.constants import CHANNELS, N_CLASSES

CELLS: dict[str, type[nn.Module]] = {"lstm": nn.LSTM, "gru": nn.GRU}
POOLINGS = ("last", "mean")


class BiRNN(nn.Module):
    """Gated recurrence over the window: ``(B, 128, 9) -> (B, 6)`` logits (no softmax).

    A stacked ``nn.LSTM`` (or ``nn.GRU`` when ``cell="gru"``, the ablation) with
    ``batch_first=True``, then pooling, dropout and a linear head. ``pooling="last"`` uses the
    final hidden states of the top layer (forward and backward concatenated when
    bidirectional); ``pooling="mean"`` averages the top-layer outputs over time. With the
    defaults the LSTM has 138,502 trainable parameters and the GRU 104,070.
    """

    def __init__(
        self,
        in_channels: int = len(CHANNELS),
        hidden: int = 64,
        num_layers: int = 2,
        cell: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.3,
        pooling: str = "last",
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {sorted(CELLS)}, got {cell!r}")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        self.cell = cell
        self.pooling = pooling
        self.bidirectional = bidirectional
        # Inter-layer dropout only exists between stacked layers; PyTorch warns if set for one layer.
        self.rnn = CELLS[cell](
            input_size=in_channels, hidden_size=hidden, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, in_channels) -> (B, n_classes)`` logits."""
        out, state = self.rnn(x)                          # out (B, T, D * hidden)
        if self.pooling == "mean":
            features = out.mean(dim=1)                    # (B, D * hidden)
        else:
            h_n = state[0] if self.cell == "lstm" else state   # (num_layers * D, B, hidden)
            if self.bidirectional:
                features = torch.cat([h_n[-2], h_n[-1]], dim=1)  # top layer: forward, backward
            else:
                features = h_n[-1]
        return self.head(self.dropout(features))
