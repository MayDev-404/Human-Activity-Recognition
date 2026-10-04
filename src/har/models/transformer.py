"""Model 4: Transformer encoder on raw inertial windows (BUILD_SPEC 7.4)."""

from __future__ import annotations

import torch
from torch import nn

from har.data.constants import CHANNELS, N_CLASSES, WINDOW_LEN


class TransformerHAR(nn.Module):
    """Self-attention over all time steps: ``(B, 128, 9) -> (B, 6)`` logits (no softmax).

    Linear input projection to ``d_model``, plus a learned positional embedding
    (init N(0, 0.02)), dropout, a pre-norm GELU ``TransformerEncoder``, a final LayerNorm,
    mean pooling over time and a linear head. With the defaults the model has 159,302
    trainable parameters. Pre-norm keeps training stable at the shared lr of 1e-3 without
    warm-up; ``enable_nested_tensor=False`` avoids a PyTorch warning with pre-norm layers.
    """

    def __init__(
        self,
        in_channels: int = len(CHANNELS),
        seq_len: int = WINDOW_LEN,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len, d_model))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, in_channels) -> (B, n_classes)`` logits, with ``T <= seq_len`` (128 for UCI HAR)."""
        if x.shape[1] > self.seq_len:
            raise ValueError(f"sequence length {x.shape[1]} exceeds seq_len={self.seq_len}")
        h = self.input_proj(x) + self.pos_embedding[:, : x.shape[1]]   # (B, T, d_model)
        h = self.encoder(self.dropout(h))                               # (B, T, d_model)
        return self.head(self.norm(h).mean(dim=1))                      # (B, n_classes)
