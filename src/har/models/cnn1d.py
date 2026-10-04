"""Model 2: 1D convolutional network on raw inertial windows (BUILD_SPEC 7.2)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from har.data.constants import CHANNELS, N_CLASSES


class CNN1D(nn.Module):
    """Temporal convolutions over the raw signal: ``(B, 128, 9) -> (B, 6)`` logits (no softmax).

    Each block is Conv1d (padding ``k // 2``, so length is kept) -> BatchNorm1d -> ReLU, with
    MaxPool1d(2) after the blocks listed in ``pool_after`` (zero-based). Global average
    pooling over time, dropout and a linear layer give the logits. The input is permuted to
    ``(B, channels, time)`` internally. With the defaults the model has 197,702 trainable
    parameters.
    """

    def __init__(
        self,
        in_channels: int = len(CHANNELS),
        channels: Sequence[int] = (64, 64, 128, 128, 128),
        kernel_sizes: Sequence[int] = (5, 5, 5, 5, 3),
        pool_after: Sequence[int] = (1, 3),
        dropout: float = 0.5,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        if len(channels) != len(kernel_sizes) or not channels:
            raise ValueError(f"need one kernel size per conv layer: {len(channels)} channels, "
                             f"{len(kernel_sizes)} kernel sizes")
        if any(not 0 <= i < len(channels) for i in pool_after):
            raise ValueError(f"pool_after indices must be in 0..{len(channels) - 1}, got {list(pool_after)}")
        layers: list[nn.Module] = []
        width_in = in_channels
        for i, (width, k) in enumerate(zip(channels, kernel_sizes)):
            layers += [nn.Conv1d(width_in, width, kernel_size=k, padding=k // 2), nn.BatchNorm1d(width), nn.ReLU()]
            if i in pool_after:
                layers.append(nn.MaxPool1d(2))
            width_in = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, in_channels) -> (B, n_classes)`` logits."""
        z = self.features(x.permute(0, 2, 1))        # (B, C, T) for Conv1d
        return self.head(self.dropout(self.pool(z).flatten(1)))
