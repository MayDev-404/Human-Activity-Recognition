"""Model 1: multilayer perceptron on the 561 engineered features (BUILD_SPEC 7.1)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from har.data.constants import N_CLASSES, N_FEATURES


class MLP(nn.Module):
    """Feed-forward baseline: ``(B, 561) -> (B, 6)`` logits (no softmax).

    Each hidden layer is Linear -> BatchNorm1d -> ReLU -> Dropout. With the default
    ``hidden=(256, 128)`` the model has 178,310 trainable parameters.
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width_in = input_dim
        for width in hidden:
            layers += [nn.Linear(width_in, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)]
            width_in = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, input_dim) -> (B, n_classes)`` logits."""
        return self.head(self.body(x))
