"""Training-time augmentation for standardised raw windows (off for every comparison run).

Two transforms, each applied with probability ``p`` when ``data.augment: true``: jitter
(additive Gaussian noise) and per-channel scaling (one factor from N(1, sigma) per channel,
constant over time). Comparison runs keep augmentation off so every model sees identical
data; it is available as a possible Phase 3 ablation.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

JITTER_SIGMA = 0.05
SCALE_SIGMA = 0.1
APPLY_PROB = 0.5


def jitter(x: torch.Tensor, sigma: float = JITTER_SIGMA,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """Add N(0, sigma^2) noise to every value. ``(..., T, C) -> (..., T, C)``."""
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
    return x + sigma * noise


def scale(x: torch.Tensor, sigma: float = SCALE_SIGMA,
          generator: torch.Generator | None = None) -> torch.Tensor:
    """Multiply each channel by a factor from N(1, sigma^2), shared over time.

    ``(..., T, C) -> (..., T, C)``; each window in a batch gets its own factors.
    """
    shape = (*x.shape[:-2], 1, x.shape[-1])
    factors = 1.0 + sigma * torch.randn(shape, generator=generator, dtype=x.dtype, device=x.device)
    return x * factors


class RandomAugment:
    """Apply ``jitter`` and ``scale`` independently, each with probability ``p``.

    Randomness comes only from ``generator``, so the same seed gives the same augmented data.
    Works on one window ``(T, C)`` (the decision is made once for the window).
    """

    def __init__(self, jitter_sigma: float = JITTER_SIGMA, scale_sigma: float = SCALE_SIGMA,
                 p: float = APPLY_PROB, generator: torch.Generator | None = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma
        self.p = p
        self.generator = generator

    def _coin(self) -> bool:
        return bool(torch.rand((), generator=self.generator) < self.p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """``(T, C) -> (T, C)`` augmented copy (the input is never modified in place)."""
        if self._coin():
            x = jitter(x, self.jitter_sigma, self.generator)
        if self._coin():
            x = scale(x, self.scale_sigma, self.generator)
        return x

    def to_dict(self) -> dict:
        """Settings for the run metadata."""
        return {"jitter_sigma": self.jitter_sigma, "scale_sigma": self.scale_sigma, "p": self.p}


class AugmentedTensorDataset(TensorDataset):
    """A ``TensorDataset`` of ``(x, y)`` whose items pass ``x`` through ``transform``.

    ``tensors`` still holds the original, un-augmented data. Use with ``num_workers=0`` (the
    project default) so the transform's generator produces one reproducible stream.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, transform: RandomAugment) -> None:
        super().__init__(x, y)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = super().__getitem__(index)
        return self.transform(x), y
