"""Per-channel standardisation of raw inertial windows, fitted on training subjects only."""

from __future__ import annotations

import numpy as np

STD_FLOOR = 1e-8


class ChannelStandardizer:
    """Z-score each channel of ``(N, T, C)`` windows with a mean and std taken over N and T.

    Statistics are computed in float64 and kept as ``(C,)`` arrays; a std below ``STD_FLOOR``
    is raised to it so a constant channel maps to zeros instead of NaN.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "ChannelStandardizer":
        """Fit on ``x`` of shape ``(N, T, C)``. Returns ``self``."""
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[0] == 0:
            raise ValueError(f"expected non-empty (N, T, C) windows, got shape {x.shape}")
        self.mean = x.mean(axis=(0, 1), dtype=np.float64)
        self.std = np.maximum(x.std(axis=(0, 1), dtype=np.float64), STD_FLOOR)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """``(N, T, C) -> (N, T, C)`` float32 standardised copy; ``x`` is not modified."""
        if self.mean is None or self.std is None:
            raise RuntimeError("ChannelStandardizer must be fitted before transform")
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[-1] != self.mean.shape[0]:
            raise ValueError(f"expected (N, T, {self.mean.shape[0]}) windows, got shape {x.shape}")
        return ((x - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """``fit(x)`` then ``transform(x)``."""
        return self.fit(x).transform(x)

    def to_dict(self) -> dict:
        """JSON-serialisable statistics: ``{'mean': [C floats], 'std': [C floats]}``."""
        if self.mean is None or self.std is None:
            raise RuntimeError("ChannelStandardizer must be fitted before to_dict")
        return {"mean": [float(v) for v in self.mean], "std": [float(v) for v in self.std]}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelStandardizer":
        """Rebuild a fitted standardiser from ``to_dict`` output."""
        mean = np.asarray(d["mean"], dtype=np.float64)
        std = np.asarray(d["std"], dtype=np.float64)
        if mean.ndim != 1 or mean.shape != std.shape:
            raise ValueError("mean and std must be 1D lists of the same length")
        out = cls()
        out.mean, out.std = mean, np.maximum(std, STD_FLOOR)
        return out
