"""Seeding helpers so every run with the same seed sees the same data order and initial weights."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices).

    If ``deterministic``, also make cuDNN pick deterministic kernels and disable its
    autotuner. ``torch.use_deterministic_algorithms(True)`` is deliberately not called:
    it raises on CUDA unless ``CUBLAS_WORKSPACE_CONFIG`` is set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded with ``seed`` (for shuffled DataLoaders)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
