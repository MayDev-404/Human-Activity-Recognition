"""The ``DataBundle`` contract returned by every data pipeline, plus a synthetic bundle for tests.

Every DataLoader yields ``(x, y)`` with ``x`` float32 of shape ``(B, *input_shape)`` and ``y``
int64 of shape ``(B,)`` holding class indices in ``0..n_classes-1``. DataLoaders use
``num_workers=0`` so they work on Windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from har.data.constants import ACTIVITY_NAMES
from har.utils.seed import make_generator


@dataclass
class DataBundle:
    """Train, validation and test loaders plus what a model and the train script need to know.

    ``input_shape`` is ``(561,)`` for features and ``(128, 9)`` for raw windows. ``meta`` holds
    the representation name, window counts per split, validation subjects and normalisation info.
    """

    train: DataLoader          # shuffled with a seeded torch.Generator
    val: DataLoader            # not shuffled
    test: DataLoader           # not shuffled
    input_shape: tuple[int, ...]
    n_classes: int
    class_names: list[str]
    meta: dict


def make_synthetic_bundle(
    input_shape: tuple[int, ...],
    n_train: int = 256,
    n_val: int = 64,
    n_test: int = 64,
    n_classes: int = 6,
    batch_size: int = 32,
    seed: int = 0,
    learnable: bool = True,
) -> DataBundle:
    """Random data for unit tests: x ``(n, *input_shape)`` float32 from N(0, 1), y ``(n,)`` int64.

    If ``learnable``, each label is the argmax of a fixed random linear projection of the
    flattened input, so a model can fit (and generalise to) the labels. Otherwise labels are
    uniform random and independent of the inputs. The same ``seed`` gives identical data and
    the same training-batch order.
    """
    input_shape = tuple(int(d) for d in input_shape)
    n_total = n_train + n_val + n_test
    generator = torch.Generator().manual_seed(seed)

    x = torch.randn((n_total, *input_shape), generator=generator, dtype=torch.float32)
    if learnable:
        projection = torch.randn((x[0].numel(), n_classes), generator=generator, dtype=torch.float32)
        y = (x.reshape(n_total, -1) @ projection).argmax(dim=1)
    else:
        y = torch.randint(0, n_classes, (n_total,), generator=generator)
    y = y.to(torch.int64)

    bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, n_total)}
    datasets = {name: TensorDataset(x[a:b], y[a:b]) for name, (a, b) in bounds.items()}

    train = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                       generator=make_generator(seed), num_workers=0)
    val = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0)
    test = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = list(ACTIVITY_NAMES) if n_classes == len(ACTIVITY_NAMES) else [f"class_{i}" for i in range(n_classes)]
    meta = {
        "representation": "synthetic",
        "learnable": learnable,
        "seed": seed,
        "counts": {"train": n_train, "val": n_val, "test": n_test},
    }
    return DataBundle(train=train, val=val, test=test, input_shape=input_shape,
                      n_classes=n_classes, class_names=class_names, meta=meta)
