"""Raw-signal pipeline: 9-channel inertial windows ``(N, 128, 9)`` (input of the CNN, RNN, Transformer).

Windowing: the dataset authors already segmented the 50 Hz signals into 2.56 s windows of 128
samples with 50% overlap, so each file row is one window and nothing is re-windowed here (the
window length is asserted). Because consecutive windows of one subject share half their
samples, a random window-level split would put near-duplicates in train and validation; the
validation split is therefore by subject (``configs/split.json``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from har.data.augment import AugmentedTensorDataset, RandomAugment
from har.data.bundle import DataBundle
from har.data.constants import ACTIVITY_NAMES, CHANNELS, N_CLASSES, WINDOW_LEN
from har.data.normalize import ChannelStandardizer
from har.data.splits import load_val_subjects, read_subjects, train_val_masks
from har.utils.paths import SPLIT_FILE, UCI_DIR
from har.utils.seed import make_generator


def _read_matrix(path: Path) -> np.ndarray:
    """Whitespace-separated numeric text file -> 2D float32 array."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run python -m har.data.download")
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float32).to_numpy()


def load_raw_split(split: Literal["train", "test"], uci_dir: Path = UCI_DIR
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one official split: X float32 ``(N, 128, 9)`` in ``CHANNELS`` order, y int64 ``(N,)``
    in 0..5, subjects int64 ``(N,)``.

    Channel files are read in the explicit ``CHANNELS`` order, never directory-listing order.
    Raises ``ValueError`` if a window is not 128 samples long, the files disagree on N, values
    are not finite, or labels are outside the file range 1..6.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    split_dir = Path(uci_dir) / split

    signals = []
    for name in CHANNELS:
        path = split_dir / "Inertial Signals" / f"{name}_{split}.txt"
        arr = _read_matrix(path)
        if arr.shape[1] != WINDOW_LEN:
            raise ValueError(f"{path}: windows have {arr.shape[1]} samples, expected {WINDOW_LEN}")
        if signals and arr.shape[0] != signals[0].shape[0]:
            raise ValueError(f"{path}: {arr.shape[0]} windows, but {CHANNELS[0]} has {signals[0].shape[0]}")
        signals.append(arr)
    x = np.stack(signals, axis=-1)

    y = _read_matrix(split_dir / f"y_{split}.txt").ravel().astype(np.int64)
    subjects = read_subjects(split, uci_dir)
    n = x.shape[0]
    if len(y) != n or len(subjects) != n:
        raise ValueError(f"{split}: row counts differ: signals {n}, y {len(y)}, subjects {len(subjects)}")
    if not np.isfinite(x).all():
        raise ValueError(f"{split}: inertial signals contain NaN or inf")
    if y.min() < 1 or y.max() > N_CLASSES:
        raise ValueError(f"{split}: labels must be in 1..{N_CLASSES}, got {y.min()}..{y.max()}")
    return x, y - 1, subjects


def _tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                         torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)))


def build_raw_loaders(
    batch_size: int,
    seed: int,
    augment: bool = False,
    uci_dir: Path = UCI_DIR,
    split_file: Path = SPLIT_FILE,
    subset: int | None = None,
) -> DataBundle:
    """Build the ``raw`` DataBundle: x ``(B, 128, 9)`` float32, y ``(B,)`` int64.

    Training windows are split by subject with the committed validation subjects. A
    ``ChannelStandardizer`` is fitted on the training-subject windows only and applied to
    train, validation and test. The train loader is shuffled with a generator seeded by
    ``seed``. ``subset`` (smoke runs only) keeps the first ``subset`` training windows after
    the split; the standardiser is still fitted on all training-subject windows. ``augment``
    (off for all comparison runs) applies ``RandomAugment`` to training windows only, after
    standardisation, with its own generator seeded from ``seed``.
    """
    x_all, y_all, subjects_all = load_raw_split("train", uci_dir)
    x_test, y_test, subjects_test = load_raw_split("test", uci_dir)
    if set(subjects_all.tolist()) & set(subjects_test.tolist()):
        raise ValueError("official train and test splits share subjects")

    val_subjects = load_val_subjects(split_file)
    train_mask, val_mask = train_val_masks(subjects_all, val_subjects)
    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_val, y_val = x_all[val_mask], y_all[val_mask]
    train_subjects = sorted(int(s) for s in np.unique(subjects_all[train_mask]))

    scaler = ChannelStandardizer().fit(x_train)
    x_train, x_val, x_test = (scaler.transform(x) for x in (x_train, x_val, x_test))

    if subset is not None:
        x_train, y_train = x_train[:subset], y_train[:subset]

    train_set = _tensor_dataset(x_train, y_train)
    augmenter = None
    if augment:
        # Separate stream from the shuffle generator, still fully determined by the seed.
        augmenter = RandomAugment(generator=make_generator(seed + 1))
        train_set = AugmentedTensorDataset(*train_set.tensors, transform=augmenter)

    train = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                       generator=make_generator(seed), num_workers=0)
    val = DataLoader(_tensor_dataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0)
    test = DataLoader(_tensor_dataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=0)

    meta = {
        "representation": "raw",
        "counts": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "val_subjects": val_subjects,
        "train_subjects": train_subjects,
        "test_subjects": sorted(int(s) for s in np.unique(subjects_test)),
        "channels": list(CHANNELS),
        "window_len": WINDOW_LEN,
        "normalisation": f"per-channel z-score fitted on {len(train_subjects)} training subjects",
        "standardizer": scaler.to_dict(),
        "augment": augment,
        "augmentation": augmenter.to_dict() if augmenter else None,
        "subset": subset,
    }
    return DataBundle(train=train, val=val, test=test, input_shape=(WINDOW_LEN, len(CHANNELS)),
                      n_classes=N_CLASSES, class_names=list(ACTIVITY_NAMES), meta=meta)
