"""Engineered-feature pipeline: the 561 time and frequency domain features (input of the MLP).

The features were computed by the dataset authors from the same 2.56 s windows as the raw
signals and are already scaled to [-1, 1]. We still standardise them with a ``StandardScaler``
fitted on the training subjects only, so every model sees zero-mean, unit-variance inputs
without information leaking from validation or test subjects.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from har.data.bundle import DataBundle
from har.data.constants import ACTIVITY_NAMES, N_CLASSES, N_FEATURES
from har.utils.paths import SPLIT_FILE, UCI_DIR
from har.utils.seed import make_generator

EXPECTED_ROWS: dict[str, int] = {"train": 7352, "test": 2947}


def _read_table(path: Path) -> np.ndarray:
    """Read a whitespace-separated numeric text file into a 2D array."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run python -m har.data.download")
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy()


def validate_feature_arrays(x: np.ndarray, y: np.ndarray, subjects: np.ndarray, n_rows: int) -> None:
    """Raise ``ValueError`` unless x is ``(n_rows, 561)`` and finite, y and subjects have one
    entry per row, and labels are the file values 1..6 (before shifting to 0..5)."""
    if x.shape != (n_rows, N_FEATURES):
        raise ValueError(f"feature matrix has shape {x.shape}, expected {(n_rows, N_FEATURES)}")
    if len(y) != n_rows or len(subjects) != n_rows:
        raise ValueError(f"row counts differ: X {n_rows}, y {len(y)}, subjects {len(subjects)}")
    if not np.isfinite(x).all():
        raise ValueError("feature matrix contains NaN or inf")
    if y.min() < 1 or y.max() > N_CLASSES:
        raise ValueError(f"labels must be in 1..{N_CLASSES}, got {y.min()}..{y.max()}")


def load_feature_split(
    split: Literal["train", "test"], uci_dir: Path = UCI_DIR
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one official split: X float32 ``(N, 561)``, y int64 ``(N,)`` in 0..5, subjects int64 ``(N,)``.

    Reads ``X_<split>.txt``, ``y_<split>.txt`` and ``subject_<split>.txt`` from
    ``uci_dir/<split>/`` and validates them (train has 7352 rows, test 2947).
    """
    if split not in EXPECTED_ROWS:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    split_dir = Path(uci_dir) / split
    x = _read_table(split_dir / f"X_{split}.txt").astype(np.float32)
    y = _read_table(split_dir / f"y_{split}.txt").ravel().astype(np.int64)
    subjects = _read_table(split_dir / f"subject_{split}.txt").ravel().astype(np.int64)
    validate_feature_arrays(x, y, subjects, EXPECTED_ROWS[split])
    return x, y - 1, subjects


def feature_names(uci_dir: Path = UCI_DIR) -> list[str]:
    """Return the 561 feature names from ``features.txt``, made unique.

    The file repeats 42 ``bandsEnergy`` names (only 477 are unique), so every occurrence of a
    repeated name gets its zero-based column index in X appended, e.g. ``..._302``.
    """
    path = Path(uci_dir) / "features.txt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run python -m har.data.download")
    names = [line.split(maxsplit=1)[1] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != N_FEATURES:
        raise ValueError(f"{path} lists {len(names)} features, expected {N_FEATURES}")
    counts = Counter(names)
    unique = [f"{name}_{i}" if counts[name] > 1 else name for i, name in enumerate(names)]
    if len(set(unique)) != N_FEATURES:
        raise ValueError("feature names are still not unique after de-duplication")
    return unique


def _tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                         torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)))


def build_feature_loaders(
    batch_size: int,
    seed: int,
    uci_dir: Path = UCI_DIR,
    split_file: Path = SPLIT_FILE,
    subset: int | None = None,
) -> DataBundle:
    """Build the ``features`` DataBundle: x ``(B, 561)`` float32, y ``(B,)`` int64.

    Training windows are split into train and validation by subject using the committed
    validation subjects in ``split_file``. The ``StandardScaler`` is fitted on the windows of
    the training subjects only, then applied to train, validation and test. The train loader
    is shuffled with a generator seeded by ``seed``. ``subset`` (smoke runs only) keeps the
    first ``subset`` training windows after the split; the scaler is still fitted on all
    training-subject windows.
    """
    # Imported here so the rest of this module works before the split module (M3.2) exists.
    from har.data.splits import load_val_subjects, train_val_masks

    x_all, y_all, subjects_all = load_feature_split("train", uci_dir)
    x_test, y_test, subjects_test = load_feature_split("test", uci_dir)

    val_subjects = [int(s) for s in load_val_subjects(split_file)]
    train_mask, val_mask = train_val_masks(subjects_all, val_subjects)
    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_val, y_val = x_all[val_mask], y_all[val_mask]
    train_subjects = sorted(int(s) for s in np.unique(subjects_all[train_mask]))

    scaler = StandardScaler().fit(x_train)
    x_train, x_val, x_test = (scaler.transform(x).astype(np.float32) for x in (x_train, x_val, x_test))

    if subset is not None:
        x_train, y_train = x_train[:subset], y_train[:subset]

    train = DataLoader(_tensor_dataset(x_train, y_train), batch_size=batch_size, shuffle=True,
                       generator=make_generator(seed), num_workers=0)
    val = DataLoader(_tensor_dataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0)
    test = DataLoader(_tensor_dataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=0)

    meta = {
        "representation": "features",
        "counts": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "val_subjects": val_subjects,
        "train_subjects": train_subjects,
        "test_subjects": sorted(int(s) for s in np.unique(subjects_test)),
        "normalisation": f"StandardScaler fitted on {len(train_subjects)} training subjects",
        "subset": subset,
    }
    return DataBundle(train=train, val=val, test=test, input_shape=(N_FEATURES,),
                      n_classes=N_CLASSES, class_names=list(ACTIVITY_NAMES), meta=meta)
