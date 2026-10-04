"""Tests for har.data.raw: raw window loading in fixed channel order and the raw DataBundle.

Most tests use a tiny fake dataset written to a temp folder (so they also run in CI); the
``needs_data`` tests check the real UCI HAR files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from har.data.constants import ACTIVITY_NAMES, CHANNELS, WINDOW_LEN
from har.data.normalize import ChannelStandardizer
from har.data.raw import build_raw_loaders, load_raw_split
from har.data.splits import load_val_subjects, read_subjects
from har.utils.paths import UCI_DIR

FAKE_SUBJECTS = {"train": [1, 2, 3, 4], "test": [5, 6]}
FAKE_VAL = [2]
PER_SUBJECT = 5


def _fake_value(c: int, w: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Signal value of channel ``c``, window ``w``, time step ``t`` in the fake dataset."""
    return c * 100 + w + t / 1000


def _write_fake_dataset(root: Path, window_len: int = WINDOW_LEN) -> tuple[Path, Path]:
    """Write a small UCI-HAR-shaped dataset and a matching split file. Returns (uci_dir, split_file)."""
    uci = root / "UCI HAR Dataset"
    for split, subjects in FAKE_SUBJECTS.items():
        signals_dir = uci / split / "Inertial Signals"
        signals_dir.mkdir(parents=True)
        per_window = np.repeat(subjects, PER_SUBJECT)
        n = len(per_window)
        np.savetxt(uci / split / f"subject_{split}.txt", per_window, fmt="%d")
        np.savetxt(uci / split / f"y_{split}.txt", np.arange(n) % 6 + 1, fmt="%d")
        w, t = np.arange(n)[:, None], np.arange(window_len)[None, :]
        # Written in reverse so directory order differs from CHANNELS order.
        for c in reversed(range(len(CHANNELS))):
            np.savetxt(signals_dir / f"{CHANNELS[c]}_{split}.txt", _fake_value(c, w, t), fmt="%.7e")
    split_file = root / "split.json"
    split_file.write_text(json.dumps({
        "seed": 42, "n_val_subjects": 1, "val_subjects": FAKE_VAL, "train_subjects": [1, 3, 4],
        "test_subjects": FAKE_SUBJECTS["test"], "note": "test fixture",
    }), encoding="utf-8")
    return uci, split_file


@pytest.fixture()
def fake(tmp_path: Path) -> tuple[Path, Path]:
    return _write_fake_dataset(tmp_path)


def _all(loader: torch.utils.data.DataLoader) -> tuple[np.ndarray, np.ndarray]:
    x, y = loader.dataset.tensors
    return x.numpy(), y.numpy()


# ---------------------------------------------------------------- load_raw_split (fake data)

def test_load_raw_split_stacks_channels_in_fixed_order(fake: tuple[Path, Path]) -> None:
    uci, _ = fake
    x, y, subjects = load_raw_split("train", uci)
    n = PER_SUBJECT * len(FAKE_SUBJECTS["train"])
    assert x.shape == (n, WINDOW_LEN, 9) and x.dtype == np.float32
    assert y.dtype == np.int64 and subjects.dtype == np.int64
    w, t = np.arange(n)[:, None], np.arange(WINDOW_LEN)[None, :]
    for c in range(len(CHANNELS)):
        np.testing.assert_allclose(x[:, :, c], _fake_value(c, w, t), atol=1e-3)


def test_load_raw_split_shifts_labels_and_reads_subjects(fake: tuple[Path, Path]) -> None:
    uci, _ = fake
    _, y, subjects = load_raw_split("test", uci)
    np.testing.assert_array_equal(y, np.arange(len(y)) % 6)
    np.testing.assert_array_equal(subjects, np.repeat(FAKE_SUBJECTS["test"], PER_SUBJECT))


def test_load_raw_split_rejects_wrong_window_length(tmp_path: Path) -> None:
    uci, _ = _write_fake_dataset(tmp_path, window_len=64)
    with pytest.raises(ValueError, match="expected 128"):
        load_raw_split("train", uci)


def test_load_raw_split_rejects_nan_bad_labels_and_count_mismatch(fake: tuple[Path, Path]) -> None:
    uci, _ = fake
    labels = uci / "test" / "y_test.txt"
    original = labels.read_text(encoding="utf-8")

    labels.write_text(original.replace("1\n", "7\n", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="labels"):
        load_raw_split("test", uci)

    labels.write_text(original + "1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row counts differ"):
        load_raw_split("test", uci)

    labels.write_text(original, encoding="utf-8")
    signal = uci / "test" / "Inertial Signals" / f"{CHANNELS[4]}_test.txt"
    lines = signal.read_text(encoding="utf-8").splitlines()
    lines[0] = " ".join(["nan"] + lines[0].split()[1:])
    signal.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="NaN"):
        load_raw_split("test", uci)


def test_load_raw_split_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="har.data.download"):
        load_raw_split("train", tmp_path)
    with pytest.raises(ValueError):
        load_raw_split("val", tmp_path)  # type: ignore[arg-type]


# ---------------------------------------------------------------- build_raw_loaders (fake data)

def test_loaders_split_by_subject_and_standardise_on_train_only(fake: tuple[Path, Path]) -> None:
    uci, split_file = fake
    bundle = build_raw_loaders(batch_size=4, seed=0, uci_dir=uci, split_file=split_file)
    assert bundle.input_shape == (WINDOW_LEN, 9)
    assert bundle.n_classes == 6 and bundle.class_names == ACTIVITY_NAMES
    assert bundle.meta["counts"] == {"train": 15, "val": 5, "test": 10}
    assert bundle.meta["val_subjects"] == FAKE_VAL
    assert bundle.meta["train_subjects"] == [1, 3, 4]
    assert bundle.meta["representation"] == "raw" and bundle.meta["channels"] == CHANNELS

    # The standardiser was fitted on the training-subject windows only.
    x_all, _, subjects = load_raw_split("train", uci)
    expected = ChannelStandardizer().fit(x_all[subjects != FAKE_VAL[0]])
    restored = ChannelStandardizer.from_dict(bundle.meta["standardizer"])
    np.testing.assert_allclose(restored.mean, expected.mean)
    np.testing.assert_allclose(restored.std, expected.std)

    x_train, _ = _all(bundle.train)
    x_val, _ = _all(bundle.val)
    np.testing.assert_allclose(x_train.mean(axis=(0, 1), dtype=np.float64), 0.0, atol=1e-5)
    np.testing.assert_allclose(x_train.std(axis=(0, 1), dtype=np.float64), 1.0, atol=1e-4)
    # Validation subject 2 sits between train subjects in the fake values, so not at mean 0 exactly.
    assert not np.allclose(x_val.mean(axis=(0, 1), dtype=np.float64), 0.0, atol=1e-3)


def test_loaders_yield_batches_of_the_contract_shape(fake: tuple[Path, Path]) -> None:
    uci, split_file = fake
    bundle = build_raw_loaders(batch_size=4, seed=0, uci_dir=uci, split_file=split_file)
    for loader in (bundle.train, bundle.val, bundle.test):
        x, y = next(iter(loader))
        assert x.shape == (4, WINDOW_LEN, 9) and x.dtype == torch.float32
        assert y.shape == (4,) and y.dtype == torch.int64


def test_train_order_is_seeded_and_eval_loaders_are_not_shuffled(fake: tuple[Path, Path]) -> None:
    uci, split_file = fake

    def train_order(seed: int) -> list[torch.Tensor]:
        """First sample of channel 0 for each window, batch by batch (identifies the window)."""
        bundle = build_raw_loaders(batch_size=4, seed=seed, uci_dir=uci, split_file=split_file)
        return [x[:, 0, 0] for x, _ in bundle.train]

    a, b, c = train_order(1), train_order(1), train_order(2)
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    assert not all(torch.equal(p, q) for p, q in zip(a, c))

    bundle = build_raw_loaders(batch_size=4, seed=1, uci_dir=uci, split_file=split_file)
    x_val, _ = _all(bundle.val)
    batches = torch.cat([x for x, _ in bundle.val]).numpy()
    np.testing.assert_array_equal(batches, x_val)


def test_subset_keeps_first_training_windows_and_full_normalisation(fake: tuple[Path, Path]) -> None:
    uci, split_file = fake
    full = build_raw_loaders(batch_size=4, seed=0, uci_dir=uci, split_file=split_file)
    small = build_raw_loaders(batch_size=4, seed=0, uci_dir=uci, split_file=split_file, subset=6)
    assert small.meta["counts"] == {"train": 6, "val": 5, "test": 10}
    np.testing.assert_array_equal(_all(small.train)[0], _all(full.train)[0][:6])
    assert small.meta["standardizer"] == full.meta["standardizer"]


def _epoch(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    return torch.cat([x for x, _ in loader])


def test_augment_false_leaves_the_data_unchanged(fake: tuple[Path, Path]) -> None:
    uci, split_file = fake
    bundle = build_raw_loaders(batch_size=4, seed=0, uci_dir=uci, split_file=split_file)
    assert bundle.meta["augment"] is False and bundle.meta["augmentation"] is None
    x_train, _ = _all(bundle.train)
    epoch = _epoch(bundle.train).numpy()
    # Same windows, only reordered by the shuffle.
    np.testing.assert_array_equal(np.sort(epoch, axis=0), np.sort(x_train, axis=0))


def test_augment_true_changes_training_windows_only_and_is_seeded(fake: tuple[Path, Path]) -> None:
    uci, split_file = fake

    def build(seed: int):
        return build_raw_loaders(batch_size=4, seed=seed, augment=True, uci_dir=uci, split_file=split_file)

    plain = build_raw_loaders(batch_size=4, seed=0, uci_dir=uci, split_file=split_file)
    aug = build(0)
    assert aug.meta["augment"] is True
    assert aug.meta["augmentation"] == {"jitter_sigma": 0.05, "scale_sigma": 0.1, "p": 0.5}
    assert aug.meta["counts"] == plain.meta["counts"]
    assert not torch.equal(_epoch(aug.train), _epoch(plain.train))
    for split in ("val", "test"):
        assert torch.equal(_epoch(getattr(aug, split)), _epoch(getattr(plain, split)))
    assert torch.equal(_epoch(build(0).train), _epoch(build(0).train))
    assert not torch.equal(_epoch(build(0).train), _epoch(build(1).train))


# ---------------------------------------------------------------- real data

@pytest.fixture(scope="module")
def real_train() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return load_raw_split("train")


@pytest.mark.needs_data
def test_real_shapes_labels_and_subjects(real_train: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    x_test, y_test, s_test = load_raw_split("test")
    for (x, y, s), split, n in ((real_train, "train", 7352), ((x_test, y_test, s_test), "test", 2947)):
        assert x.shape == (n, 128, 9) and x.dtype == np.float32
        assert y.shape == (n,) and y.dtype == np.int64
        assert set(np.unique(y).tolist()) == set(range(6))
        np.testing.assert_array_equal(s, read_subjects(split))
        file_labels = np.loadtxt(UCI_DIR / split / f"y_{split}.txt", dtype=np.int64)
        np.testing.assert_array_equal(y, file_labels - 1)
        assert np.isfinite(x).all()


@pytest.mark.needs_data
def test_real_channels_follow_the_constant_order(real_train: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    x = real_train[0]
    for c, name in enumerate(CHANNELS):
        path = UCI_DIR / "train" / "Inertial Signals" / f"{name}_train.txt"
        with path.open(encoding="utf-8") as f:
            first_window = np.array(f.readline().split(), dtype=np.float32)
        np.testing.assert_array_equal(x[0, :, c], first_window)


@pytest.mark.needs_data
def test_real_loaders_follow_the_protocol() -> None:
    bundle = build_raw_loaders(batch_size=64, seed=42)
    assert bundle.meta["counts"] == {"train": 5276, "val": 2076, "test": 2947}
    assert bundle.meta["val_subjects"] == load_val_subjects()
    assert len(bundle.meta["train_subjects"]) == 15
    assert not set(bundle.meta["train_subjects"]) & set(bundle.meta["val_subjects"])
    x_train, _ = _all(bundle.train)
    np.testing.assert_allclose(x_train.mean(axis=(0, 1), dtype=np.float64), 0.0, atol=1e-5)
    np.testing.assert_allclose(x_train.std(axis=(0, 1), dtype=np.float64), 1.0, atol=1e-4)
    x, y = next(iter(bundle.train))
    assert x.shape == (64, 128, 9) and y.shape == (64,)
