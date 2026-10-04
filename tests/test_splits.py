"""Tests for har.data.splits: subject-wise validation selection, masks and the split file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from har.data.splits import (
    N_VAL_SUBJECTS, SPLIT_SEED, load_split, load_val_subjects, main, make_split, read_subjects,
    select_val_subjects, train_val_masks,
)

# Official UCI HAR subject split (21 training subjects, 9 test subjects).
TRAIN_SUBJECTS = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
TEST_SUBJECTS = [2, 4, 9, 10, 12, 13, 18, 20, 24]


# ---------------------------------------------------------------- selection

def test_selection_is_deterministic_sorted_and_within_training_subjects() -> None:
    val = select_val_subjects(TRAIN_SUBJECTS)
    assert val == select_val_subjects(TRAIN_SUBJECTS, n=N_VAL_SUBJECTS, seed=SPLIT_SEED)
    assert len(val) == N_VAL_SUBJECTS == 6
    assert val == sorted(set(val))
    assert set(val) <= set(TRAIN_SUBJECTS)


def test_selection_ignores_window_order_and_repeats() -> None:
    per_window = np.repeat(TRAIN_SUBJECTS, 7)
    np.random.default_rng(0).shuffle(per_window)
    assert select_val_subjects(per_window) == select_val_subjects(TRAIN_SUBJECTS)


def test_selection_depends_on_the_seed() -> None:
    assert select_val_subjects(TRAIN_SUBJECTS, seed=43) != select_val_subjects(TRAIN_SUBJECTS, seed=42)


def test_selection_rejects_holding_out_every_subject() -> None:
    with pytest.raises(ValueError):
        select_val_subjects([1, 2, 3], n=3)


def test_make_split_partitions_the_subjects() -> None:
    split = make_split(np.repeat(TRAIN_SUBJECTS, 3), TEST_SUBJECTS)
    val, train, test = (set(split[k]) for k in ("val_subjects", "train_subjects", "test_subjects"))
    assert len(val) == 6 and len(train) == 15 and len(test) == 9
    assert val | train == set(TRAIN_SUBJECTS)
    assert not (val & train or val & test or train & test)
    assert split["seed"] == 42 and split["n_val_subjects"] == 6


# ---------------------------------------------------------------- masks

def test_masks_cover_every_window_exactly_once() -> None:
    subjects = np.array([1, 1, 3, 5, 3, 7, 7, 7, 1])
    train_mask, val_mask = train_val_masks(subjects, [3, 7])
    assert train_mask.dtype == bool and train_mask.shape == subjects.shape
    np.testing.assert_array_equal(train_mask ^ val_mask, np.ones_like(train_mask))
    assert set(subjects[val_mask]) == {3, 7}
    assert set(subjects[train_mask]) == {1, 5}


def test_masks_reject_validation_subjects_without_windows() -> None:
    with pytest.raises(ValueError, match=r"\[9\]"):
        train_val_masks(np.array([1, 2, 3]), [2, 9])


# ---------------------------------------------------------------- committed split

def test_committed_split_has_six_val_subjects_and_disjoint_sets() -> None:
    split = load_split()
    val, train, test = (set(split[k]) for k in ("val_subjects", "train_subjects", "test_subjects"))
    assert split["seed"] == SPLIT_SEED and split["n_val_subjects"] == N_VAL_SUBJECTS
    assert len(val) == 6 and val <= set(TRAIN_SUBJECTS)
    assert val | train == set(TRAIN_SUBJECTS) and len(train) == 15
    assert sorted(test) == TEST_SUBJECTS
    assert not (val & train or val & test or train & test)
    assert load_val_subjects() == split["val_subjects"]


def test_committed_split_equals_the_seeded_selection() -> None:
    assert load_split() == make_split(TRAIN_SUBJECTS, TEST_SUBJECTS)


@pytest.mark.needs_data
def test_committed_split_matches_the_real_subject_files() -> None:
    train_subjects, test_subjects = read_subjects("train"), read_subjects("test")
    assert sorted(set(train_subjects.tolist())) == TRAIN_SUBJECTS
    assert sorted(set(test_subjects.tolist())) == TEST_SUBJECTS
    assert load_split() == make_split(train_subjects, test_subjects)
    assert main(["--check"]) == 0


@pytest.mark.needs_data
def test_real_masks_split_7352_windows_into_5276_and_2076() -> None:
    subjects = read_subjects("train")
    train_mask, val_mask = train_val_masks(subjects, load_val_subjects())
    assert subjects.shape == (7352,)
    assert int(train_mask.sum()) == 5276 and int(val_mask.sum()) == 2076
    assert not np.any(train_mask & val_mask) and np.all(train_mask | val_mask)
    assert set(subjects[val_mask].tolist()) == set(load_val_subjects())


# ---------------------------------------------------------------- split file and CLI

def _fake_uci_dir(root: Path) -> Path:
    for split, subjects in (("train", list(range(1, 11)) * 2), ("test", [11, 12])):
        (root / split).mkdir(parents=True)
        (root / split / f"subject_{split}.txt").write_text(
            "\n".join(str(s) for s in subjects) + "\n", encoding="utf-8")
    return root


def test_read_subjects_reads_one_id_per_window(tmp_path: Path) -> None:
    uci = _fake_uci_dir(tmp_path / "uci")
    subjects = read_subjects("train", uci)
    assert subjects.dtype == np.int64 and subjects.shape == (20,)
    with pytest.raises(FileNotFoundError):
        read_subjects("train", tmp_path / "missing")


def test_cli_writes_once_and_checks(tmp_path: Path) -> None:
    uci = _fake_uci_dir(tmp_path / "uci")
    split_file = tmp_path / "split.json"
    common = ["--split-file", str(split_file), "--uci-dir", str(uci)]

    assert main(["--write", *common]) == 0
    written = split_file.read_bytes()
    assert b"\r\n" not in written
    split = load_split(split_file)
    assert split == make_split(range(1, 11), [11, 12])
    assert load_val_subjects(split_file) == split["val_subjects"]

    assert main(["--write", *common]) == 1  # never overwritten
    assert split_file.read_bytes() == written
    assert main(["--check", *common]) == 0


def test_cli_check_detects_a_different_split(tmp_path: Path) -> None:
    uci = _fake_uci_dir(tmp_path / "uci")
    split_file = tmp_path / "split.json"
    other = make_split(range(1, 11), [11, 12], seed=7)
    split_file.write_text(json.dumps(other), encoding="utf-8")
    assert main(["--check", "--split-file", str(split_file), "--uci-dir", str(uci)]) == 1


def test_load_split_rejects_missing_or_overlapping_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="never regenerate"):
        load_split(tmp_path / "nope.json")
    bad = make_split(range(1, 11), [11, 12])
    bad["train_subjects"].append(bad["val_subjects"][0])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_split(path)
