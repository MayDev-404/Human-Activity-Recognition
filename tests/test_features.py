"""Tests for the engineered-feature pipeline (har.data.features)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from har.data.bundle import DataBundle
from har.data.constants import N_FEATURES
from har.data.features import (
    build_feature_loaders,
    feature_names,
    load_feature_split,
    validate_feature_arrays,
)
from har.utils.paths import SPLIT_FILE

# Official subject split (BUILD_SPEC M3.2).
TRAIN_SUBJECTS = {1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30}
TEST_SUBJECTS = {2, 4, 9, 10, 12, 13, 18, 20, 24}
# Validation subjects used by the stand-in split below (the spec's cross-check selection).
STAND_IN_VAL = [3, 15, 19, 22, 28, 29]


# ---------------------------------------------------------------- validation (no data needed)

def _good_arrays(n: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.zeros((n, N_FEATURES), np.float32), np.arange(n) % 6 + 1, np.ones(n, np.int64)


def test_validate_accepts_well_formed_arrays() -> None:
    validate_feature_arrays(*_good_arrays(), n_rows=6)


@pytest.mark.parametrize("corrupt", ["shape", "nan", "inf", "label_0", "label_7", "rows"])
def test_validate_rejects_bad_arrays(corrupt: str) -> None:
    x, y, subjects = _good_arrays()
    if corrupt == "shape":
        x = x[:, :-1]
    elif corrupt == "nan":
        x[2, 5] = np.nan
    elif corrupt == "inf":
        x[0, 0] = np.inf
    elif corrupt == "label_0":
        y[0] = 0
    elif corrupt == "label_7":
        y[0] = 7
    elif corrupt == "rows":
        subjects = subjects[:-1]
    with pytest.raises(ValueError):
        validate_feature_arrays(x, y, subjects, n_rows=6)


def test_missing_dataset_gives_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="har.data.download"):
        load_feature_split("train", uci_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="har.data.download"):
        feature_names(uci_dir=tmp_path)


def test_unknown_split_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_feature_split("val", uci_dir=tmp_path)  # type: ignore[arg-type]


# ---------------------------------------------------------------- real data

@pytest.mark.needs_data
@pytest.mark.parametrize("split, n_rows, subjects", [("train", 7352, TRAIN_SUBJECTS), ("test", 2947, TEST_SUBJECTS)])
def test_load_feature_split_shapes_labels_and_subjects(split: str, n_rows: int, subjects: set[int]) -> None:
    x, y, s = load_feature_split(split)  # type: ignore[arg-type]
    assert x.shape == (n_rows, N_FEATURES) and x.dtype == np.float32
    assert y.shape == (n_rows,) and y.dtype == np.int64
    assert s.shape == (n_rows,) and s.dtype == np.int64
    assert set(np.unique(y)) == set(range(6))  # shifted from 1..6 to 0..5
    assert set(np.unique(s)) == subjects


@pytest.mark.needs_data
def test_feature_names_are_unique_and_keep_the_original_order() -> None:
    names = feature_names()
    assert len(names) == N_FEATURES == len(set(names))
    assert names[0] == "tBodyAcc-mean()-X"
    # 42 bandsEnergy names appear 3 times each; all 126 occurrences get the column index
    renamed = [n for i, n in enumerate(names) if n.endswith(f"_{i}") and "bandsEnergy" in n]
    assert len(renamed) == 126


def _build_with_stand_in_split(**kwargs: object) -> DataBundle:
    """Build the feature bundle with a stand-in for har.data.splits (independent of M3.2)."""
    fake = types.ModuleType("har.data.splits")
    fake.load_val_subjects = lambda split_file: list(STAND_IN_VAL)
    fake.train_val_masks = lambda subjects, val: (~np.isin(subjects, val), np.isin(subjects, val))
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "har.data.splits", fake)
        return build_feature_loaders(batch_size=64, seed=42, **kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def stand_in_bundle() -> DataBundle:
    return _build_with_stand_in_split()


@pytest.mark.needs_data
def test_bundle_shapes_dtypes_and_counts(stand_in_bundle: DataBundle) -> None:
    b = stand_in_bundle
    assert b.input_shape == (N_FEATURES,) and b.n_classes == 6
    counts = b.meta["counts"]
    assert counts["train"] + counts["val"] == 7352 and counts["test"] == 2947
    for loader in (b.train, b.val, b.test):
        assert loader.num_workers == 0
        x, y = next(iter(loader))
        assert x.shape == (64, N_FEATURES) and x.dtype == torch.float32 and y.dtype == torch.int64


@pytest.mark.needs_data
def test_train_and_val_subjects_are_disjoint(stand_in_bundle: DataBundle) -> None:
    meta = stand_in_bundle.meta
    assert meta["val_subjects"] == STAND_IN_VAL
    assert set(meta["train_subjects"]).isdisjoint(STAND_IN_VAL)
    assert set(meta["train_subjects"]) | set(STAND_IN_VAL) == TRAIN_SUBJECTS
    assert set(meta["test_subjects"]) == TEST_SUBJECTS
    assert meta["normalisation"] == "StandardScaler fitted on 15 training subjects"


@pytest.mark.needs_data
def test_scaler_is_fitted_on_training_subjects_only(stand_in_bundle: DataBundle) -> None:
    x_train = stand_in_bundle.train.dataset.tensors[0].numpy()
    x_val = stand_in_bundle.val.dataset.tensors[0].numpy()
    np.testing.assert_allclose(x_train.mean(axis=0), 0.0, atol=1e-3)
    np.testing.assert_allclose(x_train.std(axis=0), 1.0, atol=1e-3)
    # validation statistics are not forced to 0: the scaler never saw these subjects
    assert np.abs(x_val.mean(axis=0)).max() > 0.05


@pytest.mark.needs_data
def test_subset_shortens_only_the_training_set(stand_in_bundle: DataBundle) -> None:
    b = _build_with_stand_in_split(subset=100)
    assert b.meta["counts"]["train"] == 100 and len(b.train.dataset) == 100
    assert b.meta["counts"]["val"] == stand_in_bundle.meta["counts"]["val"]
    assert b.meta["counts"]["test"] == stand_in_bundle.meta["counts"]["test"]
    assert b.meta["subset"] == 100


# ---------------------------------------------------------------- integration with M3 modules

@pytest.mark.needs_data
def test_validation_subjects_match_committed_split() -> None:
    pytest.importorskip("har.data.splits", reason="har.data.splits lands in M3.2")
    if not SPLIT_FILE.is_file():
        pytest.skip("configs/split.json is committed in M3.2")
    committed = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["val_subjects"]
    b = build_feature_loaders(batch_size=64, seed=42)
    assert b.meta["val_subjects"] == sorted(committed)
    assert set(b.meta["train_subjects"]).isdisjoint(committed)
    assert b.meta["counts"]["train"] + b.meta["counts"]["val"] == 7352


@pytest.mark.needs_data
@pytest.mark.parametrize("split", ["train", "test"])
def test_raw_and_feature_pipelines_align(split: str) -> None:
    """Both representations must describe the same windows in the same order."""
    raw = pytest.importorskip("har.data.raw", reason="har.data.raw lands in M3.3")
    _, y_feat, s_feat = load_feature_split(split)  # type: ignore[arg-type]
    _, y_raw, s_raw = raw.load_raw_split(split)
    np.testing.assert_array_equal(y_feat, y_raw)
    np.testing.assert_array_equal(s_feat, s_raw)
