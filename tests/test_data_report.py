"""Tests for har.data.stats and scripts/data_report.py (dataset statistics and figures)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from har.data.constants import ACTIVITY_NAMES, CHANNELS, WINDOW_LEN
from har.data.stats import (
    activity_label, class_counts, pick_example_windows, plot_class_distribution, read_labels,
    stats_markdown, summarise_splits, write_dataset_report,
)
from har.utils.paths import REPO_ROOT

EM_DASH = "—"


def _write_fake_dataset(root: Path) -> tuple[Path, Path]:
    """Train subjects 1-4 (val subject 2) and test subjects 5-6; 12 windows each, 2 per activity."""
    uci = root / "UCI HAR Dataset"
    rng = np.random.default_rng(0)
    for split, subjects in (("train", [1, 2, 3, 4]), ("test", [5, 6])):
        (uci / split / "Inertial Signals").mkdir(parents=True)
        per_window = np.repeat(subjects, 12)
        labels = np.tile(np.repeat(np.arange(1, 7), 2), len(subjects))
        np.savetxt(uci / split / f"subject_{split}.txt", per_window, fmt="%d")
        np.savetxt(uci / split / f"y_{split}.txt", labels, fmt="%d")
        for name in CHANNELS:
            signal = 0.1 * rng.standard_normal((len(per_window), WINDOW_LEN))
            np.savetxt(uci / split / "Inertial Signals" / f"{name}_{split}.txt", signal, fmt="%.6e")
    split_file = root / "split.json"
    split_file.write_text(json.dumps({
        "seed": 42, "n_val_subjects": 1, "val_subjects": [2], "train_subjects": [1, 3, 4],
        "test_subjects": [5, 6], "note": "test fixture",
    }), encoding="utf-8")
    return uci, split_file


@pytest.fixture()
def fake(tmp_path: Path) -> tuple[Path, Path]:
    return _write_fake_dataset(tmp_path / "data")


def test_class_counts_include_absent_classes() -> None:
    np.testing.assert_array_equal(class_counts(np.array([0, 0, 5])), [2, 0, 0, 0, 0, 1])


def test_activity_label_is_sentence_case() -> None:
    assert [activity_label(n) for n in ACTIVITY_NAMES][:3] == [
        "Walking", "Walking upstairs", "Walking downstairs"]


def test_read_labels_shifts_to_zero_based_and_validates(fake: tuple[Path, Path], tmp_path: Path) -> None:
    uci, _ = fake
    y = read_labels("train", uci)
    assert y.dtype == np.int64 and y.min() == 0 and y.max() == 5
    (uci / "test" / "y_test.txt").write_text("0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="labels"):
        read_labels("test", uci)
    with pytest.raises(FileNotFoundError):
        read_labels("train", tmp_path / "missing")


def test_summary_splits_training_subjects_by_the_split_file(fake: tuple[Path, Path]) -> None:
    summary = summarise_splits(*fake)
    assert {s: set(summary[s]["subjects"].tolist()) for s in summary} == {
        "train": {1, 3, 4}, "val": {2}, "test": {5, 6}}
    np.testing.assert_array_equal(class_counts(summary["train"]["labels"]), [6] * 6)
    np.testing.assert_array_equal(class_counts(summary["val"]["labels"]), [2] * 6)


def test_markdown_tables_totals_and_no_em_dash(fake: tuple[Path, Path]) -> None:
    text = stats_markdown(summarise_splits(*fake), {"Caption": "figures/x.png"})
    assert "| WALKING | 6 (16.7%) | 2 (16.7%) | 4 (16.7%) | 12 |" in text
    assert "| **Total** | **36** | **12** | **24** | **72** |" in text
    assert "| Val | 2 | 1 | 12 | 12 to 12 |" in text
    assert "![Caption](figures/x.png)" in text
    assert EM_DASH not in text


def test_pick_example_windows_uses_one_subject_with_every_activity() -> None:
    labels = np.array([0, 0, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
    subjects = np.array([1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3])
    labels[3] = 0      # subject 1 now lacks activity 1, so subject 3 is used
    subject, indices = pick_example_windows(labels, subjects, [1, 3])
    assert subject == 3 and indices == [8, 9, 10, 11, 12, 13]
    with pytest.raises(ValueError):
        pick_example_windows(labels, subjects, [1])


def test_pick_example_windows_takes_the_middle_window() -> None:
    labels = np.repeat(np.arange(6), 5)
    subject, indices = pick_example_windows(labels, np.ones(30, dtype=int), [1])
    assert subject == 1 and indices == [2, 7, 12, 17, 22, 27]


def test_report_writes_markdown_and_both_figures(fake: tuple[Path, Path], tmp_path: Path) -> None:
    out = tmp_path / "docs" / "dataset_stats.md"
    paths = write_dataset_report(out, tmp_path / "docs" / "figures", *fake)
    for key in ("class_distribution", "sample_windows"):
        assert paths[key].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    text = out.read_text(encoding="utf-8")
    assert "](figures/class_distribution.png)" in text and "](figures/sample_windows.png)" in text
    # Output is deterministic: regenerating gives byte-identical files.
    before = {k: p.read_bytes() for k, p in paths.items()}
    write_dataset_report(out, tmp_path / "docs" / "figures", *fake)
    assert {k: p.read_bytes() for k, p in paths.items()} == before


def test_plot_creates_parent_folders(fake: tuple[Path, Path], tmp_path: Path) -> None:
    path = plot_class_distribution(summarise_splits(*fake), tmp_path / "a" / "b" / "c.png")
    assert path.is_file()


def test_script_main_passes_its_arguments(fake: tuple[Path, Path], tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("data_report", REPO_ROOT / "scripts" / "data_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    uci, split_file = fake
    out = tmp_path / "stats.md"
    paths = module.main(["--out", str(out), "--figures-dir", str(tmp_path / "figs"),
                         "--uci-dir", str(uci), "--split-file", str(split_file)])
    assert paths["stats"] == out and out.is_file()
    assert (tmp_path / "figs" / "sample_windows.png").is_file()


@pytest.mark.needs_data
def test_real_counts_match_the_protocol() -> None:
    summary = summarise_splits()
    totals = {s: int(class_counts(summary[s]["labels"]).sum()) for s in summary}
    assert totals == {"train": 5276, "val": 2076, "test": 2947}
    assert {s: len(np.unique(summary[s]["subjects"])) for s in summary} == {"train": 15, "val": 6, "test": 9}
    for s in summary:
        assert (class_counts(summary[s]["labels"]) > 0).all()
