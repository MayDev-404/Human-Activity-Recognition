"""Shared pytest setup. Tests marked ``needs_data`` skip automatically when UCI HAR is missing."""

from __future__ import annotations

from pathlib import Path

import pytest

from har.utils.paths import UCI_DIR

NEEDS_DATA_REASON = "UCI HAR not downloaded; run python -m har.data.download"

# A few files from each part of the dataset that the loaders read.
_REQUIRED_FILES = (
    "features.txt",
    "train/X_train.txt",
    "test/X_test.txt",
    "train/Inertial Signals/body_acc_x_train.txt",
    "test/Inertial Signals/body_acc_x_test.txt",
)


def data_available(uci_dir: Path = UCI_DIR) -> bool:
    """True if the extracted 'UCI HAR Dataset' folder is in place (features and raw signals)."""
    return all((uci_dir / rel).is_file() for rel in _REQUIRED_FILES)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip every test marked ``needs_data`` when the dataset has not been downloaded."""
    if data_available():
        return
    skip = pytest.mark.skip(reason=NEEDS_DATA_REASON)
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)
