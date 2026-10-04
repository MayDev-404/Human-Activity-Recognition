"""Repository paths. Every module takes paths from here; never hard-code absolute paths.

The dataset folder name contains spaces (``UCI HAR Dataset``), so always use ``pathlib.Path``.
"""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Return the nearest directory at or above ``start`` that contains ``pyproject.toml``.

    Searches upwards from this file first (editable install), then from the current
    working directory. Raises ``RuntimeError`` if neither finds the repository.
    """
    candidates = [start] if start is not None else [Path(__file__).resolve().parent, Path.cwd()]
    for origin in candidates:
        for directory in (origin, *origin.parents):
            if (directory / "pyproject.toml").is_file():
                return directory
    raise RuntimeError(
        "Could not find the repository root (no pyproject.toml above this file or the "
        "working directory). Install the package with `pip install -e .` from the repo root."
    )


REPO_ROOT: Path = find_repo_root()
DATA_ROOT: Path = REPO_ROOT / "data" / "raw"
UCI_DIR: Path = DATA_ROOT / "UCI HAR Dataset"
CONFIGS: Path = REPO_ROOT / "configs"
SPLIT_FILE: Path = CONFIGS / "split.json"
RESULTS_ROOT: Path = REPO_ROOT / "results"
