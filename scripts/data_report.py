"""Generate docs/dataset_stats.md and the dataset figures in docs/figures/.

Usage: python scripts/data_report.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from har.data.stats import write_dataset_report
from har.utils.paths import REPO_ROOT, SPLIT_FILE, UCI_DIR


def main(argv: list[str] | None = None) -> dict[str, Path]:
    """Parse arguments, write the report files and return their paths."""
    parser = argparse.ArgumentParser(description="Dataset statistics and figures for UCI HAR.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "dataset_stats.md")
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "docs" / "figures")
    parser.add_argument("--uci-dir", type=Path, default=UCI_DIR)
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    args = parser.parse_args(argv)
    paths = write_dataset_report(args.out, args.figures_dir, args.uci_dir, args.split_file)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return paths


if __name__ == "__main__":
    main()
