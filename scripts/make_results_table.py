"""Write results/summary.csv and results/summary.md and refresh the README results block.

For each model config, the full run with the best validation macro-F1 is selected.

Usage: python scripts/make_results_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from har.eval.results import write_results_table
from har.utils.paths import CONFIGS, REPO_ROOT, RESULTS_ROOT


def main(argv: list[str] | None = None) -> dict[str, Path]:
    """Parse arguments, write the summary files and return their paths."""
    parser = argparse.ArgumentParser(description="Results table from the committed run folders.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--configs-dir", type=Path, default=CONFIGS)
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md")
    parser.add_argument("--no-readme", action="store_true", help="do not touch the README")
    args = parser.parse_args(argv)
    paths = write_results_table(args.results_root, args.configs_dir, None if args.no_readme else args.readme)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return paths


if __name__ == "__main__":
    main()
