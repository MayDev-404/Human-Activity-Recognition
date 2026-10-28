"""Write the results tables and refresh the README results block.

Interim: results/summary.csv and summary.md from the untagged runs (for each model, the run
with the best validation macro-F1). Final (once runs tagged final exist): results/final_summary.csv
and final_summary.md, mean and sample std over seeds 42, 43, 44; the README then shows this table.

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
