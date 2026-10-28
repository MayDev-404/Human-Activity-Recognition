"""Error analysis of the final runs (PHASE3_SPEC section 5): per-class F1, error structure, error
overlap between models, and shared versus model-specific confusions.

Reads the run folders tagged final (metrics.json and predictions_test.csv) and writes
results/analysis/: per_class_f1.csv/.md/.png, error_structure.csv/.md, error_overlap.csv/.png and
error_summary.json/.md.

Usage: python scripts/error_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from har.eval.errors import CELL_THRESHOLD, write_error_analysis
from har.utils.paths import CONFIGS, RESULTS_ROOT


def main(argv: list[str] | None = None) -> list[Path]:
    """Parse arguments, write the analysis files and return their paths."""
    parser = argparse.ArgumentParser(description="Error analysis of the final runs.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--configs-dir", type=Path, default=CONFIGS)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <results-root>/analysis")
    parser.add_argument("--threshold", type=float, default=CELL_THRESHOLD,
                        help="share of a class's test windows for a confusion cell to count (default 0.02)")
    args = parser.parse_args(argv)
    paths = write_error_analysis(args.results_root, args.configs_dir, args.out_dir, args.threshold)
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()
