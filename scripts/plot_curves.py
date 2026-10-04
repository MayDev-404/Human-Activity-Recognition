"""Overlay validation macro-F1 curves of the selected runs and copy their test confusion matrices.

Writes results/figures/val_macro_f1_curves.png and results/figures/confusion_test_<model>.png.

Usage: python scripts/plot_curves.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from har.eval.results import write_selected_figures
from har.utils.paths import CONFIGS, RESULTS_ROOT


def main(argv: list[str] | None = None) -> list[Path]:
    """Parse arguments, write the figures and return their paths."""
    parser = argparse.ArgumentParser(description="Figures for the selected run of each model.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--configs-dir", type=Path, default=CONFIGS)
    args = parser.parse_args(argv)
    paths = write_selected_figures(args.results_root, args.configs_dir)
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()
