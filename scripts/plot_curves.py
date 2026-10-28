"""Results figures: the interim selected runs and, once they exist, the final runs.

Interim (untagged runs, as submitted in Phase 2): results/figures/val_macro_f1_curves.png and
results/figures/confusion_test_<model>.png.
Final (runs tagged final, PHASE3_SPEC 4.3): results/figures/final_val_curves.png,
results/figures/confusion_test_<model>_final.png and results/figures/cost_tradeoff.png (the last
needs results/profile_rtx4060-laptop.csv from scripts/profile_all.py).

Usage: python scripts/plot_curves.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from har.eval.results import write_final_figures, write_selected_figures
from har.utils.paths import CONFIGS, RESULTS_ROOT


def main(argv: list[str] | None = None) -> list[Path]:
    """Parse arguments, write the figures and return their paths."""
    parser = argparse.ArgumentParser(description="Figures for the selected interim run and the final runs.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--configs-dir", type=Path, default=CONFIGS)
    args = parser.parse_args(argv)
    paths = write_selected_figures(args.results_root, args.configs_dir)
    paths += write_final_figures(args.results_root, args.configs_dir)
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()
