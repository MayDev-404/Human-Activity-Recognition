"""Tune one model on the validation split over its grid in configs/tuning/<model>.yaml (PHASE3_SPEC D2).

Trains every configuration (seed 42, no test split, no profiling) into results/tuning/<model>/,
writes results/tuning/<model>/tuning_summary.csv and the selected configuration to
configs/final/<model>.yaml. Refuses to run again once that file exists, unless --force.

Usage: python scripts/tune.py --model cnn1d
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from har.train.tuning import FINAL_DIR, TUNING_DIR, TUNING_ROOT, tune_model


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse arguments, tune the model and return the summary path, final config path and selected id."""
    parser = argparse.ArgumentParser(description="Equal-budget tuning of one model on the validation split.")
    parser.add_argument("--model", required=True, help="model name, e.g. cnn1d (reads configs/tuning/<model>.yaml)")
    parser.add_argument("--force", action="store_true", help="tune again even if configs/final/<model>.yaml exists")
    parser.add_argument("--tuning-dir", type=Path, default=TUNING_DIR, help="default: configs/tuning/")
    parser.add_argument("--final-dir", type=Path, default=FINAL_DIR, help="default: configs/final/")
    parser.add_argument("--output-root", type=Path, default=TUNING_ROOT, help="default: results/tuning/")
    args = parser.parse_args(argv)
    result = tune_model(args.model, args.tuning_dir, args.final_dir, args.output_root, force=args.force)
    print(f"selected {result['selected']} -> {result['final_config']}")
    print(f"summary: {result['summary']}")
    return result


if __name__ == "__main__":
    main()
