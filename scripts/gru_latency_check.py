"""GRU versus BiLSTM CPU latency at batch size 1, with oneDNN enabled and disabled (PHASE3_SPEC F3.3).

Times configs/final/bilstm.yaml and configs/final/gru.yaml in one process and writes
results/analysis/gru_latency_check.csv and gru_latency_check.md.

Usage: python scripts/gru_latency_check.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from har.eval.latency_check import REPEATS, write_latency_check
from har.utils.paths import CONFIGS, RESULTS_ROOT


def main(argv: list[str] | None = None) -> list[Path]:
    """Parse arguments, run the check and return the written paths."""
    parser = argparse.ArgumentParser(description="GRU vs BiLSTM CPU latency with oneDNN on and off.")
    parser.add_argument("--configs-dir", type=Path, default=CONFIGS)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <results-root>/analysis")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args(argv)
    paths = write_latency_check(args.configs_dir, args.out_dir, args.results_root, args.repeats)
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()
