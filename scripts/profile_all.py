"""Profile every model config on this machine without training: parameters, size, latency.

Writes results/profile_<label>.csv. Use a neutral machine label, not the hostname. Each model
uses configs/final/<model>.yaml when it exists, otherwise configs/<model>.yaml. In Phase 3 this
runs once, in one process, on the final configs for the cost comparison (PHASE3_SPEC D4).

Usage: python scripts/profile_all.py --label rtx4060-laptop
"""

from __future__ import annotations

import argparse
from pathlib import Path

from har.eval.results import profile_configs
from har.utils.paths import CONFIGS, RESULTS_ROOT


def main(argv: list[str] | None = None) -> Path:
    """Parse arguments, profile every model config and return the CSV path."""
    parser = argparse.ArgumentParser(description="Same-machine cost profile of every model config.")
    parser.add_argument("--label", required=True, help="neutral machine label, e.g. mayank-laptop")
    parser.add_argument("--configs-dir", type=Path, default=CONFIGS)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args(argv)
    path = profile_configs(args.label, args.configs_dir, args.output_dir)
    print(path)
    return path


if __name__ == "__main__":
    main()
