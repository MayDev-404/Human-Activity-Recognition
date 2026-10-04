"""Train one model under the shared protocol and write its run folder (BUILD_SPEC M2.4).

Usage:
    python scripts/train.py --config configs/<model>.yaml            # full run, committed
    python scripts/train.py --config configs/<model>.yaml --no-test  # while debugging
    python scripts/train.py --config configs/<model>.yaml --smoke    # quick check, results/_smoke/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from har.train.runner import run_experiment
from har.utils.paths import RESULTS_ROOT


def main(argv: list[str] | None = None) -> Path:
    """Parse arguments, run training, evaluation and profiling, and return the run folder."""
    parser = argparse.ArgumentParser(description="Train a HAR model under the shared protocol.")
    parser.add_argument("--config", type=Path, required=True, help="model config, e.g. configs/cnn1d.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="512 training windows, 2 epochs, quick latency; writes under results/_smoke/")
    parser.add_argument("--no-test", action="store_true", help="skip the test split (use while debugging)")
    parser.add_argument("--allow-protocol-override", action="store_true",
                        help="allow train/seed keys in the model config (Phase 3 tuning only; recorded)")
    parser.add_argument("--tag", help="text appended to the run id")
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT, help="default: results/")
    args = parser.parse_args(argv)
    return run_experiment(
        args.config,
        smoke=args.smoke,
        no_test=args.no_test,
        allow_protocol_override=args.allow_protocol_override,
        tag=args.tag,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
