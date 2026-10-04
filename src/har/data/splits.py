"""Subject-wise validation split (fixed protocol: 6 of the 21 training subjects, seed 42).

The split is selected once and committed as ``configs/split.json``; every pipeline reads that
file through ``load_val_subjects``. Splitting by subject rather than by window matters because
UCI HAR windows overlap by 50%, so neighbouring windows of one subject are near-duplicates.

CLI: ``python -m har.data.splits`` prints the selection, ``--check`` compares it with the
committed file, ``--write`` creates the file (refuses to overwrite an existing one).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from har.utils.paths import SPLIT_FILE, UCI_DIR

N_VAL_SUBJECTS = 6
SPLIT_SEED = 42
NOTE = "Selected once; do not regenerate."


def read_subjects(split: Literal["train", "test"], uci_dir: Path = UCI_DIR) -> np.ndarray:
    """Subject id of every window in ``subject_<split>.txt``: int64 ``(N,)``."""
    path = Path(uci_dir) / split / f"subject_{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run python -m har.data.download")
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def select_val_subjects(train_subjects: Sequence[int], n: int = N_VAL_SUBJECTS,
                        seed: int = SPLIT_SEED) -> list[int]:
    """Pick ``n`` validation subjects from the training subjects, deterministically for ``seed``.

    ``train_subjects`` may contain repeats (one entry per window); the unique ids are sorted
    before sampling so the result does not depend on window order. Returns sorted ids.
    """
    unique = sorted({int(s) for s in train_subjects})
    if n >= len(unique):
        raise ValueError(f"Cannot hold out {n} of {len(unique)} training subjects")
    rng = np.random.default_rng(seed)
    return sorted(int(s) for s in rng.choice(unique, size=n, replace=False))


def make_split(train_subjects: Sequence[int], test_subjects: Sequence[int],
               n: int = N_VAL_SUBJECTS, seed: int = SPLIT_SEED) -> dict:
    """The ``split.json`` content for the given per-window (or unique) subject ids."""
    val = select_val_subjects(train_subjects, n=n, seed=seed)
    return {
        "seed": seed,
        "n_val_subjects": n,
        "val_subjects": val,
        "train_subjects": sorted({int(s) for s in train_subjects} - set(val)),
        "test_subjects": sorted({int(s) for s in test_subjects}),
        "note": NOTE,
    }


def load_split(split_file: Path = SPLIT_FILE) -> dict:
    """Read and sanity-check the committed split file."""
    split_file = Path(split_file)
    if not split_file.is_file():
        raise FileNotFoundError(
            f"{split_file} not found. It is committed in the repository (pull main); "
            "never regenerate it with a different seed."
        )
    split = json.loads(split_file.read_text(encoding="utf-8"))
    val, train, test = (set(split[k]) for k in ("val_subjects", "train_subjects", "test_subjects"))
    if len(split["val_subjects"]) != split["n_val_subjects"] or len(val) != len(split["val_subjects"]):
        raise ValueError(f"{split_file}: val_subjects does not hold {split['n_val_subjects']} distinct ids")
    if val & train or val & test or train & test:
        raise ValueError(f"{split_file}: train, val and test subject sets overlap")
    return split


def load_val_subjects(split_file: Path = SPLIT_FILE) -> list[int]:
    """The committed validation subject ids (sorted)."""
    return [int(s) for s in load_split(split_file)["val_subjects"]]


def train_val_masks(subjects: np.ndarray, val_subjects: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks ``(N,)`` over training windows: (train_mask, val_mask), exact complements.

    ``subjects`` is the per-window subject array of the official training split. Raises
    ``ValueError`` if a validation subject has no windows in it (wrong array passed).
    """
    subjects = np.asarray(subjects)
    absent = sorted(set(int(s) for s in val_subjects) - set(np.unique(subjects).tolist()))
    if absent:
        raise ValueError(f"Validation subjects {absent} have no windows in the given subject array")
    val_mask = np.isin(subjects, np.asarray(list(val_subjects)))
    return ~val_mask, val_mask


def format_split(split: dict) -> str:
    """JSON text with one key per line and each list on a single line."""
    body = ",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in split.items())
    return "{\n" + body + "\n}\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Subject-wise validation split for UCI HAR.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="create configs/split.json (once)")
    mode.add_argument("--check", action="store_true", help="compare the selection with configs/split.json")
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    parser.add_argument("--uci-dir", type=Path, default=UCI_DIR)
    args = parser.parse_args(argv)

    train_subjects = read_subjects("train", args.uci_dir)
    split = make_split(train_subjects, read_subjects("test", args.uci_dir))
    val_mask = np.isin(train_subjects, split["val_subjects"])
    print(format_split(split), end="")
    print(f"windows: train {int((~val_mask).sum())}, val {int(val_mask.sum())}")

    if args.check:
        if load_split(args.split_file) != split:
            print(f"MISMATCH with {args.split_file}", file=sys.stderr)
            return 1
        print(f"matches {args.split_file}")
    elif args.write:
        if args.split_file.exists():
            print(f"{args.split_file} already exists; the split is fixed and is not regenerated.",
                  file=sys.stderr)
            return 1
        args.split_file.write_text(format_split(split), encoding="utf-8", newline="\n")
        print(f"wrote {args.split_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
