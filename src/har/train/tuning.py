"""Equal-budget tuning on the validation split (PHASE3_SPEC D2 and 4.1 item 5).

Each model has a grid over ``model.params`` only, in ``configs/tuning/<model>.yaml``:

    model: cnn1d
    base_config: configs/cnn1d.yaml
    grid:
      dropout: [0.5, 0.3]
      kernel_sizes: [[5, 5, 5, 5, 3], [7, 7, 5, 5, 3]]

The first value of every parameter is the Phase 2 default, so configuration ``c1`` is the
interim model. Every configuration trains once under the unchanged shared protocol with seed
42, without the test split and without latency profiling. The best validation macro-F1 wins;
ties (within 0.001) go to fewer parameters, then to the default. The winner is written to
``configs/final/<model>.yaml``. Test scores are never computed, so they cannot influence the choice.
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from har.train.runner import run_experiment
from har.utils.config import load_yaml
from har.utils.paths import CONFIGS, REPO_ROOT, RESULTS_ROOT

TUNING_DIR = CONFIGS / "tuning"
FINAL_DIR = CONFIGS / "final"
TUNING_ROOT = RESULTS_ROOT / "tuning"
TUNING_SEED = 42
TIE_TOLERANCE = 0.001
SUMMARY_FILE = "tuning_summary.csv"
# tuning_summary.csv: config_id, then one column per grid parameter, then these.
RESULT_FIELDS = ("params", "val_acc", "val_macro_f1", "best_epoch", "run_id", "selected")


def _repo_path(path: Path | str) -> Path:
    """``path`` itself if absolute, otherwise relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    """Repository-relative POSIX path when possible (for messages and file headers)."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_grid(path: Path) -> dict[str, Any]:
    """Read a grid file: ``model``, ``base_config`` and ``grid`` (parameter -> non-empty list of values)."""
    spec = load_yaml(path)
    missing = [key for key in ("model", "base_config", "grid") if key not in spec]
    if missing:
        raise ValueError(f"{path}: missing key(s) {missing}")
    grid = spec["grid"]
    if not isinstance(grid, dict) or not grid or not all(isinstance(v, list) and v for v in grid.values()):
        raise ValueError(f"{path}: grid must map each parameter to a non-empty list of values")
    return spec


def expand_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Every combination in file order (the first parameter varies slowest), so the first
    combination takes the first value of every parameter."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def check_grid(grid: dict[str, list], base_params: dict[str, Any]) -> None:
    """Raise unless every grid parameter is in ``base_params``, has distinct values, and starts with the base value."""
    unknown = sorted(set(grid) - set(base_params))
    if unknown:
        raise ValueError(f"grid parameter(s) {unknown} are not in the base config's model.params")
    reserved = sorted(set(grid) & {"config_id", *RESULT_FIELDS})
    if reserved:
        raise ValueError(f"grid parameter(s) {reserved} clash with tuning_summary.csv columns")
    repeated = [k for k, values in grid.items() if len({json.dumps(v) for v in values}) != len(values)]
    if repeated:
        raise ValueError(f"grid parameter(s) {repeated} list the same value twice")
    changed = {k: (values[0], base_params[k]) for k, values in grid.items() if values[0] != base_params[k]}
    if changed:
        raise ValueError(f"the first grid value must be the Phase 2 default; (grid, base) differ for {changed}")


def select_config(rows: list[dict], tolerance: float = TIE_TOLERANCE) -> dict:
    """The winning row (PHASE3_SPEC D2). ``rows`` are in grid order, so ``rows[0]`` is the default.

    Rows whose ``val_macro_f1`` is within ``tolerance`` of the best tie; among them the fewest
    ``params`` wins, then the default, then the higher score, then the earlier configuration.
    """
    if not rows:
        raise ValueError("no tuning results to select from")
    best = max(row["val_macro_f1"] for row in rows)
    tied = [(i, row) for i, row in enumerate(rows) if best - row["val_macro_f1"] <= tolerance + 1e-12]
    _, chosen = min(tied, key=lambda item: (item[1]["params"], item[0] != 0, -item[1]["val_macro_f1"], item[0]))
    return chosen


def _cell(value: Any) -> Any:
    """CSV cell: lists as JSON (``[5, 5, 5, 5, 3]``), booleans as ``true``/``false``."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def write_tuning_summary(rows: list[dict], path: Path) -> Path:
    """One row per configuration with raw values; columns in the order of the row dicts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: _cell(v) for k, v in row.items()} for row in rows)
    return path


def write_final_config(base: dict, params: dict, path: Path, note: str) -> Path:
    """``base`` (same name, owner, data and model target) with ``model.params`` replaced, under a comment header."""
    doc = copy.deepcopy(base)
    doc["model"]["params"] = params
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "".join(f"# {line}\n" for line in note.splitlines())
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None), encoding="utf-8")
    return path


def tune_model(name: str, tuning_dir: Path = TUNING_DIR, final_dir: Path = FINAL_DIR,
               output_root: Path = TUNING_ROOT, force: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Train every configuration of ``<tuning_dir>/<name>.yaml`` and write the summary and final config.

    Run folders go to ``<output_root>/<name>/<run_id>/`` (tags ``tune-c1``, ``tune-c2``, ...),
    the summary to ``<output_root>/<name>/tuning_summary.csv`` and the winner to
    ``<final_dir>/<name>.yaml``. Refuses to start if that file exists, unless ``force``.
    Returns ``{"summary", "final_config", "selected", "rows"}``.
    """
    grid_file = Path(tuning_dir) / f"{name}.yaml"
    spec = load_grid(grid_file)
    final_config = Path(final_dir) / f"{name}.yaml"
    if final_config.exists() and not force:
        raise FileExistsError(f"{_display(final_config)} already exists; pass --force to tune {name} again")
    base_config = _repo_path(spec["base_config"])
    base = load_yaml(base_config)
    if spec["model"] != name or base.get("name") != name:
        raise ValueError(f"{grid_file}: model {spec['model']!r} and the base config name {base.get('name')!r} "
                         f"must both be {name!r}")
    base_params = base["model"].get("params") or {}
    check_grid(spec["grid"], base_params)

    rows = []
    configurations = expand_grid(spec["grid"])
    for i, overrides in enumerate(configurations, start=1):
        config_id = f"c{i}"
        if verbose:
            print(f"[tune {name}] {config_id}/{len(configurations)}: {overrides}")
        run_dir = run_experiment(base_config, no_test=True, no_profile=True, seed=TUNING_SEED,
                                 tag=f"tune-{config_id}", output_root=output_root,
                                 params_override=overrides, verbose=verbose)
        m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"config_id": config_id, **overrides, "params": m["params_trainable"],
                     "val_acc": m["val"]["accuracy"], "val_macro_f1": m["val"]["macro_f1"],
                     "best_epoch": m["best_epoch"], "run_id": m["run_id"]})
    chosen = select_config(rows)
    for row in rows:
        row["selected"] = row is chosen

    summary = write_tuning_summary(rows, Path(output_root) / name / SUMMARY_FILE)
    note = (f"Generated by scripts/tune.py from {_display(grid_file)}: configuration {chosen['config_id']} "
            f"of {len(rows)},\nselected on validation macro-F1 (PHASE3_SPEC D2; see {_display(summary)}).\n"
            f"Do not edit by hand; rerun scripts/tune.py --model {name} --force instead.")
    write_final_config(base, {**base_params, **{k: chosen[k] for k in spec["grid"]}}, final_config, note)
    return {"summary": summary, "final_config": final_config, "selected": chosen["config_id"], "rows": rows}
