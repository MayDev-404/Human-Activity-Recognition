"""Results tables, figures and same-machine profiling (BUILD_SPEC M2.6 and 8.5, PHASE3_SPEC 4.1 and 4.3).

Every number written here is read from ``results/<model>/<run_id>/metrics.json`` files produced
by ``scripts/train.py``. Two tables:

- **Interim** (``summary.csv``/``.md``): untagged runs only, so it stays as submitted in Phase 2.
  For each model the run with the best **validation** macro-F1 is selected; test scores never
  influence the choice.
- **Final** (``final_summary.csv``/``.md``): the runs tagged ``final``, exactly one per seed in
  ``final_seeds`` (base.yaml), reported as mean and sample standard deviation.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
import warnings
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from har.data.constants import ACTIVITY_NAMES, CHANNELS, N_FEATURES, WINDOW_LEN
from har.eval.plots import plot_confusion, plot_cost_tradeoff, plot_val_curves
from har.eval.profiling import best_available_device, count_params, hardware_info, measure_latency, model_size_mb
from har.train.runner import FINAL_CONFIG_DIR, FINAL_TAG
from har.utils.build import build_from_target
from har.utils.config import load_config, load_yaml
from har.utils.paths import CONFIGS, REPO_ROOT, RESULTS_ROOT
from har.utils.seed import set_seed

NON_MODEL_CONFIGS = frozenset({"base.yaml", "team.yaml"})
PROFILE_LABEL = "rtx4060-laptop"   # PHASE3_SPEC D4: the one machine for the final cost comparison
# Report order (Models 1 to 4, GRU after its BiLSTM); any other model follows alphabetically.
MODEL_ORDER = ("mlp", "cnn1d", "bilstm", "gru", "transformer")
INPUT_SHAPES: dict[str, tuple[int, ...]] = {"features": (N_FEATURES,), "raw": (WINDOW_LEN, len(CHANNELS))}
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

COLUMNS = [  # (summary.csv key, summary.md header)
    ("model", "Model"), ("run_by", "Run by"), ("representation", "Representation"), ("params", "Params"),
    ("val_acc", "Val Acc"), ("val_macro_f1", "Val Macro-F1"), ("test_acc", "Test Acc"),
    ("test_macro_f1", "Test Macro-F1"), ("best_epoch", "Best epoch"), ("train_time_s", "Train time (s)"),
    ("cpu_latency_bs1_ms", "CPU latency bs=1 (ms)"), ("hardware", "Hardware"), ("runs", "Runs"),
    ("run_id", "Run ID"),
]

# final_summary.csv: every metric below gets a <name>_mean and a <name>_std column (raw floats).
FINAL_METRICS: dict[str, Callable[[dict], float]] = {
    "val_acc": lambda m: m["val"]["accuracy"],
    "val_macro_f1": lambda m: m["val"]["macro_f1"],
    "test_acc": lambda m: m["test"]["accuracy"],
    "test_macro_f1": lambda m: m["test"]["macro_f1"],
    "train_time_s": lambda m: m["train_time_s"],
}
FINAL_CSV_COLUMNS = [
    "model", "run_by", "representation", "params",
    *(f"{name}_{stat}" for name in [*FINAL_METRICS, "cpu_latency_bs1_ms"] for stat in ("mean", "std")),
    "cpu_latency_source", "hardware", "seeds", "run_ids",
]
FINAL_MD_COLUMNS = [  # (field, final_summary.md header)
    ("model", "Model"), ("run_by", "Run by"), ("representation", "Representation"), ("params", "Params"),
    ("val_macro_f1", "Val Macro-F1"), ("test_acc", "Test Acc"), ("test_macro_f1", "Test Macro-F1"),
    ("train_time_s", "Train time (s)"), ("cpu_latency_bs1_ms", "CPU latency bs=1 (ms)"), ("seeds", "Seeds"),
]
RUNS_LATENCY_SOURCE = "mean of the final runs"


def model_configs(configs_dir: Path = CONFIGS) -> dict[str, Path]:
    """Model name -> config path for every ``configs/*.yaml`` except base and team, in report order."""
    found = {}
    for path in sorted(Path(configs_dir).glob("*.yaml")):
        if path.name not in NON_MODEL_CONFIGS:
            found[load_yaml(path).get("name") or path.stem] = path
    rank = {name: i for i, name in enumerate(MODEL_ORDER)}
    return dict(sorted(found.items(), key=lambda item: (rank.get(item[0], len(rank)), item[0])))


def final_configs(configs_dir: Path = CONFIGS) -> dict[str, Path]:
    """Like ``model_configs``, but each model's ``configs/final/<name>.yaml`` when it exists."""
    final_dir = Path(configs_dir) / FINAL_CONFIG_DIR
    return {name: final_dir / f"{name}.yaml" if (final_dir / f"{name}.yaml").is_file() else path
            for name, path in model_configs(configs_dir).items()}


def _read_runs(model: str, results_root: Path, keep: Callable[[dict], bool]) -> list[tuple[Path, dict]]:
    """``(run_dir, metrics)`` for the non-smoke, non-override runs of ``model`` that ``keep`` accepts, oldest first."""
    runs = []
    for path in sorted((Path(results_root) / model).glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if not metrics.get("smoke") and not metrics.get("protocol_override") and keep(metrics):
            runs.append((path.parent, metrics))
    return runs


def full_runs(model: str, results_root: Path = RESULTS_ROOT) -> list[tuple[Path, dict]]:
    """``(run_dir, metrics)`` for every untagged full run of ``model`` (the interim runs), oldest first.

    Smoke runs live under ``results/_smoke/`` and are never read; runs marked ``smoke`` or
    ``protocol_override`` are skipped as well, and so are tagged runs (Phase 3 final runs), so
    the interim table stays as submitted.
    """
    return _read_runs(model, results_root, lambda m: not m.get("tag"))


def final_runs(model: str, results_root: Path = RESULTS_ROOT) -> list[tuple[Path, dict]]:
    """``(run_dir, metrics)`` for every run of ``model`` tagged ``final``, oldest first."""
    return _read_runs(model, results_root, lambda m: m.get("tag") == FINAL_TAG)


def select_best(runs: list[tuple[Path, dict]]) -> tuple[Path, dict]:
    """The run with the highest validation macro-F1 (the oldest one on ties)."""
    if not runs:
        raise ValueError("no runs to select from")
    return max(runs, key=lambda run: run[1]["val"]["macro_f1"])   # max keeps the first maximum


def selected_runs(results_root: Path = RESULTS_ROOT, configs_dir: Path = CONFIGS
                  ) -> dict[str, tuple[Path, dict, int]]:
    """Model -> ``(run_dir, metrics, number of full runs)`` for every model that has a full run."""
    selected = {}
    for model in model_configs(configs_dir):
        runs = full_runs(model, results_root)
        if runs:
            selected[model] = (*select_best(runs), len(runs))
    return selected


def hardware_label(hw: dict) -> str:
    """``GPU <name>, CPU <name>`` when training used a GPU, else ``CPU <name>``."""
    cpu = f"CPU {hw.get('cpu') or hw.get('processor') or 'unknown'}"
    if hw.get("device", "cpu") != "cpu" and hw.get("gpu"):
        return f"GPU {hw['gpu']}, {cpu}"
    return cpu


def summary_rows(selected: dict[str, tuple[Path, dict, int]]) -> list[dict]:
    """One row of raw values per model (keys as in ``COLUMNS``); missing test scores are None."""
    rows = []
    for model, (_, m, n_runs) in selected.items():
        test = m.get("test") or {}
        rows.append({
            "model": model,
            "run_by": m["run_by"],
            "representation": m["representation"],
            "params": m["params_trainable"],
            "val_acc": m["val"]["accuracy"],
            "val_macro_f1": m["val"]["macro_f1"],
            "test_acc": test.get("accuracy"),
            "test_macro_f1": test.get("macro_f1"),
            "best_epoch": m["best_epoch"],
            "train_time_s": m["train_time_s"],
            "cpu_latency_bs1_ms": ((m.get("latency") or {}).get("cpu_bs1") or {}).get("mean_ms"),
            "hardware": hardware_label(m.get("hardware", {})),
            "runs": n_runs,
            "run_id": m["run_id"],
        })
    return rows


def write_summary_csv(rows: list[dict], path: Path, fieldnames: list[str] | None = None) -> Path:
    """Raw values, no formatting; None becomes an empty cell. Columns default to the interim ``COLUMNS``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or [key for key, _ in COLUMNS])
        writer.writeheader()
        writer.writerows({k: ("" if v is None else v) for k, v in row.items()} for row in rows)
    return path


def _format(key: str, value) -> str:
    if value is None:
        return "n/a"
    if key in ("val_acc", "test_acc"):
        return f"{100 * value:.2f}%"
    if key in ("val_macro_f1", "test_macro_f1"):
        return f"{value:.3f}"
    if key == "params":
        return f"{value:,}"
    if key == "train_time_s":
        return f"{value:.1f}"
    if key == "cpu_latency_bs1_ms":
        return f"{value:.3f}"
    return str(value).replace("|", "/")


def timing_note(rows: list[dict]) -> str:
    """Whether training time and latency are comparable across rows, judged from their hardware labels."""
    labels = {row["hardware"] for row in rows}
    if len(labels) == 1:
        return (f"All runs shown report the same hardware ({labels.pop()}), so training time and "
                "latency are comparable across rows.")
    if labels:
        return ("Training time and latency were measured on different machines (see Hardware), so "
                "they are not directly comparable across rows.")
    return ""


def render_summary_md(rows: list[dict], untrained: list[str]) -> str:
    """Formatted markdown table with a short provenance note (also written into the README)."""
    lines = [
        "<!-- Generated by scripts/make_results_table.py from results/<model>/<run_id>/metrics.json. "
        "Do not edit by hand. -->",
        ("Preliminary interim results: untuned, single seed (42), shared protocol. For each model the "
         "full run with the best validation macro-F1 is shown; its restored best checkpoint was "
         "evaluated once on the test split. " + timing_note(rows)).rstrip(),
        "",
        "| " + " | ".join(header for _, header in COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    lines += ["| " + " | ".join(_format(key, row[key]) for key, _ in COLUMNS) + " |" for row in rows]
    if not rows:
        lines.append("| " + " | ".join(["n/a"] * len(COLUMNS)) + " |")
    if untrained:
        lines += ["", "No full run yet: " + ", ".join(untrained) + "."]
    return "\n".join(lines) + "\n"


def update_readme(readme: Path, block: str) -> None:
    """Replace the text between the RESULTS markers in ``readme`` with ``block``."""
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(README_START), text.find(README_END)
    if start < 0 or end < start:
        raise ValueError(f"{readme} needs {README_START} followed by {README_END}")
    head = text[: start + len(README_START)]
    readme.write_text(f"{head}\n{block.rstrip()}\n{text[end:]}", encoding="utf-8")


def mean_std(values: list[float]) -> tuple[float, float | None]:
    """Mean and sample standard deviation (n - 1 in the denominator); the std is None for one value."""
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def check_final_runs(model: str, runs: list[tuple[Path, dict]], seeds: list[int]) -> None:
    """Raise ``ValueError`` unless ``runs`` hold exactly one run per seed in ``seeds``, each with a test
    split, all with the same parameter count."""
    found = [m.get("seed") for _, m in runs]
    if Counter(found) != Counter(seeds):
        raise ValueError(f"{model}: the final table needs exactly one run tagged final per seed {sorted(seeds)}, "
                         f"found seeds {found} in {[d.name for d, _ in runs]}")
    no_test = [d.name for d, m in runs if not m.get("test")]
    if no_test:
        raise ValueError(f"{model}: final runs need the test split, but {no_test} ran with --no-test")
    params = {m["params_trainable"] for _, m in runs}
    if len(params) > 1:
        raise ValueError(f"{model}: final runs have different parameter counts {sorted(params)}; "
                         f"all must use configs/{FINAL_CONFIG_DIR}/{model}.yaml")


def profile_latencies(path: Path) -> dict[str, float]:
    """Model -> CPU batch-1 mean latency (ms) from a ``profile_<label>.csv``. Only rows profiled from
    ``configs/final/`` count, so a profile of the interim configs is never mistaken for the final one."""
    if not Path(path).is_file():
        return {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        return {row["model"]: float(row["cpu_bs1_mean_ms"]) for row in csv.DictReader(f)
                if Path(row.get("config") or "").parent.name == FINAL_CONFIG_DIR and row.get("cpu_bs1_mean_ms")}


def checked_final_runs(results_root: Path = RESULTS_ROOT, configs_dir: Path = CONFIGS
                       ) -> tuple[dict[str, list[tuple[Path, dict]]], list[str]]:
    """Model -> its final runs sorted by seed, for every model that has any (in report order), and the
    models without any. Fails loudly (``ValueError``) when a model's final runs are not one per final seed."""
    seeds = load_yaml(Path(configs_dir) / "base.yaml").get("final_seeds") or []
    checked, missing = {}, []
    for model in model_configs(configs_dir):
        runs = final_runs(model, results_root)
        if not runs:
            missing.append(model)
            continue
        if not seeds:
            raise ValueError(f"{Path(configs_dir) / 'base.yaml'} has no final_seeds")
        check_final_runs(model, runs, seeds)
        checked[model] = sorted(runs, key=lambda run: run[1]["seed"])
    return checked, missing


def final_rows(results_root: Path = RESULTS_ROOT, configs_dir: Path = CONFIGS,
               profile_label: str = PROFILE_LABEL) -> tuple[list[dict], list[str]]:
    """Raw final values per model that has final runs (keys as in ``FINAL_CSV_COLUMNS``), and the models
    without any. Fails loudly (``ValueError``) when a model's final runs are not one per final seed.

    CPU latency comes from ``results/profile_<profile_label>.csv`` (one process, final configs)
    when it has the model, otherwise from the mean of the runs' own measurements.
    """
    latencies = profile_latencies(Path(results_root) / f"profile_{profile_label}.csv")
    checked, missing = checked_final_runs(results_root, configs_dir)
    rows = []
    for model, runs in checked.items():
        ms = [m for _, m in runs]
        row = {"model": model, "run_by": ", ".join(dict.fromkeys(m["run_by"] for m in ms)),
               "representation": ms[0]["representation"], "params": ms[0]["params_trainable"]}
        for name, value in FINAL_METRICS.items():
            row[f"{name}_mean"], row[f"{name}_std"] = mean_std([value(m) for m in ms])
        if model in latencies:
            row.update(cpu_latency_bs1_ms_mean=latencies[model], cpu_latency_bs1_ms_std=None,
                       cpu_latency_source=f"profile_{profile_label}.csv (single process)")
        else:
            own = [((m.get("latency") or {}).get("cpu_bs1") or {}).get("mean_ms") for m in ms]
            mean, std = mean_std(own) if None not in own else (None, None)
            row.update(cpu_latency_bs1_ms_mean=mean, cpu_latency_bs1_ms_std=std,
                       cpu_latency_source=RUNS_LATENCY_SOURCE)
        row.update(hardware="; ".join(dict.fromkeys(hardware_label(m.get("hardware", {})) for m in ms)),
                   seeds=" ".join(str(m["seed"]) for m in ms), run_ids=" ".join(m["run_id"] for m in ms))
        rows.append(row)
    return rows, missing


def _plus_minus(mean: float | None, std: float | None, spec: str, scale: float = 1.0, suffix: str = "") -> str:
    """``0.941 ± 0.004`` style; the mean alone when there is no std; ``n/a`` without a mean."""
    if mean is None:
        return "n/a"
    text = format(scale * mean, spec)
    if std is not None:
        text += f" ± {format(scale * std, spec)}"
    return text + suffix


def _format_final(key: str, row: dict) -> str:
    if key in ("val_macro_f1", "test_macro_f1"):
        return _plus_minus(row[f"{key}_mean"], row[f"{key}_std"], ".3f")
    if key == "test_acc":
        return _plus_minus(row["test_acc_mean"], row["test_acc_std"], ".1f", scale=100, suffix="%")
    if key == "train_time_s":
        return _plus_minus(row["train_time_s_mean"], row["train_time_s_std"], ".1f")
    if key == "cpu_latency_bs1_ms":
        text = _plus_minus(row["cpu_latency_bs1_ms_mean"], row["cpu_latency_bs1_ms_std"], ".3f")
        if row["cpu_latency_bs1_ms_mean"] is None:
            return text
        return text + (" (mean of runs)" if row["cpu_latency_source"] == RUNS_LATENCY_SOURCE else " (single process)")
    if key == "params":
        return f"{row['params']:,}"
    if key == "seeds":
        return ", ".join(row["seeds"].split())
    return str(row[key]).replace("|", "/")


def render_final_md(rows: list[dict], missing: list[str], seeds: list[int],
                    profile_label: str = PROFILE_LABEL) -> str:
    """Formatted final table: the one-line caption, the table, then where latency and timings come from."""
    lines = [
        "<!-- Generated by scripts/make_results_table.py from the run folders tagged final. Do not edit by hand. -->",
        f"Final results: mean ± sample std over seeds {', '.join(map(str, seeds))} of the validation-selected "
        "configuration; shared protocol.",
        "",
        "| " + " | ".join(header for _, header in FINAL_MD_COLUMNS) + " |",
        "|" + "---|" * len(FINAL_MD_COLUMNS),
    ]
    lines += ["| " + " | ".join(_format_final(key, row) for key, _ in FINAL_MD_COLUMNS) + " |" for row in rows]
    sources = {row["cpu_latency_source"] == RUNS_LATENCY_SOURCE for row in rows}
    notes = []
    if False in sources:
        notes.append(f"CPU latency (batch size 1, single thread) marked (single process) comes from one run of "
                     f"scripts/profile_all.py on {profile_label} (results/profile_{profile_label}.csv).")
    if True in sources:
        notes.append("CPU latency marked (mean of runs) is the mean of the final runs' own measurements, each "
                     f"taken in a separate process; python scripts/profile_all.py --label {profile_label} "
                     "replaces it with one single-process measurement.")
    hardware = {row["hardware"] for row in rows}
    notes.append(f"All final runs report the same hardware ({hardware.pop()})." if len(hardware) == 1 else
                 "The final runs report different hardware (see final_summary.csv), so training times are "
                 "not directly comparable across rows.")
    lines += ["", " ".join(notes)]
    if missing:
        lines += ["", "No final runs yet: " + ", ".join(missing) + "."]
    return "\n".join(lines) + "\n"


def write_results_table(results_root: Path = RESULTS_ROOT, configs_dir: Path = CONFIGS,
                        readme: Path | None = REPO_ROOT / "README.md") -> dict[str, Path]:
    """Write the interim ``summary.csv``/``summary.md`` and, once any model has final runs,
    ``final_summary.csv``/``final_summary.md`` under ``results_root``. The README block shows the
    final table when it exists, otherwise the interim one."""
    selected = selected_runs(results_root, configs_dir)
    rows = summary_rows(selected)
    untrained = [m for m in model_configs(configs_dir) if m not in selected]
    md = render_summary_md(rows, untrained)
    paths = {"csv": write_summary_csv(rows, Path(results_root) / "summary.csv"),
             "md": Path(results_root) / "summary.md"}
    paths["md"].write_text(md, encoding="utf-8")
    block = md

    final, missing = final_rows(results_root, configs_dir)
    if final:
        seeds = load_yaml(Path(configs_dir) / "base.yaml")["final_seeds"]
        block = render_final_md(final, missing, seeds)
        paths["final_csv"] = write_summary_csv(final, Path(results_root) / "final_summary.csv", FINAL_CSV_COLUMNS)
        paths["final_md"] = Path(results_root) / "final_summary.md"
        paths["final_md"].write_text(block, encoding="utf-8")
    if readme is not None:
        update_readme(Path(readme), block)
        paths["readme"] = Path(readme)
    return paths


def read_history(path: Path) -> list[dict]:
    """``history.csv`` rows with ``epoch`` as int and every other column as float."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        return [{k: int(v) if k == "epoch" else float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_selected_figures(results_root: Path = RESULTS_ROOT, configs_dir: Path = CONFIGS) -> list[Path]:
    """Overlay validation macro-F1 curves of the selected runs and copy their test confusion matrices.

    Writes ``results/figures/val_macro_f1_curves.png`` and
    ``results/figures/confusion_test_<model>.png`` (for runs that have a test split). Each
    model keeps the same colour as more runs land (colour follows the model, not its rank).
    """
    figures = Path(results_root) / "figures"
    all_models = list(model_configs(configs_dir))
    selected = selected_runs(results_root, configs_dir)
    written = []
    if selected:
        curves = {m: read_history(run_dir / "history.csv") for m, (run_dir, _, _) in selected.items()}
        slots = {m: all_models.index(m) for m in curves}
        written.append(plot_val_curves(curves, figures / "val_macro_f1_curves.png", slots))
    for model, (run_dir, _, _) in selected.items():
        source = run_dir / "confusion_test.png"
        if source.is_file():
            figures.mkdir(parents=True, exist_ok=True)
            written.append(Path(shutil.copyfile(source, figures / f"confusion_test_{model}.png")))
    return written


def summed_confusion(runs: list[tuple[Path, dict]], split: str = "test") -> list[list[int]]:
    """Cell-by-cell sum of the runs' ``split`` confusion matrices (rows = true class): ``(C, C)`` counts."""
    matrices = [m[split]["confusion"] for _, m in runs]
    return [[sum(cells) for cells in zip(*rows)] for rows in zip(*matrices)]


def write_final_figures(results_root: Path = RESULTS_ROOT, configs_dir: Path = CONFIGS,
                        profile_label: str = PROFILE_LABEL) -> list[Path]:
    """PHASE3_SPEC 4.3 figures from the runs tagged final, in ``results/figures/`` (none without final runs).

    - ``final_val_curves.png``: validation macro-F1 per epoch of each model's first-seed (42) final run.
    - ``confusion_test_<model>_final.png``: test confusion matrix summed over the seeds, row-normalised.
    - ``cost_tradeoff.png``: test macro-F1 (mean ± sample std over the seeds) against the single-process
      CPU latency at batch size 1 from ``results/profile_<profile_label>.csv`` (log x-axis), marker area
      proportional to parameters. Skipped with a warning while that profile lacks a model's final config.

    Colours follow ``write_selected_figures`` (one fixed palette slot per model). Fails like the final
    table when a model's final runs are not one per final seed.
    """
    checked, _ = checked_final_runs(results_root, configs_dir)
    if not checked:
        return []
    figures = Path(results_root) / "figures"
    all_models = list(model_configs(configs_dir))
    slots = {m: all_models.index(m) for m in checked}
    seeds = load_yaml(Path(configs_dir) / "base.yaml")["final_seeds"]

    curves = {m: read_history(next(d for d, metrics in runs if metrics["seed"] == seeds[0]) / "history.csv")
              for m, runs in checked.items()}
    written = [plot_val_curves(curves, figures / "final_val_curves.png", slots)]
    for model, runs in checked.items():
        written.append(plot_confusion(summed_confusion(runs), ACTIVITY_NAMES,
                                      figures / f"confusion_test_{model}_final.png",
                                      title=f"{model}: test set, seeds {', '.join(map(str, seeds))} summed"))

    latencies = profile_latencies(Path(results_root) / f"profile_{profile_label}.csv")
    unprofiled = [m for m in checked if m not in latencies]
    if unprofiled:
        warnings.warn(f"cost_tradeoff.png skipped: profile_{profile_label}.csv has no final-config row for "
                      f"{unprofiled}; run python scripts/profile_all.py --label {profile_label} first")
        return written
    points = []
    for model, runs in checked.items():
        f1_mean, f1_std = mean_std([m["test"]["macro_f1"] for _, m in runs])
        points.append({"name": model, "latency_ms": latencies[model], "f1_mean": f1_mean, "f1_std": f1_std,
                       "params": runs[0][1]["params_trainable"]})
    written.append(plot_cost_tradeoff(points, figures / "cost_tradeoff.png", slots))
    return written


def _config_label(path: Path) -> str:
    """Repository-relative POSIX path of a config when possible (the ``config`` column of profile CSVs)."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def profile_configs(label: str, configs_dir: Path = CONFIGS, out_dir: Path = RESULTS_ROOT) -> Path:
    """Profile every model config on this machine without training; write ``profile_<label>.csv``.

    Each model uses ``configs/final/<name>.yaml`` when it exists (Phase 3), otherwise
    ``configs/<name>.yaml``; the ``config`` column records which. Parameter count, size and
    latency do not depend on the weights, so each model is built from its config with seed 42
    and measured as is, all in this one process: CPU latency at the batch sizes, warmup,
    iterations and thread count in ``base.yaml`` ``eval``, plus GPU latency at batch size 1
    when a GPU is present. ``label`` names the machine neutrally (e.g. ``rtx4060-laptop``).
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError(f"label must use letters, digits, '-' or '_' only, got {label!r}")
    ev = load_yaml(Path(configs_dir) / "base.yaml")["eval"]
    timing = {"warmup": ev["latency_warmup"], "iters": ev["latency_iters"]}
    device = best_available_device()
    hw = hardware_info(device)
    rows = []
    for name, path in final_configs(configs_dir).items():
        cfg = load_config(path, base_cfg=Path(configs_dir) / "base.yaml")
        representation = cfg["data"]["representation"]
        shape = INPUT_SHAPES[representation]
        set_seed(cfg["seed"])
        model = build_from_target(cfg["model"]["target"], cfg["model"].get("params") or {})
        row = {"label": label, "model": name, "config": _config_label(path), "representation": representation,
               "params": count_params(model), "size_mb": model_size_mb(model)}
        runs = [(f"cpu_bs{bs}", "cpu", bs, ev["latency_threads"]) for bs in ev["latency_batch_sizes"]]
        if device != "cpu":
            runs.append(("device_bs1", device, 1, None))
        for key, dev, bs, threads in runs:
            lat = measure_latency(model, shape, dev, batch_size=bs, threads=threads, **timing)
            row.update({f"{key}_mean_ms": lat["mean_ms"], f"{key}_p50_ms": lat["p50_ms"],
                        f"{key}_p95_ms": lat["p95_ms"]})
        row.update({"cpu_threads": ev["latency_threads"], "cpu": hw["cpu"], "gpu": hw["gpu"],
                    "torch": hw["torch"], "os": hw["os"]})
        rows.append(row)

    out = Path(out_dir) / f"profile_{label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: ("" if v is None else v) for k, v in row.items()} for row in rows)
    return out
