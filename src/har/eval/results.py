"""Results table, selected-run figures and same-machine profiling (BUILD_SPEC M2.6 and 8.5).

Every number written here is read from ``results/<model>/<run_id>/metrics.json`` files produced
by ``scripts/train.py``. For each model the full run with the best **validation** macro-F1 is
selected; test scores never influence the choice.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

from har.data.constants import CHANNELS, N_FEATURES, WINDOW_LEN
from har.eval.plots import plot_val_curves
from har.eval.profiling import best_available_device, count_params, hardware_info, measure_latency, model_size_mb
from har.utils.build import build_from_target
from har.utils.config import load_config, load_yaml
from har.utils.paths import CONFIGS, REPO_ROOT, RESULTS_ROOT
from har.utils.seed import set_seed

NON_MODEL_CONFIGS = frozenset({"base.yaml", "team.yaml"})
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


def model_configs(configs_dir: Path = CONFIGS) -> dict[str, Path]:
    """Model name -> config path for every ``configs/*.yaml`` except base and team, in report order."""
    found = {}
    for path in sorted(Path(configs_dir).glob("*.yaml")):
        if path.name not in NON_MODEL_CONFIGS:
            found[load_yaml(path).get("name") or path.stem] = path
    rank = {name: i for i, name in enumerate(MODEL_ORDER)}
    return dict(sorted(found.items(), key=lambda item: (rank.get(item[0], len(rank)), item[0])))


def full_runs(model: str, results_root: Path = RESULTS_ROOT) -> list[tuple[Path, dict]]:
    """``(run_dir, metrics)`` for every full run of ``model``, oldest first.

    Smoke runs live under ``results/_smoke/`` and are never read; runs marked ``smoke`` or
    ``protocol_override`` are skipped as well.
    """
    runs = []
    for path in sorted((Path(results_root) / model).glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if not metrics.get("smoke") and not metrics.get("protocol_override"):
            runs.append((path.parent, metrics))
    return runs


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
            "cpu_latency_bs1_ms": (m["latency"].get("cpu_bs1") or {}).get("mean_ms"),
            "hardware": hardware_label(m.get("hardware", {})),
            "runs": n_runs,
            "run_id": m["run_id"],
        })
    return rows


def write_summary_csv(rows: list[dict], path: Path) -> Path:
    """Raw values, no formatting; None becomes an empty cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[key for key, _ in COLUMNS])
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


def render_summary_md(rows: list[dict], untrained: list[str]) -> str:
    """Formatted markdown table with a short provenance note (also written into the README)."""
    lines = [
        "<!-- Generated by scripts/make_results_table.py from results/<model>/<run_id>/metrics.json. "
        "Do not edit by hand. -->",
        "Preliminary interim results: untuned, single seed (42), shared protocol. For each model the "
        "full run with the best validation macro-F1 is shown; its restored best checkpoint was "
        "evaluated once on the test split. Training time and latency were measured on each member's "
        "own machine (see Hardware), so they are not directly comparable across rows.",
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


def write_results_table(results_root: Path = RESULTS_ROOT, configs_dir: Path = CONFIGS,
                        readme: Path | None = REPO_ROOT / "README.md") -> dict[str, Path]:
    """Write ``summary.csv`` and ``summary.md`` under ``results_root`` and refresh the README block."""
    selected = selected_runs(results_root, configs_dir)
    rows = summary_rows(selected)
    untrained = [m for m in model_configs(configs_dir) if m not in selected]
    md = render_summary_md(rows, untrained)
    paths = {"csv": write_summary_csv(rows, Path(results_root) / "summary.csv"),
             "md": Path(results_root) / "summary.md"}
    paths["md"].write_text(md, encoding="utf-8")
    if readme is not None:
        update_readme(Path(readme), md)
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


def profile_configs(label: str, configs_dir: Path = CONFIGS, out_dir: Path = RESULTS_ROOT) -> Path:
    """Profile every model config on this machine without training; write ``profile_<label>.csv``.

    Parameter count, size and latency do not depend on the weights, so each model is built
    from its config with seed 42 and measured as is: CPU latency at the batch sizes, warmup,
    iterations and thread count in ``base.yaml`` ``eval``, plus GPU latency at batch size 1
    when a GPU is present. ``label`` names the machine neutrally (e.g. ``mayank-laptop``).
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError(f"label must use letters, digits, '-' or '_' only, got {label!r}")
    ev = load_yaml(Path(configs_dir) / "base.yaml")["eval"]
    timing = {"warmup": ev["latency_warmup"], "iters": ev["latency_iters"]}
    device = best_available_device()
    hw = hardware_info(device)
    rows = []
    for name, path in model_configs(configs_dir).items():
        cfg = load_config(path, base_cfg=Path(configs_dir) / "base.yaml")
        representation = cfg["data"]["representation"]
        shape = INPUT_SHAPES[representation]
        set_seed(cfg["seed"])
        model = build_from_target(cfg["model"]["target"], cfg["model"].get("params") or {})
        row = {"label": label, "model": name, "representation": representation,
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
