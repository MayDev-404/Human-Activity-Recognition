"""One end-to-end training run: config -> data -> fit -> evaluate -> profile -> run folder.

Implements BUILD_SPEC M2.4 and the run folder of section 8.2. ``scripts/train.py`` only
parses the command line and calls ``run_experiment``.
"""

from __future__ import annotations

import csv
import importlib
import json
import re
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from types import ModuleType

import torch
import yaml

from har.data.bundle import DataBundle, make_synthetic_bundle
from har.eval.metrics import classification_report_text, compute_metrics
from har.eval.plots import plot_confusion, plot_history
from har.eval.profiling import count_params, hardware_info, measure_latency, model_size_mb
from har.train.trainer import HISTORY_FIELDS, TrainConfig, fit, predict, resolve_device
from har.utils.build import build_from_target
from har.utils.config import load_config
from har.utils.paths import REPO_ROOT, RESULTS_ROOT
from har.utils.seed import set_seed

STATUS = "preliminary, untuned, single seed"
SMOKE_STATUS = "smoke test, not a result"
SMOKE_DIR = "_smoke"
SMOKE_SUBSET = 512        # training windows kept by --smoke
SMOKE_EPOCHS = 2
SMOKE_EVAL = {"latency_batch_sizes": [1], "latency_warmup": 2, "latency_iters": 5}
UNKNOWN_USER = "unknown (git user.name not set)"

# Modules that provide each real representation, and the task that delivers them.
_LOADERS = {
    "raw": ("har.data.raw", "build_raw_loaders", "M3.3 raw-signal loader"),
    "features": ("har.data.features", "build_feature_loaders", "M1.1 engineered-feature pipeline"),
}


def _git(args: list[str], cwd: Path = REPO_ROOT) -> str | None:
    """Output of ``git <args>`` (stripped), or None if git is missing or the command fails."""
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state(repo_root: Path = REPO_ROOT) -> tuple[str, bool]:
    """``(short sha, dirty)``. Dirty means uncommitted changes outside ``results/``.

    Result folders from earlier runs are ignored so they do not mark the code as dirty.
    Without git the sha is ``"nogit"`` and the tree counts as dirty (it cannot be checked).
    """
    sha = _git(["rev-parse", "--short", "HEAD"], repo_root)
    status = _git(["status", "--porcelain", "--", ".", ":(exclude)results"], repo_root)
    if sha is None or status is None:
        warnings.warn("could not read git state; recording git_sha 'nogit' and git_dirty true")
        return "nogit", True
    return sha, bool(status)


def run_by(repo_root: Path = REPO_ROOT) -> str:
    """``git config user.name``, or a placeholder (with a warning) if git or the name is missing."""
    name = _git(["config", "user.name"], repo_root)
    if not name:
        warnings.warn(f"git user.name is not set; recording run_by as {UNKNOWN_USER!r}")
        return UNKNOWN_USER
    return name


def make_run_id(git_sha: str, tag: str | None = None, now: datetime | None = None) -> str:
    """``YYYYmmdd-HHMMSS_<sha>`` plus ``_<tag>`` (characters outside ``[A-Za-z0-9._-]`` become ``-``)."""
    run_id = f"{(now or datetime.now()):%Y%m%d-%H%M%S}_{git_sha}"
    if tag:
        run_id += "_" + re.sub(r"[^A-Za-z0-9._-]+", "-", tag.strip())
    return run_id


def _import_loader(module_name: str, task: str) -> ModuleType:
    """Import a data module lazily so a not-yet-merged one gives a clear message."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        if err.name != module_name:
            raise
        raise ModuleNotFoundError(
            f"{module_name} ({task}) is not in this checkout yet. Run `git pull` on main once it "
            "is merged, then rerun.", name=module_name,
        ) from err


def build_bundle(cfg: dict, subset: int | None = None) -> DataBundle:
    """DataBundle for ``cfg['data']['representation']``: ``raw``, ``features`` or ``synthetic`` (tests)."""
    data, batch_size, seed = cfg["data"], cfg["train"]["batch_size"], cfg["seed"]
    representation = data.get("representation")
    augment = bool(data.get("augment", False))
    if representation == "synthetic":
        return make_synthetic_bundle(tuple(data["input_shape"]), batch_size=batch_size, seed=seed)
    if representation not in _LOADERS:
        raise ValueError(f"unknown data.representation {representation!r}; expected raw, features or synthetic")
    if augment and representation != "raw":
        raise ValueError("data.augment applies to raw windows only")
    module_name, function, task = _LOADERS[representation]
    build = getattr(_import_loader(module_name, task), function)
    kwargs = {"augment": augment} if representation == "raw" else {}
    return build(batch_size=batch_size, seed=seed, subset=subset, **kwargs)


def _new_run_dir(parent: Path, run_id: str) -> tuple[Path, str]:
    """Create ``parent/run_id`` (adding ``-2``, ``-3``... if it already exists); return it and its id."""
    candidate, k = run_id, 1
    while (parent / candidate).exists():
        k += 1
        candidate = f"{run_id}-{k}"
    (parent / candidate).mkdir(parents=True)
    return parent / candidate, candidate


def _evaluate(model: torch.nn.Module, bundle: DataBundle, split: str, device: torch.device,
              run_dir: Path, title: str) -> dict:
    """Metrics for one split, plus its confusion-matrix PNG and text classification report."""
    y_true, y_pred, _ = predict(model, getattr(bundle, split), device)
    metrics = compute_metrics(y_true, y_pred, bundle.class_names)
    plot_confusion(metrics["confusion"], bundle.class_names, run_dir / f"confusion_{split}.png",
                   title=f"{title}: {'validation' if split == 'val' else 'test'} set")
    report = classification_report_text(y_true, y_pred, bundle.class_names)
    (run_dir / f"classification_report_{split}.txt").write_text(report, encoding="utf-8")
    return metrics


def _profile(model: torch.nn.Module, input_shape: tuple[int, ...], device: torch.device, ev: dict) -> dict:
    """CPU latency at each configured batch size, plus batch-1 latency on the GPU if training used one."""
    timing = {"warmup": ev["latency_warmup"], "iters": ev["latency_iters"]}
    latency = {f"cpu_bs{bs}": measure_latency(model, input_shape, "cpu", batch_size=bs,
                                              threads=ev["latency_threads"], **timing)
               for bs in ev["latency_batch_sizes"]}
    latency["device_bs1"] = (measure_latency(model, input_shape, device, batch_size=1, threads=None, **timing)
                             if device.type != "cpu" else None)
    return latency


def _summary_line(metrics: dict, run_dir: Path) -> str:
    parts = [f"{metrics['model']} {metrics['run_id']}",
             f"val acc {metrics['val']['accuracy']:.4f} macro-F1 {metrics['val']['macro_f1']:.4f}"]
    if metrics["test"] is not None:
        parts.append(f"test acc {metrics['test']['accuracy']:.4f} macro-F1 {metrics['test']['macro_f1']:.4f}")
    parts.append(f"best epoch {metrics['best_epoch']}/{metrics['epochs_run']}")
    parts.append(f"train {metrics['train_time_s']:.1f}s")
    if metrics["latency"].get("cpu_bs1"):
        parts.append(f"cpu bs1 {metrics['latency']['cpu_bs1']['mean_ms']:.3f} ms")
    return " | ".join(parts) + f" -> {run_dir}"


def run_experiment(
    config: Path | str,
    *,
    smoke: bool = False,
    no_test: bool = False,
    allow_protocol_override: bool = False,
    tag: str | None = None,
    output_root: Path | str = RESULTS_ROOT,
    verbose: bool = True,
) -> Path:
    """Train one model under the shared protocol and write its run folder; return the folder.

    ``smoke`` keeps 512 training windows, trains 2 epochs, measures latency at batch size 1
    with 2 warmup and 5 timed passes, and writes under ``<output_root>/_smoke/``. ``no_test``
    skips the test split (``"test": null``). The folder is
    ``<output_root>/<model name>/<run_id>/`` with the files listed in BUILD_SPEC 8.2.
    """
    config = Path(config)
    cfg = load_config(config, allow_protocol_override=allow_protocol_override)
    name = cfg.get("name") or config.stem
    subset = None
    if smoke:
        cfg["train"]["max_epochs"] = SMOKE_EPOCHS
        cfg["eval"].update(SMOKE_EVAL)
        subset = SMOKE_SUBSET
    try:
        config_file = config.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        config_file = str(config)
    cfg["run"] = {"config_file": config_file, "smoke": smoke, "no_test": no_test, "subset": subset, "tag": tag}

    git_sha, git_dirty = git_state()
    if git_dirty and not smoke:
        warnings.warn("uncommitted changes outside results/: commit the code first so git_sha identifies it")
    who = run_by()

    set_seed(cfg["seed"])
    bundle = build_bundle(cfg, subset)
    model = build_from_target(cfg["model"]["target"], cfg["model"].get("params") or {})
    train_cfg = TrainConfig.from_dict(cfg["train"])
    result = fit(model, bundle, train_cfg, verbose=verbose)
    device = resolve_device(train_cfg.device)

    parent = Path(output_root) / SMOKE_DIR / name if smoke else Path(output_root) / name
    run_dir, run_id = _new_run_dir(parent, make_run_id(git_sha, tag))

    val = _evaluate(model, bundle, "val", device, run_dir, name)
    test = None if no_test else _evaluate(model, bundle, "test", device, run_dir, name)
    latency = _profile(model, bundle.input_shape, device, cfg["eval"])

    metrics = {
        "model": name,
        "run_id": run_id,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "run_by": who,
        "status": SMOKE_STATUS if smoke else STATUS,
        "representation": cfg["data"]["representation"],
        "params_trainable": count_params(model),
        "model_size_mb": model_size_mb(model),
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "train_time_s": result.train_time_s,
        "mean_epoch_time_s": result.mean_epoch_time_s,
        "val": val,
        "test": test,
        "latency": latency,
        "hardware": hardware_info(device),
        "val_subjects": list(bundle.meta.get("val_subjects", [])),
        "protocol_override": cfg["protocol_override"],
        "smoke": smoke,
        "tag": tag,
        "counts": bundle.meta.get("counts"),
    }

    with (run_dir / "config.resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(HISTORY_FIELDS))
        writer.writeheader()
        writer.writerows(result.history)
    plot_history(result.history, run_dir / "curves.png", title=name)
    torch.save(result.best_state, run_dir / "best.pt")      # gitignored (*.pt)

    print(_summary_line(metrics, run_dir))
    return run_dir
