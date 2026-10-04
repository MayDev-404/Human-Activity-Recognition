"""Tests for har.eval.results and the results scripts (make_results_table, plot_curves, profile_all)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from har.eval.results import (
    README_END,
    README_START,
    full_runs,
    hardware_label,
    model_configs,
    profile_configs,
    read_history,
    select_best,
    timing_note,
    update_readme,
    write_results_table,
    write_selected_figures,
)
from har.utils.paths import CONFIGS, REPO_ROOT

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
BASE_YAML = """\
seed: 42
data: {augment: false}
train: {batch_size: 64, max_epochs: 50, optimizer: adam, lr: 0.001, weight_decay: 0.0, patience: 10,
        monitor: val_macro_f1, mode: max, grad_clip_norm: 1.0, device: auto}
eval: {latency_batch_sizes: [1, 4], latency_warmup: 1, latency_iters: 3, latency_threads: 1}
"""
LINEAR_ON_FEATURES = "data: {representation: features}\nmodel: {target: torch.nn.Linear, params: {in_features: 561, out_features: 6}}\n"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configs(tmp_path: Path, names: tuple[str, ...] = ("cnn1d", "mlp", "zeta")) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text(BASE_YAML, encoding="utf-8")
    (configs / "team.yaml").write_text("members: []\n", encoding="utf-8")
    for name in names:
        (configs / f"{name}.yaml").write_text(f"name: {name}\nowner: M2\n{LINEAR_ON_FEATURES}", encoding="utf-8")
    return configs


def _split(f1: float) -> dict:
    return {"accuracy": f1 - 0.01, "macro_f1": f1, "per_class": {}, "confusion": []}


def _run(results: Path, model: str, run_id: str, val_f1: float, test_f1: float | None = None,
         **extra) -> Path:
    """A minimal run folder: metrics.json, history.csv and (with a test split) confusion_test.png."""
    run_dir = results / model / run_id
    run_dir.mkdir(parents=True)
    metrics = {
        "model": model, "run_id": run_id, "run_by": "Test Person", "representation": "raw",
        "params_trainable": 197702, "best_epoch": 2, "train_time_s": 12.34,
        "val": _split(val_f1), "test": None if test_f1 is None else _split(test_f1),
        "latency": {"cpu_bs1": {"mean_ms": 0.4567}, "device_bs1": None},
        "hardware": {"cpu": "Test CPU", "device": "cuda", "gpu": "Test GPU"},
        "protocol_override": False, "smoke": False,
    } | extra
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    rows = ["epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1,lr,epoch_time_s"]
    rows += [f"{e},0.5,0.9,0.6,0.88,{val_f1 - 0.02 * (e != 2):.4f},0.001,0.4" for e in (1, 2, 3)]
    (run_dir / "history.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if test_f1 is not None:
        (run_dir / "confusion_test.png").write_bytes(PNG_MAGIC + b"fake")
    return run_dir


def _readme(tmp_path: Path) -> Path:
    path = tmp_path / "README.md"
    path.write_text(f"# Title\n\n## Results\n{README_START}\nold table\n{README_END}\n\n## After\nkeep me\n",
                    encoding="utf-8")
    return path


def test_model_configs_skip_base_and_team_and_follow_report_order(tmp_path: Path) -> None:
    configs = model_configs(_configs(tmp_path))
    assert list(configs) == ["mlp", "cnn1d", "zeta"]          # MODEL_ORDER first, others alphabetical
    assert configs["mlp"].name == "mlp.yaml"


def test_real_configs_are_all_models() -> None:
    names = list(model_configs(CONFIGS))
    assert "base" not in names and "team" not in names and "cnn1d" in names


def test_selection_uses_validation_never_test_and_skips_smoke_and_overrides(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _run(results, "cnn1d", "20261004-100000_aaa", val_f1=0.90, test_f1=0.99)     # best test, worse val
    chosen = _run(results, "cnn1d", "20261004-110000_bbb", val_f1=0.95, test_f1=0.80)
    _run(results, "cnn1d", "20261004-120000_ccc", val_f1=0.99, smoke=True)
    _run(results, "cnn1d", "20261004-130000_ddd", val_f1=0.98, protocol_override=True)
    _run(results / "_smoke", "cnn1d", "20261004-140000_eee", val_f1=0.999)        # never read
    runs = full_runs("cnn1d", results)
    assert [d.name for d, _ in runs] == ["20261004-100000_aaa", "20261004-110000_bbb"]
    assert select_best(runs)[0] == chosen


def test_ties_keep_the_oldest_run(tmp_path: Path) -> None:
    results = tmp_path / "results"
    first = _run(results, "cnn1d", "20261004-100000_aaa", val_f1=0.95)
    _run(results, "cnn1d", "20261004-110000_bbb", val_f1=0.95)
    assert select_best(full_runs("cnn1d", results))[0] == first


def test_summary_files_and_readme_block(tmp_path: Path) -> None:
    configs, results, readme = _configs(tmp_path), tmp_path / "results", _readme(tmp_path)
    _run(results, "cnn1d", "20261004-100000_aaa", val_f1=0.90, test_f1=0.93)
    _run(results, "cnn1d", "20261004-110000_bbb", val_f1=0.95, test_f1=0.80)
    _run(results, "mlp", "20261004-120000_ccc", val_f1=0.97)                       # a --no-test run
    paths = write_results_table(results, configs, readme)

    with paths["csv"].open(encoding="utf-8", newline="") as f:
        rows = {r["model"]: r for r in csv.DictReader(f)}
    assert list(rows) == ["mlp", "cnn1d"]                                         # zeta has no runs
    assert rows["cnn1d"]["val_macro_f1"] == "0.95" and rows["cnn1d"]["test_macro_f1"] == "0.8"
    assert rows["cnn1d"]["runs"] == "2" and rows["cnn1d"]["run_id"] == "20261004-110000_bbb"
    assert rows["cnn1d"]["cpu_latency_bs1_ms"] == "0.4567" and rows["cnn1d"]["params"] == "197702"
    assert rows["mlp"]["test_acc"] == "" and rows["mlp"]["test_macro_f1"] == ""

    md = paths["md"].read_text(encoding="utf-8")
    assert "| cnn1d | Test Person | raw | 197,702 | 94.00% | 0.950 | 79.00% | 0.800 | 2 | 12.3 | 0.457 |" in md
    assert "| mlp |" in md and "n/a" in md and "No full run yet: zeta." in md
    assert "—" not in md                                                      # no em dashes

    text = readme.read_text(encoding="utf-8")
    assert "old table" not in text and md.strip() in text
    assert text.startswith("# Title") and text.rstrip().endswith("keep me")
    write_results_table(results, configs, readme)                                  # idempotent
    assert readme.read_text(encoding="utf-8") == text


def test_update_readme_needs_both_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# no markers\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RESULTS"):
        update_readme(readme, "table")


def test_selected_figures(tmp_path: Path) -> None:
    configs, results = _configs(tmp_path), tmp_path / "results"
    _run(results, "cnn1d", "20261004-100000_aaa", val_f1=0.95, test_f1=0.93)
    _run(results, "mlp", "20261004-110000_bbb", val_f1=0.96)                       # no test split
    written = write_selected_figures(results, configs)
    figures = results / "figures"
    assert {p.name for p in written} == {"val_macro_f1_curves.png", "confusion_test_cnn1d.png"}
    assert (figures / "val_macro_f1_curves.png").read_bytes()[:8] == PNG_MAGIC
    assert not (figures / "confusion_test_mlp.png").exists()
    assert read_history(results / "cnn1d" / "20261004-100000_aaa" / "history.csv")[1]["val_macro_f1"] == 0.95


def test_profile_configs_writes_one_row_per_model(tmp_path: Path) -> None:
    configs = _configs(tmp_path, names=("mlp", "cnn1d"))
    out = profile_configs("test-machine", configs, tmp_path / "out")
    assert out == tmp_path / "out" / "profile_test-machine.csv"
    with out.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["model"] for r in rows] == ["mlp", "cnn1d"]
    assert all(r["params"] == str(561 * 6 + 6) and r["label"] == "test-machine" for r in rows)
    assert float(rows[0]["cpu_bs1_mean_ms"]) > 0 and float(rows[0]["cpu_bs4_mean_ms"]) > 0
    with pytest.raises(ValueError, match="label"):
        profile_configs("my laptop!", configs, tmp_path / "out")


def test_hardware_label() -> None:
    assert hardware_label({"cpu": "X", "device": "cuda", "gpu": "G"}) == "GPU G, CPU X"
    assert hardware_label({"cpu": "X", "device": "cpu", "gpu": "G"}) == "CPU X"


def test_timing_note_follows_the_hardware_of_the_rows(tmp_path: Path) -> None:
    configs, results = _configs(tmp_path), tmp_path / "results"
    _run(results, "cnn1d", "20261004-100000_aaa", val_f1=0.95, test_f1=0.93)
    _run(results, "mlp", "20261004-110000_bbb", val_f1=0.96, test_f1=0.94)
    md = write_results_table(results, configs, readme=None)["md"].read_text(encoding="utf-8")
    assert "All runs shown report the same hardware (GPU Test GPU, CPU Test CPU)" in md
    assert "different machines" not in md

    _run(results, "zeta", "20261004-120000_ccc", val_f1=0.97, test_f1=0.95,
         hardware={"cpu": "Other CPU", "device": "cpu", "gpu": None})
    md = write_results_table(results, configs, readme=None)["md"].read_text(encoding="utf-8")
    assert "measured on different machines" in md and "same hardware" not in md
    assert timing_note([]) == ""


def test_scripts_call_the_package_functions(tmp_path: Path) -> None:
    configs, results = _configs(tmp_path, names=("mlp",)), tmp_path / "results"
    _run(results, "mlp", "20261004-100000_aaa", val_f1=0.95, test_f1=0.93)
    paths = _script("make_results_table").main(["--results-root", str(results), "--configs-dir", str(configs),
                                                "--no-readme"])
    assert set(paths) == {"csv", "md"}
    figures = _script("plot_curves").main(["--results-root", str(results), "--configs-dir", str(configs)])
    assert len(figures) == 2
    profile = _script("profile_all").main(["--label", "ci", "--configs-dir", str(configs),
                                           "--output-dir", str(tmp_path)])
    assert profile.name == "profile_ci.csv"
