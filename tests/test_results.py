"""Tests for har.eval.results and the results scripts (make_results_table, plot_curves, profile_all)."""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

import har.eval.results as results_module
from har.eval.results import (
    FINAL_CSV_COLUMNS,
    README_END,
    README_START,
    final_runs,
    full_runs,
    hardware_label,
    mean_std,
    model_configs,
    profile_configs,
    read_history,
    select_best,
    summed_confusion,
    timing_note,
    update_readme,
    write_final_figures,
    write_results_table,
    write_selected_figures,
)
from har.utils.paths import CONFIGS, REPO_ROOT, RESULTS_ROOT

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
BASE_YAML = """\
seed: 42
final_seeds: [42, 43, 44]
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


def _final(results: Path, model: str, val_f1s: tuple[float, ...] = (0.95, 0.96, 0.97),
           seeds: tuple[int, ...] = (42, 43, 44), **extra) -> list[Path]:
    """Final runs (tag ``final``), one per seed: test macro-F1 = val - 0.02, train time 10, 11, 12 s,
    CPU latency 0.5, 0.6, 0.7 ms."""
    return [_run(results, model, f"20261020-1{i}0000_s{seed}_final", val_f1=f1, test_f1=f1 - 0.02,
                 tag="final", seed=seed, train_time_s=10.0 + i,
                 latency={"cpu_bs1": {"mean_ms": 0.5 + 0.1 * i}, "device_bs1": None}, **extra)
            for i, (seed, f1) in enumerate(zip(seeds, val_f1s))]


def _csv_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return {row["model"]: row for row in csv.DictReader(f)}


def _cm(seed: int) -> list[list[int]]:
    """A 6x6 test confusion matrix per seed: diagonal seed - 41, cell [3][4] = seed - 42.
    Summed over seeds 42, 43, 44: diagonal 6, cell [3][4] = 3."""
    cm = [[(seed - 41) * (i == j) for j in range(6)] for i in range(6)]
    cm[3][4] = seed - 42
    return cm


def _with_confusions(run_dirs: list[Path]) -> list[Path]:
    for run_dir in run_dirs:
        path = run_dir / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metrics["test"]["confusion"] = _cm(metrics["seed"])
        path.write_text(json.dumps(metrics), encoding="utf-8")
    return run_dirs


def _profile(results: Path, latencies: dict[str, float], config_dir: str = "configs/final") -> Path:
    """A profile_rtx4060-laptop.csv with the columns profile_latencies reads."""
    rows = ["label,model,config,cpu_bs1_mean_ms"]
    rows += [f"rtx4060-laptop,{m},{config_dir}/{m}.yaml,{ms}" for m, ms in latencies.items()]
    path = results / "profile_rtx4060-laptop.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


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


def test_interim_table_ignores_tagged_runs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    interim = _run(results, "cnn1d", "20261004-100000_aaa", val_f1=0.90, test_f1=0.93)
    finals = _final(results, "cnn1d", val_f1s=(0.99, 0.99, 0.99))           # better, but tagged
    _run(results, "cnn1d", "20261020-200000_ddd_lr-check", val_f1=0.995, tag="lr check")
    assert [d for d, _ in full_runs("cnn1d", results)] == [interim]
    assert [d for d, _ in final_runs("cnn1d", results)] == finals


def test_committed_interim_summary_is_reproduced_from_the_committed_runs(tmp_path: Path) -> None:
    """results/summary.md and summary.csv stay exactly as submitted in Phase 2 (PHASE3_SPEC 4.1 item 6)."""
    copy = tmp_path / "results"
    for metrics in RESULTS_ROOT.glob("*/*/metrics.json"):
        if not metrics.parent.parent.name.startswith("_"):
            target = copy / metrics.parent.relative_to(RESULTS_ROOT)
            target.mkdir(parents=True)
            shutil.copy(metrics, target)
    paths = write_results_table(copy, CONFIGS, readme=None)
    assert paths["md"].read_text(encoding="utf-8") == (RESULTS_ROOT / "summary.md").read_text(encoding="utf-8")
    assert _csv_rows(paths["csv"]) == _csv_rows(RESULTS_ROOT / "summary.csv")


def test_mean_std_uses_the_sample_standard_deviation() -> None:
    mean, std = mean_std([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0) and std == pytest.approx(1.0)       # population std would be 0.816
    assert mean_std([5.0]) == (5.0, None)


def test_final_summary_files_and_readme_block(tmp_path: Path) -> None:
    configs, results, readme = _configs(tmp_path), tmp_path / "results", _readme(tmp_path)
    _run(results, "cnn1d", "20261004-100000_aaa", val_f1=0.90, test_f1=0.93)    # the interim run
    _final(results, "cnn1d")
    paths = write_results_table(results, configs, readme)
    assert set(paths) == {"csv", "md", "final_csv", "final_md", "readme"}

    rows = _csv_rows(paths["final_csv"])
    with paths["final_csv"].open(encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == FINAL_CSV_COLUMNS
    assert list(rows) == ["cnn1d"]
    row = rows["cnn1d"]
    assert float(row["val_macro_f1_mean"]) == pytest.approx(0.96)
    assert float(row["val_macro_f1_std"]) == pytest.approx(0.01)
    assert float(row["test_acc_mean"]) == pytest.approx(0.93) and float(row["test_acc_std"]) == pytest.approx(0.01)
    assert float(row["train_time_s_mean"]) == pytest.approx(11.0) and float(row["train_time_s_std"]) == pytest.approx(1.0)
    assert float(row["cpu_latency_bs1_ms_mean"]) == pytest.approx(0.6)
    assert row["cpu_latency_source"] == "mean of the final runs"
    assert row["seeds"] == "42 43 44" and row["params"] == "197702" and row["run_by"] == "Test Person"
    assert row["run_ids"].split() == [f"20261020-1{i}0000_s{s}_final" for i, s in enumerate((42, 43, 44))]

    md = paths["final_md"].read_text(encoding="utf-8")
    assert ("Final results: mean ± sample std over seeds 42, 43, 44 of the validation-selected configuration; "
            "shared protocol.") in md
    assert ("| cnn1d | Test Person | raw | 197,702 | 0.960 ± 0.010 | 93.0 ± 1.0% | 0.940 ± 0.010 | 11.0 ± 1.0 "
            "| 0.600 ± 0.100 (mean of runs) | 42, 43, 44 |") in md
    assert "No final runs yet: mlp, zeta." in md and "same hardware" in md and "—" not in md

    text = readme.read_text(encoding="utf-8")
    assert md.strip() in text and "Preliminary interim results" not in text
    interim = paths["md"].read_text(encoding="utf-8")                             # still written, unchanged
    assert "Preliminary interim results" in interim and "| 1 | 20261004-100000_aaa |" in interim


def test_no_final_runs_keeps_the_interim_readme_block(tmp_path: Path) -> None:
    configs, results, readme = _configs(tmp_path), tmp_path / "results", _readme(tmp_path)
    _run(results, "cnn1d", "20261004-100000_aaa", val_f1=0.90, test_f1=0.93)
    paths = write_results_table(results, configs, readme)
    assert set(paths) == {"csv", "md", "readme"} and not (results / "final_summary.csv").exists()
    assert "Preliminary interim results" in readme.read_text(encoding="utf-8")


@pytest.mark.parametrize("seeds", [(42, 43), (42, 43, 43), (42, 43, 45), (42, 43, 44, 44)])
def test_final_summary_fails_unless_one_run_per_final_seed(tmp_path: Path, seeds: tuple[int, ...]) -> None:
    configs, results = _configs(tmp_path), tmp_path / "results"
    _final(results, "cnn1d", val_f1s=(0.95,) * len(seeds), seeds=seeds)
    with pytest.raises(ValueError, match=r"cnn1d: .*one run tagged final per seed \[42, 43, 44\]"):
        write_results_table(results, configs, readme=None)


def test_final_runs_need_the_test_split_and_one_parameter_count(tmp_path: Path) -> None:
    configs, results = _configs(tmp_path), tmp_path / "results"
    _final(results, "cnn1d")
    _run(results, "mlp", "20261020-100000_s42_final", val_f1=0.95, tag="final", seed=42)    # --no-test
    _final(results, "mlp", val_f1s=(0.95, 0.95), seeds=(43, 44))
    with pytest.raises(ValueError, match="mlp: final runs need the test split"):
        write_results_table(results, configs, readme=None)

    other = tmp_path / "other"
    _final(other, "cnn1d")
    json_path = next((other / "cnn1d").glob("*_s44_final")) / "metrics.json"
    metrics = json.loads(json_path.read_text(encoding="utf-8")) | {"params_trainable": 1}
    json_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="different parameter counts"):
        write_results_table(other, configs, readme=None)


def test_final_latency_comes_from_a_single_process_profile_of_the_final_configs(tmp_path: Path) -> None:
    configs, results = _configs(tmp_path, names=("mlp", "cnn1d")), tmp_path / "results"
    (configs / "final").mkdir()
    (configs / "final" / "cnn1d.yaml").write_text((configs / "cnn1d.yaml").read_text(encoding="utf-8"),
                                                  encoding="utf-8")
    profile = profile_configs("rtx4060-laptop", configs, results)       # cnn1d from final/, mlp from configs/
    profiled = _csv_rows(profile)
    _final(results, "cnn1d")
    _final(results, "mlp")
    paths = write_results_table(results, configs, readme=None)

    rows = _csv_rows(paths["final_csv"])
    assert float(rows["cnn1d"]["cpu_latency_bs1_ms_mean"]) == float(profiled["cnn1d"]["cpu_bs1_mean_ms"])
    assert rows["cnn1d"]["cpu_latency_bs1_ms_std"] == ""
    assert rows["cnn1d"]["cpu_latency_source"] == "profile_rtx4060-laptop.csv (single process)"
    assert rows["mlp"]["cpu_latency_source"] == "mean of the final runs"          # its profile row is the interim config
    md = paths["final_md"].read_text(encoding="utf-8")
    assert f"| {float(profiled['cnn1d']['cpu_bs1_mean_ms']):.3f} (single process) |" in md
    assert "| 0.600 ± 0.100 (mean of runs) |" in md
    assert "(results/profile_rtx4060-laptop.csv)" in md


def test_profile_configs_prefers_the_final_configs(tmp_path: Path) -> None:
    configs = _configs(tmp_path, names=("mlp", "cnn1d"))
    (configs / "final").mkdir()
    (configs / "final" / "mlp.yaml").write_text(
        "name: mlp\nowner: M1\n" + LINEAR_ON_FEATURES.replace("out_features: 6}", "out_features: 6, bias: false}"),
        encoding="utf-8")
    rows = _csv_rows(profile_configs("test-machine", configs, tmp_path / "out"))
    assert rows["mlp"]["params"] == str(561 * 6) and Path(rows["mlp"]["config"]).parent.name == "final"
    assert rows["cnn1d"]["params"] == str(561 * 6 + 6) and Path(rows["cnn1d"]["config"]).name == "cnn1d.yaml"


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


def test_summed_confusion_adds_the_seeds_cell_by_cell() -> None:
    runs = [(Path(f"s{seed}"), {"test": {"confusion": _cm(seed)}}) for seed in (42, 43, 44)]
    expected = [[6 * (i == j) for j in range(6)] for i in range(6)]
    expected[3][4] = 3
    assert summed_confusion(runs) == expected


def test_final_figures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs, results = _configs(tmp_path), tmp_path / "results"
    _run(results, "cnn1d", "20261004-100000_aaa", val_f1=0.90, test_f1=0.93)    # interim run: not used
    cnn = _with_confusions(_final(results, "cnn1d"))                            # test F1 0.93, 0.94, 0.95
    _with_confusions(_final(results, "mlp", val_f1s=(0.90, 0.91, 0.92), params_trainable=212230))
    _profile(results, {"mlp": 0.05, "cnn1d": 0.8})
    calls: dict[str, list[tuple]] = {}
    for name in ("plot_val_curves", "plot_confusion", "plot_cost_tradeoff"):       # record, then draw for real
        def spy(*args, _name=name, _original=getattr(results_module, name), **kwargs):
            calls.setdefault(_name, []).append(args)
            return _original(*args, **kwargs)
        monkeypatch.setattr(results_module, name, spy)

    written = write_final_figures(results, configs)
    assert [p.name for p in written] == ["final_val_curves.png", "confusion_test_mlp_final.png",
                                         "confusion_test_cnn1d_final.png", "cost_tradeoff.png"]
    assert all(p.parent == results / "figures" and p.read_bytes()[:8] == PNG_MAGIC for p in written)

    (curves, _, slots), = calls["plot_val_curves"]
    assert curves["cnn1d"] == read_history(cnn[0] / "history.csv")                 # the seed-42 final run
    assert curves["mlp"][1]["val_macro_f1"] == 0.90 and slots == {"mlp": 0, "cnn1d": 1}
    expected = [[6 * (i == j) for j in range(6)] for i in range(6)]
    expected[3][4] = 3
    assert [args[0] for args in calls["plot_confusion"]] == [expected, expected]

    (points, _, _), = calls["plot_cost_tradeoff"]
    assert [(p["name"], p["latency_ms"], p["params"]) for p in points] == [("mlp", 0.05, 212230),
                                                                           ("cnn1d", 0.8, 197702)]
    assert points[1]["f1_mean"] == pytest.approx(0.94) and points[1]["f1_std"] == pytest.approx(0.01)
    assert points[0]["f1_mean"] == pytest.approx(0.89)


def test_cost_tradeoff_waits_for_a_single_process_profile_of_every_final_config(tmp_path: Path) -> None:
    configs, results = _configs(tmp_path), tmp_path / "results"
    assert write_final_figures(results, configs) == []                             # no final runs yet
    _with_confusions(_final(results, "cnn1d"))
    _with_confusions(_final(results, "mlp"))
    for latencies, config_dir in (({}, "configs/final"), ({"cnn1d": 0.8}, "configs/final"),
                                  ({"mlp": 0.05, "cnn1d": 0.8}, "configs")):     # empty, partial, interim configs
        _profile(results, latencies, config_dir)
        with pytest.warns(UserWarning, match="profile_all.py --label rtx4060-laptop"):
            written = write_final_figures(results, configs)
        assert "cost_tradeoff.png" not in {p.name for p in written} and len(written) == 3
    assert not (results / "figures" / "cost_tradeoff.png").exists()


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
