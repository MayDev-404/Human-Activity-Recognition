"""Tests for har.eval.errors and scripts/error_analysis.py (PHASE3_SPEC section 5)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from har.data.constants import ACTIVITY_NAMES
from har.eval.errors import (
    analyse,
    chance_jaccard,
    confusion_counts,
    error_counts,
    load_final_predictions,
    write_error_analysis,
)
from har.eval.metrics import compute_metrics
from har.eval.results import checked_final_runs
from har.utils.paths import CONFIGS, REPO_ROOT, RESULTS_ROOT

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
WALK, UP, DOWN, SIT, STAND, LAY = range(6)
SEEDS = (42, 43, 44)
BASE_YAML = """\
seed: 42
final_seeds: [42, 43, 44]
data: {augment: false}
train: {batch_size: 64, max_epochs: 50, optimizer: adam, lr: 0.001, weight_decay: 0.0, patience: 10,
        monitor: val_macro_f1, mode: max, grad_clip_norm: 1.0, device: auto}
eval: {latency_batch_sizes: [1], latency_warmup: 1, latency_iters: 3, latency_threads: 1}
"""
# 60 test windows, 10 per class in class order: WALKING 0-9, ..., STANDING 40-49, LAYING 50-59.
Y_TRUE = np.repeat(np.arange(6), 10)
# Misclassified windows per model and seed (window -> predicted class). Wrong sets:
#   mlp   {40, 41} every seed
#   cnn1d {40, 41, 42}, plus 43 in seed 43
#   gru   {0, 40, 50} every seed (WALKING as DOWNSTAIRS, STANDING as SITTING, LAYING as UPSTAIRS)
ERRORS = {
    "mlp": {s: {40: SIT, 41: SIT} for s in SEEDS},
    "cnn1d": {42: {40: SIT, 41: SIT, 42: SIT}, 43: {40: SIT, 41: SIT, 42: SIT, 43: SIT},
              44: {40: SIT, 41: SIT, 42: SIT}},
    "gru": {s: {0: DOWN, 40: SIT, 50: UP} for s in SEEDS},
}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configs(tmp_path: Path, names=tuple(ERRORS)) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text(BASE_YAML, encoding="utf-8")
    (configs / "team.yaml").write_text("members: []\n", encoding="utf-8")
    for name in names:
        (configs / f"{name}.yaml").write_text(
            f"name: {name}\nowner: M3\ndata: {{representation: raw}}\n"
            "model: {target: torch.nn.Linear, params: {in_features: 4, out_features: 6}}\n", encoding="utf-8")
    return configs


def _final_run(results: Path, model: str, seed: int, y_pred: np.ndarray, y_true: np.ndarray = Y_TRUE) -> Path:
    """A final run folder with metrics.json (test metrics from compute_metrics) and predictions_test.csv."""
    run_dir = results / model / f"20261028-1{seed}000_abc1234_final"
    run_dir.mkdir(parents=True)
    test = compute_metrics(y_true, y_pred, ACTIVITY_NAMES)
    metrics = {"model": model, "run_id": run_dir.name, "run_by": "Test Person", "representation": "raw",
               "params_trainable": 1000, "seed": seed, "tag": "final", "smoke": False, "protocol_override": False,
               "val": test, "test": test}
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    rows = ["index,y_true,y_pred"] + [f"{i},{t},{p}" for i, (t, p) in enumerate(zip(y_true, y_pred))]
    (run_dir / "predictions_test.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return run_dir


def _results(tmp_path: Path, errors: dict = ERRORS) -> Path:
    results = tmp_path / "results"
    for model, by_seed in errors.items():
        for seed, wrong in by_seed.items():
            y_pred = Y_TRUE.copy()
            for window, pred in wrong.items():
                y_pred[window] = pred
            _final_run(results, model, seed, y_pred)
    return results


@pytest.fixture
def summary(tmp_path: Path) -> dict:
    return analyse(load_final_predictions(_results(tmp_path), _configs(tmp_path)))


def test_confusion_counts_rows_are_true_classes() -> None:
    cm = confusion_counts(np.array([0, 0, 3]), np.array([0, 4, 3]))
    assert cm.shape == (6, 6) and cm[0, 0] == 1 and cm[0, 4] == 1 and cm[3, 3] == 1 and cm.sum() == 3


def test_error_counts_on_a_hand_made_matrix() -> None:
    cm = np.diag([10] * 6)
    cm[SIT, STAND], cm[STAND, SIT] = 4, 3          # 7 sitting/standing
    cm[UP, DOWN] = 2                               # 2 among walking
    cm[LAY, UP], cm[WALK, SIT] = 1, 5              # 6 dynamic/static, 1 laying error
    assert error_counts(cm) == {"total_errors": 15, "sitting_standing": 7, "walking": 2, "dynamic_static": 6,
                                "laying": 1}


def test_chance_jaccard() -> None:
    assert chance_jaccard(0.1, 0.2) == pytest.approx(0.02 / 0.28)
    assert chance_jaccard(0.0, 0.0) == 1.0


def test_per_class_f1_is_the_mean_and_std_over_seeds(summary: dict) -> None:
    mlp = summary["per_class_f1"]["mlp"]
    assert mlp["STANDING"] == {"mean": pytest.approx(2 * 0.8 / 1.8), "std": 0.0}     # recall 8/10, precision 1
    assert mlp["SITTING"]["mean"] == pytest.approx(20 / 22)                           # precision 10/12, recall 1
    assert mlp["WALKING"]["mean"] == 1.0
    assert summary["per_class_f1"]["cnn1d"]["STANDING"]["std"] > 0                    # seed 43 differs


def test_error_structure(summary: dict) -> None:
    gru, cnn = summary["error_structure"]["gru"], summary["error_structure"]["cnn1d"]
    assert {k: gru[k]["mean"] for k in ("total_errors", "sitting_standing", "walking", "dynamic_static", "laying")} \
        == {"total_errors": 3, "sitting_standing": 1, "walking": 1, "dynamic_static": 1, "laying": 1}
    assert gru["sitting_standing_share"] == pytest.approx(1 / 3)
    assert gru["laying_main_target"] == "WALKING_UPSTAIRS" and gru["laying_main_target_share"] == 1.0
    assert cnn["total_errors"] == {"mean": pytest.approx(10 / 3), "std": pytest.approx(np.std([3, 4, 3], ddof=1))}
    assert cnn["sitting_standing_share"] == 1.0
    assert summary["error_structure"]["mlp"]["laying_main_target"] is None


def test_error_overlap(summary: dict) -> None:
    ov = summary["overlap"]
    assert ov["jaccard"]["mlp"]["cnn1d"] == pytest.approx((2 / 3 + 2 / 4 + 2 / 3) / 3)
    assert ov["jaccard"]["cnn1d"]["mlp"] == ov["jaccard"]["mlp"]["cnn1d"] and ov["jaccard"]["gru"]["gru"] == 1.0
    assert ov["jaccard"]["mlp"]["gru"] == pytest.approx(1 / 4)
    assert ov["jaccard"]["cnn1d"]["gru"] == pytest.approx((1 / 5 + 1 / 6 + 1 / 5) / 3)
    assert [p["models"] for p in ov["pairs"]] == [["mlp", "cnn1d"], ["mlp", "gru"], ["cnn1d", "gru"]]
    assert ov["pairs"][1]["jaccard_chance"] == pytest.approx(chance_jaccard(2 / 60, 3 / 60))
    assert ov["jaccard_mean"] == pytest.approx(np.mean([p["jaccard"] for p in ov["pairs"]]))
    # Window 40 is wrong for every model; 42 (+43 in seed 43) only for cnn1d; 0 and 50 only for gru.
    assert ov["all_wrong"] == {"mean": 1.0, "std": 0.0}
    assert ov["exactly_one"]["mean"] == pytest.approx(10 / 3)
    assert ov["any_wrong"]["mean"] == pytest.approx(16 / 3) and ov["none"]["mean"] == pytest.approx(60 - 16 / 3)
    assert {m: s["mean"] for m, s in ov["exactly_one_by_model"].items()} == \
        {"mlp": 0.0, "cnn1d": pytest.approx(4 / 3), "gru": 2.0}
    assert ov["all_wrong_by_class"]["STANDING"]["mean"] == 1.0 and ov["all_wrong_by_class"]["LAYING"]["mean"] == 0.0
    assert ov["all_wrong_sitting_standing_share"] == 1.0


def test_confusion_cells_shared_and_model_specific(summary: dict) -> None:
    cells = summary["confusion_cells"]
    assert [e["cell"] for e in cells["shared"]] == ["STANDING->SITTING"]
    shared = cells["shared"][0]
    assert shared["models"] == ["mlp", "cnn1d", "gru"]
    assert shared["share"] == {"mlp": pytest.approx(6 / 30), "cnn1d": pytest.approx(10 / 30), "gru": pytest.approx(3 / 30)}
    assert sorted(e["cell"] for e in cells["model_specific"]) == ["LAYING->WALKING_UPSTAIRS",
                                                                  "WALKING->WALKING_DOWNSTAIRS"]
    assert all(e["models"] == ["gru"] for e in cells["model_specific"])
    assert (cells["n_shared"], cells["n_model_specific"], cells["n_partly_shared"]) == (1, 2, 0)


def test_threshold_decides_which_cells_count(tmp_path: Path) -> None:
    data = load_final_predictions(_results(tmp_path), _configs(tmp_path))
    cells = analyse(data, threshold=0.25)["confusion_cells"]            # only cnn1d's 10/30 reaches 25%
    assert [e["cell"] for e in cells["model_specific"]] == ["STANDING->SITTING"] and cells["n_shared"] == 0


def test_partly_shared_cells(tmp_path: Path) -> None:
    errors = {m: {s: {40: SIT} for s in SEEDS} for m in ("mlp", "cnn1d", "bilstm", "gru")}
    errors["mlp"] = {s: {} for s in SEEDS}
    cells = analyse(load_final_predictions(_results(tmp_path, errors), _configs(tmp_path, tuple(errors))))
    assert [e["cell"] for e in cells["confusion_cells"]["partly_shared"]] == ["STANDING->SITTING"]


def test_fails_when_predictions_do_not_match_the_confusion_matrix(tmp_path: Path) -> None:
    results, configs = _results(tmp_path), _configs(tmp_path)
    run_dir = next((results / "gru").iterdir())
    path = run_dir / "predictions_test.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("\n1,0,0\n", "\n1,0,1\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="does not reproduce"):
        load_final_predictions(results, configs)


def test_fails_when_runs_disagree_on_the_test_labels(tmp_path: Path) -> None:
    results, configs = _results(tmp_path), _configs(tmp_path)
    shuffled = Y_TRUE[::-1].copy()
    run_dir = next((results / "gru").iterdir())
    for name in ("metrics.json", "predictions_test.csv"):
        (run_dir / name).unlink()
    run_dir.rmdir()
    _final_run(results, "gru", 42, shuffled, y_true=shuffled)
    with pytest.raises(ValueError, match="test labels differ"):
        load_final_predictions(results, configs)


def test_fails_unless_every_model_has_its_final_runs(tmp_path: Path) -> None:
    results, configs = _results(tmp_path), _configs(tmp_path, (*ERRORS, "transformer"))
    with pytest.raises(ValueError, match="transformer"):
        load_final_predictions(results, configs)


def test_index_column_must_be_in_order(tmp_path: Path) -> None:
    results, configs = _results(tmp_path), _configs(tmp_path)
    path = next((results / "mlp").iterdir()) / "predictions_test.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="index column"):
        load_final_predictions(results, configs)


def test_write_error_analysis_files(tmp_path: Path) -> None:
    results, configs = _results(tmp_path), _configs(tmp_path)
    paths = write_error_analysis(results, configs)
    out = results / "analysis"
    assert sorted(p.name for p in paths) == sorted([
        "per_class_f1.csv", "per_class_f1.md", "per_class_f1.png", "error_structure.csv", "error_structure.md",
        "error_overlap.csv", "error_overlap.png", "error_summary.json", "error_summary.md"])
    assert all(p.parent == out for p in paths)
    for name in ("per_class_f1.png", "error_overlap.png"):
        assert (out / name).read_bytes().startswith(PNG_MAGIC)
    summary = json.loads((out / "error_summary.json").read_text(encoding="utf-8"))
    assert summary["models"] == ["mlp", "cnn1d", "gru"] and summary["seeds"] == list(SEEDS)
    assert summary["overlap"]["pairs"][0]["jaccard"] == pytest.approx((2 / 3 + 2 / 4 + 2 / 3) / 3)
    with (out / "per_class_f1.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["class"] for r in rows] == ACTIVITY_NAMES
    assert list(rows[0]) == ["class"] + [f"{m}_{s}" for m in ("mlp", "cnn1d", "gru") for s in ("mean", "std")]
    md = (out / "error_summary.md").read_text(encoding="utf-8")
    assert "| mlp / cnn1d |" in md and "Standing → Sitting: mlp 20.0%, cnn1d 33.3%, gru 10.0%" in md
    for name in ("per_class_f1.md", "error_structure.md", "error_summary.md"):
        assert "—" not in (out / name).read_text(encoding="utf-8")       # no em dashes in report inputs


def test_script_writes_the_analysis(tmp_path: Path) -> None:
    results, configs = _results(tmp_path), _configs(tmp_path)
    out = tmp_path / "out"
    paths = _script("error_analysis").main(["--results-root", str(results), "--configs-dir", str(configs),
                                            "--out-dir", str(out)])
    assert len(paths) == 9 and (out / "error_summary.json").is_file()


def _real_final_runs_exist() -> bool:
    try:
        _, missing = checked_final_runs(RESULTS_ROOT, CONFIGS)
    except ValueError:
        return False
    return not missing


@pytest.mark.skipif(not _real_final_runs_exist(), reason="final runs for every model are not committed yet")
def test_committed_final_runs_can_be_compared_window_by_window() -> None:
    data = load_final_predictions(RESULTS_ROOT, CONFIGS)
    assert len(data.y_true) == 2947 and data.seeds == [42, 43, 44]


@pytest.mark.skipif(not (RESULTS_ROOT / "analysis" / "error_summary.json").is_file(),
                    reason="results/analysis is not committed yet")
def test_committed_analysis_is_reproduced_from_the_committed_runs(tmp_path: Path) -> None:
    write_error_analysis(RESULTS_ROOT, CONFIGS, tmp_path)
    for name in ("error_summary.json", "error_summary.md", "per_class_f1.csv", "per_class_f1.md",
                 "error_structure.csv", "error_structure.md", "error_overlap.csv"):
        assert (tmp_path / name).read_text(encoding="utf-8") == \
            (RESULTS_ROOT / "analysis" / name).read_text(encoding="utf-8"), name
