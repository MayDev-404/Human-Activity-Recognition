"""Tests for har.train.tuning, scripts/tune.py and the grid files in configs/tuning/ (PHASE3_SPEC D2, section 3)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from har.eval.profiling import count_params
from har.train.tuning import RESULT_FIELDS, check_grid, expand_grid, load_grid, select_config
from har.utils.build import build_from_target
from har.utils.config import load_config, load_yaml
from har.utils.paths import CONFIGS, REPO_ROOT

# PHASE3_SPEC section 3: each model's grid (parameter A, then B) and its parameter counts.
SPEC_GRIDS = {
    "mlp": ({"dropout": [0.3, 0.5], "hidden": [[256, 128], [256, 256]]}, {178_310, 212_230}),
    "cnn1d": ({"dropout": [0.5, 0.3], "kernel_sizes": [[5, 5, 5, 5, 3], [7, 7, 5, 5, 3]]}, {197_702, 207_046}),
    "bilstm": ({"dropout": [0.3, 0.5], "pooling": ["last", "mean"]}, {138_502}),
    "gru": ({"dropout": [0.3, 0.5], "pooling": ["last", "mean"]}, {104_070}),
    "transformer": ({"dropout": [0.1, 0.2], "dim_feedforward": [256, 128]}, {159_302, 109_766}),
}

TINY_BASE = """\
name: tiny
owner: M2
data: {representation: synthetic, input_shape: [16]}
model:
  target: har.models.mlp.MLP
  params: {input_dim: 16, hidden: [8], dropout: 0.3, n_classes: 6}
"""


def _load_script():
    spec = importlib.util.spec_from_file_location("tune_script", REPO_ROOT / "scripts" / "tune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tune_script = _load_script()


def _row(config_id: str, val_macro_f1: float, params: int) -> dict:
    return {"config_id": config_id, "val_macro_f1": val_macro_f1, "params": params}


@pytest.mark.parametrize("model", list(SPEC_GRIDS))
def test_grid_files_match_the_spec(model: str) -> None:
    grid, counts = SPEC_GRIDS[model]
    spec = load_grid(CONFIGS / "tuning" / f"{model}.yaml")
    assert spec["model"] == model and spec["base_config"] == f"configs/{model}.yaml"
    assert spec["grid"] == grid and list(spec["grid"]) == list(grid)
    base = load_yaml(REPO_ROOT / spec["base_config"])["model"]
    check_grid(spec["grid"], base["params"])                     # c1 is the Phase 2 default
    configurations = expand_grid(spec["grid"])
    assert len(configurations) == 4                              # equal budget (D2)
    assert {count_params(build_from_target(base["target"], {**base["params"], **c}))
            for c in configurations} == counts


def test_expand_grid_starts_with_the_defaults_and_varies_the_last_parameter_fastest() -> None:
    assert expand_grid({"a": [1, 2], "b": ["x", "y"]}) == [
        {"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 2, "b": "x"}, {"a": 2, "b": "y"},
    ]


@pytest.mark.parametrize("grid, match", [
    ({"width": [1, 2]}, "not in the base"),
    ({"dropout": [0.5, 0.3]}, "Phase 2 default"),
    ({"dropout": [0.3, 0.3]}, "same value twice"),
    ({"params": [1, 2]}, "clash"),
])
def test_check_grid_rejects_bad_grids(grid: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_grid(grid, {"dropout": 0.3, "hidden": [8], "params": 1})


def test_a_clear_validation_winner_is_selected_whatever_its_size() -> None:
    rows = [_row("c1", 0.950, 100), _row("c2", 0.960, 200), _row("c3", 0.955, 100), _row("c4", 0.940, 50)]
    assert select_config(rows)["config_id"] == "c2"


def test_ties_within_0_001_go_to_fewer_parameters() -> None:
    rows = [_row("c1", 0.9600, 200), _row("c2", 0.9605, 200), _row("c3", 0.9597, 150), _row("c4", 0.9500, 50)]
    assert select_config(rows)["config_id"] == "c3"              # c4 has fewer params but is not tied


def test_ties_with_equal_parameters_go_to_the_default() -> None:
    rows = [_row("c1", 0.9600, 100), _row("c2", 0.9608, 100), _row("c3", 0.9500, 100), _row("c4", 0.9609, 100)]
    assert select_config(rows)["config_id"] == "c1"


def test_ties_without_the_default_go_to_the_higher_score() -> None:
    rows = [_row("c1", 0.9500, 100), _row("c2", 0.9605, 100), _row("c3", 0.9610, 100), _row("c4", 0.9500, 100)]
    assert select_config(rows)["config_id"] == "c3"


def test_a_difference_of_exactly_0_001_is_a_tie() -> None:
    assert select_config([_row("c1", 0.959, 100), _row("c2", 0.960, 200)])["config_id"] == "c1"


def test_tune_script_end_to_end(tmp_path: Path) -> None:
    base = tmp_path / "tiny.yaml"
    base.write_text(TINY_BASE, encoding="utf-8")
    tuning_dir = tmp_path / "tuning"
    tuning_dir.mkdir()
    (tuning_dir / "tiny.yaml").write_text(yaml.safe_dump(
        {"model": "tiny", "base_config": str(base), "grid": {"dropout": [0.3, 0.5], "hidden": [[8], [16]]}},
        sort_keys=False), encoding="utf-8")
    final_dir, out = tmp_path / "final", tmp_path / "results" / "tuning"
    args = ["--model", "tiny", "--tuning-dir", str(tuning_dir), "--final-dir", str(final_dir),
            "--output-root", str(out)]
    result = tune_script.main(args)

    assert result["summary"] == out / "tiny" / "tuning_summary.csv"
    with result["summary"].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["config_id", "dropout", "hidden", *RESULT_FIELDS]
    assert [r["config_id"] for r in rows] == ["c1", "c2", "c3", "c4"]
    assert [r["hidden"] for r in rows] == ["[8]", "[16]", "[8]", "[16]"]
    assert [r["selected"] for r in rows].count("true") == 1
    chosen = next(r for r in rows if r["selected"] == "true")
    assert chosen["config_id"] == result["selected"]
    recorded = [_row(r["config_id"], float(r["val_macro_f1"]), int(r["params"])) for r in rows]
    assert select_config(recorded)["config_id"] == result["selected"]

    protocol = load_yaml(CONFIGS / "base.yaml")["train"]
    for r in rows:
        run_dir = out / "tiny" / r["run_id"]
        m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert m["status"] == "tuning run, validation only" and m["seed"] == 42
        assert m["test"] is None and m["latency"] is None and m["protocol_override"] is False
        assert r["run_id"].endswith(f"_tune-{r['config_id']}") and int(r["params"]) == m["params_trainable"]
        resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
        assert resolved["model"]["params"]["hidden"] == json.loads(r["hidden"])
        assert resolved["train"] == protocol                     # only model.params change
        assert not (run_dir / "predictions_test.csv").exists()

    assert result["final_config"] == final_dir / "tiny.yaml"
    assert result["final_config"].read_text(encoding="utf-8").startswith("# Generated by scripts/tune.py")
    final, original = load_yaml(result["final_config"]), load_yaml(base)
    assert {k: v for k, v in final.items() if k != "model"} == {k: v for k, v in original.items() if k != "model"}
    assert final["model"]["target"] == original["model"]["target"]
    assert final["model"]["params"] == {**original["model"]["params"], "dropout": float(chosen["dropout"]),
                                        "hidden": json.loads(chosen["hidden"])}
    load_config(result["final_config"])                          # no train or seed keys

    with pytest.raises(FileExistsError, match="--force"):        # refuses before training anything
        tune_script.main(args)
