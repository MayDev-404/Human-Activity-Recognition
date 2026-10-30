"""Tests for the demo support module (har.demo)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from har.data.splits import load_split
from har.demo import (MODEL_ORDER, final_run_dir, load_demo_model, load_test_windows, predict_proba,
                      saved_predictions)
from har.models.mlp import MLP

TINY_MLP = {"input_dim": 20, "hidden": [8], "dropout": 0.3, "n_classes": 6}


def _write_run(root: Path, name: str, run_id: str, seed: int, tag: str | None = "final",
               smoke: bool = False) -> Path:
    run_dir = root / name / run_id
    run_dir.mkdir(parents=True)
    metrics = {"model": name, "run_id": run_id, "seed": seed, "tag": tag, "smoke": smoke}
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir


def _write_checkpoint(run_dir: Path) -> MLP:
    torch.manual_seed(0)
    model = MLP(**TINY_MLP).eval()
    cfg = {"name": "mlp", "data": {"representation": "features"},
           "model": {"target": "har.models.mlp.MLP", "params": TINY_MLP}}
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "best.pt")
    return model


def test_final_run_dir_picks_the_requested_seed(tmp_path: Path) -> None:
    runs = {seed: _write_run(tmp_path, "mlp", f"2026010{seed - 40}-000000_abc_final", seed) for seed in (43, 42, 44)}
    assert final_run_dir("mlp", results_root=tmp_path) == runs[42]
    assert final_run_dir("mlp", seed=44, results_root=tmp_path) == runs[44]


def test_final_run_dir_ignores_untagged_and_smoke_runs(tmp_path: Path) -> None:
    _write_run(tmp_path, "mlp", "20260101-000000_abc", 42, tag=None)
    _write_run(tmp_path, "mlp", "20260102-000000_abc_final", 42, smoke=True)
    with pytest.raises(FileNotFoundError):
        final_run_dir("mlp", results_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        final_run_dir("cnn1d", results_root=tmp_path)


def test_load_demo_model_restores_the_checkpoint_in_eval_mode(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "mlp", "20260101-000000_abc_final", 42)
    original = _write_checkpoint(run_dir)
    loaded = load_demo_model(run_dir)
    assert loaded.name == "mlp" and loaded.representation == "features" and loaded.run_dir == run_dir
    assert loaded.metrics["seed"] == 42
    assert not loaded.model.training
    x = torch.randn(1, 20)  # batch size 1 works in eval mode (BatchNorm running statistics)
    with torch.no_grad():
        assert torch.equal(loaded.model(x), original(x))


def test_load_demo_model_without_checkpoint_raises(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "mlp", "20260101-000000_abc_final", 42)
    with pytest.raises(FileNotFoundError, match="gitignored"):
        load_demo_model(run_dir)


def test_predict_proba_rows_are_distributions_independent_of_batching() -> None:
    torch.manual_seed(0)
    model = MLP(**TINY_MLP).eval()
    x = torch.randn(10, 20)
    probs = predict_proba(model, x)
    assert probs.shape == (10, 6) and probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(predict_proba(model, x, batch_size=3), probs, rtol=1e-5, atol=1e-7)


def test_saved_predictions_reads_y_pred_in_order(tmp_path: Path) -> None:
    (tmp_path / "predictions_test.csv").write_text("index,y_true,y_pred\n0,4,4\n1,3,4\n2,0,5\n", encoding="utf-8")
    assert saved_predictions(tmp_path).tolist() == [4, 4, 5]


@pytest.fixture(scope="module")
def windows():
    return load_test_windows()


@pytest.mark.needs_data
def test_test_windows_are_aligned_across_representations(windows) -> None:
    assert windows.features.shape == (2947, 561) and windows.raw.shape == (2947, 128, 9)
    assert windows.signals.shape == (2947, 128, 9) and windows.y.shape == (2947,)
    assert set(np.unique(windows.y)) == set(range(6))
    assert sorted(np.unique(windows.subjects).tolist()) == load_split()["test_subjects"]
    with pytest.raises(ValueError):
        windows.inputs("synthetic")


@pytest.mark.needs_data
@pytest.mark.parametrize("name", MODEL_ORDER)
def test_live_predictions_match_the_committed_final_run(name: str, windows) -> None:
    try:
        run_dir = final_run_dir(name)
    except FileNotFoundError:
        pytest.skip(f"no final run for {name}")
    if not (run_dir / "best.pt").is_file():
        pytest.skip("best.pt is gitignored; it exists only on the machine that trained the run")
    demo = load_demo_model(run_dir)
    y_pred = predict_proba(demo.model, windows.inputs(demo.representation)).argmax(axis=1)
    np.testing.assert_array_equal(y_pred, saved_predictions(run_dir))
    assert float((y_pred == windows.y).mean()) == pytest.approx(demo.metrics["test"]["accuracy"])
