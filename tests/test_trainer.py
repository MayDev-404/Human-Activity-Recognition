"""Tests for har.train.trainer (synthetic data, CPU, a few seconds)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from torch import nn

from har.data.bundle import make_synthetic_bundle
from har.eval.metrics import compute_metrics
from har.train.trainer import HISTORY_FIELDS, TrainConfig, fit, predict, resolve_device
from har.utils.config import load_yaml
from har.utils.paths import CONFIGS

BATCH = 32


def _cfg(**overrides) -> TrainConfig:
    """The shared protocol from base.yaml, on CPU with the synthetic bundle's batch size."""
    base = load_yaml(CONFIGS / "base.yaml")["train"] | {"device": "cpu", "batch_size": BATCH}
    return TrainConfig.from_dict(base | overrides)


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    # Dropout makes a train-mode validation pass non-deterministic, which the restore test would catch.
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 6))


def _val_macro_f1(model: nn.Module, bundle) -> float:
    y_true, y_pred, _ = predict(model, bundle.val, torch.device("cpu"))
    return compute_metrics(y_true, y_pred, bundle.class_names)["macro_f1"]


def test_from_dict_accepts_base_yaml_and_rejects_other_optimizers() -> None:
    train = load_yaml(CONFIGS / "base.yaml")["train"]
    cfg = TrainConfig.from_dict(train)
    assert (cfg.optimizer, cfg.lr, cfg.batch_size, cfg.patience, cfg.monitor, cfg.mode) == \
        ("adam", 0.001, 64, 10, "val_macro_f1", "max")
    with pytest.raises(ValueError, match="adam"):
        TrainConfig.from_dict(train | {"optimizer": "sgd"})
    with pytest.raises(ValueError, match="unknown keys"):
        TrainConfig.from_dict(train | {"momentum": 0.9})
    with pytest.raises(ValueError, match="missing keys"):
        TrainConfig.from_dict({k: v for k, v in train.items() if k != "lr"})
    with pytest.raises(ValueError, match="monitor"):
        TrainConfig.from_dict(train | {"monitor": "test_acc"})


def test_loss_decreases_and_history_rows_match_the_spec() -> None:
    bundle = make_synthetic_bundle((16,), batch_size=BATCH, seed=0)
    result = fit(_model(), bundle, _cfg(max_epochs=8, patience=8), verbose=False)
    assert result.epochs_run == len(result.history) == 8
    assert all(tuple(row) == HISTORY_FIELDS for row in result.history)
    assert [row["epoch"] for row in result.history] == list(range(1, 9))
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
    assert result.history[-1]["val_macro_f1"] > 1 / 6        # learned something beyond chance
    assert result.train_time_s > 0 and result.mean_epoch_time_s > 0


def test_early_stopping_triggers_patience_epochs_after_the_best() -> None:
    # lr 0: the weights never change, so the first epoch stays the best and training stops
    # exactly `patience` epochs later.
    bundle = make_synthetic_bundle((16,), batch_size=BATCH, seed=0)
    result = fit(nn.Linear(16, 6), bundle, _cfg(lr=0.0, patience=2), verbose=False)
    assert (result.best_epoch, result.epochs_run) == (1, 3)


def test_restored_weights_reproduce_the_best_epoch_val_score() -> None:
    # Random labels: validation macro-F1 wanders, so the best epoch is not the last one.
    bundle = make_synthetic_bundle((16,), batch_size=BATCH, seed=1, learnable=False)
    model = _model(seed=1)
    result = fit(model, bundle, _cfg(max_epochs=40, patience=4), verbose=False)
    assert result.best_epoch < result.epochs_run < 40
    assert result.epochs_run == result.best_epoch + 4
    best_row = result.history[result.best_epoch - 1]
    assert best_row["val_macro_f1"] == max(row["val_macro_f1"] for row in result.history)
    assert _val_macro_f1(model, bundle) == pytest.approx(best_row["val_macro_f1"])
    assert all(torch.equal(v.cpu(), result.best_state[k]) for k, v in model.state_dict().items())


def test_rejects_a_bundle_built_with_another_batch_size() -> None:
    bundle = make_synthetic_bundle((16,), batch_size=16)
    with pytest.raises(ValueError, match="batch size"):
        fit(nn.Linear(16, 6), bundle, _cfg(), verbose=False)


def test_predict_shapes_and_probabilities() -> None:
    bundle = make_synthetic_bundle((16,), n_val=50, batch_size=BATCH)
    model = _model().train()
    y_true, y_pred, probs = predict(model, bundle.val, torch.device("cpu"))
    assert not model.training                                # predict switches to eval mode
    assert y_true.shape == y_pred.shape == (50,) and probs.shape == (50, 6)
    assert y_true.dtype == y_pred.dtype == np.int64 and probs.dtype == np.float32
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(y_pred, probs.argmax(axis=1))
    assert np.array_equal(y_true, bundle.val.dataset.tensors[1].numpy())


def test_resolve_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    expected = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    assert resolve_device("auto").type == expected
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            resolve_device("cuda")


def test_train_config_is_a_plain_dataclass() -> None:
    assert dataclasses.asdict(_cfg())["optimizer"] == "adam"
