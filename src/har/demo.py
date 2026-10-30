"""Support for the live demo app (``scripts/demo_app.py``): load final checkpoints, classify test windows.

The demo uses one checkpoint per model: the final run with seed ``DEMO_SEED``. It is chosen by
seed, never by test score. Checkpoints (``best.pt``) are gitignored, so the demo runs only on the
machine that trained the final runs, or after their ``best.pt`` files are copied into the same
run folders.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from har.data.features import build_feature_loaders
from har.data.raw import build_raw_loaders, load_raw_split
from har.train.runner import FINAL_TAG
from har.utils.build import build_from_target
from har.utils.paths import RESULTS_ROOT, SPLIT_FILE, UCI_DIR

DEMO_SEED = 42
MODEL_ORDER: tuple[str, ...] = ("mlp", "cnn1d", "bilstm", "gru", "transformer")


@dataclass
class DemoModel:
    """A final checkpoint on CPU in eval mode, with its run folder and recorded ``metrics.json``."""

    name: str
    representation: str      # "features" or "raw"
    model: nn.Module
    run_dir: Path
    metrics: dict


@dataclass
class TestWindows:
    """The official test windows in file order, preprocessed exactly as in training.

    ``features`` ``(N, 561)`` and ``raw`` ``(N, 128, 9)`` are standardised with statistics
    fitted on the training subjects; ``signals`` ``(N, 128, 9)`` are the unscaled raw signals
    (acceleration in g, angular velocity in rad/s) for plotting. ``y`` holds labels 0..5.
    """

    features: torch.Tensor
    raw: torch.Tensor
    signals: np.ndarray
    y: np.ndarray
    subjects: np.ndarray

    def inputs(self, representation: str) -> torch.Tensor:
        """Model input for ``representation``: ``features`` ``(N, 561)`` or ``raw`` ``(N, 128, 9)``."""
        if representation == "features":
            return self.features
        if representation == "raw":
            return self.raw
        raise ValueError(f"unknown representation {representation!r}; expected features or raw")


def final_run_dir(model_name: str, seed: int = DEMO_SEED, results_root: Path = RESULTS_ROOT) -> Path:
    """Run folder of ``model_name`` tagged final with ``seed``; raises ``FileNotFoundError`` if none."""
    for run_dir in sorted((Path(results_root) / model_name).glob(f"*_{FINAL_TAG}")):
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.is_file():
            continue
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        if metrics.get("tag") == FINAL_TAG and metrics.get("seed") == seed and not metrics.get("smoke"):
            return run_dir
    raise FileNotFoundError(f"no final run with seed {seed} for {model_name!r} under {results_root}")


def load_demo_model(run_dir: Path) -> DemoModel:
    """Rebuild the model from ``config.resolved.yaml`` and load ``best.pt``, on CPU in eval mode."""
    run_dir = Path(run_dir)
    checkpoint = run_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} not found; checkpoints are gitignored, so they exist "
                                "only on the machine that trained the run")
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    model = build_from_target(cfg["model"]["target"], cfg["model"].get("params") or {})
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return DemoModel(name=cfg["name"], representation=cfg["data"]["representation"], model=model,
                     run_dir=run_dir, metrics=metrics)


def load_test_windows(uci_dir: Path = UCI_DIR, split_file: Path = SPLIT_FILE) -> TestWindows:
    """Test split through the training pipelines, so scaling matches the checkpoints exactly.

    Batch size and seed only affect the training loader, not the test tensors.
    """
    features = build_feature_loaders(batch_size=64, seed=DEMO_SEED, uci_dir=uci_dir, split_file=split_file)
    raw = build_raw_loaders(batch_size=64, seed=DEMO_SEED, uci_dir=uci_dir, split_file=split_file)
    x_features, y_features = features.test.dataset.tensors
    x_raw, y_raw = raw.test.dataset.tensors
    signals, y, subjects = load_raw_split("test", uci_dir)
    if not (torch.equal(y_features, y_raw) and np.array_equal(y_raw.numpy(), y)):
        raise ValueError("feature and raw test windows are not in the same order")
    return TestWindows(features=x_features, raw=x_raw, signals=signals, y=y, subjects=subjects)


def predict_proba(model: nn.Module, x: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    """Softmax probabilities on CPU: ``(N, *input_shape) -> (N, 6)`` float32."""
    model.eval()
    with torch.inference_mode():
        chunks = [torch.softmax(model(chunk), dim=1) for chunk in torch.split(x, batch_size)]
    return torch.cat(chunks).numpy()


def saved_predictions(run_dir: Path, split: str = "test") -> np.ndarray:
    """``y_pred`` int64 ``(N,)`` from the run's ``predictions_<split>.csv``, in window order."""
    with (Path(run_dir) / f"predictions_{split}.csv").open(encoding="utf-8", newline="") as f:
        return np.array([int(row["y_pred"]) for row in csv.DictReader(f)], dtype=np.int64)
