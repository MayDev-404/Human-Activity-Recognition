"""Shared training loop: Adam, cross-entropy, gradient clipping, early stopping (BUILD_SPEC 5.5, 8.1, M2.3).

Every model trains through ``fit`` with the protocol from ``configs/base.yaml``, so differences
between models come from the architectures rather than from the training code.
"""

from __future__ import annotations

import math
import operator
import time
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from har.data.bundle import DataBundle
from har.eval.metrics import compute_metrics
from har.eval.profiling import best_available_device

# One row per epoch (BUILD_SPEC 8.3), also the column order of history.csv.
HISTORY_FIELDS: tuple[str, ...] = (
    "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "epoch_time_s",
)


@dataclass
class TrainConfig:
    """The ``train`` block of ``configs/base.yaml``. ``optimizer`` must be ``"adam"`` (protocol)."""

    batch_size: int
    max_epochs: int
    optimizer: str
    lr: float
    weight_decay: float
    patience: int
    monitor: str
    mode: str
    grad_clip_norm: float
    device: str

    def __post_init__(self) -> None:
        if str(self.optimizer).lower() != "adam":
            raise ValueError(f"optimizer must be 'adam' (shared protocol), got {self.optimizer!r}")
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode!r}")
        if self.monitor not in HISTORY_FIELDS[1:-2]:
            raise ValueError(f"monitor must be one of {HISTORY_FIELDS[1:-2]}, got {self.monitor!r}")
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("batch_size, max_epochs and patience must all be at least 1")

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        """Build from a config dict with exactly the dataclass fields (unknown or missing keys raise)."""
        names = {f.name for f in fields(cls)}
        unknown, missing = sorted(set(d) - names), sorted(names - set(d))
        if unknown or missing:
            raise ValueError(f"train config: unknown keys {unknown}, missing keys {missing}")
        return cls(**d)


@dataclass
class TrainResult:
    """Outcome of ``fit``. ``best_state`` is a CPU copy of the restored best ``state_dict``."""

    best_state: dict
    best_epoch: int
    epochs_run: int
    history: list[dict]
    train_time_s: float
    mean_epoch_time_s: float


def resolve_device(name: str = "auto") -> torch.device:
    """``"auto"`` picks cuda, then mps, then cpu; any other name is checked for availability."""
    if name == "auto":
        return torch.device(best_available_device())
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {name!r} requested but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError(f"device {name!r} requested but MPS is not available")
    return device


def _logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``model`` in eval mode over ``loader``: CPU tensors y ``(N,)`` and logits ``(N, C)``."""
    model.eval()
    ys, outs = [], []
    with torch.inference_mode():
        for x, y in loader:
            outs.append(model(x.to(device)).float().cpu())
            ys.append(y)
    return torch.cat(ys), torch.cat(outs)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``y_true`` int64 ``(N,)``, ``y_pred`` int64 ``(N,)`` and softmax ``probs`` float32 ``(N, C)``."""
    y, logits = _logits(model, loader, device)
    probs = torch.softmax(logits, dim=1)
    return y.numpy().astype(np.int64), logits.argmax(dim=1).numpy().astype(np.int64), probs.numpy()


def fit(model: nn.Module, bundle: DataBundle, cfg: TrainConfig, verbose: bool = True) -> TrainResult:
    """Train ``model`` on ``bundle.train`` and early-stop on ``cfg.monitor`` over ``bundle.val``.

    Each epoch is a training pass (CrossEntropyLoss, Adam, gradient norm clipped to
    ``cfg.grad_clip_norm``) followed by a validation pass in ``eval()`` mode under
    ``torch.inference_mode()``. Training stops after ``cfg.patience`` epochs without a strict
    improvement, or at ``cfg.max_epochs``. The best epoch's weights are restored into
    ``model`` (left on the training device) before returning. ``epoch_time_s`` and
    ``train_time_s`` include the validation passes.
    """
    if bundle.train.batch_size != cfg.batch_size:
        raise ValueError(f"bundle was built with batch size {bundle.train.batch_size}, "
                         f"but the train config says {cfg.batch_size}")
    device = resolve_device(cfg.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    improved = operator.gt if cfg.mode == "max" else operator.lt

    history: list[dict] = []
    best_score: float | None = None
    best_epoch, stale = 0, 0
    best_state: dict = {}
    start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), dtype=torch.int64, device=device)
        seen = 0
        for x, y in tqdm(bundle.train, desc=f"epoch {epoch}/{cfg.max_epochs}", unit="batch",
                         leave=False, disable=not verbose):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if cfg.grad_clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            loss_sum += loss.detach() * len(y)
            correct += (logits.argmax(dim=1) == y).sum()
            seen += len(y)
        train_loss = loss_sum.item() / seen
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"training loss is {train_loss} at epoch {epoch}")

        y_val, logits_val = _logits(model, bundle.val, device)
        val = compute_metrics(y_val.numpy(), logits_val.argmax(dim=1).numpy(), bundle.class_names)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": correct.item() / seen,
            "val_loss": loss_fn(logits_val, y_val).item(),
            "val_acc": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": time.perf_counter() - epoch_start,
        }
        history.append(row)

        score = row[cfg.monitor]
        if best_score is None or improved(score, best_score):
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if verbose:
            tqdm.write(f"epoch {epoch:3d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
                       f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f} macro-F1 {row['val_macro_f1']:.4f}"
                       f" | {row['epoch_time_s']:.1f}s" + ("  *best" if stale == 0 else ""))
        if stale >= cfg.patience:
            if verbose:
                tqdm.write(f"early stopping: no {cfg.monitor} improvement for {cfg.patience} epochs "
                           f"(best epoch {best_epoch})")
            break

    train_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    return TrainResult(
        best_state=best_state,
        best_epoch=best_epoch,
        epochs_run=len(history),
        history=history,
        train_time_s=train_time,
        mean_epoch_time_s=float(np.mean([r["epoch_time_s"] for r in history])),
    )
