"""Confusion-matrix and training-curve figures for run folders and the report (BUILD_SPEC M2.1).

Figures are drawn on ``matplotlib.figure.Figure`` objects, never through ``pyplot``, so PNGs
are rendered by the Agg backend without touching any global backend state (safe on headless
machines and inside tests). Every figure is saved at 200 dpi.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

DPI = 200

# Validated default palette: categorical slots 1-2 (train blue, validation orange), the blue
# sequential ramp (steps 100, 250, 400, 550, 700) for magnitude, and recessive chrome. The
# same chrome as the dataset figures in har.data.stats, so all report figures match.
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"
SURFACE = "#ffffff"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "har_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def display_name(name: str) -> str:
    """``WALKING_UPSTAIRS -> Walking upstairs``."""
    return name.replace("_", " ").capitalize()


def _style_axes(ax, grid_axis: str | None = "y") -> None:
    """Recessive chrome: no box, a hairline baseline, light gridlines behind the data."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0, labelsize=6.5)


def _save(fig: Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    return path


def plot_confusion(cm: Sequence[Sequence[int]] | np.ndarray, class_names: Sequence[str], path: Path,
                   normalize: bool = True, title: str = "") -> Path:
    """Save a confusion-matrix heatmap (rows = true class, columns = predicted) as a PNG.

    ``cm`` is a ``(C, C)`` count matrix with ``C == len(class_names)``. With ``normalize``
    each cell is coloured by its row percentage (recall per true class) and labelled with
    the percentage and the count; a row with no windows shows 0%. Otherwise cells are
    coloured and labelled by count. Returns ``path``.
    """
    counts = np.asarray(cm, dtype=np.int64)
    n = len(class_names)
    if counts.shape != (n, n):
        raise ValueError(f"confusion matrix has shape {counts.shape}, expected ({n}, {n})")

    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        shade = np.divide(counts, row_sums, out=np.zeros(counts.shape), where=row_sums > 0)
        labels = [[f"{100 * shade[i, j]:.1f}%\n{counts[i, j]:,}" for j in range(n)] for i in range(n)]
    else:
        shade = counts / max(int(counts.max()), 1)
        labels = [[f"{counts[i, j]:,}" for j in range(n)] for i in range(n)]

    fig = Figure(figsize=(4.0, 3.7), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    # Surface-coloured cell edges give the 2 px gap between neighbouring fills.
    ax.pcolormesh(shade, cmap=SEQUENTIAL, vmin=0.0, vmax=1.0, edgecolors=SURFACE, linewidth=1.5)
    for i in range(n):
        for j in range(n):
            ink = "#ffffff" if shade[i, j] > 0.55 else INK if counts[i, j] else INK_MUTED
            ax.text(j + 0.5, i + 0.5, labels[i][j], ha="center", va="center", fontsize=5.5, color=ink)

    names = [display_name(c) for c in class_names]
    ax.set_xticks(np.arange(n) + 0.5, names, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(n) + 0.5, names)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)            # first class at the top
    ax.set_aspect("equal")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=6.5, labelcolor=INK_SECONDARY)
    ax.set_xlabel("Predicted class", fontsize=7, color=INK_SECONDARY)
    ax.set_ylabel("True class", fontsize=7, color=INK_SECONDARY)
    if title:
        ax.set_title(title, fontsize=8, color=INK, loc="left")
    return _save(fig, path)


def plot_history(history: Sequence[dict], path: Path, title: str = "") -> Path:
    """Save training curves as a PNG: train and validation loss (left), validation macro-F1 (right).

    ``history`` holds one row per epoch with at least ``epoch``, ``train_loss``,
    ``val_loss`` and ``val_macro_f1`` (BUILD_SPEC 8.3). The epoch with the best validation
    macro-F1 (the first one, on ties) is marked in both panels. Returns ``path``.
    """
    if not history:
        raise ValueError("history is empty; nothing to plot")
    epochs = [int(r["epoch"]) for r in history]
    val_f1 = [float(r["val_macro_f1"]) for r in history]
    best_epoch = epochs[int(np.argmax(val_f1))]

    fig = Figure(figsize=(6.4, 2.5), facecolor=SURFACE, layout="constrained")
    ax_loss, ax_f1 = fig.subplots(1, 2)
    line = {"linewidth": 1.5, "marker": "o", "markersize": 2.5}

    ax_loss.plot(epochs, [float(r["train_loss"]) for r in history], color=TRAIN_COLOR, label="Train", **line)
    ax_loss.plot(epochs, [float(r["val_loss"]) for r in history], color=VAL_COLOR, label="Validation", **line)
    ax_loss.set_ylabel("Cross-entropy loss", fontsize=7, color=INK_SECONDARY)

    ax_f1.plot(epochs, val_f1, color=VAL_COLOR, **line)
    ax_f1.set_ylabel("Validation macro-F1", fontsize=7, color=INK_SECONDARY)

    for ax in (ax_loss, ax_f1):
        _style_axes(ax)
        ax.axvline(best_epoch, color=INK_MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.set_xlabel("Epoch", fontsize=7, color=INK_SECONDARY)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Bottom of the panel: the F1 curve starts low on the left and sits high after the best epoch.
    ax_f1.annotate(f"best epoch {best_epoch}", xy=(best_epoch, 0.03), xycoords=("data", "axes fraction"),
                   xytext=(3, 0), textcoords="offset points", fontsize=6, color=INK_SECONDARY, va="bottom")

    fig.legend(fontsize=6.5, frameon=False, loc="outside upper center", ncol=2,
               labelcolor=INK_SECONDARY, handlelength=1.5)
    if title:
        fig.suptitle(title, fontsize=8, color=INK, x=0.01, ha="left")
    return _save(fig, path)
