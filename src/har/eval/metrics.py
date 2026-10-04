"""Classification metrics shared by every model (BUILD_SPEC 5.5 and M2.1).

Every sklearn call passes ``labels=list(range(n_classes))`` (6 for UCI HAR) and
``zero_division=0``. Without ``labels``, a class missing from ``y_true`` and ``y_pred``
would drop out of the per-class arrays and silently shift every later class name.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def _check_labels(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return both arrays as 1-D int64, raising if shapes differ or a value is outside 0..n_classes-1."""
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true has {y_true.size} labels but y_pred has {y_pred.size}")
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on an empty set")
    for name, y in (("y_true", y_true), ("y_pred", y_pred)):
        if y.min() < 0 or y.max() >= n_classes:
            raise ValueError(f"{name} has values outside 0..{n_classes - 1}: min {y.min()}, max {y.max()}")
    return y_true, y_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/F1/support and the confusion matrix.

    ``y_true`` and ``y_pred`` are ``(N,)`` class indices in ``0..len(class_names)-1``. The
    confusion matrix has rows = true class and columns = predicted class, and is always
    ``len(class_names)`` square, as is the per-class dict, even if a class is absent. A class
    with no support and no predictions scores 0 and still counts in the macro average.
    Every value is a plain Python ``float``/``int``, so the result is JSON-serialisable.
    """
    n_classes = len(class_names)
    y_true, y_pred = _check_labels(y_true, y_pred, n_classes)
    labels = list(range(n_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def classification_report_text(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> str:
    """sklearn's text classification report over all classes, 4 decimals."""
    n_classes = len(class_names)
    y_true, y_pred = _check_labels(y_true, y_pred, n_classes)
    return classification_report(
        y_true, y_pred, labels=list(range(n_classes)), target_names=list(class_names),
        digits=4, zero_division=0,
    )
