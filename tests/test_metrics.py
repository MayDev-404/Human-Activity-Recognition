"""Tests for har.eval.metrics (and har.eval.plots once it exists)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from har.data.constants import ACTIVITY_NAMES
from har.eval.metrics import classification_report_text, compute_metrics

# Toy example, worked out by hand. Classes 3, 4 and 5 never occur.
#   class 0: TP 1, FP 1, FN 1 -> P 1/2, R 1/2, F1 1/2
#   class 1: TP 2, FP 1, FN 0 -> P 2/3, R 1,   F1 4/5
#   class 2: TP 1, FP 0, FN 1 -> P 1,   R 1/2, F1 2/3
TOY_TRUE = np.array([0, 0, 1, 1, 2, 2])
TOY_PRED = np.array([0, 1, 1, 1, 2, 0])


def test_hand_computed_toy_example() -> None:
    m = compute_metrics(TOY_TRUE, TOY_PRED, ACTIVITY_NAMES)
    assert m["accuracy"] == pytest.approx(4 / 6)
    # labels=list(range(6)): the three absent classes score 0 and still count in the average.
    assert m["macro_f1"] == pytest.approx((1 / 2 + 4 / 5 + 2 / 3) / 6)

    walking, upstairs, downstairs = (m["per_class"][n] for n in ACTIVITY_NAMES[:3])
    assert (walking["precision"], walking["recall"], walking["f1"]) == pytest.approx((1 / 2, 1 / 2, 1 / 2))
    assert (upstairs["precision"], upstairs["recall"], upstairs["f1"]) == pytest.approx((2 / 3, 1, 4 / 5))
    assert (downstairs["precision"], downstairs["recall"], downstairs["f1"]) == pytest.approx((1, 1 / 2, 2 / 3))
    assert [m["per_class"][n]["support"] for n in ACTIVITY_NAMES] == [2, 2, 2, 0, 0, 0]

    assert m["confusion"][:3] == [[1, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]]


def test_all_classes_present_perfect_and_one_error() -> None:
    y = np.repeat(np.arange(6), 2)
    assert compute_metrics(y, y, ACTIVITY_NAMES)["macro_f1"] == pytest.approx(1.0)

    y_pred = y.copy()
    y_pred[0] = 1          # one WALKING window predicted as WALKING_UPSTAIRS
    m = compute_metrics(y, y_pred, ACTIVITY_NAMES)
    # WALKING: P 1, R 1/2, F1 2/3; UPSTAIRS: P 2/3, R 1, F1 4/5; the other four are perfect.
    assert m["accuracy"] == pytest.approx(11 / 12)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 4 / 5 + 4) / 6)


def test_confusion_and_per_class_cover_all_six_classes_when_some_are_absent() -> None:
    m = compute_metrics(np.array([5, 5, 5]), np.array([5, 5, 3]), ACTIVITY_NAMES)
    assert np.array(m["confusion"]).shape == (6, 6)
    assert list(m["per_class"]) == ACTIVITY_NAMES
    # Absent classes stay in their own rows, not shifted onto other names.
    assert m["confusion"][5] == [0, 0, 0, 1, 0, 2]
    assert m["per_class"]["LAYING"]["support"] == 3
    assert m["per_class"]["SITTING"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}


def test_output_is_json_serialisable_with_plain_python_types() -> None:
    m = compute_metrics(TOY_TRUE, TOY_PRED, ACTIVITY_NAMES)
    assert json.loads(json.dumps(m)) == m
    assert type(m["accuracy"]) is float and type(m["macro_f1"]) is float
    assert all(type(v) is int for row in m["confusion"] for v in row)
    assert type(m["per_class"]["WALKING"]["support"]) is int


def test_rejects_mismatched_or_out_of_range_labels() -> None:
    with pytest.raises(ValueError, match="labels but"):
        compute_metrics(np.array([0, 1]), np.array([0]), ACTIVITY_NAMES)
    with pytest.raises(ValueError, match="outside"):
        compute_metrics(np.array([0, 6]), np.array([0, 1]), ACTIVITY_NAMES)   # labels still 1..6
    with pytest.raises(ValueError, match="empty"):
        compute_metrics(np.array([]), np.array([]), ACTIVITY_NAMES)


def test_classification_report_lists_every_class() -> None:
    text = classification_report_text(TOY_TRUE, TOY_PRED, ACTIVITY_NAMES)
    for name in ACTIVITY_NAMES:
        assert name in text
    assert "macro avg" in text
