"""Tests for the DataBundle contract and make_synthetic_bundle."""

from __future__ import annotations

import pytest
import torch

from har.data.bundle import DataBundle, make_synthetic_bundle
from har.data.constants import ACTIVITY_NAMES, CHANNELS, N_CLASSES, N_FEATURES, WINDOW_LEN


def test_constants_match_the_protocol() -> None:
    assert len(CHANNELS) == 9 and len(set(CHANNELS)) == 9
    assert CHANNELS[:3] == ["body_acc_x", "body_acc_y", "body_acc_z"]
    assert CHANNELS[-3:] == ["total_acc_x", "total_acc_y", "total_acc_z"]
    assert ACTIVITY_NAMES == ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                              "SITTING", "STANDING", "LAYING"]
    assert (N_CLASSES, WINDOW_LEN, N_FEATURES) == (6, 128, 561)


@pytest.mark.parametrize("input_shape", [(N_FEATURES,), (WINDOW_LEN, len(CHANNELS))])
def test_synthetic_bundle_shapes_and_dtypes(input_shape: tuple[int, ...]) -> None:
    bundle = make_synthetic_bundle(input_shape, n_train=100, n_val=40, n_test=30, batch_size=32)
    assert isinstance(bundle, DataBundle)
    assert bundle.input_shape == input_shape
    assert bundle.n_classes == 6
    assert bundle.class_names == ACTIVITY_NAMES
    assert bundle.meta["counts"] == {"train": 100, "val": 40, "test": 30}
    for name, n in (("train", 100), ("val", 40), ("test", 30)):
        loader = getattr(bundle, name)
        assert loader.num_workers == 0
        assert len(loader.dataset) == n
        x, y = next(iter(loader))
        assert x.dtype == torch.float32 and y.dtype == torch.int64
        assert x.shape == (min(32, n), *input_shape)
        assert y.shape == (min(32, n),)
        assert int(y.min()) >= 0 and int(y.max()) < 6


def test_synthetic_bundle_is_deterministic_per_seed() -> None:
    a = make_synthetic_bundle((10,), seed=3)
    b = make_synthetic_bundle((10,), seed=3)
    c = make_synthetic_bundle((10,), seed=4)
    xa, ya = a.train.dataset.tensors
    xb, yb = b.train.dataset.tensors
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert not torch.equal(xa, c.train.dataset.tensors[0])
    # the seeded generator also fixes the shuffled batch order
    assert torch.equal(next(iter(a.train))[0], next(iter(b.train))[0])


def test_val_and_test_loaders_are_not_shuffled() -> None:
    bundle = make_synthetic_bundle((10,), batch_size=16)
    for loader in (bundle.val, bundle.test):
        x, _ = next(iter(loader))
        assert torch.equal(x, loader.dataset.tensors[0][:16])


def test_non_default_class_count_gets_generic_names() -> None:
    bundle = make_synthetic_bundle((5,), n_classes=3)
    assert bundle.class_names == ["class_0", "class_1", "class_2"]
    assert int(bundle.train.dataset.tensors[1].max()) < 3


def _linear_probe_val_accuracy(bundle: DataBundle, steps: int = 300) -> float:
    """Fit a linear classifier on the train split (full batch) and return val accuracy."""
    torch.manual_seed(0)
    x_tr, y_tr = bundle.train.dataset.tensors
    x_va, y_va = bundle.val.dataset.tensors
    probe = torch.nn.Linear(x_tr[0].numel(), bundle.n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(probe(x_tr.flatten(1)), y_tr).backward()
        opt.step()
    with torch.no_grad():
        return (probe(x_va.flatten(1)).argmax(1) == y_va).float().mean().item()


def test_learnable_labels_depend_on_inputs() -> None:
    kwargs = dict(input_shape=(8,), n_train=1024, n_val=512, n_test=16, seed=1)
    learnable = _linear_probe_val_accuracy(make_synthetic_bundle(**kwargs, learnable=True))
    random_labels = _linear_probe_val_accuracy(make_synthetic_bundle(**kwargs, learnable=False))
    assert learnable > 0.8          # chance is 1/6
    assert random_labels < 0.35     # nothing to learn
