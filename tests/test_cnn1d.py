"""Tests for Model 2, the 1D-CNN on raw inertial windows."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from har.data.bundle import make_synthetic_bundle
from har.models.cnn1d import CNN1D
from har.utils.build import build_from_target
from har.utils.config import load_config
from har.utils.paths import CONFIGS

EXPECTED_PARAMS = 197_702  # BUILD_SPEC 7.2


def _n_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _from_config() -> CNN1D:
    cfg = load_config(CONFIGS / "cnn1d.yaml")
    return build_from_target(cfg["model"]["target"], cfg["model"]["params"])


def test_config_loads_within_the_protocol_and_builds_the_model() -> None:
    cfg = load_config(CONFIGS / "cnn1d.yaml")
    assert cfg["name"] == "cnn1d" and cfg["owner"] == "M2"
    assert cfg["data"]["representation"] == "raw"
    assert cfg["protocol_override"] is False
    assert isinstance(_from_config(), CNN1D)


def test_parameter_count_matches_spec() -> None:
    assert _n_trainable(_from_config()) == EXPECTED_PARAMS
    assert _n_trainable(CNN1D()) == EXPECTED_PARAMS  # defaults equal the config


def test_layer_layout_matches_spec() -> None:
    model = CNN1D()
    convs = [m for m in model.features if isinstance(m, nn.Conv1d)]
    assert [(c.in_channels, c.out_channels, c.kernel_size[0], c.padding[0]) for c in convs] == [
        (9, 64, 5, 2), (64, 64, 5, 2), (64, 128, 5, 2), (128, 128, 5, 2), (128, 128, 3, 1)]
    assert sum(isinstance(m, nn.BatchNorm1d) for m in model.features) == 5
    # Feature map after the conv stack: two poolings halve 128 time steps twice.
    with torch.no_grad():
        assert model.eval().features(torch.randn(2, 9, 128)).shape == (2, 128, 32)


def test_output_is_logits_of_shape_b_by_6() -> None:
    bundle = make_synthetic_bundle((128, 9), batch_size=32)
    x, _ = next(iter(bundle.train))
    model = _from_config().eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (32, 6) and out.dtype == torch.float32
    assert not torch.allclose(out.sum(dim=1), torch.ones(32))  # raw logits, no softmax


def test_expects_time_major_windows() -> None:
    model = CNN1D().eval()
    with pytest.raises(RuntimeError):
        model(torch.randn(2, 9, 128))       # (B, C, T) is the wrong layout; the model permutes itself


def test_batch_size_one_in_eval_mode_is_deterministic() -> None:
    model = _from_config().eval()
    x = torch.randn(1, 128, 9)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert a.shape == (1, 6)
    assert torch.equal(a, b)  # dropout off, BatchNorm uses running statistics


def test_other_sizes_work_and_bad_arguments_raise() -> None:
    model = CNN1D(in_channels=3, channels=[8, 16], kernel_sizes=[3, 3], pool_after=[0], n_classes=4).eval()
    assert model(torch.randn(5, 40, 3)).shape == (5, 4)
    with pytest.raises(ValueError, match="kernel size"):
        CNN1D(channels=[8, 16], kernel_sizes=[3])
    with pytest.raises(ValueError, match="pool_after"):
        CNN1D(channels=[8, 16], kernel_sizes=[3, 3], pool_after=[2])


def test_fits_learnable_synthetic_raw_shaped_data() -> None:
    torch.manual_seed(0)
    bundle = make_synthetic_bundle((128, 9), n_train=256, batch_size=32, seed=0)
    model = _from_config()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(15):
        model.train()
        for x, y in bundle.train:
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
    model.eval()
    x_train, y_train = bundle.train.dataset.tensors
    with torch.no_grad():
        train_acc = (model(x_train).argmax(dim=1) == y_train).float().mean().item()
    assert train_acc > 0.6  # chance is 1/6
