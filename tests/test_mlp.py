"""Tests for Model 1, the MLP on engineered features."""

from __future__ import annotations

import torch

from har.data.bundle import make_synthetic_bundle
from har.models.mlp import MLP
from har.utils.build import build_from_target
from har.utils.config import load_config
from har.utils.paths import CONFIGS

EXPECTED_PARAMS = 178_310  # BUILD_SPEC 7.1


def _n_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _from_config() -> MLP:
    cfg = load_config(CONFIGS / "mlp.yaml")
    return build_from_target(cfg["model"]["target"], cfg["model"]["params"])


def test_config_loads_within_the_protocol_and_builds_the_model() -> None:
    cfg = load_config(CONFIGS / "mlp.yaml")
    assert cfg["name"] == "mlp" and cfg["owner"] == "M1"
    assert cfg["data"]["representation"] == "features"
    assert cfg["protocol_override"] is False
    assert isinstance(_from_config(), MLP)


def test_parameter_count_matches_spec() -> None:
    assert _n_trainable(_from_config()) == EXPECTED_PARAMS
    assert _n_trainable(MLP()) == EXPECTED_PARAMS  # defaults equal the config


def test_output_is_logits_of_shape_b_by_6() -> None:
    bundle = make_synthetic_bundle((561,), batch_size=32)
    x, _ = next(iter(bundle.train))
    model = _from_config().eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (32, 6) and out.dtype == torch.float32
    assert not torch.allclose(out.sum(dim=1), torch.ones(32))  # raw logits, no softmax


def test_batch_size_one_in_eval_mode_is_deterministic() -> None:
    model = _from_config().eval()
    x = torch.randn(1, 561)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert a.shape == (1, 6)
    assert torch.equal(a, b)  # dropout off, BatchNorm uses running statistics


def test_other_hidden_sizes_work() -> None:
    model = MLP(input_dim=20, hidden=[16], n_classes=3).eval()
    assert model(torch.randn(4, 20)).shape == (4, 3)


def test_fits_learnable_synthetic_data() -> None:
    torch.manual_seed(0)
    bundle = make_synthetic_bundle((561,), n_train=256, batch_size=32, seed=0)
    model = _from_config()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(10):
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
