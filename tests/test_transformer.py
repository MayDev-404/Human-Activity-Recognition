"""Tests for Model 4, the Transformer encoder on raw windows."""

from __future__ import annotations

import warnings

import pytest
import torch

from har.data.bundle import make_synthetic_bundle
from har.models.transformer import TransformerHAR
from har.utils.build import build_from_target
from har.utils.config import load_config
from har.utils.paths import CONFIGS

EXPECTED_PARAMS = 159_302  # BUILD_SPEC 7.4


def _n_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _from_config() -> TransformerHAR:
    cfg = load_config(CONFIGS / "transformer.yaml")
    return build_from_target(cfg["model"]["target"], cfg["model"]["params"])


def test_config_loads_within_the_protocol_and_builds_the_model() -> None:
    cfg = load_config(CONFIGS / "transformer.yaml")
    assert cfg["name"] == "transformer" and cfg["owner"] == "M1"
    assert cfg["data"]["representation"] == "raw"
    assert cfg["protocol_override"] is False
    assert isinstance(_from_config(), TransformerHAR)


def test_parameter_count_matches_spec() -> None:
    assert _n_trainable(_from_config()) == EXPECTED_PARAMS
    assert _n_trainable(TransformerHAR()) == EXPECTED_PARAMS  # defaults equal the config


def test_construction_does_not_warn_about_nested_tensors() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TransformerHAR()
    assert not [w for w in caught if "nested" in str(w.message).lower()]


def test_positional_embedding_shape_and_init() -> None:
    torch.manual_seed(0)
    pos = TransformerHAR().pos_embedding
    assert pos.shape == (1, 128, 64) and pos.requires_grad
    assert abs(pos.std().item() - 0.02) < 0.003


def test_output_is_logits_of_shape_b_by_6() -> None:
    bundle = make_synthetic_bundle((128, 9), batch_size=16)
    x, _ = next(iter(bundle.train))
    model = _from_config().eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (16, 6) and out.dtype == torch.float32
    assert not torch.allclose(out.sum(dim=1), torch.ones(16))  # raw logits, no softmax


def test_batch_size_one_in_eval_mode_is_deterministic() -> None:
    model = _from_config().eval()
    x = torch.randn(1, 128, 9)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert a.shape == (1, 6)
    assert torch.equal(a, b)


def test_positional_embedding_makes_the_model_order_aware() -> None:
    """Mean pooling alone is permutation invariant; the positional embedding breaks that."""
    torch.manual_seed(0)
    model = TransformerHAR().eval()
    x = torch.randn(2, 128, 9)
    with torch.no_grad():
        out = model(x)
        out_reversed = model(x.flip(dims=[1]))
    assert not torch.allclose(out, out_reversed, atol=1e-5)


def test_sequence_longer_than_seq_len_is_rejected() -> None:
    with pytest.raises(ValueError):
        TransformerHAR().eval()(torch.randn(1, 129, 9))


def test_fits_learnable_synthetic_data() -> None:
    # Shorter windows keep this test fast on CPU; the code path is the same as for 128 steps.
    torch.manual_seed(0)
    bundle = make_synthetic_bundle((16, 9), n_train=256, batch_size=32, seed=0)
    model = TransformerHAR(seq_len=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
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
    assert train_acc > 0.4  # chance is 1/6
