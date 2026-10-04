"""Tests for Model 3, the bidirectional LSTM, and its GRU ablation."""

from __future__ import annotations

import warnings

import pytest
import torch

from har.data.bundle import make_synthetic_bundle
from har.models.birnn import BiRNN
from har.utils.build import build_from_target
from har.utils.config import load_config
from har.utils.paths import CONFIGS

EXPECTED_PARAMS = {"bilstm": 138_502, "gru": 104_070}  # BUILD_SPEC 7.3
CELL = {"bilstm": "lstm", "gru": "gru"}


def _n_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _from_config(name: str) -> BiRNN:
    cfg = load_config(CONFIGS / f"{name}.yaml")
    return build_from_target(cfg["model"]["target"], cfg["model"]["params"])


@pytest.mark.parametrize("name", ["bilstm", "gru"])
def test_config_loads_within_the_protocol_and_builds_the_model(name: str) -> None:
    cfg = load_config(CONFIGS / f"{name}.yaml")
    assert cfg["name"] == name and cfg["owner"] == "M3"
    assert cfg["data"]["representation"] == "raw"
    assert cfg["protocol_override"] is False
    model = _from_config(name)
    assert isinstance(model, BiRNN) and model.cell == CELL[name]
    assert isinstance(model.rnn, torch.nn.LSTM if name == "bilstm" else torch.nn.GRU)
    assert model.rnn.batch_first and model.rnn.bidirectional


def test_configs_differ_only_in_name_and_cell() -> None:
    lstm, gru = (load_config(CONFIGS / f"{n}.yaml") for n in ("bilstm", "gru"))
    assert {**lstm["model"]["params"], "cell": "gru"} == gru["model"]["params"]
    assert {**lstm, "name": "gru", "model": gru["model"]} == gru


@pytest.mark.parametrize("name", ["bilstm", "gru"])
def test_parameter_count_matches_spec(name: str) -> None:
    assert _n_trainable(_from_config(name)) == EXPECTED_PARAMS[name]
    assert _n_trainable(BiRNN(cell=CELL[name])) == EXPECTED_PARAMS[name]  # defaults equal the config


@pytest.mark.parametrize("name", ["bilstm", "gru"])
def test_output_is_logits_of_shape_b_by_6(name: str) -> None:
    bundle = make_synthetic_bundle((128, 9), batch_size=16)
    x, _ = next(iter(bundle.train))
    model = _from_config(name).eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (16, 6) and out.dtype == torch.float32
    assert not torch.allclose(out.sum(dim=1), torch.ones(16))  # raw logits, no softmax


@pytest.mark.parametrize("name", ["bilstm", "gru"])
def test_batch_size_one_in_eval_mode_is_deterministic(name: str) -> None:
    model = _from_config(name).eval()
    x = torch.randn(1, 128, 9)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert a.shape == (1, 6)
    assert torch.equal(a, b)


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_last_pooling_uses_final_forward_and_backward_states_of_the_top_layer(cell: str) -> None:
    """Forward direction ends at t = T-1, backward direction ends at t = 0."""
    torch.manual_seed(0)
    model = BiRNN(cell=cell).eval()
    x = torch.randn(3, 20, 9)
    with torch.no_grad():
        out, _ = model.rnn(x)
        hidden = model.rnn.hidden_size
        expected = model.head(torch.cat([out[:, -1, :hidden], out[:, 0, hidden:]], dim=1))
        torch.testing.assert_close(model(x), expected)


def test_mean_pooling_averages_top_layer_outputs_over_time() -> None:
    torch.manual_seed(0)
    model = BiRNN(pooling="mean").eval()
    x = torch.randn(3, 20, 9)
    with torch.no_grad():
        out, _ = model.rnn(x)
        torch.testing.assert_close(model(x), model.head(out.mean(dim=1)))


@pytest.mark.parametrize("cell", ["lstm", "gru"])
@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_unidirectional_and_mean_variants_work(cell: str, pooling: str) -> None:
    for bidirectional, width in ((True, 128), (False, 64)):
        model = BiRNN(cell=cell, bidirectional=bidirectional, pooling=pooling).eval()
        assert model.head.in_features == width
        with torch.no_grad():
            assert model(torch.randn(2, 128, 9)).shape == (2, 6)


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_unidirectional_last_pooling_uses_the_final_state(cell: str) -> None:
    torch.manual_seed(0)
    model = BiRNN(cell=cell, bidirectional=False, pooling="last").eval()
    x = torch.randn(2, 20, 9)
    with torch.no_grad():
        out, _ = model.rnn(x)
        torch.testing.assert_close(model(x), model.head(out[:, -1]))


def test_single_layer_does_not_warn_about_dropout() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = BiRNN(num_layers=1, dropout=0.3)
    assert not [w for w in caught if "dropout" in str(w.message).lower()]
    assert model.rnn.dropout == 0.0 and model.dropout.p == 0.3


def test_invalid_cell_or_pooling_is_rejected() -> None:
    with pytest.raises(ValueError, match="cell"):
        BiRNN(cell="rnn")
    with pytest.raises(ValueError, match="pooling"):
        BiRNN(pooling="max")


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_fits_learnable_synthetic_data(cell: str) -> None:
    # Shorter windows keep this test fast on CPU; the code path is the same as for 128 steps.
    torch.manual_seed(0)
    bundle = make_synthetic_bundle((16, 9), n_train=256, batch_size=32, seed=0)
    model = BiRNN(cell=cell)
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
