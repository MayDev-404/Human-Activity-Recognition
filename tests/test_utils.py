"""Tests for har.utils: paths, protocol-locked config loading, build_from_target and seeding."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from har.utils.build import build_from_target
from har.utils.config import ProtocolError, deep_merge, load_config, load_yaml
from har.utils.paths import CONFIGS, DATA_ROOT, REPO_ROOT, SPLIT_FILE, UCI_DIR
from har.utils.seed import make_generator, set_seed


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


MODEL_CFG = {
    "name": "dummy",
    "owner": "M1",
    "data": {"representation": "features"},
    "model": {"target": "torch.nn.Linear", "params": {"in_features": 561, "out_features": 6}},
}


# ---------------------------------------------------------------- paths

def test_paths_point_into_the_repo() -> None:
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert DATA_ROOT == REPO_ROOT / "data" / "raw"
    assert UCI_DIR.name == "UCI HAR Dataset"
    assert SPLIT_FILE == CONFIGS / "split.json"
    assert (CONFIGS / "base.yaml").is_file()


# ---------------------------------------------------------------- config

def test_base_yaml_holds_the_shared_protocol() -> None:
    base = load_yaml(CONFIGS / "base.yaml")
    assert base["seed"] == 42
    assert base["data"]["augment"] is False
    assert base["train"] == {
        "batch_size": 64, "max_epochs": 50, "optimizer": "adam", "lr": 0.001,
        "weight_decay": 0.0, "patience": 10, "monitor": "val_macro_f1", "mode": "max",
        "grad_clip_norm": 1.0, "device": "auto",
    }


def test_load_config_deep_merges_base_and_model(tmp_path: Path) -> None:
    cfg = load_config(_write_yaml(tmp_path / "m.yaml", MODEL_CFG))
    assert cfg["seed"] == 42
    assert cfg["train"]["lr"] == 0.001
    # nested merge keeps base keys next to model keys
    assert cfg["data"] == {"augment": False, "representation": "features"}
    assert cfg["model"]["target"] == "torch.nn.Linear"
    assert cfg["protocol_override"] is False


@pytest.mark.parametrize("key, value", [("train", {"lr": 0.01}), ("seed", 7)])
def test_protocol_keys_in_model_config_are_rejected(tmp_path: Path, key: str, value: object) -> None:
    path = _write_yaml(tmp_path / "m.yaml", {**MODEL_CFG, key: value})
    with pytest.raises(ProtocolError):
        load_config(path)


def test_protocol_override_is_allowed_only_explicitly_and_recorded(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "m.yaml", {**MODEL_CFG, "train": {"lr": 0.01}})
    cfg = load_config(path, allow_protocol_override=True)
    assert cfg["train"]["lr"] == 0.01
    assert cfg["train"]["batch_size"] == 64  # other protocol values still come from base
    assert cfg["protocol_override"] is True


def test_deep_merge_does_not_modify_inputs() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    override = {"a": {"y": 3}, "c": [1]}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1, "c": [1]}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}
    assert override == {"a": {"y": 3}, "c": [1]}


def test_committed_model_configs_respect_the_protocol() -> None:
    """Every model config in configs/ must load without overriding the protocol."""
    for path in sorted(CONFIGS.glob("*.yaml")):
        if path.name in {"base.yaml", "team.yaml"}:
            continue
        cfg = load_config(path)
        assert cfg["protocol_override"] is False, path.name
        assert cfg["data"].get("augment") is False, f"{path.name}: augmentation must stay off"


def test_team_file_lists_three_members_without_emails() -> None:
    members = load_yaml(CONFIGS / "team.yaml")["members"]
    assert [m["id"] for m in members] == ["M1", "M2", "M3"]
    for member in members:
        assert set(member) == {"id", "name", "github"}


# ---------------------------------------------------------------- build_from_target

def test_build_from_target_builds_with_keyword_params() -> None:
    layer = build_from_target("torch.nn.Linear", {"in_features": 4, "out_features": 2})
    assert isinstance(layer, torch.nn.Linear)
    assert layer.weight.shape == (2, 4)


def test_build_from_target_errors_are_clear() -> None:
    with pytest.raises(ValueError):
        build_from_target("Linear", {})
    with pytest.raises(ImportError):
        build_from_target("har.models.does_not_exist.Model", {})
    with pytest.raises(ImportError):
        build_from_target("torch.nn.NoSuchLayer", {})


# ---------------------------------------------------------------- seeding

def _draw() -> tuple[float, np.ndarray, torch.Tensor, torch.Tensor]:
    return random.random(), np.random.rand(3), torch.rand(3), torch.nn.Linear(4, 2).weight.detach().clone()


def test_set_seed_makes_python_numpy_and_torch_reproducible() -> None:
    set_seed(42)
    a = _draw()
    set_seed(42)
    b = _draw()
    set_seed(43)
    c = _draw()
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    assert torch.equal(a[2], b[2])
    assert torch.equal(a[3], b[3])  # identical model initialisation
    assert not torch.equal(a[2], c[2])


def test_set_seed_deterministic_sets_cudnn_flags() -> None:
    set_seed(0, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_make_generator_is_seeded() -> None:
    assert torch.equal(torch.randperm(20, generator=make_generator(5)),
                       torch.randperm(20, generator=make_generator(5)))
    assert not torch.equal(torch.randperm(20, generator=make_generator(5)),
                           torch.randperm(20, generator=make_generator(6)))
