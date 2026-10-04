"""Config loading: ``configs/base.yaml`` (shared protocol) deep-merged with a model config.

The training protocol lives only in ``base.yaml``. A model config that sets the top-level
``train`` or ``seed`` keys is rejected with ``ProtocolError`` unless the override is
explicitly allowed (Phase 3 tuning only), and the resolved config records whether the
protocol was overridden so such runs can be excluded from the comparison table.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from har.utils.paths import CONFIGS

PROTOCOL_KEYS: tuple[str, ...] = ("train", "seed")


class ProtocolError(Exception):
    """A model config tried to change the shared training protocol."""


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: ``override`` merged into ``base``.

    Nested dicts are merged key by key; any other value in ``override`` replaces the base
    value. Neither input is modified.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML file whose top level is a mapping (an empty file gives ``{}``)."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}")
    return data


def load_config(
    model_cfg: Path | str,
    base_cfg: Path | str = CONFIGS / "base.yaml",
    allow_protocol_override: bool = False,
) -> dict[str, Any]:
    """Deep-merge ``base_cfg`` with ``model_cfg`` and return the resolved config.

    Raises ``ProtocolError`` if the model config has a top-level ``train`` or ``seed`` key
    and ``allow_protocol_override`` is False. The result has ``protocol_override: True``
    only when such keys were present (and allowed), otherwise ``False``.
    """
    base = load_yaml(base_cfg)
    model = load_yaml(model_cfg)
    overridden = [key for key in PROTOCOL_KEYS if key in model]
    if overridden and not allow_protocol_override:
        raise ProtocolError(
            f"{model_cfg} sets protocol key(s) {overridden}. The shared protocol lives only in "
            f"{base_cfg}; remove these keys, or pass --allow-protocol-override "
            "(Phase 3 tuning only; the run is then recorded as a protocol override)."
        )
    resolved = deep_merge(base, model)
    resolved["protocol_override"] = bool(overridden)
    return resolved
