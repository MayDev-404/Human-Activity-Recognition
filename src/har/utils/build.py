"""Build objects from dotted import paths, so models need no shared registry file."""

from __future__ import annotations

import importlib
from typing import Any


def build_from_target(target: str, params: dict | None = None) -> Any:
    """Import ``"pkg.module.ClassName"`` and call it with ``params`` as keyword arguments.

    Example: ``build_from_target("har.models.mlp.MLP", {"input_dim": 561, ...})``.
    Raises ``ValueError`` for a target without a module part and ``ImportError`` if the
    module or attribute cannot be found.
    """
    module_name, _, attr_name = target.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"target must look like 'package.module.ClassName', got {target!r}")
    module = importlib.import_module(module_name)
    try:
        factory = getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(f"module {module_name!r} has no attribute {attr_name!r}") from exc
    return factory(**(params or {}))
