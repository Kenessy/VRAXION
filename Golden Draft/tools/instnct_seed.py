"""Seeding and modular expert-head overrides (Golden Draft facade).

Golden Draft tools often run alongside the production "Golden Code" tree, but
they are not part of the end-user (DVD) package. This module provides a stable,
tool-friendly import surface while keeping the canonical implementation in
``vraxion.instnct.seed``.
"""

from __future__ import annotations

from vraxion.instnct.seed import (  # noqa: F401
    EXPERT_HEADS,
    _maybe_override_expert_heads,
    log,
    modular_auto_experts_enabled,
    set_seed,
)

__all__ = [
    "EXPERT_HEADS",
    "_maybe_override_expert_heads",
    "log",
    "modular_auto_experts_enabled",
    "set_seed",
]


