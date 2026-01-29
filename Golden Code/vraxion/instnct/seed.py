"""Deterministic seeding and modular expert-head overrides.

This module centralizes two small helpers that used to live in the legacy
monolith:
  - set_seed(seed): best-effort deterministic seeding for Python/random, NumPy,
    and PyTorch.
  - _maybe_override_expert_heads(resume_path): when VRX_MODULAR_AUTO_EXPERTS is
    enabled (strictly "1"), read <modular_dir>/system/router.state and override
    EXPERT_HEADS from its "num_experts" field.

Constraints / semantics:
  - ASCII-only logs.
  - VRX_MODULAR_AUTO_EXPERTS enables only when the env var value is exactly "1".
  - router.state is loaded via torch.load(..., map_location="cpu").
"""

from __future__ import annotations

import os
import random
from typing import Any, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

import torch


def log(msg: str) -> None:
    """Best-effort logger (monkeypatch-friendly)."""

    try:
        print(msg, flush=True)
    except Exception:
        # Logging must never crash training.
        pass


def _read_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        log(f"WARN: invalid {name}={raw!r}; using default {default}")
        return default


# Runtime global. Callers may overwrite this symbol at runtime.
EXPERT_HEADS: int = max(1, _read_env_int("VRX_EXPERT_HEADS", 1))


def modular_auto_experts_enabled(env: Optional[dict[str, str]] = None) -> bool:
    """Return True iff VRX_MODULAR_AUTO_EXPERTS is exactly "1"."""

    envmap = env if env is not None else os.environ
    return envmap.get("VRX_MODULAR_AUTO_EXPERTS", "0") == "1"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs deterministically (best-effort)."""

    seed_int = int(seed)
    seed_u32 = seed_int % (2**32)

    # Best-effort: helps child processes spawned after this call.
    os.environ.setdefault("PYTHONHASHSEED", str(seed_u32))

    random.seed(seed_u32)
    if np is not None:
        np.random.seed(seed_u32)

    torch.manual_seed(seed_u32)

    # Guard CUDA calls for CPU-only builds.
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_u32)
    except Exception:
        pass

    # Common determinism knobs (best-effort).
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _torch_load_cpu(path: str) -> Any:
    """Load a torch payload onto CPU (best-effort, version compatible)."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch does not accept weights_only.
        return torch.load(path, map_location="cpu")
    except Exception:
        # weights_only may reject certain objects; fall back to legacy behavior.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _resolve_modular_resume_dir(resume_path: str) -> Optional[str]:
    """Resolve the modular checkpoint directory for a resume path.

    Supported shapes:
      - <modular_dir>/
      - <modular_dir>/system/router.state
      - <something>.pt  (then try <something>_modular/)
      - a file nested under the modular dir (bounded parent walk)
    """

    if not resume_path:
        return None

    pth = os.path.abspath(os.path.expanduser(resume_path))

    # Direct router.state path.
    if os.path.basename(pth) == "router.state" and os.path.basename(os.path.dirname(pth)) == "system":
        cand = os.path.dirname(os.path.dirname(pth))
        if cand and os.path.isdir(cand):
            return cand

    # Direct modular dir path.
    if os.path.isdir(pth):
        router = os.path.join(pth, "system", "router.state")
        if os.path.exists(router):
            return pth

    # Check common ".pt -> _modular" layout.
    if os.path.isfile(pth) and pth.endswith(".pt"):
        cand = os.path.splitext(pth)[0] + "_modular"
        router = os.path.join(cand, "system", "router.state")
        if os.path.isdir(cand) and os.path.exists(router):
            return cand

    # Bounded walk up from a directory anchor.
    cur = pth if os.path.isdir(pth) else os.path.dirname(pth)
    for _ in range(10):
        if not cur:
            break
        router = os.path.join(cur, "system", "router.state")
        if os.path.exists(router):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    return None


def _maybe_override_expert_heads(resume_path: str) -> None:
    """If enabled, override EXPERT_HEADS from modular router.state."""

    if not resume_path:
        return

    if not modular_auto_experts_enabled():
        return

    modular_dir = _resolve_modular_resume_dir(resume_path)
    if not modular_dir:
        log(f"WARN: modular router.state not found for resume_path={resume_path}")
        return

    router_path = os.path.join(modular_dir, "system", "router.state")
    if not os.path.exists(router_path):
        log(f"WARN: modular router.state missing: {router_path}")
        return

    try:
        ckpt = _torch_load_cpu(router_path)
    except Exception as exc:
        log(f"Modular auto-expert read failed: {exc}")
        return

    if not isinstance(ckpt, dict):
        log(f"WARN: modular router.state unexpected type: {type(ckpt).__name__}")
        return

    num_experts = ckpt.get("num_experts")
    if num_experts is None:
        return

    try:
        n = int(num_experts)
    except Exception:
        log(f"WARN: modular router.state num_experts invalid: {num_experts!r}")
        return

    if n <= 0:
        return

    global EXPERT_HEADS
    EXPERT_HEADS = n

    # Keep env in sync for code paths that re-read VRX_EXPERT_HEADS.
    os.environ["VRX_EXPERT_HEADS"] = str(EXPERT_HEADS)

    # Preserve legacy log shape.
    log(f"[modular] auto override EXPERT_HEADS={EXPERT_HEADS} from {router_path}")


