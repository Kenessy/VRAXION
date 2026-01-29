"""Runtime infrastructure helpers used by the VRAXION training runner.

Behavior-preserving extraction from the legacy monolithic training script.

This module is intentionally model-free. It contains utilities that are safe to
harden independently of the GRU / multi-circle sync implementation:

  - Logging: :func:`log`
  - Artifact housekeeping: :func:`rotate_artifacts`, :func:`sync_current_to_last`
  - Parsing helpers: :func:`_parse_csv_ints`, :func:`_parse_csv_floats`
  - Staircase batching: :class:`StaircaseController`, :class:`StaircaseBatcher`
  - NaN/Inf guard: :func:`nan_guard`
  - Misc: :func:`compute_slope`, checkpoint helpers

Contract (do not break):
  - Keep public names / signatures stable.
  - Keep env-driven overrides via module globals (ROOT, LOG_PATH, etc.).
"""

from __future__ import annotations

import math
import os
import random
import shutil
import time
import weakref
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# Minimal defaults (callers may override at runtime).
ROOT = os.getcwd()
LOG_PATH = os.path.join(ROOT, "logs", "current", "vraxion.log")

# Debug / runtime knobs.
DEBUG_NAN = False

# Window length for plateau detection; treated as 'unset' when falsy.
AGC_PLATEAU_WINDOW = 0

# Checkpoint payload defaults (callers may override at runtime).
USE_AMP = False
UPDATE_SCALE = 0.0005
PTR_INERTIA = 0.0
AGC_SCALE_MAX = 1.0
EXPERT_HEADS = 1

# torch.compile knobs (opt-in; default OFF).
TORCH_COMPILE = os.environ.get("VRX_TORCH_COMPILE", "0").strip() == "1"
TORCH_COMPILE_MODE = os.environ.get("VRX_TORCH_COMPILE_MODE", "default").strip()
TORCH_COMPILE_BACKEND = os.environ.get("VRX_TORCH_COMPILE_BACKEND", "").strip()
TORCH_COMPILE_FULLGRAPH = os.environ.get("VRX_TORCH_COMPILE_FULLGRAPH", "0").strip() == "1"
TORCH_COMPILE_DYNAMIC = os.environ.get("VRX_TORCH_COMPILE_DYNAMIC", "0").strip() == "1"
TORCH_COMPILE_FALLBACK = os.environ.get("VRX_TORCH_COMPILE_FALLBACK", "1").strip() == "1"

# Weak-cache: do NOT store compiled modules as nn.Module attributes (would register as submodules).
_TORCH_COMPILE_CACHE: "weakref.WeakKeyDictionary[nn.Module, Tuple[nn.Module, Tuple[Any, ...]]]" = (
    weakref.WeakKeyDictionary()
)


def log(msg: str) -> None:
    """Print a timestamped message and append it to ``LOG_PATH``."""

    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as filobj:
        filobj.write(line + "\n")


def maybe_torch_compile(module: nn.Module, label: str = "") -> nn.Module:
    """Return a compiled module when VRX_TORCH_COMPILE=1 (best-effort).

    This is intentionally conservative:
      - No behavior changes unless VRX_TORCH_COMPILE=1.
      - If compilation fails, it logs and returns the original module
        (unless VRX_TORCH_COMPILE_FALLBACK=0).

    Env knobs (strict "1" parsing):
      - VRX_TORCH_COMPILE=1
      - VRX_TORCH_COMPILE_MODE=default|reduce-overhead|max-autotune|...
      - VRX_TORCH_COMPILE_BACKEND=inductor|...
      - VRX_TORCH_COMPILE_FULLGRAPH=1
      - VRX_TORCH_COMPILE_DYNAMIC=1
      - VRX_TORCH_COMPILE_FALLBACK=0 (raise on errors instead of falling back)
    """

    if not TORCH_COMPILE:
        return module

    # Unwrap if we were passed an already-compiled module.
    base: nn.Module = module
    orig = getattr(module, "_orig_mod", None)
    if isinstance(orig, nn.Module):
        base = orig

    def _sig(mod: nn.Module) -> Tuple[str, str]:
        # Use the first param/buffer as a proxy for device/dtype.
        try:
            for p in mod.parameters():
                return str(p.device), str(p.dtype)
        except Exception:
            pass
        try:
            for b in mod.buffers():
                return str(b.device), str(b.dtype)
        except Exception:
            pass
        return "unknown", "unknown"

    devsig, dtysig = _sig(base)
    meta = (
        devsig,
        dtysig,
        TORCH_COMPILE_MODE,
        TORCH_COMPILE_BACKEND,
        bool(TORCH_COMPILE_FULLGRAPH),
        bool(TORCH_COMPILE_DYNAMIC),
    )

    cached = _TORCH_COMPILE_CACHE.get(base)
    if cached is not None and cached[1] == meta:
        return cached[0]

    if TORCH_COMPILE_FALLBACK:
        # When compilation fails at runtime (common when Triton isn't available),
        # PyTorch can fall back to eager instead of raising.
        try:
            import torch._dynamo  # type: ignore

            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass

    if not hasattr(torch, "compile"):
        msg = f"[torch.compile] requested{(' ' + label) if label else ''} but torch.compile is unavailable"
        if TORCH_COMPILE_FALLBACK:
            log(msg + " (continuing eager)")
            return module
        raise RuntimeError(msg)

    backend_name = TORCH_COMPILE_BACKEND.strip() or "inductor"
    if devsig.startswith("cuda") and backend_name == "inductor":
        # Inductor on CUDA requires Triton; if Triton is missing, compilation
        # will fail at first call.
        try:
            import triton  # type: ignore  # noqa: F401
        except Exception:
            msg = (
                f"[torch.compile] requested{(' ' + label) if label else ''} but Triton is unavailable "
                f"(skipping compile; install triton or set VRX_TORCH_COMPILE_BACKEND=aot_eager)"
            )
            if TORCH_COMPILE_FALLBACK:
                log(msg)
                return module
            raise RuntimeError(msg)

    kwargs: dict = {"mode": TORCH_COMPILE_MODE or "default"}
    if TORCH_COMPILE_BACKEND:
        kwargs["backend"] = TORCH_COMPILE_BACKEND
    if TORCH_COMPILE_FULLGRAPH:
        kwargs["fullgraph"] = True
    if TORCH_COMPILE_DYNAMIC:
        kwargs["dynamic"] = True

    try:
        compiled = torch.compile(base, **kwargs)
    except Exception as exc:
        msg = f"[torch.compile] failed{(' ' + label) if label else ''}: {exc}"
        if TORCH_COMPILE_FALLBACK:
            log(msg)
            return module
        raise

    _TORCH_COMPILE_CACHE[base] = (compiled, meta)
    log(
        f"[torch.compile] enabled{(' ' + label) if label else ''} "
        f"mode={kwargs.get('mode')} backend={kwargs.get('backend', 'default')} "
        f"fullgraph={bool(TORCH_COMPILE_FULLGRAPH)} dynamic={bool(TORCH_COMPILE_DYNAMIC)} "
        f"sig=({devsig},{dtysig})"
    )
    return compiled


def rotate_artifacts() -> None:
    """Move current logs/traces/summaries -> last, and last -> archive/<timestamp>."""

    ts = time.strftime("%Y%m%d_%H%M%S")

    def _rotate_dir(base_dir: str) -> None:
        curdir = os.path.join(base_dir, "current")
        lasdir = os.path.join(base_dir, "last")
        arcdir = os.path.join(base_dir, "archive")

        os.makedirs(curdir, exist_ok=True)
        os.makedirs(lasdir, exist_ok=True)
        os.makedirs(arcdir, exist_ok=True)

        # last -> archive/<ts>
        if os.path.isdir(lasdir) and os.listdir(lasdir):
            run_dir = os.path.join(arcdir, ts)
            os.makedirs(run_dir, exist_ok=True)
            for name in os.listdir(lasdir):
                shutil.move(os.path.join(lasdir, name), os.path.join(run_dir, name))

        # current -> last
        if os.path.isdir(curdir) and os.listdir(curdir):
            for name in os.listdir(curdir):
                shutil.move(os.path.join(curdir, name), os.path.join(lasdir, name))

    _rotate_dir(os.path.join(ROOT, "logs"))
    _rotate_dir(os.path.join(ROOT, "traces"))
    _rotate_dir(os.path.join(ROOT, "summaries"))


def sync_current_to_last() -> None:
    """Copy current logs/traces/summaries into ``logs/last``."""

    dstdir = os.path.join(ROOT, "logs", "last")
    os.makedirs(dstdir, exist_ok=True)

    for rel in ("logs/current", "traces/current", "summaries/current"):
        srcdir = os.path.join(ROOT, rel)
        if not os.path.isdir(srcdir):
            continue
        for name in os.listdir(srcdir):
            srcpth = os.path.join(srcdir, name)
            if os.path.isfile(srcpth):
                shutil.copy2(srcpth, os.path.join(dstdir, name))


def _parse_csv_ints(raw: str) -> Optional[List[int]]:
    """Parse a comma-separated list of positive ints.

    Returns None on empty input, parse failure, or if any value is <= 0.
    """

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    items: List[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            return None
        if value <= 0:
            return None
        items.append(value)
    return items or None


def _parse_csv_floats(raw: str) -> Optional[List[float]]:
    """Parse a comma-separated list of floats.

    Returns None on empty input or parse failure.
    """

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    items: List[float] = []
    for part in parts:
        try:
            value = float(part)
        except ValueError:
            return None
        items.append(value)
    return items or None


def _normalize_weights(weights: Sequence[float]) -> List[float]:
    """Normalize non-negative weights to sum to 1.

    Negative weights are treated as 0.0. If the total is <= 0.0, returns a
    uniform distribution of the same length.
    """

    total = sum(max(0.0, float(w)) for w in weights)
    if total <= 0.0:
        return [1.0 / len(weights)] * len(weights)
    return [max(0.0, float(w)) / total for w in weights]


def _default_staircase_weights(lens: Sequence[int]) -> List[float]:
    """Default staircase weights for common context lengths."""

    if len(lens) == 3 and lens[0] == 512 and lens[1] == 768 and lens[2] == 1024:
        return [0.95, 0.04, 0.01]
    return [1.0 / len(lens)] * len(lens)


class StaircaseController:
    """Adaptive weight shifter for :class:`StaircaseBatcher`."""

    def __init__(
        self,
        lens: Sequence[int],
        weights: Sequence[float],
        min_base: float,
        shift: float,
        stable_std: float,
        adapt_every: int,
    ) -> None:
        self.lens = list(lens)
        self.weights = _normalize_weights(weights)
        self.min_base = float(min_base)
        self.shift = float(shift)
        self.stable_std = float(stable_std)
        self.adapt_every = int(adapt_every)

    def set_weights(self, weights: Sequence[float]) -> None:
        self.weights = _normalize_weights(weights)

    def maybe_adapt(self, losses: Sequence[float], step: int) -> Optional[List[float]]:
        """Shift weights if (a) it's time, and (b) recent losses are stable."""

        if self.adapt_every <= 0 or step <= 0 or step % self.adapt_every != 0:
            return None
        if not losses:
            return None

        winlen = max(10, AGC_PLATEAU_WINDOW or 50)
        window = min(len(losses), winlen)
        recent = losses[-window:]

        meanvl = sum(recent) / len(recent)
        varval = sum((x - meanvl) ** 2 for x in recent) / len(recent)
        stdval = math.sqrt(varval)
        if not math.isfinite(stdval) or stdval > self.stable_std:
            return None

        if len(self.weights) < 2:
            return None

        basval = self.weights[0]
        if basval <= self.min_base:
            return None

        deltav = min(self.shift, basval - self.min_base)
        if deltav <= 0.0:
            return None

        peroth = deltav / (len(self.weights) - 1)
        newwts = [basval - deltav]
        for idx in range(1, len(self.weights)):
            newwts.append(self.weights[idx] + peroth)

        newwts = _normalize_weights(newwts)
        self.weights = newwts
        log(f"[staircase] shift base by {deltav:.3f} -> weights={newwts}")
        return newwts


class StaircaseBatcher:
    """Weighted iterator over multiple loaders."""

    def __init__(
        self,
        loaders: Sequence[Any],
        weights: Sequence[float],
        rng_seed: int,
        staircase: Optional[StaircaseController] = None,
    ) -> None:
        self.loaders = list(loaders)
        self.weights = _normalize_weights(weights)
        self._iters = [iter(loader) for loader in self.loaders]
        self._rng = random.Random(rng_seed)
        self.dataset = self.loaders[0].dataset if self.loaders else None
        self.staircase = staircase

    def set_weights(self, weights: Sequence[float]) -> None:
        self.weights = _normalize_weights(weights)
        if self.staircase is not None:
            self.staircase.set_weights(self.weights)

    def __iter__(self):
        return self

    def _pick_index(self) -> int:
        r = self._rng.random()
        acc = 0.0
        for idx, weight in enumerate(self.weights):
            acc += weight
            if r <= acc:
                return idx
        return len(self.weights) - 1

    def __next__(self):
        if not self.loaders:
            raise StopIteration
        idx = self._pick_index()
        try:
            return next(self._iters[idx])
        except StopIteration:
            self._iters[idx] = iter(self.loaders[idx])
            return next(self._iters[idx])


def nan_guard(name: str, tensor: torch.Tensor, step: int) -> None:
    """Raise if tensor contains NaN/Inf (only when ``DEBUG_NAN`` is True)."""

    if not DEBUG_NAN:
        return
    if not tensor.is_floating_point():
        return
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        log(f"[nan_guard] step={step:04d} tensor={name} has NaN/Inf")
        raise RuntimeError(f"NaN/Inf in {name} at step {step}")


def compute_slope(losses: Sequence[float]) -> float:
    """Compute a least-squares slope over the loss history."""

    if len(losses) < 2:
        return float("nan")

    num = float(len(losses))
    sum_x = float(len(losses) - 1) * num / 2.0
    sum_x2 = float(len(losses) - 1) * num * float(2 * len(losses) - 1) / 6.0

    sum_y = 0.0
    sum_xy = 0.0
    for idx, val in enumerate(losses):
        y = float(val)
        sum_y += y
        sum_xy += float(idx) * y

    denom = num * sum_x2 - sum_x * sum_x
    if denom == 0.0:
        return float("nan")
    return (num * sum_xy - sum_x * sum_y) / denom


def _checkpoint_payload(model: Any, optimizer: Any, scaler: Any, step: int, losses: List[float]) -> dict:
    """Build a checkpoint payload dict compatible with the Golden runtime."""

    return {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scaler": scaler.state_dict() if USE_AMP else None,
        "step": step,
        "losses": losses,
        "update_scale": getattr(model, "update_scale", UPDATE_SCALE),
        "ptr_inertia": getattr(model, "ptr_inertia", PTR_INERTIA),
        "ptr_inertia_ema": getattr(model, "ptr_inertia_ema", getattr(model, "ptr_inertia", PTR_INERTIA)),
        "ptr_inertia_floor": getattr(model, "ptr_inertia_floor", 0.0),
        "agc_scale_max": getattr(model, "agc_scale_max", AGC_SCALE_MAX),
        "ground_speed_ema": getattr(model, "ground_speed_ema", None),
        "ground_speed_limit": getattr(model, "ground_speed_limit", None),
        "num_experts": getattr(getattr(model, "head", None), "num_experts", EXPERT_HEADS),
        "param_names": [name for name, _ in model.named_parameters()],
    }


def _checkpoint_paths(base_path: str, step: int) -> Tuple[str, str, str]:
    """Return (step_path, bad_path, last_good_path) for a checkpoint base path."""

    base_dir = os.path.dirname(base_path) or ROOT
    base_name = os.path.splitext(os.path.basename(base_path))[0]

    hisdir = os.path.join(base_dir, "checkpoints")
    os.makedirs(hisdir, exist_ok=True)

    step_path = os.path.join(hisdir, f"{base_name}_step_{step:07d}.pt")
    bad_path = os.path.join(hisdir, f"{base_name}_bad_step_{step:07d}.pt")
    last_good_path = os.path.join(base_dir, f"{base_name}_last_good.pt")
    return step_path, bad_path, last_good_path


def _checkpoint_is_finite(loss_value: Any, grad_norm_value: Any, raw_delta_value: Any) -> bool:
    """Return False if any provided scalar is non-finite."""

    for value in (loss_value, grad_norm_value, raw_delta_value):
        if value is None:
            continue
        if not math.isfinite(float(value)):
            return False
    return True

