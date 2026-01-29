"""
train_steps extraction.

This file is a behavior-preserving extraction of the legacy bounded
training loop (train_steps). It intentionally avoids new dependencies and
leans on the canonical vraxion helpers when they are available.

Notes:
- Settings are loaded inside train_steps() so tests and callers can override
  behavior via environment variables.
- Environment flags are parsed with strict "1" semantics (e.g., "1" enables,
  anything else disables) where relevant.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Optional imports from the real repo. When running in isolation (unit tests),
# we provide minimal fallbacks that keep the control-flow intact.
# ---------------------------------------------------------------------------

try:
    from vraxion.settings import load_settings  # type: ignore
except Exception:  # pragma: no cover
    def load_settings() -> Any:  # type: ignore
        return {}


try:
    from vraxion.instnct.infra import log, compute_slope, _checkpoint_paths  # type: ignore
except Exception:  # pragma: no cover

    def log(msg: str) -> None:
        # Keep logs ASCII-only.
        print(str(msg))

    def compute_slope(losses: list) -> Optional[float]:
        # Simple, stable slope estimate as a fallback.
        if not losses or len(losses) < 2:
            return None
        y0 = float(losses[0])
        y1 = float(losses[-1])
        return (y1 - y0) / float(max(1, len(losses) - 1))

    def _checkpoint_paths(checkpoint_path: str, step: int) -> Tuple[str, str, str]:
        base, ext = os.path.splitext(checkpoint_path)
        step_path = f"{base}_step_{step:04d}{ext or '.pt'}"
        bad_path = f"{base}_bad_{step:04d}{ext or '.pt'}"
        last_good_path = f"{base}_last_good{ext or '.pt'}"
        return step_path, bad_path, last_good_path


try:
    from vraxion.instnct.modular_checkpoint import (  # type: ignore
        _save_modular_checkpoint,
        _load_modular_checkpoint,
        _resolve_modular_dir,
        _resolve_modular_resume_dir,
    )
except Exception:  # pragma: no cover

    def _save_modular_checkpoint(*args: Any, **kwargs: Any) -> None:
        return None

    def _load_modular_checkpoint(*args: Any, **kwargs: Any) -> None:
        return None

    def _resolve_modular_dir(modular_dir: str, root: str, checkpoint_path: str) -> str:
        # Minimal behavior: treat modular_dir as relative to root when not absolute.
        if os.path.isabs(modular_dir):
            return modular_dir
        return os.path.join(root, modular_dir)

    def _resolve_modular_resume_dir(*args: Any, **kwargs: Any) -> None:
        return None


try:
    from vraxion.instnct.thermo import ThermostatParams, apply_thermostat  # type: ignore
except Exception:  # pragma: no cover

    class ThermostatParams:  # noqa: D401 - placeholder
        """Fallback ThermostatParams placeholder."""
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    def apply_thermostat(  # type: ignore
        model: Any,
        flip_rate: float,
        flip_ema: Optional[float],
        params: Any,
        focus: Any = None,
        tension: Any = None,
        raw_delta: Any = None,
    ) -> Optional[float]:
        return flip_ema


try:
    from vraxion.instnct.agc import AGCParams, apply_update_agc  # type: ignore
except Exception:  # pragma: no cover

    class AGCParams:  # noqa: D401 - placeholder
        """Fallback AGCParams placeholder."""
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    def apply_update_agc(  # type: ignore
        model: Any,
        grad_norm_theta_ptr: Optional[float],
        params: Any,
        raw_delta: Any = None,
        step: int = 0,
        log_fn: Any = None,
    ) -> Optional[float]:
        return None


try:
    from vraxion.instnct.inertia_auto import InertiaAutoParams, apply_inertia_auto  # type: ignore
except Exception:  # pragma: no cover

    class InertiaAutoParams:  # noqa: D401 - placeholder
        """Fallback InertiaAutoParams placeholder."""
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    def apply_inertia_auto(  # type: ignore
        model: Any,
        ptr_velocity_raw: Any,
        params: Any,
        panic_active: bool = False,
    ) -> None:
        return None


try:
    from vraxion.instnct.panic import PanicReflex  # type: ignore
except Exception:  # pragma: no cover

    class PanicReflex:
        def __init__(
            self,
            ema_beta: float = 0.95,
            panic_threshold: float = 1.0,
            recovery_rate: float = 0.01,
            inertia_low: float = 0.0,
            inertia_high: float = 1.0,
            walk_prob_max: float = 0.0,
        ) -> None:
            self.ema_beta = ema_beta
            self.panic_threshold = panic_threshold
            self.recovery_rate = recovery_rate
            self.inertia_low = inertia_low
            self.inertia_high = inertia_high
            self.walk_prob_max = walk_prob_max
            self._ema: Optional[float] = None

        def update(self, loss_val: float) -> Dict[str, Any]:
            loss_val = float(loss_val)
            if self._ema is None:
                self._ema = loss_val
            else:
                self._ema = self.ema_beta * self._ema + (1.0 - self.ema_beta) * loss_val
            status = "PANIC" if self._ema >= self.panic_threshold else "OK"
            inertia = self.inertia_low if status == "PANIC" else self.inertia_high
            walk = self.walk_prob_max if status == "PANIC" else 0.0
            return {"status": status, "inertia": inertia, "walk_prob": walk}


try:
    from vraxion.instnct.cadence import CadenceGovernor  # type: ignore
except Exception:  # pragma: no cover

    class CadenceGovernor:
        def __init__(self, start_tau: float = 1.0, **kwargs: Any) -> None:
            try:
                self._tau = int(round(float(start_tau)))
            except Exception:
                self._tau = 1
            self._tau = max(1, self._tau)

        def update(
            self,
            loss_val: float,
            grad_norm: float,
            flip_rate: float,
            ptr_velocity: Any,
        ) -> int:
            return self._tau


try:
    from vraxion.instnct.vcog import VCogGovernor  # type: ignore
except Exception:  # pragma: no cover

    class VCogGovernor:
        def __init__(self, id_target: float = 0.0, sigma_floor: float = 0.0) -> None:
            self.id_target = id_target
            self.sigma_floor = sigma_floor

        def update(self, payload: Dict[str, Any]) -> str:
            # Minimal stable header.
            _ = payload
            return "VCOG"


# ---------------------------------------------------------------------------
# Local helpers for extraction completeness (lightweight, no new deps).
# ---------------------------------------------------------------------------


def _env_flag_strict_1(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return val == "1"


def _get_setting(settings: Any, name: str, default: Any = None) -> Any:
    # Support dict-like and attribute-like settings containers.
    if settings is None:
        return default
    if isinstance(settings, dict):
        return settings.get(name, default)
    if hasattr(settings, name):
        return getattr(settings, name)
    # A few callers use lower-case keys; try that too.
    low = name.lower()
    if isinstance(settings, dict):
        return settings.get(low, default)
    if hasattr(settings, low):
        return getattr(settings, low)
    return default


_DTYPE_ALIASES: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "double": torch.float64,
}


def _resolve_dtype(val: Any, default: torch.dtype) -> torch.dtype:
    if isinstance(val, torch.dtype):
        return val
    if isinstance(val, str):
        key = val.strip().lower()
        return _DTYPE_ALIASES.get(key, default)
    return default


def _checkpoint_is_finite(
    loss_value: Optional[float],
    grad_value: Optional[float],
    raw_delta_value: Optional[float],
) -> bool:
    for x in (loss_value, grad_value, raw_delta_value):
        if x is None:
            continue
        try:
            if not math.isfinite(float(x)):
                return False
        except Exception:
            return False
    return True


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    losses: list,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "losses": list(losses),
    }
    try:
        payload["scaler"] = scaler.state_dict() if scaler is not None else None
    except Exception:
        payload["scaler"] = None
    return payload


def _update_expert_usage(model: Any, num_experts: int, step: int) -> None:
    # Best-effort lightweight counter update.
    device = None
    try:
        device = next(model.parameters()).device  # type: ignore[arg-type]
    except Exception:
        device = torch.device("cpu")
    counts = getattr(model, "ptr_expert_counts", None)
    if not torch.is_tensor(counts) or counts.numel() != int(num_experts):
        counts = torch.zeros(int(num_experts), device=device)
        model.ptr_expert_counts = counts
    # Prefer router_map if present (common in expert routing setups).
    router_map = getattr(model, "router_map", None)
    if torch.is_tensor(router_map):
        idx = router_map.detach().to(torch.long).flatten()
        valid = (idx >= 0) & (idx < int(num_experts))
        if bool(valid.any()):
            binc = torch.bincount(idx[valid], minlength=int(num_experts)).to(counts.device)
            model.ptr_expert_counts += binc
            model.ptr_expert_counts_step = int(step)
            return
    active = getattr(model, "ptr_expert_active", None)
    if torch.is_tensor(active):
        idx = active.detach().to(torch.long).flatten()
        valid = (idx >= 0) & (idx < int(num_experts))
        if bool(valid.any()):
            binc = torch.bincount(idx[valid], minlength=int(num_experts)).to(counts.device)
            model.ptr_expert_counts += binc
            model.ptr_expert_counts_step = int(step)
            return
    if isinstance(active, (list, tuple, set)):
        for a in active:
            try:
                i = int(a)
            except Exception:
                continue
            if 0 <= i < int(num_experts):
                model.ptr_expert_counts[i] += 1
        model.ptr_expert_counts_step = int(step)
        return
    # No reliable signal; still stamp step for debuggability.
    model.ptr_expert_counts_step = int(step)


def _compute_expert_similarity_stats(model: Any, sim_thresh: float) -> Optional[Tuple[float, int, Tuple[int, int]]]:
    # Placeholder: similarity pruning is repo-specific; return None when unknown.
    _ = (model, sim_thresh)
    return None


def _resolve_hibernate_dir(hibernate_dir: str, root: str) -> str:
    if os.path.isabs(hibernate_dir):
        return hibernate_dir
    return os.path.join(root, hibernate_dir)


def _extract_expert_module(head: Any, idx: int) -> Optional[torch.nn.Module]:
    if head is None:
        return None
    # Common attribute names.
    for attr in ("experts", "expert_modules", "expert_heads", "heads"):
        if hasattr(head, attr):
            cand = getattr(head, attr)
            try:
                mod = cand[idx]
                if isinstance(mod, torch.nn.Module):
                    return mod
            except Exception:
                pass
    # Optional hook.
    if hasattr(head, "get_expert"):
        try:
            mod = head.get_expert(idx)
            if isinstance(mod, torch.nn.Module):
                return mod
        except Exception:
            pass
    return None


def _extract_expert_state(head: Any, idx: int) -> Optional[Dict[str, Any]]:
    mod = _extract_expert_module(head, idx)
    if mod is None:
        return None
    try:
        return mod.state_dict()
    except Exception:
        return None


def _hash_state_dict(state: Dict[str, Any]) -> Optional[str]:
    # Stable digest for simple drift detection.
    try:
        import hashlib
        h = hashlib.sha256()
        # Sort keys for stability.
        for k in sorted(state.keys()):
            v = state[k]
            if torch.is_tensor(v):
                h.update(k.encode("utf-8", errors="ignore"))
                h.update(v.detach().cpu().numpy().tobytes())
            else:
                h.update(k.encode("utf-8", errors="ignore"))
                h.update(repr(v).encode("utf-8", errors="ignore"))
        return h.hexdigest()
    except Exception:
        return None


def _save_expert_snapshot(state: Dict[str, Any], path: str) -> Optional[str]:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        return _hash_state_dict(state)
    except Exception:
        return None


def _zero_expert_weights(head: Any, idx: int) -> bool:
    mod = _extract_expert_module(head, idx)
    if mod is None:
        return False
    try:
        with torch.no_grad():
            for p in mod.parameters():
                p.zero_()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_steps(model: torch.nn.Module, loader: Any, steps: int, dataset_name: str, model_name: str) -> Dict[str, Any]:
    # Load settings inside the function so env overrides take effect in tests and callers.
    #
    # NOTE: The legacy monolith defined many knobs as module-level globals
    # sourced from both `vraxion.settings.load_settings()` and direct env reads.
    # This extraction preserves that split to avoid accidental behavior drift.
    settings = load_settings()

    # ---- Canonical settings (vraxion.settings) ----
    ROOT = str(_get_setting(settings, "root", os.getcwd()))
    CHECKPOINT_PATH = str(_get_setting(settings, "checkpoint_path", os.path.join(ROOT, "checkpoint.pt")))

    DEVICE = str(_get_setting(settings, "device", "cuda" if torch.cuda.is_available() else "cpu")).strip().lower()
    if DEVICE == "cuda" and not torch.cuda.is_available():
        DEVICE = "cpu"

    DTYPE = _resolve_dtype(_get_setting(settings, "dtype", None), torch.float32)
    USE_AMP = bool(_get_setting(settings, "use_amp", False))

    LR = float(_get_setting(settings, "lr", 1e-3))
    LAMBDA_MOVE = float(_get_setting(settings, "lambda_move", 0.0))
    GRAD_CLIP = float(_get_setting(settings, "grad_clip", 0.0))
    UPDATE_SCALE = float(_get_setting(settings, "update_scale", 1.0))

    HEARTBEAT_STEPS = int(_get_setting(settings, "heartbeat_steps", 50))
    HEARTBEAT_SECS = float(_get_setting(settings, "heartbeat_secs", 0.0))
    DEBUG_STATS = bool(_get_setting(settings, "debug_stats", False))

    SAVE_EVERY_STEPS = int(_get_setting(settings, "save_every_steps", 0))
    SAVE_HISTORY = bool(_get_setting(settings, "save_history", False))
    SAVE_BAD = bool(_get_setting(settings, "save_bad", False))
    SAVE_LAST_GOOD = bool(_get_setting(settings, "save_last_good", False))
    EVO_CKPT_INDIV = bool(_get_setting(settings, "evo_checkpoint_individual", True))

    # Training trace: strict env parsing is already handled by load_settings().
    TRAIN_TRACE = bool(_get_setting(settings, "train_trace", False))
    TRAIN_TRACE_PATH = str(
        _get_setting(
            settings,
            "train_trace_path",
            os.path.join(ROOT, "traces", "current", "train_steps_trace.jsonl"),
        )
    )

    # Controls from canonical settings (VRX_* env -> Settings -> fields).
    THERMO_ENABLED = bool(_get_setting(settings, "thermo_enabled", False))
    THERMO_EVERY = int(_get_setting(settings, "thermo_every", 20))

    PANIC_ENABLED = bool(_get_setting(settings, "panic_enabled", False))
    PANIC_BETA = float(_get_setting(settings, "panic_beta", 0.95))
    PANIC_THRESHOLD = float(_get_setting(settings, "panic_threshold", 1.0))
    PANIC_RECOVERY = float(_get_setting(settings, "panic_recovery", 0.01))
    PANIC_INERTIA_LOW = float(_get_setting(settings, "panic_inertia_low", 0.0))
    PANIC_INERTIA_HIGH = float(_get_setting(settings, "panic_inertia_high", 1.0))
    PANIC_WALK_MAX = float(_get_setting(settings, "panic_walk_max", 0.0))

    PTR_UPDATE_EVERY = int(_get_setting(settings, "ptr_update_every", 1))
    PTR_UPDATE_AUTO = bool(_get_setting(settings, "ptr_update_auto", True))
    PTR_UPDATE_MIN = int(_get_setting(settings, "ptr_update_min", 1))
    PTR_UPDATE_MAX = int(_get_setting(settings, "ptr_update_max", max(PTR_UPDATE_MIN, PTR_UPDATE_EVERY)))
    PTR_UPDATE_EMA = float(_get_setting(settings, "ptr_update_ema", 0.95))
    PTR_UPDATE_TARGET_FLIP = float(_get_setting(settings, "ptr_update_target_flip", 0.0))

    # Name differs from legacy: Settings uses ptr_update_governor.
    PTR_UPDATE_GOV = bool(_get_setting(settings, "ptr_update_governor", False))
    PTR_UPDATE_GOV_WARMUP = int(_get_setting(settings, "ptr_update_gov_warmup", 0))
    PTR_UPDATE_GOV_GRAD_HIGH = float(_get_setting(settings, "ptr_update_gov_grad_high", 0.0))
    PTR_UPDATE_GOV_GRAD_LOW = float(_get_setting(settings, "ptr_update_gov_grad_low", 0.0))
    PTR_UPDATE_GOV_LOSS_FLAT = float(_get_setting(settings, "ptr_update_gov_loss_flat", 0.0))
    PTR_UPDATE_GOV_LOSS_SPIKE = float(_get_setting(settings, "ptr_update_gov_loss_spike", 0.0))
    PTR_UPDATE_GOV_STEP_UP = int(_get_setting(settings, "ptr_update_gov_step_up", 0))
    PTR_UPDATE_GOV_STEP_DOWN = int(_get_setting(settings, "ptr_update_gov_step_down", 0))
    PTR_UPDATE_GOV_VEL_HIGH = float(_get_setting(settings, "ptr_update_gov_vel_high", 0.5))

    # ---- Env-based knobs (preserved from legacy monolithic script) ----
    DISABLE_SYNC = os.environ.get("VRX_DISABLE_SYNC", "0") == "1"
    LOSS_EMA_BETA = float(os.environ.get("VRX_LOSS_EMA_BETA", "0.95"))

    VCOG_ID_TARGET = float(os.environ.get("VRX_VCOG_ID_TARGET", "0.25"))
    VCOG_SIGMA_FLOOR = float(os.environ.get("VRX_VCOG_SIGMA_FLOOR", "1e-4"))

    METABOLIC_TELEMETRY = os.environ.get("VRX_METABOLIC_TELEMETRY", "0") == "1"
    METABOLIC_HUNGER = os.environ.get("VRX_METABOLIC_HUNGER", "0") == "1"
    METABOLIC_COST_COEFF = float(os.environ.get("VRX_METABOLIC_COST_COEFF", "0.0001"))
    METABOLIC_SIM_THRESH = float(os.environ.get("VRX_METABOLIC_SIM_THRESH", "0.98"))
    METABOLIC_EVERY = int(os.environ.get("VRX_METABOLIC_EVERY", "500"))
    METABOLIC_IDLE_STEPS = int(os.environ.get("VRX_METABOLIC_IDLE_STEPS", "2000"))

    HIBERNATE_ENABLED = os.environ.get("VRX_HIBERNATE", "0") == "1"
    HIBERNATE_MODE = os.environ.get("VRX_HIBERNATE_MODE", "shadow").strip().lower()
    HIBERNATE_IDLE_STEPS = int(os.environ.get("VRX_HIBERNATE_IDLE_STEPS", str(METABOLIC_IDLE_STEPS)))
    HIBERNATE_EVERY = int(os.environ.get("VRX_HIBERNATE_EVERY", "50"))
    HIBERNATE_DIR = os.environ.get("VRX_HIBERNATE_DIR", "hibernation")

    INERTIA_SIGNAL_ENABLED = os.environ.get("VRX_INERTIA_SIGNAL", "0") == "1"
    INERTIA_SIGNAL_REWARD = float(os.environ.get("VRX_INERTIA_SIGNAL_REWARD", "0.1"))

    # Staircase curriculum is enabled when the lens list is non-empty.
    STAIRCASE_LENS_RAW = os.environ.get("VRX_STAIRCASE_LENS", "").strip()
    STAIRCASE_ENABLED = bool(STAIRCASE_LENS_RAW)
    STAIRCASE_ADAPT = os.environ.get("VRX_STAIRCASE_ADAPT", "0") == "1"

    # Expert routing knobs.
    EXPERT_HEADS = max(1, int(os.environ.get("VRX_EXPERT_HEADS", "1")))
    EXPERT_BUDGET = int(os.environ.get("VRX_EXPERT_BUDGET", "0"))

    USAGE_LAMBDA_INIT = float(os.environ.get("VRX_USAGE_LAMBDA", "0.0"))
    USAGE_LAMBDA_MAX = float(os.environ.get("VRX_USAGE_LAMBDA_MAX", "0.5"))
    USAGE_LAMBDA_ETA = float(os.environ.get("VRX_USAGE_LAMBDA_ETA", "0.05"))
    USAGE_GATE_CONF = float(os.environ.get("VRX_USAGE_GATE_CONF", "0.95"))
    USAGE_GATE_EVAL = os.environ.get("VRX_USAGE_GATE_EVAL", "0") == "1"
    USAGE_EMA_BETA = float(os.environ.get("VRX_USAGE_EMA_BETA", "0.9"))
    USAGE_REMAP_ENABLED = os.environ.get("VRX_USAGE_REMAP", "0") == "1"
    USAGE_REMAP_EVERY = int(os.environ.get("VRX_USAGE_REMAP_EVERY", "50"))

    MODULAR_SAVE = os.environ.get("VRX_MODULAR_SAVE", "0") == "1"
    MODULAR_SAVE_MODE = os.environ.get("VRX_MODULAR_SAVE_MODE", "only").strip().lower()
    MODULAR_DIR = os.environ.get("VRX_MODULAR_DIR", "")

    TENURE_CONTRIB_THRESH = float(os.environ.get("VRX_TENURE_CONTRIB", "1000.0"))
    TENURE_PROBATION_STEPS = int(os.environ.get("VRX_TENURE_PROBATION_STEPS", "1000"))
    TENURE_TTL_STEPS = int(os.environ.get("VRX_TENURE_TTL_STEPS", "0"))
    TENURE_GC = os.environ.get("VRX_TENURE_GC", "0") == "1"

    # ---- Derived control parameter bundles ----
    THERMOSTAT_PARAMS = ThermostatParams(
        ema_beta=float(_get_setting(settings, "thermo_ema", 0.9)),
        target_flip=float(_get_setting(settings, "thermo_target_flip", 0.2)),
        inertia_step=float(_get_setting(settings, "thermo_inertia_step", 0.05)),
        deadzone_step=float(_get_setting(settings, "thermo_deadzone_step", 0.02)),
        walk_step=float(_get_setting(settings, "thermo_walk_step", 0.02)),
        inertia_min=float(_get_setting(settings, "thermo_inertia_min", 0.0)),
        inertia_max=float(_get_setting(settings, "thermo_inertia_max", 0.95)),
        deadzone_min=float(_get_setting(settings, "thermo_deadzone_min", 0.0)),
        deadzone_max=float(_get_setting(settings, "thermo_deadzone_max", 0.5)),
        walk_min=float(_get_setting(settings, "thermo_walk_min", 0.0)),
        walk_max=float(_get_setting(settings, "thermo_walk_max", 0.3)),
    )

    AGC_ENABLED = bool(_get_setting(settings, "agc_enabled", True))
    AGC_GRAD_LOW = float(_get_setting(settings, "agc_grad_low", 1.0))
    AGC_GRAD_HIGH = float(_get_setting(settings, "agc_grad_high", 5.0))
    AGC_SCALE_UP = float(os.environ.get("VRX_SCALE_UP", str(getattr(settings, "agc_scale_up", 1.05))))
    AGC_SCALE_DOWN = float(os.environ.get("VRX_SCALE_DOWN", str(getattr(settings, "agc_scale_down", 0.5))))
    AGC_SCALE_MIN = float(os.environ.get("VRX_SCALE_MIN", str(_get_setting(settings, "agc_scale_min", 0.0005))))
    AGC_SCALE_MAX = float(os.environ.get("VRX_SCALE_MAX", str(_get_setting(settings, "agc_scale_max", 1.0))))
    SCALE_WARMUP_STEPS = int(os.environ.get("VRX_SCALE_WARMUP_STEPS", "0"))
    SCALE_WARMUP_INIT = float(os.environ.get("VRX_SCALE_WARMUP_INIT", str(UPDATE_SCALE)))
    AGC_PARAMS = AGCParams(
        enabled=AGC_ENABLED,
        grad_low=AGC_GRAD_LOW,
        grad_high=AGC_GRAD_HIGH,
        scale_up=AGC_SCALE_UP,
        scale_down=AGC_SCALE_DOWN,
        scale_min=AGC_SCALE_MIN,
        scale_max_default=AGC_SCALE_MAX,
        warmup_steps=SCALE_WARMUP_STEPS,
        warmup_init=SCALE_WARMUP_INIT,
    )

    INERTIA_AUTO = bool(_get_setting(settings, "ptr_inertia_auto", False))
    INERTIA_MIN = float(_get_setting(settings, "ptr_inertia_min", 0.0))
    INERTIA_MAX = float(_get_setting(settings, "ptr_inertia_max", 1.0))
    INERTIA_VEL_FULL = float(_get_setting(settings, "ptr_inertia_vel_full", 1.0))
    INERTIA_EMA = float(_get_setting(settings, "ptr_inertia_ema", 0.9))
    DWELL_INERTIA_ENABLED = bool(int(os.environ.get("VRX_DWELL_INERTIA", "1")))
    DWELL_INERTIA_THRESH = float(os.environ.get("VRX_DWELL_INERTIA_THRESH", "50.0"))
    INERTIA_AUTO_PARAMS = InertiaAutoParams(
        enabled=INERTIA_AUTO,
        inertia_min=INERTIA_MIN,
        inertia_max=INERTIA_MAX,
        vel_full=INERTIA_VEL_FULL,
        ema_beta=INERTIA_EMA,
        dwell_enabled=DWELL_INERTIA_ENABLED,
        dwell_thresh=DWELL_INERTIA_THRESH,
    )

    # AMP helpers (match legacy call sites).
    def amp_grad_scaler() -> Any:
        enabled = bool(USE_AMP) and DEVICE == "cuda" and torch.cuda.is_available()
        # torch.cuda.amp.GradScaler is deprecated in newer PyTorch; prefer torch.amp when available.
        try:
            return torch.amp.GradScaler(device="cuda", enabled=enabled)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=enabled)

    def amp_autocast() -> Any:
        enabled = bool(USE_AMP) and DEVICE == "cuda" and torch.cuda.is_available()
        if not enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=DTYPE, enabled=True)

    # Ensure a few expected model attributes exist for telemetry even in tests.
    if not hasattr(model, "ptr_inertia"):
        model.ptr_inertia = 0.0
    if not hasattr(model, "ptr_deadzone"):
        model.ptr_deadzone = 0.0
    if not hasattr(model, "ptr_walk_prob"):
        model.ptr_walk_prob = 0.0
    if not hasattr(model, "ptr_update_every"):
        model.ptr_update_every = int(PTR_UPDATE_EVERY)

    # -----------------------------------------------------------------------
    # Original train_steps loop (extracted).
    # -----------------------------------------------------------------------

    model = model.to(DEVICE, dtype=DTYPE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = amp_grad_scaler()
    it = iter(loader)
    step = 0
    start = time.time()
    last_heartbeat = start
    train_trace_enabled = TRAIN_TRACE
    train_trace_path = TRAIN_TRACE_PATH
    if train_trace_enabled:
        os.makedirs(os.path.dirname(train_trace_path), exist_ok=True)
    pointer_hist_sum = None
    satiety_exits = 0
    losses = []
    ptr_flip_sum = 0.0
    ptr_mean_dwell_sum = 0.0
    ptr_delta_abs_sum = 0.0
    ptr_max_dwell = 0
    ptr_steps = 0
    flip_ema = None
    panic_reflex = None
    panic_status = ""
    vcog = VCogGovernor(id_target=VCOG_ID_TARGET, sigma_floor=VCOG_SIGMA_FLOOR)
    expert_last_used = []
    last_hibernate_step = -1
    hibernate_dir = None
    head = getattr(model, "head", None)
    if head is not None and HIBERNATE_ENABLED:
        if not hasattr(head, "hibernation_state"):
            head.hibernation_state = {}
        if not hasattr(head, "hibernation_saved"):
            head.hibernation_saved = 0
        if not hasattr(head, "hibernation_fetched"):
            head.hibernation_fetched = 0
        if not hasattr(head, "hibernation_corrupt"):
            head.hibernation_corrupt = 0
        if not hasattr(head, "hibernation_drift"):
            head.hibernation_drift = 0
        head.hibernate_mode = HIBERNATE_MODE
        head.hibernation_enabled = True
    hibernation_state = head.hibernation_state if head is not None and HIBERNATE_ENABLED else {}
    metabolic_stats = {"hgr": None, "prn": None, "prc": None, "prx": None, "idl": None}
    last_metabolic_step = -1
    if PANIC_ENABLED:
        panic_reflex = PanicReflex(
            ema_beta=PANIC_BETA,
            panic_threshold=PANIC_THRESHOLD,
            recovery_rate=PANIC_RECOVERY,
            inertia_low=PANIC_INERTIA_LOW,
            inertia_high=PANIC_INERTIA_HIGH,
            walk_prob_max=PANIC_WALK_MAX,
        )
    cadence_gov = None
    if PTR_UPDATE_GOV:
        cadence_gov = CadenceGovernor(
            start_tau=float(PTR_UPDATE_EVERY),
            warmup_steps=PTR_UPDATE_GOV_WARMUP,
            min_tau=PTR_UPDATE_MIN,
            max_tau=PTR_UPDATE_MAX,
            ema=PTR_UPDATE_EMA,
            target_flip=PTR_UPDATE_TARGET_FLIP,
            grad_high=PTR_UPDATE_GOV_GRAD_HIGH,
            grad_low=PTR_UPDATE_GOV_GRAD_LOW,
            loss_flat=PTR_UPDATE_GOV_LOSS_FLAT,
            loss_spike=PTR_UPDATE_GOV_LOSS_SPIKE,
            step_up=PTR_UPDATE_GOV_STEP_UP,
            step_down=PTR_UPDATE_GOV_STEP_DOWN,
            vel_high=PTR_UPDATE_GOV_VEL_HIGH,
        )
        model.ptr_update_auto = False
    else:
        model.ptr_update_auto = PTR_UPDATE_AUTO
    inertia_signal_streak = 0
    mitosis_acc_history = []
    while step < steps:
        try:
            inputs, targets = next(it)
        except StopIteration:
            it = iter(loader)
            inputs, targets = next(it)
        inputs = inputs.to(DEVICE, non_blocking=True)
        if inputs.dtype != DTYPE:
            inputs = inputs.to(DTYPE)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_autocast():
            outputs, move_pen = model(inputs)
            loss = criterion(outputs, targets) + LAMBDA_MOVE * move_pen
            if METABOLIC_HUNGER:
                head = getattr(model, "head", None)
                num_experts = getattr(head, "num_experts", EXPERT_HEADS) if head is not None else EXPERT_HEADS
                loss = loss + (METABOLIC_COST_COEFF * float(num_experts))
            loss_val = float(loss.item())
            loss_ema = getattr(model, "loss_ema", None)
            if loss_ema is None:
                loss_ema = loss_val
            else:
                loss_ema = LOSS_EMA_BETA * loss_ema + (1.0 - LOSS_EMA_BETA) * loss_val
            model.loss_ema = loss_ema
            if EXPERT_HEADS > 1:
                head = getattr(model, "head", None)
                num_experts = getattr(head, "num_experts", EXPERT_HEADS) if head is not None else EXPERT_HEADS
                _update_expert_usage(model, num_experts, step)
            if EXPERT_BUDGET > 0 and EXPERT_HEADS > 1:
                active_experts = getattr(model, "ptr_expert_active", None)
                if active_experts is not None:
                    head = getattr(model, "head", None)
                    num_experts = getattr(head, "num_experts", EXPERT_HEADS) if head is not None else EXPERT_HEADS
                    budget = min(EXPERT_BUDGET, num_experts)
                    if budget > 0:
                        usage_ratio = max(0.0, (float(active_experts) - float(budget)) / float(budget))
                        usage_ema = getattr(model, "usage_ema", None)
                        if usage_ema is None:
                            usage_ema = usage_ratio
                        else:
                            usage_ema = USAGE_EMA_BETA * usage_ema + (1.0 - USAGE_EMA_BETA) * usage_ratio
                        model.usage_ema = usage_ema
                        usage_lambda = float(getattr(model, "usage_lambda", USAGE_LAMBDA_INIT))
                        if USAGE_GATE_EVAL:
                            eval_acc_val = getattr(model, "last_eval_acc", None)
                            gate_ok = eval_acc_val is not None and float(eval_acc_val) >= USAGE_GATE_CONF
                            model.usage_conf = float(eval_acc_val) if eval_acc_val is not None else None
                        else:
                            confidence = 1.0 / (1.0 + loss_ema)
                            gate_ok = confidence >= USAGE_GATE_CONF
                            model.usage_conf = confidence
                        model.usage_gate_ok = bool(gate_ok)
                        if gate_ok:
                            usage_lambda = min(USAGE_LAMBDA_MAX, usage_lambda + (USAGE_LAMBDA_ETA * usage_ema))
                            loss = loss + (loss.new_tensor(usage_ratio) * usage_lambda)
                            if USAGE_REMAP_ENABLED and usage_ratio > 0 and step % max(1, USAGE_REMAP_EVERY) == 0:
                                counts = getattr(model, "ptr_expert_counts", None)
                                router_map = getattr(model, "router_map", None)
                                if counts is not None and router_map is not None and counts.numel() == num_experts:
                                    topk = torch.topk(counts, k=budget).indices.to(torch.long)
                                    remap_table = torch.arange(num_experts, device=router_map.device)
                                    if budget < num_experts:
                                        topk_cpu = topk.detach().cpu().tolist()
                                        for i in range(num_experts):
                                            if i not in topk_cpu:
                                                remap_table[i] = topk[i % budget]
                                    mapped = remap_table[router_map.to(remap_table.device)]
                                    model.router_map.copy_(mapped)
                                    model.usage_remap_count = int(getattr(model, "usage_remap_count", 0) + 1)
                                    model.usage_remap_last = int(step)
                        model.usage_lambda = usage_lambda
                        model.usage_ratio = usage_ratio
            if INERTIA_SIGNAL_ENABLED:
                confidence = 1.0 / (1.0 + loss_ema)
                epi = confidence * confidence
                model.ptr_inertia_epi = epi
                loss = loss + INERTIA_SIGNAL_REWARD * epi * (1.0 - model.ptr_inertia)
        scaler.scale(loss).backward()
        if hasattr(model, "ptr_inertia_dyn_tensor"):
            model.ptr_inertia_dyn_tensor = None
        if USE_AMP and scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm_step = 0.0
        if hasattr(model, "theta_ptr_reduced"):
            with torch.no_grad():
                grad = model.theta_ptr_reduced.grad
                grad_norm_step = float(grad.norm().item()) if grad is not None else 0.0
        if GRAD_CLIP > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        raw_delta = getattr(model, "ptr_delta_raw_mean", None)
        scale_after_agc = apply_update_agc(
                model, grad_norm_step if hasattr(model, "theta_ptr_reduced") else None, AGC_PARAMS, raw_delta=raw_delta, step=step, log_fn=log
            )
        if scale_after_agc is not None:
            model.update_scale = scale_after_agc
        scaler.step(optimizer)
        scaler.update()

        if DEVICE == "cuda" and not DISABLE_SYNC:
            torch.cuda.synchronize()
            if step % 10 == 0:
                torch.cuda.empty_cache()

        losses.append(loss.item())
        if STAIRCASE_ENABLED and STAIRCASE_ADAPT:
            staircase = getattr(loader, "staircase", None)
            if staircase is not None:
                new_weights = staircase.maybe_adapt(losses, step)
                if new_weights is not None:
                    loader.set_weights(new_weights)
        if hasattr(model, "pointer_hist"):
            if pointer_hist_sum is None:
                pointer_hist_sum = model.pointer_hist.clone()
            else:
                pointer_hist_sum += model.pointer_hist
        if hasattr(model, "satiety_exits"):
            satiety_exits += model.satiety_exits
        if hasattr(model, "ptr_flip_rate"):
            ptr_flip_sum += float(model.ptr_flip_rate)
            ptr_steps += 1
        if THERMO_ENABLED and step % max(1, THERMO_EVERY) == 0:
            focus_ctl = None
            tension_ctl = None
            if getattr(model, "debug_shard_info", None):
                shard_info = model.debug_shard_info
                focus_ctl = shard_info.get("focus")
                tension_ctl = shard_info.get("tension")
            flip_ema = apply_thermostat(model, float(model.ptr_flip_rate), flip_ema, THERMOSTAT_PARAMS,
                focus=focus_ctl,
                tension=tension_ctl,
                raw_delta=getattr(model, "ptr_delta_raw_mean", None),
            )
        if panic_reflex is not None:
            ctrl = panic_reflex.update(float(loss))
            panic_status = ctrl["status"]
            if panic_status == "PANIC":
                if os.environ.get("VRX_PTR_INERTIA_OVERRIDE") is None:
                    model.ptr_inertia = ctrl["inertia"]
                    model.ptr_walk_prob = ctrl["walk_prob"]
        ptr_velocity_raw = getattr(model, "ptr_delta_raw_mean", None)
        if os.environ.get("VRX_PTR_INERTIA_OVERRIDE") is None:
            apply_inertia_auto(model, ptr_velocity_raw, INERTIA_AUTO_PARAMS, panic_active=panic_status == "PANIC")
        if os.environ.get("VRX_FORCE_CADENCE_1") == "1":
            model.ptr_update_every = 1
        elif cadence_gov is not None:
            flip_rate = float(model.ptr_flip_rate) if hasattr(model, "ptr_flip_rate") else 0.0
            ptr_velocity = getattr(model, "ptr_delta_abs_mean", None)
            model.ptr_update_every = cadence_gov.update(loss.item(), grad_norm_step, flip_rate, ptr_velocity)
        now = time.time()
        grad_norm = None
        heartbeat_due = (step % HEARTBEAT_STEPS == 0) or (
            HEARTBEAT_SECS > 0.0 and (now - last_heartbeat) >= HEARTBEAT_SECS
        )
        if heartbeat_due and hasattr(model, "theta_ptr_reduced"):
            grad_norm = grad_norm_step
            log(f"{dataset_name} | {model_name} | grad_norm(theta_ptr)={grad_norm:.4e}")
            if heartbeat_due:
                last_heartbeat = now
                elapsed = now - start
                scale_log = getattr(model, "debug_scale_out", getattr(model, "update_scale", UPDATE_SCALE))
                shard_info = getattr(model, "debug_shard_info", None)
                focus_val = shard_info.get("focus") if shard_info else None
                if focus_val is not None:
                    search_val = max(0.0, min(1.0, 1.0 - float(focus_val)))
                else:
                    try:
                        search_val = max(0.0, min(1.0, float(model.ptr_walk_prob)))
                    except Exception:
                        search_val = 0.0
                eval_acc_val = getattr(model, "last_eval_acc", None)
                if eval_acc_val is None:
                    eval_acc_val = getattr(model, "ptr_inertia_reward_acc", None)
                vcog_header = vcog.update({
                    "loss": loss.item(),
                    "eval_acc": eval_acc_val,
                    "search": search_val,
                    "focus": float(focus_val) if focus_val is not None else 0.0,
                    "orb": float(getattr(model, "ptr_orbit", 0) or 0),
                    "rd": float(getattr(model, "ptr_residual_mean", 0.0) or 0.0),
                    "ac": int(getattr(model, "ptr_anchor_clicks", 0) or 0),
                    "vh": float(getattr(model, "vault_inj_rate", 0.0) or 0.0),
                    "vu": int(getattr(model, "vault_updates", 0) or 0),
                    "inertia": float(model.ptr_inertia),
                    "epi": float(getattr(model, "ptr_inertia_epi", 0.0) or 0.0),
                    "walk": float(model.ptr_walk_prob),
                    "delta": float(getattr(model, "ptr_delta_abs_mean", 0.0) or 0.0),
                    "delta_raw": float(getattr(model, "ptr_delta_raw_mean", 0.0) or 0.0),
                })
                active_experts = shard_info.get("active_experts") if shard_info else None
                inr_pre = getattr(model, "ptr_inertia_dyn_pre", None)
                inr_post = getattr(model, "ptr_inertia_dyn", None)
                inr_floor = float(getattr(model, "ptr_inertia_floor", 0.0) or 0.0)
                if inr_pre is None or inr_post is None:
                    inr_pre = float(model.ptr_inertia)
                    inr_post = float(model.ptr_inertia)
                meta_text = ""
                if METABOLIC_TELEMETRY or HIBERNATE_ENABLED:
                    head = getattr(model, "head", None)
                    num_experts = getattr(head, "num_experts", EXPERT_HEADS) if head is not None else EXPERT_HEADS
                    if len(expert_last_used) < num_experts:
                        expert_last_used.extend([None] * (num_experts - len(expert_last_used)))
                    counts = getattr(model, "ptr_expert_counts", None)
                    if counts is not None:
                        for idx, count in enumerate(counts):
                            if count > 0 and idx < len(expert_last_used):
                                expert_last_used[idx] = step
                    if METABOLIC_TELEMETRY and METABOLIC_EVERY > 0 and (step % METABOLIC_EVERY == 0 or last_metabolic_step < 0):
                        sim_stats = _compute_expert_similarity_stats(model, METABOLIC_SIM_THRESH)
                        if sim_stats is None:
                            metabolic_stats["prn"] = None
                            metabolic_stats["prc"] = None
                            metabolic_stats["prx"] = None
                        else:
                            metabolic_stats["prn"], metabolic_stats["prc"], metabolic_stats["prx"] = sim_stats
                        last_metabolic_step = step
                    idle_count = 0
                    if expert_last_used:
                        for last in expert_last_used:
                            if last is None or (step - last) >= METABOLIC_IDLE_STEPS:
                                idle_count += 1
                    metabolic_stats["idl"] = idle_count
                    if METABOLIC_TELEMETRY:
                        metabolic_stats["hgr"] = METABOLIC_COST_COEFF * float(num_experts)
                    else:
                        metabolic_stats["hgr"] = None
                    if HIBERNATE_ENABLED and HIBERNATE_EVERY > 0 and (step % HIBERNATE_EVERY == 0 or last_hibernate_step < 0):
                        if hibernate_dir is None:
                            hibernate_dir = _resolve_hibernate_dir(HIBERNATE_DIR, ROOT)
                        for idx, last in enumerate(expert_last_used):
                            idle = last is None or (step - last) >= HIBERNATE_IDLE_STEPS
                            meta = hibernation_state.get(idx)
                            if idle:
                                state = _extract_expert_state(head, idx)
                                if state is None:
                                    continue
                                ram_hash = _hash_state_dict(state)
                                if meta is not None:
                                    saved_hash = meta.get("hash")
                                    if saved_hash and ram_hash and saved_hash != ram_hash:
                                        if head is not None:
                                            head.hibernation_drift += 1
                                path = meta.get("path") if meta is not None else None
                                if not path:
                                    path = os.path.join(hibernate_dir, f"expert_{idx}.pt")
                                digest = _save_expert_snapshot(state, path)
                                hibernation_state[idx] = {
                                    "path": path,
                                    "hash": digest,
                                    "step": step,
                                    "offloaded": False,
                                }
                                if head is not None:
                                    head.hibernation_saved += 1
                                if HIBERNATE_MODE == "offload":
                                    if _zero_expert_weights(head, idx):
                                        hibernation_state[idx]["offloaded"] = True
                        last_hibernate_step = step
                    hgr = metabolic_stats.get("hgr")
                    prn = metabolic_stats.get("prn")
                    prc = metabolic_stats.get("prc")
                    prx = metabolic_stats.get("prx")
                    idl = metabolic_stats.get("idl")
                    prn_text = "-" if prn is None else f"{prn:.2f}"
                    prc_text = "-" if prc is None else f"{prc:d}"
                    prx_text = "-" if prx is None else f"{prx[0]}-{prx[1]}"
                    idl_text = "-" if idl is None else f"{idl:d}"
                    hgr_text = "-" if hgr is None else f"{hgr:.4f}"
                    hibernation_saved = getattr(head, "hibernation_saved", 0) if head is not None else 0
                    hibernation_fetched = getattr(head, "hibernation_fetched", 0) if head is not None else 0
                    hibernation_corrupt = getattr(head, "hibernation_corrupt", 0) if head is not None else 0
                    hibernation_drift = getattr(head, "hibernation_drift", 0) if head is not None else 0
                    hbr_text = "-" if not HIBERNATE_ENABLED else f"{hibernation_saved:d}"
                    hbf_text = "-" if not HIBERNATE_ENABLED else f"{hibernation_fetched:d}"
                    hibc_text = "-" if not HIBERNATE_ENABLED else f"{hibernation_corrupt:d}"
                    hibd_text = "-" if not HIBERNATE_ENABLED else f"{hibernation_drift:d}"
                    meta_text = (
                        f" META[HGR:{hgr_text} PRN:{prn_text} PRC:{prc_text} PRX:{prx_text} "
                        f"IDL:{idl_text} HIBR:{hbr_text} HIBF:{hbf_text} HIBC:{hibc_text} HIBD:{hibd_text}]"
                    )
                raw_compact = (
                    f"RAW[SCA:{float(scale_log):.3f} INR:{inr_pre:.2f}->{inr_post:.2f}/F:{inr_floor:.2f} "
                    f"DZN:{model.ptr_deadzone:.2f} WLK:{model.ptr_walk_prob:.2f} "
                    f"CAD:{model.ptr_update_every} EXP:{active_experts if active_experts is not None else EXPERT_HEADS}]"
                )
                stats_dict = getattr(model, "debug_stats", None)
                try:
                    stats_payload = json.dumps(stats_dict if stats_dict is not None else {}, separators=(",", ":"))
                except Exception:
                    stats_payload = str(stats_dict)
                debug_payload = f" | debug {stats_payload}"
                log(
                    f"{dataset_name} | {model_name} | step {step:04d}/{steps:04d} | loss {loss.item():.4f} | "
                    f"t={elapsed:.1f}s | {vcog_header}{meta_text} {raw_compact}"
                    + (f" | panic={panic_status}" if panic_reflex is not None else "")
                    + debug_payload
                )
            if train_trace_enabled:
                pointer_entropy = None
                pointer_total = None
                if hasattr(model, "pointer_hist") and model.pointer_hist is not None:
                    hist_np = model.pointer_hist.cpu().numpy()
                    total = float(hist_np.sum())
                    if total > 0.0:
                        probs = hist_np / total
                        pointer_entropy = float(-(probs * np.log(probs + 1e-12)).sum())
                        pointer_total = int(total)
                trace = {
                    "ts": time.time(),
                    "dataset": dataset_name,
                    "model": model_name,
                    "step": step,
                    "steps_total": steps,
                    "loss": float(loss.item()),
                    "grad_norm_theta_ptr": grad_norm,
                    "ctrl": {
                        "inertia": float(model.ptr_inertia),
                        "deadzone": float(model.ptr_deadzone),
                        "walk": float(model.ptr_walk_prob),
                    },
                    "panic": panic_status,
                    "ptr_flip_rate": getattr(model, "ptr_flip_rate", None),
                    "ptr_mean_dwell": getattr(model, "ptr_mean_dwell", None),
                    "ptr_max_dwell": getattr(model, "ptr_max_dwell", None),
                    "ptr_delta_abs_mean": getattr(model, "ptr_delta_abs_mean", None),
                    "ptr_delta_raw_mean": getattr(model, "ptr_delta_raw_mean", None),
                    "ptr_orbit": getattr(model, "ptr_orbit", None),
                    "ptr_residual_mean": getattr(model, "ptr_residual_mean", None),
                    "ptr_anchor_clicks": getattr(model, "ptr_anchor_clicks", None),
                    "ptr_update_allowed": getattr(model, "ptr_update_allowed", None),
                    "ptr_update_blocked": getattr(model, "ptr_update_blocked", None),
                    "ptr_lock": getattr(model, "ptr_lock", None),
                    "ptr_time_pointer": getattr(model, "time_pointer", None),
                    "ptr_warmup_steps": getattr(model, "ptr_warmup_steps", None),
                    "ptr_update_every": getattr(model, "ptr_update_every", None),
                    "update_scale": getattr(model, "debug_scale_out", getattr(model, "update_scale", UPDATE_SCALE)),
                    "ground_speed": getattr(model, "ground_speed", None),
                    "ground_speed_ema": getattr(model, "ground_speed_ema", None),
                    "ground_speed_limit": getattr(model, "ground_speed_limit", None),
                    "pointer_entropy": pointer_entropy,
                    "pointer_total": pointer_total,
                    "satiety_exits": getattr(model, "satiety_exits", None),
                    "state_loop_entropy": getattr(model, "state_loop_entropy", None),
                    "state_loop_flip_rate": getattr(model, "state_loop_flip_rate", None),
                    "state_loop_abab_rate": getattr(model, "state_loop_abab_rate", None),
                    "state_loop_mean_dwell": getattr(model, "state_loop_mean_dwell", None),
                    "state_loop_max_dwell": getattr(model, "state_loop_max_dwell", None),
                }
                if DEBUG_STATS and getattr(model, "debug_stats", None):
                    trace["debug"] = model.debug_stats
                if METABOLIC_TELEMETRY:
                    trace["metabolic_hunger"] = metabolic_stats.get("hgr")
                    trace["metabolic_prune_max_sim"] = metabolic_stats.get("prn")
                    trace["metabolic_prune_pairs"] = metabolic_stats.get("prc")
                    trace["metabolic_prune_pair"] = metabolic_stats.get("prx")
                    trace["metabolic_idle_experts"] = metabolic_stats.get("idl")
                    trace["metabolic_idle_steps"] = METABOLIC_IDLE_STEPS
                    if HIBERNATE_ENABLED:
                        trace["hibernation_saved"] = getattr(head, "hibernation_saved", 0) if head is not None else 0
                        trace["hibernation_fetched"] = getattr(head, "hibernation_fetched", 0) if head is not None else 0
                        trace["hibernation_corrupt"] = getattr(head, "hibernation_corrupt", 0) if head is not None else 0
                        trace["hibernation_drift"] = getattr(head, "hibernation_drift", 0) if head is not None else 0
                        trace["hibernation_idle_steps"] = HIBERNATE_IDLE_STEPS
                        trace["hibernation_dir"] = HIBERNATE_DIR
                try:
                    with open(train_trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(trace) + "\n")
                except Exception as e:
                    log(f"train_trace write failed: {e}")
        if hasattr(model, "ptr_mean_dwell"):
            ptr_mean_dwell_sum += float(model.ptr_mean_dwell)
        if hasattr(model, "ptr_delta_abs_mean"):
            ptr_delta_abs_sum += float(model.ptr_delta_abs_mean)
        if hasattr(model, "ptr_max_dwell"):
            ptr_max_dwell = max(ptr_max_dwell, int(model.ptr_max_dwell))
        del outputs, loss
        step += 1
        if SAVE_EVERY_STEPS > 0 and step % SAVE_EVERY_STEPS == 0:
            loss_value = losses[-1] if losses else None
            raw_delta = getattr(model, "ptr_delta_raw_mean", None)
            raw_delta_value = float(raw_delta) if raw_delta is not None else None
            grad_value = grad_norm_step if hasattr(model, "theta_ptr_reduced") else None
            is_finite = _checkpoint_is_finite(loss_value, grad_value, raw_delta_value)
            ckpt = _checkpoint_payload(model, optimizer, scaler, step, losses)
            ckpt["meta"] = {
                "loss": loss_value,
                "grad_norm": grad_value,
                "raw_delta": raw_delta_value,
                "nonfinite": not is_finite,
            }
            if model_name.startswith("evo_"):
                if not EVO_CKPT_INDIV:
                    continue
                evo_step_dir = os.path.join(ROOT, "artifacts", "evolution", "step_ckpts")
                os.makedirs(evo_step_dir, exist_ok=True)
                ckpt_path = os.path.join(evo_step_dir, f"{model_name}_step_{step:04d}.pt")
                torch.save(ckpt, ckpt_path)
                log(f"Checkpoint saved @ step {step} -> {ckpt_path}")
                continue
            step_path, bad_path, last_good_path = _checkpoint_paths(CHECKPOINT_PATH, step)
            save_monolithic = (not MODULAR_SAVE) or (MODULAR_SAVE_MODE == "dual")
            if save_monolithic and SAVE_HISTORY:
                torch.save(ckpt, step_path)
                log(f"Checkpoint saved @ step {step} -> {step_path}")
                if (not is_finite) and SAVE_BAD:
                    torch.save(ckpt, bad_path)
                    log(f"Non-finite checkpoint saved @ step {step} -> {bad_path}")
            if is_finite:
                if MODULAR_SAVE:
                    modular_dir = _resolve_modular_dir(MODULAR_DIR, ROOT, CHECKPOINT_PATH)
                    _save_modular_checkpoint(
                        model,
                        optimizer,
                        scaler,
                        step,
                        losses,
                        modular_dir,
                        TENURE_CONTRIB_THRESH,
                        TENURE_PROBATION_STEPS,
                        ttl_steps=TENURE_TTL_STEPS,
                        gc_enabled=TENURE_GC,
                    )
                    log(f"Modular checkpoint saved @ step {step} -> {modular_dir}")
                if save_monolithic:
                    torch.save(ckpt, CHECKPOINT_PATH)
                    if SAVE_LAST_GOOD:
                        torch.save(ckpt, last_good_path)
                    log(f"Checkpoint saved @ step {step} -> {CHECKPOINT_PATH}")
            else:
                log(f"Checkpoint not updated (non-finite metrics) @ step {step}")
    slope = compute_slope(losses)
    ptr_flip_rate = (ptr_flip_sum / ptr_steps) if ptr_steps else None
    ptr_mean_dwell = (ptr_mean_dwell_sum / ptr_steps) if ptr_steps else None
    ptr_delta_abs_mean = (ptr_delta_abs_sum / ptr_steps) if ptr_steps else None
    return {
        "loss_slope": slope,
        "steps": steps,
        "pointer_hist": pointer_hist_sum.tolist() if pointer_hist_sum is not None else None,
        "satiety_exits": satiety_exits,
        "ptr_flip_rate": ptr_flip_rate,
        "ptr_mean_dwell": ptr_mean_dwell,
        "ptr_max_dwell": ptr_max_dwell,
        "ptr_delta_abs_mean": ptr_delta_abs_mean,
        "ptr_delta_raw_mean": getattr(model, "ptr_delta_raw_mean", None),
    }

