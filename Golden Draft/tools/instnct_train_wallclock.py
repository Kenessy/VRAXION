"""INSTNCT train_wallclock (Golden Draft).

Behavior-preserving extraction from the legacy monolithic training script.

This module is intended to remain importable in lightweight CPU-only test
environments (i.e., when the full VRAXION runtime isn't available).
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Optional imports (real repo) with safe fallbacks (unit tests / isolated runs).
# -----------------------------------------------------------------------------


def _settings_get(settings, key: str, default=None):
    """Read a setting from either a dict-like or attribute-like settings object."""

    # Certain legacy globals were sourced from env vars that do not match the
    # corresponding setting key names. Keep the extraction behavior stable by
    # checking those env overrides explicitly.
    env_alias = {
        "HIBERNATE_ENABLED": "VRX_HIBERNATE",
        "INERTIA_SIGNAL_ENABLED": "VRX_INERTIA_SIGNAL",
        "MITOSIS_CKPT_PATH": "VRX_MITOSIS_CKPT",
        "MITOSIS_ENABLED": "VRX_MITOSIS",
        "RATCHET_ENABLED": "VRX_INERTIA_RATCHET",
        "SAVE_EVERY_STEPS": "VRX_SAVE_EVERY_STEPS",
        "SHARD_ENABLED": "VRX_SHARD_BATCH",
        "TRACTION_ENABLED": "VRX_TRACTION_LOG",
        "WALK_PULSE_ENABLED": "VRX_WALK_PULSE",
    }

    if key in env_alias:
        env_key = env_alias[key]
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return env_val

    # Generic legacy env fallback: VRX_FOO for key=FOO.
    env_val = os.environ.get(f"VRX_{key}")
    if env_val is not None:
        return env_val

    if settings is None:
        return default
    # dict-like
    if isinstance(settings, dict):
        if key in settings:
            return settings[key]
        if key.upper() in settings:
            return settings[key.upper()]
        if key.lower() in settings:
            return settings[key.lower()]
        return default
    # attribute-like
    if hasattr(settings, key):
        return getattr(settings, key)
    if hasattr(settings, key.upper()):
        return getattr(settings, key.upper())
    if hasattr(settings, key.lower()):
        return getattr(settings, key.lower())
    return default


def _coerce_dtype(value, default: torch.dtype) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        lookup = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        key = value.strip().lower()
        return lookup.get(key, default)
    return default


def _coerce_device(value, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, str):
        return value
    return default


def _coerce_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        # Preserve strict "1" semantics only where explicitly required by the
        # legacy function itself. For general settings, accept common truthy.
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


try:
    from vraxion.settings import load_settings  # type: ignore
except Exception:  # pragma: no cover
    load_settings = None  # type: ignore


try:
    from vraxion.instnct.infra import log, compute_slope, _checkpoint_paths  # type: ignore
except Exception:  # pragma: no cover

    def log(msg: str) -> None:
        # Keep output ASCII-only; tests typically patch this to silence.
        print(msg)

    def compute_slope(losses):
        if not losses or len(losses) < 2:
            return 0.0
        # Simple slope over the window.
        return (float(losses[-1]) - float(losses[0])) / float(len(losses) - 1)

    def _checkpoint_paths(checkpoint_path: str, step: int):
        root, ext = os.path.splitext(checkpoint_path)
        if not ext:
            ext = ".pt"
        step_path = f"{root}_step{step:06d}{ext}"
        bad_path = f"{root}_bad_step{step:06d}{ext}"
        last_good_path = f"{root}_last_good{ext}"
        return step_path, bad_path, last_good_path


try:
    from vraxion.instnct.modular_checkpoint import (  # type: ignore
        _save_modular_checkpoint,
        _load_modular_checkpoint,
        _resolve_modular_dir,
        _resolve_modular_resume_dir,
    )
except Exception:  # pragma: no cover

    def _save_modular_checkpoint(*args, **kwargs):
        return None

    def _load_modular_checkpoint(*args, **kwargs):
        return None

    def _resolve_modular_dir(modular_dir, root, checkpoint_path):
        # Reasonable fallback.
        if modular_dir:
            return modular_dir
        ckpt_dir = os.path.dirname(checkpoint_path) or "."
        return os.path.join(ckpt_dir, "modular")

    def _resolve_modular_resume_dir(resume_path: str):
        return resume_path if os.path.isdir(resume_path) else None


try:
    from vraxion.instnct.sharding import calculate_adaptive_vasc  # type: ignore
except Exception:  # pragma: no cover

    def calculate_adaptive_vasc(*args, **kwargs):
        # shard_count, shard_size, focus, tension, cohesion
        return None, kwargs.get("local_shard_size", 0), None, None, None


try:
    from vraxion.instnct.thermo import ThermostatParams, apply_thermostat  # type: ignore
except Exception:  # pragma: no cover

    class ThermostatParams:
        def __init__(self, *args, **kwargs):
            pass

    def apply_thermostat(*args, **kwargs):
        # Returns updated flip_ema
        return kwargs.get("flip_ema", None)


try:
    from vraxion.instnct.agc import AGCParams, apply_update_agc  # type: ignore
except Exception:  # pragma: no cover

    class AGCParams:
        def __init__(self, *args, **kwargs):
            pass

    def apply_update_agc(*args, **kwargs):
        return None


try:
    from vraxion.instnct.inertia_auto import InertiaAutoParams, apply_inertia_auto  # type: ignore
except Exception:  # pragma: no cover

    class InertiaAutoParams:
        def __init__(self, *args, **kwargs):
            pass

    def apply_inertia_auto(*args, **kwargs):
        return None


try:
    from vraxion.instnct.panic import PanicReflex  # type: ignore
except Exception:  # pragma: no cover

    class PanicReflex:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, loss_val: float):
            return {"status": "OK", "inertia": 0.0, "walk_prob": 0.0}


try:
    from vraxion.instnct.cadence import CadenceGovernor  # type: ignore
except Exception:  # pragma: no cover

    class CadenceGovernor:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, loss, grad_norm, flip_rate, ptr_velocity):
            return 1


try:
    from vraxion.instnct.vcog import VCogGovernor  # type: ignore
except Exception:  # pragma: no cover

    class VCogGovernor:
        def __init__(self, id_target=0.0, sigma_floor=0.0):
            self.id_target = id_target
            self.sigma_floor = sigma_floor

        def update(self, payload: dict) -> str:
            # Keep header stable-ish; minimal fallback.
            loss = payload.get("loss")
            try:
                loss_val = float(loss)
            except Exception:
                loss_val = 0.0
            return f"VCOG(loss={loss_val:.4f})"


# -----------------------------------------------------------------------------
# Local helpers (small, dependency-free). These mirror behavior used by the
# legacy train_wallclock for checkpointing and eval when the full stack isn't
# available.
# -----------------------------------------------------------------------------


def _checkpoint_is_finite(loss_value, grad_value, raw_delta_value) -> bool:
    for val in (loss_value, grad_value, raw_delta_value):
        if val is None:
            continue
        try:
            f = float(val)
        except Exception:
            return False
        if not math.isfinite(f):
            return False
    return True


def _checkpoint_payload(model, optimizer, scaler, step: int, losses: list[float]) -> dict:
    payload = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scaler": scaler.state_dict() if getattr(scaler, "is_enabled", lambda: False)() else None,
        "step": int(step),
        "losses": list(losses),
    }
    # Include a few dynamic fields that the resume path may look for.
    for key in (
        "update_scale",
        "ptr_inertia",
        "ptr_inertia_ema",
        "ptr_inertia_floor",
        "agc_scale_max",
        "ground_speed_ema",
        "ground_speed_limit",
    ):
        if hasattr(model, key):
            try:
                payload[key] = float(getattr(model, key))
            except Exception:
                payload[key] = getattr(model, key)
    # Optional expert count for mismatch handling.
    head = getattr(model, "head", None)
    if head is not None and hasattr(head, "num_experts"):
        try:
            payload["num_experts"] = int(getattr(head, "num_experts"))
        except Exception:
            pass
    return payload


def eval_model(model, eval_loader, dataset_name: str, model_name: str):
    """Minimal evaluator: returns {'eval_acc': float}."""
    model_was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in eval_loader:
            inputs, targets = batch
            outputs = model(inputs)[0] if isinstance(model(inputs), tuple) else model(inputs)
            preds = outputs.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.numel())
    if model_was_training:
        model.train()
    if total <= 0:
        return {"eval_acc": None}
    return {"eval_acc": float(correct) / float(total)}


def _update_expert_usage(model, num_experts: int, step: int) -> None:
    # Minimal accounting: infer mapped expert ids if present.
    last_ptr = getattr(model, "last_ptr_int", None)
    if last_ptr is None:
        return
    try:
        last_ptr = last_ptr.to(torch.long).detach().cpu()
    except Exception:
        return
    router_map = getattr(model, "router_map", None)
    if router_map is not None and getattr(router_map, "numel", lambda: 0)() > 0:
        try:
            mapped = router_map.detach().cpu()
            mapped_ids = mapped[last_ptr.clamp(0, mapped.numel() - 1)]
        except Exception:
            mapped_ids = last_ptr % max(1, int(num_experts))
    else:
        mapped_ids = last_ptr % max(1, int(num_experts))

    counts = getattr(model, "ptr_expert_counts", None)
    if counts is None or not isinstance(counts, torch.Tensor) or counts.numel() != int(num_experts):
        counts = torch.zeros(int(num_experts), dtype=torch.float32)
    try:
        binc = torch.bincount(mapped_ids.flatten(), minlength=int(num_experts)).float()
        counts = counts + binc
        total = counts.sum().clamp(min=1.0)
        model.ptr_expert_counts = counts
        model.ptr_expert_active = int((counts > 0).sum().item())
        model.ptr_expert_max_share = float((counts.max() / total).item())
        if int(num_experts) > 1:
            probs = counts / total
            ent = float((-(probs * torch.log(probs + 1e-12)).sum().item()) / math.log(int(num_experts)))
            model.ptr_expert_entropy = ent
    except Exception:
        return


def _compute_expert_similarity_stats(model, sim_thresh: float):
    # Best-effort placeholder. Real repo may provide richer telemetry.
    return None


def _resolve_hibernate_dir(hibernate_dir: str, root: str | None):
    base = hibernate_dir or "hibernation"
    if root:
        return os.path.join(root, base)
    return base


def _extract_expert_state(head, idx: int):
    # Best-effort: support nn.ModuleList of experts, or dict-like.
    if head is None:
        return None
    experts = getattr(head, "experts", None)
    if experts is None:
        return None
    try:
        expert = experts[idx]
    except Exception:
        return None
    if hasattr(expert, "state_dict"):
        return expert.state_dict()
    return None


def _hash_state_dict(state: dict | None) -> str | None:
    if not state:
        return None
    try:
        # Stable-ish hash: keys + tensor sums.
        items = []
        for k in sorted(state.keys()):
            v = state[k]
            if isinstance(v, torch.Tensor):
                items.append((k, float(v.detach().float().sum().item()), tuple(v.shape)))
            else:
                items.append((k, str(type(v))))
        blob = json.dumps(items, sort_keys=True).encode("utf-8")
        import hashlib

        return hashlib.sha256(blob).hexdigest()
    except Exception:
        return None


def _save_expert_snapshot(state: dict, path: str) -> str | None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)
        return _hash_state_dict(state)
    except Exception:
        return None


def _zero_expert_weights(head, idx: int) -> bool:
    state = _extract_expert_state(head, idx)
    if not state:
        return False
    try:
        experts = getattr(head, "experts", None)
        if experts is None:
            return False
        expert = experts[idx]
        with torch.no_grad():
            for p in expert.parameters():
                p.zero_()
        return True
    except Exception:
        return False


def amp_grad_scaler():
    enabled = bool(USE_AMP and DEVICE == "cuda" and torch.cuda.is_available() and DTYPE != torch.bfloat16)
    # Prefer torch.amp when available (avoids deprecation warnings on newer PyTorch).
    try:
        return torch.amp.GradScaler(device="cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


@contextmanager
def amp_autocast():
    enabled = bool(USE_AMP and DEVICE == "cuda" and torch.cuda.is_available())
    if not enabled:
        yield
        return

    try:
        with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=True):
            yield
    except Exception:
        with torch.cuda.amp.autocast():
            yield


# -----------------------------------------------------------------------------
# Settings / constants
# -----------------------------------------------------------------------------

_SETTINGS = load_settings() if callable(load_settings) else None

# Keep DEVICE as a string for parity with legacy checks (DEVICE == "cuda").
DEVICE = _coerce_device(_settings_get(_SETTINGS, "DEVICE", os.environ.get("VAR_COMPUTE_DEVICE", "cpu")), "cpu")
DTYPE = _coerce_dtype(_settings_get(_SETTINGS, "DTYPE", torch.float32), torch.float32)

LR = float(_settings_get(_SETTINGS, "LR", 1e-3))
LAMBDA_MOVE = float(_settings_get(_SETTINGS, "LAMBDA_MOVE", 0.0))

USE_AMP = _coerce_bool(_settings_get(_SETTINGS, "USE_AMP", False), False)
DISABLE_SYNC = _coerce_bool(_settings_get(_SETTINGS, "DISABLE_SYNC", False), False)
GRAD_CLIP = float(_settings_get(_SETTINGS, "GRAD_CLIP", 0.0))

# Checkpointing / saving defaults: keep disabled unless settings say otherwise.
RESUME = _coerce_bool(_settings_get(_SETTINGS, "RESUME", False), False)
CHECKPOINT_PATH = str(_settings_get(_SETTINGS, "CHECKPOINT_PATH", "checkpoint.pt"))
MODULAR_RESUME = _coerce_bool(_settings_get(_SETTINGS, "MODULAR_RESUME", False), False)
MODULAR_SAVE = _coerce_bool(_settings_get(_SETTINGS, "MODULAR_SAVE", False), False)
MODULAR_SAVE_MODE = str(_settings_get(_SETTINGS, "MODULAR_SAVE_MODE", "mono"))
MODULAR_DIR = str(_settings_get(_SETTINGS, "MODULAR_DIR", ""))
SAVE_HISTORY = _coerce_bool(_settings_get(_SETTINGS, "SAVE_HISTORY", False), False)
SAVE_BAD = _coerce_bool(_settings_get(_SETTINGS, "SAVE_BAD", False), False)
SAVE_LAST_GOOD = _coerce_bool(_settings_get(_SETTINGS, "SAVE_LAST_GOOD", False), False)
SAVE_EVERY_STEPS = int(_settings_get(_SETTINGS, "SAVE_EVERY_STEPS", 0))

EVAL_EVERY_STEPS = int(_settings_get(_SETTINGS, "EVAL_EVERY_STEPS", 0))
EVAL_AT_CHECKPOINT = _coerce_bool(_settings_get(_SETTINGS, "EVAL_AT_CHECKPOINT", True), True)

# Misc loop controls.
MAX_STEPS = int(_settings_get(_SETTINGS, "MAX_STEPS", 0))

# Telemetry / logging.
HEARTBEAT_STEPS = int(_settings_get(_SETTINGS, "HEARTBEAT_STEPS", 100))
HEARTBEAT_SECS = float(_settings_get(_SETTINGS, "HEARTBEAT_SECS", 0.0))
LIVE_TRACE_EVERY = int(_settings_get(_SETTINGS, "LIVE_TRACE_EVERY", 0))
LIVE_TRACE_PATH = str(_settings_get(_SETTINGS, "LIVE_TRACE_PATH", ""))

# Sharding / VASC.
SHARD_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "SHARD_ENABLED", True), True)
SHARD_ADAPT = _coerce_bool(_settings_get(_SETTINGS, "SHARD_ADAPT", True), True)
SHARD_ADAPT_EVERY = int(_settings_get(_SETTINGS, "SHARD_ADAPT_EVERY", 50))
SHARD_SIZE = int(_settings_get(_SETTINGS, "SHARD_SIZE", 19))
TRACTION_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "TRACTION_ENABLED", True), True)

# Experts / synthesis.
EXPERT_HEADS = int(_settings_get(_SETTINGS, "EXPERT_HEADS", 1))
EXPERT_BUDGET = int(_settings_get(_SETTINGS, "EXPERT_BUDGET", 0))
SYNTH_MODE = str(_settings_get(_SETTINGS, "SYNTH_MODE", ""))

# Pointer update controls / walk jitter.
PTR_WALK_PROB = float(_settings_get(_SETTINGS, "PTR_WALK_PROB", 0.0))
WALK_PULSE_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "WALK_PULSE_ENABLED", False), False)
WALK_PULSE_EVERY = int(_settings_get(_SETTINGS, "WALK_PULSE_EVERY", 0))
WALK_PULSE_VALUE = float(_settings_get(_SETTINGS, "WALK_PULSE_VALUE", PTR_WALK_PROB))

# Thermostat / panic / inertia / cadence.
THERMO_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "THERMO_ENABLED", False), False)
THERMO_EVERY = int(_settings_get(_SETTINGS, "THERMO_EVERY", 1))
THERMO_TARGET_FLIP = float(_settings_get(_SETTINGS, "THERMO_TARGET_FLIP", 0.2))
THERMO_EMA = float(_settings_get(_SETTINGS, "THERMO_EMA", 0.9))
THERMO_INERTIA_STEP = float(_settings_get(_SETTINGS, "THERMO_INERTIA_STEP", 0.05))
THERMO_DEADZONE_STEP = float(_settings_get(_SETTINGS, "THERMO_DEADZONE_STEP", 0.02))
THERMO_WALK_STEP = float(_settings_get(_SETTINGS, "THERMO_WALK_STEP", 0.02))
THERMO_INERTIA_MIN = float(_settings_get(_SETTINGS, "THERMO_INERTIA_MIN", 0.0))
THERMO_INERTIA_MAX = float(_settings_get(_SETTINGS, "THERMO_INERTIA_MAX", 0.95))
THERMO_DEADZONE_MIN = float(_settings_get(_SETTINGS, "THERMO_DEADZONE_MIN", 0.0))
THERMO_DEADZONE_MAX = float(_settings_get(_SETTINGS, "THERMO_DEADZONE_MAX", 0.5))
THERMO_WALK_MIN = float(_settings_get(_SETTINGS, "THERMO_WALK_MIN", 0.0))
THERMO_WALK_MAX = float(_settings_get(_SETTINGS, "THERMO_WALK_MAX", 0.3))
PANIC_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "PANIC_ENABLED", False), False)

INERTIA_SIGNAL_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "INERTIA_SIGNAL_ENABLED", False), False)
INERTIA_SIGNAL_REWARD = float(_settings_get(_SETTINGS, "INERTIA_SIGNAL_REWARD", 0.1))

PTR_UPDATE_GOV = _coerce_bool(_settings_get(_SETTINGS, "PTR_UPDATE_GOV", False), False)
PTR_UPDATE_EVERY = int(_settings_get(_SETTINGS, "PTR_UPDATE_EVERY", 1))
PTR_UPDATE_GOV_WARMUP = int(_settings_get(_SETTINGS, "PTR_UPDATE_GOV_WARMUP", 0))
PTR_UPDATE_MIN = int(_settings_get(_SETTINGS, "PTR_UPDATE_MIN", 1))
PTR_UPDATE_MAX = int(_settings_get(_SETTINGS, "PTR_UPDATE_MAX", 1))
PTR_UPDATE_EMA = float(_settings_get(_SETTINGS, "PTR_UPDATE_EMA", 0.9))
PTR_UPDATE_TARGET_FLIP = float(_settings_get(_SETTINGS, "PTR_UPDATE_TARGET_FLIP", 0.0))
PTR_UPDATE_GOV_GRAD_HIGH = float(_settings_get(_SETTINGS, "PTR_UPDATE_GOV_GRAD_HIGH", 0.0))
PTR_UPDATE_GOV_GRAD_LOW = float(_settings_get(_SETTINGS, "PTR_UPDATE_GOV_GRAD_LOW", 0.0))
PTR_UPDATE_GOV_LOSS_FLAT = float(_settings_get(_SETTINGS, "PTR_UPDATE_GOV_LOSS_FLAT", 0.0))
PTR_UPDATE_GOV_LOSS_SPIKE = float(_settings_get(_SETTINGS, "PTR_UPDATE_GOV_LOSS_SPIKE", 0.0))
PTR_UPDATE_GOV_STEP_UP = float(_settings_get(_SETTINGS, "PTR_UPDATE_GOV_STEP_UP", 0.0))
PTR_UPDATE_GOV_STEP_DOWN = float(_settings_get(_SETTINGS, "PTR_UPDATE_GOV_STEP_DOWN", 0.0))
PTR_UPDATE_GOV_VEL_HIGH = float(_settings_get(_SETTINGS, "PTR_UPDATE_GOV_VEL_HIGH", 0.5))
PTR_UPDATE_AUTO = _coerce_bool(_settings_get(_SETTINGS, "PTR_UPDATE_AUTO", False), False)

# AGC parameters / limits.
AGC_SCALE_MIN = float(_settings_get(_SETTINGS, "AGC_SCALE_MIN", 0.0))
AGC_SCALE_MAX = float(_settings_get(_SETTINGS, "AGC_SCALE_MAX", 1.0))
AGC_SCALE_MAX_MIN = float(_settings_get(_SETTINGS, "AGC_SCALE_MAX_MIN", 0.0))
AGC_SCALE_MAX_DECAY = float(_settings_get(_SETTINGS, "AGC_SCALE_MAX_DECAY", 1.0))
AGC_PLATEAU_WINDOW = int(_settings_get(_SETTINGS, "AGC_PLATEAU_WINDOW", 0))
AGC_PLATEAU_MIN_STEPS = int(_settings_get(_SETTINGS, "AGC_PLATEAU_MIN_STEPS", 0))
AGC_PLATEAU_STD = float(_settings_get(_SETTINGS, "AGC_PLATEAU_STD", 0.0))

LOSS_KEEP = int(_settings_get(_SETTINGS, "LOSS_KEEP", 0))
LOSS_EMA_BETA = float(_settings_get(_SETTINGS, "LOSS_EMA_BETA", 0.95))

UPDATE_SCALE = float(_settings_get(_SETTINGS, "UPDATE_SCALE", 1.0))
AGC_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "AGC_ENABLED", True), True)
AGC_GRAD_LOW = float(_settings_get(_SETTINGS, "AGC_GRAD_LOW", 1.0))
AGC_GRAD_HIGH = float(_settings_get(_SETTINGS, "AGC_GRAD_HIGH", 5.0))
# Legacy env names are VRX_SCALE_UP / VRX_SCALE_DOWN (not part of Settings).
AGC_SCALE_UP = float(os.environ.get("VRX_SCALE_UP", str(_settings_get(_SETTINGS, "AGC_SCALE_UP", 1.05))))
AGC_SCALE_DOWN = float(os.environ.get("VRX_SCALE_DOWN", str(_settings_get(_SETTINGS, "AGC_SCALE_DOWN", 0.5))))
SCALE_WARMUP_STEPS = int(os.environ.get("VRX_SCALE_WARMUP_STEPS", "0"))
SCALE_WARMUP_INIT = float(os.environ.get("VRX_SCALE_WARMUP_INIT", str(UPDATE_SCALE)))

# Ratchet controls.
RATCHET_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "RATCHET_ENABLED", False), False)
RATCHET_ACC_MIN = float(_settings_get(_SETTINGS, "RATCHET_ACC_MIN", 0.98))
RATCHET_STREAK = int(_settings_get(_SETTINGS, "RATCHET_STREAK", 2))
RATCHET_BASE = float(_settings_get(_SETTINGS, "RATCHET_BASE", 0.5))
RATCHET_SCALE = float(_settings_get(_SETTINGS, "RATCHET_SCALE", 0.98))

# Inertia-signal controls.
INERTIA_SIGNAL_ACC_MIN = float(_settings_get(_SETTINGS, "INERTIA_SIGNAL_ACC_MIN", 0.98))
INERTIA_SIGNAL_STREAK = int(_settings_get(_SETTINGS, "INERTIA_SIGNAL_STREAK", 2))
INERTIA_SIGNAL_FLOOR = float(_settings_get(_SETTINGS, "INERTIA_SIGNAL_FLOOR", 0.5))
INERTIA_SIGNAL_TARGET = float(_settings_get(_SETTINGS, "INERTIA_SIGNAL_TARGET", 0.95))

# Panic params.
PANIC_BETA = float(_settings_get(_SETTINGS, "PANIC_BETA", 0.9))
PANIC_THRESHOLD = float(_settings_get(_SETTINGS, "PANIC_THRESHOLD", 0.0))
PANIC_RECOVERY = float(_settings_get(_SETTINGS, "PANIC_RECOVERY", 0.0))
PANIC_INERTIA_LOW = float(_settings_get(_SETTINGS, "PANIC_INERTIA_LOW", 0.0))
PANIC_INERTIA_HIGH = float(_settings_get(_SETTINGS, "PANIC_INERTIA_HIGH", 1.0))
PANIC_WALK_MAX = float(_settings_get(_SETTINGS, "PANIC_WALK_MAX", 0.0))

# Mitosis.
MITOSIS_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "MITOSIS_ENABLED", False), False)
MITOSIS_STALL_WINDOW = int(_settings_get(_SETTINGS, "MITOSIS_STALL_WINDOW", 100))
MITOSIS_STALL_DELTA = float(_settings_get(_SETTINGS, "MITOSIS_STALL_DELTA", 0.005))
MITOSIS_IMBALANCE = float(_settings_get(_SETTINGS, "MITOSIS_IMBALANCE", 0.5))
MITOSIS_CKPT_PATH = str(_settings_get(_SETTINGS, "MITOSIS_CKPT_PATH", ""))
MITOSIS_EXIT_CODE = int(_settings_get(_SETTINGS, "MITOSIS_EXIT_CODE", 86))

# Mobius.
MOBIUS_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "MOBIUS_ENABLED", False), False)

# Staircase.
STAIRCASE_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "STAIRCASE_ENABLED", False), False)
STAIRCASE_ADAPT = _coerce_bool(_settings_get(_SETTINGS, "STAIRCASE_ADAPT", False), False)

# Metabolic / hibernation.
METABOLIC_HUNGER = _coerce_bool(_settings_get(_SETTINGS, "METABOLIC_HUNGER", False), False)
METABOLIC_TELEMETRY = _coerce_bool(_settings_get(_SETTINGS, "METABOLIC_TELEMETRY", False), False)
METABOLIC_COST_COEFF = float(_settings_get(_SETTINGS, "METABOLIC_COST_COEFF", 0.0001))
METABOLIC_EVERY = int(_settings_get(_SETTINGS, "METABOLIC_EVERY", 500))
METABOLIC_SIM_THRESH = float(_settings_get(_SETTINGS, "METABOLIC_SIM_THRESH", 0.98))
METABOLIC_IDLE_STEPS = int(_settings_get(_SETTINGS, "METABOLIC_IDLE_STEPS", 2000))

HIBERNATE_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "HIBERNATE_ENABLED", False), False)
HIBERNATE_EVERY = int(_settings_get(_SETTINGS, "HIBERNATE_EVERY", 50))
HIBERNATE_DIR = str(_settings_get(_SETTINGS, "HIBERNATE_DIR", "hibernation"))
HIBERNATE_IDLE_STEPS = int(_settings_get(_SETTINGS, "HIBERNATE_IDLE_STEPS", METABOLIC_IDLE_STEPS))
HIBERNATE_MODE = str(_settings_get(_SETTINGS, "HIBERNATE_MODE", "shadow"))
ROOT = str(_settings_get(_SETTINGS, "ROOT", ""))

# Domain separation.
DOMAIN_SEP_COEFF = float(_settings_get(_SETTINGS, "DOMAIN_SEP_COEFF", 0.0))

# Usage budget penalty.
USAGE_EMA_BETA = float(_settings_get(_SETTINGS, "USAGE_EMA_BETA", 0.9))
USAGE_LAMBDA_INIT = float(_settings_get(_SETTINGS, "USAGE_LAMBDA_INIT", 0.0))
USAGE_GATE_EVAL = _coerce_bool(_settings_get(_SETTINGS, "USAGE_GATE_EVAL", False), False)
USAGE_GATE_CONF = float(_settings_get(_SETTINGS, "USAGE_GATE_CONF", 0.95))
USAGE_LAMBDA_MAX = float(_settings_get(_SETTINGS, "USAGE_LAMBDA_MAX", 0.5))
USAGE_LAMBDA_ETA = float(_settings_get(_SETTINGS, "USAGE_LAMBDA_ETA", 0.05))
USAGE_REMAP_ENABLED = _coerce_bool(_settings_get(_SETTINGS, "USAGE_REMAP_ENABLED", False), False)
USAGE_REMAP_EVERY = int(_settings_get(_SETTINGS, "USAGE_REMAP_EVERY", 50))

# Tenure / modular checkpoint GC.
TENURE_CONTRIB_THRESH = float(_settings_get(_SETTINGS, "TENURE_CONTRIB_THRESH", 1000.0))
TENURE_PROBATION_STEPS = int(_settings_get(_SETTINGS, "TENURE_PROBATION_STEPS", 1000))
TENURE_TTL_STEPS = int(_settings_get(_SETTINGS, "TENURE_TTL_STEPS", 0))
TENURE_GC = _coerce_bool(_settings_get(_SETTINGS, "TENURE_GC", False), False)

# VCOG.
VCOG_ID_TARGET = float(_settings_get(_SETTINGS, "VCOG_ID_TARGET", 0.25))
VCOG_SIGMA_FLOOR = float(_settings_get(_SETTINGS, "VCOG_SIGMA_FLOOR", 1e-4))

# Parameter objects.
THERMOSTAT_PARAMS = ThermostatParams(
    ema_beta=THERMO_EMA,
    target_flip=THERMO_TARGET_FLIP,
    inertia_step=THERMO_INERTIA_STEP,
    deadzone_step=THERMO_DEADZONE_STEP,
    walk_step=THERMO_WALK_STEP,
    inertia_min=THERMO_INERTIA_MIN,
    inertia_max=THERMO_INERTIA_MAX,
    deadzone_min=THERMO_DEADZONE_MIN,
    deadzone_max=THERMO_DEADZONE_MAX,
    walk_min=THERMO_WALK_MIN,
    walk_max=THERMO_WALK_MAX,
)
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
INERTIA_AUTO = _coerce_bool(_settings_get(_SETTINGS, "ptr_inertia_auto", False), False)
INERTIA_MIN = float(_settings_get(_SETTINGS, "ptr_inertia_min", 0.0))
INERTIA_MAX = float(_settings_get(_SETTINGS, "ptr_inertia_max", 1.0))
INERTIA_VEL_FULL = float(_settings_get(_SETTINGS, "ptr_inertia_vel_full", 1.0))
INERTIA_EMA = float(_settings_get(_SETTINGS, "ptr_inertia_ema", 0.9))
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

# Wall-clock default (signature requires a module-level constant).
WALL_CLOCK_SECONDS = float(_settings_get(_SETTINGS, "WALL_CLOCK_SECONDS", 3600.0))


# -----------------------------------------------------------------------------
# Extracted train_wallclock implementation
# -----------------------------------------------------------------------------

def train_wallclock(model, loader, dataset_name, model_name, num_classes, wall_clock=WALL_CLOCK_SECONDS, eval_loader=None):
    model = model.to(DEVICE, dtype=DTYPE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = amp_grad_scaler()
    losses = []
    pointer_hist_sum = None
    satiety_exits = 0
    grad_norm = 0.0
    flip_ema = None
    ptr_flip_sum = 0.0
    ptr_mean_dwell_sum = 0.0
    ptr_delta_abs_sum = 0.0
    ptr_max_dwell = 0
    ptr_steps = 0
    panic_reflex = None
    panic_status = ""
    vcog = VCogGovernor(id_target=VCOG_ID_TARGET, sigma_floor=VCOG_SIGMA_FLOOR)
    model.last_eval_acc = None
    ratchet_streak = 0
    mitosis_acc_history = []
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
    inertia_signal_streak = 0
    ratchet_allowed = RATCHET_ENABLED and not INERTIA_SIGNAL_ENABLED
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
    # Reset adaptive scales to env defaults to avoid carrying prior runs.
    init_scale = AGC_SCALE_MIN
    model.update_scale = init_scale
    model.agc_scale_max = AGC_SCALE_MAX
    model.agc_scale_cap = AGC_SCALE_MAX
    env_inertia = os.environ.get("VRX_PTR_INERTIA_OVERRIDE")
    if env_inertia is not None:
        model.ptr_inertia = float(env_inertia)
        model.ptr_inertia_ema = model.ptr_inertia

    start = time.time()
    # Allow indefinite runs unless explicitly told to honor wall-clock limits.
    ignore_wall_clock = os.environ.get("VRX_IGNORE_WALL_CLOCK") == "1"
    end_time = float("inf") if ignore_wall_clock else (start + wall_clock)
    last_heartbeat = start
    last_live_trace = start
    step = 0
    if RESUME:
        resume_path = os.path.abspath(CHECKPOINT_PATH)
        modular_dir = _resolve_modular_resume_dir(resume_path) if (MODULAR_RESUME or os.path.isdir(resume_path)) else None
        ckpt = None
        if modular_dir:
            log(f"Resume requested, attempting modular load: {modular_dir}")
            ckpt = _load_modular_checkpoint(model, optimizer, scaler, modular_dir)
        elif os.path.exists(resume_path):
            log(f"Resume requested, attempting load: {resume_path}")
            ckpt = torch.load(resume_path, map_location=DEVICE)
            try:
                model.load_state_dict(ckpt["model"])
            except RuntimeError as exc:
                exc_text = str(exc)
                allow_relaxed = MOBIUS_ENABLED or "router_map" in exc_text
                if allow_relaxed:
                    log("Retrying load_state_dict(strict=False) due to key mismatch")
                    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
                    if missing:
                        log(f"Load missing keys: {missing}")
                    if unexpected:
                        log(f"Load unexpected keys: {unexpected}")
                else:
                    raise
            ckpt_experts = ckpt.get("num_experts")
            experts_mismatch = ckpt_experts is not None and int(ckpt_experts) != EXPERT_HEADS
            if MOBIUS_ENABLED or experts_mismatch:
                if experts_mismatch:
                    log(f"Expert count mismatch (ckpt={ckpt_experts}, env={EXPERT_HEADS}); skipping optimizer load")
                else:
                    log("MOBIUS enabled: skipping optimizer/scaler load for clean restart")
            else:
                optimizer.load_state_dict(ckpt["optim"])
                if ckpt.get("scaler") and USE_AMP:
                    scaler.load_state_dict(ckpt["scaler"])
        if ckpt is not None:
            step = int(ckpt.get("step", 0))
            losses = list(ckpt.get("losses", []))
            if "update_scale" in ckpt:
                model.update_scale = float(ckpt["update_scale"])
            if "ptr_inertia" in ckpt:
                model.ptr_inertia = float(ckpt["ptr_inertia"])
            if "ptr_inertia_ema" in ckpt:
                model.ptr_inertia_ema = float(ckpt["ptr_inertia_ema"])
            if "ptr_inertia_floor" in ckpt:
                model.ptr_inertia_floor = float(ckpt["ptr_inertia_floor"])
            if "agc_scale_max" in ckpt:
                model.agc_scale_max = float(ckpt["agc_scale_max"])
            if "ground_speed_ema" in ckpt:
                model.ground_speed_ema = ckpt["ground_speed_ema"]
            if "ground_speed_limit" in ckpt:
                model.ground_speed_limit = ckpt["ground_speed_limit"]
            # Honor env overrides after resume (no hidden reset).
            env_scale_init = os.environ.get("VRX_SCALE_INIT")
            env_scale_max = os.environ.get("VRX_SCALE_MAX")
            if env_scale_init is not None:
                model.update_scale = float(env_scale_init)
            if env_scale_max is not None:
                model.agc_scale_max = float(env_scale_max)
            env_inertia = os.environ.get("VRX_PTR_INERTIA_OVERRIDE")
            if env_inertia is not None:
                model.ptr_inertia = float(env_inertia)
                model.ptr_inertia_ema = model.ptr_inertia
            model.agc_scale_cap = model.agc_scale_max
            # Clear dynamic speed stats to avoid stale velocity.
            model.ground_speed_ema = None
            model.ground_speed_limit = None
            model.ground_speed = None
            model.debug_scale_out = model.update_scale
            log(f"Resumed from checkpoint: {resume_path} (step={step}) | update_scale={model.update_scale}")
    # cycle loader until wall clock
    # Disable early-stop flag; we rely on external stop or manual interrupt.
    stop_early = False
    xray_enabled = os.getenv("VRX_XRAY", "0") == "1"
    while time.time() <= end_time:
        for batch in loader:
            if time.time() > end_time:
                break
            inputs, targets = batch
            inputs = inputs.to(DEVICE, non_blocking=True)
            if inputs.dtype != DTYPE:
                inputs = inputs.to(DTYPE)
            targets = targets.to(DEVICE, non_blocking=True)

            # Curiosity jitter: pulse walk_prob for one step on a schedule.
            prev_walk_prob = getattr(model, "ptr_walk_prob", PTR_WALK_PROB)
            pulse_applied = False
            if WALK_PULSE_ENABLED and WALK_PULSE_EVERY > 0 and step % WALK_PULSE_EVERY == 0:
                model.ptr_walk_prob = WALK_PULSE_VALUE
                pulse_applied = True

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast():
                if xray_enabled:
                    outputs, move_pen, xray = model(inputs, return_xray=True)
                else:
                    outputs, move_pen = model(inputs)
                local_shard_size = SHARD_SIZE
                shard_count = None
                traction = None
                focus = None
                tension = None
                cohesion = None
                batch_sz = outputs.shape[0]
                dwell_val = getattr(model, "ptr_mean_dwell", 0.0)
                try:
                    dwell_val = float(dwell_val)
                except Exception:
                    dwell_val = 0.0
                tension_val = grad_norm if "grad_norm" in locals() else 0.0
                try:
                    tension_val = float(tension_val)
                except Exception:
                    tension_val = 0.0

                # Maintain dynamic dwell/grad references for scale-free VASC.
                vasc_max_dwell = getattr(model, "vasc_max_dwell", None)
                if vasc_max_dwell is None:
                    vasc_max_dwell = max(1.0, dwell_val)
                else:
                    vasc_max_dwell = max(float(vasc_max_dwell), dwell_val, 1.0)
                model.vasc_max_dwell = vasc_max_dwell

                vasc_grad_ema = getattr(model, "vasc_grad_ema", None)
                if vasc_grad_ema is None:
                    vasc_grad_ema = max(1e-8, tension_val)
                else:
                    vasc_grad_ema = PTR_UPDATE_EMA * float(vasc_grad_ema) + (1.0 - PTR_UPDATE_EMA) * float(tension_val)
                model.vasc_grad_ema = vasc_grad_ema

                dom0_top = None
                dom0_share = None
                dom1_top = None
                dom1_share = None
                dom_overlap = None
                dom_sep = None
                if EXPERT_HEADS > 1 and SYNTH_MODE == "assoc_mix":
                    last_ptr = getattr(model, "last_ptr_int", None)
                    router_map = getattr(model, "router_map", None)
                    if last_ptr is not None:
                        try:
                            last_ptr_cpu = last_ptr.to(torch.long).cpu()
                            targets_cpu = targets.detach().cpu()
                            dom_mask = targets_cpu >= 2
                            if router_map is not None and router_map.numel() > 0:
                                mapped = router_map.detach().cpu()
                                mapped_ids = mapped[last_ptr_cpu.clamp(0, mapped.numel() - 1)]
                                num_experts = int(getattr(model.head, "num_experts", EXPERT_HEADS))
                            else:
                                mapped_ids = last_ptr_cpu % EXPERT_HEADS
                                num_experts = EXPERT_HEADS
                            if mapped_ids.numel() == dom_mask.numel():
                                p0 = None
                                p1 = None
                                if (~dom_mask).any():
                                    counts0 = torch.bincount(mapped_ids[~dom_mask], minlength=num_experts).float()
                                    total0 = counts0.sum().clamp(min=1.0)
                                    dom0_top = int(torch.argmax(counts0).item())
                                    dom0_share = float((counts0[dom0_top] / total0).item())
                                    p0 = counts0 / total0
                                if dom_mask.any():
                                    counts1 = torch.bincount(mapped_ids[dom_mask], minlength=num_experts).float()
                                    total1 = counts1.sum().clamp(min=1.0)
                                    dom1_top = int(torch.argmax(counts1).item())
                                    dom1_share = float((counts1[dom1_top] / total1).item())
                                    p1 = counts1 / total1
                                if p0 is not None and p1 is not None:
                                    overlap = torch.minimum(p0, p1).sum()
                                    dom_overlap = float(overlap.item())
                                    dom_sep = 1.0 - dom_overlap
                        except Exception:
                            dom0_top = None
                if SHARD_ENABLED and SHARD_ADAPT and SHARD_ADAPT_EVERY > 0 and (step % SHARD_ADAPT_EVERY == 0):
                    shard_count, local_shard_size, focus, tension, cohesion = calculate_adaptive_vasc(
                        batch_sz, dwell_val, tension_val, vasc_max_dwell, vasc_grad_ema
                    )
                    if TRACTION_ENABLED:
                        # Traction: dwell vs flip/tension ratio.
                        flip_val = getattr(model, "ptr_flip_rate", None)
                        try:
                            flip_val = float(flip_val) if flip_val is not None else 0.0
                        except Exception:
                            flip_val = 0.0
                        denom = max(1e-6, flip_val + tension_val)
                        traction = dwell_val / denom
                    active_experts = None
                    expert_imbalance = None
                    expert_entropy = None
                    if EXPERT_HEADS > 1:
                        active_experts = getattr(model, "ptr_expert_active", None)
                        expert_imbalance = getattr(model, "ptr_expert_max_share", None)
                        expert_entropy = getattr(model, "ptr_expert_entropy", None)
                        if active_experts is None:
                            last_ptr = getattr(model, "last_ptr_int", None)
                            router_map = getattr(model, "router_map", None)
                            if last_ptr is not None:
                                try:
                                    last_ptr = last_ptr.to(torch.long)
                                    if router_map is not None and router_map.numel() > 0:
                                        mapped = router_map.detach().cpu()
                                        mapped_ids = mapped[last_ptr.clamp(0, mapped.numel() - 1)]
                                        active_experts = int(torch.unique(mapped_ids).numel())
                                        counts = torch.bincount(mapped_ids, minlength=EXPERT_HEADS).float()
                                        total = counts.sum().clamp(min=1.0)
                                        expert_imbalance = float((counts.max() / total).item())
                                        expert_entropy = float(
                                            (-(counts / total) * torch.log((counts / total) + 1e-12)).sum().item()
                                            / math.log(EXPERT_HEADS)
                                        )
                                    else:
                                        active_experts = int(torch.unique(last_ptr % EXPERT_HEADS).numel())
                                except Exception:
                                    active_experts = None
                    model.debug_shard_info = {
                        "count": shard_count,
                        "size": local_shard_size,
                        "focus": focus,
                        "tension": tension,
                        "cohesion": cohesion,
                    }
                    if traction is not None:
                        model.debug_shard_info["traction"] = traction
                    if active_experts is not None:
                        model.debug_shard_info["active_experts"] = active_experts
                    if expert_imbalance is not None:
                        model.debug_shard_info["expert_imbalance"] = expert_imbalance
                    if expert_entropy is not None:
                        model.debug_shard_info["expert_entropy"] = expert_entropy
                    if dom0_top is not None:
                        model.debug_shard_info["dom0_top"] = dom0_top
                        model.debug_shard_info["dom0_share"] = dom0_share
                    if dom1_top is not None:
                        model.debug_shard_info["dom1_top"] = dom1_top
                        model.debug_shard_info["dom1_share"] = dom1_share
                    if dom_overlap is not None:
                        model.debug_shard_info["dom_overlap"] = dom_overlap
                    if dom_sep is not None:
                        model.debug_shard_info["dom_sep"] = dom_sep
                if SHARD_ENABLED and local_shard_size > 0 and outputs.shape[0] > local_shard_size:
                    # Sub-culture partitioning: split batch into shards, mean losses.
                    loss_parts = []
                    for out_chunk, tgt_chunk in zip(
                        torch.split(outputs, local_shard_size, dim=0), torch.split(targets, local_shard_size, dim=0)
                    ):
                        loss_parts.append(criterion(out_chunk, tgt_chunk))
                    loss = torch.stack(loss_parts).mean() + LAMBDA_MOVE * move_pen
                else:
                    loss = criterion(outputs, targets) + LAMBDA_MOVE * move_pen
                if METABOLIC_HUNGER:
                    head = getattr(model, "head", None)
                    num_experts = getattr(head, "num_experts", EXPERT_HEADS) if head is not None else EXPERT_HEADS
                    loss = loss + (METABOLIC_COST_COEFF * float(num_experts))
                if DOMAIN_SEP_COEFF > 0.0 and dom_overlap is not None:
                    loss = loss + (loss.new_tensor(dom_overlap) * DOMAIN_SEP_COEFF)
                loss_val = float(loss.item())
                loss_ema = getattr(model, "loss_ema", None)
                if loss_ema is None:
                    loss_ema = loss_val
                else:
                    loss_ema = LOSS_EMA_BETA * loss_ema + (1.0 - LOSS_EMA_BETA) * loss_val
                model.loss_ema = loss_ema
                # Adaptive think-ring alpha slider (smoothly blends between min/max).
                if model.think_enabled and model.think_alpha_adapt:
                    jitter = abs(loss_val - loss_ema)
                    jitter_ema = getattr(model, "think_alpha_jitter_ema", None)
                    if jitter_ema is None:
                        jitter_ema = jitter
                    else:
                        jitter_ema = (model.think_alpha_beta * jitter_ema) + ((1.0 - model.think_alpha_beta) * jitter)
                    model.think_alpha_jitter_ema = jitter_ema
                    jlow = max(1e-9, model.think_alpha_jitter_low)
                    jhigh = max(jlow * 1.01, model.think_alpha_jitter_high)
                    jnorm = (jitter_ema - jlow) / (jhigh - jlow)
                    if jnorm < 0.0:
                        jnorm = 0.0
                    elif jnorm > 1.0:
                        jnorm = 1.0
                    target = model.think_alpha_min + (model.think_alpha_max - model.think_alpha_min) * jnorm
                    alpha_prev = model.think_alpha
                    alpha_new = (model.think_alpha_beta * alpha_prev) + ((1.0 - model.think_alpha_beta) * target)
                    # Clamp to bounds and apply
                    alpha_new = max(model.think_alpha_min, min(model.think_alpha_max, alpha_new))
                    model.think_alpha_target = target
                    model.think_alpha = alpha_new
                if model.vault_adapt and model.vault_enabled:
                    # Self-regulating vault "tap": adjust gate based on surprise + utility.
                    loss_floor = getattr(model, "vault_loss_floor", None)
                    if loss_floor is None:
                        loss_floor = loss_ema
                    loss_floor = min(loss_floor, loss_ema)
                    model.vault_loss_floor = loss_floor
                    # Periodic ablation probe to estimate utility of memory injection.
                    if step % max(1, model.vault_probe_every) == 0:
                        with torch.no_grad():
                            old_vault = model.vault_enabled
                            model.vault_enabled = False
                            out_no, move_no = model(inputs)
                            model.vault_enabled = old_vault
                            loss_no = criterion(out_no, targets) + LAMBDA_MOVE * move_no
                            loss_no_val = float(loss_no.item())
                        util = (loss_no_val - loss_val) / max(loss_no_val, 1e-6)
                        util = max(-1.0, min(1.0, util))
                        util_ema = getattr(model, "vault_util_ema", None)
                        if util_ema is None:
                            util_ema = util
                        else:
                            util_ema = (model.vault_probe_beta * util_ema) + ((1.0 - model.vault_probe_beta) * util)
                        model.vault_util_ema = util_ema
                    util_ema = float(getattr(model, "vault_util_ema", 0.0) or 0.0)
                    surprise = max(0.0, (loss_ema - loss_floor) / max(loss_floor, 1e-6))
                    alpha = model.vault_alpha_min + (model.vault_k_surprise * surprise) - (model.vault_k_utility * util_ema)
                    gate = model.vault_gate_min + (model.vault_k_utility * util_ema) - (model.vault_k_surprise * surprise)
                    # Clamp and apply.
                    alpha = max(model.vault_alpha_min, min(model.vault_alpha_max, alpha))
                    gate = max(model.vault_gate_min, min(model.vault_gate_max, gate))
                    model.vault_alpha = alpha
                    model.vault_gate = gate
                    model.vault_decay = alpha
                    model.vault_surprise = surprise
                    model.vault_utility = util_ema
                if EXPERT_HEADS > 1:
                    head = getattr(model, "head", None)
                    num_experts = getattr(head, "num_experts", EXPERT_HEADS) if head is not None else EXPERT_HEADS
                    _update_expert_usage(model, num_experts, step)
                if EXPERT_BUDGET > 0 and EXPERT_HEADS > 1:
                    active_experts = getattr(model, "ptr_expert_active", None)
                    if active_experts is None:
                        active_experts = shard_info.get("active_experts") if shard_info else None
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
            if hasattr(model, "theta_ptr_reduced"):
                with torch.no_grad():
                    grad = model.theta_ptr_reduced.grad
                    grad_norm = float(grad.norm().item()) if grad is not None else 0.0
            if GRAD_CLIP > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            raw_delta = getattr(model, "ptr_delta_raw_mean", None)
            scale_after_agc = apply_update_agc(
                model, grad_norm if hasattr(model, "theta_ptr_reduced") else None, AGC_PARAMS, raw_delta=raw_delta, step=step, log_fn=log
            )
            if scale_after_agc is not None:
                model.update_scale = scale_after_agc
            scaler.step(optimizer)
            scaler.update()

            # Force small kernels to complete and keep the watchdog happy; clear cache frequently.
            if DEVICE == "cuda" and not DISABLE_SYNC:
                torch.cuda.synchronize()
                if step % 10 == 0:
                    torch.cuda.empty_cache()

            loss_value = float(loss.item())
            losses.append(loss_value)
            if AGC_PLATEAU_WINDOW > 0 and len(losses) >= AGC_PLATEAU_WINDOW and step >= AGC_PLATEAU_MIN_STEPS:
                recent = losses[-AGC_PLATEAU_WINDOW:]
                mean_recent = sum(recent) / len(recent)
                var_recent = sum((x - mean_recent) ** 2 for x in recent) / len(recent)
                std_recent = math.sqrt(var_recent)
                if std_recent <= AGC_PLATEAU_STD and math.isfinite(std_recent):
                    new_cap = max(AGC_SCALE_MAX_MIN, float(getattr(model, "agc_scale_max", AGC_SCALE_MAX)) * AGC_SCALE_MAX_DECAY)
                    if new_cap < getattr(model, "agc_scale_max", AGC_SCALE_MAX):
                        model.agc_scale_max = new_cap
                        model.update_scale = min(getattr(model, "update_scale", new_cap), new_cap)
            if LOSS_KEEP > 0 and len(losses) > LOSS_KEEP:
                losses = losses[-LOSS_KEEP:]
            if STAIRCASE_ENABLED and STAIRCASE_ADAPT:
                staircase = getattr(loader, "staircase", None)
                if staircase is not None:
                    new_weights = staircase.maybe_adapt(losses, step)
                    if new_weights is not None:
                        loader.set_weights(new_weights)
            if THERMO_ENABLED and hasattr(model, "ptr_flip_rate") and step % max(1, THERMO_EVERY) == 0:
                focus_ctl = focus
                tension_ctl = tension
                if (focus_ctl is None or tension_ctl is None) and getattr(model, "debug_shard_info", None):
                    shard_info = model.debug_shard_info
                    if focus_ctl is None:
                        focus_ctl = shard_info.get("focus")
                    if tension_ctl is None:
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
            # Do not override manual inertia when VRX_PTR_INERTIA_OVERRIDE is set.
            if os.environ.get("VRX_PTR_INERTIA_OVERRIDE") is None:
                apply_inertia_auto(model, ptr_velocity_raw, INERTIA_AUTO_PARAMS, panic_active=panic_status == "PANIC")
            if os.environ.get("VRX_FORCE_CADENCE_1") == "1":
                model.ptr_update_every = 1
            elif cadence_gov is not None:
                flip_rate = float(model.ptr_flip_rate) if hasattr(model, "ptr_flip_rate") else 0.0
                ptr_velocity = getattr(model, "ptr_delta_abs_mean", None)
                model.ptr_update_every = cadence_gov.update(loss.item(), grad_norm, flip_rate, ptr_velocity)
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
            if hasattr(model, "ptr_mean_dwell"):
                ptr_mean_dwell_sum += float(model.ptr_mean_dwell)
            if hasattr(model, "ptr_delta_abs_mean"):
                ptr_delta_abs_sum += float(model.ptr_delta_abs_mean)
            if hasattr(model, "ptr_max_dwell"):
                ptr_max_dwell = max(ptr_max_dwell, int(model.ptr_max_dwell))
            now = time.time()
            heartbeat_due = (step % HEARTBEAT_STEPS == 0) or (
                HEARTBEAT_SECS > 0.0 and (now - last_heartbeat) >= HEARTBEAT_SECS
            )
            if heartbeat_due and hasattr(model, "theta_ptr_reduced"):
                log(f"{dataset_name} | {model_name} | grad_norm(theta_ptr)={grad_norm:.4e}")

            if heartbeat_due:
                last_heartbeat = now
                elapsed = now - start
                raw_delta = getattr(model, "ptr_delta_raw_mean", None)
                ground_speed = getattr(model, "ground_speed", None)
                ground_speed_ema = getattr(model, "ground_speed_ema", None)
                ground_speed_limit = getattr(model, "ground_speed_limit", None)
                scale_log = getattr(model, "debug_scale_out", getattr(model, "update_scale", UPDATE_SCALE))
                shard_info = getattr(model, "debug_shard_info", None)
                xray_text = ""
                if xray_enabled and isinstance(locals().get("xray"), dict):
                    parts = []
                    if "attn_mass" in xray:
                        parts.append(f"attn {xray['attn_mass']:.3f}")
                    if "gate_sat" in xray:
                        parts.append(f"sat {xray['gate_sat']:.2f}")
                    if "h_mag" in xray:
                        parts.append(f"h_mag {xray['h_mag']:.2f}")
                    if "damp_ratio" in xray:
                        parts.append(f"damp {xray['damp_ratio']:.2f}")
                    if parts:
                        xray_text = " | " + " ".join(parts)
                shard_text = ""
                if shard_info:
                    shard_text = f", shard={shard_info.get('count', '-')}/{shard_info.get('size', '-')}"
                    if TRACTION_ENABLED and "traction" in shard_info:
                        shard_text += f", traction={shard_info['traction']:.2f}"
                    if "cohesion" in shard_info:
                        shard_text += f", cohesion={shard_info['cohesion']:.2f}"
                    if "focus" in shard_info:
                        shard_text += f", focus={shard_info['focus']:.2f}"
                    if "tension" in shard_info:
                        shard_text += f", tension={shard_info['tension']:.2f}"
                if shard_info and "active_experts" in shard_info:
                    shard_text += f", experts={shard_info['active_experts']}"
                if shard_info and "expert_imbalance" in shard_info:
                    shard_text += f", imb={float(shard_info['expert_imbalance']):.2f}"
                if shard_info and "expert_entropy" in shard_info:
                    shard_text += f", ent={float(shard_info['expert_entropy']):.2f}"
                focus_val = shard_info.get("focus") if shard_info else None
                tension_val = shard_info.get("tension") if shard_info else None
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
                agc_cap = getattr(model, "agc_scale_cap", getattr(model, "agc_scale_max", AGC_SCALE_MAX))
                active_experts = shard_info.get("active_experts") if shard_info else None
                if active_experts is None:
                    active_experts = getattr(model, "ptr_expert_active", None)
                expert_imbalance = shard_info.get("expert_imbalance") if shard_info else None
                expert_entropy = shard_info.get("expert_entropy") if shard_info else None
                if active_experts is None:
                    active_experts = getattr(model, "ptr_expert_active", None)
                if expert_imbalance is None:
                    expert_imbalance = getattr(model, "ptr_expert_max_share", None)
                if expert_entropy is None:
                    expert_entropy = getattr(model, "ptr_expert_entropy", None)
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
                domain_text = ""
                shard_info = getattr(model, "debug_shard_info", None)
                if SYNTH_MODE == "assoc_mix" and shard_info:
                    dom0_top = shard_info.get("dom0_top")
                    dom0_share = shard_info.get("dom0_share")
                    dom1_top = shard_info.get("dom1_top")
                    dom1_share = shard_info.get("dom1_share")
                    if dom0_top is not None or dom1_top is not None:
                        dom0_text = "-" if dom0_top is None else f"{dom0_top}:{dom0_share:.2f}"
                        dom1_text = "-" if dom1_top is None else f"{dom1_top}:{dom1_share:.2f}"
                        dom_sep = shard_info.get("dom_sep")
                        sep_text = "" if dom_sep is None else f" SEP:{dom_sep:.2f}"
                        domain_text = f" DOM[{dom0_text} {dom1_text}{sep_text}]"
                raw_compact = (
                    f"RAW[SCA:{float(scale_log):.3f} INR:{inr_pre:.2f}->{inr_post:.2f}/F:{inr_floor:.2f} "
                    f"DZN:{model.ptr_deadzone:.2f} WLK:{model.ptr_walk_prob:.2f} "
                    f"CAD:{model.ptr_update_every} EXP:{active_experts if active_experts is not None else EXPERT_HEADS}]"
                )
                log(
                    f"{dataset_name} | {model_name} | step {step:04d} | loss {loss.item():.4f} | "
                    f"t={elapsed:.1f}s | {vcog_header}{meta_text}{domain_text} {raw_compact}{shard_text}"
                    + xray_text
                    + (f" | panic={panic_status}" if panic_reflex is not None else "")
                )
                live_due = LIVE_TRACE_EVERY > 0 and (step % LIVE_TRACE_EVERY == 0)
                if HEARTBEAT_SECS > 0.0 and (now - last_live_trace) >= HEARTBEAT_SECS:
                    live_due = True
                if LIVE_TRACE_PATH and len(LIVE_TRACE_PATH) > 0 and live_due:
                    trace = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "step": step,
                        "time_sec": round(elapsed, 3),
                        "loss": loss.item(),
                        "grad_norm_theta_ptr": grad_norm,
                    }
                    if hasattr(model, "ptr_flip_rate"):
                        trace["ptr_flip_rate"] = model.ptr_flip_rate
                        trace["ptr_pingpong_rate"] = model.ptr_pingpong_rate
                        trace["ptr_max_dwell"] = model.ptr_max_dwell
                        trace["ptr_mean_dwell"] = model.ptr_mean_dwell
                        trace["ptr_delta_abs_mean"] = getattr(model, "ptr_delta_abs_mean", None)
                        trace["ptr_delta_raw_mean"] = getattr(model, "ptr_delta_raw_mean", None)
                        trace["ptr_inertia"] = model.ptr_inertia
                        trace["ptr_orbit"] = getattr(model, "ptr_orbit", None)
                        trace["ptr_residual_mean"] = getattr(model, "ptr_residual_mean", None)
                        trace["ptr_anchor_clicks"] = getattr(model, "ptr_anchor_clicks", None)
                        trace["ptr_update_allowed"] = getattr(model, "ptr_update_allowed", None)
                        trace["ptr_update_blocked"] = getattr(model, "ptr_update_blocked", None)
                        trace["ptr_lock"] = getattr(model, "ptr_lock", None)
                        trace["ptr_time_pointer"] = getattr(model, "time_pointer", None)
                        trace["ptr_warmup_steps"] = getattr(model, "ptr_warmup_steps", None)
                        trace["ptr_update_every"] = getattr(model, "ptr_update_every", None)
                        trace["ptr_expert_active"] = getattr(model, "ptr_expert_active", None)
                        trace["ptr_expert_max_share"] = getattr(model, "ptr_expert_max_share", None)
                        trace["ptr_expert_entropy"] = getattr(model, "ptr_expert_entropy", None)
                        trace["ptr_deadzone"] = model.ptr_deadzone
                        trace["ptr_walk_prob"] = model.ptr_walk_prob
                        trace["update_scale"] = getattr(model, "update_scale", UPDATE_SCALE)
                        trace["ground_speed"] = getattr(model, "ground_speed", None)
                        trace["ground_speed_ema"] = getattr(model, "ground_speed_ema", None)
                        trace["ground_speed_limit"] = getattr(model, "ground_speed_limit", None)
                        if panic_reflex is not None:
                            trace["panic_status"] = panic_status
                    if SYNTH_MODE == "assoc_mix" and shard_info:
                        trace["dom0_top"] = shard_info.get("dom0_top")
                        trace["dom0_share"] = shard_info.get("dom0_share")
                        trace["dom1_top"] = shard_info.get("dom1_top")
                        trace["dom1_share"] = shard_info.get("dom1_share")
                    if getattr(model, "debug_stats", None):
                        trace.update(model.debug_stats)
                    if hasattr(model, "state_loop_entropy"):
                        trace["state_loop_entropy"] = model.state_loop_entropy
                        trace["state_loop_flip_rate"] = getattr(model, "state_loop_flip_rate", None)
                        trace["state_loop_abab_rate"] = getattr(model, "state_loop_abab_rate", None)
                        trace["state_loop_mean_dwell"] = getattr(model, "state_loop_mean_dwell", None)
                        trace["state_loop_max_dwell"] = getattr(model, "state_loop_max_dwell", None)
                    if METABOLIC_TELEMETRY:
                        trace["metabolic_hunger"] = metabolic_stats.get("hgr")
                        trace["metabolic_prune_max_sim"] = metabolic_stats.get("prn")
                        trace["metabolic_prune_pairs"] = metabolic_stats.get("prc")
                        trace["metabolic_prune_pair"] = metabolic_stats.get("prx")
                        trace["metabolic_idle_experts"] = metabolic_stats.get("idl")
                        trace["metabolic_idle_steps"] = METABOLIC_IDLE_STEPS
                    if EXPERT_BUDGET > 0:
                        trace["usage_budget"] = EXPERT_BUDGET
                        trace["usage_ratio"] = getattr(model, "usage_ratio", None)
                        trace["usage_lambda"] = getattr(model, "usage_lambda", None)
                        trace["usage_ema"] = getattr(model, "usage_ema", None)
                        trace["usage_conf"] = getattr(model, "usage_conf", None)
                        trace["usage_gate_ok"] = getattr(model, "usage_gate_ok", None)
                        trace["usage_remap_count"] = getattr(model, "usage_remap_count", None)
                        trace["usage_remap_last"] = getattr(model, "usage_remap_last", None)
                    if HIBERNATE_ENABLED:
                        trace["hibernation_saved"] = getattr(head, "hibernation_saved", 0) if head is not None else 0
                        trace["hibernation_fetched"] = getattr(head, "hibernation_fetched", 0) if head is not None else 0
                        trace["hibernation_corrupt"] = getattr(head, "hibernation_corrupt", 0) if head is not None else 0
                        trace["hibernation_drift"] = getattr(head, "hibernation_drift", 0) if head is not None else 0
                        trace["hibernation_idle_steps"] = HIBERNATE_IDLE_STEPS
                        trace["hibernation_dir"] = HIBERNATE_DIR
                    if DEVICE == "cuda":
                        trace["cuda_mem_alloc_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 2)
                        trace["cuda_mem_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 2)
                    if pointer_hist_sum is not None:
                        hist_np = pointer_hist_sum.cpu().numpy()
                        total = hist_np.sum()
                        if total > 0:
                            probs = hist_np / total
                            entropy = float(-(probs * np.log(probs + 1e-12)).sum())
                        else:
                            entropy = None
                        trace["pointer_entropy"] = entropy
                        trace["pointer_total"] = int(total)
                    try:
                        with open(LIVE_TRACE_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(trace) + "\n")
                    except Exception as e:
                        log(f"live_trace write failed: {e}")
                    last_live_trace = now

            # Explicitly release intermediates to reduce fragmentation risk.
            del outputs, loss
            if pulse_applied:
                model.ptr_walk_prob = prev_walk_prob
            step += 1
            # Respect MAX_STEPS unless explicitly disabled (used by infinite wrapper).
            ignore_max_steps = os.environ.get("VRX_IGNORE_MAX_STEPS") == "1"
            if (not ignore_max_steps) and MAX_STEPS > 0 and step >= MAX_STEPS:
                break
            eval_stats = None
            eval_due = (
                EVAL_EVERY_STEPS > 0
                and eval_loader is not None
                and step % EVAL_EVERY_STEPS == 0
            )
            if eval_due:
                eval_stats = eval_model(model, eval_loader, dataset_name, model_name)
                if eval_stats is not None:
                    acc = eval_stats.get("eval_acc")
                    if acc is not None:
                        model.last_eval_acc = float(acc)
                if ratchet_allowed and eval_stats is not None:
                    acc = eval_stats.get("eval_acc")
                    if acc is not None:
                        if acc >= RATCHET_ACC_MIN:
                            ratchet_streak += 1
                        else:
                            ratchet_streak = 0
                        if ratchet_streak >= RATCHET_STREAK:
                            new_floor = max(RATCHET_BASE, float(acc) * RATCHET_SCALE)
                            if new_floor > getattr(model, "ptr_inertia_floor", 0.0):
                                model.ptr_inertia_floor = new_floor
                                log(
                                    f"Ratchet inertia floor -> {model.ptr_inertia_floor:.2f} "
                                    f"(acc={acc:.4f}, streak={ratchet_streak})"
                                )
                if INERTIA_SIGNAL_ENABLED and eval_stats is not None:
                    acc = eval_stats.get("eval_acc")
                    if acc is not None:
                        if acc >= INERTIA_SIGNAL_ACC_MIN:
                            inertia_signal_streak += 1
                        else:
                            inertia_signal_streak = 0
                        model.ptr_inertia_reward_ready = inertia_signal_streak >= INERTIA_SIGNAL_STREAK
                        model.ptr_inertia_reward_acc = float(acc)
                        model.ptr_inertia_reward_streak = inertia_signal_streak
                        if model.ptr_inertia_reward_ready:
                            target_floor = max(
                                INERTIA_SIGNAL_FLOOR,
                                min(INERTIA_SIGNAL_TARGET, 0.99),
                            )
                            if target_floor > getattr(model, "ptr_inertia_floor", 0.0):
                                model.ptr_inertia_floor = target_floor
                                log(f"Inertia signal floor -> {model.ptr_inertia_floor:.2f} (acc={acc:.4f})")
                if MITOSIS_ENABLED and eval_stats is not None:
                    acc = eval_stats.get("eval_acc")
                    if acc is not None:
                        acc = float(acc)
                        mitosis_acc_history.append(acc)
                        if len(mitosis_acc_history) > MITOSIS_STALL_WINDOW:
                            mitosis_acc_history.pop(0)
                        if len(mitosis_acc_history) >= MITOSIS_STALL_WINDOW:
                            slope = mitosis_acc_history[-1] - mitosis_acc_history[0]
                            imbalance = eval_stats.get("mitosis_expert_imbalance")
                            if imbalance is None:
                                imbalance = getattr(model, "ptr_expert_max_share", None)
                            if imbalance is None:
                                shard_info = getattr(model, "debug_shard_info", None)
                                if shard_info is not None:
                                    imbalance = shard_info.get("expert_imbalance")
                            parent_expert = eval_stats.get("mitosis_parent_expert")
                            hot_addresses = eval_stats.get("mitosis_hot_addresses")
                            if (
                                imbalance is not None
                                and acc < 0.99
                                and slope < MITOSIS_STALL_DELTA
                                and float(imbalance) > MITOSIS_IMBALANCE
                            ):
                                ckpt_path = MITOSIS_CKPT_PATH or CHECKPOINT_PATH
                                ckpt_dir = os.path.dirname(ckpt_path)
                                if ckpt_dir:
                                    os.makedirs(ckpt_dir, exist_ok=True)
                                payload = _checkpoint_payload(model, optimizer, scaler, step, losses)
                                payload["mitosis"] = {
                                    "acc": acc,
                                    "slope": slope,
                                    "imbalance": float(imbalance),
                                    "window": MITOSIS_STALL_WINDOW,
                                    "parent_expert": parent_expert,
                                    "hot_addresses": hot_addresses,
                                }
                                payload["num_experts"] = getattr(model.head, "num_experts", EXPERT_HEADS)
                                torch.save(payload, ckpt_path)
                                meta = {
                                    "checkpoint": ckpt_path,
                                    "acc": acc,
                                    "slope": slope,
                                    "imbalance": float(imbalance),
                                    "window": MITOSIS_STALL_WINDOW,
                                    "parent_expert": parent_expert,
                                    "hot_addresses": hot_addresses,
                                }
                                meta_path = os.path.splitext(ckpt_path)[0] + "_meta.json"
                                try:
                                    with open(meta_path, "w", encoding="utf-8") as f:
                                        json.dump(meta, f, indent=2)
                                except Exception as e:
                                    log(f"mitosis meta write failed: {e}")
                                log(
                                    f"MITOSIS requested: acc={acc:.4f} slope={slope:.4f} "
                                    f"imb={float(imbalance):.3f} -> {ckpt_path}"
                                )
                                raise SystemExit(MITOSIS_EXIT_CODE)
            if SAVE_EVERY_STEPS > 0 and step % SAVE_EVERY_STEPS == 0:
                if EVAL_AT_CHECKPOINT and (not eval_due) and eval_loader is not None:
                    eval_stats = eval_model(model, eval_loader, dataset_name, model_name)
                    if eval_stats is not None:
                        acc = eval_stats.get("eval_acc")
                        if acc is not None:
                            model.last_eval_acc = float(acc)
                    if ratchet_allowed and eval_stats is not None:
                        acc = eval_stats.get("eval_acc")
                        if acc is not None:
                            if acc >= RATCHET_ACC_MIN:
                                ratchet_streak += 1
                            else:
                                ratchet_streak = 0
                            if ratchet_streak >= RATCHET_STREAK:
                                new_floor = max(RATCHET_BASE, float(acc) * RATCHET_SCALE)
                                if new_floor > getattr(model, "ptr_inertia_floor", 0.0):
                                    model.ptr_inertia_floor = new_floor
                                    log(
                                        f"Ratchet inertia floor -> {model.ptr_inertia_floor:.2f} "
                                        f"(acc={acc:.4f}, streak={ratchet_streak})"
                                    )
                    if INERTIA_SIGNAL_ENABLED and eval_stats is not None:
                        acc = eval_stats.get("eval_acc")
                        if acc is not None:
                            if acc >= INERTIA_SIGNAL_ACC_MIN:
                                inertia_signal_streak += 1
                            else:
                                inertia_signal_streak = 0
                            model.ptr_inertia_reward_ready = inertia_signal_streak >= INERTIA_SIGNAL_STREAK
                            model.ptr_inertia_reward_acc = float(acc)
                            model.ptr_inertia_reward_streak = inertia_signal_streak
                            if model.ptr_inertia_reward_ready:
                                target_floor = max(
                                    INERTIA_SIGNAL_FLOOR,
                                    min(INERTIA_SIGNAL_TARGET, 0.99),
                                )
                                if target_floor > getattr(model, "ptr_inertia_floor", 0.0):
                                    model.ptr_inertia_floor = target_floor
                                    log(f"Inertia signal floor -> {model.ptr_inertia_floor:.2f} (acc={acc:.4f})")
                loss_value = losses[-1] if losses else None
                raw_delta = getattr(model, "ptr_delta_raw_mean", None)
                raw_delta_value = float(raw_delta) if raw_delta is not None else None
                grad_value = grad_norm if hasattr(model, "theta_ptr_reduced") else None
                is_finite = _checkpoint_is_finite(loss_value, grad_value, raw_delta_value)
                ckpt = _checkpoint_payload(model, optimizer, scaler, step, losses)
                ckpt["meta"] = {
                    "loss": loss_value,
                    "grad_norm": grad_value,
                    "raw_delta": raw_delta_value,
                    "nonfinite": not is_finite,
                }
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
        # No early break; continue looping until externally stopped.
    # end while

    slope = compute_slope(losses)
    log(f"{dataset_name} | {model_name} | slope {slope:.6f} over {len(losses)} steps")
    ptr_flip_rate = (ptr_flip_sum / ptr_steps) if ptr_steps else None
    ptr_mean_dwell = (ptr_mean_dwell_sum / ptr_steps) if ptr_steps else None
    ptr_delta_abs_mean = (ptr_delta_abs_sum / ptr_steps) if ptr_steps else None
    return {
        "loss_slope": slope,
        "steps": step,
        "losses": losses,
        "pointer_hist": pointer_hist_sum.tolist() if pointer_hist_sum is not None else None,
        "satiety_exits": satiety_exits,
        "ptr_flip_rate": ptr_flip_rate,
        "ptr_pingpong_rate": getattr(model, "ptr_pingpong_rate", None),
        "ptr_max_dwell": ptr_max_dwell,
        "ptr_mean_dwell": ptr_mean_dwell,
        "ptr_delta_abs_mean": ptr_delta_abs_mean,
        "ptr_delta_raw_mean": getattr(model, "ptr_delta_raw_mean", None),
        "state_loop_entropy": getattr(model, "state_loop_entropy", None),
        "state_loop_flip_rate": getattr(model, "state_loop_flip_rate", None),
        "state_loop_abab_rate": getattr(model, "state_loop_abab_rate", None),
        "state_loop_max_dwell": getattr(model, "state_loop_max_dwell", None),
        "state_loop_mean_dwell": getattr(model, "state_loop_mean_dwell", None),
    }

