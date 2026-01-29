"""VRAXION settings.

Environment-driven configuration for the INSTNCT training runner.

This module is intentionally conservative: many other components may depend on
its parsing quirks. Behavior is locked by tests; keep semantics stable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Tuple

import torch


DTMAPS: Mapping[str, torch.dtype] = MappingProxyType(
    {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "amp": torch.float16,
    }
)

PTMAPS: Mapping[str, torch.dtype] = MappingProxyType(
    {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
)

AMPSET = frozenset({"fp16", "bf16", "amp"})
DEVSET = frozenset({"cuda", "cpu"})


def _env_flag(name: str, default: bool) -> bool:
    """Strict legacy env flag.

    Only the literal string "1" (after strip) is treated as True.
    """

    valsix = os.environ.get(name, "1" if default else "0").strip()
    return valsix == "1"


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_str(name: str, default: str = "") -> str:
    valsix = os.environ.get(name, default)
    return valsix.strip()


def _def_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def _def_log(root: str) -> str:
    return os.path.join(root, "logs", "current", "vraxion.log")


def _log_path(root: str) -> str:
    dfltl = _def_log(root)
    patval = _env_str("VAR_LOGGING_PATH", dfltl)
    # NOTE: whitespace becomes empty via _env_str(...).strip(); empty must fall back.
    return patval or dfltl


def _pick_dev() -> str:
    devval = _env_str("VAR_COMPUTE_DEVICE", "").lower()
    if devval in DEVSET:
        return devval
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prec_dt(device: str) -> Tuple[str, torch.dtype, bool]:
    precvl = _env_str("VRX_PRECISION", "fp32").lower()
    dtyval = DTMAPS.get(precvl, torch.float32)

    useamp = device == "cuda" and precvl in AMPSET
    if precvl == "fp64":
        useamp = False

    return precvl, dtyval, useamp


def _ptr_dt() -> torch.dtype:
    namval = _env_str("VRX_PTR_DTYPE", "fp64").lower()
    return PTMAPS.get(namval, torch.float32)


def _flg_ovr(primary: str, alias: str, default: bool = False) -> bool:
    """primary overrides alias (both using strict "1" parsing)."""

    alival = _env_flag(alias, default)
    return _env_flag(primary, alival)


@dataclass(frozen=True)
class Settings:
    root: str
    data_dir: str
    log_path: str

    seed: int
    device: str
    offline_only: bool
    audio_backend: str

    max_samples: int
    eval_samples: int
    eval_split: str
    eval_ptr_deterministic: bool
    batch_size: int
    lr: float
    wall_clock_seconds: int
    max_steps: int
    heartbeat_steps: int
    heartbeat_secs: float
    live_trace_every: int
    satiety_thresh: float

    ring_len: int
    slot_dim: int
    ptr_param_stride: int
    ptr_dtype: torch.dtype
    gauss_k: int
    gauss_tau: float
    ptr_kernel: str
    ptr_kappa: float
    ptr_edge_eps: float
    lambda_move: float
    ptr_inertia: float
    ptr_inertia_auto: bool
    ptr_inertia_min: float
    ptr_inertia_max: float
    ptr_inertia_vel_full: float
    ptr_inertia_ema: float
    ptr_deadzone: float
    ptr_deadzone_tau: float
    ptr_warmup_steps: int
    ptr_walk_prob: float
    context_scale_init: float
    ptr_no_round: bool
    ptr_phantom: bool
    ptr_phantom_off: float
    ptr_phantom_read: bool
    soft_readout: bool
    soft_readout_k: int
    soft_readout_tau: float
    ptr_vel: bool
    ptr_vel_decay: float
    ptr_vel_cap: float
    ptr_vel_scale: float
    ptr_lock: bool
    ptr_lock_value: float
    ptr_update_every: int
    ptr_update_auto: bool
    ptr_update_min: int
    ptr_update_max: int
    ptr_update_every_step: int
    ptr_update_target_flip: float
    ptr_update_ema: float
    ptr_update_governor: bool
    ptr_update_gov_warmup: int
    ptr_update_gov_grad_high: float
    ptr_update_gov_grad_low: float
    ptr_update_gov_loss_flat: float
    ptr_update_gov_loss_spike: float
    ptr_update_gov_step_up: float
    ptr_update_gov_step_down: float
    ptr_gate_mode: str
    ptr_gate_steps: str
    ptr_soft_gate: bool
    ptr_jump_disabled: bool
    ptr_jump_cap: float

    thermo_enabled: bool
    thermo_every: int
    thermo_target_flip: float
    thermo_ema: float
    thermo_inertia_step: float
    thermo_deadzone_step: float
    thermo_walk_step: float
    thermo_inertia_min: float
    thermo_inertia_max: float
    thermo_deadzone_min: float
    thermo_deadzone_max: float
    thermo_walk_min: float
    thermo_walk_max: float

    panic_enabled: bool
    panic_threshold: float
    panic_beta: float
    panic_recovery: float
    panic_inertia_low: float
    panic_inertia_high: float
    panic_walk_max: float

    mobius_enabled: bool
    mobius_emb_scale: float
    act_name: str
    c13_p: float

    debug_nan: bool
    debug_stats: bool
    debug_every: int

    precision: str
    dtype: torch.dtype
    use_amp: bool

    mi_shuffle: bool
    state_loop_metrics: bool
    state_loop_every: int
    state_loop_samples: int
    state_loop_dim: int
    grad_clip: float
    state_clip: float
    state_decay: float
    update_scale: float
    agc_scale_min: float
    agc_scale_max_min: float
    agc_scale_max_decay: float
    agc_plateau_window: int
    agc_plateau_std: float
    agc_plateau_min_steps: int
    agc_enabled: bool
    agc_grad_low: float
    agc_grad_high: float
    agc_scale_max: float
    speed_gov_enabled: bool
    speed_gov_rho: float
    speed_gov_l_min: float
    speed_gov_l_max: float
    speed_gov_l_ema: float
    speed_gov_l_k: float
    live_trace_path: str
    run_mode: str

    checkpoint_path: str
    save_every_steps: int
    save_history: bool
    save_last_good: bool
    save_bad: bool
    resume: bool
    loss_keep: int

    phase_a_steps: int
    phase_b_steps: int
    synth_len: int
    synth_shuffle: bool
    synth_mode: str
    hand_min: int
    assoc_keys: int
    assoc_pairs: int
    assoc_val_range: int

    evo_pop: int
    evo_gens: int
    evo_steps: int
    evo_mut_std: float
    evo_pointer_only: bool
    evo_checkpoint_every: int
    evo_resume: bool
    evo_checkpoint_individual: bool
    evo_progress: bool
    train_trace: bool
    train_trace_path: str


def load_settings() -> Settings:
    # ---- Paths ----
    root = _env_str("VAR_PROJECT_ROOT", _def_root())
    data_dir = os.path.join(root, "data")
    log_path = _log_path(root)

    # ---- Runtime ----
    seed = _env_int("VAR_RUN_SEED", 123)
    device = _pick_dev()
    offline_only = _env_flag("VRX_OFFLINE_ONLY", True)
    audio_backend = _env_str("VAR_TORCHAUDIO_BACKEND", "")

    # ---- Core training loop ----
    max_samples = _env_int("VRX_MAX_SAMPLES", 5000)
    eval_samples = _env_int("VRX_EVAL_SAMPLES", 1024)
    eval_split = _env_str("VRX_EVAL_SPLIT", "test").lower()
    eval_ptr_deterministic = _env_flag("VRX_EVAL_PTR_DETERMINISTIC", True)
    batch_size = _env_int("VRX_BATCH_SIZE", 16)
    lr = _env_float("VRX_LR", 1e-3)
    wall_clock_seconds = _env_int("VRX_WALL", 15 * 60)
    max_steps = _env_int("VRX_MAX_STEPS", 0)

    # Default to per-step heartbeat/logging; callers can raise via env if desired.
    heartbeat_steps = _env_int("VAR_LOG_EVERY_N_STEPS", 1)
    heartbeat_secs = _env_float("VAR_LOG_EVERY_N_SECS", 0.0)
    live_trace_every = _env_int("VAR_LIVE_TRACE_EVERY_N_STEPS", heartbeat_steps)
    satiety_thresh = _env_float("VRX_SATIETY", 0.98)

    # ---- Pointer / ring buffer ----
    ring_len = _env_int("VRX_RING_LEN", 8192)
    slot_dim = _env_int("VRX_SLOT_DIM", 576)
    ptr_param_stride = _env_int("VRX_PTR_STRIDE", 1)

    gauss_k = _env_int("VRX_GAUSS_K", 2)
    gauss_tau = _env_float("VRX_GAUSS_TAU", 0.5)
    ptr_kernel = _env_str("VRX_PTR_KERNEL", "gauss").lower()
    ptr_kappa = _env_float("VRX_PTR_KAPPA", 4.0)
    ptr_edge_eps = _env_float("VRX_PTR_EDGE_EPS", 0.0)
    lambda_move = _env_float("VRX_LMOVE", 1e-3)

    ptr_inertia = _env_float("VRX_PTR_INERTIA", 0.0)
    ptr_inertia_auto = _env_flag("VRX_PTR_INERTIA_AUTO", False)
    ptr_inertia_min = _env_float("VRX_PTR_INERTIA_MIN", 0.5)
    ptr_inertia_max = _env_float("VRX_PTR_INERTIA_MAX", 0.9)
    ptr_inertia_vel_full = _env_float("VRX_PTR_INERTIA_VEL_FULL", 0.5)
    ptr_inertia_ema = _env_float("VRX_PTR_INERTIA_EMA", 0.9)

    ptr_deadzone = _env_float("VRX_PTR_DEADZONE", 0.0)
    ptr_deadzone_tau = _env_float("VRX_PTR_DEADZONE_TAU", 1e-3)
    ptr_warmup_steps = _env_int("VRX_PTR_WARMUP_STEPS", 0)
    ptr_walk_prob = _env_float("PARAM_POINTER_FORWARD_STEP_PROB", 0.2)
    context_scale_init = _env_float("VRX_CONTEXT_SCALE_INIT", 0.2)

    ptr_no_round = _env_flag("VRX_PTR_NO_ROUND", False)
    ptr_phantom = _env_flag("VRX_PTR_PHANTOM", False)
    ptr_phantom_off = _env_float("VRX_PTR_PHANTOM_OFF", 0.5)
    ptr_phantom_read = _env_flag("VRX_PTR_PHANTOM_READ", False)

    soft_readout = _env_flag("VRX_SOFT_READOUT", False)
    soft_readout_k = _env_int("VRX_SOFT_READOUT_K", 2)
    soft_readout_tau = _env_float("VRX_SOFT_READOUT_TAU", gauss_tau)

    ptr_vel = _env_flag("VRX_PTR_VEL", False)
    ptr_vel_decay = _env_float("VRX_PTR_VEL_DECAY", 0.9)
    ptr_vel_cap = _env_float("VRX_PTR_VEL_CAP", 0.5)
    ptr_vel_scale = _env_float("VRX_PTR_VEL_SCALE", 1.0)

    ptr_lock = _env_flag("VRX_PTR_LOCK", False)
    ptr_lock_value = _env_float("VRX_PTR_LOCK_VALUE", 0.5)

    ptr_update_every = _env_int("VRX_PTR_UPDATE_EVERY", 1)
    ptr_update_auto = _env_flag("VRX_PTR_UPDATE_AUTO", False)
    ptr_update_min = _env_int("VRX_PTR_UPDATE_MIN", 1)
    ptr_update_max = _env_int("VRX_PTR_UPDATE_MAX", 16)
    ptr_update_every_step = _env_int("VRX_PTR_UPDATE_EVERY_STEP", 20)
    ptr_update_target_flip = _env_float("VRX_PTR_UPDATE_TARGET_FLIP", 0.2)
    ptr_update_ema = _env_float("VRX_PTR_UPDATE_EMA", 0.9)

    ptr_update_governor = _env_flag("VRX_PTR_UPDATE_GOV", False)
    ptr_update_gov_warmup = _env_int("VRX_PTR_UPDATE_GOV_WARMUP", ptr_warmup_steps)
    ptr_update_gov_grad_high = _env_float("VRX_PTR_UPDATE_GOV_GRAD_HIGH", 45.0)
    ptr_update_gov_grad_low = _env_float("VRX_PTR_UPDATE_GOV_GRAD_LOW", 2.0)
    ptr_update_gov_loss_flat = _env_float("VRX_PTR_UPDATE_GOV_LOSS_FLAT", 0.001)
    ptr_update_gov_loss_spike = _env_float("VRX_PTR_UPDATE_GOV_LOSS_SPIKE", 0.1)
    ptr_update_gov_step_up = _env_float("VRX_PTR_UPDATE_GOV_STEP_UP", 0.5)
    ptr_update_gov_step_down = _env_float("VRX_PTR_UPDATE_GOV_STEP_DOWN", 0.2)

    ptr_gate_mode = _env_str("VRX_PTR_GATE_MODE", "none").lower()
    ptr_gate_steps = _env_str("VRX_PTR_GATE_STEPS", "")
    ptr_soft_gate = _env_flag("VRX_PTR_SOFT_GATE", False)
    ptr_jump_disabled = _env_flag("VRX_PTR_JUMP_DISABLED", False)
    ptr_jump_cap = _env_float("VRX_PTR_JUMP_CAP", 1.0)

    # ---- Thermo / panic governors ----
    thermo_enabled = _flg_ovr("VRX_THERMO", "VRX_THERMO_ENABLED")
    thermo_every = _env_int("VRX_THERMO_EVERY", 20)
    thermo_target_flip = _env_float("VRX_THERMO_TARGET_FLIP", 0.2)
    thermo_ema = _env_float("VRX_THERMO_EMA", 0.9)
    thermo_inertia_step = _env_float("VRX_THERMO_INERTIA_STEP", 0.05)
    thermo_deadzone_step = _env_float("VRX_THERMO_DEADZONE_STEP", 0.02)
    thermo_walk_step = _env_float("VRX_THERMO_WALK_STEP", 0.02)
    thermo_inertia_min = _env_float("VRX_THERMO_INERTIA_MIN", 0.0)
    thermo_inertia_max = _env_float("VRX_THERMO_INERTIA_MAX", 0.95)
    thermo_deadzone_min = _env_float("VRX_THERMO_DEADZONE_MIN", 0.0)
    thermo_deadzone_max = _env_float("VRX_THERMO_DEADZONE_MAX", 0.5)
    thermo_walk_min = _env_float("VRX_THERMO_WALK_MIN", 0.0)
    thermo_walk_max = _env_float("VRX_THERMO_WALK_MAX", 0.3)

    panic_enabled = _flg_ovr("VRX_PANIC", "VRX_PANIC_ENABLED")
    panic_threshold = _env_float("VRX_PANIC_THRESHOLD", 1.5)
    panic_beta = _env_float("VRX_PANIC_BETA", 0.9)
    panic_recovery = _env_float("VRX_PANIC_RECOVERY", 0.01)
    panic_inertia_low = _env_float("VRX_PANIC_INERTIA_LOW", 0.1)
    panic_inertia_high = _env_float("VRX_PANIC_INERTIA_HIGH", 0.95)
    panic_walk_max = _env_float("VRX_PANIC_WALK_MAX", 0.2)

    # ---- Model toggles ----
    mobius_enabled = _env_flag("VRX_MOBIUS", False)
    mobius_emb_scale = _env_float("VRX_MOBIUS_EMB", 0.1)
    act_name = _env_str("VRX_ACT", "c19").lower()
    c13_p = _env_float("VRX_C13_P", 2.0)

    # ---- Debug ----
    debug_nan = _env_flag("VRX_DEBUG_NAN", False)
    debug_stats = _env_flag("VRX_DEBUG_STATS", False)
    debug_every = _env_int("VRX_DEBUG_EVERY", 0)

    # ---- Precision / AMP ----
    precision, dtype, use_amp = _prec_dt(device)
    ptr_dtype = _ptr_dt()

    # ---- Monitoring / governors / misc ----
    mi_shuffle = _env_flag("VRX_MI_SHUFFLE", False)
    state_loop_metrics = _env_flag("VRX_STATE_LOOP_METRICS", False)
    state_loop_every = _env_int("VRX_STATE_LOOP_EVERY", 1)
    state_loop_samples = _env_int("VRX_STATE_LOOP_SAMPLES", 0)
    state_loop_dim = _env_int("VRX_STATE_LOOP_DIM", 16)
    grad_clip = _env_float("VRX_GRAD_CLIP", 0.0)
    state_clip = _env_float("VRX_STATE_CLIP", 0.0)
    state_decay = _env_float("VRX_STATE_DECAY", 1.0)
    update_scale = _env_float("VRX_UPDATE_SCALE", 1.0)

    agc_scale_min = _env_float("VRX_SCALE_MIN", 0.01)
    agc_scale_max_min = _env_float("VRX_SCALE_MAX_MIN", 1e-4)
    agc_scale_max_decay = _env_float("VRX_SCALE_MAX_DECAY", 0.3)
    agc_plateau_window = _env_int("VRX_PLATEAU_WINDOW", 200)
    agc_plateau_std = _env_float("VRX_PLATEAU_STD", 0.02)
    agc_plateau_min_steps = _env_int("VRX_PLATEAU_MIN_STEPS", 400)
    agc_scale_max = _env_float("VRX_SCALE_MAX", 1.0)
    agc_enabled = _env_flag("VRX_AGC_ENABLED", True)
    agc_grad_low = _env_float("VRX_AGC_GRAD_LOW", 1.0)
    agc_grad_high = _env_float("VRX_AGC_GRAD_HIGH", 5.0)

    speed_gov_enabled = _env_flag("VRX_SPEED_GOV", False)
    speed_gov_rho = _env_float("VRX_SPEED_GOV_RHO", 4.0)
    speed_gov_l_min = _env_float("VRX_SPEED_GOV_L_MIN", 5.0)
    speed_gov_l_max = _env_float("VRX_SPEED_GOV_L_MAX", 60.0)
    speed_gov_l_ema = _env_float("VRX_SPEED_GOV_L_EMA", 0.9)
    speed_gov_l_k = _env_float("VRX_SPEED_GOV_L_K", 2.0)

    live_trace_path = _env_str(
        "VAR_LIVE_TRACE_PATH",
        os.path.join(root, "traces", "current", "live_trace.json"),
    )
    run_mode = _env_str("VRX_MODE", "train")

    # ---- Checkpoint / saving ----
    checkpoint_path = _env_str("VRX_CKPT", os.path.join(root, "checkpoint.pt"))
    save_every_steps = _env_int("VRX_SAVE_EVERY", 100)
    save_history = _env_flag("VRX_SAVE_HISTORY", True)
    save_last_good = _env_flag("VRX_SAVE_LAST_GOOD", True)
    save_bad = _env_flag("VRX_SAVE_BAD", True)
    resume = _env_flag("VRX_RESUME", False)
    loss_keep = _env_int("VAR_LOSS_HISTORY_LEN", 2000)

    # ---- Synthesis / dataset tasks ----
    phase_a_steps = _env_int("VRX_PHASE_A_STEPS", 50)
    phase_b_steps = _env_int("VRX_PHASE_B_STEPS", 50)
    synth_len = _env_int("VRX_SYNTH_LEN", 256)
    synth_shuffle = _env_flag("VRX_SYNTH_SHUFFLE", False)
    synth_mode = _env_str("VRX_SYNTH_MODE", "random").lower()
    hand_min = _env_int("VRX_HAND_MIN", 256)
    assoc_keys = _env_int("VRX_ASSOC_KEYS", 4)
    assoc_pairs = _env_int("VRX_ASSOC_PAIRS", 3)
    assoc_val_range = _env_int("VRX_ASSOC_VAL_RANGE", 256)

    # ---- Evolutionary ----
    evo_pop = _env_int("VRX_EVO_POP", 6)
    evo_gens = _env_int("VRX_EVO_GENS", 3)
    evo_steps = _env_int("VRX_EVO_STEPS", 100)
    evo_mut_std = _env_float("VRX_EVO_MUT_STD", 0.02)
    evo_pointer_only = _env_flag("VRX_EVO_POINTER_ONLY", False)
    evo_checkpoint_every = _env_int("VRX_EVO_CKPT_EVERY", 1)
    evo_resume = _env_flag("VRX_EVO_RESUME", False)
    evo_checkpoint_individual = _env_flag("VRX_EVO_CKPT_INDIV", True)
    evo_progress = _env_flag("VRX_EVO_PROGRESS", True)

    train_trace = _env_flag("VAR_TRAINING_TRACE_ENABLED", False)
    train_trace_path = _env_str(
        "VAR_TRAINING_TRACE_PATH",
        os.path.join(root, "traces", "current", "train_steps_trace.jsonl"),
    )

    locmap = locals()
    kwdsix = {namkey: locmap[namkey] for namkey in Settings.__dataclass_fields__}
    return Settings(**kwdsix)


__all__ = [
    "Settings",
    "load_settings",
]

