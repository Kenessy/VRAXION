import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "1" if default else "0").strip()
    return value == "1"


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


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
    root = _env_str("VAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    data_dir = os.path.join(root, "data")
    default_log_path = os.path.join(root, "logs", "current", "tournament_phase6.log")
    log_path = _env_str("VAR_LOGGING_PATH", default_log_path)
    if not log_path:
        log_path = default_log_path

    seed = _env_int("VAR_RUN_SEED", 123)
    device = _env_str("VAR_COMPUTE_DEVICE", "").lower()
    if device not in {"cuda", "cpu"}:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    offline_only = _env_flag("PILOT_OFFLINE", True)
    audio_backend = _env_str("VAR_TORCHAUDIO_BACKEND", "")

    max_samples = _env_int("TP6_MAX_SAMPLES", 5000)
    eval_samples = _env_int("TP6_EVAL_SAMPLES", 1024)
    eval_split = _env_str("TP6_EVAL_SPLIT", "test").lower()
    eval_ptr_deterministic = _env_flag("TP6_EVAL_PTR_DETERMINISTIC", True)
    batch_size = _env_int("TP6_BATCH_SIZE", 128)
    lr = _env_float("TP6_LR", 1e-3)
    wall_clock_seconds = _env_int("TP6_WALL", 15 * 60)
    max_steps = _env_int("TP6_MAX_STEPS", 0)
    # Default to per-step heartbeat/logging; callers can raise via env if desired.
    heartbeat_steps = _env_int("VAR_LOG_EVERY_N_STEPS", 1)
    heartbeat_secs = _env_float("VAR_LOG_EVERY_N_SECS", 0.0)
    live_trace_every = _env_int("VAR_LIVE_TRACE_EVERY_N_STEPS", heartbeat_steps)
    satiety_thresh = _env_float("TP6_SATIETY", 0.98)

    ring_len = _env_int("TP6_RING_LEN", 4096)
    slot_dim = _env_int("TP6_SLOT_DIM", 8)
    ptr_param_stride = _env_int("TP6_PTR_STRIDE", 1)
    gauss_k = _env_int("TP6_GAUSS_K", 2)
    gauss_tau = _env_float("TP6_GAUSS_TAU", 0.5)
    ptr_kernel = _env_str("TP6_PTR_KERNEL", "gauss").lower()
    ptr_kappa = _env_float("TP6_PTR_KAPPA", 4.0)
    ptr_edge_eps = _env_float("TP6_PTR_EDGE_EPS", 0.0)
    lambda_move = _env_float("TP6_LMOVE", 1e-3)
    ptr_inertia = _env_float("TP6_PTR_INERTIA", 0.0)
    ptr_inertia_auto = _env_flag("TP6_PTR_INERTIA_AUTO", False)
    ptr_inertia_min = _env_float("TP6_PTR_INERTIA_MIN", 0.5)
    ptr_inertia_max = _env_float("TP6_PTR_INERTIA_MAX", 0.9)
    ptr_inertia_vel_full = _env_float("TP6_PTR_INERTIA_VEL_FULL", 0.5)
    ptr_inertia_ema = _env_float("TP6_PTR_INERTIA_EMA", 0.9)
    ptr_deadzone = _env_float("TP6_PTR_DEADZONE", 0.0)
    ptr_deadzone_tau = _env_float("TP6_PTR_DEADZONE_TAU", 1e-3)
    ptr_warmup_steps = _env_int("TP6_PTR_WARMUP_STEPS", 0)
    ptr_walk_prob = _env_float("PARAM_POINTER_FORWARD_STEP_PROB", 0.2)
    context_scale_init = _env_float("TP6_CONTEXT_SCALE_INIT", 0.2)
    ptr_no_round = _env_flag("TP6_PTR_NO_ROUND", False)
    ptr_phantom = _env_flag("TP6_PTR_PHANTOM", False)
    ptr_phantom_off = _env_float("TP6_PTR_PHANTOM_OFF", 0.5)
    ptr_phantom_read = _env_flag("TP6_PTR_PHANTOM_READ", False)
    soft_readout = _env_flag("TP6_SOFT_READOUT", False)
    soft_readout_k = _env_int("TP6_SOFT_READOUT_K", 2)
    soft_readout_tau = _env_float("TP6_SOFT_READOUT_TAU", gauss_tau)
    ptr_vel = _env_flag("TP6_PTR_VEL", False)
    ptr_vel_decay = _env_float("TP6_PTR_VEL_DECAY", 0.9)
    ptr_vel_cap = _env_float("TP6_PTR_VEL_CAP", 0.5)
    ptr_vel_scale = _env_float("TP6_PTR_VEL_SCALE", 1.0)
    ptr_lock = _env_flag("TP6_PTR_LOCK", False)
    ptr_lock_value = _env_float("TP6_PTR_LOCK_VALUE", 0.5)
    ptr_update_every = _env_int("TP6_PTR_UPDATE_EVERY", 1)
    ptr_update_auto = _env_flag("TP6_PTR_UPDATE_AUTO", False)
    ptr_update_min = _env_int("TP6_PTR_UPDATE_MIN", 1)
    ptr_update_max = _env_int("TP6_PTR_UPDATE_MAX", 16)
    ptr_update_every_step = _env_int("TP6_PTR_UPDATE_EVERY_STEP", 20)
    ptr_update_target_flip = _env_float("TP6_PTR_UPDATE_TARGET_FLIP", 0.2)
    ptr_update_ema = _env_float("TP6_PTR_UPDATE_EMA", 0.9)
    ptr_update_governor = _env_flag("TP6_PTR_UPDATE_GOV", False)
    ptr_update_gov_warmup = _env_int("TP6_PTR_UPDATE_GOV_WARMUP", ptr_warmup_steps)
    ptr_update_gov_grad_high = _env_float("TP6_PTR_UPDATE_GOV_GRAD_HIGH", 45.0)
    ptr_update_gov_grad_low = _env_float("TP6_PTR_UPDATE_GOV_GRAD_LOW", 2.0)
    ptr_update_gov_loss_flat = _env_float("TP6_PTR_UPDATE_GOV_LOSS_FLAT", 0.001)
    ptr_update_gov_loss_spike = _env_float("TP6_PTR_UPDATE_GOV_LOSS_SPIKE", 0.1)
    ptr_update_gov_step_up = _env_float("TP6_PTR_UPDATE_GOV_STEP_UP", 0.5)
    ptr_update_gov_step_down = _env_float("TP6_PTR_UPDATE_GOV_STEP_DOWN", 0.2)
    ptr_gate_mode = _env_str("TP6_PTR_GATE_MODE", "none").lower()
    ptr_gate_steps = _env_str("TP6_PTR_GATE_STEPS", "")
    ptr_soft_gate = _env_flag("TP6_PTR_SOFT_GATE", False)
    ptr_jump_disabled = _env_flag("TP6_PTR_JUMP_DISABLED", False)
    ptr_jump_cap = _env_float("TP6_PTR_JUMP_CAP", 1.0)

    thermo_enabled = _env_flag("TP6_THERMO", _env_flag("TP6_THERMO_ENABLED", False))
    thermo_every = _env_int("TP6_THERMO_EVERY", 20)
    thermo_target_flip = _env_float("TP6_THERMO_TARGET_FLIP", 0.2)
    thermo_ema = _env_float("TP6_THERMO_EMA", 0.9)
    thermo_inertia_step = _env_float("TP6_THERMO_INERTIA_STEP", 0.05)
    thermo_deadzone_step = _env_float("TP6_THERMO_DEADZONE_STEP", 0.02)
    thermo_walk_step = _env_float("TP6_THERMO_WALK_STEP", 0.02)
    thermo_inertia_min = _env_float("TP6_THERMO_INERTIA_MIN", 0.0)
    thermo_inertia_max = _env_float("TP6_THERMO_INERTIA_MAX", 0.95)
    thermo_deadzone_min = _env_float("TP6_THERMO_DEADZONE_MIN", 0.0)
    thermo_deadzone_max = _env_float("TP6_THERMO_DEADZONE_MAX", 0.5)
    thermo_walk_min = _env_float("TP6_THERMO_WALK_MIN", 0.0)
    thermo_walk_max = _env_float("TP6_THERMO_WALK_MAX", 0.3)

    panic_enabled = _env_flag("TP6_PANIC", _env_flag("TP6_PANIC_ENABLED", False))
    panic_threshold = _env_float("TP6_PANIC_THRESHOLD", 1.5)
    panic_beta = _env_float("TP6_PANIC_BETA", 0.9)
    panic_recovery = _env_float("TP6_PANIC_RECOVERY", 0.01)
    panic_inertia_low = _env_float("TP6_PANIC_INERTIA_LOW", 0.1)
    panic_inertia_high = _env_float("TP6_PANIC_INERTIA_HIGH", 0.95)
    panic_walk_max = _env_float("TP6_PANIC_WALK_MAX", 0.2)

    mobius_enabled = _env_flag("TP6_MOBIUS", False)
    mobius_emb_scale = _env_float("TP6_MOBIUS_EMB", 0.1)
    act_name = _env_str("TP6_ACT", "c19").lower()
    c13_p = _env_float("TP6_C13_P", 2.0)

    debug_nan = _env_flag("TP6_DEBUG_NAN", False)
    debug_stats = _env_flag("TP6_DEBUG_STATS", False)
    debug_every = _env_int("TP6_DEBUG_EVERY", 0)

    precision = _env_str("TP6_PRECISION", "fp32").lower()
    dtype_map: Dict[str, torch.dtype] = {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "amp": torch.float16,
    }
    dtype = dtype_map.get(precision, torch.float32)
    use_amp = device == "cuda" and precision in {"fp16", "bf16", "amp"}
    if precision == "fp64":
        use_amp = False
    ptr_dtype_name = _env_str("TP6_PTR_DTYPE", "fp32").lower()
    ptr_dtype_map: Dict[str, torch.dtype] = {
        "fp64": torch.float64,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    ptr_dtype = ptr_dtype_map.get(ptr_dtype_name, torch.float32)

    mi_shuffle = _env_flag("TP6_MI_SHUFFLE", False)
    state_loop_metrics = _env_flag("TP6_STATE_LOOP_METRICS", False)
    state_loop_every = _env_int("TP6_STATE_LOOP_EVERY", 1)
    state_loop_samples = _env_int("TP6_STATE_LOOP_SAMPLES", 0)
    state_loop_dim = _env_int("TP6_STATE_LOOP_DIM", 16)
    grad_clip = _env_float("TP6_GRAD_CLIP", 0.0)
    state_clip = _env_float("TP6_STATE_CLIP", 0.0)
    state_decay = _env_float("TP6_STATE_DECAY", 1.0)
    update_scale = _env_float("TP6_UPDATE_SCALE", 1.0)
    agc_scale_min = _env_float("TP6_SCALE_MIN", 0.01)
    agc_scale_max_min = _env_float("TP6_SCALE_MAX_MIN", 1e-4)
    agc_scale_max_decay = _env_float("TP6_SCALE_MAX_DECAY", 0.3)
    agc_plateau_window = _env_int("TP6_PLATEAU_WINDOW", 200)
    agc_plateau_std = _env_float("TP6_PLATEAU_STD", 0.02)
    agc_plateau_min_steps = _env_int("TP6_PLATEAU_MIN_STEPS", 400)
    agc_scale_max = _env_float("TP6_SCALE_MAX", 1.0)
    agc_enabled = _env_flag("TP6_AGC_ENABLED", True)
    agc_grad_low = _env_float("TP6_AGC_GRAD_LOW", 1.0)
    agc_grad_high = _env_float("TP6_AGC_GRAD_HIGH", 5.0)
    speed_gov_enabled = _env_flag("TP6_SPEED_GOV", False)
    speed_gov_rho = _env_float("TP6_SPEED_GOV_RHO", 4.0)
    speed_gov_l_min = _env_float("TP6_SPEED_GOV_L_MIN", 5.0)
    speed_gov_l_max = _env_float("TP6_SPEED_GOV_L_MAX", 60.0)
    speed_gov_l_ema = _env_float("TP6_SPEED_GOV_L_EMA", 0.9)
    speed_gov_l_k = _env_float("TP6_SPEED_GOV_L_K", 2.0)
    live_trace_path = _env_str("VAR_LIVE_TRACE_PATH", os.path.join(root, "traces", "current", "live_trace.json"))
    run_mode = _env_str("TP6_MODE", "train")

    checkpoint_path = _env_str("TP6_CKPT", os.path.join(root, "checkpoint.pt"))
    save_every_steps = _env_int("TP6_SAVE_EVERY", 100)
    save_history = _env_flag("TP6_SAVE_HISTORY", True)
    save_last_good = _env_flag("TP6_SAVE_LAST_GOOD", True)
    save_bad = _env_flag("TP6_SAVE_BAD", True)
    resume = _env_flag("TP6_RESUME", False)
    loss_keep = _env_int("VAR_LOSS_HISTORY_LEN", 2000)

    phase_a_steps = _env_int("TP6_PHASE_A_STEPS", 50)
    phase_b_steps = _env_int("TP6_PHASE_B_STEPS", 50)
    synth_len = _env_int("TP6_SYNTH_LEN", 256)
    synth_shuffle = _env_flag("TP6_SYNTH_SHUFFLE", False)
    synth_mode = _env_str("TP6_SYNTH_MODE", "random").lower()
    hand_min = _env_int("TP6_HAND_MIN", 256)
    assoc_keys = _env_int("TP6_ASSOC_KEYS", 4)
    assoc_pairs = _env_int("TP6_ASSOC_PAIRS", 3)
    assoc_val_range = _env_int("TP6_ASSOC_VAL_RANGE", 256)

    evo_pop = _env_int("TP6_EVO_POP", 6)
    evo_gens = _env_int("TP6_EVO_GENS", 3)
    evo_steps = _env_int("TP6_EVO_STEPS", 100)
    evo_mut_std = _env_float("TP6_EVO_MUT_STD", 0.02)
    evo_pointer_only = _env_flag("TP6_EVO_POINTER_ONLY", False)
    evo_checkpoint_every = _env_int("TP6_EVO_CKPT_EVERY", 1)
    evo_resume = _env_flag("TP6_EVO_RESUME", False)
    evo_checkpoint_individual = _env_flag("TP6_EVO_CKPT_INDIV", True)
    evo_progress = _env_flag("TP6_EVO_PROGRESS", True)
    train_trace = _env_flag("VAR_TRAINING_TRACE_ENABLED", False)
    train_trace_path = _env_str(
        "VAR_TRAINING_TRACE_PATH",
        os.path.join(root, "traces", "current", "train_steps_trace.jsonl"),
    )

    return Settings(
        root=root,
        data_dir=data_dir,
        log_path=log_path,
        seed=seed,
        device=device,
        offline_only=offline_only,
        audio_backend=audio_backend,
        max_samples=max_samples,
        eval_samples=eval_samples,
        eval_split=eval_split,
        eval_ptr_deterministic=eval_ptr_deterministic,
        batch_size=batch_size,
        lr=lr,
        wall_clock_seconds=wall_clock_seconds,
        max_steps=max_steps,
        heartbeat_steps=heartbeat_steps,
        heartbeat_secs=heartbeat_secs,
        live_trace_every=live_trace_every,
        satiety_thresh=satiety_thresh,
        ring_len=ring_len,
        slot_dim=slot_dim,
        ptr_param_stride=ptr_param_stride,
        ptr_dtype=ptr_dtype,
        gauss_k=gauss_k,
        gauss_tau=gauss_tau,
        ptr_kernel=ptr_kernel,
        ptr_kappa=ptr_kappa,
        ptr_edge_eps=ptr_edge_eps,
        lambda_move=lambda_move,
        ptr_inertia=ptr_inertia,
        ptr_inertia_auto=ptr_inertia_auto,
        ptr_inertia_min=ptr_inertia_min,
        ptr_inertia_max=ptr_inertia_max,
        ptr_inertia_vel_full=ptr_inertia_vel_full,
        ptr_inertia_ema=ptr_inertia_ema,
        ptr_deadzone=ptr_deadzone,
        ptr_deadzone_tau=ptr_deadzone_tau,
        ptr_warmup_steps=ptr_warmup_steps,
        ptr_walk_prob=ptr_walk_prob,
        ptr_no_round=ptr_no_round,
        ptr_phantom=ptr_phantom,
        ptr_phantom_off=ptr_phantom_off,
        ptr_phantom_read=ptr_phantom_read,
        soft_readout=soft_readout,
        soft_readout_k=soft_readout_k,
        soft_readout_tau=soft_readout_tau,
        ptr_vel=ptr_vel,
        ptr_vel_decay=ptr_vel_decay,
        ptr_vel_cap=ptr_vel_cap,
        ptr_vel_scale=ptr_vel_scale,
        ptr_lock=ptr_lock,
        ptr_lock_value=ptr_lock_value,
        ptr_update_every=ptr_update_every,
        ptr_update_auto=ptr_update_auto,
        ptr_update_min=ptr_update_min,
        ptr_update_max=ptr_update_max,
        ptr_update_every_step=ptr_update_every_step,
        ptr_update_target_flip=ptr_update_target_flip,
        ptr_update_ema=ptr_update_ema,
        ptr_update_governor=ptr_update_governor,
        ptr_update_gov_warmup=ptr_update_gov_warmup,
        ptr_update_gov_grad_high=ptr_update_gov_grad_high,
        ptr_update_gov_grad_low=ptr_update_gov_grad_low,
        ptr_update_gov_loss_flat=ptr_update_gov_loss_flat,
        ptr_update_gov_loss_spike=ptr_update_gov_loss_spike,
        ptr_update_gov_step_up=ptr_update_gov_step_up,
        ptr_update_gov_step_down=ptr_update_gov_step_down,
        ptr_gate_mode=ptr_gate_mode,
        ptr_gate_steps=ptr_gate_steps,
        ptr_soft_gate=ptr_soft_gate,
        ptr_jump_disabled=ptr_jump_disabled,
        ptr_jump_cap=ptr_jump_cap,
        context_scale_init=context_scale_init,
        thermo_enabled=thermo_enabled,
        thermo_every=thermo_every,
        thermo_target_flip=thermo_target_flip,
        thermo_ema=thermo_ema,
        thermo_inertia_step=thermo_inertia_step,
        thermo_deadzone_step=thermo_deadzone_step,
        thermo_walk_step=thermo_walk_step,
        thermo_inertia_min=thermo_inertia_min,
        thermo_inertia_max=thermo_inertia_max,
        thermo_deadzone_min=thermo_deadzone_min,
        thermo_deadzone_max=thermo_deadzone_max,
        thermo_walk_min=thermo_walk_min,
        thermo_walk_max=thermo_walk_max,
        panic_enabled=panic_enabled,
        panic_threshold=panic_threshold,
        panic_beta=panic_beta,
        panic_recovery=panic_recovery,
        panic_inertia_low=panic_inertia_low,
        panic_inertia_high=panic_inertia_high,
        panic_walk_max=panic_walk_max,
        mobius_enabled=mobius_enabled,
        mobius_emb_scale=mobius_emb_scale,
        act_name=act_name,
        c13_p=c13_p,
        debug_nan=debug_nan,
        debug_stats=debug_stats,
        debug_every=debug_every,
        precision=precision,
        dtype=dtype,
        use_amp=use_amp,
        mi_shuffle=mi_shuffle,
        state_loop_metrics=state_loop_metrics,
        state_loop_every=state_loop_every,
        state_loop_samples=state_loop_samples,
        state_loop_dim=state_loop_dim,
        grad_clip=grad_clip,
        state_clip=state_clip,
        state_decay=state_decay,
        update_scale=update_scale,
        agc_scale_min=agc_scale_min,
        agc_scale_max=agc_scale_max,
        agc_scale_max_min=agc_scale_max_min,
        agc_scale_max_decay=agc_scale_max_decay,
        agc_plateau_window=agc_plateau_window,
        agc_plateau_std=agc_plateau_std,
        agc_plateau_min_steps=agc_plateau_min_steps,
        agc_enabled=agc_enabled,
        agc_grad_low=agc_grad_low,
        agc_grad_high=agc_grad_high,
        speed_gov_enabled=speed_gov_enabled,
        speed_gov_rho=speed_gov_rho,
        speed_gov_l_min=speed_gov_l_min,
        speed_gov_l_max=speed_gov_l_max,
        speed_gov_l_ema=speed_gov_l_ema,
        speed_gov_l_k=speed_gov_l_k,
        live_trace_path=live_trace_path,
        run_mode=run_mode,
        checkpoint_path=checkpoint_path,
        save_every_steps=save_every_steps,
        save_history=save_history,
        save_last_good=save_last_good,
        save_bad=save_bad,
        resume=resume,
        loss_keep=loss_keep,
        phase_a_steps=phase_a_steps,
        phase_b_steps=phase_b_steps,
        synth_len=synth_len,
        synth_shuffle=synth_shuffle,
        synth_mode=synth_mode,
        hand_min=hand_min,
        assoc_keys=assoc_keys,
        assoc_pairs=assoc_pairs,
        assoc_val_range=assoc_val_range,
        evo_pop=evo_pop,
        evo_gens=evo_gens,
        evo_steps=evo_steps,
        evo_mut_std=evo_mut_std,
        evo_pointer_only=evo_pointer_only,
        evo_checkpoint_every=evo_checkpoint_every,
        evo_resume=evo_resume,
        evo_checkpoint_individual=evo_checkpoint_individual,
        evo_progress=evo_progress,
        train_trace=train_trace,
        train_trace_path=train_trace_path,
    )










