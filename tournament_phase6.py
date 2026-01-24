import os
import time
import math
import json
import random
from itertools import count
import shutil
import urllib.request
import zipfile
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except Exception:
    torchaudio = None
    HAS_TORCHAUDIO = False

from prime_c19.settings import load_settings

CFG = load_settings()

ROOT = CFG.root
DATA_DIR = CFG.data_dir
LOG_PATH = CFG.log_path

SEED = CFG.seed
DEVICE = CFG.device
OFFLINE_ONLY = CFG.offline_only

if HAS_TORCHAUDIO and CFG.audio_backend:
    try:
        torchaudio.set_audio_backend(CFG.audio_backend)
    except Exception:
        pass

# experiment budget (env overrides for quick sanity)
MAX_SAMPLES = CFG.max_samples
EVAL_SAMPLES = CFG.eval_samples
EVAL_SPLIT = CFG.eval_split
EVAL_PTR_DETERMINISTIC = CFG.eval_ptr_deterministic
BATCH_SIZE = CFG.batch_size
LR = CFG.lr
WALL_CLOCK_SECONDS = CFG.wall_clock_seconds  # seconds
MAX_STEPS = CFG.max_steps
HEARTBEAT_STEPS = CFG.heartbeat_steps
HEARTBEAT_SECS = CFG.heartbeat_secs
LIVE_TRACE_EVERY = CFG.live_trace_every
SATIETY_THRESH = CFG.satiety_thresh
RING_LEN = CFG.ring_len
SLOT_DIM = CFG.slot_dim
EXPERT_HEADS = max(1, int(os.environ.get("TP6_EXPERT_HEADS", "1")))
PTR_DTYPE = CFG.ptr_dtype
PTR_PARAM_STRIDE = CFG.ptr_param_stride
# Gaussian window + movement penalty controls
GAUSS_K = CFG.gauss_k  # neighbors on each side; window size = 2*K+1
GAUSS_TAU = CFG.gauss_tau
PTR_KERNEL = CFG.ptr_kernel
PTR_KAPPA = CFG.ptr_kappa
PTR_EDGE_EPS = CFG.ptr_edge_eps
LAMBDA_MOVE = CFG.lambda_move
PTR_INERTIA = CFG.ptr_inertia  # 0=no inertia, 0.9=strong stay-bias
PTR_DEADZONE = CFG.ptr_deadzone  # distance below which pointer resists moving
PTR_DEADZONE_TAU = CFG.ptr_deadzone_tau
PTR_WARMUP_STEPS = CFG.ptr_warmup_steps
PTR_WALK_PROB = CFG.ptr_walk_prob  # 0=stay when not jumping, 1=always walk
PTR_JUMP_DISABLED = CFG.ptr_jump_disabled  # when true, disable jump mix (walk/stay only)
PTR_JUMP_CAP = CFG.ptr_jump_cap  # optional cap on jump probability (<=1.0)
PTR_NO_ROUND = CFG.ptr_no_round
PTR_PHANTOM = CFG.ptr_phantom
PTR_PHANTOM_OFF = CFG.ptr_phantom_off
PTR_PHANTOM_READ = CFG.ptr_phantom_read
SOFT_READOUT = CFG.soft_readout
SOFT_READOUT_K = CFG.soft_readout_k
SOFT_READOUT_TAU = CFG.soft_readout_tau
PTR_VEL = bool(int(os.environ.get("TP6_PTR_VEL", str(int(CFG.ptr_vel)))))
PTR_VEL_DECAY = CFG.ptr_vel_decay
PTR_VEL_CAP = CFG.ptr_vel_cap
PTR_VEL_SCALE = CFG.ptr_vel_scale
PTR_LOCK = CFG.ptr_lock
PTR_LOCK_VALUE = CFG.ptr_lock_value
PTR_UPDATE_EVERY = CFG.ptr_update_every
PTR_UPDATE_AUTO = CFG.ptr_update_auto
PTR_UPDATE_MIN = CFG.ptr_update_min
PTR_UPDATE_MAX = CFG.ptr_update_max
PTR_UPDATE_EVERY_STEP = CFG.ptr_update_every_step
PTR_UPDATE_TARGET_FLIP = CFG.ptr_update_target_flip
PTR_UPDATE_EMA = CFG.ptr_update_ema
PTR_UPDATE_GOV = CFG.ptr_update_governor
PTR_UPDATE_GOV_WARMUP = CFG.ptr_update_gov_warmup
PTR_UPDATE_GOV_GRAD_HIGH = CFG.ptr_update_gov_grad_high
PTR_UPDATE_GOV_GRAD_LOW = CFG.ptr_update_gov_grad_low
PTR_UPDATE_GOV_LOSS_FLAT = CFG.ptr_update_gov_loss_flat
PTR_UPDATE_GOV_LOSS_SPIKE = CFG.ptr_update_gov_loss_spike
PTR_UPDATE_GOV_STEP_UP = CFG.ptr_update_gov_step_up
PTR_UPDATE_GOV_STEP_DOWN = CFG.ptr_update_gov_step_down
PTR_GATE_MODE = CFG.ptr_gate_mode
PTR_GATE_STEPS = CFG.ptr_gate_steps
PTR_SOFT_GATE = CFG.ptr_soft_gate
THERMO_ENABLED = CFG.thermo_enabled
THERMO_EVERY = CFG.thermo_every
THERMO_TARGET_FLIP = CFG.thermo_target_flip
THERMO_EMA = CFG.thermo_ema
THERMO_INERTIA_STEP = CFG.thermo_inertia_step
THERMO_DEADZONE_STEP = CFG.thermo_deadzone_step
THERMO_WALK_STEP = CFG.thermo_walk_step
THERMO_INERTIA_MIN = CFG.thermo_inertia_min
THERMO_INERTIA_MAX = CFG.thermo_inertia_max
THERMO_DEADZONE_MIN = CFG.thermo_deadzone_min
THERMO_DEADZONE_MAX = CFG.thermo_deadzone_max
THERMO_WALK_MIN = CFG.thermo_walk_min
THERMO_WALK_MAX = CFG.thermo_walk_max

PANIC_ENABLED = CFG.panic_enabled
PANIC_THRESHOLD = CFG.panic_threshold
PANIC_BETA = CFG.panic_beta
PANIC_RECOVERY = CFG.panic_recovery
PANIC_INERTIA_LOW = CFG.panic_inertia_low
PANIC_INERTIA_HIGH = CFG.panic_inertia_high
PANIC_WALK_MAX = CFG.panic_walk_max
MOBIUS_ENABLED = CFG.mobius_enabled
MOBIUS_EMB_SCALE = CFG.mobius_emb_scale
ACT_NAME = CFG.act_name
C13_P = CFG.c13_p
DEBUG_NAN = CFG.debug_nan
# Force per-step debug logging for live runs regardless of env flags.
DEBUG_STATS = True
DEBUG_EVERY = int(os.environ.get("TP6_DEBUG_EVERY", "1"))
PRECISION = CFG.precision
DTYPE = CFG.dtype
USE_AMP = CFG.use_amp
DISABLE_SYNC = os.environ.get("TP6_DISABLE_SYNC", "0") == "1"
# Auto-pick bf16 on supported CUDA devices when precision is not explicitly set.
if os.environ.get("TP6_PRECISION") is None and DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    PRECISION = "bf16"
    DTYPE = torch.bfloat16
    USE_AMP = True
torch.set_default_dtype(DTYPE)

if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
    def amp_autocast():
        return torch.amp.autocast(device_type="cuda", enabled=USE_AMP)

    def amp_grad_scaler():
        enabled = USE_AMP and DTYPE != torch.bfloat16
        return torch.amp.GradScaler("cuda", enabled=enabled)
else:
    def amp_autocast():
        return torch.cuda.amp.autocast(enabled=USE_AMP)

    def amp_grad_scaler():
        enabled = USE_AMP and DTYPE != torch.bfloat16
        return torch.cuda.amp.GradScaler(enabled=enabled)
MI_SHUFFLE = CFG.mi_shuffle
STATE_LOOP_METRICS = CFG.state_loop_metrics
STATE_LOOP_EVERY = CFG.state_loop_every
STATE_LOOP_SAMPLES = CFG.state_loop_samples
STATE_LOOP_DIM = CFG.state_loop_dim
GRAD_CLIP = float(os.environ.get("TP6_GRAD_CLIP", str(CFG.grad_clip)))
STATE_CLIP = CFG.state_clip
STATE_DECAY = float(os.environ.get("TP6_STATE_DECAY", str(CFG.state_decay)))
# Hardcode tiny update scale floor to kill the 0.1 ghost. (Now env/CFG driven.)
UPDATE_SCALE = float(os.environ.get("TP6_UPDATE_SCALE", str(getattr(CFG, "update_scale", 0.0005))))
AGC_ENABLED = CFG.agc_enabled
AGC_GRAD_LOW = CFG.agc_grad_low
AGC_GRAD_HIGH = CFG.agc_grad_high
AGC_SCALE_UP = float(os.environ.get("TP6_SCALE_UP", str(getattr(CFG, "agc_scale_up", 1.05))))
AGC_SCALE_DOWN = float(os.environ.get("TP6_SCALE_DOWN", str(getattr(CFG, "agc_scale_down", 0.5))))
AGC_SCALE_MIN = float(os.environ.get("TP6_SCALE_MIN", str(getattr(CFG, "agc_scale_min", 0.0005))))
AGC_SCALE_MAX = float(os.environ.get("TP6_SCALE_MAX", str(getattr(CFG, "agc_scale_max", 1.0))))
AGC_SCALE_MAX_MIN = float(os.environ.get("TP6_SCALE_MAX_MIN", str(getattr(CFG, "agc_scale_max_min", 1e-4))))
AGC_SCALE_MAX_DECAY = float(os.environ.get("TP6_SCALE_MAX_DECAY", str(getattr(CFG, "agc_scale_max_decay", 0.3))))
AGC_PLATEAU_WINDOW = int(os.environ.get("TP6_AGC_PLATEAU_WINDOW", str(getattr(CFG, "agc_plateau_window", 0))))
AGC_PLATEAU_STD = float(os.environ.get("TP6_AGC_PLATEAU_STD", str(getattr(CFG, "agc_plateau_std", 0.0))))
AGC_PLATEAU_MIN_STEPS = int(os.environ.get("TP6_AGC_PLATEAU_MIN_STEPS", str(getattr(CFG, "agc_plateau_min_steps", 0))))
SPEED_GOV_ENABLED = CFG.speed_gov_enabled
SPEED_GOV_RHO = CFG.speed_gov_rho
SPEED_GOV_L_MIN = CFG.speed_gov_l_min
SPEED_GOV_L_MAX = CFG.speed_gov_l_max
SPEED_GOV_L_EMA = CFG.speed_gov_l_ema
SPEED_GOV_L_K = CFG.speed_gov_l_k
INERTIA_AUTO = CFG.ptr_inertia_auto
INERTIA_MIN = CFG.ptr_inertia_min
INERTIA_MAX = CFG.ptr_inertia_max
INERTIA_VEL_FULL = CFG.ptr_inertia_vel_full
INERTIA_EMA = CFG.ptr_inertia_ema
# Optional dwell-driven inertia (kinetic tempering)
DWELL_INERTIA_ENABLED = bool(int(os.environ.get("TP6_DWELL_INERTIA", "1")))
DWELL_INERTIA_THRESH = float(os.environ.get("TP6_DWELL_INERTIA_THRESH", "50.0"))
WALK_PULSE_ENABLED = bool(int(os.environ.get("TP6_WALK_PULSE", "0")))
WALK_PULSE_EVERY = int(os.environ.get("TP6_WALK_PULSE_EVERY", "50"))
WALK_PULSE_VALUE = float(os.environ.get("TP6_WALK_PULSE_VALUE", "0.5"))
SCALE_WARMUP_STEPS = int(os.environ.get("TP6_SCALE_WARMUP_STEPS", "0"))
SCALE_WARMUP_INIT = float(os.environ.get("TP6_SCALE_WARMUP_INIT", str(UPDATE_SCALE)))
SHARD_ENABLED = bool(int(os.environ.get("TP6_SHARD_BATCH", "1")))
SHARD_SIZE = int(os.environ.get("TP6_SHARD_SIZE", "19"))
SHARD_ADAPT = bool(int(os.environ.get("TP6_SHARD_ADAPT", "1")))
SHARD_ADAPT_EVERY = int(os.environ.get("TP6_SHARD_ADAPT_EVERY", "50"))
SHARD_ADAPT_GRAD = float(os.environ.get("TP6_SHARD_ADAPT_GRAD", "10.0"))
SHARD_ADAPT_DWELL = float(os.environ.get("TP6_SHARD_ADAPT_DWELL", "40.0"))
SHARD_MIN_PER_SHARD = int(os.environ.get("TP6_SHARD_MIN_PER_SHARD", "1"))
VASC_MIN_GROUP_RATIO = float(os.environ.get("TP6_VASC_MIN_GROUP_RATIO", "0.02"))
TRACTION_ENABLED = bool(int(os.environ.get("TP6_TRACTION_LOG", "1")))
PTR_UPDATE_GOV_VEL_HIGH = getattr(CFG, "ptr_update_gov_vel_high", 0.5)
LIVE_TRACE_PATH = CFG.live_trace_path
RUN_MODE = CFG.run_mode
# Cold start controls
COLD_START_STEPS = int(os.environ.get("TP6_COLD_START_STEPS", "0"))
COLD_PTR_UPDATE_MIN = int(os.environ.get("TP6_COLD_PTR_UPDATE_MIN", PTR_UPDATE_EVERY))
COLD_UPDATE_SCALE = float(os.environ.get("TP6_COLD_UPDATE_SCALE", UPDATE_SCALE))
# Checkpoint / resume controls
CHECKPOINT_PATH = CFG.checkpoint_path
SAVE_EVERY_STEPS = int(os.environ.get("TP6_SAVE_EVERY_STEPS", str(CFG.save_every_steps)))
SAVE_HISTORY = CFG.save_history
SAVE_LAST_GOOD = CFG.save_last_good
SAVE_BAD = CFG.save_bad
RESUME = CFG.resume
LOSS_KEEP = CFG.loss_keep
# Lockout test controls (synthetic, deterministic)
PHASE_A_STEPS = CFG.phase_a_steps
PHASE_B_STEPS = CFG.phase_b_steps
SYNTH_LEN = CFG.synth_len
SYNTH_SHUFFLE = CFG.synth_shuffle
SYNTH_MODE = CFG.synth_mode
HAND_MIN = CFG.hand_min
ASSOC_KEYS = CFG.assoc_keys
ASSOC_PAIRS = CFG.assoc_pairs
ASSOC_VAL_RANGE = getattr(CFG, "assoc_val_range", 256)
SYNTH_META = {}
# Evolution defaults (small to keep runs short/safe)
EVO_POP = CFG.evo_pop
EVO_GENS = CFG.evo_gens
EVO_STEPS = CFG.evo_steps
EVO_MUT_STD = CFG.evo_mut_std
EVO_POINTER_ONLY = CFG.evo_pointer_only
EVO_CKPT_EVERY = CFG.evo_checkpoint_every
EVO_RESUME = CFG.evo_resume
EVO_CKPT_INDIV = CFG.evo_checkpoint_individual
EVO_PROGRESS = CFG.evo_progress
TRAIN_TRACE = CFG.train_trace
TRAIN_TRACE_PATH = CFG.train_trace_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def rotate_artifacts() -> None:
    """Move current logs/traces/summaries -> last, and last -> archive/<timestamp>."""
    ts = time.strftime("%Y%m%d_%H%M%S")

    def _rotate_dir(base_dir: str) -> None:
        current_dir = os.path.join(base_dir, "current")
        last_dir = os.path.join(base_dir, "last")
        archive_dir = os.path.join(base_dir, "archive")

        os.makedirs(current_dir, exist_ok=True)
        os.makedirs(last_dir, exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)

        if os.path.isdir(last_dir) and os.listdir(last_dir):
            archive_run = os.path.join(archive_dir, ts)
            os.makedirs(archive_run, exist_ok=True)
            for name in os.listdir(last_dir):
                shutil.move(os.path.join(last_dir, name), os.path.join(archive_run, name))

        if os.path.isdir(current_dir) and os.listdir(current_dir):
            for name in os.listdir(current_dir):
                shutil.move(os.path.join(current_dir, name), os.path.join(last_dir, name))

    _rotate_dir(os.path.join(ROOT, "logs"))
    _rotate_dir(os.path.join(ROOT, "traces"))
    _rotate_dir(os.path.join(ROOT, "summaries"))


def sync_current_to_last() -> None:
    """Copy current logs/traces/summaries into logs/last for quick inspection."""
    dest_dir = os.path.join(ROOT, "logs", "last")
    os.makedirs(dest_dir, exist_ok=True)
    for rel in ("logs/current", "traces/current", "summaries/current"):
        src_dir = os.path.join(ROOT, rel)
        if not os.path.isdir(src_dir):
            continue
        for name in os.listdir(src_dir):
            src = os.path.join(src_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest_dir, name))


def nan_guard(name: str, tensor: torch.Tensor, step: int) -> None:
    if not DEBUG_NAN:
        return
    if not tensor.is_floating_point():
        return
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        log(f"[nan_guard] step={step:04d} tensor={name} has NaN/Inf")
        raise RuntimeError(f"NaN/Inf in {name} at step {step}")


def apply_thermostat(
    model,
    flip_rate: float,
    ema: float | None,
    *,
    focus: float | None = None,
    tension: float | None = None,
    raw_delta: float | None = None,
):
    """Adaptive pointer control: reduce flapping without freezing forever."""
    if ema is None:
        ema = flip_rate
    else:
        ema = THERMO_EMA * ema + (1.0 - THERMO_EMA) * flip_rate
    # Respect manual inertia override: do not mutate ptr_inertia/deadzone/walk.
    if os.environ.get("TP6_PTR_INERTIA_OVERRIDE") is not None:
        return ema

    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    if focus is not None and tension is not None:
        f = _clamp01(float(focus))
        t = _clamp01(float(tension))
        stuck = 0.0
        if raw_delta is not None:
            try:
                rd = float(raw_delta)
            except (TypeError, ValueError):
                rd = None
            if rd is not None and math.isfinite(rd):
                stuck = 1.0 / (1.0 + max(0.0, rd))
        drive = max(t, 1.0 - f, stuck)
        target_inertia = THERMO_INERTIA_MIN + (THERMO_INERTIA_MAX - THERMO_INERTIA_MIN) * (f * (1.0 - t))
        target_deadzone = THERMO_DEADZONE_MIN + (THERMO_DEADZONE_MAX - THERMO_DEADZONE_MIN) * t
        target_walk = THERMO_WALK_MIN + (THERMO_WALK_MAX - THERMO_WALK_MIN) * drive
        blend = max(1e-3, 1.0 - THERMO_EMA)
        model.ptr_inertia = model.ptr_inertia + (target_inertia - model.ptr_inertia) * blend
        model.ptr_deadzone = model.ptr_deadzone + (target_deadzone - model.ptr_deadzone) * blend
        model.ptr_walk_prob = model.ptr_walk_prob + (target_walk - model.ptr_walk_prob) * blend
        return ema

    if ema > THERMO_TARGET_FLIP:
        model.ptr_inertia = min(THERMO_INERTIA_MAX, model.ptr_inertia + THERMO_INERTIA_STEP)
        model.ptr_deadzone = min(THERMO_DEADZONE_MAX, model.ptr_deadzone + THERMO_DEADZONE_STEP)
        model.ptr_walk_prob = max(THERMO_WALK_MIN, model.ptr_walk_prob - THERMO_WALK_STEP)
    elif ema < THERMO_TARGET_FLIP * 0.5:
        model.ptr_inertia = max(THERMO_INERTIA_MIN, model.ptr_inertia - THERMO_INERTIA_STEP)
        model.ptr_deadzone = max(THERMO_DEADZONE_MIN, model.ptr_deadzone - THERMO_DEADZONE_STEP)
        model.ptr_walk_prob = min(THERMO_WALK_MAX, model.ptr_walk_prob + THERMO_WALK_STEP)

    return ema


class PanicReflex:
    """Loss-based unlock: reduce friction when loss spikes."""

    def __init__(
        self,
        ema_beta: float = 0.9,
        panic_threshold: float = 1.5,
        recovery_rate: float = 0.01,
        inertia_low: float = 0.1,
        inertia_high: float = 0.95,
        walk_prob_max: float = 0.2,
    ) -> None:
        self.loss_ema: float | None = None
        self.beta = ema_beta
        self.threshold = panic_threshold
        self.recovery = recovery_rate
        self.inertia_low = inertia_low
        self.inertia_high = inertia_high
        self.walk_prob_max = walk_prob_max
        self.panic_state = 0.0

    def update(self, loss_value: float) -> dict:
        if self.loss_ema is None:
            self.loss_ema = loss_value
            return {"status": "INIT", "inertia": self.inertia_high, "walk_prob": 0.0}
        ratio = loss_value / (self.loss_ema + 1e-6)
        if ratio > self.threshold:
            self.panic_state = 1.0
        else:
            self.panic_state = max(0.0, self.panic_state - self.recovery)
        self.loss_ema = (self.beta * self.loss_ema) + ((1.0 - self.beta) * loss_value)
        if self.panic_state > 0.1:
            inertia = self.inertia_low + (self.inertia_high - self.inertia_low) * (1.0 - self.panic_state)
            walk_prob = self.walk_prob_max * self.panic_state
            return {"status": "PANIC", "inertia": inertia, "walk_prob": walk_prob}
        return {"status": "LOCKED", "inertia": self.inertia_high, "walk_prob": 0.0}


class CadenceGovernor:
    """Adaptive cadence controller combining flip-rate and gradient shock signals."""

    def __init__(
        self,
        start_tau: float,
        warmup_steps: int,
        min_tau: int,
        max_tau: int,
        ema: float,
        target_flip: float,
        grad_high: float,
        grad_low: float,
        loss_flat: float,
        loss_spike: float,
        step_up: float,
        step_down: float,
        vel_high: float = PTR_UPDATE_GOV_VEL_HIGH,
    ):
        self.tau = float(start_tau)
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_tau = max(1, int(min_tau))
        self.max_tau = max(self.min_tau, int(max_tau))
        self.ema = float(ema)
        self.target_flip = float(target_flip)
        self.grad_high = float(grad_high)
        self.grad_low = float(grad_low)
        self.loss_flat = float(loss_flat)
        self.loss_spike = float(loss_spike)
        self.step_up = float(step_up)
        self.step_down = float(step_down)
        self.vel_high = float(vel_high)
        self.step_count = 0
        self.grad_ema = None
        self.flip_ema = None
        self.vel_ema = None
        self.prev_loss = None

    def update(self, loss_value: float, grad_norm: float, flip_rate: float, ptr_velocity=None) -> int:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            return int(round(self.tau))

        if ptr_velocity is not None:
            if self.vel_ema is None:
                self.vel_ema = float(ptr_velocity)
            else:
                self.vel_ema = self.ema * self.vel_ema + (1.0 - self.ema) * float(ptr_velocity)
            # High pointer velocity requires higher sampling (lower cadence).
            if self.vel_ema > self.vel_high:
                self.tau = float(self.min_tau)
                self.prev_loss = loss_value
                return int(round(self.tau))

        if grad_norm > self.grad_high:
            self.tau = float(self.max_tau)
            return int(round(self.tau))

        if self.grad_ema is None:
            self.grad_ema = grad_norm
        else:
            self.grad_ema = self.ema * self.grad_ema + (1.0 - self.ema) * grad_norm

        if self.flip_ema is None:
            self.flip_ema = flip_rate
        else:
            self.flip_ema = self.ema * self.flip_ema + (1.0 - self.ema) * flip_rate

        if self.prev_loss is None:
            loss_delta = 0.0
        else:
            loss_delta = self.prev_loss - loss_value
        self.prev_loss = loss_value

        # Slow down when turbulence is high or loss spikes.
        if self.flip_ema > self.target_flip or self.grad_ema > self.grad_high or loss_delta < -self.loss_spike:
            self.tau = min(self.max_tau, self.tau + self.step_up)
        # Speed up only when laminar and loss is flat.
        elif self.grad_ema < self.grad_low and self.flip_ema < self.target_flip * 0.5 and abs(loss_delta) < self.loss_flat:
            self.tau = max(self.min_tau, self.tau - self.step_down)

        self.tau = max(self.min_tau, min(self.max_tau, self.tau))
        return int(round(self.tau))


def compute_slope(losses: List[float]) -> float:
    if len(losses) < 2:
        return float("nan")
    x = np.arange(len(losses), dtype=np.float64)
    y = np.array(losses, dtype=np.float64)
    a, _ = np.polyfit(x, y, 1)
    return float(a)


def _checkpoint_payload(model, optimizer, scaler, step: int, losses: List[float]) -> dict:
    return {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scaler": scaler.state_dict() if USE_AMP else None,
        "step": step,
        "losses": losses,
        "update_scale": getattr(model, "update_scale", UPDATE_SCALE),
        "ptr_inertia": getattr(model, "ptr_inertia", PTR_INERTIA),
        "ptr_inertia_ema": getattr(model, "ptr_inertia_ema", getattr(model, "ptr_inertia", PTR_INERTIA)),
        "agc_scale_max": getattr(model, "agc_scale_max", AGC_SCALE_MAX),
        "ground_speed_ema": getattr(model, "ground_speed_ema", None),
        "ground_speed_limit": getattr(model, "ground_speed_limit", None),
    }


def _checkpoint_paths(base_path: str, step: int) -> tuple[str, str, str]:
    base_dir = os.path.dirname(base_path) or ROOT
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    history_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(history_dir, exist_ok=True)
    step_path = os.path.join(history_dir, f"{base_name}_step_{step:07d}.pt")
    bad_path = os.path.join(history_dir, f"{base_name}_bad_step_{step:07d}.pt")
    last_good_path = os.path.join(base_dir, f"{base_name}_last_good.pt")
    return step_path, bad_path, last_good_path


def _checkpoint_is_finite(loss_value, grad_norm_value, raw_delta_value) -> bool:
    for value in (loss_value, grad_norm_value, raw_delta_value):
        if value is None:
            continue
        if not math.isfinite(float(value)):
            return False
    return True


def apply_update_agc(model, grad_norm, raw_delta=None, step: int | None = None):
    """
    Surgical reset: single-floor AGC. No hidden clamps, no speed governor.
    Rules:
      * start from AGC_SCALE_MIN
      * adjust up/down by grad_norm
      * clamp to [AGC_SCALE_MIN, agc_scale_max/agc_scale_cap]
    """
    base_cap = float(getattr(model, "agc_scale_max", AGC_SCALE_MAX))
    cap = float(getattr(model, "agc_scale_cap", base_cap))
    if not math.isfinite(cap) or cap <= 0:
        cap = base_cap
    cap = max(AGC_SCALE_MIN, min(base_cap, cap))

    # Warmup floor: ramp linearly to AGC_SCALE_MIN over SCALE_WARMUP_STEPS from SCALE_WARMUP_INIT.
    if step is not None:
        warmup_horizon = max(1, SCALE_WARMUP_STEPS)
        warmup = max(0.0, min(1.0, step / float(warmup_horizon)))
        floor = SCALE_WARMUP_INIT + (AGC_SCALE_MIN - SCALE_WARMUP_INIT) * warmup
        floor = max(0.0, min(AGC_SCALE_MIN, floor))
    else:
        floor = AGC_SCALE_MIN

    scale = float(getattr(model, "update_scale", floor))
    if not math.isfinite(scale) or scale <= 0:
        scale = floor
    if step is not None and step == 0:
        scale = floor

    if AGC_ENABLED and grad_norm is not None and math.isfinite(grad_norm):
        if grad_norm < AGC_GRAD_LOW:
            scale *= AGC_SCALE_UP
        elif grad_norm > AGC_GRAD_HIGH:
            scale *= AGC_SCALE_DOWN

    scale = max(floor, min(cap, scale))
    model.agc_scale_cap = cap
    model.update_scale = scale
    model.debug_scale_out = scale
    if step is not None and step == 0:
        dbg = {
            "scale_in": scale,
            "scale_out": scale,
            "agc_scale_min": AGC_SCALE_MIN,
            "warmup_floor": floor,
            "cap": cap,
            "base_cap": base_cap,
        }
        log(f"[debug_scale_step0] {dbg}")
    return scale


def apply_inertia_auto(model, ptr_velocity, panic_active=False):
    if not INERTIA_AUTO or panic_active:
        return
    # Dwell-driven kinetic tempering: glue when dwell is high, agile when low.
    if DWELL_INERTIA_ENABLED:
        dwell = getattr(model, "ptr_mean_dwell", None)
        max_dwell = getattr(model, "ptr_max_dwell", dwell)
        try:
            dwell = float(dwell) if dwell is not None else 0.0
            max_dwell = float(max_dwell) if max_dwell is not None else dwell
        except (TypeError, ValueError):
            dwell, max_dwell = 0.0, 0.0
        dwell_metric = max(dwell, max_dwell)
        if DWELL_INERTIA_THRESH > 0:
            weight = max(0.0, min(1.0, dwell_metric / DWELL_INERTIA_THRESH))
            target = INERTIA_MIN + weight * (INERTIA_MAX - INERTIA_MIN)
        else:
            target = INERTIA_MAX
    else:
        if ptr_velocity is None:
            return
        if INERTIA_VEL_FULL <= 0:
            return
        try:
            velocity = float(ptr_velocity)
        except (TypeError, ValueError):
            return
        velocity = max(0.0, velocity)
        ratio = min(1.0, velocity / INERTIA_VEL_FULL)
        target = INERTIA_MIN + ratio * (INERTIA_MAX - INERTIA_MIN)
    ema = float(getattr(model, "ptr_inertia_ema", model.ptr_inertia))
    ema = INERTIA_EMA * ema + (1.0 - INERTIA_EMA) * target
    ema = max(INERTIA_MIN, min(INERTIA_MAX, ema))
    model.ptr_inertia_ema = ema
    model.ptr_inertia = ema


def calculate_adaptive_vasc(batch_size, dwell, grad_norm, max_dwell_limit, ema_grad_norm, min_group_ratio=VASC_MIN_GROUP_RATIO):
    """Scale-free VASC: choose shard count from ratio-based cohesion signals."""
    eps = 1e-8
    batch_size = int(batch_size)
    if batch_size <= 0:
        return 1, 0, 0.0, 0.0, 0.0

    max_dwell_limit = max(eps, float(max_dwell_limit))
    ema_grad_norm = max(eps, float(ema_grad_norm))

    focus = max(0.0, min(1.0, float(dwell) / max_dwell_limit))
    tension = max(0.0, min(1.0, float(grad_norm) / (ema_grad_norm + eps)))
    cohesion = max(0.0, min(1.0, focus - tension))

    ceiling = float(batch_size)
    floor = max(1.0, ceiling * float(min_group_ratio))

    target_group_size = floor + (ceiling - floor) * cohesion
    raw_shards = ceiling / max(eps, target_group_size)

    valid_counts = [c for c in range(1, batch_size + 1) if batch_size % c == 0]
    shard_count = min(valid_counts, key=lambda c: abs(c - raw_shards)) if valid_counts else 1
    shard_count = max(1, min(shard_count, batch_size))
    group_size = batch_size // shard_count
    return shard_count, group_size, focus, tension, cohesion


class LocationExpertRouter(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, num_experts: int = 1):
        super().__init__()
        self.num_experts = max(1, int(num_experts))
        if self.num_experts == 1:
            self.single = nn.Linear(d_model, vocab_size)
            self.experts = None
        else:
            self.single = None
            self.experts = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(self.num_experts)])

    def reset_parameters(self):
        def init_layer(layer):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        if self.single is not None:
            init_layer(self.single)
        else:
            for expert in self.experts:
                init_layer(expert)

    def forward(self, x: torch.Tensor, pointer_addresses: torch.Tensor | None = None) -> torch.Tensor:
        if self.single is not None:
            return self.single(x)

        if pointer_addresses is None:
            expert_indices = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        else:
            expert_indices = pointer_addresses.to(torch.long, non_blocking=True) % self.num_experts

        out_dtype = self.experts[0].weight.dtype
        out = torch.zeros(x.shape[0], self.experts[0].out_features, device=x.device, dtype=out_dtype)
        for i, expert in enumerate(self.experts):
            mask = expert_indices == i
            if mask.any():
                out[mask] = expert(x[mask]).to(out_dtype)
        return out


class AbsoluteHallway(nn.Module):
    @staticmethod
    def wrap_delta(a, b, ring_range):
        # Shortest signed delta from a to b on a ring.
        return torch.remainder(b - a + ring_range / 2, ring_range) - ring_range / 2

    @staticmethod
    def circ_lerp(a, b, w, ring_range):
        # Move from a toward b by fraction w along shortest arc.
        return torch.remainder(a + w * AbsoluteHallway.wrap_delta(a, b, ring_range), ring_range)

    """
    Boundaryless ring with intrinsic pointer params per neuron.
    - Each neuron has theta_ptr (target coord) and theta_gate (bias).
    - Pointer update is a soft mix of jump target and walk/stay, then optional inertia/deadzone.
    - Readout: average of states at last K pointers (tensorized) or soft readout window.
    - Satiety exit: if max prob > SATIETY_THRESH, stop processing further timesteps for that sample.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        ring_len=RING_LEN,
        slot_dim=SLOT_DIM,
        ptr_stride=PTR_PARAM_STRIDE,
        gauss_k=GAUSS_K,
        gauss_tau=GAUSS_TAU,
        bypass_ring: bool = False,
        time_pointer: bool = False,
        aux_ring: bool = False,
        context_scale_init: float = 0.2,
    ):
        super().__init__()
        self.ring_len = ring_len
        self.slot_dim = slot_dim
        self.num_classes = num_classes
        self.ptr_stride = max(1, ptr_stride)
        self.gauss_k = gauss_k
        self.gauss_tau = gauss_tau
        self.ptr_kernel = PTR_KERNEL if PTR_KERNEL in {"gauss", "vonmises"} else "gauss"
        self.ptr_kappa = PTR_KAPPA
        self.ptr_edge_eps = PTR_EDGE_EPS
        self.act_name = ACT_NAME
        # Telemetry switch for optional x-ray logging.
        self.collect_xray = False
        # Optional: bypass ring scatter/read and use GRU output directly (diagnostic).
        self.bypass_ring = bypass_ring
        # Optional: force pointer to follow time index deterministically (diagnostic).
        self.time_pointer = time_pointer
        # Optional: keep a global GRU hidden and treat ring as auxiliary buffer (do not replace hidden with ring reads).
        self.aux_ring = aux_ring
        self.c13_p = max(C13_P, 1e-6)
        self.c19_rho = 4.0
        self.c14_rho = self.c19_rho
        self.mobius = MOBIUS_ENABLED
        self.mobius_scale = 2 if self.mobius else 1
        self.ring_range = ring_len * self.mobius_scale
        # Pointer control (modifiable at runtime)
        self.ptr_inertia = PTR_INERTIA
        self.ptr_inertia_ema = self.ptr_inertia
        self.ptr_deadzone = PTR_DEADZONE
        self.ptr_deadzone_tau = PTR_DEADZONE_TAU
        self.ptr_walk_prob = PTR_WALK_PROB
        self.update_scale = float(UPDATE_SCALE)
        self.agc_scale_max = float(AGC_SCALE_MAX)
        self.ground_speed_ema = None
        self.ground_speed_limit = None
        self.ground_speed = None
        self.ptr_vel_enabled = PTR_VEL
        self.ptr_vel_decay = PTR_VEL_DECAY
        self.ptr_vel_cap = PTR_VEL_CAP
        self.ptr_vel_scale = PTR_VEL_SCALE
        self.ptr_lock = PTR_LOCK
        self.ptr_lock_value = PTR_LOCK_VALUE
        self.ptr_update_every = max(1, PTR_UPDATE_EVERY)
        self.ptr_update_auto = PTR_UPDATE_AUTO
        self.ptr_update_min = max(1, PTR_UPDATE_MIN)
        self.ptr_update_max = max(self.ptr_update_min, PTR_UPDATE_MAX)
        self.ptr_update_every_step = max(1, PTR_UPDATE_EVERY_STEP)
        self.ptr_update_target_flip = PTR_UPDATE_TARGET_FLIP
        self.ptr_update_ema = PTR_UPDATE_EMA
        self.ptr_update_ema_state = None
        self.ptr_gate_mode = PTR_GATE_MODE
        self.ptr_gate_steps = set()
        if self.ptr_gate_mode == "steps" and PTR_GATE_STEPS:
            for token in PTR_GATE_STEPS.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    self.ptr_gate_steps.add(int(token))
                except ValueError:
                    continue
        self.ptr_soft_gate = PTR_SOFT_GATE
        if self.ptr_soft_gate:
            self.gate_head = nn.Linear(slot_dim, 1)
        else:
            self.gate_head = None
        self.ptr_warmup_steps = PTR_WARMUP_STEPS
        self.ptr_no_round = PTR_NO_ROUND
        self.ptr_phantom = PTR_PHANTOM
        self.ptr_phantom_off = PTR_PHANTOM_OFF
        self.ptr_phantom_read = PTR_PHANTOM_READ
        self.soft_readout = SOFT_READOUT
        self.soft_readout_k = max(0, SOFT_READOUT_K)
        self.soft_readout_tau = max(SOFT_READOUT_TAU, 1e-6)
        self.state_loop_metrics = STATE_LOOP_METRICS
        self.state_loop_every = max(1, STATE_LOOP_EVERY)
        self.state_loop_samples = max(0, STATE_LOOP_SAMPLES)
        self.state_loop_dim = max(2, STATE_LOOP_DIM)
        if self.state_loop_metrics:
            g = torch.Generator()
            g.manual_seed(1337)
            proj = torch.randn(self.slot_dim, self.state_loop_dim, generator=g)
            proj = torch.nn.functional.normalize(proj, dim=0)
            self.register_buffer("state_loop_proj", proj)
        else:
            self.register_buffer("state_loop_proj", torch.empty(0))
        self.input_proj = nn.Linear(input_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.jump_score = nn.Linear(slot_dim, 1)
        # Adaptive control heads for pointer behavior.
        self.inertia_head = nn.Linear(slot_dim, 1)
        self.deadzone_head = nn.Linear(slot_dim, 1)
        self.walk_head = nn.Linear(slot_dim, 1)
        # intrinsic params (downsampled for memory, then upsampled by gather)
        reduced = (self.ring_range + self.ptr_stride - 1) // self.ptr_stride
        self.theta_ptr_reduced = nn.Parameter(torch.zeros(reduced))  # mapped via sigmoid to [0,1]
        self.theta_gate_reduced = nn.Parameter(torch.zeros(reduced))
        if self.mobius:
            self.phase_embed = nn.Parameter(torch.zeros(2, slot_dim))
        else:
            self.register_parameter("phase_embed", None)
        self.head = LocationExpertRouter(slot_dim, num_classes, num_experts=EXPERT_HEADS)
        # Learnable ring context scale (sigmoid-bounded) to avoid hard-coded mixing; init is configurable.
        c0 = max(1e-3, min(0.999, context_scale_init))
        logit = math.log(c0 / (1.0 - c0))
        self.context_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))
        self.pointer_hist_bins = 128
        self.register_buffer("bin_edges", torch.linspace(0, self.ring_range, self.pointer_hist_bins + 1))
        self.pointer_hist = torch.zeros(self.pointer_hist_bins, dtype=torch.long)
        self.satiety_exits = 0
        # Default to single-pointer read to avoid averaging away sparse signals.
        self.blur_window = 1
        self.debug_stats = None
        self.reset_parameters()

    def reset_parameters(self):
        # Spread ptr targets across ring; bias gates near zero
        nn.init.uniform_(self.theta_ptr_reduced, -4.0, 4.0)
        nn.init.uniform_(self.theta_gate_reduced, -0.5, 0.5)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        # Adjust GRU power smoothly by ring length to avoid runaway accumulation on long tasks.
        # std interpolates 0.1 -> 0.05 as ring_len goes 0 -> 200 (clamped), bias 0.5 -> 0.2.
        f = max(0.0, min(1.0, self.ring_len / 200.0))
        gru_std = 0.1 - 0.05 * f
        gru_bias = 0.5 - 0.3 * f
        nn.init.normal_(self.gru.weight_ih, mean=0.0, std=gru_std)
        nn.init.normal_(self.gru.weight_hh, mean=0.0, std=gru_std)
        nn.init.constant_(self.gru.bias_ih, gru_bias)
        nn.init.constant_(self.gru.bias_hh, gru_bias)
        nn.init.xavier_uniform_(self.jump_score.weight)
        nn.init.zeros_(self.jump_score.bias)
        if hasattr(self.head, "reset_parameters"):
            self.head.reset_parameters()
        if self.gate_head is not None:
            nn.init.xavier_uniform_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)
        if self.mobius:
            nn.init.normal_(self.phase_embed, mean=0.0, std=MOBIUS_EMB_SCALE)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_name in ("identity", "none"):
            return x
        if self.act_name == "tanh":
            return torch.tanh(x)
        if self.act_name == "softsign":
            return x / (1.0 + x.abs())
        if self.act_name == "arctan":
            return torch.atan(x)
        if self.act_name in ("silu", "swish"):
            return torch.nn.functional.silu(x)
        if self.act_name == "relu":
            return torch.relu(x)
        if self.act_name in ("c13", "c-13"):
            t = 1.0 + (x / self.c13_p)
            t = torch.clamp(t, 0.0, 1.0)
            return x * t * t
        if self.act_name in ("c13-static", "c-13-static"):
            t = 1.0 + (x / 2.0)
            t = torch.clamp(t, 0.0, 1.0)
            return x * t * t
        if self.act_name in ("c19", "c-19", "candidate-19", "c14", "c-14"):
            u = x
            l = 6.0 * math.pi
            inv_pi = 1.0 / math.pi
            scaled = u * inv_pi
            n = torch.floor(scaled)
            t = scaled - n
            h = t * (1.0 - t)
            is_even = torch.remainder(n, 2.0) < 1.0
            sgn = torch.where(is_even, torch.ones_like(u), -torch.ones_like(u))
            core = math.pi * (sgn * h + (self.c19_rho * h * h))
            return torch.where(u >= l, u - l, torch.where(u <= -l, u + l, core))
        return x

    def _gather_params(self, ptr):
        # ptr: [B] float or long, map to reduced indices with linear interpolation.
        ptr_f = ptr.to(torch.float32)
        idx_float = ptr_f / self.ptr_stride
        idx_base = torch.floor(idx_float)
        frac = (idx_float - idx_base).clamp(0.0, 1.0).detach().unsqueeze(1)
        n = self.theta_ptr_reduced.numel()
        idx0 = torch.remainder(idx_base, n).long()
        idx1 = torch.remainder(idx0 + 1, n).long()
        ring_range = self.ring_range

        theta_ptr0 = torch.sigmoid(self.theta_ptr_reduced[idx0]) * (ring_range - 1)
        theta_ptr1 = torch.sigmoid(self.theta_ptr_reduced[idx1]) * (ring_range - 1)
        theta_ptr = theta_ptr0 + (theta_ptr1 - theta_ptr0) * frac.squeeze(1)

        theta_gate0 = self.theta_gate_reduced[idx0]
        theta_gate1 = self.theta_gate_reduced[idx1]
        theta_gate = theta_gate0 + (theta_gate1 - theta_gate0) * frac.squeeze(1)
        return theta_ptr, theta_gate

    def _compute_kernel_weights(self, ptr_float, offsets, ring_range, tau_override=None):
        # Integer centers (non-differentiable) + fractional pointer (differentiable).
        base = torch.floor(ptr_float).long().clamp(0, ring_range - 1)
        offsets_long = offsets.to(torch.long)
        centers = torch.remainder(base.unsqueeze(1) + offsets_long.unsqueeze(0), ring_range)
        centers = torch.nan_to_num(centers, nan=0, posinf=ring_range - 1, neginf=0)
        centers_f = centers.to(ptr_float.dtype)
        if self.ptr_kernel == "vonmises":
            angle_scale = (2.0 * math.pi) / max(float(ring_range), 1e-6)
            delta = (centers_f - ptr_float.unsqueeze(1)) * angle_scale
            kappa = max(self.ptr_kappa, 1e-6)
            logits = kappa * torch.cos(delta)
        else:
            delta = self.wrap_delta(ptr_float.unsqueeze(1), centers_f, ring_range)
            d2 = delta ** 2
            tau = max(self.gauss_tau if tau_override is None else tau_override, 1e-4)
            logits = -d2 / tau
        weights = torch.softmax(logits, dim=1)
        return centers, weights, centers_f

    def _compute_gru_gates(self, inp, cur):
        """Recompute GRU gates for telemetry (sigmoid/tanh activations)."""
        wi, wh = self.gru.weight_ih, self.gru.weight_hh
        bi, bh = self.gru.bias_ih, self.gru.bias_hh
        gates = F.linear(inp, wi, bi) + F.linear(cur, wh, bh)
        r, z, n = gates.chunk(3, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n)
        return r, z, n

    def forward(self, x, return_xray: bool = False):
        B, T, _ = x.shape
        device = x.device
        ring_range = self.ring_range
        ptr_dtype = PTR_DTYPE
        # Diagnostic bypass: skip ring read/write and use GRU output directly.
        if self.bypass_ring:
            h = torch.zeros(B, self.slot_dim, device=device, dtype=x.dtype)
            movement_cost = torch.tensor(0.0, device=device, dtype=x.dtype)
            pointer_addresses = torch.zeros(B, device=device, dtype=torch.long)
            for t in range(T):
                inp = self.input_proj(x[:, t, :])
                inp = self._apply_activation(inp)
                nan_guard("inp", inp, t)
                h = self.gru(inp, h)
                nan_guard("upd", h, t)
            logits = self.head(h, pointer_addresses)
            nan_guard("logits_final", logits, T)
            return logits, movement_cost

        state = torch.zeros(B, ring_range, self.slot_dim, device=device, dtype=x.dtype)
        # Persistent global hidden: ring acts as an external logbook, not the hidden state itself.
        h = torch.zeros(B, self.slot_dim, device=device, dtype=x.dtype)
        # randomize start pointer per sample to break symmetry (float for STE)
        if self.ptr_lock:
            ptr_float = torch.full((B,), self.ptr_lock_value, device=device, dtype=ptr_dtype) * (ring_range - 1)
        elif (not self.training) and EVAL_PTR_DETERMINISTIC:
            ptr_float = torch.zeros(B, device=device, dtype=ptr_dtype)
        else:
            ptr_float = torch.rand(B, device=device, dtype=ptr_dtype) * (ring_range - 1)
        # last K pointers tensorized (initialize from starting pointer)
        ptr_int_init = torch.floor(torch.remainder(ptr_float, ring_range)).clamp(0, ring_range - 1).long()
        last_ptrs = ptr_int_init.view(B, 1).repeat(1, self.blur_window)
        hist = torch.zeros(self.pointer_hist_bins, device=device, dtype=torch.long)
        satiety_exited = torch.zeros(B, device=device, dtype=torch.bool)
        ptr_vel = torch.zeros(B, device=device, dtype=ptr_dtype)

        movement_cost = 0.0
        raw_movement_cost = 0.0
        # Dynamic pointer trace (loops/motion)
        prev_ptr_int = None
        prev_prev_ptr_int = None
        dwell_len = torch.zeros(B, device=device, dtype=torch.long)
        max_dwell = torch.zeros(B, device=device, dtype=torch.long)
        flip_count = torch.zeros(B, device=device, dtype=torch.long)
        pingpong_count = torch.zeros(B, device=device, dtype=torch.long)
        total_active_steps = 0
        active_steps_per_sample = torch.zeros(B, device=device, dtype=torch.long)
        collect_xray = return_xray or self.collect_xray
        target_mask = None
        gate_sat_count = 0.0
        gate_sat_total = 0.0
        h_abs_sum = 0.0
        h_abs_count = 0
        if collect_xray:
            target_mask = torch.zeros(B, int(ring_range), device=device, dtype=x.dtype)
            attn_max = torch.zeros(B, device=device, dtype=x.dtype)
        ctrl_inertia_mean = None
        ctrl_deadzone_mean = None
        ctrl_walk_mean = None
        # Internal state loop metrics (A-B-A patterns in hidden state modes)
        if self.state_loop_metrics:
            loop_samples = B if self.state_loop_samples <= 0 else min(B, self.state_loop_samples)
            mode_prev = torch.full((loop_samples,), -1, device=device, dtype=torch.long)
            mode_prevprev = torch.full((loop_samples,), -1, device=device, dtype=torch.long)
            mode_dwell = torch.zeros(loop_samples, device=device, dtype=torch.long)
            mode_max_dwell = torch.zeros(loop_samples, device=device, dtype=torch.long)
            mode_flip = torch.zeros(loop_samples, device=device, dtype=torch.long)
            mode_abab = torch.zeros(loop_samples, device=device, dtype=torch.long)
            mode_counts = torch.zeros(self.state_loop_dim, device=device, dtype=torch.long)
            mode_steps = 0

        satiety_enabled = SATIETY_THRESH > 0.0 and self.num_classes > 1

        for t in range(T):
            active_mask = ~satiety_exited
            if not active_mask.any():
                break
            # Per-step magnitude accumulator for adaptive decay.
            h_abs_sum_step = 0.0
            h_abs_count_step = 0

            # Hard guard against NaN/Inf pointer values before any indexing.
            ptr_float = torch.nan_to_num(ptr_float, nan=0.0, posinf=ring_range - 1, neginf=0.0)

            # Adaptive decay: bleed a bit of state only if decay is enabled and activations spike.
            if STATE_DECAY < 1.0:
                decay_base = min(max(STATE_DECAY, 0.0), 1.0)
                # Smooth decay base by ring length: longer rings get slightly lower base.
                decay_base = max(0.9, decay_base - min(0.1, (self.ring_len / 2000.0)))
                h_mag_step = 0.0
                if 'h_abs_sum_step' in locals() and h_abs_count_step > 0:
                    h_mag_step = h_abs_sum_step / max(h_abs_count_step, 1)
                target = 0.3
                k = 0.5
                extra = k * max(0.0, h_mag_step - target)
                decay = max(0.9, min(1.0, decay_base - extra))
                decay_vec = active_mask.to(state.dtype) * decay + (~active_mask).to(state.dtype)
                state = state * decay_vec.view(B, 1, 1)

            inp = self.input_proj(x[:, t, :])  # [B, slot_dim]
            inp = self._apply_activation(inp)
            nan_guard("inp", inp, t)
            # Gaussian soft neighborhood over offsets [-K..K]
            offsets = torch.arange(-self.gauss_k, self.gauss_k + 1, device=device, dtype=ptr_float.dtype)
            pos_idx, weights, pos_mod = self._compute_kernel_weights(ptr_float, offsets, ring_range)
            nan_guard("weights", weights, t)
            # gather neighbors
            pos_idx_exp = pos_idx.unsqueeze(-1).expand(-1, -1, self.slot_dim).clamp(0, ring_range - 1)
            neigh = state.gather(1, pos_idx_exp)  # [B,2K+1,slot_dim]
            cur = (weights.unsqueeze(-1) * neigh.to(weights.dtype)).sum(dim=1)
            if self.mobius:
                # Continuous Riemann helix: smooth phase over [0, 2*ring_len]
                # theta = 0..2pi as ptr_float goes 0..2*ring_len
                theta = (ptr_float / self.ring_len) * math.pi
                phase_cos = torch.cos(theta).unsqueeze(1)
                phase_sin = torch.sin(theta).unsqueeze(1)
                cur = cur + phase_cos * self.phase_embed[0] + phase_sin * self.phase_embed[1]

            if cur.dtype != inp.dtype:
                cur = cur.to(inp.dtype)
            # Feed ring context as an additive cue; learnable sigmoid scale avoids hard-coded mixing.
            context_scale = torch.sigmoid(self.context_logit).to(inp.dtype)
            gru_in = inp if self.aux_ring else inp + context_scale * cur
            nan_guard("gru_in", gru_in, t)
            prev_h = h
            if collect_xray:
                r_gate, z_gate, n_gate = self._compute_gru_gates(gru_in, prev_h)
                sat = (
                    (r_gate < 0.05) | (r_gate > 0.95) |
                    (z_gate < 0.05) | (z_gate > 0.95) |
                    (n_gate.abs() > 0.95)
                ).float()
                gate_sat_count += float(sat.sum().item())
                gate_sat_total += float(sat.numel())
            h_new = self.gru(gru_in, prev_h)
            # Freeze hidden for inactive samples to avoid drift after satiety exit.
            upd = torch.where(active_mask.unsqueeze(1), h_new, prev_h)
            h = upd
            nan_guard("upd", upd, t)
            if collect_xray:
                h_abs_sum += float(upd.abs().sum().item())
                h_abs_count += upd.numel()
                h_abs_sum_step += float(upd.abs().sum().item())
                h_abs_count_step += upd.numel()
            # satiety_exited: keep updating logits, but do not write state for inactive samples.
            # (Pointer freeze handled later.)
            # State loop metrics (mode sequence)
            if self.state_loop_metrics and (t % self.state_loop_every == 0):
                loop_active = active_mask[:loop_samples]
                proj = upd[:loop_samples] @ self.state_loop_proj.to(device)
                mode = torch.argmax(proj, dim=1)
                mode_counts += torch.bincount(mode, minlength=self.state_loop_dim)
                if mode_prev[0] == -1:
                    mode_prev = mode
                    mode_prevprev = mode
                    mode_dwell = torch.where(loop_active, torch.ones_like(mode_dwell), mode_dwell)
                    mode_max_dwell = torch.maximum(mode_max_dwell, mode_dwell)
                else:
                    mflip = loop_active & (mode != mode_prev)
                    mode_flip = mode_flip + mflip.long()
                    mode_dwell = torch.where(
                        loop_active,
                        torch.where(mflip, torch.ones_like(mode_dwell), mode_dwell + 1),
                        mode_dwell,
                    )
                    mode_max_dwell = torch.maximum(mode_max_dwell, mode_dwell)
                    mabab = loop_active & (mode == mode_prevprev) & (mode != mode_prev)
                    mode_abab = mode_abab + mabab.long()
                    mode_prevprev = mode_prev
                    mode_prev = mode
                mode_steps += int(loop_active.sum().item())
            # scatter-add updates using the same Gaussian weights
            upd_exp = upd.unsqueeze(1).expand(-1, weights.size(1), -1)
            contrib = (weights.unsqueeze(-1) * upd_exp).to(state.dtype)
            scale = float(self.update_scale)
            if scale != 1.0:
                contrib = contrib * scale
            contrib = contrib * active_mask.view(B, 1, 1).to(contrib.dtype)
            # scatter_add under bf16 autocast can crash on some Windows/CUDA combos.
            # Run this op in fp32, then cast back to preserve bf16 for the rest.
            state_dtype = state.dtype
            if state_dtype in (torch.float16, torch.bfloat16):
                state_fp32 = state.float()
                contrib_fp32 = contrib.float()
                state_fp32 = state_fp32.scatter_add(1, pos_idx_exp, contrib_fp32)
                state = state_fp32.to(state_dtype)
            else:
                state = state.scatter_add(1, pos_idx_exp, contrib)
            if STATE_CLIP > 0.0:
                state = state.clamp(-STATE_CLIP, STATE_CLIP)
            if collect_xray and x.size(-1) > 1:
                marker = x[:, t, 1] > 0.5
                # Align target_mask with actual pointer index (center), not the soft window.
                # This marks the ring position where the marker was seen.
                if marker.any():
                    # Use the absolute token position (t) mapped to ring index to avoid coupling to pointer state.
                    token_idx = torch.full((B,), int(t % ring_range), device=device, dtype=torch.long)
                    target_mask.scatter_add_(1, token_idx.view(B, 1), marker.float().unsqueeze(1))
                    # Warp-drive: snap pointer to token index to ensure direct marker focus.
                    ptr_float = token_idx.to(ptr_dtype)
                # Inertia drop on marker to prevent orbital limit cycles: brake when marker seen.
                # If any marker in batch, lower inertia; otherwise restore baseline.
                base_inertia = getattr(self, "ptr_inertia_base", self.ptr_inertia)
                inertia_override = os.environ.get("TP6_PTR_INERTIA_OVERRIDE")
                if inertia_override is not None:
                    base_inertia = float(inertia_override)
                # Do not override inertia based on marker; honor configured/dynamic value.
                self.ptr_inertia = base_inertia
            if collect_xray and target_mask is not None:
                # Recompute weights after any warp so attention reflects true pointer position.
                offsets_post = torch.arange(-self.gauss_k, self.gauss_k + 1, device=device, dtype=ptr_float.dtype)
                pos_idx_post, weights_post, _ = self._compute_kernel_weights(ptr_float, offsets_post, ring_range)
                step_attn = (weights_post * target_mask.gather(1, pos_idx_post)).sum(dim=1)
                attn_max = torch.maximum(attn_max, step_attn)

            prev_ptr = ptr_float
            # Read pointer (pre-update) so reads align with the location just written.
            ptr_read_phys = torch.remainder(prev_ptr, ring_range)
            if self.time_pointer:
                ptr_read_int = torch.full((B,), int(t % ring_range), device=device, dtype=torch.long)
            elif self.ptr_phantom and prev_ptr_int is not None:
                ptr_read_base = torch.floor(ptr_read_phys)
                ptr_read_off = torch.floor(torch.remainder(ptr_read_phys + self.ptr_phantom_off, ring_range))
                read_agree = ptr_read_base == ptr_read_off
                ptr_read_int = torch.where(read_agree, ptr_read_base, prev_ptr_int.float())
            else:
                ptr_read_int = torch.floor(ptr_read_phys)
            ptr_read_int = torch.clamp(ptr_read_int, 0, ring_range - 1).long()
            if self.ptr_phantom_read:
                ptr_read_phys = ptr_read_int.float()
            jump_p = None
            move_mask = None
            gate = None
            delta_pre = None
            update_allowed = (t % self.ptr_update_every) == 0
            if self.ptr_gate_mode == "steps" and self.ptr_gate_steps:
                update_allowed = update_allowed and (t in self.ptr_gate_steps)
            if self.time_pointer:
                # Deterministic pointer: follow time index directly.
                ptr_float = torch.full((B,), float(t % ring_range), device=device, dtype=ptr_dtype)
            elif self.ptr_lock or not update_allowed:
                ptr_float = prev_ptr
            elif self.ptr_warmup_steps > 0 and t < self.ptr_warmup_steps:
                # Warmup lock: keep pointer fixed to build basic features first.
                ptr_float = prev_ptr
            else:
                # Adaptive controls from the current update vector.
                inertia_use = torch.full((B,), float(self.ptr_inertia), device=device, dtype=ptr_dtype)
                deadzone_use = torch.full((B,), float(self.ptr_deadzone), device=device, dtype=ptr_dtype)
                walk_use = torch.full((B,), float(self.ptr_walk_prob), device=device, dtype=ptr_dtype)
                # Respect manual steering override: disable neural heads for inertia/deadzone/walk.
                if os.environ.get("TP6_PTR_INERTIA_OVERRIDE") is None:
                    if self.inertia_head is not None:
                        inertia_use = torch.sigmoid(self.inertia_head(upd)).squeeze(1).to(ptr_dtype)
                    if self.deadzone_head is not None:
                        # Softplus to ensure non-negative deadzone.
                        deadzone_use = torch.nn.functional.softplus(self.deadzone_head(upd)).squeeze(1).to(ptr_dtype)
                    if self.walk_head is not None:
                        walk_use = torch.sigmoid(self.walk_head(upd)).squeeze(1).to(ptr_dtype)
                # Clamp ranges
                inertia_use = torch.clamp(inertia_use, 0.0, 0.99)
                deadzone_use = torch.clamp(deadzone_use, min=0.0)
                walk_use = torch.clamp(walk_use, 0.0, 1.0)
                ctrl_inertia_mean = float(inertia_use.mean().item())
                ctrl_deadzone_mean = float(deadzone_use.mean().item())
                ctrl_walk_mean = float(walk_use.mean().item())

                theta_ptr, theta_gate = self._gather_params(ptr_float)  # base idx for params
                jump_logits = self.jump_score(upd).squeeze(1) + theta_gate
                nan_guard("jump_logits", jump_logits, t)
                p = torch.sigmoid(jump_logits)
                if 0.0 < PTR_JUMP_CAP < 1.0:
                    p = torch.clamp(p, max=PTR_JUMP_CAP)
                if PTR_JUMP_DISABLED:
                    p = torch.zeros_like(p)
                jump_p = p
                # straight-through estimator for pointer target (continuous)
                target_cont = theta_ptr  # already in [0, ring_len)
                if self.ptr_no_round:
                    target_ste = target_cont
                else:
                    target_ste = (target_cont.round() - target_cont).detach() + target_cont
                walk_ptr = torch.remainder(ptr_float + 1, ring_range)
                # allow "stay" when not jumping to reduce flapping
                walk_prob = walk_use
                stay_ptr = prev_ptr
                non_jump_ptr = self.circ_lerp(stay_ptr, walk_ptr, walk_prob, ring_range)
                # soft mix keeps gradients flowing through p and target_ste
                ptr_float = self.circ_lerp(non_jump_ptr, target_ste, p, ring_range)
                # Raw (pre-inertia) velocity for adaptive inertia control.
                ptr_float_pre = torch.where(active_mask, ptr_float, prev_ptr)
                delta_pre = self.wrap_delta(prev_ptr, ptr_float_pre, ring_range)
                raw_movement_cost = raw_movement_cost + delta_pre.abs().mean()
                # optional inertia (stay-bias)
                inertia = inertia_use
                if (inertia > 0.0).any():
                    ptr_float = self.circ_lerp(prev_ptr, ptr_float, 1.0 - inertia, ring_range)
                # optional deadzone with smooth mask (keeps gradients flowing)
                if (deadzone_use > 0.0).any():
                    delta_raw = self.wrap_delta(prev_ptr, ptr_float, ring_range)
                    tau = max(self.ptr_deadzone_tau, 1e-6)
                    move_mask = torch.sigmoid((delta_raw.abs() - deadzone_use) / tau)
                    ptr_float = torch.remainder(prev_ptr + move_mask * delta_raw, ring_range)
                # Optional velocity governor: smooths large pointer jumps into bounded motion.
                if self.ptr_vel_enabled:
                    delta_to_target = self.wrap_delta(prev_ptr, ptr_float, ring_range)
                    scale = max(self.ptr_vel_scale, 1e-6)
                    torque = torch.tanh(delta_to_target / scale) * self.ptr_vel_cap
                    ptr_vel = self.ptr_vel_decay * ptr_vel + (1.0 - self.ptr_vel_decay) * torque
                    ptr_float = prev_ptr + ptr_vel
                # Optional learned soft gate: modulates how strongly the pointer updates.
                if self.ptr_soft_gate and self.gate_head is not None:
                    gate = torch.sigmoid(self.gate_head(upd)).squeeze(1)
                    delta_gate = self.wrap_delta(prev_ptr, ptr_float, ring_range)
                    ptr_float = torch.remainder(prev_ptr + gate * delta_gate, ring_range)
            if self.ptr_vel_enabled:
                ptr_vel = torch.where(active_mask, ptr_vel, torch.zeros_like(ptr_vel))
            ptr_float = torch.where(active_mask, ptr_float, prev_ptr)
            # hard clamp for safety (keeps pointer in-bounds)
            ptr_float = torch.nan_to_num(ptr_float, nan=0.0, posinf=ring_range - 1, neginf=0.0)
            ptr_float = torch.remainder(ptr_float, ring_range)
            nan_guard("ptr_float", ptr_float, t)
            # movement cost (wrap-aware)
            delta = torch.remainder(ptr_float - prev_ptr + ring_range / 2, ring_range) - ring_range / 2
            movement_cost = movement_cost + delta.abs().mean()

            # update history tensorized: prepend read ptr, drop last
            ptr_float_phys = torch.remainder(ptr_float, ring_range)
            ptr_base = torch.floor(ptr_float_phys)
            if self.ptr_phantom and prev_ptr_int is not None:
                # Dual-grid hysteresis: use an offset quantizer; if they disagree, hold prior bin.
                ptr_off = torch.floor(torch.remainder(ptr_float_phys + self.ptr_phantom_off, ring_range))
                agree = ptr_base == ptr_off
                ptr_int = torch.where(agree, ptr_base, prev_ptr_int.float())
            else:
                ptr_int = ptr_base
            ptr_int = torch.clamp(ptr_int, 0, ring_range - 1).long()
            if self.ptr_phantom_read:
                ptr_float_phys = ptr_int.float()
            if DEBUG_STATS and (DEBUG_EVERY <= 0 or t % DEBUG_EVERY == 0):
                stats = {
                    "active_rate": float(active_mask.float().mean().item()),
                    "ptr_float_min": float(ptr_float.min().item()),
                    "ptr_float_max": float(ptr_float.max().item()),
                    "ptr_float_mean": float(ptr_float.mean().item()),
                    "ptr_float_std": float(ptr_float.std(unbiased=False).item()),
                    "ptr_delta_abs_mean": float(delta.abs().mean().item()),
                    "ptr_delta_abs_max": float(delta.abs().max().item()),
                    "ptr_int_unique": int(ptr_int.unique().numel()),
                    "ptr_int_min": int(ptr_int.min().item()),
                    "ptr_int_max": int(ptr_int.max().item()),
                    "cur_abs_max": float(cur.abs().max().item()),
                    "cur_abs_mean": float(cur.abs().mean().item()),
                    "upd_abs_max": float(upd.abs().max().item()),
                    "upd_abs_mean": float(upd.abs().mean().item()),
                }
                if jump_p is not None:
                    stats["jump_p_mean"] = float(jump_p.mean().item())
                    stats["jump_p_min"] = float(jump_p.min().item())
                    stats["jump_p_max"] = float(jump_p.max().item())
                    stats["jump_p_std"] = float(jump_p.std(unbiased=False).item())
                if move_mask is not None:
                    stats["move_mask_mean"] = float(move_mask.mean().item())
                    stats["move_mask_std"] = float(move_mask.std(unbiased=False).item())
                if gate is not None:
                    stats["gate_mean"] = float(gate.mean().item())
                    stats["gate_std"] = float(gate.std(unbiased=False).item())
                if self.ptr_vel_enabled:
                    stats["ptr_vel_abs_mean"] = float(ptr_vel.abs().mean().item())
                    stats["ptr_vel_abs_max"] = float(ptr_vel.abs().max().item())
                    stats["ptr_vel_std"] = float(ptr_vel.std(unbiased=False).item())
                if delta_pre is not None:
                    stats["ptr_delta_raw_mean"] = float(delta_pre.abs().mean().item())
                stats["ptr_update_every"] = int(self.ptr_update_every)
                stats["ptr_soft_gate"] = int(self.ptr_soft_gate)
                stats["ptr_vel_enabled"] = int(self.ptr_vel_enabled)
                if self.ptr_edge_eps > 0.0:
                    eps = self.ptr_edge_eps
                    edge_mask = (ptr_float_phys < eps) | (ptr_float_phys > (ring_range - eps))
                    stats["ptr_edge_rate"] = float(edge_mask.float().mean().item())
                stats["ptr_kernel"] = self.ptr_kernel
                self.debug_stats = stats
            last_ptrs = torch.cat([ptr_read_int.view(B, 1), last_ptrs[:, :-1]], dim=1)
            bins = torch.bucketize(ptr_int.float(), self.bin_edges.to(device)) - 1
            bins = bins.clamp(0, self.pointer_hist_bins - 1)
            # Count all samples this step; bincount avoids accidental batch collapse
            if active_mask.any():
                active_bins = bins[active_mask]
                step_counts = torch.bincount(active_bins, minlength=self.pointer_hist_bins)
            else:
                step_counts = torch.zeros_like(hist)
            hist = hist + step_counts

            # Dynamic pointer trace metrics (only count active samples)
            if prev_ptr_int is None:
                prev_ptr_int = ptr_int
                prev_prev_ptr_int = ptr_int
                dwell_len = torch.where(active_mask, torch.ones_like(dwell_len), dwell_len)
                max_dwell = torch.maximum(max_dwell, dwell_len)
            else:
                flip = active_mask & (ptr_int != prev_ptr_int)
                flip_count = flip_count + flip.long()
                # Dwell length updates only for active samples
                dwell_len = torch.where(
                    active_mask,
                    torch.where(flip, torch.ones_like(dwell_len), dwell_len + 1),
                    dwell_len,
                )
                max_dwell = torch.maximum(max_dwell, dwell_len)
                pingpong = active_mask & (ptr_int == prev_prev_ptr_int) & (ptr_int != prev_ptr_int)
                pingpong_count = pingpong_count + pingpong.long()
                prev_prev_ptr_int = prev_ptr_int
                prev_ptr_int = ptr_int
            total_active_steps += int(active_mask.sum().item())
            active_steps_per_sample += active_mask.long()

            # Optional auto-adjust of pointer update cadence (simple EMA controller)
            if self.ptr_update_auto and (t % self.ptr_update_every_step == 0) and total_active_steps > 0:
                flip_rate = float(flip_count.sum().item() / max(1, total_active_steps))
                if self.ptr_update_ema_state is None:
                    ema = flip_rate
                else:
                    ema = self.ptr_update_ema * self.ptr_update_ema_state + (1.0 - self.ptr_update_ema) * flip_rate
                self.ptr_update_ema_state = ema
                # If flapping is high, slow down pointer updates (larger stride)
                if ema > self.ptr_update_target_flip:
                    self.ptr_update_every = min(self.ptr_update_max, self.ptr_update_every + 1)
                elif ema < self.ptr_update_target_flip * 0.5:
                    self.ptr_update_every = max(self.ptr_update_min, self.ptr_update_every - 1)

            # satiety check (optionally soft slice readout)
            if self.soft_readout:
                k = self.soft_readout_k
                offsets = torch.arange(-k, k + 1, device=device, dtype=ptr_float_phys.dtype)
                pos_idx, w, _ = self._compute_kernel_weights(
                    ptr_read_phys, offsets, ring_range, tau_override=self.soft_readout_tau
                )
                pos_idx_exp = pos_idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
                gathered = state.gather(1, pos_idx_exp)
                fused = (w.unsqueeze(-1) * gathered.to(w.dtype)).sum(dim=1)
                if fused.dtype != state.dtype:
                    fused = fused.to(state.dtype)
            else:
                gather_idx = last_ptrs.clamp(0, ring_range - 1)
                gather_idx_exp = gather_idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
                gathered = state.gather(1, gather_idx_exp)
                fused = gathered.mean(dim=1)
            if fused.dtype != h.dtype:
                fused = fused.to(h.dtype)
            read_vec = fused + h
            logits = self.head(read_vec, ptr_int)
            nan_guard("logits_step", logits, t)
            if satiety_enabled:
                probs = torch.softmax(logits, dim=1)
                confident = probs.max(dim=1).values > SATIETY_THRESH
                satiety_exited = satiety_exited | confident

        # final readout
        if self.soft_readout:
            k = self.soft_readout_k
            offsets = torch.arange(-k, k + 1, device=device, dtype=ptr_float_phys.dtype)
            pos_idx, w, _ = self._compute_kernel_weights(
                ptr_read_phys, offsets, ring_range, tau_override=self.soft_readout_tau
            )
            pos_idx_exp = pos_idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
            gathered = state.gather(1, pos_idx_exp)
            fused = (w.unsqueeze(-1) * gathered.to(w.dtype)).sum(dim=1)
            if fused.dtype != state.dtype:
                fused = fused.to(state.dtype)
        else:
            # With blur_window=1, just read the current pointer bin.
            gather_idx = ptr_int.clamp(0, ring_range - 1).unsqueeze(1).unsqueeze(2)  # [B,1,1]
            gather_idx_exp = gather_idx.expand(-1, 1, self.slot_dim)  # [B,1,D]
            fused = state.gather(1, gather_idx_exp).squeeze(1)
        if fused.dtype != h.dtype:
            fused = fused.to(h.dtype)
        read_vec = fused + h
        logits = self.head(read_vec, ptr_int)
        nan_guard("logits_final", logits, T)

        self.pointer_hist = hist.detach().cpu()
        self.satiety_exits = int(satiety_exited.sum().item())
        # Expose last pointer bins for MI/TEI evaluation.
        last_bins = torch.bucketize(ptr_int.float(), self.bin_edges.to(device)) - 1
        last_bins = last_bins.clamp(0, self.pointer_hist_bins - 1).detach().cpu()
        self.last_ptr_bins = last_bins
        self.last_ptr_int = ptr_int.detach().cpu()
        # Pointer dynamics summary
        denom = max(1, total_active_steps)
        self.ptr_flip_rate = float(flip_count.sum().item()) / denom
        self.ptr_pingpong_rate = float(pingpong_count.sum().item()) / denom
        self.ptr_max_dwell = int(max_dwell.max().item()) if max_dwell.numel() else 0
        # Mean dwell as active_steps / (flip_count + 1) per sample, averaged.
        mean_dwell = active_steps_per_sample.float() / (flip_count.float() + 1.0)
        self.ptr_mean_dwell = float(mean_dwell.mean().item()) if mean_dwell.numel() else 0.0
        if self.state_loop_metrics:
            mode_denom = max(1, mode_steps)
            self.state_loop_flip_rate = float(mode_flip.sum().item()) / mode_denom
            self.state_loop_abab_rate = float(mode_abab.sum().item()) / mode_denom
            self.state_loop_max_dwell = int(mode_max_dwell.max().item()) if mode_max_dwell.numel() else 0
            self.state_loop_mean_dwell = float(mode_dwell.float().mean().item()) if mode_dwell.numel() else 0.0
            if mode_counts.sum() > 0:
                probs = mode_counts.float() / mode_counts.sum()
                ent = -(probs * torch.log(probs + 1e-12)).sum() / math.log(2.0)
                self.state_loop_entropy = float(ent.item())
            else:
                self.state_loop_entropy = None
        steps_used = max(1, t + 1)
        move_penalty = movement_cost / steps_used
        self.ptr_delta_abs_mean = float(move_penalty)
        raw_move_penalty = raw_movement_cost / steps_used
        self.ptr_delta_raw_mean = float(raw_move_penalty)
        if collect_xray:
            xray = {}
            if target_mask is not None and "attn_max" in locals():
                attn_mass = attn_max.mean()
                xray["attn_mass"] = float(attn_mass.item())
            if gate_sat_total > 0:
                xray["gate_sat"] = float(gate_sat_count / gate_sat_total)
            if h_abs_count > 0:
                xray["h_mag"] = float(h_abs_sum / h_abs_count)
            # Pointer motion telemetry
            xray["ptr_delta_abs_mean"] = self.ptr_delta_abs_mean
            xray["ptr_delta_raw_mean"] = self.ptr_delta_raw_mean
            raw = getattr(self, "ptr_delta_raw_mean", None)
            gs = getattr(self, "ground_speed", None)
            if raw is not None and gs is not None and math.isfinite(raw) and math.isfinite(gs):
                xray["damp_ratio"] = float(raw / max(gs, 1e-6))
            if ctrl_inertia_mean is not None:
                xray["ptr_inertia_dyn"] = ctrl_inertia_mean
            if ctrl_deadzone_mean is not None:
                xray["ptr_deadzone_dyn"] = ctrl_deadzone_mean
            if ctrl_walk_mean is not None:
                xray["ptr_walk_dyn"] = ctrl_walk_mean
            return logits, move_penalty, xray
        return logits, move_penalty


class FileAudioDataset(torch.utils.data.Dataset):
    def __init__(self, items, num_classes, sample_rate=16000, max_len=16000, n_mels=64, max_frames=100):
        self.items = items
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.max_frames = max_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.size(1) < self.max_len:
            pad = self.max_len - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            wav = wav[:, : self.max_len]
        with torch.no_grad():
            mel = self.melspec(wav)  # [1, n_mels, frames]
            mel = torch.log(mel + 1e-6)
        mel = mel.squeeze(0).transpose(0, 1)  # [frames, n_mels]
        if mel.size(0) < self.max_frames:
            pad = self.max_frames - mel.size(0)
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad))
        else:
            mel = mel[: self.max_frames]
        return mel, label


def _download_zip(url, dest_dir, tag):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, f"{tag}.zip")
    if not os.path.exists(zip_path):
        log(f"Downloading {tag}...")
        urllib.request.urlretrieve(url, zip_path)
    return zip_path


def _extract_zip(zip_path, dest_dir):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def get_fsdd_loader():
    if not HAS_TORCHAUDIO:
        raise RuntimeError("torchaudio not available")
    root = os.path.join(DATA_DIR, "fsdd")
    if OFFLINE_ONLY and not os.path.exists(root):
        raise RuntimeError("offline mode and fsdd not present")
    zip_url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
    zip_path = _download_zip(zip_url, root, "fsdd")
    extract_root = os.path.join(root, "free-spoken-digit-dataset-master")
    if not os.path.exists(extract_root):
        _extract_zip(zip_path, root)

    recordings = os.path.join(extract_root, "recordings")
    items = []
    for fname in sorted(os.listdir(recordings)):
        if not fname.endswith(".wav"):
            continue
        label = int(fname.split("_")[0])
        items.append((os.path.join(recordings, fname), label))
    items = items[:MAX_SAMPLES]
    dataset = FileAudioDataset(items, num_classes=10)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    return loader, 10


def get_seq_mnist_loader():
    SYNTH_META.clear()
    synth_env = os.environ.get("TP6_SYNTH", "0").strip()
    if synth_env == "1":
        synth_mode = SYNTH_MODE
        base_seq_len = max(1, int(SYNTH_LEN))
        SYNTH_META.update({"enabled": True, "mode": synth_mode, "synth_len": base_seq_len})
        n_samples = max(1, MAX_SAMPLES)
        if synth_mode == "assoc_clean":
            seq_len = base_seq_len
            pairs = max(1, int(ASSOC_PAIRS))
            keys = max(2, int(ASSOC_KEYS))
            min_len = pairs * 2 + 1
            bump_attempts = 0
            max_bumps = 5

            def _build_assoc(seq_len_local: int):
                x_local = torch.zeros((n_samples, seq_len_local, 1), dtype=torch.float32)
                y_local = torch.zeros((n_samples,), dtype=torch.long)
                max_start_local = seq_len_local - 3  # reserve last token for query
                for idx in range(n_samples):
                    used = set()
                    pair_specs = []
                    starts = list(range(0, max_start_local + 1))
                    random.shuffle(starts)
                    for cand in starts:
                        if cand in used or (cand + 1) in used:
                            continue
                        used.add(cand)
                        used.add(cand + 1)
                        key_id = random.randint(0, keys - 1)
                        val = random.randint(0, 1)
                        key_token = float(2 + key_id)
                        val_token = -1.0 if val == 0 else -2.0
                        x_local[idx, cand, 0] = key_token
                        x_local[idx, cand + 1, 0] = val_token
                        pair_specs.append((key_id, val, key_token))
                        if len(pair_specs) >= pairs:
                            break
                    if len(pair_specs) < pairs:
                        return None, None
                    _, q_val, q_token = random.choice(pair_specs)
                    x_local[idx, -1, 0] = q_token
                    y_local[idx] = q_val
                return x_local, y_local

            if seq_len < min_len:
                log(f"[synth] assoc_clean bump len from {seq_len} to {min_len} (min_len)")
                seq_len = min_len

            x = None
            y = None
            while bump_attempts <= max_bumps:
                x, y = _build_assoc(seq_len)
                if x is not None:
                    break
                bump_attempts += 1
                new_len = seq_len + max(2, pairs) * 2
                log(f"[synth] assoc_clean bump len from {seq_len} to {new_len} (placement failed)")
                seq_len = new_len

            if x is None:
                raise RuntimeError("assoc_clean: failed to place non-overlapping pairs after bumps")

            SYNTH_META.update({"assoc_keys": keys, "assoc_pairs": pairs, "synth_len": seq_len})
            num_classes = 2
            log(f"[synth] mode=assoc_clean rows={int(n_samples)} keys={keys} pairs={pairs} len={seq_len}")
            class _Synth(torch.utils.data.Dataset):
                def __len__(self):
                    return n_samples

                def __getitem__(self, item):
                    return x[item], y[item]

            ds = _Synth()

            def collate(batch):
                xs, ys = zip(*batch)
                return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate,
            )
            return loader, num_classes, collate
        elif synth_mode == "assoc_byte":
            seq_len = base_seq_len
            pairs = max(1, int(ASSOC_PAIRS))
            keys = max(2, int(ASSOC_KEYS))
            val_range = max(2, int(ASSOC_VAL_RANGE))
            min_len = pairs * 2 + 1
            bump_attempts = 0
            max_bumps = 5

            def _build_assoc_byte(seq_len_local: int):
                x_local = torch.zeros((n_samples, seq_len_local, 1), dtype=torch.float32)
                y_local = torch.zeros((n_samples,), dtype=torch.long)
                max_start_local = seq_len_local - 3  # reserve last token for query
                for idx in range(n_samples):
                    used = set()
                    pair_specs = []
                    starts = list(range(0, max_start_local + 1))
                    random.shuffle(starts)
                    for cand in starts:
                        if cand in used or (cand + 1) in used:
                            continue
                        used.add(cand)
                        used.add(cand + 1)
                        key_id = random.randint(0, keys - 1)
                        val = random.randint(0, val_range - 1)
                        key_token = float(2 + key_id)
                        val_token = -float(val + 1)  # keep value tokens distinct from keys/distractors
                        x_local[idx, cand, 0] = key_token
                        x_local[idx, cand + 1, 0] = val_token
                        pair_specs.append((key_id, val, key_token))
                        if len(pair_specs) >= pairs:
                            break
                    if len(pair_specs) < pairs:
                        return None, None
                    _, q_val, q_token = random.choice(pair_specs)
                    x_local[idx, -1, 0] = q_token
                    y_local[idx] = q_val
                return x_local, y_local

            if seq_len < min_len:
                log(f"[synth] assoc_byte bump len from {seq_len} to {min_len} (min_len)")
                seq_len = min_len

            x = None
            y = None
            while bump_attempts <= max_bumps:
                x, y = _build_assoc_byte(seq_len)
                if x is not None:
                    break
                bump_attempts += 1
                new_len = seq_len + max(2, pairs) * 2
                log(f"[synth] assoc_byte bump len from {seq_len} to {new_len} (placement failed)")
                seq_len = new_len

            if x is None:
                raise RuntimeError("assoc_byte: failed to place non-overlapping pairs after bumps")

            SYNTH_META.update(
                {"assoc_keys": keys, "assoc_pairs": pairs, "assoc_val_range": val_range, "synth_len": seq_len}
            )
            num_classes = val_range
            log(
                f"[synth] mode=assoc_byte rows={int(n_samples)} keys={keys} vals={val_range} "
                f"pairs={pairs} len={seq_len}"
            )

            class _SynthByte(torch.utils.data.Dataset):
                def __len__(self):
                    return n_samples

                def __getitem__(self, item):
                    return x[item], y[item]

            ds = _SynthByte()

            def collate(batch):
                xs, ys = zip(*batch)
                return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate,
            )
            return loader, num_classes, collate
        elif synth_mode == "hand_kv":
            hand_path = os.environ.get("TP6_HAND_PATH", os.path.join(DATA_DIR, "hand_kv.jsonl"))
            pad_len = int(os.environ.get("TP6_HAND_PAD_LEN", "0"))
            rows = []
            with open(hand_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            if MAX_SAMPLES and MAX_SAMPLES < len(rows):
                rows = rows[:MAX_SAMPLES]
            if len(rows) < HAND_MIN:
                raise RuntimeError(
                    f"hand_kv dataset too small: {len(rows)} rows < HAND_MIN={HAND_MIN} ({hand_path})"
                )
            SYNTH_META.update({"hand_path": hand_path, "rows": len(rows), "pad_len": pad_len})
            log(f"[synth] mode=hand_kv rows={len(rows)} pad_len={pad_len} path={hand_path}")

            xs = []
            ys = []
            for row in rows:
                seq = row.get("x", [])
                label = row.get("y", 0)
                if pad_len > 0:
                    if len(seq) < pad_len:
                        seq = seq + [0] * (pad_len - len(seq))
                    else:
                        seq = seq[:pad_len]
                xs.append(torch.tensor(seq, dtype=torch.float32).view(-1, 1))
                ys.append(int(label))

            class _ListSynth(torch.utils.data.Dataset):
                def __len__(self):
                    return len(xs)

                def __getitem__(self, idx):
                    return xs[idx], ys[idx]

            ds = _ListSynth()

            def collate(batch):
                xs_b, ys_b = zip(*batch)
                return torch.stack(xs_b, dim=0), torch.tensor(ys_b, dtype=torch.long)

            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate,
            )
            num_classes = max(2, max(ys) + 1 if ys else 2)
            return loader, num_classes, collate
        else:
            x = torch.randint(0, 2, (n_samples, base_seq_len, 1), dtype=torch.float32)
            if synth_mode == "markov0":
                y = x[:, -1, 0].to(torch.long)
            elif synth_mode == "markov0_flip":
                y = (1 - x[:, -1, 0]).to(torch.long)
            elif synth_mode == "const0":
                y = torch.zeros((n_samples,), dtype=torch.long)
            else:
                y = torch.randint(0, 2, (n_samples,), dtype=torch.long)
        SYNTH_META.update({"rows": int(n_samples)})
        log(f"[synth] mode={synth_mode} rows={int(n_samples)}")

        class _Synth(torch.utils.data.Dataset):
            def __len__(self):
                return n_samples

            def __getitem__(self, idx):
                return x[idx], y[idx]

        ds = _Synth()

        def collate(batch):
            xs, ys = zip(*batch)
            return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=SYNTH_SHUFFLE,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate,
        )
        return loader, 2, collate

    try:
        import torchvision.transforms as T
        from torchvision.datasets import MNIST
    except Exception as exc:
        raise RuntimeError("torchvision is required for MNIST mode") from exc

    transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
    ds = MNIST(os.path.join(DATA_DIR, "mnist_seq"), train=True, download=not OFFLINE_ONLY, transform=transform)
    if MAX_SAMPLES and MAX_SAMPLES < len(ds):
        ds = Subset(ds, list(range(MAX_SAMPLES)))

    def collate(batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)  # [B,1,16,16]
        x = x.view(x.size(0), -1, 1)  # [B,256,1]
        y = torch.tensor(ys, dtype=torch.long)
        return x, y

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )
    return loader, 10, collate


def build_synth_pair_loaders():
    n_samples = max(1, MAX_SAMPLES)
    seq_len = max(1, SYNTH_LEN)
    g = torch.Generator()
    g.manual_seed(SEED)
    x = torch.randint(0, 2, (n_samples, seq_len, 1), dtype=torch.float32, generator=g)
    y_a = x[:, -1, 0].to(torch.long)
    y_b = (1 - x[:, -1, 0]).to(torch.long)

    class _FixedSynth(torch.utils.data.Dataset):
        def __init__(self, xs, ys):
            self.xs = xs
            self.ys = ys

        def __len__(self):
            return self.xs.size(0)

        def __getitem__(self, idx):
            return self.xs[idx], self.ys[idx]

    ds_a = _FixedSynth(x, y_a)
    ds_b = _FixedSynth(x, y_b)

    def collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

    loader_a = DataLoader(
        ds_a,
        batch_size=BATCH_SIZE,
        shuffle=SYNTH_SHUFFLE,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )
    loader_b = DataLoader(
        ds_b,
        batch_size=BATCH_SIZE,
        shuffle=SYNTH_SHUFFLE,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )
    return loader_a, loader_b, collate


def build_eval_loader_from_subset(train_ds, input_collate=None):
    eval_size = min(EVAL_SAMPLES, len(train_ds))
    if isinstance(train_ds, Subset):
        indices = train_ds.indices[:eval_size]
        eval_subset = Subset(train_ds.dataset, indices)
    else:
        eval_subset = Subset(train_ds, list(range(eval_size)))

    def _collate(batch):
        if input_collate:
            return input_collate(batch)
        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

    loader = DataLoader(
        eval_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_collate,
    )
    return loader, eval_size


def build_eval_loader_from_dataset(eval_ds, input_collate=None):
    eval_size = min(EVAL_SAMPLES, len(eval_ds))
    eval_subset = Subset(eval_ds, list(range(eval_size)))

    def _collate(batch):
        if input_collate:
            return input_collate(batch)
        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

    loader = DataLoader(
        eval_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_collate,
    )
    return loader, eval_size


def log_eval_overlap(train_ds, eval_ds, eval_size, label):
    def base_and_indices(ds):
        if isinstance(ds, Subset):
            return ds.dataset, set(ds.indices)
        return ds, None

    train_base, train_idx = base_and_indices(train_ds)
    eval_base, eval_idx = base_and_indices(eval_ds)

    if train_base is eval_base:
        if eval_idx is None:
            overlap = eval_size
        elif train_idx is None:
            overlap = len(eval_idx)
        else:
            overlap = len(train_idx.intersection(eval_idx))
        log(f"[eval] split={label} overlap={overlap}/{eval_size} (shared base dataset)")
    else:
        log(f"[eval] split={label} overlap=0/{eval_size} (disjoint datasets)")


def train_wallclock(model, loader, dataset_name, model_name, num_classes, wall_clock=WALL_CLOCK_SECONDS):
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
        )
        model.ptr_update_auto = False
    else:
        model.ptr_update_auto = PTR_UPDATE_AUTO
    # Reset adaptive scales to env defaults to avoid carrying prior runs.
    init_scale = AGC_SCALE_MIN
    model.update_scale = init_scale
    model.agc_scale_max = AGC_SCALE_MAX
    model.agc_scale_cap = AGC_SCALE_MAX
    env_inertia = os.environ.get("TP6_PTR_INERTIA_OVERRIDE")
    if env_inertia is not None:
        model.ptr_inertia = float(env_inertia)
        model.ptr_inertia_ema = model.ptr_inertia

    start = time.time()
    # Allow indefinite runs unless explicitly told to honor wall-clock limits.
    ignore_wall_clock = os.environ.get("TP6_IGNORE_WALL_CLOCK") == "1"
    end_time = float("inf") if ignore_wall_clock else (start + wall_clock)
    last_heartbeat = start
    last_live_trace = start
    step = 0
    if RESUME and os.path.exists(CHECKPOINT_PATH):
        resume_path = os.path.abspath(CHECKPOINT_PATH)
        log(f"Resume requested, attempting load: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE)
        try:
            model.load_state_dict(ckpt["model"])
        except RuntimeError:
            if MOBIUS_ENABLED:
                log("MOBIUS enabled: retrying load_state_dict(strict=False) due to key mismatch")
                missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
                if missing:
                    log(f"MOBIUS load missing keys: {missing}")
                if unexpected:
                    log(f"MOBIUS load unexpected keys: {unexpected}")
            else:
                raise
        if MOBIUS_ENABLED:
            log("MOBIUS enabled: skipping optimizer/scaler load for clean restart")
        else:
            optimizer.load_state_dict(ckpt["optim"])
            if ckpt.get("scaler") and USE_AMP:
                scaler.load_state_dict(ckpt["scaler"])
        step = int(ckpt.get("step", 0))
        losses = list(ckpt.get("losses", []))
        if "update_scale" in ckpt:
            model.update_scale = float(ckpt["update_scale"])
        if "ptr_inertia" in ckpt:
            model.ptr_inertia = float(ckpt["ptr_inertia"])
        if "ptr_inertia_ema" in ckpt:
            model.ptr_inertia_ema = float(ckpt["ptr_inertia_ema"])
        if "agc_scale_max" in ckpt:
            model.agc_scale_max = float(ckpt["agc_scale_max"])
        if "ground_speed_ema" in ckpt:
            model.ground_speed_ema = ckpt["ground_speed_ema"]
        if "ground_speed_limit" in ckpt:
            model.ground_speed_limit = ckpt["ground_speed_limit"]
        # Honor env overrides after resume (no hidden reset).
        env_scale_init = os.environ.get("TP6_SCALE_INIT")
        env_scale_max = os.environ.get("TP6_SCALE_MAX")
        if env_scale_init is not None:
            model.update_scale = float(env_scale_init)
        if env_scale_max is not None:
            model.agc_scale_max = float(env_scale_max)
        env_inertia = os.environ.get("TP6_PTR_INERTIA_OVERRIDE")
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
    xray_enabled = os.getenv("TP6_XRAY", "0") == "1"
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
                    if EXPERT_HEADS > 1:
                        last_ptr = getattr(model, "last_ptr_int", None)
                        if last_ptr is not None:
                            try:
                                active_experts = int(torch.unique(last_ptr.to(torch.long) % EXPERT_HEADS).numel())
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
            scaler.scale(loss).backward()
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
                model, grad_norm if hasattr(model, "theta_ptr_reduced") else None, raw_delta, step=step
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
            if THERMO_ENABLED and hasattr(model, "ptr_flip_rate") and step % max(1, THERMO_EVERY) == 0:
                focus_ctl = focus
                tension_ctl = tension
                if (focus_ctl is None or tension_ctl is None) and getattr(model, "debug_shard_info", None):
                    shard_info = model.debug_shard_info
                    if focus_ctl is None:
                        focus_ctl = shard_info.get("focus")
                    if tension_ctl is None:
                        tension_ctl = shard_info.get("tension")
                flip_ema = apply_thermostat(
                    model,
                    float(model.ptr_flip_rate),
                    flip_ema,
                    focus=focus_ctl,
                    tension=tension_ctl,
                    raw_delta=getattr(model, "ptr_delta_raw_mean", None),
                )
            if panic_reflex is not None:
                ctrl = panic_reflex.update(float(loss))
                panic_status = ctrl["status"]
                if panic_status == "PANIC":
                    if os.environ.get("TP6_PTR_INERTIA_OVERRIDE") is None:
                        model.ptr_inertia = ctrl["inertia"]
                        model.ptr_walk_prob = ctrl["walk_prob"]
            ptr_velocity_raw = getattr(model, "ptr_delta_raw_mean", None)
            # Do not override manual inertia when TP6_PTR_INERTIA_OVERRIDE is set.
            if os.environ.get("TP6_PTR_INERTIA_OVERRIDE") is None:
                apply_inertia_auto(model, ptr_velocity_raw, panic_status == "PANIC")
            if os.environ.get("TP6_FORCE_CADENCE_1") == "1":
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
                raw_delta_text = f", raw_delta={float(raw_delta):.3f}" if raw_delta is not None else ""
                ground_speed_text = (
                    f", ground_speed={float(ground_speed):.3f}" if ground_speed is not None else ""
                )
                ground_speed_ema_text = (
                    f", g_ema={float(ground_speed_ema):.3f}" if ground_speed_ema is not None else ""
                )
                ground_speed_limit_text = (
                    f", g_L={float(ground_speed_limit):.3f}" if ground_speed_limit is not None else ""
                )
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
                    if "active_experts" in shard_info:
                        shard_text += f", experts={shard_info['active_experts']}"
                ptr_stats = []
                for label, val, fmt in [
                    ("flip", getattr(model, "ptr_flip_rate", None), "{:.3f}"),
                    ("dwell", getattr(model, "ptr_mean_dwell", None), "{:.2f}"),
                    ("dwell_max", getattr(model, "ptr_max_dwell", None), "{:.0f}"),
                    ("delta", getattr(model, "ptr_delta_abs_mean", None), "{:.4f}"),
                    ("delta_raw", getattr(model, "ptr_delta_raw_mean", None), "{:.4f}"),
                ]:
                    if val is None:
                        continue
                    try:
                        v = float(val)
                    except Exception:
                        continue
                    if math.isfinite(v):
                        ptr_stats.append(f"{label}={fmt.format(v)}")
                ptr_text = f", ptr[{'; '.join(ptr_stats)}]" if ptr_stats else ""
                agc_cap = getattr(model, "agc_scale_cap", getattr(model, "agc_scale_max", AGC_SCALE_MAX))
                log(
                    f"{dataset_name} | {model_name} | step {step:04d} | loss {loss.item():.4f} | "
                    f"t={elapsed:.1f}s | ctrl(inertia={model.ptr_inertia:.2f}, deadzone={model.ptr_deadzone:.2f}, walk={model.ptr_walk_prob:.2f}, cadence={model.ptr_update_every}, scale={float(scale_log):.3f}, cap={agc_cap:.3f}"
                    f"{raw_delta_text}{ground_speed_text}{ground_speed_ema_text}{ground_speed_limit_text}{shard_text}{ptr_text})"
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
                        trace["ptr_deadzone"] = model.ptr_deadzone
                        trace["ptr_walk_prob"] = model.ptr_walk_prob
                        trace["update_scale"] = getattr(model, "update_scale", UPDATE_SCALE)
                        trace["ground_speed"] = getattr(model, "ground_speed", None)
                        trace["ground_speed_ema"] = getattr(model, "ground_speed_ema", None)
                        trace["ground_speed_limit"] = getattr(model, "ground_speed_limit", None)
                        if panic_reflex is not None:
                            trace["panic_status"] = panic_status
                    if getattr(model, "debug_stats", None):
                        trace.update(model.debug_stats)
                    if hasattr(model, "state_loop_entropy"):
                        trace["state_loop_entropy"] = model.state_loop_entropy
                        trace["state_loop_flip_rate"] = getattr(model, "state_loop_flip_rate", None)
                        trace["state_loop_abab_rate"] = getattr(model, "state_loop_abab_rate", None)
                        trace["state_loop_mean_dwell"] = getattr(model, "state_loop_mean_dwell", None)
                        trace["state_loop_max_dwell"] = getattr(model, "state_loop_max_dwell", None)
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
                            f.write(json.dumps(trace) + "\\n")
                    except Exception as e:
                        log(f"live_trace write failed: {e}")
                    last_live_trace = now

            # Explicitly release intermediates to reduce fragmentation risk.
            del outputs, loss
            if pulse_applied:
                model.ptr_walk_prob = prev_walk_prob
            step += 1
            # Respect MAX_STEPS unless explicitly disabled (used by infinite wrapper).
            ignore_max_steps = os.environ.get("TP6_IGNORE_MAX_STEPS") == "1"
            if (not ignore_max_steps) and MAX_STEPS > 0 and step >= MAX_STEPS:
                break
            if SAVE_EVERY_STEPS > 0 and step % SAVE_EVERY_STEPS == 0:
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
                if SAVE_HISTORY:
                    torch.save(ckpt, step_path)
                    log(f"Checkpoint saved @ step {step} -> {step_path}")
                    if (not is_finite) and SAVE_BAD:
                        torch.save(ckpt, bad_path)
                        log(f"Non-finite checkpoint saved @ step {step} -> {bad_path}")
                if is_finite:
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


def eval_model(model, loader, dataset_name, model_name):
    model = model.to(DEVICE, dtype=DTYPE)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    mi_bins = getattr(model, "pointer_hist_bins", 128)
    joint = torch.zeros((model.head.out_features, mi_bins), dtype=torch.long)
    joint_shuffle = torch.zeros_like(joint) if MI_SHUFFLE else None
    ptr_flip_sum = 0.0
    ptr_steps = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            if inputs.dtype != DTYPE:
                inputs = inputs.to(DTYPE)
            targets = targets.to(DEVICE, non_blocking=True)
            with amp_autocast():
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_seen += inputs.size(0)
            if hasattr(model, "last_ptr_bins"):
                bins = model.last_ptr_bins.to(torch.long)
                labels = targets.detach().cpu().to(torch.long)
                idx = labels * mi_bins + bins
                joint += torch.bincount(idx, minlength=joint.numel()).view_as(joint)
                if MI_SHUFFLE:
                    perm = torch.randperm(labels.numel())
                    labels_shuf = labels[perm]
                    idx_shuf = labels_shuf * mi_bins + bins
                    joint_shuffle += torch.bincount(idx_shuf, minlength=joint.numel()).view_as(joint)
            if hasattr(model, "ptr_flip_rate"):
                ptr_flip_sum += float(model.ptr_flip_rate)
                ptr_steps += 1
    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    mi_bits = None
    mi_bits_shuffle = None
    if joint.sum() > 0:
        p = joint.float() / joint.sum()
        pc = p.sum(dim=1, keepdim=True)
        pb = p.sum(dim=0, keepdim=True)
        mi = (p * (torch.log(p + 1e-12) - torch.log(pc * pb + 1e-12))).sum()
        mi_bits = float(mi / math.log(2.0))
    if joint_shuffle is not None and joint_shuffle.sum() > 0:
        p = joint_shuffle.float() / joint_shuffle.sum()
        pc = p.sum(dim=1, keepdim=True)
        pb = p.sum(dim=0, keepdim=True)
        mi = (p * (torch.log(p + 1e-12) - torch.log(pc * pb + 1e-12))).sum()
        mi_bits_shuffle = float(mi / math.log(2.0))
    ptr_flip_rate = (ptr_flip_sum / ptr_steps) if ptr_steps else None
    tei = None
    if mi_bits is not None and ptr_flip_rate is not None:
        tei = acc * mi_bits * (1.0 - ptr_flip_rate)
    log(f"{dataset_name} | {model_name} | eval_loss {avg_loss:.4f} | eval_acc {acc:.4f} | eval_n {total_seen}")
    return {
        "eval_loss": avg_loss,
        "eval_acc": acc,
        "eval_n": total_seen,
        "eval_mi_bits": mi_bits,
        "eval_mi_bits_shuffled": mi_bits_shuffle,
        "eval_ptr_flip_rate": ptr_flip_rate,
        "eval_tei": tei,
    }


def train_steps(model, loader, steps, dataset_name, model_name):
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
        )
        model.ptr_update_auto = False
    else:
        model.ptr_update_auto = PTR_UPDATE_AUTO
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
        scaler.scale(loss).backward()
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
            model, grad_norm_step if hasattr(model, "theta_ptr_reduced") else None, raw_delta, step=step
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
            flip_ema = apply_thermostat(
                model,
                float(model.ptr_flip_rate),
                flip_ema,
                focus=focus_ctl,
                tension=tension_ctl,
                raw_delta=getattr(model, "ptr_delta_raw_mean", None),
            )
        if panic_reflex is not None:
            ctrl = panic_reflex.update(float(loss))
            panic_status = ctrl["status"]
            if panic_status == "PANIC":
                if os.environ.get("TP6_PTR_INERTIA_OVERRIDE") is None:
                    model.ptr_inertia = ctrl["inertia"]
                    model.ptr_walk_prob = ctrl["walk_prob"]
        ptr_velocity_raw = getattr(model, "ptr_delta_raw_mean", None)
        if os.environ.get("TP6_PTR_INERTIA_OVERRIDE") is None:
            apply_inertia_auto(model, ptr_velocity_raw, panic_status == "PANIC")
        if os.environ.get("TP6_FORCE_CADENCE_1") == "1":
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
                raw_delta = getattr(model, "ptr_delta_raw_mean", None)
                raw_delta_text = f", raw_delta={float(raw_delta):.3f}" if raw_delta is not None else ""
                ground_speed = getattr(model, "ground_speed", None)
                ground_speed_text = (
                    f", ground_speed={float(ground_speed):.3f}" if ground_speed is not None else ""
                )
                ground_speed_ema = getattr(model, "ground_speed_ema", None)
                ground_speed_ema_text = (
                    f", g_ema={float(ground_speed_ema):.3f}" if ground_speed_ema is not None else ""
                )
                ground_speed_limit = getattr(model, "ground_speed_limit", None)
                ground_speed_limit_text = (
                    f", g_L={float(ground_speed_limit):.3f}" if ground_speed_limit is not None else ""
                )
                scale_log = getattr(model, "debug_scale_out", getattr(model, "update_scale", UPDATE_SCALE))
                stats_dict = getattr(model, "debug_stats", None)
                try:
                    stats_payload = json.dumps(stats_dict if stats_dict is not None else {}, separators=(",", ":"))
                except Exception:
                    stats_payload = str(stats_dict)
                debug_payload = f" | debug {stats_payload}"
                log(
                    f"{dataset_name} | {model_name} | step {step:04d}/{steps:04d} | loss {loss.item():.4f} | "
                    f"t={elapsed:.1f}s | ctrl(inertia={model.ptr_inertia:.2f}, deadzone={model.ptr_deadzone:.2f}, walk={model.ptr_walk_prob:.2f}, cadence={model.ptr_update_every}, scale={float(scale_log):.3f}{raw_delta_text}{ground_speed_text}{ground_speed_ema_text}{ground_speed_limit_text})"
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
            if SAVE_HISTORY:
                torch.save(ckpt, step_path)
                log(f"Checkpoint saved @ step {step} -> {step_path}")
                if (not is_finite) and SAVE_BAD:
                    torch.save(ckpt, bad_path)
                    log(f"Non-finite checkpoint saved @ step {step} -> {bad_path}")
            if is_finite:
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


def _is_pointer_param(name: str) -> bool:
    return (
        name.startswith("theta_ptr_reduced")
        or name.startswith("theta_gate_reduced")
        or name.startswith("jump_score")
        or name.startswith("gate_head")
    )


def mutate_state_dict(parent_state, std=EVO_MUT_STD, pointer_only: bool = False):
    child = {}
    for k, v in parent_state.items():
        if not torch.is_floating_point(v):
            child[k] = v.clone()
            continue
        if pointer_only and not _is_pointer_param(k):
            child[k] = v.clone()
            continue
        noise = torch.randn_like(v, device="cpu") * std
        child[k] = (v.cpu() + noise).to(v.dtype)
    return child


def save_evo_checkpoint(gen: int, model, train_stats, eval_stats, fitness: float) -> None:
    evo_dir = os.path.join(ROOT, "artifacts", "evolution")
    os.makedirs(evo_dir, exist_ok=True)
    payload = {
        "gen": gen,
        "model": model.state_dict(),
        "train": train_stats,
        "eval": eval_stats,
        "fitness": fitness,
    }
    latest_path = os.path.join(evo_dir, "evo_latest.pt")
    torch.save(payload, latest_path)
    if EVO_CKPT_EVERY > 0 and gen % EVO_CKPT_EVERY == 0:
        gen_path = os.path.join(evo_dir, f"evo_gen_{gen:06d}.pt")
        torch.save(payload, gen_path)
        log(f"Evolution checkpoint saved @ gen {gen} -> {gen_path}")


def run_evolution(dataset_name, loader, eval_loader, input_dim, num_classes):
    evo_dir = os.path.join(ROOT, "artifacts", "evolution")
    evo_latest = os.path.join(evo_dir, "evo_latest.pt")
    resume_state = None
    start_gen = 0
    if EVO_RESUME and os.path.exists(evo_latest):
        try:
            payload = torch.load(evo_latest, map_location="cpu")
            resume_state = payload.get("model")
            start_gen = int(payload.get("gen", -1)) + 1
            log(f"Evolution resume: loaded {evo_latest} (start_gen={start_gen})")
        except Exception as e:
            log(f"Evolution resume failed: {e}; starting fresh.")
    log(
        "=== Evolution mode | dataset="
        f"{dataset_name} | pop={EVO_POP} gens={EVO_GENS} steps/ind={EVO_STEPS} "
        f"pointer_only={int(EVO_POINTER_ONLY)} resume={int(EVO_RESUME)} start_gen={start_gen} ==="
    )
    # init population
    population = []
    if resume_state is not None:
        elite = AbsoluteHallway(input_dim=input_dim, num_classes=num_classes, ring_len=RING_LEN, slot_dim=SLOT_DIM)
        elite.load_state_dict(resume_state)
        population.append(elite)
        while len(population) < EVO_POP:
            child = AbsoluteHallway(input_dim=input_dim, num_classes=num_classes, ring_len=RING_LEN, slot_dim=SLOT_DIM)
            child.load_state_dict(
                mutate_state_dict(resume_state, std=EVO_MUT_STD, pointer_only=EVO_POINTER_ONLY)
            )
            population.append(child)
    else:
        for _ in range(EVO_POP):
            m = AbsoluteHallway(input_dim=input_dim, num_classes=num_classes, ring_len=RING_LEN, slot_dim=SLOT_DIM)
            population.append(m)

    best_eval = None
    if EVO_GENS > 0:
        gen_iter = range(start_gen, start_gen + EVO_GENS)
    else:
        gen_iter = count(start_gen)
    for gen in gen_iter:
        gen_fitness = []
        for idx, model in enumerate(population):
            train_stats = train_steps(model, loader, EVO_STEPS, dataset_name, f"evo_{gen}_{idx}")
            eval_stats = eval_model(model, eval_loader, dataset_name, f"evo_{gen}_{idx}")
            fitness = 1.0 - eval_stats["eval_loss"]
            gen_fitness.append((fitness, model, train_stats, eval_stats))

        gen_fitness.sort(key=lambda x: x[0], reverse=True)
        topk = max(1, EVO_POP // 3)
        elites = gen_fitness[:topk]
        if best_eval is None or elites[0][0] > best_eval[0]:
            best_eval = elites[0]
        if EVO_PROGRESS:
            log(f"Gen {gen}: best_acc={elites[0][3]['eval_acc']:.4f}, loss={elites[0][3]['eval_loss']:.4f}")
        save_evo_checkpoint(gen, elites[0][1], elites[0][2], elites[0][3], elites[0][0])

        # Refill population
        new_population = [e[1] for e in elites]  # keep elites
        while len(new_population) < EVO_POP:
            parent = random.choice(elites)[1]
            child = AbsoluteHallway(input_dim=input_dim, num_classes=num_classes, ring_len=RING_LEN, slot_dim=SLOT_DIM)
            child.load_state_dict(
                mutate_state_dict(parent.state_dict(), std=EVO_MUT_STD, pointer_only=EVO_POINTER_ONLY)
            )
            new_population.append(child)
        population = new_population

    # final best eval stats already stored in best_eval
    fitness, best_model, train_stats, eval_stats = best_eval
    return {
        "mode": "evolution",
        "best_train": train_stats,
        "best_eval": eval_stats,
        "best_fitness": fitness,
    }


def run_phase(dataset_name: str, loader, eval_loader, input_dim: int, num_classes: int):
    extra = ""
    if SYNTH_META.get("enabled"):
        extra = f" | synth_mode={SYNTH_META.get('mode')}"
    log(f"=== Phase 6.5 | dataset={dataset_name} | num_classes={num_classes}{extra} ===")
    hallway = AbsoluteHallway(input_dim=input_dim, num_classes=num_classes, ring_len=RING_LEN, slot_dim=SLOT_DIM)
    hall_train = train_wallclock(hallway, loader, dataset_name, "absolute_hallway", num_classes)
    hall_eval = eval_model(hallway, eval_loader, dataset_name, "absolute_hallway")
    result = {"dataset": dataset_name, "absolute_hallway": {"train": hall_train, "eval": hall_eval}}
    if SYNTH_META:
        result["meta"] = dict(SYNTH_META)
    return result


def run_lockout_test():
    log("=== Lockout test | deterministic synth A->B (label flip) ===")
    loader_a, loader_b, collate = build_synth_pair_loaders()
    eval_a, _ = build_eval_loader_from_subset(loader_a.dataset, input_collate=collate)
    eval_b, _ = build_eval_loader_from_subset(loader_b.dataset, input_collate=collate)

    model = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=RING_LEN, slot_dim=SLOT_DIM)

    train_a = train_steps(model, loader_a, PHASE_A_STEPS, "synthA", "absolute_hallway")
    eval_a_post = eval_model(model, eval_a, "synthA", "absolute_hallway")
    eval_b_pre = eval_model(model, eval_b, "synthB_pre", "absolute_hallway")

    train_b = train_steps(model, loader_b, PHASE_B_STEPS, "synthB", "absolute_hallway")
    eval_b_post = eval_model(model, eval_b, "synthB_post", "absolute_hallway")
    eval_a_post_b = eval_model(model, eval_a, "synthA_postB", "absolute_hallway")

    return {
        "mode": "lockout",
        "phase_a": {"train": train_a, "eval": eval_a_post},
        "phase_b": {"pre_eval": eval_b_pre, "train": train_b, "eval": eval_b_post},
        "forgetting_check": eval_a_post_b,
        "meta": {
            "phase_a_steps": PHASE_A_STEPS,
            "phase_b_steps": PHASE_B_STEPS,
            "synth_len": SYNTH_LEN,
            "synth_shuffle": SYNTH_SHUFFLE,
        },
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not RESUME:
        rotate_artifacts()
    set_seed(SEED)
    # Reduce kernel search overhead / variance.
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    log(f"Phase 6.5 Absolute Hallway start | device={DEVICE} | offline_only={OFFLINE_ONLY}")

    summary = []

    # Synthetic short-circuit: if TP6_SYNTH=1, bypass MNIST entirely.
    synth_env = os.environ.get("TP6_SYNTH", "0").strip()
    synth_once = os.environ.get("TP6_SYNTH_ONCE") == "1"
    if synth_env == "1":
        synth_loader, synth_classes, synth_collate = get_seq_mnist_loader()
        synth_eval_loader, eval_size = build_eval_loader_from_subset(
            synth_loader.dataset, input_collate=synth_collate
        )
        log_eval_overlap(synth_loader.dataset, synth_eval_loader.dataset, eval_size, "synth_subset")
        # Run once for probes; otherwise run forever until manually stopped.
        if synth_once:
            run_phase("synth", synth_loader, synth_eval_loader, input_dim=1, num_classes=synth_classes)
            return
        while True:
            run_phase("synth", synth_loader, synth_eval_loader, input_dim=1, num_classes=synth_classes)
        return
    elif RUN_MODE == "lockout":
        summary.append(run_lockout_test())
    else:
        mnist_loader, mnist_classes, mnist_collate = get_seq_mnist_loader()
        eval_label = "train_subset"
        if SYNTH_META.get("enabled") or EVAL_SPLIT == "subset":
            mnist_eval_loader, eval_size = build_eval_loader_from_subset(
                mnist_loader.dataset, input_collate=mnist_collate
            )
            eval_label = "train_subset"
        else:
            try:
                import torchvision.transforms as T
                from torchvision.datasets import MNIST
                transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
                eval_ds = MNIST(
                    os.path.join(DATA_DIR, "mnist_seq"),
                    train=False,
                    download=not OFFLINE_ONLY,
                    transform=transform,
                )
                mnist_eval_loader, eval_size = build_eval_loader_from_dataset(
                    eval_ds, input_collate=mnist_collate
                )
                eval_label = "mnist_test"
            except Exception as exc:
                log(f"[eval] test split unavailable ({exc}); falling back to train subset")
                mnist_eval_loader, eval_size = build_eval_loader_from_subset(
                    mnist_loader.dataset, input_collate=mnist_collate
                )
                eval_label = "train_subset_fallback"
        log_eval_overlap(mnist_loader.dataset, mnist_eval_loader.dataset, eval_size, eval_label)
        if RUN_MODE == "evolution":
            summary.append(
                run_evolution("seq_mnist", mnist_loader, mnist_eval_loader, input_dim=1, num_classes=mnist_classes)
            )
        else:
            summary.append(run_phase("seq_mnist", mnist_loader, mnist_eval_loader, input_dim=1, num_classes=mnist_classes))

    # Ensure GPU work is complete before writing summary to avoid partial/hung writes.
    # For synthetic indefinite run we skip summary writing and syncing to keep process alive.
    if torch.cuda.is_available() and not DISABLE_SYNC:
        torch.cuda.synchronize()
    if summary:
        summary_path = os.path.join(ROOT, "summaries", "current", "tournament_phase6_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            log(f"Tournament done. Summary saved to {summary_path}")
        except Exception as e:
            log(f"Summary write skipped: {e}")
        try:
            sync_current_to_last()
        except Exception as e:
            log(f"Summary sync skipped: {e}")


if __name__ == "__main__":
    main()
