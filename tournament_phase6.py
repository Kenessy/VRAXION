import os
import time
import math
import json
import hashlib
import random
from itertools import count
import shutil
import urllib.request
import zipfile
from typing import List, Optional, Tuple

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

from prime_c19.tp6.controls import (
    ThermostatParams,
    AGCParams,
    InertiaAutoParams,
    PanicReflex,
    CadenceGovernor,
    apply_thermostat,
    apply_update_agc,
    apply_inertia_auto,
)
from prime_c19.tp6.sharding import calculate_adaptive_vasc
from prime_c19.tp6.experts import LocationExpertRouter

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
PTR_ANCHOR_CONF_MIN = float(os.environ.get("TP6_ANCHOR_CONF_MIN", "0.9"))
PTR_ANCHOR_MIN_STEP = float(os.environ.get("TP6_ANCHOR_MIN_STEP", "0"))
PTR_ANCHOR_CLICK_INJECT = bool(int(os.environ.get("TP6_ANCHOR_CLICK_INJECT", "0")))
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
EVAL_EVERY_STEPS = int(os.environ.get("TP6_EVAL_EVERY_STEPS", "0"))
EVAL_AT_CHECKPOINT = os.environ.get("TP6_EVAL_AT_CHECKPOINT", "1") == "1"
VCOG_ID_TARGET = float(os.environ.get("TP6_VCOG_ID_TARGET", "0.25"))
VCOG_SIGMA_FLOOR = float(os.environ.get("TP6_VCOG_SIGMA_FLOOR", "1e-4"))
VCOG_PRG_FROM_ACC = os.environ.get("TP6_VCOG_PRG_FROM_ACC", "0") == "1"
VCOG_PRG_ACC_TARGET = float(os.environ.get("TP6_VCOG_PRG_ACC_TARGET", "1.0"))
LOSS_EMA_BETA = float(os.environ.get("TP6_LOSS_EMA_BETA", "0.95"))
RATCHET_ENABLED = os.environ.get("TP6_INERTIA_RATCHET", "0") == "1"
RATCHET_ACC_MIN = float(os.environ.get("TP6_RATCHET_ACC_MIN", "0.98"))
RATCHET_STREAK = int(os.environ.get("TP6_RATCHET_STREAK", "2"))
RATCHET_BASE = float(os.environ.get("TP6_RATCHET_BASE", "0.5"))
RATCHET_SCALE = float(os.environ.get("TP6_RATCHET_SCALE", "0.98"))
INERTIA_SIGNAL_ENABLED = os.environ.get("TP6_INERTIA_SIGNAL", "0") == "1"
INERTIA_SIGNAL_ACC_MIN = float(os.environ.get("TP6_INERTIA_SIGNAL_ACC_MIN", "0.98"))
INERTIA_SIGNAL_STREAK = int(os.environ.get("TP6_INERTIA_SIGNAL_STREAK", "2"))
INERTIA_SIGNAL_TARGET = float(os.environ.get("TP6_INERTIA_SIGNAL_TARGET", "0.95"))
INERTIA_SIGNAL_FLOOR = float(os.environ.get("TP6_INERTIA_SIGNAL_FLOOR", "0.5"))
INERTIA_SIGNAL_REWARD = float(os.environ.get("TP6_INERTIA_SIGNAL_REWARD", "0.1"))
INERTIA_SIGNAL_PAIN = float(os.environ.get("TP6_INERTIA_SIGNAL_PAIN", "0.05"))
MITOSIS_ENABLED = os.environ.get("TP6_MITOSIS", "0") == "1"
MITOSIS_STALL_WINDOW = int(os.environ.get("TP6_MITOSIS_STALL_WINDOW", "100"))
MITOSIS_STALL_DELTA = float(os.environ.get("TP6_MITOSIS_STALL_DELTA", "0.005"))
MITOSIS_IMBALANCE = float(os.environ.get("TP6_MITOSIS_IMBALANCE", "0.5"))
MITOSIS_EXIT_CODE = int(os.environ.get("TP6_MITOSIS_EXIT_CODE", "86"))
MITOSIS_CKPT_PATH = os.environ.get("TP6_MITOSIS_CKPT", "")
METABOLIC_TELEMETRY = os.environ.get("TP6_METABOLIC_TELEMETRY", "0") == "1"
METABOLIC_HUNGER = os.environ.get("TP6_METABOLIC_HUNGER", "0") == "1"
METABOLIC_COST_COEFF = float(os.environ.get("TP6_METABOLIC_COST_COEFF", "0.0001"))
DOMAIN_SEP_COEFF = float(os.environ.get("TP6_DOMAIN_SEP_COEFF", "0.0"))
EXPERT_BUDGET = int(os.environ.get("TP6_EXPERT_BUDGET", "0"))
USAGE_LAMBDA_INIT = float(os.environ.get("TP6_USAGE_LAMBDA", "0.0"))
USAGE_LAMBDA_MAX = float(os.environ.get("TP6_USAGE_LAMBDA_MAX", "0.5"))
USAGE_LAMBDA_ETA = float(os.environ.get("TP6_USAGE_LAMBDA_ETA", "0.05"))
USAGE_GATE_CONF = float(os.environ.get("TP6_USAGE_GATE_CONF", "0.95"))
USAGE_GATE_EVAL = os.environ.get("TP6_USAGE_GATE_EVAL", "0") == "1"
USAGE_EMA_BETA = float(os.environ.get("TP6_USAGE_EMA_BETA", "0.9"))
USAGE_REMAP_ENABLED = os.environ.get("TP6_USAGE_REMAP", "0") == "1"
USAGE_REMAP_EVERY = int(os.environ.get("TP6_USAGE_REMAP_EVERY", "50"))
USAGE_REMAP_MODE = os.environ.get("TP6_USAGE_REMAP_MODE", "round_robin").strip().lower()
METABOLIC_SIM_THRESH = float(os.environ.get("TP6_METABOLIC_SIM_THRESH", "0.98"))
METABOLIC_EVERY = int(os.environ.get("TP6_METABOLIC_EVERY", "500"))
METABOLIC_IDLE_STEPS = int(os.environ.get("TP6_METABOLIC_IDLE_STEPS", "2000"))
HIBERNATE_ENABLED = os.environ.get("TP6_HIBERNATE", "0") == "1"
HIBERNATE_MODE = os.environ.get("TP6_HIBERNATE_MODE", "shadow").strip().lower()
HIBERNATE_IDLE_STEPS = int(os.environ.get("TP6_HIBERNATE_IDLE_STEPS", str(METABOLIC_IDLE_STEPS)))
HIBERNATE_EVERY = int(os.environ.get("TP6_HIBERNATE_EVERY", "50"))
HIBERNATE_DIR = os.environ.get("TP6_HIBERNATE_DIR", "hibernation")
MODULAR_SAVE = os.environ.get("TP6_MODULAR_SAVE", "0") == "1"
MODULAR_SAVE_MODE = os.environ.get("TP6_MODULAR_SAVE_MODE", "only").strip().lower()
MODULAR_DIR = os.environ.get("TP6_MODULAR_DIR", "")
MODULAR_RESUME = os.environ.get("TP6_MODULAR_RESUME", "0") == "1"
MODULAR_AUTO_EXPERTS = os.environ.get("TP6_MODULAR_AUTO_EXPERTS", "0") == "1"
TENURE_CONTRIB_THRESH = float(os.environ.get("TP6_TENURE_CONTRIB", "1000.0"))
TENURE_PROBATION_STEPS = int(os.environ.get("TP6_TENURE_PROBATION_STEPS", "1000"))
TENURE_TTL_STEPS = int(os.environ.get("TP6_TENURE_TTL_STEPS", "0"))
TENURE_GC = os.environ.get("TP6_TENURE_GC", "0") == "1"
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


class VCogGovernor:
    def __init__(self, id_target=0.25, beta=0.95, sigma_floor=1e-4):
        self.id_target = id_target
        self.beta = beta
        self.sigma_floor = sigma_floor
        self.search_ema = None
        self.search_var = 0.0
        self.loss_ema = None
        self.loss_var = 0.0

    def update(self, telemetry):
        search = float(telemetry.get("search", 0.0))
        loss = float(telemetry.get("loss", 0.0))
        eval_acc = telemetry.get("eval_acc")
        if self.search_ema is None:
            self.search_ema = search
        else:
            self.search_ema = self.beta * self.search_ema + (1.0 - self.beta) * search
        self.search_var = self.beta * self.search_var + (1.0 - self.beta) * (search - self.search_ema) ** 2
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = self.beta * self.loss_ema + (1.0 - self.beta) * loss
        self.loss_var = self.beta * self.loss_var + (1.0 - self.beta) * (loss - self.loss_ema) ** 2

        inertia = float(telemetry.get("inertia", 0.0))
        epi = float(telemetry.get("epi", 0.0))
        walk = float(telemetry.get("walk", 0.0))
        focus = float(telemetry.get("focus", 0.0))
        grip = max(0.0, min(1.0, inertia * (1.0 - walk) * focus))

        lk = grip * (1.0 - search)
        delta = float(telemetry.get("delta", 0.0))
        delta_raw = float(telemetry.get("delta_raw", 0.0))
        mob = max(0.0, min(1.0, delta / (delta_raw + 1e-9)))
        fl = grip * mob

        search_std = max(math.sqrt(abs(self.search_var)), self.sigma_floor)
        sz = (search - self.search_ema) / (search_std + self.sigma_floor)
        sn = max(0.0, sz) * grip

        loss_std = max(math.sqrt(abs(self.loss_var)), self.sigma_floor)
        ident = 1.0 / (1.0 + self.loss_ema * (1.0 + loss_std))
        prg = None
        if eval_acc is not None:
            try:
                acc_val = float(eval_acc)
                prg = max(0.0, min(100.0, acc_val * 100.0))
            except Exception:
                prg = None
        if prg is None:
            prg = max(0.0, min(1.0, ident / max(self.id_target, 1e-6))) * 100.0
        orb = telemetry.get("orb")
        rd = telemetry.get("rd")
        ac = telemetry.get("ac")
        try:
            orb = int(orb) if orb is not None else 0
        except Exception:
            orb = 0
        try:
            rd = float(rd) if rd is not None else 0.0
        except Exception:
            rd = 0.0
        try:
            ac = int(ac) if ac is not None else 0
        except Exception:
            ac = 0

        header = (
            f"V_COG[PRGRS:{prg:.1f}% ORB:{orb} RD:{rd:.2e} AC:{ac} EPI:{epi:.2f} "
            f"LOCKS:{lk:.2f} FLOWS:{fl:.2f} "
            f"SNAPS:{sn:.2f} IDENT:{ident:.3f}]"
        )
        return header


def _compute_expert_similarity_stats(model, sim_thresh):
    head = getattr(model, "head", None)
    experts = getattr(head, "experts", None) if head is not None else None
    if not experts:
        return None
    with torch.no_grad():
        flat = []
        for expert in experts:
            vec = expert.weight.detach().float().reshape(-1)
            if expert.bias is not None:
                vec = torch.cat([vec, expert.bias.detach().float().reshape(-1)])
            flat.append(vec)
        mat = torch.stack(flat, dim=0)
        mat = F.normalize(mat, dim=1)
        sim = mat @ mat.T
        upper = torch.triu(sim, diagonal=1)
        if upper.numel() == 0:
            return None
        max_sim = float(upper.max().item())
        count = int((upper > sim_thresh).sum().item())
        flat_idx = int(torch.argmax(upper).item())
        i = flat_idx // upper.size(1)
        j = flat_idx % upper.size(1)
    return max_sim, count, (int(i), int(j))


def _resolve_hibernate_dir(base_dir, root_dir):
    path = base_dir
    if not os.path.isabs(path):
        path = os.path.join(root_dir, base_dir)
    os.makedirs(path, exist_ok=True)
    return path


def _extract_expert_state(head, expert_id):
    if head is None:
        return None
    experts = getattr(head, "experts", None)
    if experts is None or expert_id >= len(experts):
        return None
    state = {}
    with torch.no_grad():
        exp_state = experts[expert_id].state_dict()
        for name, tensor in exp_state.items():
            state[name] = tensor.detach().cpu()
    return state


def _zero_expert_weights(head, expert_id):
    if head is None:
        return False
    experts = getattr(head, "experts", None)
    if experts is None or expert_id >= len(experts):
        return False
    with torch.no_grad():
        for param in experts[expert_id].parameters():
            param.zero_()
    return True


def _hash_state_dict(state):
    if not state:
        return None
    hasher = hashlib.sha256()
    for key in sorted(state.keys()):
        tensor = state[key]
        hasher.update(key.encode("utf-8"))
        if torch.is_tensor(tensor):
            arr = tensor.contiguous().cpu().numpy()
            hasher.update(arr.tobytes())
        else:
            hasher.update(repr(tensor).encode("utf-8"))
    return hasher.hexdigest()


def _save_expert_snapshot(state, path):
    torch.save(state, path)
    return _hash_state_dict(state)


def _load_expert_snapshot(path):
    if not os.path.exists(path):
        return None, None
    state = torch.load(path, map_location="cpu")
    return state, _hash_state_dict(state)


def _resolve_modular_dir(base_dir, root_dir, fallback_path):
    path = base_dir or ""
    if not path:
        path = fallback_path
    if not path:
        path = os.path.join(root_dir, "checkpoints", "modular")
    if path.endswith(".pt"):
        path = os.path.splitext(path)[0] + "_modular"
    if not os.path.isabs(path):
        path = os.path.join(root_dir, path)
    os.makedirs(path, exist_ok=True)
    return path


def _modular_paths(base_dir):
    system_dir = os.path.join(base_dir, "system")
    experts_dir = os.path.join(base_dir, "experts")
    os.makedirs(system_dir, exist_ok=True)
    os.makedirs(experts_dir, exist_ok=True)
    router_path = os.path.join(system_dir, "router.state")
    meta_path = os.path.join(experts_dir, "meta.json")
    return router_path, experts_dir, meta_path


def _split_model_state_dict(state_dict):
    core = {}
    experts = {}
    prefix = "head.experts."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            rest = key[len(prefix):]
            parts = rest.split(".", 1)
            if len(parts) != 2:
                core[key] = value
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                core[key] = value
                continue
            expert_state = experts.setdefault(idx, {})
            expert_state[parts[1]] = value.detach().cpu()
        else:
            core[key] = value.detach().cpu()
    return core, experts


def _ensure_expert_tracking(model, num_experts, step):
    contrib = getattr(model, "expert_contrib", None)
    last_used = getattr(model, "expert_last_used", None)
    created = getattr(model, "expert_created_step", None)
    tenured = getattr(model, "expert_tenured", None)
    if contrib is None or last_used is None or created is None or tenured is None:
        model.expert_contrib = [0.0 for _ in range(num_experts)]
        model.expert_last_used = [int(step) for _ in range(num_experts)]
        model.expert_created_step = [int(step) for _ in range(num_experts)]
        model.expert_tenured = [False for _ in range(num_experts)]
        return
    if len(contrib) < num_experts:
        missing = num_experts - len(contrib)
        model.expert_contrib.extend([0.0 for _ in range(missing)])
        model.expert_last_used.extend([int(step) for _ in range(missing)])
        model.expert_created_step.extend([int(step) for _ in range(missing)])
        model.expert_tenured.extend([False for _ in range(missing)])


def _update_expert_usage(model, num_experts, step):
    counts = getattr(model, "ptr_expert_counts", None)
    if counts is None or counts.numel() != num_experts:
        return
    _ensure_expert_tracking(model, num_experts, step)
    counts_list = counts.detach().cpu().tolist()
    for idx, count in enumerate(counts_list):
        if count <= 0:
            continue
        model.expert_last_used[idx] = int(step)
        model.expert_contrib[idx] += float(count)


def _build_expert_meta(model, step, num_experts, contrib_thresh, probation_steps):
    _ensure_expert_tracking(model, num_experts, step)
    meta = []
    for idx in range(num_experts):
        contrib = float(model.expert_contrib[idx])
        created = int(model.expert_created_step[idx])
        last_used = int(model.expert_last_used[idx])
        age = max(0, int(step) - created)
        probation = age < probation_steps
        tenured = contrib >= contrib_thresh
        model.expert_tenured[idx] = bool(tenured)
        meta.append(
            {
                "id": idx,
                "contrib": contrib,
                "created_step": created,
                "last_used_step": last_used,
                "age_steps": age,
                "probation": bool(probation),
                "tenured": bool(tenured),
            }
        )
    return meta


def _load_modular_meta(model, meta_path):
    if not os.path.exists(meta_path):
        return
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    experts = data.get("experts", [])
    if not experts:
        return
    num_experts = len(experts)
    model.expert_contrib = [float(item.get("contrib", 0.0)) for item in experts]
    model.expert_last_used = [int(item.get("last_used_step", 0)) for item in experts]
    model.expert_created_step = [int(item.get("created_step", 0)) for item in experts]
    model.expert_tenured = [bool(item.get("tenured", False)) for item in experts]


def _save_modular_checkpoint(model, optimizer, scaler, step, losses, base_dir, contrib_thresh, probation_steps,
                             ttl_steps=0, gc_enabled=False):
    router_path, experts_dir, meta_path = _modular_paths(base_dir)
    state_dict = model.state_dict()
    core_state, expert_states = _split_model_state_dict(state_dict)
    payload = {
        "model": core_state,
        "optim": optimizer.state_dict(),
        "scaler": scaler.state_dict() if USE_AMP else None,
        "step": int(step),
        "losses": list(losses),
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
    torch.save(payload, router_path)
    num_experts = int(payload["num_experts"])
    expert_meta = _build_expert_meta(model, step, num_experts, contrib_thresh, probation_steps)
    deleted = []
    for idx, state in expert_states.items():
        if idx >= num_experts:
            continue
        if gc_enabled and ttl_steps > 0:
            tenured = model.expert_tenured[idx]
            idle = step - model.expert_last_used[idx]
            age = step - model.expert_created_step[idx]
            probation = age < probation_steps
            if not tenured and not probation and idle >= ttl_steps:
                deleted.append(idx)
                continue
        path = os.path.join(experts_dir, f"expert_{idx:03d}.pt")
        torch.save(state, path)
    meta = {
        "num_experts": num_experts,
        "step": int(step),
        "experts": expert_meta,
        "deleted": deleted,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_modular_checkpoint(model, optimizer, scaler, base_dir):
    router_path, experts_dir, meta_path = _modular_paths(base_dir)
    ckpt = torch.load(router_path, map_location=DEVICE)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        log(f"Modular load strict failed: {exc}; retrying strict=False")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            log(f"Modular load missing keys: {missing}")
        if unexpected:
            log(f"Modular load unexpected keys: {unexpected}")
    head = getattr(model, "head", None)
    if head is not None and getattr(head, "experts", None):
        for idx, expert in enumerate(head.experts):
            path = os.path.join(experts_dir, f"expert_{idx:03d}.pt")
            if not os.path.exists(path):
                continue
            state = torch.load(path, map_location="cpu")
            expert.load_state_dict(state, strict=False)
    if "optim" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optim"])
    if ckpt.get("scaler") and USE_AMP and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])
    _load_modular_meta(model, meta_path)
    return ckpt


def _resolve_modular_resume_dir(resume_path):
    if resume_path and os.path.isdir(resume_path):
        router_path = os.path.join(resume_path, "system", "router.state")
        if os.path.exists(router_path):
            return resume_path
    if resume_path and resume_path.endswith(".pt"):
        candidate = os.path.splitext(resume_path)[0] + "_modular"
        router_path = os.path.join(candidate, "system", "router.state")
        if os.path.exists(router_path):
            return candidate
    return None
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
STAIRCASE_LENS_RAW = os.environ.get("TP6_STAIRCASE_LENS", "").strip()
STAIRCASE_WEIGHTS_RAW = os.environ.get("TP6_STAIRCASE_WEIGHTS", "").strip()
STAIRCASE_ENABLED = bool(STAIRCASE_LENS_RAW)
STAIRCASE_ADAPT = os.environ.get("TP6_STAIRCASE_ADAPT", "0") == "1"
STAIRCASE_ADAPT_EVERY = int(os.environ.get("TP6_STAIRCASE_ADAPT_EVERY", "2000"))
STAIRCASE_MIN_BASE = float(os.environ.get("TP6_STAIRCASE_MIN_BASE", "0.60"))
STAIRCASE_SHIFT = float(os.environ.get("TP6_STAIRCASE_SHIFT", "0.02"))
STAIRCASE_STABLE_STD = float(os.environ.get("TP6_STAIRCASE_STABLE_STD", "0.02"))
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

# Control-loop parameter bundles (extracted for testability).
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

INERTIA_AUTO_PARAMS = InertiaAutoParams(
    enabled=INERTIA_AUTO,
    inertia_min=INERTIA_MIN,
    inertia_max=INERTIA_MAX,
    vel_full=INERTIA_VEL_FULL,
    ema_beta=INERTIA_EMA,
    dwell_enabled=DWELL_INERTIA_ENABLED,
    dwell_thresh=DWELL_INERTIA_THRESH,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _maybe_override_expert_heads(resume_path: str) -> None:
    if not resume_path:
        return
    modular_dir = _resolve_modular_resume_dir(resume_path)
    if not modular_dir or not MODULAR_AUTO_EXPERTS:
        return
    router_path = os.path.join(modular_dir, "system", "router.state")
    if not os.path.exists(router_path):
        return
    try:
        ckpt = torch.load(router_path, map_location="cpu")
    except Exception as exc:
        log(f"Modular auto-expert read failed: {exc}")
        return
    num_experts = ckpt.get("num_experts")
    if num_experts:
        global EXPERT_HEADS
        EXPERT_HEADS = int(num_experts)
        log(f"[modular] auto override EXPERT_HEADS={EXPERT_HEADS} from {router_path}")


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


def _parse_csv_ints(raw: str) -> Optional[List[int]]:
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
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    items: List[float] = []
    for part in parts:
        try:
            value = float(part)
        except ValueError:
            return None
        items.append(value)
    return items or None


def _normalize_weights(weights: List[float]) -> List[float]:
    total = sum(max(0.0, float(w)) for w in weights)
    if total <= 0.0:
        return [1.0 / len(weights)] * len(weights)
    return [max(0.0, float(w)) / total for w in weights]


def _default_staircase_weights(lens: List[int]) -> List[float]:
    if len(lens) == 3 and lens[0] == 512 and lens[1] == 768 and lens[2] == 1024:
        return [0.95, 0.04, 0.01]
    return [1.0 / len(lens)] * len(lens)


class StaircaseController:
    def __init__(
        self,
        lens: List[int],
        weights: List[float],
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

    def set_weights(self, weights: List[float]) -> None:
        self.weights = _normalize_weights(weights)

    def maybe_adapt(self, losses: List[float], step: int) -> Optional[List[float]]:
        if self.adapt_every <= 0 or step <= 0 or step % self.adapt_every != 0:
            return None
        if not losses:
            return None
        window = min(len(losses), max(10, AGC_PLATEAU_WINDOW or 50))
        recent = losses[-window:]
        mean_recent = sum(recent) / len(recent)
        var_recent = sum((x - mean_recent) ** 2 for x in recent) / len(recent)
        std_recent = math.sqrt(var_recent)
        if not math.isfinite(std_recent) or std_recent > self.stable_std:
            return None
        if len(self.weights) < 2:
            return None
        base_weight = self.weights[0]
        if base_weight <= self.min_base:
            return None
        delta = min(self.shift, base_weight - self.min_base)
        if delta <= 0.0:
            return None
        per_other = delta / (len(self.weights) - 1)
        new_weights = [base_weight - delta]
        for idx in range(1, len(self.weights)):
            new_weights.append(self.weights[idx] + per_other)
        new_weights = _normalize_weights(new_weights)
        self.weights = new_weights
        log(f"[staircase] shift base by {delta:.3f} -> weights={new_weights}")
        return new_weights


class StaircaseBatcher:
    def __init__(
        self,
        loaders: List[torch.utils.data.DataLoader],
        weights: List[float],
        rng_seed: int,
        staircase: Optional[StaircaseController] = None,
    ) -> None:
        self.loaders = list(loaders)
        self.weights = _normalize_weights(weights)
        self._iters = [iter(loader) for loader in self.loaders]
        self._rng = random.Random(rng_seed)
        self.dataset = self.loaders[0].dataset if self.loaders else None
        self.staircase = staircase

    def set_weights(self, weights: List[float]) -> None:
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
    if not DEBUG_NAN:
        return
    if not tensor.is_floating_point():
        return
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        log(f"[nan_guard] step={step:04d} tensor={name} has NaN/Inf")
        raise RuntimeError(f"NaN/Inf in {name} at step {step}")


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
        "ptr_inertia_floor": getattr(model, "ptr_inertia_floor", 0.0),
        "agc_scale_max": getattr(model, "agc_scale_max", AGC_SCALE_MAX),
        "ground_speed_ema": getattr(model, "ground_speed_ema", None),
        "ground_speed_limit": getattr(model, "ground_speed_limit", None),
        "num_experts": getattr(getattr(model, "head", None), "num_experts", EXPERT_HEADS),
        "param_names": [name for name, _ in model.named_parameters()],
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
        # Inertia floor for post-snap stability (ratchet mode).
        self.ptr_inertia_floor = 0.0
        self.ptr_inertia_dyn_pre = None
        self.ptr_inertia_dyn = None
        self.ptr_inertia_dyn_tensor = None
        self.ptr_inertia_reward_ready = False
        self.ptr_inertia_reward_acc = None
        self.ptr_inertia_reward_streak = 0
        self.ptr_inertia_epi = 0.0
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
        self.usage_soft_enabled = os.environ.get("TP6_USAGE_SOFT", "0") == "1"
        self.usage_soft_temp = float(os.environ.get("TP6_USAGE_SOFT_TEMP", "0.5"))
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
        sensory_flag = os.environ.get("TP6_SENSORY_RING", "1").strip().lower()
        self.sensory_enabled = sensory_flag not in {"0", "false", "off", "no"}
        sensory_len_env = os.environ.get("TP6_SENSORY_RING_LEN")
        sensory_dim_env = os.environ.get("TP6_SENSORY_SLOT_DIM")
        default_sensory_len = max(1, self.ring_len // 2)
        default_sensory_dim = max(8, self.slot_dim // 3)
        self.sensory_len = max(1, int(sensory_len_env)) if sensory_len_env else default_sensory_len
        self.sensory_dim = max(8, int(sensory_dim_env)) if sensory_dim_env else default_sensory_dim
        if self.sensory_enabled:
            # Small sensory ring: converts raw input into a richer sequence for the core ring.
            self.sensory_proj_in = nn.Linear(input_dim, self.sensory_dim)
            self.sensory_gru = nn.GRUCell(self.sensory_dim, self.sensory_dim)
            self.sensory_bridge = nn.Linear(self.sensory_dim, slot_dim)
        else:
            self.sensory_proj_in = None
            self.sensory_gru = None
            self.sensory_bridge = None
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
        # Router map: decouple address -> expert from modulo routing.
        map_len = max(1, int(self.ring_range))
        router_init = torch.arange(map_len, dtype=torch.long) % self.head.num_experts
        self.register_buffer("router_map", router_init)
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
        if self.sensory_enabled:
            nn.init.xavier_uniform_(self.sensory_proj_in.weight)
            nn.init.zeros_(self.sensory_proj_in.bias)
            nn.init.normal_(self.sensory_gru.weight_ih, mean=0.0, std=0.05)
            nn.init.normal_(self.sensory_gru.weight_hh, mean=0.0, std=0.05)
            nn.init.zeros_(self.sensory_gru.bias_ih)
            nn.init.zeros_(self.sensory_gru.bias_hh)
            nn.init.xavier_uniform_(self.sensory_bridge.weight)
            nn.init.zeros_(self.sensory_bridge.bias)
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

    def _map_expert_ids(self, ptr_int: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "router_map") or self.router_map.numel() == 0:
            return ptr_int
        router_map = self.router_map
        if router_map.device != ptr_int.device:
            router_map = router_map.to(ptr_int.device)
        idx = ptr_int.clamp(0, router_map.numel() - 1)
        expert_ids = router_map[idx].to(torch.long)
        # Guard against stale maps if expert count changed.
        if expert_ids.numel() and expert_ids.max() >= self.head.num_experts:
            expert_ids = expert_ids % self.head.num_experts
        return expert_ids

    def _update_expert_stats(self, expert_ids: torch.Tensor | None) -> None:
        self.ptr_expert_ids = None
        self.ptr_expert_counts = None
        self.ptr_expert_active = 0
        self.ptr_expert_max_share = None
        self.ptr_expert_entropy = None
        if expert_ids is None or self.head.num_experts <= 1:
            return
        counts = torch.bincount(expert_ids, minlength=self.head.num_experts).float()
        total = counts.sum().clamp(min=1.0)
        probs = counts / total
        max_share = (counts.max() / total).item()
        active = int((counts > 0).sum().item())
        entropy = 0.0
        if self.head.num_experts > 1:
            entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item() / math.log(self.head.num_experts))
        self.ptr_expert_ids = expert_ids.detach().cpu()
        self.ptr_expert_counts = counts.detach().cpu()
        self.ptr_expert_active = active
        self.ptr_expert_max_share = float(max_share)
        self.ptr_expert_entropy = float(entropy)

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
        sensory_seq = None
        if self.sensory_enabled:
            s_state = torch.zeros(B, self.sensory_len, self.sensory_dim, device=device, dtype=x.dtype)
            s_h = torch.zeros(B, self.sensory_dim, device=device, dtype=x.dtype)
            s_ptr = torch.zeros(B, device=device, dtype=torch.long)
            s_out = []
            for t in range(T):
                s_inp = self.sensory_proj_in(x[:, t, :])
                s_inp = self._apply_activation(s_inp)
                s_ctx = s_state[torch.arange(B, device=device), s_ptr]
                s_h = self.sensory_gru(s_inp + s_ctx, s_h)
                s_state[torch.arange(B, device=device), s_ptr] = s_h
                s_out.append(s_h)
                s_ptr = (s_ptr + 1) % self.sensory_len
            sensory_seq = torch.stack(s_out, dim=1)
        # Diagnostic bypass: skip ring read/write and use GRU output directly.
        if self.bypass_ring:
            h = torch.zeros(B, self.slot_dim, device=device, dtype=x.dtype)
            movement_cost = torch.tensor(0.0, device=device, dtype=x.dtype)
            pointer_addresses = torch.zeros(B, device=device, dtype=torch.long)
            for t in range(T):
                if self.sensory_enabled:
                    inp = self.sensory_bridge(sensory_seq[:, t, :])
                else:
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
        ptr_int = ptr_int_init
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
        # Dual-pointer (anchor + residual) state for Phase A stereoscopic foundation.
        loss_ema = getattr(self, "loss_ema", None)
        confidence = 0.0
        if loss_ema is not None:
            confidence = 1.0 / (1.0 + float(loss_ema))
        kernel_width = max(self.gauss_tau, 1e-6)
        min_step_floor = max(
            ring_range * torch.finfo(torch.float64).eps,
            kernel_width * 1e-3,
        )
        min_step = max(min_step_floor, float(self.ptr_stride))
        if PTR_ANCHOR_MIN_STEP > 0:
            # Allow a smaller actuation lattice for Phase A click testing.
            min_step = max(min_step_floor, PTR_ANCHOR_MIN_STEP)
        self.ptr_min_step = float(min_step)
        ptr_anchor = ptr_float.to(torch.float64)
        ptr_anchor = torch.remainder(torch.round(ptr_anchor / min_step) * min_step, ring_range)
        ptr_residual = ptr_float.to(torch.float64) - ptr_anchor
        ptr_float = (ptr_anchor + ptr_residual).to(ptr_dtype)
        res_mean_init = float(ptr_residual.abs().mean().item())
        self.ptr_residual_mean = res_mean_init
        self.ptr_orbit = 2 if res_mean_init >= (min_step * 0.1) else 1
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
            anchor_clicks = 0
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

            if self.sensory_enabled:
                inp = self.sensory_bridge(sensory_seq[:, t, :])
            else:
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
            self.ptr_update_allowed = bool(update_allowed)
            self.ptr_update_blocked = bool(
                self.ptr_lock
                or self.time_pointer
                or not update_allowed
                or (self.ptr_warmup_steps > 0 and t < self.ptr_warmup_steps)
            )
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
                ctrl_inertia_tensor_pre = inertia_use.mean()
                ctrl_inertia_pre = float(ctrl_inertia_tensor_pre.item())
                inertia_floor = float(getattr(self, "ptr_inertia_floor", 0.0) or 0.0)
                if inertia_floor > 0.0:
                    inertia_floor = min(inertia_floor, 0.99)
                    inertia_use = torch.clamp(inertia_use, min=inertia_floor)
                ctrl_inertia_tensor = inertia_use.mean()
                ctrl_inertia_mean = float(ctrl_inertia_tensor.item())
                ctrl_deadzone_mean = float(deadzone_use.mean().item())
                ctrl_walk_mean = float(walk_use.mean().item())
                self.ptr_inertia_dyn_pre = ctrl_inertia_pre
                self.ptr_inertia_dyn = ctrl_inertia_mean
                self.ptr_inertia_dyn_tensor = ctrl_inertia_tensor

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
            # Dual-pointer annealing: accumulate residuals and click anchor on min_step lattice.
            delta_step = self.wrap_delta(prev_ptr, ptr_float, ring_range)
            delta_step = delta_step * active_mask.to(delta_step.dtype)
            ptr_residual = ptr_residual + delta_step.to(ptr_residual.dtype)
            if PTR_ANCHOR_CLICK_INJECT and t == 0:
                ptr_residual = ptr_residual + min_step
            if confidence >= PTR_ANCHOR_CONF_MIN:
                step_units = torch.floor(ptr_residual.abs() / min_step) * torch.sign(ptr_residual)
                step_units = torch.clamp(step_units, -1.0, 1.0)
                click_mask = step_units != 0
                if click_mask.any():
                    anchor_clicks = int(click_mask.sum().item())
                    ptr_anchor = torch.remainder(ptr_anchor + step_units * min_step, ring_range)
                    ptr_residual = ptr_residual - step_units * min_step
            ptr_float = torch.remainder(ptr_anchor + ptr_residual, ring_range)
            ptr_residual = self.wrap_delta(ptr_anchor, ptr_float, ring_range).to(ptr_residual.dtype)
            ptr_float = ptr_float.to(ptr_dtype)
            self.ptr_anchor_clicks = anchor_clicks
        if self.ptr_lock or self.time_pointer or not update_allowed or (self.ptr_warmup_steps > 0 and t < self.ptr_warmup_steps):
            ptr_anchor = torch.remainder(torch.round(ptr_float.to(torch.float64) / min_step) * min_step, ring_range)
            ptr_residual = self.wrap_delta(ptr_anchor, ptr_float.to(torch.float64), ring_range)
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
        res_mean = float(ptr_residual.abs().mean().item())
        self.ptr_residual_mean = res_mean
        self.ptr_orbit = 2 if res_mean >= (min_step * 0.1) else 1
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
        expert_ids = self._map_expert_ids(ptr_int)
        logits = self.head(read_vec, expert_ids)
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
        expert_ids = self._map_expert_ids(ptr_int)
        logits = self.head(read_vec, expert_ids)
        nan_guard("logits_final", logits, T)

        self.pointer_hist = hist.detach().cpu()
        self.satiety_exits = int(satiety_exited.sum().item())
        # Expose last pointer bins for MI/TEI evaluation.
        last_bins = torch.bucketize(ptr_int.float(), self.bin_edges.to(device)) - 1
        last_bins = last_bins.clamp(0, self.pointer_hist_bins - 1).detach().cpu()
        self.last_ptr_bins = last_bins
        self.last_ptr_int = ptr_int.detach().cpu()
        self._update_expert_stats(expert_ids)
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
        staircase_lens = _parse_csv_ints(STAIRCASE_LENS_RAW) if STAIRCASE_ENABLED else None
        if STAIRCASE_ENABLED and not staircase_lens:
            log(f"[staircase] invalid lens '{STAIRCASE_LENS_RAW}', disabling")
        staircase_weights = None
        if staircase_lens:
            if STAIRCASE_WEIGHTS_RAW:
                staircase_weights = _parse_csv_floats(STAIRCASE_WEIGHTS_RAW)
                if not staircase_weights or len(staircase_weights) != len(staircase_lens):
                    log(f"[staircase] invalid weights '{STAIRCASE_WEIGHTS_RAW}', using defaults")
                    staircase_weights = None
            if staircase_weights is None:
                staircase_weights = _default_staircase_weights(staircase_lens)

        def _wrap_dataset(x_src: torch.Tensor, y_src: torch.Tensor):
            class _SynthDataset(torch.utils.data.Dataset):
                def __len__(self):
                    return x_src.size(0)

                def __getitem__(self, item):
                    return x_src[item], y_src[item]

            def collate(batch):
                xs, ys = zip(*batch)
                return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

            return _SynthDataset(), collate
        if synth_mode == "assoc_clean":
            pairs = max(1, int(ASSOC_PAIRS))
            keys = max(2, int(ASSOC_KEYS))
            min_len = pairs * 2 + 1
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

            def _make_assoc_clean(seq_len_local: int):
                seq_len = seq_len_local
                bump_attempts = 0
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
                return x, y, seq_len

            if staircase_lens:
                loaders = []
                lens_actual = []
                for seq_len_local in staircase_lens:
                    x, y, used_len = _make_assoc_clean(seq_len_local)
                    ds, collate = _wrap_dataset(x, y)
                    loader = DataLoader(
                        ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                        collate_fn=collate,
                    )
                    loaders.append(loader)
                    lens_actual.append(used_len)
                num_classes = 2
                if lens_actual != staircase_lens:
                    log(f"[staircase] assoc_clean adjusted lens {staircase_lens} -> {lens_actual}")
                staircase = StaircaseController(
                    lens_actual,
                    staircase_weights,
                    STAIRCASE_MIN_BASE,
                    STAIRCASE_SHIFT,
                    STAIRCASE_STABLE_STD,
                    STAIRCASE_ADAPT_EVERY,
                )
                batcher = StaircaseBatcher(loaders, staircase_weights, SEED, staircase=staircase)
                SYNTH_META.update(
                    {
                        "assoc_keys": keys,
                        "assoc_pairs": pairs,
                        "synth_len": lens_actual[0] if lens_actual else base_seq_len,
                        "staircase_lens": lens_actual,
                        "staircase_weights": staircase.weights,
                    }
                )
                log(
                    f"[synth] mode=assoc_clean rows={int(n_samples)} keys={keys} pairs={pairs} "
                    f"lens={lens_actual} weights={staircase.weights}"
                )
                return batcher, num_classes, collate

            x, y, seq_len = _make_assoc_clean(base_seq_len)
            SYNTH_META.update({"assoc_keys": keys, "assoc_pairs": pairs, "synth_len": seq_len})
            num_classes = 2
            log(f"[synth] mode=assoc_clean rows={int(n_samples)} keys={keys} pairs={pairs} len={seq_len}")
            ds, collate = _wrap_dataset(x, y)
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
            pairs = max(1, int(ASSOC_PAIRS))
            keys = max(2, int(ASSOC_KEYS))
            val_range = max(2, int(ASSOC_VAL_RANGE))
            min_len = pairs * 2 + 1
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

            def _make_assoc_byte(seq_len_local: int):
                seq_len = seq_len_local
                bump_attempts = 0
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
                return x, y, seq_len

            if staircase_lens:
                loaders = []
                lens_actual = []
                for seq_len_local in staircase_lens:
                    x, y, used_len = _make_assoc_byte(seq_len_local)
                    ds, collate = _wrap_dataset(x, y)
                    loader = DataLoader(
                        ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                        collate_fn=collate,
                    )
                    loaders.append(loader)
                    lens_actual.append(used_len)
                num_classes = val_range
                if lens_actual != staircase_lens:
                    log(f"[staircase] assoc_byte adjusted lens {staircase_lens} -> {lens_actual}")
                staircase = StaircaseController(
                    lens_actual,
                    staircase_weights,
                    STAIRCASE_MIN_BASE,
                    STAIRCASE_SHIFT,
                    STAIRCASE_STABLE_STD,
                    STAIRCASE_ADAPT_EVERY,
                )
                batcher = StaircaseBatcher(loaders, staircase_weights, SEED, staircase=staircase)
                SYNTH_META.update(
                    {
                        "assoc_keys": keys,
                        "assoc_pairs": pairs,
                        "assoc_val_range": val_range,
                        "synth_len": lens_actual[0] if lens_actual else base_seq_len,
                        "staircase_lens": lens_actual,
                        "staircase_weights": staircase.weights,
                    }
                )
                log(
                    f"[synth] mode=assoc_byte rows={int(n_samples)} keys={keys} vals={val_range} "
                    f"pairs={pairs} lens={lens_actual} weights={staircase.weights}"
                )
                return batcher, num_classes, collate

            x, y, seq_len = _make_assoc_byte(base_seq_len)
            SYNTH_META.update(
                {"assoc_keys": keys, "assoc_pairs": pairs, "assoc_val_range": val_range, "synth_len": seq_len}
            )
            num_classes = val_range
            log(
                f"[synth] mode=assoc_byte rows={int(n_samples)} keys={keys} vals={val_range} "
                f"pairs={pairs} len={seq_len}"
            )
            ds, collate = _wrap_dataset(x, y)
            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate,
            )
            return loader, num_classes, collate
        elif synth_mode == "assoc_mix":
            pairs = max(1, int(ASSOC_PAIRS))
            keys = max(2, int(ASSOC_KEYS))
            val_range = max(2, int(ASSOC_VAL_RANGE))
            min_len = pairs * 2 + 1
            max_bumps = 5
            n_samples_int = int(n_samples)
            n_clean = n_samples_int // 2
            n_byte = n_samples_int - n_clean

            def _build_assoc_clean(seq_len_local: int, n_samples_local: int):
                x_local = torch.zeros((n_samples_local, seq_len_local, 1), dtype=torch.float32)
                y_local = torch.zeros((n_samples_local,), dtype=torch.long)
                max_start_local = seq_len_local - 3  # reserve last token for query
                for idx in range(n_samples_local):
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

            def _build_assoc_byte(seq_len_local: int, n_samples_local: int):
                x_local = torch.zeros((n_samples_local, seq_len_local, 1), dtype=torch.float32)
                y_local = torch.zeros((n_samples_local,), dtype=torch.long)
                max_start_local = seq_len_local - 3  # reserve last token for query
                for idx in range(n_samples_local):
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

            mix_offset = float(os.environ.get("TP6_ASSOC_MIX_OFFSET", "100.0"))
            mix_clean_offset = float(os.environ.get("TP6_ASSOC_MIX_CLEAN_OFFSET", "0.0"))
            mix_domain_token = os.environ.get("TP6_ASSOC_MIX_DOMAIN_TOKEN") == "1"
            mix_clean_sentinel = float(os.environ.get("TP6_ASSOC_MIX_CLEAN_SENTINEL", "50.0"))
            mix_byte_sentinel = float(os.environ.get("TP6_ASSOC_MIX_BYTE_SENTINEL", "150.0"))

            def _prepend_domain_token(x_src, token):
                x_pad = torch.zeros(
                    (x_src.size(0), x_src.size(1) + 1, x_src.size(2)),
                    dtype=x_src.dtype,
                )
                x_pad[:, 1:, :] = x_src
                x_pad[:, 0, 0] = token
                return x_pad

            def _make_assoc_mix(seq_len_local: int):
                seq_len = seq_len_local
                bump_attempts = 0
                if seq_len < min_len:
                    log(f"[synth] assoc_mix bump len from {seq_len} to {min_len} (min_len)")
                    seq_len = min_len
                x_clean = None
                y_clean = None
                x_byte = None
                y_byte = None
                while bump_attempts <= max_bumps:
                    x_clean, y_clean = _build_assoc_clean(seq_len, n_clean)
                    x_byte, y_byte = _build_assoc_byte(seq_len, n_byte)
                    if x_clean is not None and x_byte is not None:
                        break
                    bump_attempts += 1
                    new_len = seq_len + max(2, pairs) * 2
                    log(f"[synth] assoc_mix bump len from {seq_len} to {new_len} (placement failed)")
                    seq_len = new_len
                if x_clean is None or x_byte is None:
                    raise RuntimeError("assoc_mix: failed to place non-overlapping pairs after bumps")
                if mix_clean_offset:
                    mask = x_clean > 0
                    x_clean[mask] = x_clean[mask] + mix_clean_offset
                if mix_offset:
                    mask = x_byte > 0
                    x_byte[mask] = x_byte[mask] + mix_offset
                if mix_domain_token:
                    x_clean = _prepend_domain_token(x_clean, mix_clean_sentinel)
                    x_byte = _prepend_domain_token(x_byte, mix_byte_sentinel)
                    seq_len = seq_len + 1
                y_byte = y_byte + 2
                x = torch.cat([x_clean, x_byte], dim=0)
                y = torch.cat([y_clean, y_byte], dim=0)
                perm = torch.randperm(x.size(0))
                x = x[perm]
                y = y[perm]
                return x, y, seq_len

            if staircase_lens:
                loaders = []
                lens_actual = []
                for seq_len_local in staircase_lens:
                    x, y, used_len = _make_assoc_mix(seq_len_local)
                    ds, collate = _wrap_dataset(x, y)
                    loader = DataLoader(
                        ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                        collate_fn=collate,
                    )
                    loaders.append(loader)
                    lens_actual.append(used_len)
                num_classes = val_range + 2
                if lens_actual != staircase_lens:
                    log(f"[staircase] assoc_mix adjusted lens {staircase_lens} -> {lens_actual}")
                staircase = StaircaseController(
                    lens_actual,
                    staircase_weights,
                    STAIRCASE_MIN_BASE,
                    STAIRCASE_SHIFT,
                    STAIRCASE_STABLE_STD,
                    STAIRCASE_ADAPT_EVERY,
                )
                batcher = StaircaseBatcher(loaders, staircase_weights, SEED, staircase=staircase)
                SYNTH_META.update(
                    {
                        "assoc_keys": keys,
                        "assoc_pairs": pairs,
                        "assoc_val_range": val_range,
                        "synth_len": lens_actual[0] if lens_actual else base_seq_len,
                        "assoc_mix_clean": int(n_clean),
                        "assoc_mix_byte": int(n_byte),
                        "assoc_mix_offset": mix_offset,
                        "assoc_mix_clean_offset": mix_clean_offset,
                        "assoc_mix_domain_token": int(mix_domain_token),
                        "assoc_mix_clean_sentinel": mix_clean_sentinel,
                        "assoc_mix_byte_sentinel": mix_byte_sentinel,
                        "staircase_lens": lens_actual,
                        "staircase_weights": staircase.weights,
                    }
                )
                log(
                    f"[synth] mode=assoc_mix rows={int(n_samples_int)} clean={n_clean} byte={n_byte} "
                    f"keys={keys} vals={val_range} pairs={pairs} lens={lens_actual} "
                    f"offsets=clean:{mix_clean_offset:g} byte:{mix_offset:g} "
                    f"sentinel={int(mix_domain_token)} weights={staircase.weights}"
                )
                return batcher, num_classes, collate

            x, y, seq_len = _make_assoc_mix(base_seq_len)
            SYNTH_META.update(
                {
                    "assoc_keys": keys,
                    "assoc_pairs": pairs,
                    "assoc_val_range": val_range,
                    "synth_len": seq_len,
                    "assoc_mix_clean": int(n_clean),
                    "assoc_mix_byte": int(n_byte),
                    "assoc_mix_offset": mix_offset,
                    "assoc_mix_clean_offset": mix_clean_offset,
                    "assoc_mix_domain_token": int(mix_domain_token),
                    "assoc_mix_clean_sentinel": mix_clean_sentinel,
                    "assoc_mix_byte_sentinel": mix_byte_sentinel,
                }
            )
            num_classes = val_range + 2
            log(
                f"[synth] mode=assoc_mix rows={int(n_samples_int)} clean={n_clean} byte={n_byte} "
                f"keys={keys} vals={val_range} pairs={pairs} len={seq_len} "
                f"offsets=clean:{mix_clean_offset:g} byte:{mix_offset:g} "
                f"sentinel={int(mix_domain_token)}"
            )
            ds, collate = _wrap_dataset(x, y)
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
                    if os.environ.get("TP6_PTR_INERTIA_OVERRIDE") is None:
                        model.ptr_inertia = ctrl["inertia"]
                        model.ptr_walk_prob = ctrl["walk_prob"]
            ptr_velocity_raw = getattr(model, "ptr_delta_raw_mean", None)
            # Do not override manual inertia when TP6_PTR_INERTIA_OVERRIDE is set.
            if os.environ.get("TP6_PTR_INERTIA_OVERRIDE") is None:
                apply_inertia_auto(model, ptr_velocity_raw, INERTIA_AUTO_PARAMS, panic_active=panic_status == "PANIC")
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
            ignore_max_steps = os.environ.get("TP6_IGNORE_MAX_STEPS") == "1"
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


def eval_model(model, loader, dataset_name, model_name):
    model = model.to(DEVICE, dtype=DTYPE)
    model.eval()
    if hasattr(model, "head") and not hasattr(model.head, "out_features"):
        if hasattr(model.head, "experts") and model.head.experts:
            model.head.out_features = model.head.experts[0].out_features
        elif hasattr(model.head, "single") and model.head.single is not None:
            model.head.out_features = model.head.single.out_features
    criterion = nn.CrossEntropyLoss()
    collect_mitosis = MITOSIS_ENABLED
    addr_loss_sum = None
    addr_count = None
    expert_counts = None
    ring_len = int(getattr(model, "ring_range", getattr(model, "ring_len", 0)) or 0)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    dom0_correct = 0
    dom0_seen = 0
    dom1_correct = 0
    dom1_seen = 0
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
                if collect_mitosis:
                    loss_vec = F.cross_entropy(outputs, targets, reduction="none")
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_seen += inputs.size(0)
            if SYNTH_MODE == "assoc_mix":
                dom0_mask = targets < 2
                dom1_mask = ~dom0_mask
                dom0_seen += dom0_mask.sum().item()
                dom1_seen += dom1_mask.sum().item()
                dom0_correct += ((preds == targets) & dom0_mask).sum().item()
                dom1_correct += ((preds == targets) & dom1_mask).sum().item()
            if collect_mitosis and ring_len > 0:
                addr = getattr(model, "last_ptr_int", None)
                router_map = getattr(model, "router_map", None)
                if addr is not None:
                    addr = addr.to(torch.long).cpu()
                    loss_cpu = loss_vec.detach().cpu().to(torch.float32)
                    if addr_loss_sum is None:
                        addr_loss_sum = torch.zeros(ring_len, dtype=torch.float32)
                        addr_count = torch.zeros(ring_len, dtype=torch.float32)
                    addr_loss_sum += torch.bincount(addr, weights=loss_cpu, minlength=ring_len)[:ring_len]
                    addr_count += torch.bincount(addr, minlength=ring_len)[:ring_len]
                    if router_map is not None and router_map.numel() > 0:
                        mapped = router_map.detach().cpu()
                        mapped_ids = mapped[addr.clamp(0, mapped.numel() - 1)]
                        num_experts = int(getattr(model.head, "num_experts", 1))
                        if expert_counts is None:
                            expert_counts = torch.zeros(num_experts, dtype=torch.float32)
                        expert_counts += torch.bincount(mapped_ids, minlength=num_experts).float()[:num_experts]
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
    dom0_acc = dom0_correct / max(dom0_seen, 1) if SYNTH_MODE == "assoc_mix" else None
    dom1_acc = dom1_correct / max(dom1_seen, 1) if SYNTH_MODE == "assoc_mix" else None
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
    try:
        model.last_eval_acc = float(acc)
    except Exception:
        model.last_eval_acc = None
    mitosis_parent = None
    mitosis_hot = None
    mitosis_imbalance = None
    if collect_mitosis and expert_counts is not None and addr_loss_sum is not None:
        total = expert_counts.sum().item()
        if total > 0:
            shares = expert_counts / total
            mitosis_parent = int(torch.argmax(shares).item())
            mitosis_imbalance = float(shares[mitosis_parent].item())
            router_map = getattr(model, "router_map", None)
            if router_map is not None and router_map.numel() > 0:
                loss_mean = addr_loss_sum / addr_count.clamp(min=1.0)
                router_cpu = router_map.detach().cpu()
                parent_mask = router_cpu == mitosis_parent
                addr_idx = torch.nonzero(parent_mask, as_tuple=False).view(-1)
                if addr_idx.numel() > 0:
                    parent_losses = loss_mean[addr_idx]
                    top_k = min(5, parent_losses.numel())
                    top_vals, top_idx = torch.topk(parent_losses, k=top_k, largest=True)
                    mitosis_hot = [int(addr_idx[i].item()) for i in top_idx]
    if SYNTH_MODE == "assoc_mix":
        log(
            f"{dataset_name} | {model_name} | eval_loss {avg_loss:.4f} | eval_acc {acc:.4f} | "
            f"eval_acc_d0 {dom0_acc:.4f} | eval_acc_d1 {dom1_acc:.4f} | eval_n {total_seen}"
        )
    else:
        log(f"{dataset_name} | {model_name} | eval_loss {avg_loss:.4f} | eval_acc {acc:.4f} | eval_n {total_seen}")
    return {
        "eval_loss": avg_loss,
        "eval_acc": acc,
        "eval_acc_d0": dom0_acc,
        "eval_acc_d1": dom1_acc,
        "eval_n": total_seen,
        "eval_mi_bits": mi_bits,
        "eval_mi_bits_shuffled": mi_bits_shuffle,
        "eval_ptr_flip_rate": ptr_flip_rate,
        "eval_tei": tei,
        "mitosis_parent_expert": mitosis_parent,
        "mitosis_hot_addresses": mitosis_hot,
        "mitosis_expert_imbalance": mitosis_imbalance,
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
                if os.environ.get("TP6_PTR_INERTIA_OVERRIDE") is None:
                    model.ptr_inertia = ctrl["inertia"]
                    model.ptr_walk_prob = ctrl["walk_prob"]
        ptr_velocity_raw = getattr(model, "ptr_delta_raw_mean", None)
        if os.environ.get("TP6_PTR_INERTIA_OVERRIDE") is None:
            apply_inertia_auto(model, ptr_velocity_raw, INERTIA_AUTO_PARAMS, panic_active=panic_status == "PANIC")
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
    hall_train = train_wallclock(hallway, loader, dataset_name, "absolute_hallway", num_classes, eval_loader=eval_loader)
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
    if RESUME:
        _maybe_override_expert_heads(os.path.abspath(CHECKPOINT_PATH))
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
