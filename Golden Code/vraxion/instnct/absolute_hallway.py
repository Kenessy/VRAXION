# Copyright (c) Vraxion
# SPDX-License-Identifier: MIT
"""
AbsoluteHallway core model.

This is a "goldenized" extraction of the legacy monolithic model. The goal is
to preserve the original behavior and telemetry as closely as possible while
making the code safe to import as a library module (no import-time side effects).

Key characteristics (from the legacy design):
- Boundaryless ring memory with intrinsic per-neuron pointer parameters.
- Pointer update is a soft mix of jump target and walk/stay, with optional
  inertia and deadzone.
- Readout uses either a tensorized history of pointers or a soft window.
- Optional "satiety" early-exit per sample when prediction confidence exceeds a
  threshold.
- Optional auxiliary rings: sensory pre-ring, vault ring, think ring.

Environment variable semantics are preserved for all env vars referenced in the
legacy excerpt (VRX_*).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import torch
from torch import nn
from torch.nn import functional as F


# -----------------------------
# Small helpers (no side effects)
# -----------------------------


def _env_is_one(name: str, default: bool = False) -> bool:
    """Legacy boolean semantics: only the literal string '1' enables."""
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip() == "1"


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return int(default)
    try:
        return int(float(val))
    except Exception:
        return int(default)


def nan_guard(tag: str, x: torch.Tensor, step: int) -> None:
    """
    Optional NaN/Inf guard for debugging.

    The legacy code used nan_guard for telemetry. Here we keep it as an opt-in
    runtime check controlled by VRX_NAN_GUARD:

    - VRX_NAN_GUARD unset / '0': no-op (default).
    - VRX_NAN_GUARD == '1': raise RuntimeError on first non-finite detection.
    """
    if os.environ.get("VRX_NAN_GUARD", "0").strip() != "1":
        return
    if x is None:
        return
    if not torch.isfinite(x).all():
        # Avoid heavy reductions unless needed.
        with torch.no_grad():
            finite = torch.isfinite(x)
            bad = int((~finite).sum().item())
            total = int(x.numel())
            x_min = float(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).min().item())
            x_max = float(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).max().item())
        raise RuntimeError(
            f"Non-finite detected at step={step} tag={tag} bad={bad}/{total} "
            f"min={x_min:.6g} max={x_max:.6g}"
        )


# -----------------------------
# Optional dependencies (fallbacks)
# -----------------------------


try:
    # If the wider Vraxion runtime provides these, prefer them.
    from .experts import LocationExpertRouter  # type: ignore
except Exception:  # pragma: no cover

    class LocationExpertRouter(nn.Module):
        """
        Minimal per-sample expert router.

        The legacy code routes each sample to an expert based on pointer address.
        This fallback keeps the same interface:
            head(x: [B,D], expert_ids: [B]) -> logits: [B,C]
        """

        def __init__(self, input_dim: int, num_classes: int, num_experts: int = 1) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.num_classes = int(num_classes)
            self.num_experts = int(max(1, num_experts))
            self.weight = nn.Parameter(torch.empty(self.num_experts, self.num_classes, self.input_dim))
            self.bias = nn.Parameter(torch.empty(self.num_experts, self.num_classes))
            self.reset_parameters()

        def reset_parameters(self) -> None:
            for e in range(self.num_experts):
                nn.init.xavier_uniform_(self.weight[e])
            nn.init.zeros_(self.bias)

        def forward(self, x: torch.Tensor, expert_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
            if expert_ids is None or self.num_experts <= 1:
                return F.linear(x, self.weight[0], self.bias[0])

            ids = expert_ids.to(device=x.device, dtype=torch.long)
            if ids.dim() != 1:
                ids = ids.view(-1)
            if ids.numel() == 1 and x.shape[0] != 1:
                ids = ids.expand(x.shape[0])
            if ids.shape[0] != x.shape[0]:
                raise ValueError(f"expert_ids shape {tuple(ids.shape)} incompatible with batch {x.shape[0]}")
            ids = ids.clamp(0, self.num_experts - 1)

            w = self.weight.index_select(0, ids)  # [B,C,D]
            b = self.bias.index_select(0, ids)  # [B,C]
            # logits[b,c] = sum_d x[b,d] * w[b,c,d]
            logits = torch.einsum("bd,bcd->bc", x, w) + b
            return logits


try:
    from .brainstem import BrainstemMixer  # type: ignore
except Exception:  # pragma: no cover

    class BrainstemMixer:
        """
        Minimal BrainstemMixer fallback.

        The legacy code uses a BrainstemMixer to compute a FAST/SHIELD share from
        an entropy proxy. This implementation provides a stable interface without
        introducing new dependencies.

        update(...) returns:
            w_shield: float in [0, 1]
            info: Dict with debug fields
        """

        def __init__(self, beta: float = 0.9) -> None:
            self.beta = float(beta)
            self.entropy_ema: Optional[float] = None

        def update(self, current_entropy: float, repetition_count: int = 0, dt: float = 1.0) -> Tuple[float, Dict]:
            ent = float(current_entropy)
            if not math.isfinite(ent):
                ent = 0.0
            if self.entropy_ema is None:
                self.entropy_ema = ent
            else:
                b = self.beta
                self.entropy_ema = (b * self.entropy_ema) + ((1.0 - b) * ent)
            w_shield = max(0.0, min(1.0, float(self.entropy_ema)))
            info = {
                "mode": "ema",
                "danger": 0.0,
                "entropy": ent,
                "entropy_ema": float(self.entropy_ema),
                "repetition_count": int(repetition_count),
                "dt": float(dt),
            }
            return w_shield, info


# -----------------------------
# Module defaults (legacy constants)
# -----------------------------
# NOTE: These are conservative defaults suitable for small tests. The legacy
# monolithic script likely overwrote some of these via globals. Callers that
# rely on specific values should pass explicit args or set env vars (VRX_*).


RING_LEN = 256
SLOT_DIM = 128
PTR_PARAM_STRIDE = 1
GAUSS_K = 2
GAUSS_TAU = 8.0

# Pointer kernel configuration.
PTR_KERNEL = "gauss"  # "gauss" or "vonmises"
PTR_KAPPA = 8.0
PTR_EDGE_EPS = 0.0

# Activation and special non-linearities.
ACT_NAME = "tanh"
C13_P = 2.0

# Mobius / phase embedding.
MOBIUS_ENABLED = False
MOBIUS_EMB_SCALE = 0.02

# Pointer control defaults.
PTR_INERTIA = 0.0
PTR_DEADZONE = 0.0
PTR_DEADZONE_TAU = 0.25
PTR_WALK_PROB = 1.0

# State write scaling and clamps.
UPDATE_SCALE = 1.0
AGC_SCALE_MAX = 1.0
STATE_DECAY = 1.0
STATE_CLIP = 0.0

# Pointer velocity governor.
PTR_VEL = False
PTR_VEL_DECAY = 0.9
PTR_VEL_CAP = 1.0
PTR_VEL_SCALE = 1.0

# Pointer locking / update cadence.
PTR_LOCK = False
PTR_LOCK_VALUE = 0.0
PTR_UPDATE_EVERY = 1
PTR_UPDATE_AUTO = False
PTR_UPDATE_MIN = 1
PTR_UPDATE_MAX = 16
PTR_UPDATE_EVERY_STEP = 1
PTR_UPDATE_TARGET_FLIP = 0.2
PTR_UPDATE_EMA = 0.9

# Pointer gating and rounding.
PTR_GATE_MODE = "none"  # "none" or "steps"
PTR_GATE_STEPS = ""  # comma-separated ints
PTR_SOFT_GATE = False
PTR_WARMUP_STEPS = 0
PTR_NO_ROUND = False
PTR_PHANTOM = False
PTR_PHANTOM_OFF = 0.5
PTR_PHANTOM_READ = False

# Readout.
SOFT_READOUT = False
SOFT_READOUT_K = 2
SOFT_READOUT_TAU = 8.0

# Satiety early-exit.
SATIETY_THRESH = 0.0

# Token decay controls (legacy BOS/EOS semantics).
BOS_DECAY = 1.0
BOS_ID = 1
EOS_ID = 2

# Deterministic pointer in eval.
EVAL_PTR_DETERMINISTIC = False

# Debug telemetry.
DEBUG_STATS = False
DEBUG_EVERY = 0

# Pointer jump controls.
PTR_JUMP_CAP = 1.0
PTR_JUMP_DISABLED = False

# Dual-pointer anchor controls (Phase A).
PTR_ANCHOR_MIN_STEP = 0.0
PTR_ANCHOR_CLICK_INJECT = False
PTR_ANCHOR_CONF_MIN = 0.0

# State loop metrics.
STATE_LOOP_METRICS = False
STATE_LOOP_EVERY = 1
STATE_LOOP_SAMPLES = 0
STATE_LOOP_DIM = 8

# Pointer dtype (legacy: a global torch dtype).
PTR_DTYPE = torch.float32

# Expert head count.
EXPERT_HEADS = 1


# -----------------------------
# Main model
# -----------------------------


class AbsoluteHallway(nn.Module):
    @staticmethod
    def wrap_delta(a: torch.Tensor, b: torch.Tensor, ring_range: int) -> torch.Tensor:
        """Shortest signed delta from a to b on a ring."""
        rr = float(ring_range)
        return torch.remainder(b - a + rr / 2.0, rr) - rr / 2.0

    @staticmethod
    def circ_lerp(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, ring_range: int) -> torch.Tensor:
        """Move from a toward b by fraction w along shortest arc."""
        rr = float(ring_range)
        return torch.remainder(a + w * AbsoluteHallway.wrap_delta(a, b, ring_range), rr)

    """
    Boundaryless ring with intrinsic pointer params per neuron.

    - Each neuron has theta_ptr (target coord) and theta_gate (bias).
    - Pointer update is a soft mix of jump target and walk/stay, then optional
      inertia/deadzone.
    - Readout: average of states at last K pointers (tensorized) or soft readout
      window.
    - Satiety exit: if max prob > SATIETY_THRESH, stop processing further
      timesteps for that sample.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        ring_len: int = RING_LEN,
        slot_dim: int = SLOT_DIM,
        ptr_stride: int = PTR_PARAM_STRIDE,
        gauss_k: int = GAUSS_K,
        gauss_tau: float = GAUSS_TAU,
        bypass_ring: bool = False,
        time_pointer: bool = False,
        aux_ring: bool = False,
        context_scale_init: float = 0.2,
    ) -> None:
        super().__init__()

        self.ring_len = int(ring_len)
        self.slot_dim = int(slot_dim)
        self.num_classes = int(num_classes)
        self.ptr_stride = int(max(1, ptr_stride))
        self.gauss_k = int(gauss_k)
        self.gauss_tau = float(gauss_tau)

        # Pointer kernel.
        self.ptr_kernel = PTR_KERNEL if PTR_KERNEL in {"gauss", "vonmises"} else "gauss"
        self.ptr_kappa = float(PTR_KAPPA)
        self.ptr_edge_eps = float(PTR_EDGE_EPS)

        # Activation.
        self.act_name = str(ACT_NAME)

        # Telemetry switch for optional x-ray logging.
        self.collect_xray = False

        # Optional diagnostics.
        self.bypass_ring = bool(bypass_ring)
        self.time_pointer = bool(time_pointer)
        self.aux_ring = bool(aux_ring)

        # Parameters for special activations.
        self.c13_p = max(float(C13_P), 1e-6)
        self.c19_rho = 4.0
        self.c14_rho = self.c19_rho

        # Mobius / ring_range.
        self.mobius = bool(MOBIUS_ENABLED)
        self.mobius_scale = 2 if self.mobius else 1
        self.ring_range = int(self.ring_len * self.mobius_scale)

        # Pointer control (modifiable at runtime).
        self.ptr_inertia = float(PTR_INERTIA)
        self.ptr_inertia_ema = float(self.ptr_inertia)
        self.ptr_inertia_floor = 0.0
        self.ptr_inertia_dyn_pre: Optional[float] = None
        self.ptr_inertia_dyn: Optional[float] = None
        self.ptr_inertia_dyn_tensor: Optional[torch.Tensor] = None
        self.ptr_inertia_reward_ready = False
        self.ptr_inertia_reward_acc: Optional[float] = None
        self.ptr_inertia_reward_streak = 0
        self.ptr_inertia_epi = 0.0

        self.ptr_deadzone = float(PTR_DEADZONE)
        self.ptr_deadzone_tau = float(PTR_DEADZONE_TAU)
        self.ptr_walk_prob = float(PTR_WALK_PROB)

        self.update_scale = float(UPDATE_SCALE)
        self.agc_scale_max = float(AGC_SCALE_MAX)

        self.ground_speed_ema: Optional[float] = None
        self.ground_speed_limit: Optional[float] = None
        self.ground_speed: Optional[float] = None

        self.ptr_vel_enabled = bool(PTR_VEL)
        self.ptr_vel_decay = float(PTR_VEL_DECAY)
        self.ptr_vel_cap = float(PTR_VEL_CAP)
        self.ptr_vel_scale = float(PTR_VEL_SCALE)

        self.ptr_lock = bool(PTR_LOCK)
        self.ptr_lock_value = float(PTR_LOCK_VALUE)

        self.ptr_update_every = int(max(1, PTR_UPDATE_EVERY))
        self.ptr_update_auto = bool(PTR_UPDATE_AUTO)
        self.ptr_update_min = int(max(1, PTR_UPDATE_MIN))
        self.ptr_update_max = int(max(self.ptr_update_min, PTR_UPDATE_MAX))
        self.ptr_update_every_step = int(max(1, PTR_UPDATE_EVERY_STEP))
        self.ptr_update_target_flip = float(PTR_UPDATE_TARGET_FLIP)
        self.ptr_update_ema = float(PTR_UPDATE_EMA)
        self.ptr_update_ema_state: Optional[float] = None

        # Usage smoothing.
        self.usage_soft_enabled = _env_is_one("VRX_USAGE_SOFT", default=False)
        self.usage_soft_temp = float(_env_float("VRX_USAGE_SOFT_TEMP", 0.5))

        # Pointer gating schedule.
        self.ptr_gate_mode = str(PTR_GATE_MODE)
        self.ptr_gate_steps: Set[int] = set()
        if self.ptr_gate_mode == "steps" and PTR_GATE_STEPS:
            for token in str(PTR_GATE_STEPS).split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    self.ptr_gate_steps.add(int(token))
                except ValueError:
                    continue

        # Optional learned soft gate for pointer update strength.
        self.ptr_soft_gate = bool(PTR_SOFT_GATE)
        self.gate_head = nn.Linear(self.slot_dim, 1) if self.ptr_soft_gate else None

        self.ptr_warmup_steps = int(PTR_WARMUP_STEPS)
        self.ptr_no_round = bool(PTR_NO_ROUND)
        self.ptr_phantom = bool(PTR_PHANTOM)
        self.ptr_phantom_off = float(PTR_PHANTOM_OFF)
        self.ptr_phantom_read = bool(PTR_PHANTOM_READ)

        # Readout.
        self.soft_readout = bool(SOFT_READOUT)
        self.soft_readout_k = int(max(0, SOFT_READOUT_K))
        self.soft_readout_tau = max(float(SOFT_READOUT_TAU), 1e-6)

        # Optional hidden-state loop metrics.
        self.state_loop_metrics = bool(STATE_LOOP_METRICS)
        self.state_loop_every = int(max(1, STATE_LOOP_EVERY))
        self.state_loop_samples = int(max(0, STATE_LOOP_SAMPLES))
        self.state_loop_dim = int(max(2, STATE_LOOP_DIM))
        if self.state_loop_metrics:
            g = torch.Generator()
            g.manual_seed(1337)
            proj = torch.randn(self.slot_dim, self.state_loop_dim, generator=g)
            proj = torch.nn.functional.normalize(proj, dim=0)
            self.register_buffer("state_loop_proj", proj)
        else:
            self.register_buffer("state_loop_proj", torch.empty(0))

        # Sensory ring (env-controlled).
        sensory_flag = os.environ.get("VRX_SENSORY_RING", "1").strip().lower()
        self.sensory_enabled = sensory_flag not in {"0", "false", "off", "no"}
        sensory_len_env = os.environ.get("VRX_SENSORY_RING_LEN")
        sensory_dim_env = os.environ.get("VRX_SENSORY_SLOT_DIM")
        default_sensory_len = max(1, self.ring_len // 2)
        default_sensory_dim = max(8, self.slot_dim // 3)
        self.sensory_len = max(1, int(sensory_len_env)) if sensory_len_env else default_sensory_len
        self.sensory_dim = max(8, int(sensory_dim_env)) if sensory_dim_env else default_sensory_dim
        if self.sensory_enabled:
            self.sensory_proj_in = nn.Linear(int(input_dim), self.sensory_dim)
            self.sensory_gru = nn.GRUCell(self.sensory_dim, self.sensory_dim)
            self.sensory_bridge = nn.Linear(self.sensory_dim, self.slot_dim)
        else:
            self.sensory_proj_in = None
            self.sensory_gru = None
            self.sensory_bridge = None

        # Vault ring (env-controlled).
        self.vault_enabled = _env_is_one("VRX_VAULT", default=False)
        self.vault_len = int(max(1, _env_int("VRX_VAULT_LEN", 32)))
        self.vault_dim = int(max(8, _env_int("VRX_VAULT_DIM", 128)))
        self.vault_decay = float(_env_float("VRX_VAULT_DECAY", 0.3))
        self.vault_inject_scale = float(_env_float("VRX_VAULT_INJECT_SCALE", 0.2))

        # Adaptive vault control (self-regulating "tap").
        self.vault_adapt = _env_is_one("VRX_VAULT_ADAPT", default=False)
        self.vault_gate_min = float(_env_float("VRX_VAULT_GATE_MIN", 0.0))
        self.vault_gate_max = float(_env_float("VRX_VAULT_GATE_MAX", 0.6))
        self.vault_alpha_min = float(_env_float("VRX_VAULT_ALPHA_MIN", 0.05))
        self.vault_alpha_max = float(_env_float("VRX_VAULT_ALPHA_MAX", 0.5))
        self.vault_k_surprise = float(_env_float("VRX_VAULT_K_SURPRISE", 0.25))
        self.vault_k_utility = float(_env_float("VRX_VAULT_K_UTILITY", 0.25))
        self.vault_probe_every = int(_env_int("VRX_VAULT_PROBE_EVERY", 200))
        self.vault_probe_beta = float(_env_float("VRX_VAULT_PROBE_BETA", 0.9))

        self.vault_util_ema: Optional[float] = None
        self.vault_loss_floor: Optional[float] = None
        self.vault_gate = float(self.vault_inject_scale)
        self.vault_alpha = float(self.vault_decay)

        if self.vault_enabled:
            self.vault_down = nn.Linear(self.slot_dim, self.vault_dim)
            self.vault_up = nn.Linear(self.vault_dim, self.slot_dim)
        else:
            self.vault_down = None
            self.vault_up = None

        # Think ring (env-controlled).
        self.think_enabled = _env_is_one("VRX_THINK_RING", default=False)
        self.think_mode = os.environ.get("VRX_THINK_RING_MODE", "parallel").strip().lower()
        self.think_len = int(max(1, _env_int("VRX_THINK_RING_LEN", 128)))
        self.think_dim = int(max(8, _env_int("VRX_THINK_RING_DIM", 21)))
        self.think_alpha = float(_env_float("VRX_THINK_RING_ALPHA", 0.1))

        # Dual-core think ring.
        self.think_dual = _env_is_one("VRX_THINK_RING_DUAL", default=False)
        self.think_alpha_low = float(_env_float("VRX_THINK_RING_ALPHA_LOW", self.think_alpha))
        self.think_alpha_high = float(_env_float("VRX_THINK_RING_ALPHA_HIGH", self.think_alpha))

        # Mix semantics: VRX_THINK_RING_MIX is *slow* share.
        self.think_mix = float(_env_float("VRX_THINK_RING_MIX", 0.5))
        fast_share = os.environ.get("VRX_THINK_RING_FAST_SHARE")
        if fast_share is not None and str(fast_share).strip() != "":
            try:
                self.think_mix = 1.0 - float(fast_share)
            except Exception:
                pass

        # Optional brainstem mixer (entropy-driven).
        self.think_brainstem = _env_is_one("VRX_THINK_RING_BRAINSTEM", default=False)
        self.think_brainstem_every = int(max(1, _env_int("VRX_THINK_RING_BRAINSTEM_EVERY", 1)))
        self.think_brainstem_dt = float(_env_float("VRX_THINK_RING_BRAINSTEM_DT", 1.0))
        self.brainstem = BrainstemMixer() if self.think_brainstem else None
        self.think_brainstem_w = 0.0
        self.think_brainstem_mix = float(self.think_mix)
        self.think_brainstem_info: Optional[Dict] = None

        # Optional EMA writeback for think ring state.
        self.think_write_ema = _env_is_one("VRX_THINK_RING_WRITE_EMA", default=False)

        # Adaptive think-alpha slider (kept for compatibility; used externally).
        self.think_alpha_adapt = _env_is_one("VRX_THINK_ALPHA_ADAPT", default=False)
        self.think_alpha_min = float(_env_float("VRX_THINK_ALPHA_MIN", self.think_alpha))
        self.think_alpha_max = float(_env_float("VRX_THINK_ALPHA_MAX", self.think_alpha))
        self.think_alpha_beta = float(_env_float("VRX_THINK_ALPHA_BETA", 0.95))
        self.think_alpha_jitter_low = float(_env_float("VRX_THINK_ALPHA_JITTER_LOW", 0.02))
        self.think_alpha_jitter_high = float(_env_float("VRX_THINK_ALPHA_JITTER_HIGH", 0.08))
        self.think_alpha_jitter_ema: Optional[float] = None
        self.think_alpha_target = float(self.think_alpha)

        if self.think_enabled:
            self.think_proj_in = nn.Linear(self.slot_dim, self.think_dim)
            self.think_gru = nn.GRUCell(self.think_dim, self.think_dim)
            self.think_proj_out = nn.Linear(self.think_dim, self.slot_dim)
        else:
            self.think_proj_in = None
            self.think_gru = None
            self.think_proj_out = None

        # Core modules.
        self.input_proj = nn.Linear(int(input_dim), self.slot_dim)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.jump_score = nn.Linear(self.slot_dim, 1)

        # Adaptive control heads for pointer behavior.
        self.inertia_head = nn.Linear(self.slot_dim, 1)
        self.deadzone_head = nn.Linear(self.slot_dim, 1)
        self.walk_head = nn.Linear(self.slot_dim, 1)

        # Intrinsic params (downsampled for memory, then upsampled by gather).
        reduced = (self.ring_range + self.ptr_stride - 1) // self.ptr_stride
        self.theta_ptr_reduced = nn.Parameter(torch.zeros(int(reduced)))
        self.theta_gate_reduced = nn.Parameter(torch.zeros(int(reduced)))

        if self.mobius:
            self.phase_embed = nn.Parameter(torch.zeros(2, self.slot_dim))
        else:
            self.register_parameter("phase_embed", None)

        # Output head.
        self.head = LocationExpertRouter(self.slot_dim, self.num_classes, num_experts=int(EXPERT_HEADS))

        # Router map: decouple address -> expert from modulo routing.
        map_len = max(1, int(self.ring_range))
        router_init = torch.arange(map_len, dtype=torch.long) % self.head.num_experts
        self.register_buffer("router_map", router_init)

        # Learnable ring context scale (sigmoid-bounded).
        c0 = max(1e-3, min(0.999, float(context_scale_init)))
        logit = math.log(c0 / (1.0 - c0))
        self.context_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

        # Pointer histogram.
        self.pointer_hist_bins = 128
        self.register_buffer("bin_edges", torch.linspace(0, float(self.ring_range), self.pointer_hist_bins + 1))
        self.pointer_hist = torch.zeros(self.pointer_hist_bins, dtype=torch.long)

        # Telemetry.
        self.satiety_exits = 0
        self.blur_window = 1
        self.debug_stats: Optional[Dict] = None

        self.reset_parameters()

    # -----------------------------
    # Init / activation
    # -----------------------------

    def reset_parameters(self) -> None:
        # Spread ptr targets across ring; bias gates near zero.
        nn.init.uniform_(self.theta_ptr_reduced, -4.0, 4.0)
        nn.init.uniform_(self.theta_gate_reduced, -0.5, 0.5)

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        if self.sensory_enabled:
            assert self.sensory_proj_in is not None
            assert self.sensory_gru is not None
            assert self.sensory_bridge is not None
            nn.init.xavier_uniform_(self.sensory_proj_in.weight)
            nn.init.zeros_(self.sensory_proj_in.bias)
            nn.init.normal_(self.sensory_gru.weight_ih, mean=0.0, std=0.05)
            nn.init.normal_(self.sensory_gru.weight_hh, mean=0.0, std=0.05)
            nn.init.zeros_(self.sensory_gru.bias_ih)
            nn.init.zeros_(self.sensory_gru.bias_hh)
            nn.init.xavier_uniform_(self.sensory_bridge.weight)
            nn.init.zeros_(self.sensory_bridge.bias)

        if self.vault_enabled:
            assert self.vault_down is not None
            assert self.vault_up is not None
            nn.init.xavier_uniform_(self.vault_down.weight)
            nn.init.zeros_(self.vault_down.bias)
            nn.init.xavier_uniform_(self.vault_up.weight)
            nn.init.zeros_(self.vault_up.bias)

        if self.think_enabled:
            assert self.think_proj_in is not None
            assert self.think_gru is not None
            assert self.think_proj_out is not None
            nn.init.xavier_uniform_(self.think_proj_in.weight)
            nn.init.zeros_(self.think_proj_in.bias)
            nn.init.normal_(self.think_gru.weight_ih, mean=0.0, std=0.05)
            nn.init.normal_(self.think_gru.weight_hh, mean=0.0, std=0.05)
            nn.init.zeros_(self.think_gru.bias_ih)
            nn.init.zeros_(self.think_gru.bias_hh)
            nn.init.xavier_uniform_(self.think_proj_out.weight)
            nn.init.zeros_(self.think_proj_out.bias)

        # Adjust GRU power smoothly by ring length to avoid runaway accumulation.
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
            self.head.reset_parameters()  # type: ignore[attr-defined]

        if self.gate_head is not None:
            nn.init.xavier_uniform_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)

        if self.mobius and self.phase_embed is not None:
            nn.init.normal_(self.phase_embed, mean=0.0, std=float(MOBIUS_EMB_SCALE))

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

    # -----------------------------
    # Routing / telemetry helpers
    # -----------------------------

    def _map_expert_ids(self, ptr_int: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "router_map") or self.router_map.numel() == 0:
            return ptr_int
        router_map = self.router_map
        if router_map.device != ptr_int.device:
            router_map = router_map.to(ptr_int.device)
        idx = ptr_int.clamp(0, router_map.numel() - 1)
        expert_ids = router_map[idx].to(torch.long)
        # Guard against stale maps if expert count changed.
        if expert_ids.numel() and int(expert_ids.max().item()) >= int(self.head.num_experts):
            expert_ids = expert_ids % int(self.head.num_experts)
        return expert_ids

    def _update_expert_stats(self, expert_ids: Optional[torch.Tensor]) -> None:
        self.ptr_expert_ids = None
        self.ptr_expert_counts = None
        self.ptr_expert_active = 0
        self.ptr_expert_max_share = None
        self.ptr_expert_entropy = None
        if expert_ids is None or int(getattr(self.head, "num_experts", 1)) <= 1:
            return
        counts = torch.bincount(expert_ids, minlength=int(self.head.num_experts)).float()
        total = counts.sum().clamp(min=1.0)
        probs = counts / total
        max_share = (counts.max() / total).item()
        active = int((counts > 0).sum().item())
        entropy = 0.0
        if int(self.head.num_experts) > 1:
            entropy = float(-(probs * torch.log(probs + 1e-12)).sum().item() / math.log(int(self.head.num_experts)))
        self.ptr_expert_ids = expert_ids.detach().cpu()
        self.ptr_expert_counts = counts.detach().cpu()
        self.ptr_expert_active = active
        self.ptr_expert_max_share = float(max_share)
        self.ptr_expert_entropy = float(entropy)

    def _compute_step_entropy(self, ptr_int: torch.Tensor, active_mask: Optional[torch.Tensor]) -> float:
        """
        Per-step entropy proxy in [0,1] used by the BrainstemMixer.

        Prefer router-map expert IDs when available; fall back to raw pointer bins otherwise.
        """
        if ptr_int is None or ptr_int.numel() == 0:
            return 0.0
        ids = self._map_expert_ids(ptr_int)
        num_experts = int(getattr(self.head, "num_experts", 0) or 0)
        if ids.numel() and num_experts > 1 and int(ids.max().item()) < num_experts:
            norm = num_experts
        else:
            norm = int(self.ring_range) if int(self.ring_range) > 1 else num_experts
        if active_mask is not None and bool(active_mask.any()):
            ids = ids[active_mask]
        if ids.numel() == 0 or norm <= 1:
            return 0.0
        counts = torch.bincount(ids.to(torch.long), minlength=norm).float()
        total = counts.sum().clamp(min=1.0)
        probs = counts / total
        ent = float(-(probs * torch.log(probs + 1e-12)).sum().item() / math.log(norm))
        return ent

    # -----------------------------
    # Pointer parameter gathering / kernels
    # -----------------------------

    def _gather_params(self, ptr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ptr: [B] float or long, map to reduced indices with linear interpolation.
        ptr_f = ptr.to(torch.float32)
        idx_float = ptr_f / float(self.ptr_stride)
        idx_base = torch.floor(idx_float)
        frac = (idx_float - idx_base).clamp(0.0, 1.0).detach()
        n = int(self.theta_ptr_reduced.numel())
        idx0 = torch.remainder(idx_base, n).long()
        idx1 = torch.remainder(idx0 + 1, n).long()
        ring_range = int(self.ring_range)

        theta_ptr0 = torch.sigmoid(self.theta_ptr_reduced[idx0]) * float(ring_range - 1)
        theta_ptr1 = torch.sigmoid(self.theta_ptr_reduced[idx1]) * float(ring_range - 1)
        theta_ptr = theta_ptr0 + (theta_ptr1 - theta_ptr0) * frac

        theta_gate0 = self.theta_gate_reduced[idx0]
        theta_gate1 = self.theta_gate_reduced[idx1]
        theta_gate = theta_gate0 + (theta_gate1 - theta_gate0) * frac
        return theta_ptr, theta_gate

    def _compute_kernel_weights(
        self,
        ptr_float: torch.Tensor,
        offsets: torch.Tensor,
        ring_range: int,
        tau_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Integer centers (non-differentiable) + fractional pointer (differentiable).
        rr = int(ring_range)
        base = torch.floor(ptr_float).long().clamp(0, rr - 1)
        offsets_long = offsets.to(torch.long)
        centers = torch.remainder(base.unsqueeze(1) + offsets_long.unsqueeze(0), rr)
        centers = torch.nan_to_num(centers, nan=0, posinf=rr - 1, neginf=0)
        centers_f = centers.to(ptr_float.dtype)
        if self.ptr_kernel == "vonmises":
            angle_scale = (2.0 * math.pi) / max(float(rr), 1e-6)
            delta = (centers_f - ptr_float.unsqueeze(1)) * angle_scale
            kappa = max(float(self.ptr_kappa), 1e-6)
            logits = kappa * torch.cos(delta)
        else:
            delta = self.wrap_delta(ptr_float.unsqueeze(1), centers_f, rr)
            d2 = delta ** 2
            tau = max(float(self.gauss_tau if tau_override is None else tau_override), 1e-4)
            logits = -d2 / tau
        weights = torch.softmax(logits, dim=1)
        return centers, weights, centers_f

    def _compute_gru_gates(self, inp: torch.Tensor, cur: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute GRU gates for telemetry (sigmoid/tanh activations)."""
        wi, wh = self.gru.weight_ih, self.gru.weight_hh
        bi, bh = self.gru.bias_ih, self.gru.bias_hh
        gates = F.linear(inp, wi, bi) + F.linear(cur, wh, bh)
        r, z, n = gates.chunk(3, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n)
        return r, z, n

    # -----------------------------
    # Forward
    # -----------------------------

    def forward(self, x: torch.Tensor, return_xray: bool = False):
        """
        Args:
            x: [B,T,input_dim]
            return_xray: if True, returns an extra dict with telemetry.

        Returns:
            (logits, move_penalty) or (logits, move_penalty, xray)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B,T,D], got {tuple(x.shape)}")
        B, T, _ = x.shape
        device = x.device

        # Legacy BOS/EOS decay mode only triggers on scalar token streams.
        bos_decay = max(0.0, min(1.0, float(BOS_DECAY)))
        bos_mask = None
        eos_mask = None
        decay_on = None
        decay_off = None
        if bos_decay < 1.0 and x.shape[2] == 1:
            bos_mask = x[:, :, 0] == float(BOS_ID)
            eos_mask = x[:, :, 0] == float(EOS_ID)
            decay_on = torch.tensor(bos_decay, device=device, dtype=x.dtype)
            decay_off = torch.tensor(1.0, device=device, dtype=x.dtype)

        ring_range = int(self.ring_range)
        ptr_dtype = PTR_DTYPE

        # -----------------------------
        # Sensory ring pre-processing
        # -----------------------------
        sensory_seq = None
        if self.sensory_enabled:
            assert self.sensory_proj_in is not None
            assert self.sensory_gru is not None
            assert self.sensory_bridge is not None
            s_dtype = self.sensory_gru.weight_ih.dtype
            s_state = torch.zeros(B, self.sensory_len, self.sensory_dim, device=device, dtype=s_dtype)
            s_h = torch.zeros(B, self.sensory_dim, device=device, dtype=s_dtype)
            s_ptr = torch.zeros(B, device=device, dtype=torch.long)
            s_out = []
            s_decay_on = decay_on.to(s_dtype) if decay_on is not None else None
            s_decay_off = decay_off.to(s_dtype) if decay_off is not None else None
            idx = torch.arange(B, device=device)
            for t in range(T):
                if bos_mask is not None:
                    mask_t = bos_mask[:, t]
                    if bool(mask_t.any()):
                        decay = torch.where(mask_t, s_decay_on, s_decay_off).view(B, 1, 1)
                        s_state = s_state * decay
                        s_h = s_h * decay.view(B, 1)
                s_inp = self.sensory_proj_in(x[:, t, :])
                s_inp = self._apply_activation(s_inp)
                s_ctx = s_state[idx, s_ptr]
                s_h = self.sensory_gru(s_inp + s_ctx, s_h)
                if s_h.dtype != s_state.dtype:
                    s_h = s_h.to(s_state.dtype)
                s_state[idx, s_ptr] = s_h
                s_out.append(s_h)
                s_ptr = (s_ptr + 1) % int(self.sensory_len)
            sensory_seq = torch.stack(s_out, dim=1)

        # -----------------------------
        # Diagnostic bypass
        # -----------------------------
        if self.bypass_ring:
            h = torch.zeros(B, self.slot_dim, device=device, dtype=x.dtype)
            movement_cost = torch.tensor(0.0, device=device, dtype=x.dtype)
            pointer_addresses = torch.zeros(B, device=device, dtype=torch.long)
            for t in range(T):
                if bos_mask is not None:
                    mask_t = bos_mask[:, t]
                    if bool(mask_t.any()):
                        decay = torch.where(mask_t, decay_on, decay_off).view(B, 1)
                        h = h * decay
                if self.sensory_enabled:
                    assert sensory_seq is not None
                    assert self.sensory_bridge is not None
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

        # -----------------------------
        # Main ring state
        # -----------------------------
        state = torch.zeros(B, ring_range, self.slot_dim, device=device, dtype=x.dtype)
        h = torch.zeros(B, self.slot_dim, device=device, dtype=x.dtype)

        # Randomize start pointer per sample to break symmetry (float for STE).
        if self.ptr_lock:
            ptr_float = torch.full((B,), float(self.ptr_lock_value), device=device, dtype=ptr_dtype) * float(ring_range - 1)
        elif (not self.training) and bool(EVAL_PTR_DETERMINISTIC):
            ptr_float = torch.zeros(B, device=device, dtype=ptr_dtype)
        else:
            ptr_float = torch.rand(B, device=device, dtype=ptr_dtype) * float(ring_range - 1)

        ptr_int_init = torch.floor(torch.remainder(ptr_float, float(ring_range))).clamp(0, ring_range - 1).long()
        ptr_int = ptr_int_init
        last_ptrs = ptr_int_init.view(B, 1).repeat(1, int(self.blur_window))
        hist = torch.zeros(self.pointer_hist_bins, device=device, dtype=torch.long)
        satiety_exited = torch.zeros(B, device=device, dtype=torch.bool)
        ptr_vel = torch.zeros(B, device=device, dtype=ptr_dtype)

        movement_cost: torch.Tensor | float = 0.0
        raw_movement_cost: torch.Tensor | float = 0.0

        # Vault ring is only active for BOS/EOS token streams.
        vault_active = bool(self.vault_enabled and bos_mask is not None and eos_mask is not None)
        vault_ring = None
        vault_ptr = None
        vault_injections = 0
        vault_updates = 0
        if vault_active:
            assert self.vault_down is not None
            assert self.vault_up is not None
            vault_dtype = self.vault_down.weight.dtype
            vault_ring = torch.zeros(B, self.vault_len, self.vault_dim, device=device, dtype=vault_dtype)
            vault_ptr = torch.zeros(B, device=device, dtype=torch.long)

        # Think ring.
        think_active = bool(self.think_enabled)
        think_ring = None
        think_ptr = None
        think_ring2 = None
        think_ptr2 = None
        think_dual = False
        if think_active:
            assert self.think_proj_in is not None
            assert self.think_gru is not None
            assert self.think_proj_out is not None
            think_dtype = self.think_proj_in.weight.dtype
            think_ring = torch.zeros(B, self.think_len, self.think_dim, device=device, dtype=think_dtype)
            think_ptr = torch.zeros(B, device=device, dtype=torch.long)
            think_dual = bool(getattr(self, "think_dual", False))
            if think_dual:
                think_ring2 = torch.zeros(B, self.think_len, self.think_dim, device=device, dtype=think_dtype)
                think_ptr2 = torch.zeros(B, device=device, dtype=torch.long)

        # Dynamic pointer trace metrics.
        prev_ptr_int = None
        prev_prev_ptr_int = None
        dwell_len = torch.zeros(B, device=device, dtype=torch.long)
        max_dwell = torch.zeros(B, device=device, dtype=torch.long)
        flip_count = torch.zeros(B, device=device, dtype=torch.long)
        pingpong_count = torch.zeros(B, device=device, dtype=torch.long)
        total_active_steps = 0
        active_steps_per_sample = torch.zeros(B, device=device, dtype=torch.long)

        collect_xray = bool(return_xray or self.collect_xray)
        target_mask = None
        gate_sat_count = 0.0
        gate_sat_total = 0.0
        h_abs_sum = 0.0
        h_abs_count = 0
        if collect_xray:
            target_mask = torch.zeros(B, ring_range, device=device, dtype=x.dtype)
            attn_max = torch.zeros(B, device=device, dtype=x.dtype)

        ctrl_inertia_mean = None
        ctrl_deadzone_mean = None
        ctrl_walk_mean = None

        # Dual-pointer (anchor + residual).
        loss_ema = getattr(self, "loss_ema", None)
        confidence = 0.0
        if loss_ema is not None:
            try:
                confidence = 1.0 / (1.0 + float(loss_ema))
            except Exception:
                confidence = 0.0

        kernel_width = max(float(self.gauss_tau), 1e-6)
        min_step_floor = max(
            ring_range * torch.finfo(torch.float64).eps,
            kernel_width * 1e-3,
        )
        min_step = max(min_step_floor, float(self.ptr_stride))
        if float(PTR_ANCHOR_MIN_STEP) > 0.0:
            min_step = max(min_step_floor, float(PTR_ANCHOR_MIN_STEP))
        self.ptr_min_step = float(min_step)

        ptr_anchor = ptr_float.to(torch.float64)
        ptr_anchor = torch.remainder(torch.round(ptr_anchor / min_step) * min_step, float(ring_range))
        ptr_residual = ptr_float.to(torch.float64) - ptr_anchor
        ptr_float = (ptr_anchor + ptr_residual).to(ptr_dtype)

        res_mean_init = float(ptr_residual.abs().mean().item())
        self.ptr_residual_mean = res_mean_init
        self.ptr_orbit = 2 if res_mean_init >= (min_step * 0.1) else 1

        # Internal state loop metrics.
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

        satiety_enabled = float(SATIETY_THRESH) > 0.0 and self.num_classes > 1

        # Pre-allocate offsets for the main kernel.
        offsets = torch.arange(-self.gauss_k, self.gauss_k + 1, device=device, dtype=ptr_float.dtype)

        logits = torch.zeros(B, self.num_classes, device=device, dtype=x.dtype)
        upd = h  # placeholder for type-checkers

        for t in range(T):
            active_mask = ~satiety_exited
            if not bool(active_mask.any()):
                break

            # BOS decay.
            if bos_mask is not None:
                mask_t = bos_mask[:, t]
                if bool(mask_t.any()):
                    decay = torch.where(mask_t, decay_on, decay_off).view(B, 1, 1)
                    state = state * decay
                    h = h * decay.view(B, 1)
                    if vault_active and vault_ring is not None:
                        vault_ring = vault_ring * decay

            anchor_clicks = 0

            # Per-step magnitude accumulator for adaptive decay.
            h_abs_sum_step = 0.0
            h_abs_count_step = 0

            # Guard against NaN/Inf pointer values before any indexing.
            ptr_float = torch.nan_to_num(ptr_float, nan=0.0, posinf=float(ring_range - 1), neginf=0.0)

            # Adaptive decay (kept as in legacy excerpt).
            if float(STATE_DECAY) < 1.0:
                decay_base = min(max(float(STATE_DECAY), 0.0), 1.0)
                decay_base = max(0.9, decay_base - min(0.1, (self.ring_len / 2000.0)))
                h_mag_step = 0.0
                if h_abs_count_step > 0:
                    h_mag_step = h_abs_sum_step / max(h_abs_count_step, 1)
                target = 0.3
                k = 0.5
                extra = k * max(0.0, h_mag_step - target)
                decay_val = max(0.9, min(1.0, decay_base - extra))
                decay_vec = active_mask.to(state.dtype) * decay_val + (~active_mask).to(state.dtype)
                state = state * decay_vec.view(B, 1, 1)

            # Input projection.
            if self.sensory_enabled:
                assert sensory_seq is not None
                assert self.sensory_bridge is not None
                inp = self.sensory_bridge(sensory_seq[:, t, :])
            else:
                inp = self.input_proj(x[:, t, :])

            # Vault inject on BOS.
            if vault_active and bos_mask is not None and vault_ring is not None and vault_ptr is not None:
                mask_t = bos_mask[:, t]
                if bool(mask_t.any()):
                    idx = torch.arange(B, device=device)
                    read_idx = (vault_ptr - 1) % int(self.vault_len)
                    read_vec = vault_ring[idx, read_idx]
                    gate = float(self.vault_gate if self.vault_adapt else self.vault_inject_scale)
                    assert self.vault_up is not None
                    inp = inp + self.vault_up(read_vec).to(inp.dtype) * gate
                    vault_injections += int(mask_t.sum().item())

            inp = self._apply_activation(inp)
            nan_guard("inp", inp, t)

            # Kernel weights around pointer.
            pos_idx, weights, _ = self._compute_kernel_weights(ptr_float, offsets, ring_range)
            nan_guard("weights", weights, t)

            # Gather neighborhood.
            pos_idx_exp = pos_idx.unsqueeze(-1).expand(-1, -1, self.slot_dim).clamp(0, ring_range - 1)
            neigh = state.gather(1, pos_idx_exp)  # [B,2K+1,D]
            cur = (weights.unsqueeze(-1) * neigh.to(weights.dtype)).sum(dim=1)

            # Mobius phase embedding.
            if self.mobius and self.phase_embed is not None:
                theta = (ptr_float / float(self.ring_len)) * math.pi
                phase_cos = torch.cos(theta).unsqueeze(1)
                phase_sin = torch.sin(theta).unsqueeze(1)
                cur = cur + phase_cos * self.phase_embed[0] + phase_sin * self.phase_embed[1]

            if cur.dtype != inp.dtype:
                cur = cur.to(inp.dtype)

            # Feed ring context as an additive cue.
            context_scale = torch.sigmoid(self.context_logit).to(inp.dtype)
            gru_in = inp if self.aux_ring else (inp + context_scale * cur)
            nan_guard("gru_in", gru_in, t)

            prev_h = h

            # Gate saturation telemetry.
            if collect_xray:
                r_gate, z_gate, n_gate = self._compute_gru_gates(gru_in, prev_h)
                sat = (
                    (r_gate < 0.05)
                    | (r_gate > 0.95)
                    | (z_gate < 0.05)
                    | (z_gate > 0.95)
                    | (n_gate.abs() > 0.95)
                ).float()
                gate_sat_count += float(sat.sum().item())
                gate_sat_total += float(sat.numel())

            h_new = self.gru(gru_in, prev_h)

            # Think ring step.
            if think_active and think_ring is not None and think_ptr is not None:
                assert self.think_proj_in is not None
                assert self.think_gru is not None
                assert self.think_proj_out is not None
                idx = torch.arange(B, device=device)
                think_inp = self.think_proj_in(gru_in)

                def _think_step(ring: torch.Tensor, ptr: torch.Tensor, alpha: float):
                    prev = ring[idx, ptr]
                    st = self.think_gru(think_inp, prev)
                    if st.dtype != ring.dtype:
                        st = st.to(ring.dtype)

                    # Optional EMA writeback so alpha behaves like a time constant.
                    if self.think_write_ema or think_dual:
                        a = float(alpha)
                        if a < 0.0:
                            a = 0.0
                        elif a > 1.0:
                            a = 1.0
                        write = (1.0 - a) * prev + a * st
                    else:
                        write = st

                    if bool(active_mask.any()):
                        ring[idx[active_mask], ptr[active_mask]] = write[active_mask]
                        ptr = (ptr + active_mask.to(torch.long)) % int(self.think_len)

                    delta = self.think_proj_out(write)
                    if delta.dtype != h_new.dtype:
                        delta = delta.to(h_new.dtype)
                    return ring, ptr, delta

                if think_dual and think_ring2 is not None and think_ptr2 is not None:
                    mix = float(getattr(self, "think_mix", 0.5))
                    if self.think_brainstem and self.brainstem is not None:
                        if (t % int(self.think_brainstem_every)) == 0:
                            entropy_step = self._compute_step_entropy(ptr_int, active_mask)
                            w_shield, info = self.brainstem.update(
                                current_entropy=float(entropy_step),
                                repetition_count=0,
                                dt=float(self.think_brainstem_dt),
                            )
                            mix = 1.0 - float(w_shield)
                            self.think_brainstem_w = float(w_shield)
                            self.think_brainstem_info = info
                        else:
                            mix = float(getattr(self, "think_brainstem_mix", mix))
                    mix = max(0.0, min(1.0, mix))
                    self.think_brainstem_mix = float(mix)

                    think_ring, think_ptr, delta_fast = _think_step(think_ring, think_ptr, float(self.think_alpha_high))
                    think_ring2, think_ptr2, delta_slow = _think_step(think_ring2, think_ptr2, float(self.think_alpha_low))
                    think_delta = (mix * delta_slow) + ((1.0 - mix) * delta_fast)
                else:
                    think_ring, think_ptr, think_delta = _think_step(think_ring, think_ptr, float(self.think_alpha))

                if str(self.think_mode) == "replace":
                    h_new = think_delta
                else:
                    h_new = h_new + float(self.think_alpha) * think_delta

            # Freeze hidden for inactive samples.
            upd = torch.where(active_mask.unsqueeze(1), h_new, prev_h)
            h = upd
            nan_guard("upd", upd, t)

            # Vault write on EOS.
            if vault_active and eos_mask is not None and vault_ring is not None and vault_ptr is not None:
                mask_t = eos_mask[:, t]
                if bool(mask_t.any()):
                    idx = torch.arange(B, device=device)
                    assert self.vault_down is not None
                    write_vec = self.vault_down(upd)
                    if write_vec.dtype != vault_ring.dtype:
                        write_vec = write_vec.to(vault_ring.dtype)
                    vault_ring[idx[mask_t], vault_ptr[mask_t]] = write_vec[mask_t]
                    vault_ptr = (vault_ptr + mask_t.to(torch.long)) % int(self.vault_len)
                    vault_updates += int(mask_t.sum().item())

            if collect_xray:
                h_abs_sum += float(upd.abs().sum().item())
                h_abs_count += int(upd.numel())
                h_abs_sum_step += float(upd.abs().sum().item())
                h_abs_count_step += int(upd.numel())

            # State loop metrics (mode sequence).
            if self.state_loop_metrics and (t % int(self.state_loop_every) == 0):
                loop_active = active_mask[:loop_samples]
                proj = upd[:loop_samples] @ self.state_loop_proj.to(device)
                mode = torch.argmax(proj, dim=1)
                mode_counts += torch.bincount(mode, minlength=int(self.state_loop_dim))
                if int(mode_prev[0].item()) == -1:
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

            # Scatter-add updates (same weights).
            upd_exp = upd.unsqueeze(1).expand(-1, weights.size(1), -1)
            contrib = (weights.unsqueeze(-1) * upd_exp).to(state.dtype)
            scale = float(self.update_scale)
            if scale != 1.0:
                contrib = contrib * scale
            contrib = contrib * active_mask.view(B, 1, 1).to(contrib.dtype)

            state_dtype = state.dtype
            if state_dtype in (torch.float16, torch.bfloat16):
                state_fp32 = state.float()
                contrib_fp32 = contrib.float()
                state_fp32 = state_fp32.scatter_add(1, pos_idx_exp, contrib_fp32)
                state = state_fp32.to(state_dtype)
            else:
                state = state.scatter_add(1, pos_idx_exp, contrib)

            if float(STATE_CLIP) > 0.0:
                state = state.clamp(-float(STATE_CLIP), float(STATE_CLIP))

            # Optional x-ray marker path.
            if collect_xray and x.size(-1) > 1 and target_mask is not None:
                marker = x[:, t, 1] > 0.5
                if bool(marker.any()):
                    token_idx = torch.full((B,), int(t % ring_range), device=device, dtype=torch.long)
                    target_mask.scatter_add_(1, token_idx.view(B, 1), marker.float().unsqueeze(1))
                    ptr_float = token_idx.to(ptr_dtype)

                # Baseline inertia override (compat).
                base_inertia = float(getattr(self, "ptr_inertia_base", self.ptr_inertia))
                inertia_override = os.environ.get("VRX_PTR_INERTIA_OVERRIDE")
                if inertia_override is not None:
                    try:
                        base_inertia = float(inertia_override)
                    except Exception:
                        pass
                self.ptr_inertia = float(base_inertia)

                # Attention mass after any warp.
                offsets_post = offsets.to(ptr_float.dtype)
                pos_idx_post, weights_post, _ = self._compute_kernel_weights(ptr_float, offsets_post, ring_range)
                step_attn = (weights_post * target_mask.gather(1, pos_idx_post)).sum(dim=1)
                attn_max = torch.maximum(attn_max, step_attn)

            prev_ptr = ptr_float

            # Read pointer (pre-update) so reads align with the location just written.
            ptr_read_phys = torch.remainder(prev_ptr, float(ring_range))

            if self.time_pointer:
                ptr_read_int = torch.full((B,), int(t % ring_range), device=device, dtype=torch.long)
            elif self.ptr_phantom and prev_ptr_int is not None:
                ptr_read_base = torch.floor(ptr_read_phys)
                ptr_read_off = torch.floor(torch.remainder(ptr_read_phys + float(self.ptr_phantom_off), float(ring_range)))
                read_agree = ptr_read_base == ptr_read_off
                ptr_read_int = torch.where(read_agree, ptr_read_base, prev_ptr_int.float())
            else:
                ptr_read_int = torch.floor(ptr_read_phys)

            ptr_read_int = torch.clamp(ptr_read_int, 0, ring_range - 1).long()
            if self.ptr_phantom_read:
                ptr_read_phys = ptr_read_int.float()

            # Pointer update gating.
            update_allowed = (t % int(self.ptr_update_every)) == 0
            if self.ptr_gate_mode == "steps" and self.ptr_gate_steps:
                update_allowed = update_allowed and (t in self.ptr_gate_steps)

            self.ptr_update_allowed = bool(update_allowed)
            self.ptr_update_blocked = bool(
                self.ptr_lock
                or self.time_pointer
                or (not update_allowed)
                or (int(self.ptr_warmup_steps) > 0 and t < int(self.ptr_warmup_steps))
            )

            jump_p = None
            move_mask = None
            gate = None
            delta_pre = None

            if self.time_pointer:
                ptr_float = torch.full((B,), float(t % ring_range), device=device, dtype=ptr_dtype)
            elif self.ptr_lock or (not update_allowed):
                ptr_float = prev_ptr
            elif int(self.ptr_warmup_steps) > 0 and t < int(self.ptr_warmup_steps):
                ptr_float = prev_ptr
            else:
                # Adaptive controls from the current update vector.
                inertia_use = torch.full((B,), float(self.ptr_inertia), device=device, dtype=ptr_dtype)
                deadzone_use = torch.full((B,), float(self.ptr_deadzone), device=device, dtype=ptr_dtype)
                walk_use = torch.full((B,), float(self.ptr_walk_prob), device=device, dtype=ptr_dtype)

                # Respect manual steering override: disable neural heads.
                if os.environ.get("VRX_PTR_INERTIA_OVERRIDE") is None:
                    inertia_use = torch.sigmoid(self.inertia_head(upd)).squeeze(1).to(ptr_dtype)
                    deadzone_use = F.softplus(self.deadzone_head(upd)).squeeze(1).to(ptr_dtype)
                    walk_use = torch.sigmoid(self.walk_head(upd)).squeeze(1).to(ptr_dtype)

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

                theta_ptr, theta_gate = self._gather_params(ptr_float)
                jump_logits = self.jump_score(upd).squeeze(1) + theta_gate
                nan_guard("jump_logits", jump_logits, t)

                p = torch.sigmoid(jump_logits)
                if 0.0 < float(PTR_JUMP_CAP) < 1.0:
                    p = torch.clamp(p, max=float(PTR_JUMP_CAP))
                if bool(PTR_JUMP_DISABLED):
                    p = torch.zeros_like(p)

                jump_p = p

                # Straight-through estimator for target (continuous).
                target_cont = theta_ptr
                if self.ptr_no_round:
                    target_ste = target_cont
                else:
                    target_ste = (target_cont.round() - target_cont).detach() + target_cont

                walk_ptr = torch.remainder(ptr_float + 1.0, float(ring_range))
                stay_ptr = prev_ptr
                non_jump_ptr = self.circ_lerp(stay_ptr, walk_ptr, walk_use, ring_range)

                ptr_float = self.circ_lerp(non_jump_ptr, target_ste, p, ring_range)

                # Raw (pre-inertia) velocity.
                ptr_float_pre = torch.where(active_mask, ptr_float, prev_ptr)
                delta_pre = self.wrap_delta(prev_ptr, ptr_float_pre, ring_range)
                raw_movement_cost = raw_movement_cost + delta_pre.abs().mean()

                # Inertia (stay-bias).
                if bool((inertia_use > 0.0).any()):
                    ptr_float = self.circ_lerp(prev_ptr, ptr_float, 1.0 - inertia_use, ring_range)

                # Deadzone with smooth mask.
                if bool((deadzone_use > 0.0).any()):
                    delta_raw = self.wrap_delta(prev_ptr, ptr_float, ring_range)
                    tau = max(float(self.ptr_deadzone_tau), 1e-6)
                    move_mask = torch.sigmoid((delta_raw.abs() - deadzone_use) / tau)
                    ptr_float = torch.remainder(prev_ptr + move_mask * delta_raw, float(ring_range))

                # Velocity governor.
                if self.ptr_vel_enabled:
                    delta_to_target = self.wrap_delta(prev_ptr, ptr_float, ring_range)
                    scale = max(float(self.ptr_vel_scale), 1e-6)
                    torque = torch.tanh(delta_to_target / scale) * float(self.ptr_vel_cap)
                    ptr_vel = float(self.ptr_vel_decay) * ptr_vel + (1.0 - float(self.ptr_vel_decay)) * torque
                    ptr_float = prev_ptr + ptr_vel

            # Optional learned soft gate for pointer update strength.
            if self.ptr_soft_gate and self.gate_head is not None:
                gate = torch.sigmoid(self.gate_head(upd)).squeeze(1)
                delta_gate = self.wrap_delta(prev_ptr, ptr_float, ring_range)
                ptr_float = torch.remainder(prev_ptr + gate * delta_gate, float(ring_range))

            # Dual-pointer annealing: accumulate residuals and click anchor on min_step lattice.
            delta_step = self.wrap_delta(prev_ptr, ptr_float, ring_range)
            delta_step = delta_step * active_mask.to(delta_step.dtype)
            ptr_residual = ptr_residual + delta_step.to(ptr_residual.dtype)

            if bool(PTR_ANCHOR_CLICK_INJECT) and t == 0:
                ptr_residual = ptr_residual + min_step

            if confidence >= float(PTR_ANCHOR_CONF_MIN):
                step_units = torch.floor(ptr_residual.abs() / min_step) * torch.sign(ptr_residual)
                step_units = torch.clamp(step_units, -1.0, 1.0)
                click_mask = step_units != 0
                if bool(click_mask.any()):
                    anchor_clicks = int(click_mask.sum().item())
                    ptr_anchor = torch.remainder(ptr_anchor + step_units * min_step, float(ring_range))
                    ptr_residual = ptr_residual - step_units * min_step

            ptr_float = torch.remainder(ptr_anchor + ptr_residual, float(ring_range))
            ptr_residual = self.wrap_delta(ptr_anchor, ptr_float, ring_range).to(ptr_residual.dtype)
            ptr_float = ptr_float.to(ptr_dtype)
            self.ptr_anchor_clicks = anchor_clicks

            # When pointer is blocked, keep anchor/residual consistent.
            if self.ptr_lock or self.time_pointer or (not update_allowed) or (int(self.ptr_warmup_steps) > 0 and t < int(self.ptr_warmup_steps)):
                ptr_anchor = torch.remainder(torch.round(ptr_float.to(torch.float64) / min_step) * min_step, float(ring_range))
                ptr_residual = self.wrap_delta(ptr_anchor, ptr_float.to(torch.float64), ring_range)
                if self.ptr_vel_enabled:
                    ptr_vel = torch.where(active_mask, ptr_vel, torch.zeros_like(ptr_vel))
                ptr_float = torch.where(active_mask, ptr_float, prev_ptr)

            # Hard clamp for safety.
            ptr_float = torch.nan_to_num(ptr_float, nan=0.0, posinf=float(ring_range - 1), neginf=0.0)
            ptr_float = torch.remainder(ptr_float, float(ring_range))
            nan_guard("ptr_float", ptr_float, t)

            # Movement cost (wrap-aware).
            delta = torch.remainder(ptr_float - prev_ptr + float(ring_range) / 2.0, float(ring_range)) - float(ring_range) / 2.0
            movement_cost = movement_cost + delta.abs().mean()

            # Update history tensorized: prepend read ptr, drop last.
            ptr_float_phys = torch.remainder(ptr_float, float(ring_range))
            ptr_base = torch.floor(ptr_float_phys)
            if self.ptr_phantom and prev_ptr_int is not None:
                ptr_off = torch.floor(torch.remainder(ptr_float_phys + float(self.ptr_phantom_off), float(ring_range)))
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

            # Debug stats (opt-in).
            if bool(DEBUG_STATS) and (int(DEBUG_EVERY) <= 0 or (t % int(DEBUG_EVERY) == 0)):
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
                if float(self.ptr_edge_eps) > 0.0:
                    eps = float(self.ptr_edge_eps)
                    edge_mask = (ptr_float_phys < eps) | (ptr_float_phys > (float(ring_range) - eps))
                    stats["ptr_edge_rate"] = float(edge_mask.float().mean().item())
                if think_dual and self.think_brainstem:
                    info = self.think_brainstem_info or {}
                    stats["brainstem_w_shield"] = float(getattr(self, "think_brainstem_w", 0.0))
                    stats["brainstem_mix_slow"] = float(getattr(self, "think_brainstem_mix", 0.0))
                    stats["brainstem_mode"] = str(info.get("mode", ""))
                    stats["brainstem_danger"] = float(info.get("danger", 0.0))
                stats["ptr_kernel"] = str(self.ptr_kernel)
                self.debug_stats = stats

            last_ptrs = torch.cat([ptr_read_int.view(B, 1), last_ptrs[:, :-1]], dim=1)

            bins = torch.bucketize(ptr_int.float(), self.bin_edges.to(device)) - 1
            bins = bins.clamp(0, self.pointer_hist_bins - 1)

            if bool(active_mask.any()):
                active_bins = bins[active_mask]
                step_counts = torch.bincount(active_bins, minlength=self.pointer_hist_bins)
            else:
                step_counts = torch.zeros_like(hist)
            hist = hist + step_counts

            # Pointer trace metrics (only count active samples).
            if prev_ptr_int is None:
                prev_ptr_int = ptr_int
                prev_prev_ptr_int = ptr_int
                dwell_len = torch.where(active_mask, torch.ones_like(dwell_len), dwell_len)
                max_dwell = torch.maximum(max_dwell, dwell_len)
            else:
                flip = active_mask & (ptr_int != prev_ptr_int)
                flip_count = flip_count + flip.long()
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

            # Optional auto-adjust pointer update cadence.
            if self.ptr_update_auto and (t % int(self.ptr_update_every_step) == 0) and total_active_steps > 0:
                flip_rate = float(flip_count.sum().item() / max(1, total_active_steps))
                if self.ptr_update_ema_state is None:
                    ema = flip_rate
                else:
                    ema = float(self.ptr_update_ema) * float(self.ptr_update_ema_state) + (1.0 - float(self.ptr_update_ema)) * flip_rate
                self.ptr_update_ema_state = ema
                if ema > float(self.ptr_update_target_flip):
                    self.ptr_update_every = min(int(self.ptr_update_max), int(self.ptr_update_every) + 1)
                elif ema < float(self.ptr_update_target_flip) * 0.5:
                    self.ptr_update_every = max(int(self.ptr_update_min), int(self.ptr_update_every) - 1)

            # Satiety check (optionally soft slice readout).
            if self.soft_readout:
                k = int(self.soft_readout_k)
                offsets_sr = torch.arange(-k, k + 1, device=device, dtype=ptr_float_phys.dtype)
                pos_idx_sr, w_sr, _ = self._compute_kernel_weights(
                    ptr_read_phys, offsets_sr, ring_range, tau_override=float(self.soft_readout_tau)
                )
                pos_idx_exp_sr = pos_idx_sr.unsqueeze(-1).expand(-1, -1, self.slot_dim)
                gathered_sr = state.gather(1, pos_idx_exp_sr)
                fused_sr = (w_sr.unsqueeze(-1) * gathered_sr.to(w_sr.dtype)).sum(dim=1)
                if fused_sr.dtype != state.dtype:
                    fused_sr = fused_sr.to(state.dtype)
                fused = fused_sr
            else:
                gather_idx = last_ptrs.clamp(0, ring_range - 1)
                gather_idx_exp = gather_idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
                gathered = state.gather(1, gather_idx_exp)
                fused = gathered.mean(dim=1)

            if fused.dtype != h.dtype:
                fused = fused.to(h.dtype)

            read_vec = fused + h
            expert_ids = self._map_expert_ids(ptr_int)
            logits_step = self.head(read_vec, expert_ids)
            nan_guard("logits_step", logits_step, t)

            if satiety_enabled:
                probs = torch.softmax(logits_step, dim=1)
                confident = probs.max(dim=1).values > float(SATIETY_THRESH)
                satiety_exited = satiety_exited | confident

            # Final readout for return value: by default, read current ptr bin.
            if self.soft_readout:
                logits = logits_step
            else:
                gather_idx2 = ptr_int.clamp(0, ring_range - 1).unsqueeze(1).unsqueeze(2)  # [B,1,1]
                gather_idx2_exp = gather_idx2.expand(-1, 1, self.slot_dim)
                fused2 = state.gather(1, gather_idx2_exp).squeeze(1)
                if fused2.dtype != h.dtype:
                    fused2 = fused2.to(h.dtype)
                read_vec2 = fused2 + h
                logits = self.head(read_vec2, expert_ids)
                nan_guard("logits_final_step", logits, t)

        # ---------------------------------
        # End loop: finalize telemetry
        # ---------------------------------
        self.pointer_hist = hist.detach().cpu()
        self.satiety_exits = int(satiety_exited.sum().item())

        last_bins = torch.bucketize(ptr_int.float(), self.bin_edges.to(device)) - 1
        last_bins = last_bins.clamp(0, self.pointer_hist_bins - 1).detach().cpu()
        self.last_ptr_bins = last_bins
        self.last_ptr_int = ptr_int.detach().cpu()
        self._update_expert_stats(self._map_expert_ids(ptr_int))

        denom = max(1, total_active_steps)
        self.ptr_flip_rate = float(flip_count.sum().item()) / denom
        self.ptr_pingpong_rate = float(pingpong_count.sum().item()) / denom
        self.ptr_max_dwell = int(max_dwell.max().item()) if max_dwell.numel() else 0
        mean_dwell = active_steps_per_sample.float() / (flip_count.float() + 1.0)
        self.ptr_mean_dwell = float(mean_dwell.mean().item()) if mean_dwell.numel() else 0.0

        if self.state_loop_metrics:
            mode_denom = max(1, mode_steps)
            self.state_loop_flip_rate = float(mode_flip.sum().item()) / mode_denom
            self.state_loop_abab_rate = float(mode_abab.sum().item()) / mode_denom
            self.state_loop_max_dwell = int(mode_max_dwell.max().item()) if mode_max_dwell.numel() else 0
            self.state_loop_mean_dwell = float(mode_dwell.float().mean().item()) if mode_dwell.numel() else 0.0
            if int(mode_counts.sum().item()) > 0:
                probs = mode_counts.float() / mode_counts.sum()
                ent = -(probs * torch.log(probs + 1e-12)).sum() / math.log(2.0)
                self.state_loop_entropy = float(ent.item())
            else:
                self.state_loop_entropy = None

        steps_used = max(1, t + 1 if T > 0 else 1)

        if vault_active:
            self.vault_inj_rate = float(vault_injections / max(1, T))
            self.vault_updates = int(vault_updates)
        else:
            self.vault_inj_rate = 0.0
            self.vault_updates = 0

        move_penalty = movement_cost / steps_used
        self.ptr_delta_abs_mean = float(move_penalty) if not isinstance(move_penalty, torch.Tensor) else float(move_penalty.item())
        raw_move_penalty = raw_movement_cost / steps_used
        self.ptr_delta_raw_mean = float(raw_move_penalty) if not isinstance(raw_move_penalty, torch.Tensor) else float(raw_move_penalty.item())

        # Ensure tensor output for move_penalty.
        if not isinstance(move_penalty, torch.Tensor):
            move_penalty = torch.tensor(float(move_penalty), device=device, dtype=x.dtype)

        if collect_xray:
            xray: Dict[str, float] = {}
            if target_mask is not None:
                try:
                    attn_mass = float(attn_max.mean().item())
                    xray["attn_mass"] = attn_mass
                except Exception:
                    pass
            if gate_sat_total > 0:
                xray["gate_sat"] = float(gate_sat_count / gate_sat_total)
            if h_abs_count > 0:
                xray["h_mag"] = float(h_abs_sum / h_abs_count)
            xray["ptr_delta_abs_mean"] = float(self.ptr_delta_abs_mean)
            xray["ptr_delta_raw_mean"] = float(self.ptr_delta_raw_mean)

            raw = getattr(self, "ptr_delta_raw_mean", None)
            gs = getattr(self, "ground_speed", None)
            if raw is not None and gs is not None and math.isfinite(float(raw)) and math.isfinite(float(gs)):
                xray["damp_ratio"] = float(raw) / max(float(gs), 1e-6)

            if ctrl_inertia_mean is not None:
                xray["ptr_inertia_dyn"] = float(ctrl_inertia_mean)
            if ctrl_deadzone_mean is not None:
                xray["ptr_deadzone_dyn"] = float(ctrl_deadzone_mean)
            if ctrl_walk_mean is not None:
                xray["ptr_walk_dyn"] = float(ctrl_walk_mean)

            return logits, move_penalty, xray

        return logits, move_penalty

