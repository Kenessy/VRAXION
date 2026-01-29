"""Automatic gain control (AGC) for INSTNCT.

This is a behavior-preserving extraction of the AGC logic from the original
kernel. The public API is re-exported by :mod:`vraxion.instnct.controls`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional


def _clamp(valnum: float, lowlim: float, hilimx: float) -> float:
    # Use min/max to match the original NaN behavior.
    return max(lowlim, min(hilimx, valnum))


@dataclass(frozen=True)
class AGCParams:
    """Parameters for :func:`apply_update_agc`."""

    enabled: bool

    grad_low: float
    grad_high: float

    scale_up: float
    scale_down: float

    scale_min: float
    scale_max_default: float

    warmup_steps: int
    warmup_init: float


def apply_update_agc(
    model,
    grad_norm: float | None,
    params: AGCParams,
    *,
    raw_delta: float | None = None,
    step: int | None = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> float:
    """Auto-gain control for the state update scale.

    Behavior-preserving extraction of the legacy monolith's ``apply_update_agc``.

    The function mutates:
      - ``model.update_scale``
      - ``model.agc_scale_cap``
      - ``model.debug_scale_out``

    Args:
        model: object with ``update_scale`` attribute (and optionally
            ``agc_scale_max``/``agc_scale_cap``).
        grad_norm: gradient norm signal.
        params: AGC parameters.
        raw_delta: kept for API parity (not used).
        step: global optimizer step.
        log_fn: optional logger callback.

    Returns:
        The new ``update_scale``.
    """

    # NOTE: locals follow the 6-char convention; args keep callsite names.
    # raw_delta is kept for callsite parity.

    bascap = float(getattr(model, "agc_scale_max", params.scale_max_default))
    capval = float(getattr(model, "agc_scale_cap", bascap))
    if (not math.isfinite(capval)) or capval <= 0:
        capval = bascap
    capval = _clamp(capval, params.scale_min, bascap)

    # Warmup floor: ramp linearly to scale_min over warmup_steps from warmup_init.
    if step is not None:
        warhor = max(1, int(params.warmup_steps))
        warmup = _clamp(step / float(warhor), 0.0, 1.0)
        flrval = params.warmup_init + (params.scale_min - params.warmup_init) * warmup
        flrval = _clamp(flrval, 0.0, params.scale_min)
    else:
        flrval = params.scale_min

    scaval = float(getattr(model, "update_scale", flrval))
    if (not math.isfinite(scaval)) or scaval <= 0:
        scaval = flrval
    if step is not None and step == 0:
        scaval = flrval

    if params.enabled and grad_norm is not None and math.isfinite(float(grad_norm)):
        if grad_norm < params.grad_low:
            scaval *= params.scale_up
        elif grad_norm > params.grad_high:
            scaval *= params.scale_down

    scaval = _clamp(scaval, flrval, capval)
    model.agc_scale_cap = capval
    model.update_scale = scaval
    model.debug_scale_out = scaval

    if step is not None and step == 0 and log_fn is not None:
        dbgdat = {
            "scale_in": scaval,
            "scale_out": scaval,
            "agc_scale_min": params.scale_min,
            "warmup_floor": flrval,
            "cap": capval,
            "base_cap": bascap,
        }
        log_fn(f"[debug_scale_step0] {dbgdat}")

    return scaval
