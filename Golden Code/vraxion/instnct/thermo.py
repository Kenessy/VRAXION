"""Thermostat (adaptive pointer control) for INSTNCT.

The public API is re-exported by :mod:`vraxion.instnct.controls`.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


OVRENV = "VRX_PTR_INERTIA_OVERRIDE"
BLNMIN = 1e-3


def _clp01(valnum: float) -> float:
    """Clamp ``valnum`` to the closed interval [0.0, 1.0]."""

    # Use min/max to match the original NaN behavior.
    return max(0.0, min(1.0, valnum))


@dataclass(frozen=True)
class ThermostatParams:
    """Parameters for :func:`apply_thermostat`.

    Names match their meaning in the original kernel:
    - ``ema_beta``: EMA coefficient for flip-rate smoothing.
    - ``target_flip``: target flip-rate.
    - ``*_step``: discrete step sizes (fallback mode).
    - ``*_min``/``*_max``: clamp bounds.
    """

    ema_beta: float
    target_flip: float

    inertia_step: float
    deadzone_step: float
    walk_step: float

    inertia_min: float
    inertia_max: float

    deadzone_min: float
    deadzone_max: float

    walk_min: float
    walk_max: float


def apply_thermostat(
    model,
    flip_rate: float,
    ema: float | None,
    params: ThermostatParams,
    *,
    focus: float | None = None,
    tension: float | None = None,
    raw_delta: float | None = None,
) -> float:
    """Adaptive pointer control: reduce flapping without freezing forever.

    Behavior matches the original implementation in the legacy monolith.

    Returns:
        The updated EMA of ``flip_rate``.
    """

    betval = params.ema_beta
    if ema is None:
        emaval = flip_rate
    else:
        emaval = (betval * ema) + ((1.0 - betval) * flip_rate)

    # Respect manual inertia override: do not mutate ptr_inertia/deadzone/walk.
    if os.environ.get(OVRENV) is not None:
        return emaval

    if focus is not None and tension is not None:
        focval = _clp01(float(focus))
        tenval = _clp01(float(tension))

        stkval = 0.0
        if raw_delta is not None:
            try:
                delval = float(raw_delta)
            except (TypeError, ValueError):
                delval = None
            if delval is not None and math.isfinite(delval):
                stkval = 1.0 / (1.0 + max(0.0, delval))

        drvval = max(tenval, 1.0 - focval, stkval)

        trginr = params.inertia_min + (params.inertia_max - params.inertia_min) * (focval * (1.0 - tenval))
        trgdea = params.deadzone_min + (params.deadzone_max - params.deadzone_min) * tenval
        trgwlk = params.walk_min + (params.walk_max - params.walk_min) * drvval

        blnval = max(BLNMIN, 1.0 - betval)
        model.ptr_inertia = model.ptr_inertia + (trginr - model.ptr_inertia) * blnval
        model.ptr_deadzone = model.ptr_deadzone + (trgdea - model.ptr_deadzone) * blnval
        model.ptr_walk_prob = model.ptr_walk_prob + (trgwlk - model.ptr_walk_prob) * blnval
        return emaval

    if emaval > params.target_flip:
        model.ptr_inertia = min(params.inertia_max, model.ptr_inertia + params.inertia_step)
        model.ptr_deadzone = min(params.deadzone_max, model.ptr_deadzone + params.deadzone_step)
        model.ptr_walk_prob = max(params.walk_min, model.ptr_walk_prob - params.walk_step)
    elif emaval < params.target_flip * 0.5:
        model.ptr_inertia = max(params.inertia_min, model.ptr_inertia - params.inertia_step)
        model.ptr_deadzone = max(params.deadzone_min, model.ptr_deadzone - params.deadzone_step)
        model.ptr_walk_prob = min(params.walk_max, model.ptr_walk_prob + params.walk_step)

    return emaval

