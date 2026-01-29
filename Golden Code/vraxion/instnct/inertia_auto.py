"""Pointer inertia auto-tuning for INSTNCT."""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(valnum: float, lowlim: float, hilimx: float) -> float:
    # Use min/max to match the original NaN behavior.
    return max(lowlim, min(hilimx, valnum))


@dataclass(frozen=True)
class InertiaAutoParams:
    """Parameters for :func:`apply_inertia_auto`."""

    enabled: bool

    inertia_min: float
    inertia_max: float

    vel_full: float
    ema_beta: float

    dwell_enabled: bool
    dwell_thresh: float


def apply_inertia_auto(model, ptr_velocity, params: InertiaAutoParams, *, panic_active: bool = False) -> None:
    """Update ``model.ptr_inertia`` using either dwell or pointer velocity signals."""

    if (not params.enabled) or panic_active:
        return

    # Dwell-driven kinetic tempering: glue when dwell is high, agile when low.
    if params.dwell_enabled:
        dwlraw = getattr(model, "ptr_mean_dwell", None)
        maxraw = getattr(model, "ptr_max_dwell", dwlraw)
        try:
            dwlval = float(dwlraw) if dwlraw is not None else 0.0
            maxval = float(maxraw) if maxraw is not None else dwlval
        except (TypeError, ValueError):
            dwlval, maxval = 0.0, 0.0

        dwlmet = max(dwlval, maxval)
        if params.dwell_thresh > 0:
            wgtval = _clamp(dwlmet / params.dwell_thresh, 0.0, 1.0)
            tarval = params.inertia_min + wgtval * (params.inertia_max - params.inertia_min)
        else:
            tarval = params.inertia_max
    else:
        if ptr_velocity is None or params.vel_full <= 0:
            return
        try:
            velval = float(ptr_velocity)
        except (TypeError, ValueError):
            return
        velval = max(0.0, velval)
        ratval = min(1.0, velval / params.vel_full)
        tarval = params.inertia_min + ratval * (params.inertia_max - params.inertia_min)

    emaval = float(getattr(model, "ptr_inertia_ema", model.ptr_inertia))
    betval = params.ema_beta
    emaval = (betval * emaval) + ((1.0 - betval) * tarval)
    emaval = _clamp(emaval, params.inertia_min, params.inertia_max)
    model.ptr_inertia_ema = emaval
    model.ptr_inertia = emaval
