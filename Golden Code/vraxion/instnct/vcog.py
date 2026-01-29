"""V_COG telemetry header governor.

Behavior-preserving extraction from the legacy monolithic training script.

Contract (do not break):
  - Keep the emitted header format stable: ``V_COG[...]``.
  - Keep math and rounding stable for regression tests.

The governor maintains EMAs/variances for a small set of telemetry signals and
formats a compact header string suitable for log prefixes.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional


def _clamp(valnum: float, minval: float, maxval: float) -> float:
    return max(minval, min(maxval, valnum))


def _to_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except Exception:
        return 0


def _to_flt(value: Any) -> float:
    try:
        return float(value) if value is not None else 0.0
    except Exception:
        return 0.0


class VCogGovernor:
    def __init__(self, id_target: float = 0.25, beta: float = 0.95, sigma_floor: float = 1e-4):
        self.id_target = id_target
        self.beta = beta
        self.sigma_floor = sigma_floor

        self.search_ema: Optional[float] = None
        self.search_var: float = 0.0
        self.loss_ema: Optional[float] = None
        self.loss_var: float = 0.0

    def update(self, telemetry: Mapping[str, Any]) -> str:
        """Update internal statistics and emit a stable ``V_COG[...]`` header."""

        search = float(telemetry.get("search", 0.0))
        losval = float(telemetry.get("loss", 0.0))
        evlacc = telemetry.get("eval_acc")

        # EMA + variance (order preserved).
        if self.search_ema is None:
            self.search_ema = search
        else:
            self.search_ema = self.beta * self.search_ema + (1.0 - self.beta) * search
        self.search_var = self.beta * self.search_var + (1.0 - self.beta) * (search - self.search_ema) ** 2

        if self.loss_ema is None:
            self.loss_ema = losval
        else:
            self.loss_ema = self.beta * self.loss_ema + (1.0 - self.beta) * losval
        self.loss_var = self.beta * self.loss_var + (1.0 - self.beta) * (losval - self.loss_ema) ** 2

        inrval = float(telemetry.get("inertia", 0.0))
        epival = float(telemetry.get("epi", 0.0))
        walkvl = float(telemetry.get("walk", 0.0))
        focval = float(telemetry.get("focus", 0.0))
        gripvl = _clamp(inrval * (1.0 - walkvl) * focval, 0.0, 1.0)

        lkvalx = gripvl * (1.0 - search)
        deltav = float(telemetry.get("delta", 0.0))
        drwval = float(telemetry.get("delta_raw", 0.0))
        mobval = _clamp(deltav / (drwval + 1e-9), 0.0, 1.0)
        floval = gripvl * mobval

        srstdv = max(math.sqrt(abs(self.search_var)), self.sigma_floor)
        szvalx = (search - self.search_ema) / (srstdv + self.sigma_floor)
        snapvl = max(0.0, szvalx) * gripvl

        lsstdv = max(math.sqrt(abs(self.loss_var)), self.sigma_floor)
        # loss_ema is set above.
        idnval = 1.0 / (1.0 + self.loss_ema * (1.0 + lsstdv))

        prgval: Optional[float] = None
        if evlacc is not None:
            try:
                accval = float(evlacc)
                prgval = _clamp(accval * 100.0, 0.0, 100.0)
            except Exception:
                prgval = None
        if prgval is None:
            prgval = _clamp(idnval / max(self.id_target, 1e-6), 0.0, 1.0) * 100.0

        orbval = _to_int(telemetry.get("orb"))
        rdvalx = _to_flt(telemetry.get("rd"))
        acvalx = _to_int(telemetry.get("ac"))
        vhvalx = _to_flt(telemetry.get("vh"))
        vuvalx = _to_int(telemetry.get("vu"))

        # Do not change formatting / spacing (hard requirement).
        header = (
            f"V_COG[PRGRS:{prgval:.1f}% ORB:{orbval} RD:{rdvalx:.2e} AC:{acvalx} VH:{vhvalx:.2f} VU:{vuvalx} EPI:{epival:.2f} "
            f"LOCKS:{lkvalx:.2f} FLOWS:{floval:.2f} "
            f"SNAPS:{snapvl:.2f} IDENT:{idnval:.3f}]"
        )
        return header
