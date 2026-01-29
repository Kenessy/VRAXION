"""Panic reflex for INSTNCT.

The public API is re-exported by :mod:`vraxion.instnct.controls`.
"""

from __future__ import annotations


MINEPS = 1e-6


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
        """Update internal loss EMA and return current panic controls."""

        if self.loss_ema is None:
            self.loss_ema = loss_value
            return {"status": "INIT", "inertia": self.inertia_high, "walk_prob": 0.0}

        ratval = loss_value / (self.loss_ema + MINEPS)
        if ratval > self.threshold:
            self.panic_state = 1.0
        else:
            self.panic_state = max(0.0, self.panic_state - self.recovery)

        self.loss_ema = (self.beta * self.loss_ema) + ((1.0 - self.beta) * loss_value)

        if self.panic_state > 0.1:
            invval = 1.0 - self.panic_state
            inrval = self.inertia_low + (self.inertia_high - self.inertia_low) * invval
            wlkval = self.walk_prob_max * self.panic_state
            return {"status": "PANIC", "inertia": inrval, "walk_prob": wlkval}

        return {"status": "LOCKED", "inertia": self.inertia_high, "walk_prob": 0.0}
