"""Cadence governor for INSTNCT.

The cadence governor adapts the sampling period (tau) based on flip rate,
gradient norms, loss shocks, and (optionally) pointer velocity.

The public API is re-exported by :mod:`vraxion.instnct.controls`.
"""

from __future__ import annotations


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
        *,
        vel_high: float,
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
        """Update cadence and return the current integer tau."""

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
            losdel = 0.0
        else:
            losdel = self.prev_loss - loss_value
        self.prev_loss = loss_value

        # Slow down when turbulence is high or loss spikes.
        if self.flip_ema > self.target_flip or self.grad_ema > self.grad_high or losdel < -self.loss_spike:
            self.tau = min(self.max_tau, self.tau + self.step_up)
        # Speed up only when laminar and loss is flat.
        elif self.grad_ema < self.grad_low and self.flip_ema < self.target_flip * 0.5 and abs(losdel) < self.loss_flat:
            self.tau = max(self.min_tau, self.tau - self.step_down)

        self.tau = max(self.min_tau, min(self.max_tau, self.tau))
        return int(round(self.tau))
