"""Behavior locks for :class:`vraxion.instnct.controls.CadenceGovernor`.

Ported from a legacy pytest suite to stdlib-only ``unittest``.

These tests pin:
- warmup behavior
- velocity/gradient short-circuiting
- flip/laminar cadence dynamics
"""

from __future__ import annotations

import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from vraxion.instnct.controls import CadenceGovernor


class CadenceGovernorTests(unittest.TestCase):
    def test_cadence_governor_warmup_returns_start_tau(self) -> None:
        govobj = CadenceGovernor(
            start_tau=5.0,
            warmup_steps=2,
            min_tau=1,
            max_tau=10,
            ema=0.5,
            target_flip=0.1,
            grad_high=10.0,
            grad_low=0.1,
            loss_flat=0.001,
            loss_spike=0.5,
            step_up=1.0,
            step_down=1.0,
            vel_high=0.5,
        )

        outone = govobj.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0, ptr_velocity=0.0)
        outtwo = govobj.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0, ptr_velocity=0.0)
        self.assertEqual(outone, 5)
        self.assertEqual(outtwo, 5)

    def test_cadence_governor_velocity_and_grad_short_circuit(self) -> None:
        govobj = CadenceGovernor(
            start_tau=5.0,
            warmup_steps=0,
            min_tau=1,
            max_tau=10,
            ema=0.5,
            target_flip=0.1,
            grad_high=2.0,
            grad_low=0.1,
            loss_flat=0.001,
            loss_spike=0.5,
            step_up=1.0,
            step_down=1.0,
            vel_high=0.5,
        )

        # Velocity high => force min_tau.
        outone = govobj.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0, ptr_velocity=1.0)
        self.assertEqual(outone, 1)

        # Grad norm high => force max_tau.
        outtwo = govobj.update(loss_value=1.0, grad_norm=100.0, flip_rate=0.0, ptr_velocity=0.0)
        self.assertEqual(outtwo, 10)

    def test_cadence_governor_increases_on_high_flip_and_decreases_on_laminar(self) -> None:
        govone = CadenceGovernor(
            start_tau=5.0,
            warmup_steps=0,
            min_tau=1,
            max_tau=10,
            ema=0.0,  # make EMA fully follow the current value for deterministic tests
            target_flip=0.1,
            grad_high=10.0,
            grad_low=0.1,
            loss_flat=0.001,
            loss_spike=0.5,
            step_up=1.0,
            step_down=1.0,
            vel_high=0.5,
        )

        # High flip => step_up.
        tauone = govone.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.9, ptr_velocity=0.0)
        self.assertEqual(tauone, 6)

        # Create a new governor to test laminar speed-up without history.
        govtwo = CadenceGovernor(
            start_tau=5.0,
            warmup_steps=0,
            min_tau=1,
            max_tau=10,
            ema=0.0,
            target_flip=0.2,
            grad_high=10.0,
            grad_low=0.1,
            loss_flat=0.001,
            loss_spike=0.5,
            step_up=1.0,
            step_down=1.0,
            vel_high=0.5,
        )

        # First call seeds prev_loss and EMAs.
        govtwo.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0, ptr_velocity=0.0)
        # Second call with same loss => abs(loss_delta)=0, laminar => step_down.
        tautwo = govtwo.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0, ptr_velocity=0.0)
        self.assertEqual(tautwo, 3)


if __name__ == "__main__":
    unittest.main()
