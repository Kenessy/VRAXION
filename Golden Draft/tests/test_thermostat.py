"""Behavior locks for :func:`vraxion.instnct.controls.apply_thermostat`.

Ported from a legacy pytest suite to stdlib-only ``unittest``.

These tests pin:
- discrete step adjustments
- continuous (focus, tension) targeting path
- env override short-circuiting
"""

from __future__ import annotations

import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from vraxion.instnct.controls import ThermostatParams, apply_thermostat


class DummyModel:
    def __init__(self, ptr_inertia: float = 0.5, ptr_deadzone: float = 0.25, ptr_walk_prob: float = 0.1):
        self.ptr_inertia = ptr_inertia
        self.ptr_deadzone = ptr_deadzone
        self.ptr_walk_prob = ptr_walk_prob


class ThermostatTests(unittest.TestCase):
    def test_apply_thermostat_discrete_steps_adjust_values(self) -> None:
        parobj = ThermostatParams(
            ema_beta=0.0,
            target_flip=0.1,
            inertia_step=0.1,
            deadzone_step=0.05,
            walk_step=0.02,
            inertia_min=0.0,
            inertia_max=1.0,
            deadzone_min=0.0,
            deadzone_max=1.0,
            walk_min=0.0,
            walk_max=0.5,
        )

        modobj = DummyModel(ptr_inertia=0.5, ptr_deadzone=0.25, ptr_walk_prob=0.1)

        # High flip -> increase inertia & deadzone, decrease walk.
        with conftest.temporary_env(VRX_PTR_INERTIA_OVERRIDE=None):
            emaout = apply_thermostat(modobj, flip_rate=0.9, ema=None, params=parobj)
        self.assertAlmostEqual(emaout, 0.9, places=12)
        self.assertAlmostEqual(modobj.ptr_inertia, 0.6, places=12)
        self.assertAlmostEqual(modobj.ptr_deadzone, 0.3, places=12)
        self.assertAlmostEqual(modobj.ptr_walk_prob, 0.08, places=12)

        # Very low flip -> decrease inertia/deadzone, increase walk.
        with conftest.temporary_env(VRX_PTR_INERTIA_OVERRIDE=None):
            emaout = apply_thermostat(modobj, flip_rate=0.0, ema=emaout, params=parobj)
        self.assertAlmostEqual(modobj.ptr_inertia, 0.5, places=12)
        self.assertAlmostEqual(modobj.ptr_deadzone, 0.25, places=12)
        self.assertAlmostEqual(modobj.ptr_walk_prob, 0.1, places=12)

    def test_apply_thermostat_continuous_focus_tension_targets(self) -> None:
        parobj = ThermostatParams(
            ema_beta=0.9,
            target_flip=0.1,
            inertia_step=0.1,
            deadzone_step=0.05,
            walk_step=0.02,
            inertia_min=0.0,
            inertia_max=1.0,
            deadzone_min=0.0,
            deadzone_max=1.0,
            walk_min=0.0,
            walk_max=0.5,
        )

        modobj = DummyModel(ptr_inertia=0.0, ptr_deadzone=1.0, ptr_walk_prob=0.5)

        # focus=1, tension=0 => target_inertia=1, target_deadzone=0,
        # target_walk=walk_min.
        with conftest.temporary_env(VRX_PTR_INERTIA_OVERRIDE=None):
            emaout = apply_thermostat(
                modobj,
                flip_rate=0.0,
                ema=0.0,
                params=parobj,
                focus=1.0,
                tension=0.0,
            )

        blndvl = max(1e-3, 1.0 - parobj.ema_beta)
        self.assertAlmostEqual(modobj.ptr_inertia, 0.0 + (1.0 - 0.0) * blndvl, places=12)
        self.assertAlmostEqual(modobj.ptr_deadzone, 1.0 + (0.0 - 1.0) * blndvl, places=12)
        self.assertAlmostEqual(modobj.ptr_walk_prob, 0.5 + (0.0 - 0.5) * blndvl, places=12)
        self.assertAlmostEqual(emaout, 0.0, places=12)

    def test_apply_thermostat_respects_env_inertia_override(self) -> None:
        parobj = ThermostatParams(
            ema_beta=0.0,
            target_flip=0.1,
            inertia_step=0.1,
            deadzone_step=0.05,
            walk_step=0.02,
            inertia_min=0.0,
            inertia_max=1.0,
            deadzone_min=0.0,
            deadzone_max=1.0,
            walk_min=0.0,
            walk_max=0.5,
        )

        modobj = DummyModel(ptr_inertia=0.5, ptr_deadzone=0.25, ptr_walk_prob=0.1)

        with conftest.temporary_env(VRX_PTR_INERTIA_OVERRIDE="0.33"):
            emaout = apply_thermostat(modobj, flip_rate=0.9, ema=None, params=parobj)

        # No mutations when override present.
        self.assertAlmostEqual(modobj.ptr_inertia, 0.5, places=12)
        self.assertAlmostEqual(modobj.ptr_deadzone, 0.25, places=12)
        self.assertAlmostEqual(modobj.ptr_walk_prob, 0.1, places=12)
        self.assertAlmostEqual(emaout, 0.9, places=12)


if __name__ == "__main__":
    unittest.main()

