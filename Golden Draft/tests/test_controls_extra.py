"""Additional behavior locks for :mod:`vraxion.instnct.controls`.

The golden verifier covers a few representative cases. These unit tests add
coverage for edge cases and control-flow boundaries without changing behavior.
"""

from __future__ import annotations

import math
import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from vraxion.instnct import controls as C


class Model:
    def __init__(self, **kw):
        self.ptr_inertia = kw.get("ptr_inertia", 0.0)
        self.ptr_deadzone = kw.get("ptr_deadzone", 0.0)
        self.ptr_walk_prob = kw.get("ptr_walk_prob", 0.0)

        self.update_scale = kw.get("update_scale", 0.01)
        self.agc_scale_max = kw.get("agc_scale_max", 1.0)
        self.agc_scale_cap = kw.get("agc_scale_cap", self.agc_scale_max)
        self.debug_scale_out = kw.get("debug_scale_out", None)

        self.ptr_mean_dwell = kw.get("ptr_mean_dwell", 0.0)
        self.ptr_max_dwell = kw.get("ptr_max_dwell", 0.0)
        self.ptr_inertia_ema = kw.get("ptr_inertia_ema", kw.get("ptr_inertia", 0.0))


class ControlsExtraTests(unittest.TestCase):
    def test_inertia_auto_dwell_path(self) -> None:
        """Dwell-driven inertia path matches the legacy EMA behavior."""

        parobj = C.InertiaAutoParams(
            enabled=True,
            inertia_min=0.1,
            inertia_max=0.9,
            vel_full=1.0,
            ema_beta=0.5,
            dwell_enabled=True,
            dwell_thresh=10.0,
        )

        modobj = Model(ptr_inertia=0.2, ptr_inertia_ema=0.2, ptr_mean_dwell=0.0, ptr_max_dwell=0.0)
        C.apply_inertia_auto(modobj, ptr_velocity=None, params=parobj, panic_active=False)
        self.assertTrue(math.isclose(modobj.ptr_inertia, 0.15, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(modobj.ptr_inertia_ema, 0.15, rel_tol=0.0, abs_tol=1e-12))

        modobj.ptr_mean_dwell = 20.0
        modobj.ptr_max_dwell = 5.0
        C.apply_inertia_auto(modobj, ptr_velocity=None, params=parobj, panic_active=False)
        self.assertTrue(math.isclose(modobj.ptr_inertia, 0.525, rel_tol=0.0, abs_tol=1e-12))

    def test_thermostat_focus_ignored_without_tension(self) -> None:
        """Only the (focus, tension) pair triggers the continuous path."""

        parobj = C.ThermostatParams(
            ema_beta=0.9,
            target_flip=0.2,
            inertia_step=0.05,
            deadzone_step=0.02,
            walk_step=0.01,
            inertia_min=0.1,
            inertia_max=0.95,
            deadzone_min=0.0,
            deadzone_max=0.5,
            walk_min=0.0,
            walk_max=0.2,
        )
        modobj = Model(ptr_inertia=0.2, ptr_deadzone=0.01, ptr_walk_prob=0.2)

        with conftest.temporary_env(VRX_PTR_INERTIA_OVERRIDE=None):
            emaout = C.apply_thermostat(
                modobj,
                flip_rate=0.3,
                ema=0.4,
                params=parobj,
                focus=0.9,
                tension=None,
            )

        self.assertTrue(math.isclose(emaout, 0.39, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(modobj.ptr_inertia, 0.25, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(modobj.ptr_deadzone, 0.03, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(modobj.ptr_walk_prob, 0.19, rel_tol=0.0, abs_tol=1e-12))

    def test_agc_invalid_cap_resets_to_base(self) -> None:
        """Non-finite/invalid caps fall back to agc_scale_max then clamp."""

        parobj = C.AGCParams(
            enabled=True,
            grad_low=0.01,
            grad_high=0.2,
            scale_up=1.1,
            scale_down=0.9,
            scale_min=0.01,
            scale_max_default=1.0,
            warmup_steps=10,
            warmup_init=0.01,
        )
        modobj = Model(update_scale=0.5, agc_scale_max=1.0, agc_scale_cap=math.nan)

        outval = C.apply_update_agc(modobj, grad_norm=0.0, params=parobj, step=None)
        self.assertTrue(math.isclose(outval, 0.55, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(modobj.agc_scale_cap, 1.0, rel_tol=0.0, abs_tol=1e-12))

    def test_apply_update_agc_warmup_floor_and_logging(self) -> None:
        logs = []

        def log_fn(msg: str) -> None:
            logs.append(msg)

        parobj = C.AGCParams(
            enabled=True,
            grad_low=0.5,
            grad_high=2.0,
            scale_up=2.0,
            scale_down=0.5,
            scale_min=0.01,
            scale_max_default=1.0,
            warmup_steps=10,
            warmup_init=0.001,
        )

        modobj = Model(update_scale=123.0, agc_scale_max=1.0, agc_scale_cap=1.0)

        # step=0 forces scale to the warmup floor first, then applies AGC scaling.
        scale = C.apply_update_agc(modobj, grad_norm=0.0, params=parobj, step=0, log_fn=log_fn)
        self.assertTrue(math.isclose(scale, 0.002, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(modobj.update_scale, 0.002, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(modobj.debug_scale_out, 0.002, rel_tol=0.0, abs_tol=1e-12))
        self.assertTrue(any("[debug_scale_step0]" in s for s in logs))

    def test_apply_update_agc_scales_up_and_down_with_clamps(self) -> None:
        parobj = C.AGCParams(
            enabled=True,
            grad_low=1.0,
            grad_high=3.0,
            scale_up=2.0,
            scale_down=0.25,
            scale_min=0.01,
            scale_max_default=0.5,
            warmup_steps=0,
            warmup_init=0.01,
        )

        modobj = Model(update_scale=0.1, agc_scale_max=0.5, agc_scale_cap=0.5)

        # Low grad -> scale up
        s1 = C.apply_update_agc(modobj, grad_norm=0.5, params=parobj, step=1)
        self.assertTrue(math.isclose(s1, 0.2, rel_tol=0.0, abs_tol=1e-12))

        # High grad -> scale down
        s2 = C.apply_update_agc(modobj, grad_norm=10.0, params=parobj, step=2)
        self.assertTrue(math.isclose(s2, 0.05, rel_tol=0.0, abs_tol=1e-12))

        # Clamp to min floor
        modobj.update_scale = 1e-9
        s3 = C.apply_update_agc(modobj, grad_norm=10.0, params=parobj, step=3)
        self.assertGreaterEqual(s3, parobj.scale_min)

        # Clamp to cap
        modobj.update_scale = 100.0
        s4 = C.apply_update_agc(modobj, grad_norm=0.0, params=parobj, step=4)
        self.assertLessEqual(s4, modobj.agc_scale_cap)

    def test_cadence_warmup_is_noop(self) -> None:
        """Warmup should return current cadence without updating tau."""

        govobj = C.CadenceGovernor(
            start_tau=7.0,
            warmup_steps=3,
            min_tau=2,
            max_tau=10,
            ema=0.9,
            target_flip=0.2,
            grad_high=0.5,
            grad_low=0.1,
            loss_flat=0.01,
            loss_spike=0.2,
            step_up=1.0,
            step_down=1.0,
            vel_high=999.0,
        )

        outone = govobj.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0)
        outtwo = govobj.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0)
        outtre = govobj.update(loss_value=1.0, grad_norm=0.0, flip_rate=0.0)
        self.assertEqual(outone, 7)
        self.assertEqual(outtwo, 7)
        self.assertEqual(outtre, 7)


if __name__ == "__main__":
    unittest.main()

