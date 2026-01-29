from __future__ import annotations

"""
Verifies that the production "Golden Code" controls module matches the locked
golden behavior contract.

This mirrors the locked contract used by the Golden Disk verifier, but keeps
the contract JSON in Golden Draft so verification does not depend on external
mirror folders.

Run from repo root:
  python tools/verify_golden_instnct_controls.py
"""

import json
import math
import os
import sys
from pathlib import Path


# Locked contract JSON is kept in Golden Draft so verification does not depend on
# external mirror folders.
CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"
EXPECTED = CONTRACTS_DIR / "instnct_controls_golden_expected.json"

# By default, verify against the production "Golden Code" tree if present.
# Override with VRAXION_GOLDEN_SRC=<path>.
DEFAULT_GOLDEN_SRC = Path(r"S:\AI\Golden Code")
_override = os.environ.get("VRAXION_GOLDEN_SRC")
GOLDEN_SRC = Path(_override).expanduser().resolve() if _override else DEFAULT_GOLDEN_SRC
if not GOLDEN_SRC.exists():
    raise SystemExit(f"Golden source path not found: {GOLDEN_SRC!s}")
sys.path.insert(0, str(GOLDEN_SRC))

from vraxion.instnct import controls as C  # noqa: E402


def _isclose(a: float, b: float, *, rel: float = 1e-9, abs_: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


class Model:
    def __init__(self, **kw):
        # Pointer controls
        self.ptr_inertia = kw.get("ptr_inertia", 0.0)
        self.ptr_deadzone = kw.get("ptr_deadzone", 0.0)
        self.ptr_walk_prob = kw.get("ptr_walk_prob", 0.0)

        # AGC
        self.update_scale = kw.get("update_scale", 0.01)
        self.agc_scale_max = kw.get("agc_scale_max", 1.0)
        self.agc_scale_cap = kw.get("agc_scale_cap", self.agc_scale_max)
        self.agc_scale_min = kw.get("agc_scale_min", 0.01)
        self.agc_scale_floor = kw.get("agc_scale_floor", 0.01)
        self.agc_warmup = kw.get("agc_warmup", 0)
        self.agc_decay = kw.get("agc_decay", 0.0)

        # Thermostat
        self.focus_ema = kw.get("focus_ema", 0.0)
        self.tension_ema = kw.get("tension_ema", 0.0)

        # Inertia auto
        self.ptr_mean_dwell = kw.get("ptr_mean_dwell", 0.0)
        self.ptr_max_dwell = kw.get("ptr_max_dwell", 0.0)
        self.ptr_inertia_ema = kw.get("ptr_inertia_ema", kw.get("ptr_inertia", 0.0))

        # Debug
        self.debug_scale_in = kw.get("debug_scale_in", None)
        self.debug_scale_out = kw.get("debug_scale_out", None)


def main() -> int:
    exp = json.loads(EXPECTED.read_text(encoding="utf-8"))
    fail = 0

    # --- facade sanity: keep pickling / introspection stable ---
    modnam = C.__name__
    for symnam in (
        "ThermostatParams",
        "AGCParams",
        "InertiaAutoParams",
        "PanicReflex",
        "CadenceGovernor",
        "apply_thermostat",
        "apply_update_agc",
        "apply_inertia_auto",
    ):
        symobj = getattr(C, symnam)
        if getattr(symobj, "__module__", modnam) != modnam:
            print(f"[FAIL] facade module mismatch {symnam} module={getattr(symobj, '__module__', None)} exp={modnam}")
            fail += 1

    # --- apply_thermostat ---
    tp = C.ThermostatParams(**exp["thermo"]["params"])
    for case in exp["thermo"]["cases"]:
        mdl = Model(**case["model_in"])
        ema_in = case["ema_in"]
        env = case.get("env") or {}
        old_env = dict(os.environ)
        try:
            os.environ.update({k: str(v) for k, v in env.items()})
            ema_out = C.apply_thermostat(
                mdl,
                case["flip_rate"],
                ema_in,
                tp,
                focus=case.get("focus"),
                tension=case.get("tension"),
                raw_delta=case.get("raw_delta"),
            )
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        if not _isclose(ema_out, case["expect"]["ema_out"]):
            print(f"[FAIL] thermostat ema case={case['id']} got={ema_out} exp={case['expect']['ema_out']}")
            fail += 1
        for k, vexp in case["expect"]["model_out"].items():
            vgot = getattr(mdl, k)
            if not _isclose(vgot, vexp):
                print(f"[FAIL] thermostat {k} case={case['id']} got={vgot} exp={vexp}")
                fail += 1

    # --- apply_update_agc ---
    ap = C.AGCParams(**exp["agc"]["params"])
    for case in exp["agc"]["cases"]:
        mdl = Model(**case["model_in"])

        logs: list[str] = []

        def _log(msg: str) -> None:
            logs.append(msg)

        out = C.apply_update_agc(
            mdl,
            case["grad_norm"],
            ap,
            step=case.get("step"),
            log_fn=_log,
        )
        # Invariant: the function returns exactly what it stores.
        if not (_isclose(out, mdl.update_scale) and _isclose(out, mdl.debug_scale_out)):
            print(f"[FAIL] agc invariant case={case['id']} out/update_scale/debug mismatch")
            fail += 1
        if not _isclose(out, case["expect"]["scale_out"]):
            print(f"[FAIL] agc scale case={case['id']} got={out} exp={case['expect']['scale_out']}")
            fail += 1
        for k, vexp in case["expect"]["model_out"].items():
            vgot = getattr(mdl, k)
            if not _isclose(vgot, vexp):
                print(f"[FAIL] agc {k} case={case['id']} got={vgot} exp={vexp}")
                fail += 1
        if case["expect"].get("log_contains"):
            needle = case["expect"]["log_contains"]
            if not any(needle in s for s in logs):
                print(f"[FAIL] agc log case={case['id']} missing '{needle}'")
                fail += 1

    # --- apply_inertia_auto ---
    ip = C.InertiaAutoParams(**exp["inrtao"]["params"])
    for case in exp["inrtao"]["cases"]:
        mdl = Model(**case["model_in"])
        C.apply_inertia_auto(mdl, case.get("ptr_velocity"), ip, panic_active=case.get("panic_active", False))
        for k, vexp in case["expect"]["model_out"].items():
            vgot = getattr(mdl, k)
            if not _isclose(vgot, vexp):
                print(f"[FAIL] inertia_auto {k} case={case['id']} got={vgot} exp={vexp}")
                fail += 1

    # --- PanicReflex ---
    pr_cfg = exp["panic"]["params"]
    pr = C.PanicReflex(**pr_cfg)
    for step in exp["panic"]["steps"]:
        out = pr.update(step["loss"])
        for k, vexp in step["expect"].items():
            vgot = out.get(k)
            if isinstance(vexp, (int, float)):
                if not _isclose(vgot, vexp, rel=1e-7, abs_=1e-7):
                    print(f"[FAIL] panic {k} step={step['id']} got={vgot} exp={vexp}")
                    fail += 1
            else:
                if vgot != vexp:
                    print(f"[FAIL] panic {k} step={step['id']} got={vgot} exp={vexp}")
                    fail += 1

    # --- extra: inertia_auto is a no-op when disabled ---
    ip_off = C.InertiaAutoParams(**{**exp["inrtao"]["params"], "enabled": False})
    mdl = Model(ptr_inertia=0.12, ptr_inertia_ema=0.34)
    C.apply_inertia_auto(mdl, 999.0, ip_off, panic_active=False)
    if not (_isclose(mdl.ptr_inertia, 0.12) and _isclose(mdl.ptr_inertia_ema, 0.34)):
        print("[FAIL] inertia_auto disabled mutated model")
        fail += 1

    # --- extra: PanicReflex is deterministic for a fixed loss sequence ---
    pr_a = C.PanicReflex(**pr_cfg)
    pr_b = C.PanicReflex(**pr_cfg)
    for step in exp["panic"]["steps"]:
        out_a = pr_a.update(step["loss"])
        out_b = pr_b.update(step["loss"])
        if out_a.get("status") != out_b.get("status"):
            print(f"[FAIL] panic deterministic mismatch step={step['id']}")
            fail += 1
        if not (
            _isclose(out_a.get("inertia", 0.0), out_b.get("inertia", 0.0), rel=1e-7, abs_=1e-7)
            and _isclose(out_a.get("walk_prob", 0.0), out_b.get("walk_prob", 0.0), rel=1e-7, abs_=1e-7)
        ):
            print(f"[FAIL] panic deterministic values mismatch step={step['id']}")
            fail += 1

    # --- extra: CadenceGovernor deterministic scenario (warmup + velocity + grad shock) ---
    gov = C.CadenceGovernor(
        start_tau=5,
        warmup_steps=2,
        min_tau=1,
        max_tau=10,
        ema=0.5,
        target_flip=0.2,
        grad_high=1.0,
        grad_low=0.1,
        loss_flat=0.01,
        loss_spike=0.5,
        step_up=1.0,
        step_down=1.0,
        vel_high=2.0,
    )
    seqdat = [
        (1.0, 0.0, 0.0, None),
        (0.9, 0.0, 0.0, None),
        (0.8, 0.0, 0.0, 3.0),
        (0.7, 2.0, 0.0, None),
        (0.71, 2.0, 0.0, None),
        (0.72, 2.0, 0.0, None),
    ]
    expout = [5, 5, 1, 10, 10, 10]
    gotout: list[int] = []
    for losval, grdval, flpval, velval in seqdat:
        gotout.append(gov.update(losval, grdval, flpval, ptr_velocity=velval))
    if gotout != expout:
        print(f"[FAIL] cadence deterministic scenario got={gotout} exp={expout}")
        fail += 1

    if fail:
        raise SystemExit(2)
    print("OK: Golden Code controls match golden_expected.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
