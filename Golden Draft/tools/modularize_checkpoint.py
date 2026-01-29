#!/usr/bin/env python
"""Explode a monolithic checkpoint into a modular directory layout.

The modular layout mirrors the expert lifecycle format used by the VRAXION runner:

- ``<out>/system/router.state``: core model + training state
- ``<out>/experts/expert_###.pt``: per-expert tensors
- ``<out>/experts/meta.json``: lightweight metadata

Return codes (stable):
- 2: checkpoint missing
- 3: output directory not empty without --force
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from ._checkpoint_io import (
    atomic_json_dump,
    atomic_torch_save,
    infer_num_experts,
    safe_torch_load,
    split_model_state,
)


def _out_non_empty(outdir: str) -> bool:
    """Return True if ``outdir`` exists and is non-empty.

    For robustness, any error reading directory contents is treated as
    "non-empty" (cannot safely overwrite without --force).
    """

    if not os.path.exists(outdir):
        return False
    try:
        return bool(os.listdir(outdir))
    except OSError:
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Explode a monolithic checkpoint into modular format.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--output", required=True, help="Output directory for modular checkpoint")
    parser.add_argument("--tenure-all", action="store_true", help="Mark all experts as tenured")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    argobj = parser.parse_args()

    ckppth = os.path.abspath(argobj.checkpoint)
    outdir = os.path.abspath(argobj.output)

    if not os.path.exists(ckppth):
        print(f"[modularize] checkpoint not found: {ckppth}", file=sys.stderr)
        return 2

    # Keep behavior stable: only error if output dir is non-empty and --force not set.
    if _out_non_empty(outdir) and not argobj.force:
        print(f"[modularize] output directory not empty: {outdir}", file=sys.stderr)
        return 3

    ckptob = safe_torch_load(ckppth)

    # Support both full checkpoint dicts and raw state dicts.
    if isinstance(ckptob, dict):
        statem = ckptob.get("model", ckptob)
        numexp = ckptob.get("num_experts")
        optimx = ckptob.get("optim")
        scalrx = ckptob.get("scaler")
        stepix = ckptob.get("step", 0)
        loslst = ckptob.get("losses", [])
        updscx = ckptob.get("update_scale")
        ptrinr = ckptob.get("ptr_inertia")
        ptrema = ckptob.get("ptr_inertia_ema")
        ptrifl = ckptob.get("ptr_inertia_floor")
        agcmax = ckptob.get("agc_scale_max")
        gsema6 = ckptob.get("ground_speed_ema")
        gslim6 = ckptob.get("ground_speed_limit")
        pnames = ckptob.get("param_names")
    else:
        statem = ckptob
        numexp = None
        optimx = None
        scalrx = None
        stepix = 0
        loslst = []
        updscx = None
        ptrinr = None
        ptrema = None
        ptrifl = None
        agcmax = None
        gsema6 = None
        gslim6 = None
        pnames = None

    if not isinstance(statem, dict):
        raise TypeError("Checkpoint must be either a state_dict (dict) or contain a 'model' dict")

    core6x, expmap = split_model_state(statem)
    if numexp is None:
        numexp = infer_num_experts(statem)

    sysdir = os.path.join(outdir, "system")
    expdir = os.path.join(outdir, "experts")
    os.makedirs(sysdir, exist_ok=True)
    os.makedirs(expdir, exist_ok=True)

    # Keep insertion order stable.
    paylod: dict[str, Any] = {
        "model": core6x,
        "optim": optimx,
        "scaler": scalrx,
        "step": stepix,
        "losses": loslst,
        "update_scale": updscx,
        "ptr_inertia": ptrinr,
        "ptr_inertia_ema": ptrema,
        "ptr_inertia_floor": ptrifl,
        "agc_scale_max": agcmax,
        "ground_speed_ema": gsema6,
        "ground_speed_limit": gslim6,
        "num_experts": int(numexp),
        "param_names": pnames,
    }

    rtpath = os.path.join(sysdir, "router.state")
    atomic_torch_save(paylod, rtpath)

    metlst: list[dict[str, Any]] = []
    step6x = int(paylod.get("step", 0) or 0)

    for idxval in range(int(numexp)):
        expsta = expmap.get(idxval)
        if expsta is None:
            continue
        exppth = os.path.join(expdir, f"expert_{idxval:03d}.pt")
        atomic_torch_save(expsta, exppth)
        metlst.append(
            {
                "id": idxval,
                "tenured": bool(argobj.tenure_all),
                "created_step": step6x,
                "last_used_step": step6x,
                "contrib": 0.0,
            }
        )

    metpth = os.path.join(expdir, "meta.json")
    metdat = {
        "source_checkpoint": ckppth,
        "num_experts": int(numexp),
        "experts": metlst,
    }
    atomic_json_dump(metdat, metpth, indent=2)

    print(f"[modularize] wrote router.state to {rtpath}")
    print(f"[modularize] wrote {len(metlst)} experts to {expdir}")
    print(f"[modularize] wrote meta to {metpth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
