#!/usr/bin/env python
"""VRAXION offline pruning (expert merge).

This CLI tool removes the highest-index expert from a checkpoint by:

1) Remapping ``router_map`` entries that point to the removed expert into a
   specified "kept" expert.
2) Deleting the removed expert's tensors from the model state dict.
3) Best-effort trimming of optimizer state and ``param_names`` entries.

Safety / contract:
- Only allows pruning the *highest* expert id to keep expert ids dense.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch

from ._checkpoint_io import (
    atomic_torch_save,
    expert_param_keys,
    infer_num_experts,
    safe_torch_load,
)


def _drop_optimizer_entries(
    optim: dict | None,
    param_names: list[str] | None,
    drop_names: set[str],
) -> tuple[dict | None, list[str] | None]:
    """Best-effort removal of optimizer state + param_names for dropped params.

    Contract:
    - Always trims ``param_names`` when present.
    - Trims optimizer state only when we can confidently map names -> param ids.
    """

    if not drop_names:
        return optim, param_names

    # Always trim param_names when present (even if optimizer is absent).
    pnamls = None
    if isinstance(param_names, list):
        pnamls = [namstr for namstr in param_names if namstr not in drop_names]
    else:
        pnamls = param_names

    if not isinstance(optim, dict) or not isinstance(param_names, list):
        return optim, pnamls

    pgrpsx = optim.get("param_groups")
    if not isinstance(pgrpsx, list) or not pgrpsx:
        return optim, pnamls

    # Gather all param ids across groups.
    allpid: list[int] = []
    for grpobj in pgrpsx:
        if not isinstance(grpobj, dict):
            continue
        params = grpobj.get("params", [])
        if not isinstance(params, list):
            continue
        for pidval in params:
            if isinstance(pidval, int):
                allpid.append(pidval)

    if not allpid:
        return optim, pnamls

    dropid: set[int] = set()
    uniqid = sorted(set(allpid))

    # Convention A: pid == index into param_names (common in some setups).
    if uniqid and uniqid[0] == 0 and uniqid[-1] == len(param_names) - 1:
        for idxval, namstr in enumerate(param_names):
            if namstr in drop_names:
                dropid.add(idxval)
    else:
        # Convention B: param_names aligns with param_groups[0]['params'] order.
        firgrp = pgrpsx[0] if isinstance(pgrpsx[0], dict) else {}
        firlis = firgrp.get("params", []) if isinstance(firgrp, dict) else []
        if isinstance(firlis, list):
            nam2id = {namstr: pidval for namstr, pidval in zip(param_names, firlis) if isinstance(pidval, int)}
            for namstr in drop_names:
                pidval = nam2id.get(namstr)
                if isinstance(pidval, int):
                    dropid.add(pidval)

    if not dropid:
        return optim, pnamls

    # Trim param ids from all groups.
    for grpobj in pgrpsx:
        if not isinstance(grpobj, dict):
            continue
        params = grpobj.get("params", [])
        if not isinstance(params, list):
            continue
        grpobj["params"] = [pidval for pidval in params if pidval not in dropid]

    # Trim optimizer state by pid.
    stmap = optim.get("state")
    if isinstance(stmap, dict):
        for pidval in dropid:
            stmap.pop(pidval, None)
        optim["state"] = stmap

    optim["param_groups"] = pgrpsx
    return optim, pnamls


def main() -> int:
    parser = argparse.ArgumentParser(description="VRAXION prune: merge redundant experts offline.")
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Output checkpoint (.pt)")
    parser.add_argument("--merge-from", type=int, required=True, help="Expert id to remove")
    parser.add_argument("--merge-into", type=int, required=True, help="Expert id to keep")
    argobj = parser.parse_args()

    ckppth = os.path.abspath(argobj.checkpoint)
    outpth = os.path.abspath(argobj.output)
    mrgfrm = int(argobj.merge_from)
    mrgint = int(argobj.merge_into)

    ckptob = safe_torch_load(ckppth)
    if not isinstance(ckptob, dict):
        raise TypeError("Checkpoint payload must be a dict")
    if "model" not in ckptob:
        raise KeyError("Checkpoint missing 'model' state dict")

    statem = ckptob["model"]
    if not isinstance(statem, dict):
        raise TypeError("Checkpoint 'model' must be a state dict (dict)")

    numexp = ckptob.get("num_experts")
    if numexp is None:
        numexp = infer_num_experts(statem)
    numexp = int(numexp)

    if numexp <= 1:
        raise ValueError("Cannot prune with <= 1 expert")
    if mrgfrm != numexp - 1:
        raise ValueError(f"merge-from {mrgfrm} must be the highest expert id ({numexp - 1})")
    if mrgfrm == mrgint:
        raise ValueError("merge-from and merge-into must be different")
    if mrgint < 0 or mrgint >= numexp:
        raise ValueError(f"merge-into {mrgint} must be within [0, {numexp - 1}]")

    if "router_map" not in statem:
        raise KeyError("Checkpoint missing router_map; cannot prune")

    rtmapx = torch.as_tensor(statem["router_map"], dtype=torch.long).clone()
    rtmapx[rtmapx == mrgfrm] = mrgint
    statem["router_map"] = rtmapx

    drpkey = set(expert_param_keys(statem, mrgfrm))
    if not drpkey:
        raise ValueError(f"No parameters found for expert {mrgfrm}")
    for keystr in drpkey:
        statem.pop(keystr, None)

    ckptob["model"] = statem
    ckptob["num_experts"] = numexp - 1
    ckptob["prune"] = {
        "merge_from": mrgfrm,
        "merge_into": mrgint,
        "router_remap": f"{mrgfrm}->{mrgint}",
    }

    optimx = ckptob.get("optim")
    pnames = ckptob.get("param_names")
    optimx, pnames = _drop_optimizer_entries(optimx, pnames, drpkey)

    if optimx is not None:
        ckptob["optim"] = optimx
    if pnames is not None:
        ckptob["param_names"] = pnames

    atomic_torch_save(ckptob, outpth)

    print(
        f"[prune] merged expert {mrgfrm} -> {mrgint}, "
        f"removed {len(drpkey)} tensors, new experts={ckptob['num_experts']}, "
        f"saved {outpth}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
