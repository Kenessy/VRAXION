#!/usr/bin/env python
"""VRAXION offline mitosis (expert split).

This CLI tool performs checkpoint-only expert "fission":

  - Clone all tensors under ``head.experts.<parent_id>.`` into a new, highest-id
    expert slot.
  - Redirect selected ring addresses in ``router_map`` to the new expert id.
  - If present, clone optimizer entries for the cloned parameters (using the
    lightweight layout exercised by ``tests/verify_golden.py``).

Behavior is validated by the golden verifier; keep semantics stable.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any

import torch

from tools._checkpoint_io import atomic_torch_save, infer_num_experts, safe_torch_load


def _parse_addresses(arg: str) -> list[int]:
    """Parse a comma-separated list of integer addresses."""

    addlst: list[int] = []
    for tokstr in arg.split(","):
        tokstr = tokstr.strip()
        if not tokstr:
            continue
        addlst.append(int(tokstr))

    if not addlst:
        raise ValueError("No target addresses provided.")

    return addlst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VRAXION mitosis: split an expert offline.")
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Output checkpoint (.pt)")
    parser.add_argument("--parent", type=int, help="Overloaded expert id to clone")
    parser.add_argument(
        "--addresses",
        help="Comma-separated ring addresses to redirect to the new expert",
    )
    parser.add_argument("--meta", help="Path to mitosis_meta.json for auto addresses/parent")
    parser.add_argument("--noise", type=float, default=1e-5, help="Noise scale for cloned weights")
    args = parser.parse_args(argv)

    ckppth = os.path.abspath(args.checkpoint)
    outpth = os.path.abspath(args.output)

    metpar = None
    metadd = None
    if args.meta:
        metpth = os.path.abspath(args.meta)
        with open(metpth, "r", encoding="utf-8") as filobj:
            metdat = json.load(filobj)
        metpar = metdat.get("parent_expert")
        metadd = metdat.get("hot_addresses")

    parent = args.parent if args.parent is not None else metpar

    addarg: str | None
    if args.addresses is not None:
        addarg = args.addresses
    elif metadd is not None:
        addarg = ",".join(str(a) for a in metadd)
    else:
        addarg = None

    if parent is None or addarg is None:
        raise ValueError("Parent expert and addresses are required (via --parent/--addresses or --meta).")

    addlst = _parse_addresses(addarg)

    ckpt = safe_torch_load(ckppth)
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint payload must be a dict")
    if "model" not in ckpt:
        raise KeyError("Checkpoint missing 'model' state dict")

    state = ckpt["model"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint 'model' must be a state dict (dict)")

    numexp = ckpt.get("num_experts")
    if numexp is None:
        numexp = infer_num_experts(state)
    numexp = int(numexp)
    if numexp <= 0:
        raise ValueError("Could not infer num_experts from checkpoint")

    paridx = int(parent)
    newidx = int(numexp)

    parpre = f"head.experts.{paridx}."
    newpre = f"head.experts.{newidx}."

    cloned = 0
    clnmap: list[tuple[str, str]] = []
    for keystr, valobj in list(state.items()):
        if not keystr.startswith(parpre):
            continue
        newkey = newpre + keystr[len(parpre) :]
        # Behavior-preserving: always consume RNG per tensor (even if noise==0).
        noise6 = torch.randn_like(valobj) * float(args.noise)
        state[newkey] = valobj.clone() + noise6
        clnmap.append((keystr, newkey))
        cloned += 1

    if cloned == 0:
        raise ValueError(f"No expert parameters found for parent {paridx}")

    if "router_map" not in state:
        raise KeyError("Checkpoint missing router_map; run a router-map build first.")

    rtmapx = state["router_map"].clone()
    maxadr = int(rtmapx.numel()) - 1
    for adrval in addlst:
        if adrval < 0 or adrval > maxadr:
            raise ValueError(f"Address {adrval} out of range [0, {maxadr}]")
        rtmapx[adrval] = newidx
    state["router_map"] = rtmapx

    ckpt["model"] = state
    ckpt["num_experts"] = newidx + 1
    ckpt["mitosis"] = {
        "parent": paridx,
        "child": newidx,
        "addresses": addlst,
        "noise": float(args.noise),
    }

    # Clone optimizer momentum if available and param names are present.
    pnames = ckpt.get("param_names")
    optimx = ckpt.get("optim")
    if optimx and pnames:
        pgrpsx = optimx.get("param_groups", [])
        if pgrpsx:
            group0 = pgrpsx[0]
            pidsix = list(group0.get("params", []))
            nam2id = {name: pid for name, pid in zip(pnames, pidsix)}

            nxtpid = (max(pidsix) + 1) if pidsix else 0
            ostmap = optimx.get("state", {})

            for parnam, chlnam in clnmap:
                parpid = nam2id.get(parnam)
                if parpid is None:
                    continue
                chnpid = nxtpid
                nxtpid += 1

                pidsix.append(chnpid)
                ostmap[chnpid] = copy.deepcopy(ostmap.get(parpid, {}))
                pnames.append(chlnam)

            group0["params"] = pidsix
            pgrpsx[0] = group0
            optimx["param_groups"] = pgrpsx
            optimx["state"] = ostmap
            ckpt["optim"] = optimx
            ckpt["param_names"] = pnames

    atomic_torch_save(ckpt, outpth)

    print(
        f"[mitosis] cloned expert {paridx} -> {newidx} "
        f"({cloned} tensors), redirected {len(addlst)} addresses -> {outpth}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
