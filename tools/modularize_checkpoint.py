#!/usr/bin/env python
"""Explode a monolithic VRAXION checkpoint into a modular directory layout."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def _infer_num_experts(state: dict) -> int:
    max_idx = -1
    prefix = "head.experts."
    for key in state.keys():
        if key.startswith(prefix):
            try:
                idx = int(key[len(prefix):].split(".", 1)[0])
            except ValueError:
                continue
            max_idx = max(max_idx, idx)
    return max_idx + 1


def _split_model_state(state_dict: dict) -> tuple[dict, dict]:
    core = {}
    experts = {}
    prefix = "head.experts."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            rest = key[len(prefix):]
            parts = rest.split(".", 1)
            if len(parts) != 2:
                core[key] = value
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                core[key] = value
                continue
            experts.setdefault(idx, {})[parts[1]] = value.detach().cpu()
        else:
            core[key] = value.detach().cpu()
    return core, experts


def main() -> int:
    parser = argparse.ArgumentParser(description="Explode a monolithic checkpoint into modular format.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--output", required=True, help="Output directory for modular checkpoint")
    parser.add_argument("--tenure-all", action="store_true", help="Mark all experts as tenured")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.output)
    if not os.path.exists(ckpt_path):
        print(f"[modularize] checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    if os.path.exists(out_dir) and os.listdir(out_dir):
        if not args.force:
            print(f"[modularize] output directory not empty: {out_dir}", file=sys.stderr)
            return 3

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    core_state, expert_states = _split_model_state(state)
    num_experts = ckpt.get("num_experts") or _infer_num_experts(state)

    system_dir = os.path.join(out_dir, "system")
    experts_dir = os.path.join(out_dir, "experts")
    os.makedirs(system_dir, exist_ok=True)
    os.makedirs(experts_dir, exist_ok=True)

    payload = {
        "model": core_state,
        "optim": ckpt.get("optim"),
        "scaler": ckpt.get("scaler"),
        "step": ckpt.get("step", 0),
        "losses": ckpt.get("losses", []),
        "update_scale": ckpt.get("update_scale"),
        "ptr_inertia": ckpt.get("ptr_inertia"),
        "ptr_inertia_ema": ckpt.get("ptr_inertia_ema"),
        "ptr_inertia_floor": ckpt.get("ptr_inertia_floor"),
        "agc_scale_max": ckpt.get("agc_scale_max"),
        "ground_speed_ema": ckpt.get("ground_speed_ema"),
        "ground_speed_limit": ckpt.get("ground_speed_limit"),
        "num_experts": int(num_experts),
        "param_names": ckpt.get("param_names"),
    }
    router_path = os.path.join(system_dir, "router.state")
    torch.save(payload, router_path)

    experts_meta = []
    for idx in range(int(num_experts)):
        state = expert_states.get(idx)
        if state is None:
            continue
        expert_path = os.path.join(experts_dir, f"expert_{idx:03d}.pt")
        torch.save(state, expert_path)
        experts_meta.append(
            {
                "id": idx,
                "tenured": bool(args.tenure_all),
                "created_step": int(payload.get("step", 0)),
                "last_used_step": int(payload.get("step", 0)),
                "contrib": 0.0,
            }
        )

    meta_path = os.path.join(experts_dir, "meta.json")
    meta = {
        "source_checkpoint": ckpt_path,
        "num_experts": int(num_experts),
        "experts": experts_meta,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[modularize] wrote router.state to {router_path}")
    print(f"[modularize] wrote {len(experts_meta)} experts to {experts_dir}")
    print(f"[modularize] wrote meta to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
