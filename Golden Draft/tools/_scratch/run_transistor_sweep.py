#!/usr/bin/env python
"""
Deterministic sweep runner for the transistor-style NAND suite (logic_nand).

Why a script:
- Avoid brittle inline PowerShell / quoting.
- Run a small matrix of configs and print a compact PASS/FAIL table.

This script only launches short runs via the existing boot runner:
Golden Draft/benchmarks/boot_synth_assoc_byte/run_boot_synth_assoc_byte.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
BOOT_DIR = REPO_ROOT / "Golden Draft" / "benchmarks" / "boot_synth_assoc_byte"
BOOT_PY = BOOT_DIR / "run_boot_synth_assoc_byte.py"
BENCH_ROOT = REPO_ROOT / "bench_vault" / "benchmarks"


@dataclass(frozen=True)
class RunCfg:
    n: int
    out_dim: int
    slot_dim: int
    ring_len: int
    seq_len: int
    mode: str
    shared: int
    seed: int
    steps: int
    eval_samples: int
    max_samples: int
    lr: float

    def tag(self) -> str:
        return (
            f"logic_nand_n{self.n}_od{self.out_dim}_sd{self.slot_dim}"
            f"_rl{self.ring_len}_sl{self.seq_len}_m{self.mode}_sh{self.shared}"
            f"_seed{self.seed}_st{self.steps}_en{self.eval_samples}"
        )


def _run(cmd: List[str], cwd: Path) -> None:
    # Keep output streaming to console for visibility.
    p = subprocess.Popen(cmd, cwd=str(cwd), env=os.environ.copy())
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {' '.join(cmd)}")


def _latest_run_dir(tag: str) -> Path:
    # The boot runner may append extra suffixes to the tag directory
    # (e.g. "_prismion_hallway_bank_n1_mean"). Match by prefix.
    candidates = [p for p in BENCH_ROOT.glob(f"{tag}*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(str(BENCH_ROOT / tag))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    base = candidates[0]

    # run_root naming is timestamped; pick newest directory.
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"no run dirs under: {base}")
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_report(run_root: Path) -> Dict[str, Any]:
    rep_path = run_root / "report.json"
    if not rep_path.exists():
        raise FileNotFoundError(str(rep_path))
    rep = _read_json(rep_path)
    eval_sum = rep.get("eval") or {}
    extra = rep.get("eval_extra") or {}
    out: Dict[str, Any] = {
        "run_root": str(run_root),
        "checkpoint_step": int(rep.get("train", {}).get("steps") or rep.get("checkpoint_step") or 0),
        # Report schema uses eval_acc/eval_loss keys.
        "eval_acc": float(eval_sum.get("eval_acc") or 0.0),
        "eval_loss": float(eval_sum.get("eval_loss") or 0.0),
        "pass_strict": bool(extra.get("pass_strict") or False),
        "min_case_acc": float(extra.get("min_case_acc") or 0.0),
        "min_p_true": float(extra.get("min_p_true") or 0.0),
        "min_margin": float(extra.get("min_margin") or 0.0),
    }
    return out


def _format_row(cfg: RunCfg, s: Dict[str, Any]) -> str:
    # Small, stable, grep-friendly line.
    status = "PASS" if s["pass_strict"] else "FAIL"
    return (
        f"{status} "
        f"n={cfg.n} od={cfg.out_dim} sd={cfg.slot_dim} rl={cfg.ring_len} sl={cfg.seq_len} "
        f"mode={cfg.mode} sh={cfg.shared} seed={cfg.seed} "
        f"step={s['checkpoint_step']} acc={s['eval_acc']:.4f} "
        f"minP={s['min_p_true']:.3f} minM={s['min_margin']:.3f} "
        f"run_root={s['run_root']}"
    )


def _build_cmd(cfg: RunCfg) -> List[str]:
    return [
        sys.executable,
        "-u",
        str(BOOT_PY),
        "--synth-mode",
        "logic_nand",
        "--model",
        "prismion_hallway_bank",
        "--prismn-n",
        str(cfg.n),
        "--prismn-mode",
        cfg.mode,
        "--prismn-shared",
        str(cfg.shared),
        "--prismn-out-dim",
        str(cfg.out_dim),
        "--slot-dim",
        str(cfg.slot_dim),
        "--ring-len",
        str(cfg.ring_len),
        "--seq-len",
        str(cfg.seq_len),
        "--steps",
        str(cfg.steps),
        "--max-samples",
        str(cfg.max_samples),
        "--eval-samples",
        str(cfg.eval_samples),
        "--seed",
        str(cfg.seed),
        "--lr",
        str(cfg.lr),
        "--device",
        "cpu",
        "--eval-disjoint",
        "--tag",
        cfg.tag(),
    ]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Run only the N=1 slot_dim sweep (4→1).")
    p.add_argument("--seed", type=int, default=126)
    p.add_argument(
        "--slot-dims",
        type=str,
        default="",
        help="Comma list override for slot_dim sweep (e.g. '1,2,3'). Default is '4,3,2,1'.",
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--eval-samples", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=512)
    p.add_argument("--ring-len", type=int, default=7)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-3)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if not BOOT_PY.exists():
        raise SystemExit(f"boot runner not found: {BOOT_PY}")
    if not BENCH_ROOT.exists():
        raise SystemExit(f"bench root not found: {BENCH_ROOT}")

    # Sweep 1: N=1, out_dim=1, slot_dim down to 1.
    if str(args.slot_dims).strip():
        slot_dims = [int(x) for x in str(args.slot_dims).split(",") if x.strip()]
    else:
        slot_dims = [4, 3, 2, 1]

    cfgs: List[RunCfg] = []
    for sd in slot_dims:
        cfgs.append(
            RunCfg(
                n=1,
                out_dim=1,
                slot_dim=sd,
                ring_len=int(args.ring_len),
                seq_len=int(args.seq_len),
                mode="mean",
                shared=1,
                seed=int(args.seed),
                steps=int(args.steps),
                eval_samples=int(args.eval_samples),
                max_samples=int(args.max_samples),
                lr=float(args.lr),
            )
        )

    # Sweep 2: small “more ants / smaller ants” grid (kept tiny on purpose).
    if not args.quick:
        for n in (2, 4, 8):
            for sd in (1, 2):
                cfgs.append(
                    RunCfg(
                        n=n,
                        out_dim=1,
                        slot_dim=sd,
                        ring_len=int(args.ring_len),
                        seq_len=int(args.seq_len),
                        mode="mosaic",
                        shared=1,
                        seed=int(args.seed),
                        steps=int(args.steps),
                        eval_samples=int(args.eval_samples),
                        max_samples=int(args.max_samples),
                        lr=float(args.lr),
                    )
                )

    print(f"[sweep] count={len(cfgs)} bench_root={BENCH_ROOT}")
    t0 = time.time()
    results: List[Tuple[RunCfg, Dict[str, Any]]] = []
    for i, cfg in enumerate(cfgs, start=1):
        print(f"[sweep] ({i}/{len(cfgs)}) START {cfg.tag()}")
        cmd = _build_cmd(cfg)
        _run(cmd, cwd=BOOT_DIR)
        run_root = _latest_run_dir(cfg.tag())
        s = _summarize_report(run_root)
        results.append((cfg, s))
        print(_format_row(cfg, s))

    dt = time.time() - t0
    print(f"[sweep] done_s={dt:.1f}")

    # Compact PASS frontier by (slot_dim, n).
    print("[frontier] passing configs (sorted by cost proxy: n*slot_dim):")
    passing = [(cfg, s) for (cfg, s) in results if s["pass_strict"]]
    passing.sort(key=lambda cs: (cs[0].n * cs[0].slot_dim, cs[0].slot_dim, cs[0].n))
    for cfg, s in passing:
        cost = cfg.n * cfg.slot_dim
        print(f"PASS cost={cost:3d} n={cfg.n} sd={cfg.slot_dim} od={cfg.out_dim} mode={cfg.mode} run_root={s['run_root']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
