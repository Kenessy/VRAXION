#!/usr/bin/env python3
"""Entry for supervised VRA-79 saturation runs.

This wrapper exists for 2 reasons:
1) Centralize the env var contract used by `Golden Draft/vraxion_run.py`.
2) Make runs reproducible and easy to parameter-sweep without editing files.

Artifacts are written under `<run_root>/train/` and are expected to be gitignored
when `<run_root>` lives under `bench_vault/_tmp/...`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
RUNNER = REPO_ROOT / r"Golden Draft\vraxion_run.py"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a supervised VRA-79 saturation job.")
    ap.add_argument("--run-root", required=True, help="Root dir for run artifacts.")

    # Compute / size knobs (the ones we want to A/B).
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--ring-len", type=int, required=True)
    ap.add_argument("--slot-dim", type=int, required=True)
    ap.add_argument("--expert-heads", type=int, default=1)
    ap.add_argument("--batch-size", type=int, required=True)

    # Stop conditions. Convention:
    # - max_steps=0 means "no limit" when ignore_max_steps=1.
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--ignore-max-steps", type=int, default=0, choices=[0, 1])
    ap.add_argument("--ignore-wall-clock", type=int, default=1, choices=[0, 1])

    # Evidence cadence.
    ap.add_argument("--save-every-steps", type=int, default=100)
    ap.add_argument("--save-last-good", type=int, default=1, choices=[0, 1])
    ap.add_argument("--save-history", type=int, default=0, choices=[0, 1])

    # Synthetic task settings (kept stable for apples-to-apples A/B).
    ap.add_argument("--synth-len", type=int, default=256)
    ap.add_argument("--assoc-keys", type=int, default=64)
    ap.add_argument("--assoc-pairs", type=int, default=4)
    ap.add_argument("--assoc-val-range", type=int, default=256)

    # Eval loader size (used by instnct runner to build the subset).
    ap.add_argument("--eval-samples", type=int, default=512)

    # Misc.
    ap.add_argument("--ptr-dtype", default="fp64", choices=["fp32", "fp64"])
    ap.add_argument("--resume", type=int, default=0, choices=[0, 1])
    ap.add_argument("--offline-only", type=int, default=1, choices=[0, 1])
    return ap.parse_args()


def main() -> int:
    if not RUNNER.exists():
        raise FileNotFoundError(f"Missing runner: {RUNNER}")

    args = _parse_args()
    run_root = Path(args.run_root).resolve()
    train_root = run_root / "train"
    train_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "VAR_PROJECT_ROOT": str(train_root),
            "VAR_LOGGING_PATH": str(train_root / "vraxion.log"),
            "VAR_COMPUTE_DEVICE": str(args.device),
            "VRX_OFFLINE_ONLY": "1" if int(args.offline_only) else "0",
            "VRX_SYNTH": "1",
            "VRX_SYNTH_MODE": "assoc_byte",
            "VRX_SYNTH_LEN": str(int(args.synth_len)),
            "VRX_ASSOC_KEYS": str(int(args.assoc_keys)),
            "VRX_ASSOC_PAIRS": str(int(args.assoc_pairs)),
            "VRX_ASSOC_VAL_RANGE": str(int(args.assoc_val_range)),
            "VRX_RING_LEN": str(int(args.ring_len)),
            "VRX_SLOT_DIM": str(int(args.slot_dim)),
            "VRX_EXPERT_HEADS": str(int(args.expert_heads)),
            "VRX_BATCH_SIZE": str(int(args.batch_size)),
            "VRX_MAX_STEPS": str(int(args.max_steps)),
            "VRX_IGNORE_MAX_STEPS": "1" if int(args.ignore_max_steps) else "0",
            "VRX_IGNORE_WALL_CLOCK": "1" if int(args.ignore_wall_clock) else "0",
            "VRX_SAVE_EVERY_STEPS": str(int(args.save_every_steps)),
            "VRX_EVAL_SAMPLES": str(int(args.eval_samples)),
            "VRX_SAVE_LAST_GOOD": "1" if int(args.save_last_good) else "0",
            "VRX_SAVE_HISTORY": "1" if int(args.save_history) else "0",
            "VRX_PTR_DTYPE": str(args.ptr_dtype),
            "VRX_RESUME": "1" if int(args.resume) else "0",
        }
    )

    cmd = [sys.executable, "-u", str(RUNNER)]
    cp = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return int(cp.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

