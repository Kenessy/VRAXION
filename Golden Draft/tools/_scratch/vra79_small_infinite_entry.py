#!/usr/bin/env python3
"""Entry for supervised infinite small-E1 synthetic saturation run."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
RUNNER = REPO_ROOT / r"Golden Draft\vraxion_run.py"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run infinite small-E1 synth loop.")
    ap.add_argument("--run-root", required=True, help="Root dir for run artifacts.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root).resolve()
    train_root = run_root / "train"
    train_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # Uninterrupted saturation: run synth continuously and ignore wall-clock/max-step
    # cutoffs so `step` stays monotonic unless the process itself restarts.
    env.update(
        {
            "VAR_PROJECT_ROOT": str(train_root),
            "VAR_LOGGING_PATH": str(train_root / "vraxion.log"),
            "VAR_COMPUTE_DEVICE": "cuda",
            "VRX_SYNTH": "1",
            "VRX_SYNTH_MODE": "assoc_byte",
            "VRX_SYNTH_LEN": "256",
            "VRX_ASSOC_KEYS": "64",
            "VRX_ASSOC_PAIRS": "4",
            "VRX_ASSOC_VAL_RANGE": "256",
            "VRX_RING_LEN": "2048",
            "VRX_SLOT_DIM": "256",
            "VRX_EXPERT_HEADS": "1",
            "VRX_BATCH_SIZE": "27",
            # Keep eval cost bounded but rankable (>=512).
            "VRX_EVAL_SAMPLES": "512",
            "VRX_SAVE_LAST_GOOD": "1",
            "VRX_SAVE_HISTORY": "0",
            "VRX_EVAL_AT_CHECKPOINT": "0",
            # Ensure the run produces regular "smartness" signals in the log.
            "VRX_EVAL_EVERY_STEPS": "100",
            "VRX_IGNORE_WALL_CLOCK": "1",
            "VRX_IGNORE_MAX_STEPS": "1",
            "VRX_PTR_DTYPE": "fp64",
            "VRX_RESUME": "0",
        }
    )

    cmd = [sys.executable, "-u", str(RUNNER)]
    cp = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return int(cp.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
