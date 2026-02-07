#!/usr/bin/env python3
"""Start one supervised small-E1 infinite saturation run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
SUPERVISOR_TOOL = REPO_ROOT / r"Golden Draft\tools\vraxion_lab_supervisor.py"
ENTRY_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_small_infinite_entry.py"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main() -> int:
    if not SUPERVISOR_TOOL.exists():
        raise FileNotFoundError(f"Missing supervisor tool: {SUPERVISOR_TOOL}")
    if not ENTRY_TOOL.exists():
        raise FileNotFoundError(f"Missing entry tool: {ENTRY_TOOL}")

    stamp = _utc_stamp()
    run_root = REPO_ROOT / "bench_vault" / "_tmp" / "vra79_small_saturation" / f"{stamp}-small_e1_infinite"
    run_root.mkdir(parents=True, exist_ok=True)
    job_root = run_root / "supervisor_job"
    launch_log = run_root / "launcher_boot.log"

    child_cmd = [
        sys.executable,
        "-u",
        str(ENTRY_TOOL),
        "--run-root",
        str(run_root),
    ]
    sup_cmd = [
        sys.executable,
        "-u",
        str(SUPERVISOR_TOOL),
        "--job-name",
        "vra79_small_saturation_infinite",
        "--job-root",
        str(job_root),
        "--wake-window-title",
        "VRAXION_CLI",
        "--wake-after-s",
        "120",
        "--watchdog-no-output-s",
        "600",
        "--watchdog-surge-s",
        "180",
        "--watchdog-spill-s",
        "420",
        "--watchdog-abort-after-kills",
        "3",
        "--poll-interval-s",
        "10",
        "--max-restarts",
        "3",
        "--",
    ] + child_cmd

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    with launch_log.open("a", encoding="utf-8") as logh:
        proc = subprocess.Popen(
            sup_cmd,
            cwd=str(REPO_ROOT),
            stdout=logh,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            start_new_session=(os.name != "nt"),
        )

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "supervisor_pid": int(proc.pid),
        "run_root": str(run_root),
        "job_root": str(job_root),
        "launch_log": str(launch_log),
        "run_log": str(run_root / r"train\vraxion.log"),
        "child_stdout_log": str(job_root / "child_stdout.log"),
        "child_stderr_log": str(job_root / "child_stderr.log"),
        "supervisor_log": str(job_root / "supervisor.log"),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
