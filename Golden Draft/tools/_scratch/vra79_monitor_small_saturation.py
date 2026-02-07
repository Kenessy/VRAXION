#!/usr/bin/env python3
"""Simple live monitor for VRA-79 small saturation supervisor job."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


def _tail_from(path: Path, pos: int) -> tuple[int, list[str]]:
    if not path.exists():
        return pos, []
    size = path.stat().st_size
    if pos > size:
        pos = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(pos)
        text = handle.read()
        new_pos = handle.tell()
    if not text:
        return new_pos, []
    return new_pos, [line for line in text.splitlines() if line.strip()]


def main() -> int:
    # Ensure monitor output appears immediately even when stdout is piped.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Monitor supervisor + child logs for VRA-79 small saturation.")
    ap.add_argument("--job-root", required=True)
    ap.add_argument("--poll-s", type=int, default=15)
    ap.add_argument("--max-minutes", type=int, default=120)
    args = ap.parse_args()

    job_root = Path(args.job_root).resolve()
    supervisor_log = job_root / "supervisor.log"
    child_stdout = job_root / "child_stdout.log"
    child_stderr = job_root / "child_stderr.log"
    # Prefer run log if present; this is what the model writes via VAR_LOGGING_PATH.
    run_log = job_root.parent / r"train\vraxion.log"

    sup_pos = 0
    out_pos = 0
    err_pos = 0
    run_pos = 0
    start = time.time()
    last_pulse = 0.0

    print(f"[monitor] start={datetime.now().isoformat(timespec='seconds')} job_root={job_root}")
    print(f"[monitor] logs: sup={supervisor_log} out={child_stdout} err={child_stderr}")
    print(f"[monitor] run_log={run_log}")

    max_seconds = int(max(1, args.max_minutes)) * 60
    while True:
        now = time.time()
        elapsed = int(now - start)
        if elapsed >= max_seconds:
            print(f"[monitor] stop: reached max runtime {args.max_minutes}m")
            return 0

        sup_pos, sup_lines = _tail_from(supervisor_log, sup_pos)
        out_pos, out_lines = _tail_from(child_stdout, out_pos)
        err_pos, err_lines = _tail_from(child_stderr, err_pos)
        run_pos, run_lines = _tail_from(run_log, run_pos)

        for line in sup_lines[-8:]:
            print(f"[sup] {line}")
        key_out = [ln for ln in out_lines if "heartbeat" in ln.lower() or "step" in ln.lower() or "eval" in ln.lower()]
        for line in key_out[-12:]:
            print(f"[out] {line}")
        key_run = [ln for ln in run_lines if "heartbeat" in ln.lower() or "step" in ln.lower() or "eval" in ln.lower()]
        for line in key_run[-12:]:
            print(f"[run] {line}")
        for line in err_lines[-6:]:
            print(f"[err] {line}")

        if now - last_pulse >= 60:
            print(f"[pulse] elapsed={elapsed}s sup_bytes={sup_pos} out_bytes={out_pos} err_bytes={err_pos}")
            last_pulse = now

        time.sleep(max(5, int(args.poll_s)))


if __name__ == "__main__":
    raise SystemExit(main())
