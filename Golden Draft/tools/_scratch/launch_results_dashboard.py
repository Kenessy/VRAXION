"""Launch the Streamlit results dashboard in a detached process.

Why this exists:
- Running Streamlit directly from the CLI tool session can terminate the
  server when the session ends.
- PowerShell one-liners are brittle in this environment (quoting / `$`).
- We want a single, reliable command that launches the dashboard and keeps
  it alive, with logs written to bench_vault/_tmp.

Usage:
  python "Golden Draft/tools/_scratch/launch_results_dashboard.py" --port 8520
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def _is_port_open(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", int(port))) == 0
    finally:
        try:
            s.close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8520)
    args = p.parse_args(argv)

    # .../Golden Draft/tools/_scratch/ -> parents[0]=_scratch, [1]=tools,
    # [2]=Golden Draft, [3]=repo root
    repo_root = Path(__file__).resolve().parents[3]
    app_path = repo_root / "Golden Draft" / "tools" / "results_dashboard.py"
    if not app_path.exists():
        raise SystemExit(f"results_dashboard.py not found: {app_path}")

    port = int(args.port)
    if _is_port_open(port):
        print(f"[dashboard] already listening on port {port}")
        print(f"[dashboard] open: http://localhost:{port}")
        return 0

    log_dir = repo_root / "bench_vault" / "_tmp"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"results_dashboard_{port}_{ts}.log"

    # Detach on Windows so the dashboard survives the parent session.
    creationflags = 0
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "DETACHED_PROCESS", 0)) | int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=lf,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )

    print(f"[dashboard] launched pid={proc.pid} port={port}")
    print(f"[dashboard] log: {log_path}")
    print(f"[dashboard] open: http://localhost:{port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

