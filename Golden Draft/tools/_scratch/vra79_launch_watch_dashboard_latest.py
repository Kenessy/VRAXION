#!/usr/bin/env python3
"""Launch dashboard against the latest VRA-79 small saturation run log."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
RUNS_ROOT = REPO_ROOT / r"bench_vault\_tmp\vra79_small_saturation"
DASHBOARD_SCRIPT = REPO_ROOT / r"Golden Draft\tools\live_dashboard.py"


def _pick_port(start: int = 8501, end: int = 8510) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found in 8501..8510")


def _latest_run_root() -> Path:
    if not RUNS_ROOT.exists():
        raise FileNotFoundError(f"Missing runs root: {RUNS_ROOT}")
    candidates = [path for path in RUNS_ROOT.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"No run dirs found under: {RUNS_ROOT}")
    candidates.sort(key=lambda path: path.name, reverse=True)
    return candidates[0]


def main() -> int:
    if not DASHBOARD_SCRIPT.exists():
        raise FileNotFoundError(f"Missing dashboard script: {DASHBOARD_SCRIPT}")

    run_root = _latest_run_root()
    # Prefer the run's own log file (always produced by VAR_LOGGING_PATH),
    # fall back to supervisor-captured stdout if needed.
    preferred_log = run_root / r"train\vraxion.log"
    fallback_log = run_root / r"supervisor_job\child_stdout.log"
    if preferred_log.exists():
        log_path = preferred_log
    elif fallback_log.exists():
        log_path = fallback_log
    else:
        raise FileNotFoundError(f"Missing logs: {preferred_log} and {fallback_log}")

    port = _pick_port()
    url = f"http://localhost:{port}"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_SCRIPT),
        "--server.port",
        str(port),
        "--",
        "--log",
        str(log_path),
        "--refresh",
        "2",
        "--max-rows",
        "5000",
    ]

    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        start_new_session=not sys.platform.startswith("win"),
    )

    boot_info = run_root / "dashboard_boot.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dashboard_pid": int(proc.pid),
        "url": url,
        "port": int(port),
        "run_root": str(run_root),
        "log_path": str(log_path),
        "cmd": cmd,
    }
    boot_info.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
