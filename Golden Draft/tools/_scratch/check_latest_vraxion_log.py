"""Print the most recently modified bench_vault/benchmarks/**/vraxion.log.

Used by PLANLAB to prove "time-to-visible-output" quickly without relying on
PowerShell variables/quoting.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=str, default=r"S:\AI\work\VRAXION_DEV")
    p.add_argument("--tail", type=int, default=3)
    args = p.parse_args()

    repo_root = Path(args.repo_root)
    bench_root = repo_root / "bench_vault" / "benchmarks"
    logs = list(bench_root.rglob("vraxion.log"))
    if not logs:
        print(f"[latest_log] none found under {bench_root}")
        return 2

    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    log_path = logs[0]
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        lines = []

    print(f"[latest_log] {log_path}")
    if lines:
        for ln in lines[-int(max(1, args.tail)) :]:
            print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

