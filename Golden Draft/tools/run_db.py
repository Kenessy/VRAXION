from __future__ import annotations

"""VRAXION run recorder.

This script wraps an arbitrary command (training/eval/etc.) and records:
- stdout/stderr append-only logs
- parsed metrics JSONL (step/loss/V_COG)
- meta.json + summary.json
- a small SQLite index (runs.sqlite) for fast querying

Non-negotiable behavior:
- CLI entrypoint stays:  python tools/run_db.py ... -- <cmd...>
- Prints the created run_dir to stdout (single line at end)
- Exits with the child process exit code

Only stdlib dependencies.
"""

import argparse
import codecs
import datetime as _dt
import getpass
import json
import os
import platform
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, TextIO, Tuple

# Support both:
# - running as a script:   python tools/run_db.py
# - importing as package:  import tools.run_db
try:  # pragma: no cover
    from .vcog_parse import OnlineStats, dump_json, parse_line
except ImportError:  # pragma: no cover
    # When executed as a script, `sys.path[0]` is the tools/ directory, not the repo root.
    # Add the repo root so `import tools.vcog_parse` works.
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.vcog_parse import OnlineStats, dump_json, parse_line


def _utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _repo_root() -> Path:
    # tools/run_db.py -> repo root
    return Path(__file__).resolve().parents[1]


def _sanitize_run_slug(s: str) -> str:
    """Make a filesystem-friendly slug for a human run name.

    We keep ASCII alnum plus '-' '_' and map whitespace to '_' (everything else dropped).
    """

    keep: list[str] = []
    for ch in s.strip():
        o = ord(ch)
        if o < 128 and (ch.isalnum() or ch in ("-", "_")):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
        # drop everything else

    out = "".join(keep).strip("_")
    return (out[:64] if out else "run")


def _git_info(root: Path) -> Dict[str, Any]:
    """Best-effort git commit + dirty flag (fails gracefully if git unavailable)."""

    info: Dict[str, Any] = {"commit": None, "dirty": None}

    def _run_git(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(args),
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )

    try:
        r = _run_git(["git", "rev-parse", "HEAD"])
        if r.returncode == 0:
            info["commit"] = r.stdout.strip() or None

        s = _run_git(["git", "status", "--porcelain"])
        if s.returncode == 0:
            info["dirty"] = 1 if s.stdout.strip() else 0
    except Exception:
        # git not installed / not a repo / permission issues / etc.
        pass

    return info


def _ensure_sqlite(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY,
              start_utc TEXT,
              end_utc TEXT,
              duration_s REAL,
              cwd TEXT,
              cmd TEXT,
              git_commit TEXT,
              git_dirty INTEGER,
              exit_code INTEGER,
              run_name TEXT,
              tags TEXT,
              meta_path TEXT,
              summary_path TEXT,
              loss_mean REAL,
              loss_std REAL,
              loss_min REAL,
              loss_max REAL,
              n_loss INTEGER
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS runs_start ON runs(start_utc)")
        con.commit()
    finally:
        con.close()


def _insert_sqlite(db_path: Path, row: Dict[str, Any]) -> None:
    _ensure_sqlite(db_path)
    con = sqlite3.connect(str(db_path))
    try:
        cols = [
            "run_id",
            "start_utc",
            "end_utc",
            "duration_s",
            "cwd",
            "cmd",
            "git_commit",
            "git_dirty",
            "exit_code",
            "run_name",
            "tags",
            "meta_path",
            "summary_path",
            "loss_mean",
            "loss_std",
            "loss_min",
            "loss_max",
            "n_loss",
        ]
        vals = [row.get(c) for c in cols]
        placeholders = ",".join(["?"] * len(cols))
        con.execute(
            f"INSERT OR REPLACE INTO runs ({','.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        con.commit()
    finally:
        con.close()


def _create_run_dir(runs_dir: Path, run_id_base: str) -> Tuple[str, Path]:
    """Create a unique run directory under runs_dir.

    Keeps the historical run_id pattern (timestamp + slug). If a collision happens,
    appends _2, _3, ... and finally a short UUID.
    """

    runs_dir.mkdir(parents=True, exist_ok=True)

    def _candidates() -> Iterable[str]:
        yield run_id_base
        for i in range(2, 1000):
            yield f"{run_id_base}_{i}"
        yield f"{run_id_base}_{uuid.uuid4().hex[:8]}"

    for run_id in _candidates():
        run_dir = runs_dir / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue

    raise RuntimeError("failed to create a unique run directory")


def _safe_json_dumps(obj: Any) -> str:
    # Keep metrics.jsonl stable and easy to diff (sorted keys, no trailing spaces).
    return json.dumps(obj, sort_keys=True)


def _record_metrics_line(
    *,
    stream_name: str,
    line: str,
    metrics_fp: TextIO,
    lock: threading.Lock,
    loss_stats: OnlineStats,
    last_vcog: MutableMapping[str, Any],
) -> None:
    ev, vcog = parse_line(line)
    if ev is None and vcog is None:
        return

    if ev is None:
        ev = {"ts_utc": _utc_iso()}

    ev["stream"] = stream_name

    with lock:
        if vcog is not None:
            ev["vcog"] = vcog
            last_vcog.clear()
            last_vcog.update(vcog)
        if "loss" in ev:
            try:
                loss_stats.update(float(ev["loss"]))
            except Exception:
                # If loss is somehow non-numeric, skip stats update but still log event.
                pass

        metrics_fp.write(_safe_json_dumps(ev) + "\n")
        metrics_fp.flush()


def _pump_stream(
    *,
    stream_name: str,
    stream,  # BinaryIO
    log_fp: TextIO,
    metrics_fp: TextIO,
    lock: threading.Lock,
    loss_stats: OnlineStats,
    last_vcog: MutableMapping[str, Any],
) -> None:
    """Continuously drain a subprocess pipe.

    Uses chunked reads (not readline) to avoid deadlocks when a process emits a long
    line without a newline.
    """

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""

    def _flush_lines() -> None:
        nonlocal buf
        while True:
            nl = buf.find("\n")
            if nl < 0:
                return
            line = buf[: nl + 1]
            buf = buf[nl + 1 :]
            _record_metrics_line(
                stream_name=stream_name,
                line=line,
                metrics_fp=metrics_fp,
                lock=lock,
                loss_stats=loss_stats,
                last_vcog=last_vcog,
            )

    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break

            text = decoder.decode(chunk)
            if not text:
                continue

            # Append-only logs: write exactly what we decoded, as soon as we decoded it.
            log_fp.write(text)
            log_fp.flush()

            buf += text
            _flush_lines()

        # Final decoder flush (rare, but can happen for partial UTF-8 sequences).
        tail = decoder.decode(b"", final=True)
        if tail:
            log_fp.write(tail)
            log_fp.flush()
            buf += tail

        if buf:
            # Process a last partial line (no trailing newline).
            _record_metrics_line(
                stream_name=stream_name,
                line=buf,
                metrics_fp=metrics_fp,
                lock=lock,
                loss_stats=loss_stats,
                last_vcog=last_vcog,
            )
    except Exception:
        # Never let a reader thread crash the whole recorder.
        return


def _parse_cmd_remainder(cmd_remainder: Sequence[str]) -> List[str]:
    # argparse.REMAINDER keeps the leading "--" in many setups; only strip a single
    # leading marker. Do NOT strip subsequent "--" which may be part of the command.
    cmd = list(cmd_remainder)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=r"Run a command and archive logs/metrics into S:\AI\Golden Draft\vault\db."
    )
    ap.add_argument(
        "--db-root",
        default=r"S:\AI\Golden Draft\vault\db",
        help=r"DB root dir (default: S:\AI\Golden Draft\vault\db)",
    )
    ap.add_argument("--run-name", default="", help="Human-readable run name (slugged)")
    ap.add_argument("--tag", action="append", default=[], help="Tag (repeatable)")
    ap.add_argument("--workdir", default="", help="Working directory for the command (default: repo root)")
    ap.add_argument("--env", action="append", default=[], help="Env override KEY=VAL (repeatable)")
    ap.add_argument("--no-sqlite", action="store_true", help="Disable sqlite indexing")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run. Use -- before the command.")
    args = ap.parse_args(argv)

    cmd = _parse_cmd_remainder(args.cmd)
    if not cmd:
        ap.error("Missing command. Example: python tools/run_db.py -- python vraxion_run.py")

    repo = _repo_root()
    workdir = Path(args.workdir).resolve() if args.workdir else repo

    db_root = Path(args.db_root)
    runs_dir = db_root / "runs"

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _sanitize_run_slug(args.run_name) if args.run_name else "run"
    run_id_base = f"{ts}_{slug}"
    run_id, run_dir = _create_run_dir(runs_dir, run_id_base)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    metrics_path = run_dir / "metrics.jsonl"
    meta_path = run_dir / "meta.json"
    summary_path = run_dir / "summary.json"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for kv in args.env:
        if "=" not in kv:
            raise SystemExit(f"--env expects KEY=VAL, got: {kv}")
        k, v = kv.split("=", 1)
        env[k] = v

    g = _git_info(repo)
    meta: Dict[str, Any] = {
        "run_id": run_id,
        "run_name": args.run_name,
        "tags": args.tag,
        "start_utc": _utc_iso(),
        "host": platform.node(),
        "user": getpass.getuser(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(workdir),
        "cmd": cmd,
        "env_overrides": args.env,
        "git": g,
    }
    dump_json(meta_path, meta)

    loss_stats = OnlineStats()
    last_vcog: Dict[str, Any] = {}
    lock = threading.Lock()

    t0 = time.monotonic()
    exit_code: int

    # Ensure the key output files always exist (even if Popen fails).
    with (
        open(stdout_path, "w", encoding="utf-8", errors="replace", newline="") as out_fp,
        open(stderr_path, "w", encoding="utf-8", errors="replace", newline="") as err_fp,
        open(metrics_path, "w", encoding="utf-8", newline="\n") as met_fp,
    ):
        try:
            p = subprocess.Popen(
                cmd,
                cwd=str(workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
            )
        except FileNotFoundError as e:
            # Command couldn't be launched; record the failure and continue to summary.
            err_fp.write(f"[run_db] failed to launch: {e}\n")
            err_fp.flush()
            exit_code = 127
        else:
            assert p.stdout is not None
            assert p.stderr is not None

            th_out = threading.Thread(
                target=_pump_stream,
                kwargs={
                    "stream_name": "stdout",
                    "stream": p.stdout,
                    "log_fp": out_fp,
                    "metrics_fp": met_fp,
                    "lock": lock,
                    "loss_stats": loss_stats,
                    "last_vcog": last_vcog,
                },
                daemon=True,
            )
            th_err = threading.Thread(
                target=_pump_stream,
                kwargs={
                    "stream_name": "stderr",
                    "stream": p.stderr,
                    "log_fp": err_fp,
                    "metrics_fp": met_fp,
                    "lock": lock,
                    "loss_stats": loss_stats,
                    "last_vcog": last_vcog,
                },
                daemon=True,
            )
            th_out.start()
            th_err.start()

            try:
                exit_code = p.wait()
            except KeyboardInterrupt:
                # Best-effort: terminate child and wait.
                try:
                    p.terminate()
                except Exception:
                    pass
                exit_code = p.wait()

            # Help threads reach EOF quickly.
            try:
                if p.stdout:
                    p.stdout.close()
            except Exception:
                pass
            try:
                if p.stderr:
                    p.stderr.close()
            except Exception:
                pass

            th_out.join()
            th_err.join()

    dur_s = time.monotonic() - t0
    end_utc = _utc_iso()

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "start_utc": meta["start_utc"],
        "end_utc": end_utc,
        "duration_s": dur_s,
        "exit_code": exit_code,
        "loss": loss_stats.to_dict(),
        "vcog_last": last_vcog or None,
    }
    try:
        dump_json(summary_path, summary)
    except Exception as errval:
        print(f"[run_db] failed to write summary.json: {errval!r}", file=sys.stderr)

    if not args.no_sqlite:
        db_path = db_root / "runs.sqlite"
        row = {
            "run_id": run_id,
            "start_utc": meta["start_utc"],
            "end_utc": end_utc,
            "duration_s": dur_s,
            "cwd": str(workdir),
            "cmd": " ".join(cmd),
            "git_commit": g.get("commit"),
            "git_dirty": g.get("dirty"),
            "exit_code": exit_code,
            "run_name": args.run_name,
            "tags": json.dumps(args.tag),
            "meta_path": str(meta_path),
            "summary_path": str(summary_path),
            "loss_mean": summary["loss"]["mean"],
            "loss_std": summary["loss"]["std"],
            "loss_min": summary["loss"]["min"],
            "loss_max": summary["loss"]["max"],
            "n_loss": summary["loss"]["n"],
        }
        try:
            _insert_sqlite(db_path, row)
        except Exception as errval:
            print(f"[run_db] sqlite index failed: {errval!r}", file=sys.stderr)

    # Non-negotiable: single line at end with the run_dir path.
    print(str(run_dir))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
