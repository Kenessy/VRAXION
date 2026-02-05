"""Minimal run supervisor + loopback trigger for unattended VRAXION experiments.

This tool exists because Codex turns are finite. For long runs we need:
  - a supervisor that continues running after the chat turn ends
  - crash recovery (auto-restart with backoff)
  - a "wake trigger" file that an external "Ghost Hand" script can use to
    re-activate the chat when something meaningful happens

Design constraints:
  - Stdlib only (no new deps)
  - Append-only logs (never rewrite)
  - Result-safe: treat the child as the source of truth; we only manage process
    lifecycle + wake triggers.
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _utc_iso() -> str:
    # time.strftime has no UTC ISO with millis; keep it simple + stable.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def _jsonl_append(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


class WatchdogStage(enum.IntEnum):
    OK = 0
    SURGE = 1
    SPILL = 2
    BREACH = 3
    WASHOUT = 4


def _compute_watchdog_stage(no_output_s: float, surge_s: float, spill_s: float, breach_s: float) -> WatchdogStage:
    # Strictly monotone staging as "no output" grows. Stages are only meaningful
    # if breach_s > 0. Caller must ensure ordered thresholds.
    if breach_s <= 0.0:
        return WatchdogStage.OK
    if no_output_s >= breach_s:
        return WatchdogStage.BREACH
    if spill_s > 0.0 and no_output_s >= spill_s:
        return WatchdogStage.SPILL
    if surge_s > 0.0 and no_output_s >= surge_s:
        return WatchdogStage.SURGE
    return WatchdogStage.OK


def _tail_lines(path: Path, max_lines: int) -> List[str]:
    if max_lines <= 0 or not path.exists():
        return []
    # Read whole file for simplicity; logs are typically small-ish. If this ever
    # becomes too large, replace with a bounded tail reader by bytes.
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    lines = txt.splitlines()
    return lines[-max_lines:]


def _write_log_tail(job_root: Path, out_log: Path, err_log: Path, tail_lines: int) -> Path:
    tail_path = job_root / "child_log_tail.txt"
    out_tail = _tail_lines(out_log, tail_lines)
    err_tail = _tail_lines(err_log, tail_lines)
    chunks: List[str] = []
    chunks.append("=== child_stdout.log (tail) ===")
    chunks.extend(out_tail or ["<empty>"])
    chunks.append("")
    chunks.append("=== child_stderr.log (tail) ===")
    chunks.extend(err_tail or ["<empty>"])
    tail_path.write_text("\n".join(chunks).rstrip("\n") + "\n", encoding="utf-8")
    return tail_path


def _append_failure_summary(
    job_root: Path,
    *,
    stage: WatchdogStage,
    reason: str,
    attempt: int,
    child_pid: int,
    no_output_s: float,
    watchdog_s: float,
) -> Path:
    summ_path = job_root / "failure_summary.md"
    lines: List[str] = []
    lines.append(f"## {stage.name} - {reason}")
    lines.append("")
    lines.append(f"- utc: `{_utc_iso()}`")
    lines.append(f"- attempt: `{attempt}`")
    lines.append(f"- child_pid: `{child_pid}`")
    lines.append(f"- no_output_s: `{no_output_s:.1f}`")
    lines.append(f"- watchdog_no_output_s: `{watchdog_s:.1f}`")
    lines.append("")
    with summ_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return summ_path


def _default_status_path(repo_root: Path) -> Path:
    return repo_root / "bench_vault" / "_tmp" / "nightmode_status.md"


def _write_watchdog_state(
    state_path: Path,
    *,
    job_root: Path,
    stage: WatchdogStage,
    reason: str,
    attempt: int,
    failures: int,
    child_pid: int,
    no_output_s: float,
    watchdog_s: float,
    kill_count: int,
    degrade_level: str,
) -> None:
    payload = {
        "version": 1,
        "updated_utc": _utc_iso(),
        "job_root": str(job_root),
        "stage": stage.name,
        "reason": str(reason),
        "attempt": int(attempt),
        "failures": int(failures),
        "child_pid": int(child_pid),
        "no_output_s": float(no_output_s),
        "watchdog_no_output_s": float(watchdog_s),
        "kill_count": int(kill_count),
        "degrade_level": str(degrade_level),
        # Placeholders for future signal integration (kept explicit for schema stability).
        "step_ratio_p95_over_median": None,
        "vram_reserved_ratio": None,
        "gpu_util_zero_s": None,
    }
    _atomic_write_json(state_path, payload)


def _append_line(path: Path, line: str) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _kill_pid_tree(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        # /T = tree, /F = force
        subprocess.call(["taskkill", "/PID", str(int(pid)), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    try:
        os.killpg(int(pid), 9)
    except Exception:
        try:
            os.kill(int(pid), 9)
        except Exception:
            pass


@dataclass(frozen=True)
class WakeTrigger:
    wake_after_s: int
    memo: str
    reason: str
    job_root: str
    attempt: int
    pid: int
    window_title: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "created_utc": _utc_iso(),
            "wake_after_s": int(self.wake_after_s),
            "memo": str(self.memo),
            "reason": str(self.reason),
            "job_root": str(self.job_root),
            "attempt": int(self.attempt),
            "pid": int(self.pid),
            "window_title": str(self.window_title),
        }


def _default_repo_root() -> Path:
    # Golden Draft/tools/<this_file>.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _default_job_root(repo_root: Path, job_name: str) -> Path:
    safe = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in job_name)
    return repo_root / "bench_vault" / "jobs" / f"{_now_ts()}_{safe}"


def _default_wake_trigger(repo_root: Path) -> Path:
    return repo_root / "bench_vault" / "wake_trigger.json"


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vraxion_lab_supervisor.py")
    p.add_argument("--job-name", required=True, help="Human-readable name for logs/state.")
    p.add_argument("--job-root", default="", help="Where to write logs/state. Default: bench_vault/jobs/<ts>_<job-name>")
    p.add_argument("--wake-trigger", default="", help="Where to write wake_trigger.json. Default: bench_vault/wake_trigger.json")
    p.add_argument("--wake-after-s", type=int, default=720, help="Delay before Ghost Hand sends the memo (seconds).")
    p.add_argument(
        "--wake-window-title",
        default="",
        help="Optional window title for ghost_wake.ps1 AppActivate (recommended).",
    )
    p.add_argument(
        "--heartbeat-interval-s",
        type=int,
        default=0,
        help="If >0, emit a wake trigger periodically while running (best-effort; won't overwrite an existing trigger).",
    )
    p.add_argument(
        "--watchdog-no-output-s",
        type=int,
        default=0,
        help="If >0, kill+restart child if it produces no stdout/stderr bytes for this many seconds.",
    )
    p.add_argument(
        "--watchdog-surge-s",
        type=int,
        default=0,
        help="If >0, emit SURGE warnings after this many seconds with no stdout/stderr growth. Default: derived from --watchdog-no-output-s.",
    )
    p.add_argument(
        "--watchdog-spill-s",
        type=int,
        default=0,
        help="If >0, emit SPILL warnings after this many seconds with no stdout/stderr growth. Default: derived from --watchdog-no-output-s.",
    )
    p.add_argument(
        "--watchdog-abort-after-kills",
        type=int,
        default=0,
        help="If >0, abort the supervisor after this many watchdog kill events (WASHOUT). Default: 0 (never).",
    )
    p.add_argument(
        "--status-path",
        default="",
        help="Append-only status file path. Default: bench_vault/_tmp/nightmode_status.md under repo root.",
    )
    p.add_argument(
        "--log-tail-lines",
        type=int,
        default=200,
        help="Number of lines to capture from child stdout/stderr when writing failure artifacts.",
    )
    p.add_argument(
        "--poll-interval-s",
        type=float,
        default=1.0,
        help="Supervisor polling interval for checking child status/log growth (seconds).",
    )
    p.add_argument("--max-restarts", type=int, default=10, help="Maximum restart attempts after failures.")
    p.add_argument("--restart-backoff-s", type=float, default=5.0, help="Initial backoff between restarts (seconds).")
    p.add_argument(
        "--stop-on-file",
        default="",
        help="If set, stop once this file exists (absolute or relative to repo root).",
    )
    p.add_argument(
        "--stop-on-success",
        action="store_true",
        default=True,
        help="Stop when the child exits with code 0 (default).",
    )
    p.add_argument(
        "--no-stop-on-success",
        action="store_false",
        dest="stop_on_success",
        help="Keep restarting even on exit code 0 (rare).",
    )
    p.add_argument(
        "--",
        dest="cmd_sep",
        action="store_true",
        help="Separator before the child command (everything after is the child command).",
    )
    p.add_argument("child_cmd", nargs=argparse.REMAINDER, help="Child command (prefix with --).")
    return p.parse_args(list(argv) if argv is not None else None)


def _resolve_stop_file(repo_root: Path, raw: str) -> Optional[Path]:
    if not raw.strip():
        return None
    p = Path(raw.strip())
    if p.is_absolute():
        return p
    return repo_root / p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if not args.child_cmd:
        raise SystemExit("Missing child command. Usage: python vraxion_lab_supervisor.py --job-name X -- <cmd...>")
    if args.child_cmd and args.child_cmd[0] == "--":
        args.child_cmd = args.child_cmd[1:]

    repo_root = _default_repo_root()
    job_root = Path(args.job_root).expanduser() if str(args.job_root).strip() else _default_job_root(repo_root, args.job_name)
    wake_trigger = Path(args.wake_trigger).expanduser() if str(args.wake_trigger).strip() else _default_wake_trigger(repo_root)
    stop_file = _resolve_stop_file(repo_root, str(args.stop_on_file))
    status_path = Path(args.status_path).expanduser() if str(args.status_path).strip() else _default_status_path(repo_root)

    poll_s_base = float(args.poll_interval_s)
    if poll_s_base <= 0.0:
        poll_s_base = 1.0

    tail_lines = int(args.log_tail_lines)
    abort_after_kills = int(max(0, int(args.watchdog_abort_after_kills)))

    watchdog_s = float(max(0, int(args.watchdog_no_output_s)))
    surge_s = float(max(0, int(args.watchdog_surge_s)))
    spill_s = float(max(0, int(args.watchdog_spill_s)))
    if watchdog_s > 0.0:
        if surge_s <= 0.0 or surge_s >= watchdog_s:
            surge_s = max(1.0, watchdog_s / 3.0)
        if spill_s <= 0.0 or spill_s >= watchdog_s:
            spill_s = max(surge_s + 1.0, 2.0 * watchdog_s / 3.0)
        if spill_s <= surge_s:
            spill_s = surge_s + 1.0
    else:
        surge_s = 0.0
        spill_s = 0.0

    _ensure_dir(job_root)
    sup_log = job_root / "supervisor.log"
    out_log = job_root / "child_stdout.log"
    err_log = job_root / "child_stderr.log"
    state_path = job_root / "state.json"
    config_path = job_root / "job.json"
    wd_state_path = job_root / "watchdog_state.json"
    wd_events_path = job_root / "watchdog_events.jsonl"
    degrade_path = job_root / "degrade_mode.json"

    _atomic_write_json(
        config_path,
        {
            "version": 1,
            "created_utc": _utc_iso(),
            "job_name": args.job_name,
            "job_root": str(job_root),
            "wake_trigger": str(wake_trigger),
            "wake_after_s": int(args.wake_after_s),
            "wake_window_title": str(args.wake_window_title),
            "heartbeat_interval_s": int(args.heartbeat_interval_s),
            "watchdog_no_output_s": int(args.watchdog_no_output_s),
            "watchdog_surge_s": int(args.watchdog_surge_s),
            "watchdog_spill_s": int(args.watchdog_spill_s),
            "watchdog_abort_after_kills": int(args.watchdog_abort_after_kills),
            "status_path": str(args.status_path),
            "log_tail_lines": int(args.log_tail_lines),
            "poll_interval_s": float(args.poll_interval_s),
            "max_restarts": int(args.max_restarts),
            "restart_backoff_s": float(args.restart_backoff_s),
            "stop_on_file": str(stop_file) if stop_file else "",
            "stop_on_success": bool(args.stop_on_success),
            "child_cmd": list(args.child_cmd),
            "python": sys.executable,
            "pid": int(os.getpid()),
        },
    )

    attempt = 0
    failures = 0
    kill_count = 0
    exit_code = 0
    last_hb = 0.0
    last_out_sz = 0
    last_err_sz = 0
    last_activity = time.monotonic()

    _append_line(sup_log, f"[{_utc_iso()}] supervisor start pid={os.getpid()} job={args.job_name!r}")
    _append_line(sup_log, f"[{_utc_iso()}] child cmd: {args.child_cmd!r}")
    _append_line(sup_log, f"[{_utc_iso()}] job_root={job_root}")
    if watchdog_s > 0.0:
        _append_line(
            sup_log,
            f"[{_utc_iso()}] watchdog clock: surge_s={surge_s:.1f} spill_s={spill_s:.1f} breach_s={watchdog_s:.1f} "
            f"abort_after_kills={abort_after_kills} poll_s={poll_s_base:.2f} status_path={status_path}",
        )
        _append_line(
            status_path,
            f"[{_utc_iso()}] watchdog clock armed: job={args.job_name!r} job_root={job_root} "
            f"surge_s={surge_s:.1f} spill_s={spill_s:.1f} breach_s={watchdog_s:.1f} abort_after_kills={abort_after_kills}",
        )
        try:
            _write_watchdog_state(
                wd_state_path,
                job_root=job_root,
                stage=WatchdogStage.OK,
                reason="armed",
                attempt=attempt,
                failures=failures,
                child_pid=0,
                no_output_s=0.0,
                watchdog_s=watchdog_s,
                kill_count=kill_count,
                degrade_level="",
            )
            _jsonl_append(
                wd_events_path,
                {
                    "version": 1,
                    "created_utc": _utc_iso(),
                    "stage": WatchdogStage.OK.name,
                    "reason": "armed",
                    "attempt": int(attempt),
                    "job_root": str(job_root),
                },
            )
        except Exception as e:
            _append_line(sup_log, f"[{_utc_iso()}] watchdog arm emit failed: {e!r}")

    while True:
        if stop_file is not None and stop_file.exists():
            _append_line(sup_log, f"[{_utc_iso()}] stop_file exists: {stop_file} (stopping)")
            break

        if failures > int(args.max_restarts):
            _append_line(sup_log, f"[{_utc_iso()}] max restarts exceeded: failures={failures} (stopping)")
            if watchdog_s > 0.0:
                try:
                    _append_failure_summary(
                        job_root,
                        stage=WatchdogStage.WASHOUT,
                        reason="max_restarts",
                        attempt=attempt,
                        child_pid=0,
                        no_output_s=0.0,
                        watchdog_s=watchdog_s,
                    )
                    _write_log_tail(job_root, out_log, err_log, tail_lines)
                    _write_watchdog_state(
                        wd_state_path,
                        job_root=job_root,
                        stage=WatchdogStage.WASHOUT,
                        reason="max_restarts",
                        attempt=attempt,
                        failures=failures,
                        child_pid=0,
                        no_output_s=0.0,
                        watchdog_s=watchdog_s,
                        kill_count=kill_count,
                        degrade_level="",
                    )
                    _jsonl_append(
                        wd_events_path,
                        {
                            "version": 1,
                            "created_utc": _utc_iso(),
                            "stage": WatchdogStage.WASHOUT.name,
                            "reason": "max_restarts",
                            "attempt": int(attempt),
                            "failures": int(failures),
                            "kill_count": int(kill_count),
                            "job_root": str(job_root),
                        },
                    )
                    _append_line(status_path, f"[{_utc_iso()}] WASHOUT: max_restarts exceeded failures={failures} kill_count={kill_count}")
                except Exception as e:
                    _append_line(sup_log, f"[{_utc_iso()}] washout artifact write failed: {e!r}")
            # Wake the user because this needs manual attention.
            trig = WakeTrigger(
                wake_after_s=int(args.wake_after_s),
                memo=f"[vraxion_lab_supervisor] STOPPED: max restarts exceeded. job_root={job_root}",
                reason="max_restarts",
                job_root=str(job_root),
                attempt=attempt,
                pid=int(os.getpid()),
                window_title=str(args.wake_window_title),
            )
            _atomic_write_json(wake_trigger, trig.to_json())
            break

        attempt += 1
        _append_line(sup_log, f"[{_utc_iso()}] attempt={attempt} launching child")

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Helpful breadcrumb in child logs.
        env["VRX_SUPERVISOR_JOB_ROOT"] = str(job_root)
        env["VRX_SUPERVISOR_ATTEMPT"] = str(int(attempt))
        # If the supervisor previously entered SPILL, re-signal degrade mode on restarts.
        if degrade_path.exists():
            try:
                payload = json.loads(degrade_path.read_text(encoding="utf-8"))
                lvl = str(payload.get("level") or "").strip()
            except Exception:
                lvl = ""
            if lvl:
                env["VRX_SUPERVISOR_DEGRADE_LEVEL"] = lvl

        # Append-only: open in append mode.
        with out_log.open("ab") as out_f, err_log.open("ab") as err_f:
            popen_kwargs: Dict[str, Any] = {
                "env": env,
                "stdout": out_f,
                "stderr": err_f,
                "stdin": subprocess.DEVNULL,
            }

            # Create a new process group so we can kill the tree.
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                popen_kwargs["creationflags"] = int(creationflags)
            else:
                popen_kwargs["start_new_session"] = True

            try:
                child = subprocess.Popen(list(args.child_cmd), **popen_kwargs)  # noqa: S603
            except Exception as e:
                failures += 1
                _append_line(sup_log, f"[{_utc_iso()}] failed to launch child: {e!r}")
                time.sleep(float(args.restart_backoff_s))
                continue

        child_pid = int(getattr(child, "pid", 0) or 0)
        _atomic_write_json(
            state_path,
            {
                "version": 1,
                "updated_utc": _utc_iso(),
                "attempt": int(attempt),
                "failures": int(failures),
                "child_pid": int(child_pid),
                "child_running": True,
                "child_cmd": list(args.child_cmd),
            },
        )

        start_mono = time.monotonic()
        last_hb = start_mono
        last_activity = start_mono
        last_out_sz = out_log.stat().st_size if out_log.exists() else 0
        last_err_sz = err_log.stat().st_size if err_log.exists() else 0

        degrade_level = ""
        stage_prev = WatchdogStage.OK
        last_status_emit = 0.0
        abort_job = False

        poll_s = poll_s_base
        hb_s = float(max(0, int(args.heartbeat_interval_s)))

        while True:
            if stop_file is not None and stop_file.exists():
                _append_line(sup_log, f"[{_utc_iso()}] stop_file exists: {stop_file} (killing child pid={child_pid})")
                _kill_pid_tree(child_pid)
                break

            rc = child.poll()
            if rc is not None:
                break

            # Watchdog: did output logs move?
            try:
                out_sz = out_log.stat().st_size
                err_sz = err_log.stat().st_size
            except Exception:
                out_sz = last_out_sz
                err_sz = last_err_sz

            if out_sz != last_out_sz or err_sz != last_err_sz:
                last_activity = time.monotonic()
                last_out_sz = out_sz
                last_err_sz = err_sz

            no_output_s = time.monotonic() - last_activity
            stage_now = _compute_watchdog_stage(no_output_s, surge_s, spill_s, watchdog_s)

            if stage_now != stage_prev and watchdog_s > 0.0:
                reason = "output_resumed" if stage_now < stage_prev else "no_output"
                ev = {
                    "version": 1,
                    "created_utc": _utc_iso(),
                    "stage": stage_now.name,
                    "reason": reason,
                    "attempt": int(attempt),
                    "child_pid": int(child_pid),
                    "no_output_s": float(no_output_s),
                    "watchdog_no_output_s": float(watchdog_s),
                    "kill_count": int(kill_count),
                    "job_root": str(job_root),
                }
                try:
                    _jsonl_append(wd_events_path, ev)
                    _write_watchdog_state(
                        wd_state_path,
                        job_root=job_root,
                        stage=stage_now,
                        reason=reason,
                        attempt=attempt,
                        failures=failures,
                        child_pid=child_pid,
                        no_output_s=no_output_s,
                        watchdog_s=watchdog_s,
                        kill_count=kill_count,
                        degrade_level=degrade_level,
                    )
                    _append_line(
                        status_path,
                        f"[{_utc_iso()}] {stage_now.name}: no_output_s={no_output_s:.1f} attempt={attempt} pid={child_pid} job_root={job_root}",
                    )
                except Exception as e:
                    _append_line(sup_log, f"[{_utc_iso()}] watchdog emit failed: {e!r}")

                # SPILL: signal "degrade mode" for the *next* restart (child opt-in).
                if stage_now == WatchdogStage.SPILL and degrade_level != "spill":
                    degrade_level = "spill"
                    try:
                        _atomic_write_json(
                            degrade_path,
                            {
                                "version": 1,
                                "updated_utc": _utc_iso(),
                                "level": degrade_level,
                                "job_root": str(job_root),
                            },
                        )
                    except Exception as e:
                        _append_line(sup_log, f"[{_utc_iso()}] degrade_mode write failed: {e!r}")

                stage_prev = stage_now

            # Periodic countdown lines once we enter SURGE/SPILL (append-only eye-sore).
            if watchdog_s > 0.0 and stage_prev >= WatchdogStage.SURGE:
                now_m = time.monotonic()
                if (now_m - last_status_emit) >= max(5.0, poll_s):
                    last_status_emit = now_m
                    if stage_prev == WatchdogStage.SURGE:
                        next_name = "SPILL"
                        next_in = max(0.0, spill_s - no_output_s)
                    elif stage_prev == WatchdogStage.SPILL:
                        next_name = "BREACH"
                        next_in = max(0.0, watchdog_s - no_output_s)
                    else:
                        next_name = ""
                        next_in = 0.0
                    _append_line(
                        status_path,
                        f"[{_utc_iso()}] clock: stage={stage_prev.name} no_output_s={no_output_s:.1f} next={next_name} in {next_in:.1f}s attempt={attempt} pid={child_pid}",
                    )

            # BREACH: kill+restart (legacy watchdog). Optionally abort after N kills (WASHOUT).
            if watchdog_s > 0.0 and stage_now == WatchdogStage.BREACH:
                _append_line(
                    sup_log,
                    f"[{_utc_iso()}] watchdog: no output for {watchdog_s:.1f}s; killing child pid={child_pid}",
                )
                try:
                    _append_failure_summary(
                        job_root,
                        stage=WatchdogStage.BREACH,
                        reason="no_output",
                        attempt=attempt,
                        child_pid=child_pid,
                        no_output_s=no_output_s,
                        watchdog_s=watchdog_s,
                    )
                    _write_log_tail(job_root, out_log, err_log, tail_lines)
                except Exception as e:
                    _append_line(sup_log, f"[{_utc_iso()}] failure artifact write failed: {e!r}")

                kill_count += 1
                if abort_after_kills > 0 and kill_count >= abort_after_kills:
                    abort_job = True
                    try:
                        _append_failure_summary(
                            job_root,
                            stage=WatchdogStage.WASHOUT,
                            reason="abort_after_kills",
                            attempt=attempt,
                            child_pid=child_pid,
                            no_output_s=no_output_s,
                            watchdog_s=watchdog_s,
                        )
                        _write_watchdog_state(
                            wd_state_path,
                            job_root=job_root,
                            stage=WatchdogStage.WASHOUT,
                            reason="abort_after_kills",
                            attempt=attempt,
                            failures=failures,
                            child_pid=child_pid,
                            no_output_s=no_output_s,
                            watchdog_s=watchdog_s,
                            kill_count=kill_count,
                            degrade_level=degrade_level,
                        )
                        _jsonl_append(
                            wd_events_path,
                            {
                                "version": 1,
                                "created_utc": _utc_iso(),
                                "stage": WatchdogStage.WASHOUT.name,
                                "reason": "abort_after_kills",
                                "attempt": int(attempt),
                                "child_pid": int(child_pid),
                                "no_output_s": float(no_output_s),
                                "watchdog_no_output_s": float(watchdog_s),
                                "kill_count": int(kill_count),
                                "job_root": str(job_root),
                            },
                        )
                    except Exception:
                        pass
                    _append_line(status_path, f"[{_utc_iso()}] WASHOUT: abort_after_kills={abort_after_kills} (aborting)")
                _kill_pid_tree(child_pid)
                time.sleep(0.5)
                break

            # Heartbeat wake trigger (best-effort; do not overwrite an existing trigger).
            if hb_s > 0.0 and (time.monotonic() - last_hb) >= hb_s:
                last_hb = time.monotonic()
                if not wake_trigger.exists():
                    elapsed = time.monotonic() - start_mono
                    trig = WakeTrigger(
                        wake_after_s=int(args.wake_after_s),
                        memo=(
                            f"[vraxion_lab_supervisor] heartbeat: still running (attempt={attempt} pid={child_pid} "
                            f"elapsed_s={int(elapsed)}). job_root={job_root}"
                        ),
                        reason="heartbeat",
                        job_root=str(job_root),
                        attempt=int(attempt),
                        pid=int(os.getpid()),
                        window_title=str(args.wake_window_title),
                    )
                    _atomic_write_json(wake_trigger, trig.to_json())
                    _append_line(sup_log, f"[{_utc_iso()}] wrote heartbeat wake trigger: {wake_trigger}")

            time.sleep(poll_s)

        # Child ended (or watchdog killed it).
        rc = child.poll()
        if rc is None:
            try:
                rc = child.wait(timeout=5.0)
            except Exception:
                rc = -1
        rc = int(rc)

        dur_s = time.monotonic() - start_mono
        _append_line(sup_log, f"[{_utc_iso()}] child exit rc={rc} dur_s={dur_s:.1f} pid={child_pid}")

        _atomic_write_json(
            state_path,
            {
                "version": 1,
                "updated_utc": _utc_iso(),
                "attempt": int(attempt),
                "failures": int(failures),
                "child_pid": int(child_pid),
                "child_running": False,
                "child_exit_code": int(rc),
                "child_duration_s": float(dur_s),
                "child_cmd": list(args.child_cmd),
            },
        )

        if abort_job:
            exit_code = 3
            _append_line(sup_log, f"[{_utc_iso()}] WASHOUT: aborting supervisor after watchdog kill (kill_count={kill_count})")
            if not wake_trigger.exists():
                trig = WakeTrigger(
                    wake_after_s=int(args.wake_after_s),
                    memo=f"[vraxion_lab_supervisor] WASHOUT: watchdog abort (kill_count={kill_count}). job_root={job_root}",
                    reason="washout",
                    job_root=str(job_root),
                    attempt=int(attempt),
                    pid=int(os.getpid()),
                    window_title=str(args.wake_window_title),
                )
                _atomic_write_json(wake_trigger, trig.to_json())
                _append_line(sup_log, f"[{_utc_iso()}] wrote washout wake trigger: {wake_trigger}")
            break

        if rc == 0 and bool(args.stop_on_success):
            trig = WakeTrigger(
                wake_after_s=int(args.wake_after_s),
                memo=f"[vraxion_lab_supervisor] DONE rc=0 attempt={attempt} job_root={job_root}",
                reason="done",
                job_root=str(job_root),
                attempt=int(attempt),
                pid=int(os.getpid()),
                window_title=str(args.wake_window_title),
            )
            _atomic_write_json(wake_trigger, trig.to_json())
            _append_line(sup_log, f"[{_utc_iso()}] wrote done wake trigger: {wake_trigger}")
            break

        failures += 1

        # Wake user on crash, but do not overwrite an existing trigger (avoid spam).
        if not wake_trigger.exists():
            trig = WakeTrigger(
                wake_after_s=int(args.wake_after_s),
                memo=(
                    f"[vraxion_lab_supervisor] CRASH rc={rc} attempt={attempt} failures={failures} "
                    f"(auto-restarting). job_root={job_root}"
                ),
                reason="crash",
                job_root=str(job_root),
                attempt=int(attempt),
                pid=int(os.getpid()),
                window_title=str(args.wake_window_title),
            )
            _atomic_write_json(wake_trigger, trig.to_json())
            _append_line(sup_log, f"[{_utc_iso()}] wrote crash wake trigger: {wake_trigger}")

        # Exponential-ish backoff with a cap.
        backoff = float(args.restart_backoff_s) * min(6.0, 1.0 + 0.5 * float(failures))
        _append_line(sup_log, f"[{_utc_iso()}] backoff {backoff:.1f}s before restart")
        time.sleep(backoff)

    _append_line(sup_log, f"[{_utc_iso()}] supervisor exit job_root={job_root}")
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
