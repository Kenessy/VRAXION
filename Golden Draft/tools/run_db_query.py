from __future__ import annotations

"""Query helper for the run_db SQLite index.

This intentionally stays simple and dependency-free:
- list recent runs
- show a run's metadata
- show env snapshot (VRX_/VAR_)
- tail streaming events (step/loss/V_COG)
- grep stdout/stderr logs for a pattern
"""

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


DEFAULT_DB_ROOT = Path(r"S:\AI\Golden Draft\vault\db")


def _db_path(db_root: Path) -> Path:
    return db_root / "runs.sqlite"


def _iter_rows(con: sqlite3.Connection, sql: str, params: Sequence[object]) -> Iterable[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    cur = con.execute(sql, list(params))
    for row in cur:
        yield row


def _parse_kv_filters(items: Sequence[str]) -> list[Tuple[str, str]]:
    out: list[Tuple[str, str]] = []
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--env expects KEY=VAL, got: {it}")
        k, v = it.split("=", 1)
        out.append((k, v))
    return out


def cmd_list(*, db_root: Path, limit: int, exit_code: Optional[int], env_filters: Sequence[str]) -> int:
    dbp = _db_path(db_root)
    if not dbp.exists():
        print(f"[run_db_query] missing db: {dbp}", file=sys.stderr)
        return 2

    where = ""
    params: list[object] = []
    if exit_code is not None:
        where = "WHERE exit_code = ?"
        params.append(int(exit_code))

    env_pairs = _parse_kv_filters(list(env_filters))
    if env_pairs:
        # AND EXISTS(...) per filter so multiple filters are conjunctive.
        joiner = " AND " if where else "WHERE "
        for k, v in env_pairs:
            where += (
                f"{joiner}EXISTS ("
                "SELECT 1 FROM env_kv e "
                "WHERE e.run_id = runs.run_id AND e.env_key = ? AND e.env_val = ?"
                ")"
            )
            params.extend([k, v])
            joiner = " AND "

    sql = (
        "SELECT run_id, start_utc, duration_s, exit_code, run_name, tags "
        "FROM runs "
        f"{where} "
        "ORDER BY start_utc DESC "
        "LIMIT ?"
    )
    params.append(int(limit))

    con = sqlite3.connect(str(dbp))
    try:
        for row in _iter_rows(con, sql, params):
            tags = row["tags"] or ""
            rn = row["run_name"] or ""
            dur = row["duration_s"]
            dur_s = f"{dur:.3f}" if isinstance(dur, (int, float)) else ""
            print(f"{row['start_utc']}  exit={row['exit_code']}  dur_s={dur_s:>8}  {row['run_id']}  {rn}  {tags}")
    finally:
        con.close()
    return 0


def cmd_show(*, db_root: Path, run_id: str) -> int:
    dbp = _db_path(db_root)
    if not dbp.exists():
        print(f"[run_db_query] missing db: {dbp}", file=sys.stderr)
        return 2

    con = sqlite3.connect(str(dbp))
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT * FROM runs WHERE run_id = ?", [run_id]).fetchone()
    finally:
        con.close()
    if row is None:
        print(f"[run_db_query] unknown run_id: {run_id}", file=sys.stderr)
        return 2

    for k in row.keys():
        print(f"{k}: {row[k]}")
    return 0


def cmd_env(*, db_root: Path, run_id: str) -> int:
    dbp = _db_path(db_root)
    if not dbp.exists():
        print(f"[run_db_query] missing db: {dbp}", file=sys.stderr)
        return 2

    con = sqlite3.connect(str(dbp))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT env_key, env_val FROM env_kv WHERE run_id = ? ORDER BY env_key ASC",
            [run_id],
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return 1

    for r in rows:
        print(f"{r['env_key']}={r['env_val']}")
    return 0


def _latest_run_id(con: sqlite3.Connection) -> Optional[str]:
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT run_id FROM runs ORDER BY start_utc DESC LIMIT 1").fetchone()
    if row is None:
        return None
    return str(row["run_id"])


def cmd_events(*, db_root: Path, run_id: str, limit: int, follow: bool, interval_s: float) -> int:
    dbp = _db_path(db_root)
    if not dbp.exists():
        print(f"[run_db_query] missing db: {dbp}", file=sys.stderr)
        return 2

    con = sqlite3.connect(str(dbp))
    con.row_factory = sqlite3.Row

    try:
        if not run_id:
            rid = _latest_run_id(con)
            if rid is None:
                print("[run_db_query] no runs in db", file=sys.stderr)
                return 2
            run_id = rid

        last_seq = -1
        while True:
            if follow:
                rows = con.execute(
                    "SELECT seq, ts_utc, stream, step, loss, vcog_json "
                    "FROM events WHERE run_id = ? AND seq > ? "
                    "ORDER BY seq ASC",
                    [run_id, last_seq],
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT seq, ts_utc, stream, step, loss, vcog_json "
                    "FROM events WHERE run_id = ? "
                    "ORDER BY seq DESC LIMIT ?",
                    [run_id, int(limit)],
                ).fetchall()
                rows = list(reversed(rows))

            for r in rows:
                last_seq = max(last_seq, int(r["seq"]))
                step = r["step"]
                loss = r["loss"]
                msg = f"{r['ts_utc']}  {r['stream']}"
                if step is not None:
                    msg += f"  step={int(step)}"
                if loss is not None:
                    msg += f"  loss={float(loss):.6g}"
                if r["vcog_json"]:
                    msg += "  vcog=1"
                print(msg)

            if not follow:
                break
            time.sleep(max(0.05, float(interval_s)))
    finally:
        con.close()

    return 0


def _iter_log_lines(path: Path) -> Iterable[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                yield ln.rstrip("\n")
    except OSError:
        return


def cmd_grep(*, db_root: Path, pattern: str, run_id: Optional[str], ignore_case: bool) -> int:
    runs_dir = db_root / "runs"
    if not runs_dir.exists():
        print(f"[run_db_query] missing runs dir: {runs_dir}", file=sys.stderr)
        return 2

    flags = re.IGNORECASE if ignore_case else 0
    rx = re.compile(pattern, flags=flags)

    targets: list[Path] = []
    if run_id:
        targets.append(runs_dir / run_id)
    else:
        targets.extend(sorted([p for p in runs_dir.iterdir() if p.is_dir()], reverse=True))

    hits = 0
    for run_dir in targets:
        for fn in ("stdout.log", "stderr.log", "metrics.jsonl"):
            fp = run_dir / fn
            if not fp.exists():
                continue
            for ln in _iter_log_lines(fp):
                if rx.search(ln):
                    print(f"{run_dir.name}/{fn}: {ln}")
                    hits += 1

    if hits == 0:
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Query run_db sqlite + logs.")
    ap.add_argument("--db-root", default=str(DEFAULT_DB_ROOT), help="DB root dir (contains runs.sqlite + runs/).")

    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_list = sub.add_parser("list", help="List recent runs.")
    ap_list.add_argument("--limit", type=int, default=30)
    ap_list.add_argument("--exit-code", type=int, default=None)
    ap_list.add_argument("--env", action="append", default=[], help="Filter by env_kv KEY=VAL (repeatable).")

    ap_show = sub.add_parser("show", help="Show full DB row for a run_id.")
    ap_show.add_argument("run_id")

    ap_env = sub.add_parser("env", help="Show env_kv snapshot (VRX_/VAR_) for a run_id.")
    ap_env.add_argument("run_id")

    ap_events = sub.add_parser("events", help="Show/tail parsed events (step/loss/V_COG) from sqlite.")
    ap_events.add_argument("--run-id", default="", help="Run id (default: latest run).")
    ap_events.add_argument("--limit", type=int, default=50, help="Show at most N events (non-follow mode).")
    ap_events.add_argument("--follow", action="store_true", help="Poll for new events until interrupted.")
    ap_events.add_argument("--interval", type=float, default=0.5, help="Polling interval seconds for --follow.")

    ap_grep = sub.add_parser("grep", help="Search stdout/stderr/metrics for a regex pattern.")
    ap_grep.add_argument("pattern")
    ap_grep.add_argument("--run-id", default=None)
    ap_grep.add_argument("-i", "--ignore-case", action="store_true")

    args = ap.parse_args(list(argv) if argv is not None else None)
    db_root = Path(args.db_root)

    if args.cmd == "list":
        return cmd_list(
            db_root=db_root,
            limit=int(args.limit),
            exit_code=args.exit_code,
            env_filters=list(args.env),
        )
    if args.cmd == "show":
        return cmd_show(db_root=db_root, run_id=str(args.run_id))
    if args.cmd == "env":
        return cmd_env(db_root=db_root, run_id=str(args.run_id))
    if args.cmd == "events":
        return cmd_events(
            db_root=db_root,
            run_id=str(args.run_id),
            limit=int(args.limit),
            follow=bool(args.follow),
            interval_s=float(args.interval),
        )
    if args.cmd == "grep":
        return cmd_grep(
            db_root=db_root,
            pattern=str(args.pattern),
            run_id=(str(args.run_id) if args.run_id else None),
            ignore_case=bool(args.ignore_case),
        )

    print(f"[run_db_query] unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

