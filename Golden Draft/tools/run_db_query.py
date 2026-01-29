from __future__ import annotations

"""Query helper for the run_db SQLite index.

This intentionally stays simple and dependency-free:
- list recent runs
- show a run's metadata
- grep stdout/stderr logs for a pattern
"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence


DEFAULT_DB_ROOT = Path(r"S:\AI\Golden Draft\vault\db")


def _db_path(db_root: Path) -> Path:
    return db_root / "runs.sqlite"


def _iter_rows(con: sqlite3.Connection, sql: str, params: Sequence[object]) -> Iterable[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    cur = con.execute(sql, list(params))
    for row in cur:
        yield row


def cmd_list(*, db_root: Path, limit: int, exit_code: Optional[int]) -> int:
    dbp = _db_path(db_root)
    if not dbp.exists():
        print(f"[run_db_query] missing db: {dbp}", file=sys.stderr)
        return 2

    where = ""
    params: list[object] = []
    if exit_code is not None:
        where = "WHERE exit_code = ?"
        params.append(int(exit_code))

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

    ap_show = sub.add_parser("show", help="Show full DB row for a run_id.")
    ap_show.add_argument("run_id")

    ap_grep = sub.add_parser("grep", help="Search stdout/stderr/metrics for a regex pattern.")
    ap_grep.add_argument("pattern")
    ap_grep.add_argument("--run-id", default=None)
    ap_grep.add_argument("-i", "--ignore-case", action="store_true")

    args = ap.parse_args(list(argv) if argv is not None else None)
    db_root = Path(args.db_root)

    if args.cmd == "list":
        return cmd_list(db_root=db_root, limit=int(args.limit), exit_code=args.exit_code)
    if args.cmd == "show":
        return cmd_show(db_root=db_root, run_id=str(args.run_id))
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

