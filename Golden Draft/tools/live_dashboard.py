"""Live training dashboard for VRAXION logs.

This is "Golden Draft" internal tooling.

Recommended usage:
  streamlit run tools/live_dashboard.py -- --log logs/current/vraxion.log

Design goals
- Import-safe without Streamlit installed (parsers are stdlib-only).
- Keep parsing behavior stable vs the original draft script.

Parsing API
- ``parse_log_lines(lines)`` returns stdlib-only parsed rows.
- ``parse_log(log_path)`` returns a pandas DataFrame (requires pandas).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence


REFLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
RESTEP = re.compile(rf"step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>{REFLOAT})\s+\|(?P<tail>.*)")
REGRAD = re.compile(rf"grad_norm\(theta_ptr\)=(?P<grad>{REFLOAT})")
RERAW = re.compile(rf"raw_delta=(?P<raw_delta>{REFLOAT})")
RERD = re.compile(rf"\bRD:(?P<raw_delta>{REFLOAT})\b")
RESHARD = re.compile(r"shard=(?P<shard_count>\d+)/(?P<shard_size>\d+)")
RETRACT = re.compile(rf"traction=(?P<traction>{REFLOAT})")


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _tryflt(valstr: str) -> Optional[float]:
    try:
        return float(valstr)
    except Exception:
        return None


def parse_log_lines(lines: Iterable[str]) -> List[Dict[str, Any]]:
    """Parse log lines into dict rows.

    Behavior contract (matches original draft script):
    - A ``grad_norm(theta_ptr)=...`` line sets a temporary grad value.
    - That grad value is attached to the *next* parsed step line only.
    - Step rows contain: step, loss, raw_delta, shard_count, shard_size,
      traction (optional), grad_norm (optional).
    """

    rows6x: List[Dict[str, Any]] = []
    grad6x: Optional[float] = None

    for linstr in lines:
        grdmat = REGRAD.search(linstr)
        if grdmat is not None:
            grdstr = grdmat.group("grad")
            grad6x = _tryflt(grdstr)
            continue

        stpmat = RESTEP.search(linstr)
        if stpmat is None:
            continue

        tail = stpmat.group("tail")
        rawmat = RERAW.search(tail)
        if rawmat is None:
            rawmat = RERD.search(tail)

        shdmat = RESHARD.search(tail)
        trcmat = RETRACT.search(tail)

        raw_delta = _tryflt(rawmat.group("raw_delta")) if rawmat is not None else None
        shard_count = _tryflt(shdmat.group("shard_count")) if shdmat is not None else None
        shard_size = _tryflt(shdmat.group("shard_size")) if shdmat is not None else None
        trac6x = _tryflt(trcmat.group("traction")) if trcmat is not None else None

        # For legacy step lines without explicit shard metadata.
        if shard_count is None:
            shard_count = 0.0
        if shard_size is None:
            shard_size = 0.0
        if raw_delta is None:
            raw_delta = 0.0

        rowdat: Dict[str, Any] = {
            "step": int(stpmat.group("step")),
            "loss": float(stpmat.group("loss")),
            "raw_delta": float(raw_delta),
            "shard_count": float(shard_count),
            "shard_size": float(shard_size),
            "traction": trac6x,
            "grad_norm": grad6x,
        }
        rows6x.append(rowdat)

        # Grad applies to only the next step.
        grad6x = None

    return rows6x


def parse_log_file(log_path: str) -> List[Dict[str, Any]]:
    """Read and parse a log file.

    Missing/unreadable files return an empty list.
    """

    if not os.path.exists(log_path):
        return []

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as filobj:
            linlst = filobj.readlines()
    except OSError:
        return []

    return parse_log_lines(linlst)


def _read_tail_lines(path: str, max_lines: int = 40, max_bytes: int = 2_000_000) -> List[str]:
    """Read up to max_lines from the end of a text file.

    Implementation is intentionally streaming-friendly: it avoids reading the
    whole file into memory (important for long-running saturation runs).
    """
    if not os.path.exists(path):
        return []
    if int(max_lines) <= 0:
        return []
    try:
        with open(path, "rb") as filobj:
            filobj.seek(0, os.SEEK_END)
            size = int(filobj.tell())
            start = max(0, size - int(max(1024, max_bytes)))
            filobj.seek(start, os.SEEK_SET)
            data = filobj.read()
    except OSError:
        return []

    text = data.decode("utf-8", errors="replace")
    lines = [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]
    return lines[-int(max_lines) :]


def _read_new_lines(path: str, pos: int, max_bytes: int = 2_000_000) -> tuple[int, List[str]]:
    """Read newly appended lines starting from pos (bytes)."""
    if not os.path.exists(path):
        return pos, []
    try:
        size = int(os.stat(path).st_size)
    except OSError:
        return pos, []

    # Handle truncation/rotation.
    if int(pos) > size:
        pos = 0

    try:
        with open(path, "rb") as filobj:
            filobj.seek(int(pos), os.SEEK_SET)
            data = filobj.read(int(max(0, max_bytes)))
            new_pos = int(filobj.tell())
    except OSError:
        return pos, []

    if not data:
        return new_pos, []

    text = data.decode("utf-8", errors="replace")
    return new_pos, [line for line in text.splitlines() if line.strip()]


def collect_live_status(log_path: str) -> Dict[str, Any]:
    """Collect file-level liveness signals for pre-step phases.

    This avoids a false "nothing running" appearance when training step lines
    are not yet emitted (for example during probe stage).
    """
    out: Dict[str, Any] = {
        "log_exists": bool(os.path.exists(log_path)),
        "log_size_bytes": 0,
        "log_mtime_epoch": None,
        "log_age_s": None,
        "log_tail": [],
        "stderr_tail": [],
        "supervisor_tail": [],
    }
    if not out["log_exists"]:
        return out

    try:
        st = os.stat(log_path)
        out["log_size_bytes"] = int(st.st_size)
        out["log_mtime_epoch"] = float(st.st_mtime)
        out["log_age_s"] = max(0.0, float(time.time()) - float(st.st_mtime))
    except OSError:
        pass

    out["log_tail"] = _read_tail_lines(log_path, max_lines=30)
    base_dir = os.path.dirname(log_path)
    # Support both:
    # - supervisor-captured logs (log_path == .../supervisor_job/child_stdout.log)
    # - run log (log_path == .../train/vraxion.log) where supervisor_job is a sibling.
    cand_dirs = [
        base_dir,
        os.path.join(base_dir, "supervisor_job"),
        os.path.join(os.path.dirname(base_dir), "supervisor_job"),
    ]
    stderr_path = ""
    supervisor_path = ""
    for d in cand_dirs:
        p_err = os.path.join(d, "child_stderr.log")
        p_sup = os.path.join(d, "supervisor.log")
        if not stderr_path and os.path.exists(p_err):
            stderr_path = p_err
        if not supervisor_path and os.path.exists(p_sup):
            supervisor_path = p_sup
    if not stderr_path:
        stderr_path = os.path.join(base_dir, "child_stderr.log")
    if not supervisor_path:
        supervisor_path = os.path.join(base_dir, "supervisor.log")
    out["stderr_tail"] = _read_tail_lines(stderr_path, max_lines=20)
    out["supervisor_tail"] = _read_tail_lines(supervisor_path, max_lines=20)
    return out


def parse_log(log_path: str):
    """Parse a log file into a pandas DataFrame.

    This function requires pandas. Importing this module does not.
    """

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required for parse_log()") from exc

    rows6x = parse_log_file(log_path)
    dfobj6 = pd.DataFrame(rows6x)

    if dfobj6.empty:
        return dfobj6

    if "step" in dfobj6.columns:
        dfobj6 = dfobj6.drop_duplicates(subset=["step"]).sort_values("step")

    # Derived metric used by the dashboard.
    dfobj6["tension"] = dfobj6["grad_norm"] * dfobj6["raw_delta"] / 100.0

    # Clip outliers for plotting readability.
    capval = dfobj6["tension"].quantile(0.99)
    dfobj6["tension"] = dfobj6["tension"].clip(upper=capval)

    return dfobj6


def _req_st() -> Any:
    try:
        import streamlit as stmod6  # type: ignore

        return stmod6
    except Exception:
        _eprint("[live_dashboard] ERROR: streamlit is required to run the dashboard")
        _eprint("[live_dashboard] Hint: pip install streamlit")
        raise


def _opt_mod(modnam: str) -> Any:
    try:
        __import__(modnam)
        return sys.modules[modnam]
    except Exception:
        return None


def _autorf() -> Any:
    """Return an autorefresh callable if available, else None."""

    # Prefer streamlit-autorefresh if installed.
    modobj = _opt_mod("streamlit_autorefresh")
    if modobj is not None:
        try:
            fncobj = getattr(modobj, "st_autorefresh")
            return fncobj
        except Exception:
            pass

    # Back-compat: some environments vend an autorefresh helper under streamlit.
    try:
        from streamlit import autorefresh as fncobj  # type: ignore

        return fncobj
    except Exception:
        return None


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="VRAXION live dashboard")
    parser.add_argument("--log", default=os.path.join("logs", "current", "vraxion.log"))
    parser.add_argument("--refresh", type=int, default=10, help="Auto-refresh seconds (0 disables)")
    parser.add_argument("--max-rows", type=int, default=5000, help="Display at most N rows (0 = all)")

    argobj, _unk6x = parser.parse_known_args(list(argv) if argv is not None else None)

    # Late import so tests can import parsers without Streamlit.
    try:
        stmod6 = _req_st()
    except Exception:
        return

    stmod6.set_page_config(page_title="VRAXION Live Dashboard", layout="wide")
    stmod6.title("VRAXION Live Dashboard")

    logpth = stmod6.sidebar.text_input("Log path", value=str(argobj.log))
    rfrsec = stmod6.sidebar.number_input(
        "Refresh interval (sec)",
        min_value=0,
        max_value=600,
        value=int(argobj.refresh),
    )
    maxrow = stmod6.sidebar.number_input(
        "Max rows",
        min_value=0,
        max_value=500000,
        value=int(argobj.max_rows),
    )

    if int(rfrsec) > 0:
        fncobj = _autorf()
        if fncobj is not None:
            fncobj(interval=int(rfrsec) * 1000, key="auto_refresh")
        else:
            stmod6.sidebar.caption(
                "Auto-refresh helper not available; install streamlit-autorefresh for periodic refresh."
            )

    # Stream-friendly parsing: keep a rolling window of parsed rows in session
    # state, and only read newly appended bytes each refresh. This avoids
    # re-reading huge logs and prevents RAM blowups on long runs.
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        stmod6.error(f"pandas is required for the dashboard: {exc}")
        return

    if "live_log_path" not in stmod6.session_state:
        stmod6.session_state["live_log_path"] = ""
    if "live_log_pos" not in stmod6.session_state:
        stmod6.session_state["live_log_pos"] = 0
    if "live_rows" not in stmod6.session_state:
        stmod6.session_state["live_rows"] = []

    if stmod6.session_state["live_log_path"] != str(logpth):
        stmod6.session_state["live_log_path"] = str(logpth)
        stmod6.session_state["live_log_pos"] = 0
        stmod6.session_state["live_rows"] = []

    # On first load, start near the tail to avoid ingesting a massive file.
    try:
        size0 = int(os.stat(logpth).st_size)
    except OSError:
        size0 = 0
    if stmod6.session_state["live_log_pos"] == 0 and not stmod6.session_state["live_rows"] and size0 > 0:
        stmod6.session_state["live_log_pos"] = max(0, size0 - 2_000_000)

    new_pos, new_lines = _read_new_lines(str(logpth), int(stmod6.session_state["live_log_pos"]))
    stmod6.session_state["live_log_pos"] = int(new_pos)
    if new_lines:
        stmod6.session_state["live_rows"].extend(parse_log_lines(new_lines))

    dfobj6 = pd.DataFrame(stmod6.session_state["live_rows"])
    if not dfobj6.empty and "step" in dfobj6.columns:
        dfobj6 = dfobj6.drop_duplicates(subset=["step"]).sort_values("step")

    # Keep memory bounded even if maxrow is set very high.
    keep_rows = 0
    if int(maxrow) > 0:
        keep_rows = int(maxrow) * 2
    else:
        keep_rows = 20000
    if keep_rows > 0 and len(dfobj6) > keep_rows:
        dfobj6 = dfobj6.tail(keep_rows)

    # Persist the bounded window back to session state.
    stmod6.session_state["live_rows"] = dfobj6.to_dict(orient="records")

    try:
        # Recompute derived metric for plotting readability.
        if not dfobj6.empty:
            if "grad_norm" in dfobj6.columns and "raw_delta" in dfobj6.columns:
                dfobj6["tension"] = dfobj6["grad_norm"] * dfobj6["raw_delta"] / 100.0
                capval = dfobj6["tension"].quantile(0.99)
                dfobj6["tension"] = dfobj6["tension"].clip(upper=capval)
    except Exception as exc:
        stmod6.error(f"Failed to parse log: {exc}")
        return

    if dfobj6.empty:
        stmod6.warning("No step rows parsed yet.")
        stmod6.caption(f"Watching log: {logpth}")

        status = collect_live_status(logpth)
        col1, col2, col3 = stmod6.columns(3)
        with col1:
            stmod6.metric("Log exists", "yes" if status.get("log_exists") else "no")
        with col2:
            stmod6.metric("Log size (bytes)", int(status.get("log_size_bytes") or 0))
        with col3:
            age = status.get("log_age_s")
            stmod6.metric("Last write age (s)", int(age) if isinstance(age, (float, int)) else -1)

        stmod6.info(
            "Run may still be active in probe/eval stage before train step lines. "
            "Use tails below to confirm liveness."
        )

        if status.get("supervisor_tail"):
            stmod6.subheader("Supervisor tail")
            stmod6.code("\n".join(status["supervisor_tail"]), language="text")
        if status.get("stderr_tail"):
            stmod6.subheader("Child stderr tail")
            stmod6.code("\n".join(status["stderr_tail"]), language="text")
        if status.get("log_tail"):
            stmod6.subheader("Child stdout tail")
            stmod6.code("\n".join(status["log_tail"]), language="text")
        return

    if int(maxrow) > 0 and len(dfobj6) > int(maxrow):
        dfobj6 = dfobj6.tail(int(maxrow))

    last6x = dfobj6.iloc[-1]
    stmod6.metric("Step", int(last6x.get("step", 0)))

    # Chart backend: prefer plotly if installed.
    pltx6x = _opt_mod("plotly.express")
    if pltx6x is not None:
        try:
            figlos = pltx6x.line(dfobj6, x="step", y="loss", title="Loss")
            stmod6.plotly_chart(figlos, use_container_width=True)

            figten = pltx6x.line(dfobj6, x="step", y="tension", title="Tension")
            stmod6.plotly_chart(figten, use_container_width=True)
        except Exception as exc:
            stmod6.warning(f"Plotly render failed, falling back to table only: {exc}")
    else:
        stmod6.caption("Plotly not installed; showing table only.")

    stmod6.subheader("Latest rows")
    stmod6.dataframe(dfobj6.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()
