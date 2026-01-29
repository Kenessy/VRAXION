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
from typing import Any, Dict, Iterable, List, Optional, Sequence


RESTEP = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[\d\.]+)\s+\|"
    r".*?raw_delta=(?P<raw_delta>[\d\.\-]+).*?shard=(?P<shard_count>[\d\-]+)/(?P<shard_size>[\d\-]+)"
    r"(?:,\s*traction=(?P<traction>[\d\.\-]+))?"
)
REGRAD = re.compile(r"grad_norm\(theta_ptr\)=(?P<grad>[\d\.\+eE\-]+)")


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

        trcstr = stpmat.group("traction")
        trac6x = _tryflt(trcstr) if trcstr is not None else None

        rowdat: Dict[str, Any] = {
            "step": int(stpmat.group("step")),
            "loss": float(stpmat.group("loss")),
            "raw_delta": float(stpmat.group("raw_delta")),
            "shard_count": float(stpmat.group("shard_count")),
            "shard_size": float(stpmat.group("shard_size")),
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

    try:
        dfobj6 = parse_log(logpth)
    except Exception as exc:
        stmod6.error(f"Failed to parse log: {exc}")
        return

    if dfobj6.empty:
        stmod6.info("No data parsed yet.")
        stmod6.caption(f"Waiting for log to populate: {logpth}")
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
