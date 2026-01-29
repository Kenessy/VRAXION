"""V_COG log parsing utilities.

This module is a small, stdlib-only parser for VRAXION training logs that
include a stable header of the form:

    V_COG[KEY:VAL KEY2:VAL2 ...]

The contract is intentionally simple and behavior-locked:
  - Numeric values are coerced to **float** (even integer-looking tokens).
  - Percent values like ``100.0%`` are coerced to float without the ``%``.
  - ``parse_line`` returns ``(None, None)`` when the line contains no metrics.
  - ``OnlineStats`` uses Welford mean/variance and reports *sample* std.

These helpers are used by ``tools.parse_vcog``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import re
from typing import Any, Dict, Optional, Tuple


RESTEP = re.compile(r"\bstep\s+(\d+)(?:/\d+)?\b.*?\bloss\s+([0-9.+\-eE]+)")
REVCOG = re.compile(r"V_COG\[(.*?)\]")


def _now_utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _try_float(valstr: str) -> Optional[float]:
    try:
        return float(valstr)
    except Exception:
        return None


def parse_vcog_kv(blob6x: str) -> Dict[str, Any]:
    """Parse the inner content of a ``V_COG[...]`` header into a dict."""

    outmap: Dict[str, Any] = {}
    for tokstr in blob6x.split():
        if ":" not in tokstr:
            continue
        keystr, valstr = tokstr.split(":", 1)
        keystr = keystr.strip()
        valstr = valstr.strip()
        if not keystr:
            continue

        # Common pattern: 100.0%
        if valstr.endswith("%"):
            numval = _try_float(valstr[:-1])
            outmap[keystr] = numval if numval is not None else valstr
            continue

        numval = _try_float(valstr)
        outmap[keystr] = numval if numval is not None else valstr

    return outmap


def parse_line(linstr: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Parse a single log line.

    Returns:
        (metrics_event, vcog_dict)
          - metrics_event contains: ts_utc, step (optional), loss (optional)
          - vcog_dict contains parsed V_COG fields if present
    """

    evmapx: Dict[str, Any] = {"ts_utc": _now_utc_iso()}

    stpmat = RESTEP.search(linstr)
    if stpmat:
        evmapx["step"] = int(stpmat.group(1))
        evmapx["loss"] = float(stpmat.group(2))

    vcgmat = REVCOG.search(linstr)
    vcog = None
    if vcgmat:
        vcog = parse_vcog_kv(vcgmat.group(1))

    if len(evmapx) == 1 and vcog is None:
        return None, None
    return evmapx, vcog


@dataclasses.dataclass
class OnlineStats:
    """Welford online mean/variance (sample std)."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    vmin: float = float("inf")
    vmax: float = float("-inf")

    def update(self, valnum: float) -> None:
        if math.isnan(valnum) or math.isinf(valnum):
            return
        self.n += 1
        if valnum < self.vmin:
            self.vmin = valnum
        if valnum > self.vmax:
            self.vmax = valnum
        delval = valnum - self.mean
        self.mean += delval / self.n
        del2vl = valnum - self.mean
        self.m2 += delval * del2vl

    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean if self.n else None,
            "std": self.std() if self.n else None,
            "min": self.vmin if self.n else None,
            "max": self.vmax if self.n else None,
        }


def dump_json(pthobj, objval: Any) -> None:
    """Write JSON with stable indentation + key ordering."""

    with open(pthobj, "w", encoding="utf-8") as filobj:
        json.dump(objval, filobj, indent=2, sort_keys=True)
