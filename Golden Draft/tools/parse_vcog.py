"""Parse a VRAXION training log into JSONL metrics.

Produces:
  - ``metrics.jsonl``: one JSON object per parsed event
  - ``summary.json``: online loss stats and the most recent V_COG header

This is an offline helper and intentionally lives in the top-level ``tools``
namespace (Golden Draft).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    # Preferred invocation: `python -m tools.parse_vcog ...`
    from .vcog_parse import OnlineStats, dump_json, parse_line
except ImportError:  # pragma: no cover
    # Fallback for: `python tools/parse_vcog.py ...` (adds repo root to sys.path).
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.vcog_parse import OnlineStats, dump_json, parse_line


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse a VRAXION log file into metrics.jsonl + summary.json."
    )
    ap.add_argument("--log", required=True, help="Path to a log file (stdout/stderr).")
    ap.add_argument("--out-dir", required=True, help="Output directory (will be created).")
    args = ap.parse_args(argv)

    logpth = Path(args.log)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    metpth = outdir / "metrics.jsonl"
    sumpth = outdir / "summary.json"

    lossts = OnlineStats()
    lastvc: Dict[str, Any] = {}

    with open(logpth, "r", encoding="utf-8", errors="replace") as finobj, open(
        metpth, "w", encoding="utf-8"
    ) as metobj:
        for linstr in finobj:
            evmapx, vcog = parse_line(linstr)
            if evmapx is None and vcog is None:
                continue
            if evmapx is None:
                evmapx = {}
            if vcog is not None:
                evmapx["vcog"] = vcog
                lastvc.clear()
                lastvc.update(vcog)
            if "loss" in evmapx:
                lossts.update(float(evmapx["loss"]))
            metobj.write(json.dumps(evmapx, sort_keys=True) + "\n")

    summary = {"loss": lossts.to_dict(), "vcog_last": lastvc or None}
    dump_json(sumpth, summary)
    print(str(outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
