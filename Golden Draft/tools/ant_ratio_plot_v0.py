"""VRA-78: Plot an ant-ratio frontier (single-file HTML, no pip deps).

Input: JSONL packets produced by ant_ratio_packet_v0.py / ant_ratio_sweep_v0.py.
Output: A single HTML file with embedded data and a 3D scatter plot.

Implementation choice (v0):
- Use Plotly via CDN to avoid Python dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def _as_float(val: Any) -> Optional[float]:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val)


def _as_int(val: Any) -> Optional[int]:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return int(val)


def _html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _tier_color(tier: str) -> str:
    t = str(tier).strip().lower()
    if t == "small":
        return "#58a6ff"  # blue
    if t == "real":
        return "#2ea043"  # green
    if t == "stress":
        return "#d29922"  # yellow
    return "#8b949e"  # gray


def build_html(*, packets: List[Dict[str, Any]], title: str) -> str:
    # Partition points.
    pts: List[Dict[str, Any]] = []
    for p in packets:
        x = _as_float(p.get("vram_ratio_reserved"))
        y = _as_float(p.get("throughput_tokens_per_s"))
        z = _as_float(p.get("assoc_byte_disjoint_accuracy"))
        if x is None or y is None or z is None:
            continue
        pts.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "tier": p.get("ant_tier") or "unknown",
                "heads": _as_int(p.get("expert_heads")) or 0,
                "batch": _as_int(p.get("batch_size")) or 0,
                "pass": bool(p.get("stability_pass") is True),
                "probe": p.get("probe_run_root") or "",
                "assoc": p.get("assoc_run_root") or "",
                "fail": ",".join([str(x) for x in (p.get("fail_reasons") or [])]),
            }
        )

    # Build traces by tier for readability.
    tiers = sorted({str(r["tier"]) for r in pts})
    traces: List[Dict[str, Any]] = []
    for tier in tiers:
        tier_color = _tier_color(tier)
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        sizes: List[float] = []
        symbols: List[str] = []
        texts: List[str] = []
        for r in pts:
            if str(r["tier"]) != tier:
                continue
            xs.append(float(r["x"]))
            ys.append(float(r["y"]))
            zs.append(float(r["z"]))
            sizes.append(float(max(6, min(18, 6 + int(r["heads"])))))
            symbols.append("circle" if r["pass"] else "x")
            status = "PASS" if r["pass"] else "FAIL"
            tip = (
                f"{tier} | E={r['heads']} | B={r['batch']} | {status}"
                f"<br>vram={r['x']:.3f} | toks/s={r['y']:.1f} | acc={r['z']:.4f}"
                f"<br>probe={_html_escape(r['probe'])}"
                f"<br>assoc={_html_escape(r['assoc'])}"
            )
            if r["fail"]:
                tip += f"<br>fail={_html_escape(r['fail'])}"
            texts.append(tip)

        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "name": tier,
                "x": xs,
                "y": ys,
                "z": zs,
                "text": texts,
                "hoverinfo": "text",
                "marker": {
                    "size": sizes,
                    "color": tier_color,
                    "symbol": symbols,
                    "opacity": 0.9,
                },
            }
        )

    data_json = json.dumps(traces, separators=(",", ":"), ensure_ascii=True)
    title_esc = _html_escape(title)

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title_esc}</title>
    <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        height: 100%;
        background: #0b0f14;
        color: #e6edf3;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      }}
      #plot {{
        width: 100%;
        height: 100%;
      }}
      .note {{
        position: absolute;
        top: 12px;
        left: 12px;
        padding: 10px 12px;
        background: rgba(22, 27, 34, 0.75);
        border: 1px solid rgba(240, 246, 252, 0.12);
        border-radius: 10px;
        font-size: 12px;
        line-height: 1.35;
        max-width: 520px;
      }}
      .note b {{ color: #ffffff; }}
      .note code {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 11px;
      }}
    </style>
  </head>
  <body>
    <div class="note">
      <b>Ant Ratio Frontier v0</b><br />
      X = reserved VRAM ratio, Y = tokens/sec, Z = assoc_byte disjoint accuracy.<br />
      Marker size scales with expert heads (out_dim). Color = ant tier. Symbol: PASS circle, FAIL x.<br />
      Source: <code>ant_ratio_packets.jsonl</code>
    </div>
    <div id="plot"></div>
    <script>
      const data = {data_json};
      const layout = {{
        paper_bgcolor: "#0b0f14",
        plot_bgcolor: "#0b0f14",
        font: {{ color: "#e6edf3" }},
        margin: {{ l: 0, r: 0, t: 30, b: 0 }},
        title: {{ text: "{title_esc}", x: 0.01, xanchor: "left", font: {{ size: 16 }} }},
        scene: {{
          xaxis: {{ title: "vram_ratio_reserved", gridcolor: "rgba(240,246,252,0.10)", zerolinecolor: "rgba(240,246,252,0.15)" }},
          yaxis: {{ title: "throughput_tokens_per_s", gridcolor: "rgba(240,246,252,0.10)", zerolinecolor: "rgba(240,246,252,0.15)" }},
          zaxis: {{ title: "assoc_byte_disjoint_accuracy", gridcolor: "rgba(240,246,252,0.10)", zerolinecolor: "rgba(240,246,252,0.15)" }},
          bgcolor: "#0b0f14"
        }},
        legend: {{ x: 0.01, y: 0.98 }}
      }};
      Plotly.newPlot("plot", data, layout, {{ responsive: true }});
    </script>
  </body>
</html>
"""


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate ant_ratio_frontier_v0.html from packets JSONL.")
    ap.add_argument("--packets", required=True, help="Path to ant_ratio_packets.jsonl")
    ap.add_argument("--out", required=True, help="Output HTML path")
    ap.add_argument("--title", default="VRAXION Ant Ratio Frontier v0")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    packets_path = Path(args.packets).resolve()
    out_path = Path(args.out).resolve()
    if not packets_path.exists():
        print(f"ERROR: packets not found: {packets_path}", file=sys.stderr)
        return 2

    packets = _load_jsonl(packets_path)
    html = build_html(packets=packets, title=str(args.title))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[plot] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
