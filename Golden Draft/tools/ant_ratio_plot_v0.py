"""VRA-78: Plot an ant-ratio frontier (single-file HTML, no pip deps).

Input: JSONL packets produced by ant_ratio_packet_v0.py / ant_ratio_sweep_v0.py.
Output: A single HTML file with embedded data and a small dashboard.

Implementation choice (v0):
- Use Plotly via CDN to avoid Python dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _tier_symbol(tier: str) -> str:
    t = str(tier).strip().lower()
    if t == "small":
        return "circle"
    if t == "real":
        return "diamond"
    if t == "stress":
        return "square"
    return "circle-open"


def _compute_desirability(
    *,
    vram_ratio_reserved: float,
    throughput_tokens_per_s: float,
    stability_pass: bool,
    target_vram_ratio: float,
    ratio_sigma: float,
    max_log_tok: float,
) -> Tuple[float, float, float]:
    tokens = max(0.0, float(throughput_tokens_per_s))
    ratio = max(0.0, float(vram_ratio_reserved))
    tok_log_norm = (math.log1p(tokens) / max_log_tok) if max_log_tok > 0.0 else 0.0
    ratio_fit = math.exp(-(((ratio - target_vram_ratio) / max(ratio_sigma, 1e-9)) ** 2))
    if not stability_pass:
        return (tok_log_norm, ratio_fit, 0.0)
    return (tok_log_norm, ratio_fit, max(0.0, min(1.0, tok_log_norm * ratio_fit)))


def build_html(*, packets: List[Dict[str, Any]], title: str) -> str:
    # Partition points.
    pts: List[Dict[str, Any]] = []
    for p in packets:
        x = _as_float(p.get("vram_ratio_reserved"))
        y = _as_float(p.get("throughput_tokens_per_s"))
        z = _as_float(p.get("assoc_byte_disjoint_accuracy"))
        if x is None or y is None or z is None:
            continue
        token_budget_steps = _as_int(p.get("token_budget_steps"))
        assoc_eval_n = _as_int(p.get("assoc_eval_n"))
        is_pass = bool(p.get("stability_pass") is True)
        is_rankable = bool(is_pass and (token_budget_steps or 0) >= 30 and (assoc_eval_n or 0) >= 512)
        pts.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "tier": p.get("ant_tier") or "unknown",
                "heads": _as_int(p.get("expert_heads")) or 0,
                "batch": _as_int(p.get("batch_size")) or 0,
                "pass": is_pass,
                "probe": p.get("probe_run_root") or "",
                "assoc": p.get("assoc_run_root") or "",
                "fail": ",".join([str(x) for x in (p.get("fail_reasons") or [])]),
                "generated_utc": p.get("generated_utc") or "",
                "git_commit": p.get("git_commit") or "",
                "workload_id": p.get("workload_id") or "",
                "token_budget_steps": token_budget_steps,
                "assoc_eval_n": assoc_eval_n,
                "rankable": is_rankable,
            }
        )

    target_vram_ratio = 0.85
    ratio_sigma = 0.10
    pass_pts = [row for row in pts if row.get("pass") is True]
    max_log_tok = (
        max(math.log1p(max(0.0, float(row["y"]))) for row in pass_pts) if pass_pts else 0.0
    )

    for row in pts:
        tok_log_norm, ratio_fit, desirability = _compute_desirability(
            vram_ratio_reserved=float(row["x"]),
            throughput_tokens_per_s=float(row["y"]),
            stability_pass=bool(row.get("pass") is True),
            target_vram_ratio=target_vram_ratio,
            ratio_sigma=ratio_sigma,
            max_log_tok=max_log_tok,
        )
        row["tok_log_norm"] = tok_log_norm
        row["ratio_fit"] = ratio_fit
        row["desirability"] = desirability

    # Build traces by tier for readability.
    tiers = sorted({str(r["tier"]) for r in pts})
    traces3d: List[Dict[str, Any]] = []
    traces_vram_tok: List[Dict[str, Any]] = []
    traces_heads_tok: List[Dict[str, Any]] = []
    traces_heads_acc: List[Dict[str, Any]] = []

    show_scale = True
    for tier in tiers:
        tier_color = _tier_color(tier)
        tier_symbol = _tier_symbol(tier)
        tier_pts = [r for r in pts if str(r.get("tier")) == tier]
        if not tier_pts:
            continue

        # 3D: show all points (PASS and FAIL); symbol marks failures.
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        sizes: List[float] = []
        symbols: List[str] = []
        colors: List[float] = []
        texts: List[str] = []

        for r in tier_pts:
            xs.append(float(r["x"]))
            ys.append(float(r["y"]))
            zs.append(float(r["z"]))
            colors.append(float(r.get("desirability") or 0.0))
            sizes.append(float(max(6, min(18, 6 + int(r["heads"])))))
            symbols.append(tier_symbol if r["pass"] else "x")
            status = "PASS" if r["pass"] else "FAIL"
            tip = (
                f"{tier} | E={r['heads']} | B={r['batch']} | {status}"
                f"<br>vram={r['x']:.3f} | toks/s={r['y']:.1f} | acc={r['z']:.4f}"
                f"<br>des={float(r.get('desirability') or 0.0):.3f} | fit={float(r.get('ratio_fit') or 0.0):.3f} | logtok={float(r.get('tok_log_norm') or 0.0):.3f}"
                f"<br>probe={_html_escape(r['probe'])}"
                f"<br>assoc={_html_escape(r['assoc'])}"
            )
            if r["fail"]:
                tip += f"<br>fail={_html_escape(r['fail'])}"
            texts.append(tip)

        traces3d.append(
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
                    "color": colors,
                    "colorscale": "RdYlGn",
                    "cmin": 0.0,
                    "cmax": 1.0,
                    "showscale": show_scale,
                    "colorbar": {"title": "desirability"} if show_scale else None,
                    "symbol": symbols,
                    "opacity": 0.9,
                    "line": {"width": 0.6, "color": "rgba(240,246,252,0.16)"},
                },
            }
        )
        show_scale = False

        # 2D projections: draw lines only through PASS points (sorted), show FAIL as x markers.
        pass_pts_tier = [r for r in tier_pts if bool(r.get("pass") is True)]
        fail_pts_tier = [r for r in tier_pts if not bool(r.get("pass") is True)]

        def _tip(r: Dict[str, Any]) -> str:
            status = "PASS" if r.get("pass") else "FAIL"
            tip = (
                f"{tier} | E={int(r.get('heads') or 0)} | B={int(r.get('batch') or 0)} | {status}"
                f"<br>vram={float(r.get('x') or 0.0):.3f} | toks/s={float(r.get('y') or 0.0):.1f} | acc={float(r.get('z') or 0.0):.4f}"
                f"<br>des={float(r.get('desirability') or 0.0):.3f}"
            )
            if r.get("probe"):
                tip += f"<br>probe={_html_escape(str(r.get('probe')))}"
            if r.get("assoc"):
                tip += f"<br>assoc={_html_escape(str(r.get('assoc')))}"
            if r.get("fail"):
                tip += f"<br>fail={_html_escape(str(r.get('fail')))}"
            return tip

        pass_pts_vram = sorted(pass_pts_tier, key=lambda r: float(r.get("x") or 0.0))
        fail_pts_vram = sorted(fail_pts_tier, key=lambda r: float(r.get("x") or 0.0))
        traces_vram_tok.append(
            {
                "type": "scatter",
                "mode": "markers+lines",
                "name": tier,
                "x": [float(r.get("x") or 0.0) for r in pass_pts_vram],
                "y": [float(r.get("y") or 0.0) for r in pass_pts_vram],
                "text": [_tip(r) for r in pass_pts_vram],
                "hoverinfo": "text",
                "marker": {"size": [float(max(6, min(18, 6 + int(r.get("heads") or 0)))) for r in pass_pts_vram], "color": tier_color, "opacity": 0.9},
                "line": {"width": 1.2, "color": tier_color},
            }
        )
        if fail_pts_vram:
            traces_vram_tok.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "name": f"{tier} (fail)",
                    "x": [float(r.get("x") or 0.0) for r in fail_pts_vram],
                    "y": [float(r.get("y") or 0.0) for r in fail_pts_vram],
                    "text": [_tip(r) for r in fail_pts_vram],
                    "hoverinfo": "text",
                    "marker": {"size": 10, "color": tier_color, "opacity": 0.55, "symbol": "x"},
                }
            )

        pass_pts_heads = sorted(pass_pts_tier, key=lambda r: int(r.get("heads") or 0))
        fail_pts_heads = sorted(fail_pts_tier, key=lambda r: int(r.get("heads") or 0))
        traces_heads_tok.append(
            {
                "type": "scatter",
                "mode": "markers+lines",
                "name": tier,
                "x": [int(r.get("heads") or 0) for r in pass_pts_heads],
                "y": [float(r.get("y") or 0.0) for r in pass_pts_heads],
                "text": [_tip(r) for r in pass_pts_heads],
                "hoverinfo": "text",
                "marker": {"size": 9, "color": tier_color, "opacity": 0.9},
                "line": {"width": 1.2, "color": tier_color},
            }
        )
        if fail_pts_heads:
            traces_heads_tok.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "name": f"{tier} (fail)",
                    "x": [int(r.get("heads") or 0) for r in fail_pts_heads],
                    "y": [float(r.get("y") or 0.0) for r in fail_pts_heads],
                    "text": [_tip(r) for r in fail_pts_heads],
                    "hoverinfo": "text",
                    "marker": {"size": 10, "color": tier_color, "opacity": 0.55, "symbol": "x"},
                }
            )

        traces_heads_acc.append(
            {
                "type": "scatter",
                "mode": "markers+lines",
                "name": tier,
                "x": [int(r.get("heads") or 0) for r in pass_pts_heads],
                "y": [float(r.get("z") or 0.0) for r in pass_pts_heads],
                "text": [_tip(r) for r in pass_pts_heads],
                "hoverinfo": "text",
                "marker": {"size": 9, "color": tier_color, "opacity": 0.9},
                "line": {"width": 1.2, "color": tier_color},
            }
        )
        if fail_pts_heads:
            traces_heads_acc.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "name": f"{tier} (fail)",
                    "x": [int(r.get("heads") or 0) for r in fail_pts_heads],
                    "y": [float(r.get("z") or 0.0) for r in fail_pts_heads],
                    "text": [_tip(r) for r in fail_pts_heads],
                    "hoverinfo": "text",
                    "marker": {"size": 10, "color": tier_color, "opacity": 0.55, "symbol": "x"},
                }
            )

    data3d_json = json.dumps(traces3d, separators=(",", ":"), ensure_ascii=True)
    vram_tok_json = json.dumps(traces_vram_tok, separators=(",", ":"), ensure_ascii=True)
    heads_tok_json = json.dumps(traces_heads_tok, separators=(",", ":"), ensure_ascii=True)
    heads_acc_json = json.dumps(traces_heads_acc, separators=(",", ":"), ensure_ascii=True)
    title_esc = _html_escape(title)

    packets_total = len(packets)
    points_total = len(pts)
    pass_count = sum(1 for r in pts if r.get("pass") is True)
    fail_count = points_total - pass_count
    rankable_count = sum(1 for r in pts if r.get("rankable") is True)
    tier_list = ", ".join(tiers) if tiers else "-"
    gen_times = sorted([str(r.get("generated_utc") or "") for r in pts if r.get("generated_utc")])
    gen_span = f"{gen_times[0]} .. {gen_times[-1]}" if gen_times else "-"

    # Compact HTML table: keep it copy/paste friendly and readable offline.
    table_rows: List[str] = []
    pts_sorted = sorted(
        pts,
        key=lambda r: (0 if r.get("pass") is True else 1, -(float(r.get("desirability") or 0.0))),
    )
    for r in pts_sorted[:80]:
        table_rows.append(
            "<tr>"
            f"<td>{_html_escape(str(r.get('tier')))}</td>"
            f"<td>E{int(r.get('heads') or 0)}</td>"
            f"<td>B{int(r.get('batch') or 0)}</td>"
            f"<td>{'PASS' if r.get('pass') else 'FAIL'}</td>"
            f"<td>{float(r.get('x') or 0.0):.3f}</td>"
            f"<td>{float(r.get('y') or 0.0):.1f}</td>"
            f"<td>{float(r.get('z') or 0.0):.4f}</td>"
            f"<td>{float(r.get('desirability') or 0.0):.3f}</td>"
            f"<td><code>{_html_escape(str(r.get('probe') or ''))}</code></td>"
            f"<td><code>{_html_escape(str(r.get('assoc') or ''))}</code></td>"
            "</tr>"
        )
    table_html = "\n".join(table_rows)

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
      .wrap {{
        max-width: 1400px;
        margin: 0 auto;
        padding: 12px 12px 18px 12px;
      }}
      .top {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 10px;
        margin-bottom: 12px;
      }}
      .card {{
        background: rgba(22, 27, 34, 0.75);
        border: 1px solid rgba(240, 246, 252, 0.12);
        border-radius: 12px;
        padding: 10px 12px;
      }}
      h1 {{
        margin: 0 0 6px 0;
        font-size: 16px;
        font-weight: 700;
      }}
      .meta {{
        font-size: 12px;
        line-height: 1.45;
        color: rgba(230, 237, 243, 0.92);
      }}
      .meta b {{ color: #ffffff; }}
      .meta code {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 11px;
      }}
      .grid {{
        display: grid;
        grid-template-columns: 1.3fr 1fr;
        gap: 12px;
      }}
      @media (max-width: 1100px) {{
        .grid {{ grid-template-columns: 1fr; }}
      }}
      .plot {{
        width: 100%;
        height: 520px;
      }}
      .plot.small {{
        height: 320px;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
      }}
      th, td {{
        border-bottom: 1px solid rgba(240, 246, 252, 0.10);
        padding: 6px 6px;
        vertical-align: top;
      }}
      th {{
        text-align: left;
        font-weight: 700;
        color: rgba(230, 237, 243, 0.98);
        position: sticky;
        top: 0;
        background: rgba(22, 27, 34, 0.92);
      }}
      .scroll {{
        max-height: 360px;
        overflow: auto;
        border-radius: 10px;
        border: 1px solid rgba(240, 246, 252, 0.08);
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="top">
        <div class="card meta">
          <h1>{title_esc}</h1>
          <b>Source:</b> <code>ant_ratio_packets.jsonl</code><br />
          <b>Packets loaded:</b> {packets_total} (points plotted: {points_total}, pass: {pass_count}, fail: {fail_count}, rankable: {rankable_count})<br />
          <b>Rankable gate:</b> PASS and token_budget_steps &gt;= 30 and assoc_eval_n &gt;= 512<br />
          <b>Tiers:</b> {tier_list}<br />
          <b>Generated UTC span:</b> <code>{_html_escape(gen_span)}</code><br />
          <b>3D axes:</b> X=vram_ratio_reserved, Y=throughput_tokens_per_s, Z=assoc_byte_disjoint_accuracy<br />
          <b>Color:</b> desirability = log_norm(tok/s) * exp(-((vram-{target_vram_ratio:.2f})/{ratio_sigma:.2f})^2), zeroed on FAIL
        </div>
      </div>

      <div class="grid">
        <div class="card">
          <div id="plot3d" class="plot"></div>
        </div>
        <div class="card">
          <div id="plot_vram_tok" class="plot small"></div>
          <div id="plot_heads_tok" class="plot small"></div>
          <div id="plot_heads_acc" class="plot small"></div>
        </div>
      </div>

      <div style="height:12px;"></div>

      <div class="card">
        <div class="meta"><b>Top points</b> (sorted by PASS then desirability; first 80 shown)</div>
        <div class="scroll">
          <table>
            <thead>
              <tr>
                <th>tier</th>
                <th>E</th>
                <th>B</th>
                <th>status</th>
                <th>vram</th>
                <th>tok/s</th>
                <th>acc</th>
                <th>des</th>
                <th>probe</th>
                <th>assoc</th>
              </tr>
            </thead>
            <tbody>
{table_html}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <script>
      const data3d = {data3d_json};
      const dataVramTok = {vram_tok_json};
      const dataHeadsTok = {heads_tok_json};
      const dataHeadsAcc = {heads_acc_json};

      const layout3d = {{
        paper_bgcolor: "#0b0f14",
        plot_bgcolor: "#0b0f14",
        font: {{ color: "#e6edf3" }},
        margin: {{ l: 0, r: 0, t: 40, b: 0 }},
        title: {{ text: "3D frontier", x: 0.01, xanchor: "left", font: {{ size: 14 }} }},
        scene: {{
          xaxis: {{ title: "vram_ratio_reserved", gridcolor: "rgba(240,246,252,0.10)", zerolinecolor: "rgba(240,246,252,0.15)" }},
          yaxis: {{ title: "throughput_tokens_per_s", gridcolor: "rgba(240,246,252,0.10)", zerolinecolor: "rgba(240,246,252,0.15)" }},
          zaxis: {{ title: "assoc_byte_disjoint_accuracy", gridcolor: "rgba(240,246,252,0.10)", zerolinecolor: "rgba(240,246,252,0.15)" }},
          bgcolor: "#0b0f14"
        }},
        legend: {{ x: 0.01, y: 0.98, bgcolor: "rgba(0,0,0,0)" }}
      }};

      const layout2d = (title, xTitle, yTitle) => ({{
        paper_bgcolor: "#0b0f14",
        plot_bgcolor: "#0b0f14",
        font: {{ color: "#e6edf3" }},
        margin: {{ l: 48, r: 10, t: 36, b: 38 }},
        title: {{ text: title, x: 0.01, xanchor: "left", font: {{ size: 13 }} }},
        xaxis: {{ title: xTitle, gridcolor: "rgba(240,246,252,0.10)", zerolinecolor: "rgba(240,246,252,0.15)" }},
        yaxis: {{ title: yTitle, gridcolor: "rgba(240,246,252,0.10)", zerolinecolor: "rgba(240,246,252,0.15)" }},
        legend: {{ orientation: "h", x: 0.01, y: 1.15, bgcolor: "rgba(0,0,0,0)" }},
      }});

      Plotly.newPlot("plot3d", data3d, layout3d, {{ responsive: true }});
      Plotly.newPlot("plot_vram_tok", dataVramTok, layout2d("2D: tok/s vs VRAM ratio", "vram_ratio_reserved", "throughput_tokens_per_s"), {{ responsive: true }});
      Plotly.newPlot("plot_heads_tok", dataHeadsTok, layout2d("2D: tok/s vs expert_heads", "expert_heads", "throughput_tokens_per_s"), {{ responsive: true }});
      Plotly.newPlot("plot_heads_acc", dataHeadsAcc, layout2d("2D: accuracy vs expert_heads", "expert_heads", "assoc_byte_disjoint_accuracy"), {{ responsive: true }});
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
