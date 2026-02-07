"""VRA-79: publish sanitized ant-ratio datapoints + public frontiers.

Input:
- One or more sweep roots from bench_vault/_tmp/vra78_ant_ratio_sweep_v0/<ts>/

Outputs (repo-tracked):
- docs/results/ant_ratio/db_v0.csv
- docs/results/ant_ratio/db_v0.jsonl
- docs/results/ant_ratio/frontier_latest.html
- docs/results/ant_ratio/frontier_latest.svg
- docs/results/ant_ratio/index.html
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROW_SCHEMA = "ant_ratio_db_row_v0"
DB_JSONL = "db_v0.jsonl"
DB_CSV = "db_v0.csv"
FRONTIER_HTML = "frontier_latest.html"
FRONTIER_SVG = "frontier_latest.svg"
INDEX_HTML = "index.html"

TIER_ORDER = {"small": 0, "real": 1, "stress": 2}
SEED_RE = re.compile(r"seed(\d+)", re.IGNORECASE)

DB_FIELDS: Sequence[str] = (
    "schema_version",
    "run_tag",
    "generated_utc",
    "ant_tier",
    "expert_heads",
    "batch",
    "steps",
    "cap_seed",
    "status",
    "cap_train_rc",
    "cap_train_nonzero_rc",
    "cap_eval_device",
    "eval_heartbeat_seen",
    "attempt_eval_count",
    "probe_pass",
    "vram_ratio_reserved",
    "throughput_tokens_per_s",
    "tokens_per_vram_ratio",
    "assoc_byte_disjoint_accuracy",
    "assoc_eval_n",
    "token_budget_steps",
    "vram_target_abs_error",
    "assoc_acc_per_token_per_s",
    "assoc_acc_per_vram_ratio",
    "rankable",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(float(stripped))
        except Exception:
            return default
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return float(stripped)
        except Exception:
            return default
    return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    chunks = [json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rows]
    path.write_text("\n".join(chunks) + ("\n" if chunks else ""), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_seed(value: Any) -> int:
    match = SEED_RE.search(str(value or ""))
    if not match:
        return 0
    try:
        return int(match.group(1))
    except Exception:
        return 0


def _stable_sort(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(row: Dict[str, Any]) -> Tuple[Any, ...]:
        tier = str(row.get("ant_tier") or "").strip().lower()
        return (
            str(row.get("run_tag") or ""),
            TIER_ORDER.get(tier, 99),
            _safe_int(row.get("expert_heads"), 0),
            _safe_int(row.get("cap_seed"), 0),
            _safe_int(row.get("batch"), 0),
            _safe_int(row.get("steps"), 0),
        )

    return sorted(rows, key=key_fn)


def _row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(row.get("run_tag") or ""),
        str(row.get("ant_tier") or "").strip().lower(),
        _safe_int(row.get("expert_heads"), 0),
        _safe_int(row.get("cap_seed"), 0),
    )


def _load_existing_db(out_dir: Path) -> List[Dict[str, Any]]:
    path = out_dir / DB_JSONL
    if not path.exists():
        return []
    loaded = _jsonl_rows(path)
    rows: List[Dict[str, Any]] = []
    for row in loaded:
        if not isinstance(row, dict):
            continue
        sanitized = {field: row.get(field) for field in DB_FIELDS}
        rows.append(sanitized)
    return rows


def _packet_key(packet: Dict[str, Any]) -> Tuple[str, int, int]:
    return (
        str(packet.get("ant_tier") or "").strip().lower(),
        _safe_int(packet.get("expert_heads"), 0),
        _safe_int(packet.get("batch_size"), 0),
    )


def _packet_map(sweep_root: Path) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    packets_path = sweep_root / "ant_ratio_packets.jsonl"
    if not packets_path.exists():
        return {}
    out: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for packet in _jsonl_rows(packets_path):
        if isinstance(packet, dict):
            out[_packet_key(packet)] = packet
    return out


def _sanitize_row(
    *,
    source: Dict[str, Any],
    packet: Optional[Dict[str, Any]],
    run_tag: str,
    generated_utc: str,
    vram_target: float,
) -> Dict[str, Any]:
    tier = str(source.get("ant_tier") or "").strip().lower()
    heads = _safe_int(source.get("expert_heads"), _safe_int((packet or {}).get("expert_heads"), 0))
    batch = _safe_int(source.get("batch"), _safe_int(source.get("batch_size"), _safe_int((packet or {}).get("batch_size"), 0)))
    steps = _safe_int(source.get("steps"), _safe_int((packet or {}).get("token_budget_steps"), 0))
    cap_seed = _extract_seed(source.get("assoc_run_root")) or _extract_seed((packet or {}).get("assoc_run_root"))

    status = str(source.get("status") or ("ok" if _safe_bool((packet or {}).get("stability_pass"), False) else "error")).strip().lower()
    probe_pass = _safe_bool(source.get("probe_pass"), _safe_bool((packet or {}).get("stability_pass"), False))

    vram_ratio = _safe_float(source.get("vram_ratio_reserved"), _safe_float((packet or {}).get("vram_ratio_reserved"), 0.0))
    throughput = _safe_float(source.get("throughput_tokens_per_s"), _safe_float((packet or {}).get("throughput_tokens_per_s"), 0.0))
    assoc_acc = _safe_float(source.get("assoc_byte_disjoint_accuracy"), _safe_float((packet or {}).get("assoc_byte_disjoint_accuracy"), 0.0))
    assoc_eval_n = _safe_int(source.get("assoc_eval_n"), _safe_int((packet or {}).get("assoc_eval_n"), 0))
    token_budget_steps = _safe_int(source.get("token_budget_steps"), _safe_int((packet or {}).get("token_budget_steps"), steps))
    cap_train_rc = _safe_int(source.get("cap_train_rc"), _safe_int((packet or {}).get("cap_train_rc"), 0))
    cap_train_nonzero_rc = _safe_bool(source.get("cap_train_nonzero_rc"), _safe_bool((packet or {}).get("cap_train_nonzero_rc"), cap_train_rc != 0))
    cap_eval_device = str(source.get("cap_eval_device") or (packet or {}).get("cap_eval_device") or "").strip().lower()
    eval_heartbeat_seen = _safe_bool(source.get("eval_heartbeat_seen"), _safe_bool((packet or {}).get("eval_heartbeat_seen"), False))
    attempt_eval_count = _safe_int(source.get("attempt_eval_count"), _safe_int((packet or {}).get("attempt_eval_count"), 0))

    tokens_per_vram = throughput / vram_ratio if vram_ratio > 0.0 else 0.0
    acc_per_tok = assoc_acc / throughput if throughput > 0.0 else 0.0
    acc_per_vram = assoc_acc / vram_ratio if vram_ratio > 0.0 else 0.0
    rankable = bool(status == "ok" and probe_pass and token_budget_steps >= 30 and assoc_eval_n >= 512)

    return {
        "schema_version": ROW_SCHEMA,
        "run_tag": run_tag,
        "generated_utc": generated_utc,
        "ant_tier": tier,
        "expert_heads": heads,
        "batch": batch,
        "steps": steps,
        "cap_seed": cap_seed,
        "status": status,
        "cap_train_rc": cap_train_rc,
        "cap_train_nonzero_rc": cap_train_nonzero_rc,
        "cap_eval_device": cap_eval_device,
        "eval_heartbeat_seen": eval_heartbeat_seen,
        "attempt_eval_count": attempt_eval_count,
        "probe_pass": probe_pass,
        "vram_ratio_reserved": vram_ratio,
        "throughput_tokens_per_s": throughput,
        "tokens_per_vram_ratio": tokens_per_vram,
        "assoc_byte_disjoint_accuracy": assoc_acc,
        "assoc_eval_n": assoc_eval_n,
        "token_budget_steps": token_budget_steps,
        "vram_target_abs_error": abs(vram_ratio - float(vram_target)),
        "assoc_acc_per_token_per_s": acc_per_tok,
        "assoc_acc_per_vram_ratio": acc_per_vram,
        "rankable": rankable,
    }


def _load_rows_from_sweep(sweep_root: Path, vram_target: float, rankable_only: bool) -> List[Dict[str, Any]]:
    summary_csv = sweep_root / "ant_ratio_summary.csv"
    if not summary_csv.exists():
        return []

    run_tag = sweep_root.name
    meta_path = sweep_root / "sweep_meta.json"
    generated_utc = _now_utc()
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            generated_utc = str(meta.get("generated_utc") or generated_utc)
        except Exception:
            pass

    packets = _packet_map(sweep_root)
    rows: List[Dict[str, Any]] = []
    for raw in _csv_rows(summary_csv):
        tier = str(raw.get("ant_tier") or "").strip().lower()
        heads = _safe_int(raw.get("expert_heads"), 0)
        batch = _safe_int(raw.get("batch"), 0)
        packet = packets.get((tier, heads, batch))
        row = _sanitize_row(
            source=raw,
            packet=packet,
            run_tag=run_tag,
            generated_utc=generated_utc,
            vram_target=vram_target,
        )
        if rankable_only and not bool(row.get("rankable")):
            continue
        rows.append(row)
    return rows


def _rows_to_packets(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    packets: List[Dict[str, Any]] = []
    for row in rows:
        packets.append(
            {
                "ant_tier": row.get("ant_tier"),
                "expert_heads": _safe_int(row.get("expert_heads"), 0),
                "batch_size": _safe_int(row.get("batch"), 0),
                "stability_pass": _safe_bool(row.get("probe_pass"), False),
                "fail_reasons": [] if _safe_bool(row.get("probe_pass"), False) else ["not_pass"],
                "vram_ratio_reserved": _safe_float(row.get("vram_ratio_reserved"), 0.0),
                "throughput_tokens_per_s": _safe_float(row.get("throughput_tokens_per_s"), 0.0),
                "assoc_byte_disjoint_accuracy": _safe_float(row.get("assoc_byte_disjoint_accuracy"), 0.0),
                "generated_utc": row.get("generated_utc"),
                "token_budget_steps": _safe_int(row.get("token_budget_steps"), 0),
                "assoc_eval_n": _safe_int(row.get("assoc_eval_n"), 0),
                "probe_run_root": "",
                "assoc_run_root": "",
            }
        )
    return packets


def _build_frontier_html(rows: Sequence[Dict[str, Any]]) -> str:
    try:
        from ant_ratio_plot_v0 import build_html  # type: ignore
    except Exception:
        from tools.ant_ratio_plot_v0 import build_html  # type: ignore
    packets = _rows_to_packets(rows)
    return build_html(packets=packets, title="VRAXION Ant Ratio Frontier v0 (Public)")


def _svg_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _tier_color(tier: str) -> str:
    if tier == "small":
        return "#58a6ff"
    if tier == "real":
        return "#2ea043"
    if tier == "stress":
        return "#d29922"
    return "#8b949e"


def _write_svg(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    width = 1200
    height = 720
    left = 90
    right = 40
    top = 40
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    points = [
        row for row in rows
        if _safe_float(row.get("throughput_tokens_per_s"), -1.0) >= 0.0
    ]
    if not points:
        path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='720'><text x='20' y='40'>No points</text></svg>",
            encoding="utf-8",
        )
        return

    x_min = min(_safe_float(r.get("throughput_tokens_per_s"), 0.0) for r in points)
    x_max = max(_safe_float(r.get("throughput_tokens_per_s"), 0.0) for r in points)
    y_min = min(_safe_float(r.get("assoc_byte_disjoint_accuracy"), 0.0) for r in points)
    y_max = max(_safe_float(r.get("assoc_byte_disjoint_accuracy"), 0.0) for r in points)
    if math.isclose(x_min, x_max):
        x_min, x_max = (x_min - 1.0, x_max + 1.0)
    if math.isclose(y_min, y_max):
        y_min, y_max = (y_min - 0.01, y_max + 0.01)

    def sx(value: float) -> float:
        frac = (value - x_min) / max(x_max - x_min, 1e-9)
        return left + frac * plot_w

    def sy(value: float) -> float:
        frac = (value - y_min) / max(y_max - y_min, 1e-9)
        return top + (1.0 - frac) * plot_h

    lines: List[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>",
        "  .bg { fill: #0d1117; }",
        "  .axis { stroke: #30363d; stroke-width: 1; }",
        "  .label { fill: #c9d1d9; font-size: 13px; font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; }",
        "  .title { fill: #f0f6fc; font-size: 18px; font-weight: 700; font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; }",
        "</style>",
        f"<rect class='bg' x='0' y='0' width='{width}' height='{height}' />",
        f"<text class='title' x='{left}' y='28'>VRA-78 Public Frontier (2D)</text>",
        f"<line class='axis' x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' />",
        f"<line class='axis' x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' />",
        f"<text class='label' x='{left + plot_w - 220}' y='{height - 24}'>Throughput (tokens/s)</text>",
        f"<text class='label' x='14' y='{top + 16}'>Assoc disjoint accuracy</text>",
    ]

    for row in points:
        tier = str(row.get("ant_tier") or "")
        x = sx(_safe_float(row.get("throughput_tokens_per_s"), 0.0))
        y = sy(_safe_float(row.get("assoc_byte_disjoint_accuracy"), 0.0))
        r = max(4.0, min(14.0, 4.0 + 0.5 * _safe_int(row.get("expert_heads"), 0)))
        fill = _tier_color(tier)
        status = "PASS" if _safe_bool(row.get("probe_pass"), False) else "FAIL"
        title = (
            f"{tier} E={_safe_int(row.get('expert_heads'), 0)} B={_safe_int(row.get('batch'), 0)} {status} "
            f"tok/s={_safe_float(row.get('throughput_tokens_per_s'), 0.0):.2f} "
            f"acc={_safe_float(row.get('assoc_byte_disjoint_accuracy'), 0.0):.6f}"
        )
        lines.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{r:.2f}' fill='{fill}' fill-opacity='0.86' stroke='#f0f6fc' stroke-opacity='0.25' stroke-width='0.8'>"
            f"<title>{_svg_escape(title)}</title></circle>"
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_table(rows: Sequence[Dict[str, Any]]) -> str:
    def rank_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
        return (
            _safe_float(row.get("vram_target_abs_error"), 9e9),
            -_safe_float(row.get("tokens_per_vram_ratio"), -9e9),
            -_safe_float(row.get("assoc_acc_per_vram_ratio"), -9e9),
        )

    rankable = [r for r in rows if _safe_bool(r.get("rankable"), False)]
    rankable_sorted = sorted(rankable, key=rank_key)
    top = rankable_sorted[:5]
    bottom = list(reversed(rankable_sorted[-5:])) if rankable_sorted else []

    def tr(row: Dict[str, Any], label: str) -> str:
        return (
            "<tr>"
            f"<td>{label}</td>"
            f"<td>{_svg_escape(str(row.get('run_tag') or ''))}</td>"
            f"<td>{_svg_escape(str(row.get('ant_tier') or ''))}</td>"
            f"<td>{_safe_int(row.get('expert_heads'), 0)}</td>"
            f"<td>{_safe_int(row.get('batch'), 0)}</td>"
            f"<td>{_safe_float(row.get('vram_ratio_reserved'), 0.0):.3f}</td>"
            f"<td>{_safe_float(row.get('throughput_tokens_per_s'), 0.0):.1f}</td>"
            f"<td>{_safe_float(row.get('assoc_byte_disjoint_accuracy'), 0.0):.6f}</td>"
            f"<td>{_safe_float(row.get('vram_target_abs_error'), 0.0):.3f}</td>"
            "</tr>"
        )

    rows_html: List[str] = []
    for row in top:
        rows_html.append(tr(row, "TOP"))
    for row in bottom:
        rows_html.append(tr(row, "BOTTOM"))
    return "\n".join(rows_html)


def _write_index(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    generated = _now_utc()
    total = len(rows)
    rankable = sum(1 for r in rows if _safe_bool(r.get("rankable"), False))
    pass_count = sum(1 for r in rows if _safe_bool(r.get("probe_pass"), False))
    fail_count = total - pass_count
    table_html = _render_table(rows)
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>VRAXION Ant Ratio Public Frontier</title>
    <style>
      body {{ font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial; margin: 24px; color: #0d1117; }}
      h1 {{ margin: 0 0 8px 0; }}
      .meta {{ margin-bottom: 16px; color: #57606a; }}
      a {{ color: #0969da; }}
      table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
      th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; font-size: 13px; text-align: left; }}
      th {{ background: #f6f8fa; }}
    </style>
  </head>
  <body>
    <h1>VRAXION Ant Ratio Frontier (Public)</h1>
    <div class="meta">
      Last updated: <code>{generated}</code><br />
      Rows: <code>{total}</code> | PASS: <code>{pass_count}</code> | FAIL: <code>{fail_count}</code> | Rankable: <code>{rankable}</code>
    </div>
    <div>
      <a href="{DB_CSV}">db_v0.csv</a> |
      <a href="{DB_JSONL}">db_v0.jsonl</a> |
      <a href="{FRONTIER_HTML}">interactive 3D frontier</a> |
      <a href="{FRONTIER_SVG}">static 2D frontier (SVG)</a>
    </div>
    <h2>Rankable Top/Bottom</h2>
    <table>
      <thead>
        <tr>
          <th>Band</th><th>Run</th><th>Tier</th><th>Heads</th><th>Batch</th>
          <th>VRAM</th><th>Tok/s</th><th>Acc</th><th>|VRAM-0.85|</th>
        </tr>
      </thead>
      <tbody>
{table_html}
      </tbody>
    </table>
  </body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _merge_rows(existing: Sequence[Dict[str, Any]], new_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in existing:
        merged[_row_key(row)] = {field: row.get(field) for field in DB_FIELDS}
    for row in new_rows:
        merged[_row_key(row)] = {field: row.get(field) for field in DB_FIELDS}
    return _stable_sort(list(merged.values()))


def _signature(root: Path) -> str:
    parts = [str(root.resolve())]
    for name in ("ant_ratio_summary.csv", "ant_ratio_packets.jsonl", "sweep_meta.json"):
        path = root / name
        if path.exists():
            stat = path.stat()
            parts.append(f"{name}:{int(stat.st_mtime)}:{int(stat.st_size)}")
        else:
            parts.append(f"{name}:missing")
    return "|".join(parts)


def _publish_once(*, sweep_roots: Sequence[Path], out_dir: Path, vram_target: float, rankable_only: bool) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_db(out_dir)
    incoming: List[Dict[str, Any]] = []
    for root in sweep_roots:
        incoming.extend(_load_rows_from_sweep(root, vram_target=vram_target, rankable_only=rankable_only))
    rows = _merge_rows(existing, incoming)
    _write_jsonl(out_dir / DB_JSONL, rows)
    _write_csv(out_dir / DB_CSV, rows, DB_FIELDS)
    (out_dir / FRONTIER_HTML).write_text(_build_frontier_html(rows), encoding="utf-8")
    _write_svg(out_dir / FRONTIER_SVG, rows)
    _write_index(out_dir / INDEX_HTML, rows)
    return rows


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Publish VRA-78 sweeps into a public ant-ratio DB + plots.")
    ap.add_argument("--sweep-root", action="append", default=[], help="Path to a sweep root; can be repeated.")
    ap.add_argument("--out-dir", default="docs/results/ant_ratio", help="Output directory relative to repo root by default.")
    ap.add_argument("--vram-target", type=float, default=0.85)
    ap.add_argument("--rankable-only", type=int, choices=[0, 1], default=0)
    ap.add_argument("--watch", type=int, choices=[0, 1], default=0)
    ap.add_argument("--poll-s", type=int, default=10)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    repo_root = _repo_root()
    sweep_roots = [Path(p).resolve() for p in args.sweep_root if str(p).strip()]
    if not sweep_roots:
        raise SystemExit("at least one --sweep-root is required")
    out_dir_raw = Path(args.out_dir)
    out_dir = out_dir_raw if out_dir_raw.is_absolute() else (repo_root / out_dir_raw)
    rankable_only = bool(int(args.rankable_only))
    vram_target = float(args.vram_target)

    rows = _publish_once(
        sweep_roots=sweep_roots,
        out_dir=out_dir,
        vram_target=vram_target,
        rankable_only=rankable_only,
    )
    print(f"[publish] rows={len(rows)} out={out_dir}")

    if not bool(int(args.watch)):
        return 0

    signatures = {str(root): _signature(root) for root in sweep_roots}
    last_heartbeat = time.monotonic()
    poll_s = max(1, int(args.poll_s))
    while True:
        changed = False
        for root in sweep_roots:
            key = str(root)
            sig = _signature(root)
            if signatures.get(key) != sig:
                signatures[key] = sig
                changed = True
        if changed:
            rows = _publish_once(
                sweep_roots=sweep_roots,
                out_dir=out_dir,
                vram_target=vram_target,
                rankable_only=rankable_only,
            )
            print(f"[publish] refresh rows={len(rows)} utc={_now_utc()}")
        now = time.monotonic()
        if now - last_heartbeat >= 60.0:
            print(f"[publish] heartbeat watch=1 utc={_now_utc()} rows={len(rows)}")
            last_heartbeat = now
        time.sleep(poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
