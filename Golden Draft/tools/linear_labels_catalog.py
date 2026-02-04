"""Normalize a Linear label dump into a searchable catalog (CSV + JSON).

This tool intentionally does NOT call Linear APIs. Instead, it consumes a raw
JSON dump produced by the Linear MCP tool:

  mcp__linear__list_issue_labels(team="VRAXION", limit=250)

Why:
  - OAuth/session handling is owned by the MCP server.
  - We want a deterministic, repo-local catalog and diffable artifacts.

Input shape (minimum):
  {"labels":[{"id","parentId","name","color","description"}, ...]}

Outputs:
  - labels_catalog_v1.csv (grep-friendly + ASCII-normalized columns)
  - labels_catalog_v1.json (machine-friendly, includes IDs + warnings)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ARROW = "\u2192"  # "→"


def _to_ascii(s: str) -> str:
    # Keep output predictable on Windows consoles while preserving raw columns.
    return s.replace(ARROW, "->")


def _split_group_child(name: str) -> Tuple[str, str, str]:
    """Return (group_raw, child_raw, sep_used). sep_used is "", "->", or "→"."""
    if ARROW in name:
        left, right = name.split(ARROW, 1)
        return left.strip(), right.strip(), ARROW
    if "->" in name:
        left, right = name.split("->", 1)
        return left.strip(), right.strip(), "->"
    return "", name.strip(), ""


def labels_to_rows(labels: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Convert raw Linear labels into catalog rows + warnings.

    Row schema matches docs/linear/labels_catalog_v1.csv columns.
    """
    warnings: List[str] = []

    # Dedupe by label_id in case callers concatenate multiple pages.
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for lab in labels:
        lid = str(lab.get("id") or "")
        if not lid:
            warnings.append("Label missing id; skipping one record.")
            continue
        if lid in seen:
            continue
        seen.add(lid)
        deduped.append(lab)

    rows: List[Dict[str, Any]] = []
    for lab in deduped:
        label_id = str(lab.get("id") or "")
        parent_id = lab.get("parentId")
        parent_id_str = str(parent_id) if parent_id is not None else ""
        name = str(lab.get("name") or "").strip()
        color = str(lab.get("color") or "").strip()
        desc = lab.get("description")
        description = "" if desc is None else str(desc)

        group_raw = ""
        child_raw = ""
        group_id = ""

        if parent_id is None:
            # Group headers are parentId=null in Linear. Keep them in the catalog
            # so we can reconstruct tree shape even if children are missing arrows.
            group_raw = name
            child_raw = ""
            group_id = label_id
            # If a group label contains an arrow, it's unusual; warn but keep.
            if ARROW in name or "->" in name:
                warnings.append(f"Group label contains arrow separator: {name!r}")
        else:
            group_raw, child_raw, sep = _split_group_child(name)
            group_id = parent_id_str
            if not sep:
                warnings.append(f"Child label missing '{ARROW}' separator: {name!r}")

        if not description:
            warnings.append(f"Label missing description: {name!r}")

        row = {
            "group_name_raw": group_raw,
            "group_name_ascii": _to_ascii(group_raw),
            "group_id": group_id,
            "label_name_raw": name,
            "label_name_ascii": _to_ascii(name),
            "label_child_raw": child_raw,
            "label_child_ascii": _to_ascii(child_raw),
            "label_id": label_id,
            "color": color,
            "description": description,
        }
        rows.append(row)

    # Duplicate child detection within a group_id (excluding group header rows).
    child_counts: Dict[Tuple[str, str], int] = {}
    for r in rows:
        gid = str(r["group_id"])
        child = str(r["label_child_raw"])
        if not child:
            continue
        key = (gid, child)
        child_counts[key] = child_counts.get(key, 0) + 1
    for (gid, child), n in sorted(child_counts.items()):
        if n > 1:
            warnings.append(f"Duplicate child label within group_id={gid}: {child!r} ({n}x)")

    # Deterministic order.
    rows.sort(key=lambda r: (r["group_name_raw"], r["label_child_raw"], r["label_name_raw"]))
    return rows, warnings


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "group_name_raw",
        "group_name_ascii",
        "group_id",
        "label_name_raw",
        "label_name_ascii",
        "label_child_raw",
        "label_child_ascii",
        "label_id",
        "color",
        "description",
    ]
    with path.open("w", encoding="utf-8", errors="replace", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_json(path: Path, team: str, rows: List[Dict[str, Any]], warnings: List[str]) -> None:
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "team": team,
        "labels": [
            {
                "group_name_raw": r["group_name_raw"],
                "group_id": r["group_id"],
                "label_name_raw": r["label_name_raw"],
                "label_child_raw": r["label_child_raw"],
                "label_id": r["label_id"],
                "color": r["color"],
                "description": r["description"],
            }
            for r in rows
        ],
        "warnings": warnings,
    }
    with path.open("w", encoding="utf-8", errors="replace") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to raw MCP JSON (labels list).")
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for labels_catalog_v1.{csv,json}",
    )
    p.add_argument("--team", default="VRAXION", help="Team name to stamp into JSON output.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    in_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(in_path.read_text(encoding="utf-8", errors="replace"))
    labels = raw.get("labels")
    if not isinstance(labels, list):
        raise SystemExit("Input JSON missing 'labels' list.")

    rows, warns = labels_to_rows(labels)

    csv_path = out_dir / "labels_catalog_v1.csv"
    json_path = out_dir / "labels_catalog_v1.json"
    _write_csv(csv_path, rows)
    _write_json(json_path, str(args.team), rows, warns)

    groups = {r["group_id"] for r in rows if r["group_id"]}
    print(f"labels={len(rows)} groups={len(groups)} warnings={len(warns)}")
    for w in warns[:5]:
        print(f"WARNING: {w}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

