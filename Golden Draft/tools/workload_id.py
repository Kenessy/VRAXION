"""Workload spec validation + canonical stable workload ID (VRA-31).

Stdlib-only by design.

This module defines:
- schema_version: workload_schema_v1
- strict validation (unknown keys are errors; no implicit coercion)
- deterministic ID: sha256(canonical_json) -> wl_v1_<12 hex>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "workload_schema_v1"

PTR_DTYPE_ALLOWED = ("fp64", "fp32", "fp16", "bf16")
PRECISION_ALLOWED = ("fp64", "fp32", "fp16", "bf16", "amp")

ANT_REQUIRED = ("ring_len", "slot_dim", "ptr_dtype", "precision")
COL_REQUIRED = ("seq_len", "synth_len", "batch_size", "ptr_update_every", "state_loop_samples")

_TOP_ALLOWED = frozenset({"schema_version", "ant_spec", "colony_spec", "name", "notes"})
_ANT_ALLOWED = frozenset(set(ANT_REQUIRED) | {"name", "notes"})
_COL_ALLOWED = frozenset(set(COL_REQUIRED) | {"name", "notes"})


def load_workload_spec(path: str) -> dict[str, Any]:
    """Load a workload spec JSON file."""

    pth = Path(path)
    with pth.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Workload spec must be a JSON object at top-level.")
    return obj


def _unknown_keys(obj: Mapping[str, Any], *, allowed: frozenset[str]) -> set[str]:
    return {k for k in obj.keys() if k not in allowed}


def _expect_dict(val: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(val, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return val


def _expect_int(val: Any, *, label: str, min_value: int) -> int:
    if not isinstance(val, int) or isinstance(val, bool):
        raise ValueError(f"{label} must be an integer.")
    if val < min_value:
        raise ValueError(f"{label} must be >= {min_value}.")
    return val


def _expect_enum(val: Any, *, label: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(val, str):
        raise ValueError(f"{label} must be a string.")
    if val not in allowed:
        raise ValueError(f"{label} must be one of: {', '.join(allowed)}.")
    return val


def canonicalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and return canonical spec (drops non-ID fields).

    Canonical output includes only:
      {schema_version, ant_spec(required keys), colony_spec(required keys)}
    """

    bad_top = _unknown_keys(spec, allowed=_TOP_ALLOWED)
    if bad_top:
        raise ValueError(f"Unknown top-level key(s): {sorted(bad_top)!r}")

    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}.")

    ant = _expect_dict(spec.get("ant_spec"), label="ant_spec")
    col = _expect_dict(spec.get("colony_spec"), label="colony_spec")

    bad_ant = _unknown_keys(ant, allowed=_ANT_ALLOWED)
    if bad_ant:
        raise ValueError(f"Unknown ant_spec key(s): {sorted(bad_ant)!r}")

    bad_col = _unknown_keys(col, allowed=_COL_ALLOWED)
    if bad_col:
        raise ValueError(f"Unknown colony_spec key(s): {sorted(bad_col)!r}")

    ant_canon = {
        "ring_len": _expect_int(ant.get("ring_len"), label="ant_spec.ring_len", min_value=1),
        "slot_dim": _expect_int(ant.get("slot_dim"), label="ant_spec.slot_dim", min_value=1),
        "ptr_dtype": _expect_enum(ant.get("ptr_dtype"), label="ant_spec.ptr_dtype", allowed=PTR_DTYPE_ALLOWED),
        "precision": _expect_enum(ant.get("precision"), label="ant_spec.precision", allowed=PRECISION_ALLOWED),
    }

    col_canon = {
        "seq_len": _expect_int(col.get("seq_len"), label="colony_spec.seq_len", min_value=1),
        "synth_len": _expect_int(col.get("synth_len"), label="colony_spec.synth_len", min_value=1),
        "batch_size": _expect_int(col.get("batch_size"), label="colony_spec.batch_size", min_value=1),
        "ptr_update_every": _expect_int(col.get("ptr_update_every"), label="colony_spec.ptr_update_every", min_value=1),
        "state_loop_samples": _expect_int(
            col.get("state_loop_samples"),
            label="colony_spec.state_loop_samples",
            min_value=0,
        ),
    }

    return {"schema_version": SCHEMA_VERSION, "ant_spec": ant_canon, "colony_spec": col_canon}


def _canonical_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_workload_id(canon: Mapping[str, Any]) -> str:
    """Compute stable workload ID from a canonical workload spec."""

    payload = _canonical_json(canon).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"wl_v1_{digest[:12]}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("spec", help="Path to workload spec JSON (workload_schema_v1)")
    p.add_argument("--json", action="store_true", help="Print canonicalized spec + workload_id as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        spec = load_workload_spec(args.spec)
        canon = canonicalize_spec(spec)
        wid = compute_workload_id(canon)
    except Exception as exc:
        print(f"[workload_id] error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        out = {
            "workload_id": wid,
            "schema_version": canon["schema_version"],
            "ant_spec": canon["ant_spec"],
            "colony_spec": canon["colony_spec"],
        }
        print(json.dumps(out, indent=2, ensure_ascii=True, sort_keys=True))
    else:
        print(wid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

