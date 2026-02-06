"""VRA-76: Join probe + capability artifacts into one ant-ratio packet (v0).

This tool is intentionally "artifact-truth":
- PASS/FAIL is read from probe metrics.json (never from process exit codes).
- Run roots are stored repo-relative when possible for portability.

Inputs
- probe run dir: metrics.json, env.json, run_cmd.txt (from gpu_capacity_probe.py)
- assoc run dir: report.json (capability eval; see eval_ckpt_assoc_byte.py)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


SCHEMA_VERSION = "ant_ratio_packet_v0"
SMALL_ANT_BODY_CELLS = 2048 * 256


KNOWN_ANT_TIERS: dict[tuple[int, int], str] = {
    (2048, 256): "small",
    (8192, 576): "real",
    (16384, 768): "stress",
}


class PacketError(RuntimeError):
    pass


def _repo_root() -> Path:
    # .../Golden Draft/tools/ant_ratio_packet_v0.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PacketError(f"missing required artifact: {path}")
    except Exception as exc:
        raise PacketError(f"failed to parse JSON: {path} ({exc})")


def _rel_repo_path(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(_repo_root().resolve())
        return rel.as_posix()
    except Exception:
        return path.resolve().as_posix()


def _try_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_repo_root()), text=True)
        return out.strip() or None
    except Exception:
        return None


def _is_probe_pass(metrics: dict[str, Any]) -> bool:
    # Never use process exit code as PASS/FAIL. Contract is metrics.json truth.
    return bool(
        metrics.get("stability_pass") is True
        and metrics.get("had_oom") is False
        and metrics.get("had_nan") is False
        and metrics.get("had_inf") is False
    )


@dataclass(frozen=True)
class TokenBudget:
    token_budget: int
    min_steps: int
    max_steps: int

    def derive_steps(self, *, batch: int, seq_len: int) -> Tuple[int, int]:
        tokens_per_step = int(batch) * int(seq_len)
        if tokens_per_step <= 0:
            raise PacketError(f"invalid tokens_per_step={tokens_per_step} (batch={batch}, seq_len={seq_len})")
        raw = int(self.token_budget) // int(tokens_per_step)
        steps = max(int(self.min_steps), min(int(self.max_steps), int(raw)))
        return steps, tokens_per_step


def build_packet(
    *,
    probe_run_root: Path,
    assoc_run_root: Path,
    ant_tier_override: Optional[str] = None,
    token_budget: Optional[TokenBudget] = None,
    capability_steps_override: Optional[int] = None,
) -> dict[str, Any]:
    probe_run_root = Path(probe_run_root)
    assoc_run_root = Path(assoc_run_root)

    metrics = _load_json(probe_run_root / "metrics.json")
    env = _load_json(probe_run_root / "env.json")
    cmd = _load_json(probe_run_root / "run_cmd.txt")
    report = _load_json(assoc_run_root / "report.json")

    canon = cmd.get("canonical_spec") or {}
    ant_spec = canon.get("ant_spec") or {}
    colony_spec = canon.get("colony_spec") or {}

    ring_len = ant_spec.get("ring_len")
    slot_dim = ant_spec.get("slot_dim")
    ant_body_cells: Optional[int] = None
    ant_body_scale_vs_small: Optional[float] = None
    if isinstance(ring_len, int) and isinstance(slot_dim, int):
        ant_body_cells = int(ring_len) * int(slot_dim)
        if SMALL_ANT_BODY_CELLS > 0:
            ant_body_scale_vs_small = float(ant_body_cells) / float(SMALL_ANT_BODY_CELLS)
    ant_tier = ant_tier_override
    if ant_tier is None and isinstance(ring_len, int) and isinstance(slot_dim, int):
        ant_tier = KNOWN_ANT_TIERS.get((int(ring_len), int(slot_dim)))
        if ant_tier is None:
            ant_tier = "custom"

    reserved = metrics.get("peak_vram_reserved_bytes")
    total_vram = env.get("total_vram_bytes")
    vram_ratio_reserved: Optional[float] = None
    if isinstance(reserved, int) and isinstance(total_vram, int) and total_vram > 0:
        vram_ratio_reserved = float(reserved) / float(total_vram)

    eval_obj = report.get("eval") or {}
    settings = report.get("settings") or {}
    assoc_acc = eval_obj.get("eval_acc")
    assoc_eval_disjoint = settings.get("eval_disjoint")

    seq_len = metrics.get("seq_len")
    batch = metrics.get("batch_size")
    derived_steps: Optional[int] = None
    tokens_per_step: Optional[int] = None
    if token_budget is not None and isinstance(batch, int) and isinstance(seq_len, int):
        # Record the fairness-normalization basis even when steps are supplied
        # explicitly by the sweep runner.
        tokens_per_step = int(batch) * int(seq_len)
    if capability_steps_override is not None:
        derived_steps = int(capability_steps_override)
    elif token_budget is not None and isinstance(batch, int) and isinstance(seq_len, int):
        derived_steps, _ = token_budget.derive_steps(batch=int(batch), seq_len=int(seq_len))

    pkt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _try_git_commit(),
        # Config identity (from probe canonical_spec when available)
        "ant_tier": ant_tier,
        "ant_ring_len": int(ring_len) if isinstance(ring_len, int) else None,
        "ant_slot_dim": int(slot_dim) if isinstance(slot_dim, int) else None,
        "ant_body_cells": ant_body_cells,
        "ant_body_scale_vs_small": ant_body_scale_vs_small,
        "expert_heads": int(metrics.get("out_dim")) if isinstance(metrics.get("out_dim"), int) else None,
        "batch_size": int(batch) if isinstance(batch, int) else None,
        "precision": metrics.get("precision"),
        "amp": int(metrics.get("amp")) if isinstance(metrics.get("amp"), int) else None,
        "seq_len": int(seq_len) if isinstance(seq_len, int) else None,
        # Cost metrics (probe)
        "vram_ratio_reserved": vram_ratio_reserved,
        "peak_vram_reserved_bytes": int(reserved) if isinstance(reserved, int) else None,
        "total_vram_bytes": int(total_vram) if isinstance(total_vram, int) else None,
        "throughput_tokens_per_s": metrics.get("throughput_tokens_per_s"),
        "throughput_samples_per_s": metrics.get("throughput_samples_per_s"),
        # Capability metric (assoc eval)
        "assoc_byte_disjoint_accuracy": float(assoc_acc) if isinstance(assoc_acc, (int, float)) else None,
        "assoc_eval_disjoint": bool(assoc_eval_disjoint) if isinstance(assoc_eval_disjoint, bool) else None,
        "assoc_eval_n": int(eval_obj.get("eval_n")) if isinstance(eval_obj.get("eval_n"), int) else None,
        # Fairness metadata (optional)
        "token_budget": int(token_budget.token_budget) if token_budget is not None else None,
        "token_budget_tokens_per_step": int(tokens_per_step) if tokens_per_step is not None else None,
        "token_budget_steps": int(derived_steps) if derived_steps is not None else None,
        # Provenance
        "probe_run_root": _rel_repo_path(probe_run_root),
        "assoc_run_root": _rel_repo_path(assoc_run_root),
        # Stability truth (probe)
        "stability_pass": bool(_is_probe_pass(metrics)),
        "fail_reasons": list(metrics.get("fail_reasons") or []),
        "had_oom": bool(metrics.get("had_oom") or False),
        "had_nan": bool(metrics.get("had_nan") or False),
        "had_inf": bool(metrics.get("had_inf") or False),
        # Optional: include workload ids for join/debug
        "workload_id": metrics.get("workload_id"),
        "probe_id": metrics.get("probe_id"),
    }

    # Keep the output JSON stable and ASCII-only for diffs.
    json.dumps(pkt, sort_keys=True, ensure_ascii=True)
    return pkt


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join probe+assoc artifacts into ant_ratio_packet_v0 JSON.")
    p.add_argument("--probe-run-root", required=True, help="Probe run directory (contains metrics.json/env.json/run_cmd.txt).")
    p.add_argument("--assoc-run-root", required=True, help="Assoc run directory (contains report.json).")
    p.add_argument("--out", default="", help="Output JSON path. If omitted, prints to stdout.")
    p.add_argument("--ant-tier", default="", help="Override ant_tier (small/real/stress/custom).")
    p.add_argument("--token-budget", type=int, default=0, help="Optional fixed token budget used for fairness metadata.")
    p.add_argument("--min-steps", type=int, default=50, help="Clamp for derived steps when using --token-budget.")
    p.add_argument("--max-steps", type=int, default=5000, help="Clamp for derived steps when using --token-budget.")
    p.add_argument("--capability-steps", type=int, default=0, help="Optional explicit steps used for capability run.")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    ant_tier = args.ant_tier.strip() or None
    tb: Optional[TokenBudget] = None
    if int(args.token_budget) > 0:
        tb = TokenBudget(token_budget=int(args.token_budget), min_steps=int(args.min_steps), max_steps=int(args.max_steps))
    cap_steps = int(args.capability_steps) if int(args.capability_steps) > 0 else None

    try:
        pkt = build_packet(
            probe_run_root=Path(args.probe_run_root),
            assoc_run_root=Path(args.assoc_run_root),
            ant_tier_override=ant_tier,
            token_budget=tb,
            capability_steps_override=cap_steps,
        )
    except PacketError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_txt = json.dumps(pkt, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    if args.out.strip():
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_txt, encoding="utf-8")
    else:
        sys.stdout.write(out_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
