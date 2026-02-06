"""VRA-78: Ant-ratio sweep v0 (probe cost + assoc capability + frontier plot).

This tool is an orchestrator. It produces runtime artifacts under repo-root:
  bench_vault/_tmp/vra78_ant_ratio_sweep_v0/<ts>/

Committed outputs are only code/tests/docs. The sweep artifacts are gitignored.

Important invariants:
- PASS/FAIL is always read from metrics.json/report.json, not exit codes.
- All run roots are created under repo-root bench_vault/_tmp/... by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class SweepError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_probe_pass(metrics: Dict[str, Any]) -> bool:
    return bool(
        metrics.get("stability_pass") is True
        and metrics.get("had_oom") is False
        and metrics.get("had_nan") is False
        and metrics.get("had_inf") is False
    )


def _safe_tag(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _ant_preset_for_tier(tier: str) -> str:
    t = str(tier).strip().lower()
    if t == "small":
        return "OD1_CANON_SMALL"
    if t == "real":
        return "OD1_CANON_REAL"
    if t == "stress":
        return "OD1_CANON_STRESS"
    raise SweepError(f"unknown ant_tier: {tier!r} (expected small,real,stress)")


def _ant_shape_for_tier(tier: str) -> Tuple[int, int]:
    t = str(tier).strip().lower()
    if t == "small":
        return 2048, 256
    if t == "real":
        return 8192, 576
    if t == "stress":
        return 16384, 768
    raise SweepError(f"unknown ant_tier: {tier!r} (expected small,real,stress)")


@dataclass(frozen=True)
class TokenBudget:
    token_budget: int
    min_steps: int
    max_steps: int
    seq_len: int

    def steps_for_batch(self, batch: int) -> int:
        tokens_per_step = int(batch) * int(self.seq_len)
        if tokens_per_step <= 0:
            return int(self.min_steps)
        raw = int(self.token_budget) // int(tokens_per_step)
        return max(int(self.min_steps), min(int(self.max_steps), int(raw)))


def _run_probe(
    *,
    repo_root: Path,
    out_dir: Path,
    ant: str,
    colony: str,
    out_dim: int,
    batch: int,
    warmup_steps: int,
    measure_steps: int,
    precision: str,
    amp: int,
    force_device: str,
) -> Tuple[Path, Dict[str, Any]]:
    tool = repo_root / "Golden Draft" / "tools" / "gpu_capacity_probe.py"
    if not tool.exists():
        raise SweepError(f"probe harness missing: {tool}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(tool),
        "--ant",
        str(ant),
        "--colony",
        str(colony),
        "--out-dim",
        str(int(out_dim)),
        "--batch",
        str(int(batch)),
        "--warmup-steps",
        str(int(warmup_steps)),
        "--measure-steps",
        str(int(measure_steps)),
        "--precision",
        str(precision),
        "--amp",
        str(int(amp)),
        "--output-dir",
        str(out_dir),
    ]
    env = dict(os.environ)
    if force_device:
        env["VRX_FORCE_DEVICE"] = str(force_device)

    rc = subprocess.call(cmd, cwd=str(repo_root), env=env)
    if int(rc) == 2:
        raise SweepError(f"probe harness rc=2 (invalid args or cannot write): out_dir={out_dir}")

    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        raise SweepError(f"probe metrics missing: {metrics_path}")
    metrics = _load_json(metrics_path)
    return out_dir, metrics


def _run_capability_train(
    *,
    repo_root: Path,
    run_root: Path,
    seed: int,
    device: str,
    precision: str,
    ring_len: int,
    slot_dim: int,
    expert_heads: int,
    batch: int,
    steps: int,
    synth_len: int,
    assoc_keys: int,
    assoc_pairs: int,
    assoc_val_range: int,
    max_samples: int,
    eval_samples: int,
    ptr_dtype: str,
    offline_only: bool,
    save_every: int,
) -> Path:
    """Run one synth assoc_byte training job and return checkpoint path."""

    run_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_root / "checkpoint.pt"
    log_path = run_root / "vraxion.log"

    env = dict(os.environ)
    env.update(
        {
            "VAR_PROJECT_ROOT": str(run_root),
            "VAR_LOGGING_PATH": str(log_path),
            "VAR_COMPUTE_DEVICE": str(device),
            "VRX_MODE": "train",
            "VRX_PRECISION": str(precision),
            "VRX_PTR_DTYPE": str(ptr_dtype),
            "VAR_RUN_SEED": str(int(seed)),
            "VRX_OFFLINE_ONLY": "1" if offline_only else "0",
            # Shape
            "VRX_RING_LEN": str(int(ring_len)),
            "VRX_SLOT_DIM": str(int(slot_dim)),
            "VRX_EXPERT_HEADS": str(int(expert_heads)),
            # Synth assoc_byte
            "VRX_SYNTH": "1",
            "VRX_SYNTH_ONCE": "1",
            "VRX_SYNTH_MODE": "assoc_byte",
            "VRX_SYNTH_LEN": str(int(synth_len)),
            "VRX_ASSOC_KEYS": str(int(assoc_keys)),
            "VRX_ASSOC_PAIRS": str(int(assoc_pairs)),
            "VRX_ASSOC_VAL_RANGE": str(int(assoc_val_range)),
            # Loop
            "VRX_BATCH_SIZE": str(int(batch)),
            "VRX_MAX_SAMPLES": str(int(max_samples)),
            "VRX_EVAL_SAMPLES": str(int(eval_samples)),
            "VRX_MAX_STEPS": str(int(steps)),
            # The modern runner path is phase-driven; set both phase and max caps
            # so capability loops remain bounded and deterministic.
            "VRX_PHASE_A_STEPS": str(int(steps)),
            "VRX_PHASE_B_STEPS": "0",
            # Saving
            "VRX_RESUME": "0",
            "VRX_CKPT": str(ckpt_path),
            "VRX_SAVE_LAST_GOOD": "1",
            # Keep both names: modern runner consumes SAVE_EVERY_STEPS, while
            # older paths may still inspect SAVE_EVERY.
            "VRX_SAVE_EVERY_STEPS": str(int(save_every)),
            "VRX_SAVE_EVERY": str(int(save_every)),
            "VRX_SAVE_HISTORY": "0",
        }
    )

    cmd = [sys.executable, "-u", str(repo_root / "Golden Draft" / "vraxion_run.py")]
    rc = subprocess.call(cmd, cwd=str(repo_root), env=env)
    if int(rc) != 0:
        raise SweepError(f"capability train rc={rc} (run_root={run_root})")

    if not ckpt_path.exists():
        # Fallbacks used by other tooling.
        for name in ("checkpoint_last_good.pt", "checkpoint.pt"):
            cand = run_root / name
            if cand.exists():
                return cand
        raise SweepError(f"checkpoint not found under run_root={run_root}")

    return ckpt_path


def _run_capability_eval(
    *,
    repo_root: Path,
    run_root: Path,
    checkpoint: Path,
    eval_samples: int,
    batch_size: int,
    device: str,
    eval_seed_offset: int,
    force_disjoint: bool,
) -> Path:
    tool = repo_root / "Golden Draft" / "tools" / "eval_ckpt_assoc_byte.py"
    if not tool.exists():
        raise SweepError(f"capability eval tool missing: {tool}")

    cmd: List[str] = [
        sys.executable,
        str(tool),
        "--run-root",
        str(run_root),
        "--checkpoint",
        str(checkpoint),
        "--eval-samples",
        str(int(eval_samples)),
        "--batch-size",
        str(int(batch_size)),
        "--device",
        str(device),
        "--eval-seed-offset",
        str(int(eval_seed_offset)),
    ]
    if force_disjoint:
        cmd.append("--force-eval-disjoint")

    rc = subprocess.call(cmd, cwd=str(repo_root))
    if int(rc) != 0:
        raise SweepError(f"capability eval rc={rc} (run_root={run_root})")

    rep = run_root / "report.json"
    if not rep.exists():
        raise SweepError(f"missing report.json after eval: {rep}")
    return rep


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run ant-ratio sweep v0 (VRA-78).")
    ap.add_argument("--batch-targets", required=True, help="Path to ant_ratio_batch_targets_v0.json")
    ap.add_argument("--out-root", default="", help="Default: bench_vault/_tmp/vra78_ant_ratio_sweep_v0/<ts>/")
    # Probe track
    ap.add_argument("--probe-colony", default="OD1_CANON_REAL")
    ap.add_argument("--probe-precision", default="fp16")
    ap.add_argument("--probe-amp", type=int, default=1)
    ap.add_argument("--probe-warmup-steps", type=int, default=5)
    ap.add_argument("--probe-measure-steps", type=int, default=50)
    ap.add_argument("--probe-force-device", default="", help="Optional VRX_FORCE_DEVICE override passed to probe.")
    # Capability track
    ap.add_argument("--cap-device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--cap-precision", default="fp32")
    ap.add_argument("--cap-seed", type=int, default=123)
    ap.add_argument("--cap-ptr-dtype", default="fp64")
    ap.add_argument("--cap-offline-only", action="store_true", default=True)
    ap.add_argument("--cap-save-every", type=int, default=50)
    ap.add_argument("--cap-eval-samples", type=int, default=4096)
    ap.add_argument("--cap-eval-seed-offset", type=int, default=1000003)
    ap.add_argument("--cap-force-disjoint", action="store_true", default=True)
    ap.add_argument("--cap-synth-len", type=int, default=256)
    ap.add_argument("--cap-assoc-keys", type=int, default=64)
    ap.add_argument("--cap-assoc-pairs", type=int, default=4)
    ap.add_argument("--cap-assoc-val-range", type=int, default=256)
    ap.add_argument("--cap-max-samples", type=int, default=8192)
    # Fairness
    ap.add_argument("--seq-len", type=int, default=256, help="Accounting seq_len used for token budget computation.")
    ap.add_argument("--token-budget", type=int, default=1_000_000, help="Fixed token budget for capability runs.")
    ap.add_argument("--min-steps", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=2000)
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    repo_root = _repo_root()

    out_root = Path(args.out_root).resolve() if str(args.out_root).strip() else (repo_root / "bench_vault" / "_tmp" / "vra78_ant_ratio_sweep_v0" / _now_ts())
    out_root.mkdir(parents=True, exist_ok=True)

    targets = _load_json(Path(args.batch_targets).resolve())
    rows = targets.get("rows") or []
    if not isinstance(rows, list):
        raise SystemExit("batch-targets JSON missing rows[]")

    # These live next to this script under Golden Draft/tools/.
    from ant_ratio_packet_v0 import TokenBudget as PacketBudget, build_packet  # type: ignore
    from ant_ratio_plot_v0 import build_html  # type: ignore

    tb = TokenBudget(
        token_budget=int(args.token_budget),
        min_steps=int(args.min_steps),
        max_steps=int(args.max_steps),
        seq_len=int(args.seq_len),
    )
    pkt_budget = PacketBudget(token_budget=int(args.token_budget), min_steps=int(args.min_steps), max_steps=int(args.max_steps))

    packets_jsonl = out_root / "ant_ratio_packets.jsonl"
    summary_csv = out_root / "ant_ratio_summary.csv"
    html_out = out_root / "ant_ratio_frontier_v0.html"

    csv_rows: List[Dict[str, Any]] = []

    for r in rows:
        if not isinstance(r, dict):
            continue
        if bool(r.get("unusable")):
            continue
        tier = str(r.get("ant_tier") or "").strip().lower()
        eh = r.get("expert_heads")
        batch = r.get("chosen_batch")
        if tier not in ("small", "real", "stress"):
            continue
        if not isinstance(eh, int) or not isinstance(batch, int):
            continue

        cfg_tag = _safe_tag(f"{tier}_E{eh}_B{batch:04d}")

        # 1) Probe run (cost metrics)
        probe_dir = out_root / "runs_probe" / cfg_tag
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_out_dir = probe_dir / "probe"
        probe_out, probe_metrics = _run_probe(
            repo_root=repo_root,
            out_dir=probe_out_dir,
            ant=_ant_preset_for_tier(tier),
            colony=str(args.probe_colony),
            out_dim=int(eh),
            batch=int(batch),
            warmup_steps=int(args.probe_warmup_steps),
            measure_steps=int(args.probe_measure_steps),
            precision=str(args.probe_precision),
            amp=int(args.probe_amp),
            force_device=str(args.probe_force_device).strip(),
        )

        # 2) Capability run (train + postmortem eval)
        ring_len, slot_dim = _ant_shape_for_tier(tier)
        steps = tb.steps_for_batch(int(batch))
        assoc_dir = out_root / "runs_assoc" / cfg_tag / f"seed{int(args.cap_seed)}"
        # Capability runs in this sweep are intentionally short/bounded. The
        # wallclock trainer checks MAX_STEPS before checkpoint cadence, so a
        # larger save interval can miss all saves. Force every-step saving to
        # guarantee checkpoint availability for postmortem eval.
        save_every = 1
        ckpt = _run_capability_train(
            repo_root=repo_root,
            run_root=assoc_dir,
            seed=int(args.cap_seed),
            device=str(args.cap_device),
            precision=str(args.cap_precision),
            ring_len=int(ring_len),
            slot_dim=int(slot_dim),
            expert_heads=int(eh),
            batch=int(batch),
            steps=int(steps),
            synth_len=int(args.cap_synth_len),
            assoc_keys=int(args.cap_assoc_keys),
            assoc_pairs=int(args.cap_assoc_pairs),
            assoc_val_range=int(args.cap_assoc_val_range),
            max_samples=int(args.cap_max_samples),
            eval_samples=int(args.cap_eval_samples),
            ptr_dtype=str(args.cap_ptr_dtype),
            offline_only=bool(args.cap_offline_only),
            save_every=int(save_every),
        )
        _run_capability_eval(
            repo_root=repo_root,
            run_root=assoc_dir,
            checkpoint=ckpt,
            eval_samples=int(args.cap_eval_samples),
            batch_size=int(batch),
            device=str(args.cap_device),
            eval_seed_offset=int(args.cap_eval_seed_offset),
            force_disjoint=bool(args.cap_force_disjoint),
        )

        # 3) Join into packet and append.
        pkt = build_packet(
            probe_run_root=probe_out,
            assoc_run_root=assoc_dir,
            ant_tier_override=str(tier),
            token_budget=pkt_budget,
            capability_steps_override=int(steps),
        )
        with packets_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pkt, sort_keys=True, ensure_ascii=True) + "\n")

        vram_ratio = pkt.get("vram_ratio_reserved")
        throughput_tokens = pkt.get("throughput_tokens_per_s")
        capability_acc = pkt.get("assoc_byte_disjoint_accuracy")
        cap_per_token: Optional[float] = None
        cap_per_vram: Optional[float] = None
        tokens_per_vram: Optional[float] = None
        target_abs_error: Optional[float] = None
        if isinstance(vram_ratio, (int, float)) and float(vram_ratio) > 0.0:
            if isinstance(throughput_tokens, (int, float)) and float(throughput_tokens) > 0.0:
                tokens_per_vram = float(throughput_tokens) / float(vram_ratio)
            if isinstance(capability_acc, (int, float)):
                cap_per_vram = float(capability_acc) / float(vram_ratio)
                target_abs_error = abs(float(vram_ratio) - 0.85)
        if isinstance(capability_acc, (int, float)) and isinstance(throughput_tokens, (int, float)) and float(throughput_tokens) > 0.0:
            cap_per_token = float(capability_acc) / float(throughput_tokens)

        csv_rows.append(
            {
                "ant_tier": tier,
                "ant_body_cells": pkt.get("ant_body_cells"),
                "ant_body_scale_vs_small": pkt.get("ant_body_scale_vs_small"),
                "expert_heads": int(eh),
                "batch": int(batch),
                "steps": int(steps),
                "probe_pass": bool(_is_probe_pass(probe_metrics)),
                "vram_ratio_reserved": vram_ratio,
                "vram_target_abs_error": target_abs_error,
                "throughput_tokens_per_s": throughput_tokens,
                "tokens_per_vram_ratio": tokens_per_vram,
                "assoc_byte_disjoint_accuracy": capability_acc,
                "assoc_acc_per_token_per_s": cap_per_token,
                "assoc_acc_per_vram_ratio": cap_per_vram,
                "probe_run_root": pkt.get("probe_run_root"),
                "assoc_run_root": pkt.get("assoc_run_root"),
            }
        )

    _write_csv(summary_csv, csv_rows)
    packets = []
    if packets_jsonl.exists():
        for ln in packets_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            s = ln.strip()
            if not s:
                continue
            packets.append(json.loads(s))

    html = build_html(packets=packets, title="VRAXION Ant Ratio Frontier v0")
    html_out.write_text(html, encoding="utf-8")

    meta = {
        "schema_version": "ant_ratio_sweep_v0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_root": str(out_root),
        "batch_targets": str(Path(args.batch_targets).resolve()),
        "token_budget": int(args.token_budget),
        "seq_len": int(args.seq_len),
        "min_steps": int(args.min_steps),
        "max_steps": int(args.max_steps),
    }
    (out_root / "sweep_meta.json").write_text(_stable_json(meta), encoding="utf-8")

    print(f"[vra78] packets: {packets_jsonl}")
    print(f"[vra78] csv: {summary_csv}")
    print(f"[vra78] html: {html_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
