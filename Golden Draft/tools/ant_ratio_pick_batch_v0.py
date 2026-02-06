"""VRA-77: Pick batch per (ant_tier, expert_heads) to target reserved VRAM ratio.

This tool runs the VRA-32 probe harness and selects a batch that lands near a
target reserved VRAM ratio while remaining PASS.

Hard rules (v0):
- Run artifacts live under repo-root bench_vault/_tmp/... (gitignored).
- PASS/FAIL is read from metrics.json (never from process exit code).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


SCHEMA_VERSION = "ant_ratio_batch_targets_v0"


class PickBatchError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def _is_pass(metrics: Dict[str, Any]) -> bool:
    return bool(
        metrics.get("stability_pass") is True
        and metrics.get("had_oom") is False
        and metrics.get("had_nan") is False
        and metrics.get("had_inf") is False
    )


@dataclass(frozen=True)
class ProbeObservation:
    batch: int
    run_root: str
    vram_ratio_reserved: Optional[float]
    stability_pass: bool
    fail_reasons: List[str]

    @property
    def is_pass(self) -> bool:
        return bool(self.stability_pass)


@dataclass(frozen=True)
class PickResult:
    chosen_batch: Optional[int]
    chosen_ratio: Optional[float]
    chosen_run_root: Optional[str]
    boundary_fail_batch: Optional[int]
    boundary_fail_run_root: Optional[str]
    unusable: bool
    notes: List[str]
    observations: List[ProbeObservation]


def _best_pass(observations: List[ProbeObservation], target: float) -> Optional[ProbeObservation]:
    cands = [o for o in observations if o.is_pass and o.vram_ratio_reserved is not None]
    if not cands:
        return None
    cands.sort(key=lambda o: (abs(float(o.vram_ratio_reserved) - float(target)), int(o.batch)))
    return cands[0]


def pick_batch_for_target(
    *,
    eval_at_batch: Callable[[int], ProbeObservation],
    target_ratio: float,
    accept_low: float,
    accept_high: float,
    max_calls: int,
) -> PickResult:
    """Select a batch near *target_ratio* while staying PASS.

    The algorithm is kill-fast and assumes VRAM ratio is mostly monotonic with batch.
    If non-monotonic behavior is observed, it stops refinement and returns the best
    observed PASS closest to target.
    """

    notes: List[str] = []
    obs: List[ProbeObservation] = []
    obs_by_batch: Dict[int, ProbeObservation] = {}
    calls = 0

    def _eval(b: int) -> ProbeObservation:
        nonlocal calls
        b = int(b)
        existing = obs_by_batch.get(b)
        if existing is not None:
            return existing
        if calls >= int(max_calls):
            raise PickBatchError(f"probe call budget exceeded (max_calls={max_calls})")
        calls += 1
        o = eval_at_batch(b)
        obs.append(o)
        obs_by_batch[b] = o
        return o

    # Step 1: start at B=1.
    o1 = _eval(1)
    if not o1.is_pass:
        return PickResult(
            chosen_batch=None,
            chosen_ratio=o1.vram_ratio_reserved,
            chosen_run_root=o1.run_root,
            boundary_fail_batch=1,
            boundary_fail_run_root=o1.run_root,
            unusable=True,
            notes=["FAIL at B=1"],
            observations=obs,
        )

    # Bracket by doubling while ratio < accept_low.
    b = 1
    last_pass = o1
    boundary_fail: Optional[ProbeObservation] = None

    while calls < int(max_calls):
        if last_pass.vram_ratio_reserved is None:
            notes.append("missing vram_ratio_reserved; cannot bracket (keep best observed PASS)")
            break
        if float(last_pass.vram_ratio_reserved) >= float(accept_low):
            break

        # Prefer a proportional jump toward the target when it's smaller than doubling.
        # This reduces the number of "too-high" probes (which can be slow/unstable under WDDM)
        # without assuming the ratio is perfectly linear.
        ratio = float(last_pass.vram_ratio_reserved)
        if ratio <= 0.0:
            b_next = int(b) * 2
        else:
            est = int(math.ceil(float(b) * (float(target_ratio) / ratio)))
            b_next = min(int(b) * 2, max(int(b) + 1, est))
        b = int(b_next)

        o = _eval(b)
        if not o.is_pass:
            boundary_fail = o
            break
        # Detect gross non-monotonicity early.
        if o.vram_ratio_reserved is not None and last_pass.vram_ratio_reserved is not None:
            if float(o.vram_ratio_reserved) + 1e-9 < float(last_pass.vram_ratio_reserved):
                notes.append("non_monotonic_ratio_detected; abort refinement")
                break
        last_pass = o

    # Optional: if still below target, step up once to get a high bracket.
    if (
        boundary_fail is None
        and last_pass.is_pass
        and last_pass.vram_ratio_reserved is not None
        and float(last_pass.vram_ratio_reserved) < float(target_ratio)
        and calls < int(max_calls)
    ):
        b2 = max(2, int(b) * 2)
        o = _eval(b2)
        if not o.is_pass:
            boundary_fail = o
        else:
            if o.vram_ratio_reserved is not None and last_pass.vram_ratio_reserved is not None:
                if float(o.vram_ratio_reserved) + 1e-9 < float(last_pass.vram_ratio_reserved):
                    notes.append("non_monotonic_ratio_detected; abort refinement")
            last_pass = o

    # If we already have something in-band, we can stop early.
    best = _best_pass(obs, target_ratio)
    if best is not None and best.vram_ratio_reserved is not None:
        if float(accept_low) <= float(best.vram_ratio_reserved) <= float(accept_high):
            return PickResult(
                chosen_batch=int(best.batch),
                chosen_ratio=float(best.vram_ratio_reserved),
                chosen_run_root=str(best.run_root),
                boundary_fail_batch=int(boundary_fail.batch) if boundary_fail is not None else None,
                boundary_fail_run_root=str(boundary_fail.run_root) if boundary_fail is not None else None,
                unusable=False,
                notes=notes,
                observations=obs,
            )

    # Refinement: binary-search-ish around target with max 6 extra evals.
    if "non_monotonic_ratio_detected; abort refinement" in notes:
        best = _best_pass(obs, target_ratio)
        return PickResult(
            chosen_batch=int(best.batch) if best else None,
            chosen_ratio=float(best.vram_ratio_reserved) if best and best.vram_ratio_reserved is not None else None,
            chosen_run_root=str(best.run_root) if best else None,
            boundary_fail_batch=int(boundary_fail.batch) if boundary_fail is not None else None,
            boundary_fail_run_root=str(boundary_fail.run_root) if boundary_fail is not None else None,
            unusable=False,
            notes=notes,
            observations=obs,
        )

    # Establish low/high brackets.
    pass_obs = [o for o in obs if o.is_pass and o.vram_ratio_reserved is not None]
    pass_obs.sort(key=lambda o: o.batch)
    lo = pass_obs[0].batch
    hi: Optional[int] = boundary_fail.batch if boundary_fail is not None else None

    # Prefer a PASS high bracket if we have one over target.
    for o in pass_obs:
        if o.vram_ratio_reserved is not None and float(o.vram_ratio_reserved) >= float(target_ratio):
            hi = o.batch
            break

    refine_steps_used = 0
    while calls < int(max_calls) and refine_steps_used < 6:
        if hi is None:
            break
        if int(hi) - int(lo) <= 1:
            break
        mid = (int(lo) + int(hi)) // 2
        mid_already = int(mid) in obs_by_batch
        o = _eval(mid)
        if not mid_already:
            refine_steps_used += 1
        if not o.is_pass:
            hi = mid
            boundary_fail = o
            continue
        if o.vram_ratio_reserved is None:
            notes.append("missing vram_ratio_reserved during refine; abort refinement")
            break
        # Non-monotonic guard.
        prev = max([x for x in pass_obs if x.batch <= o.batch], key=lambda x: x.batch, default=None)
        if prev is not None and prev.vram_ratio_reserved is not None:
            if float(o.vram_ratio_reserved) + 1e-9 < float(prev.vram_ratio_reserved):
                notes.append("non_monotonic_ratio_detected; abort refinement")
                break
        pass_obs.append(o)
        pass_obs.sort(key=lambda x: x.batch)
        if float(o.vram_ratio_reserved) < float(target_ratio):
            lo = mid
        else:
            hi = mid

    best = _best_pass(obs, target_ratio)
    return PickResult(
        chosen_batch=int(best.batch) if best else None,
        chosen_ratio=float(best.vram_ratio_reserved) if best and best.vram_ratio_reserved is not None else None,
        chosen_run_root=str(best.run_root) if best else None,
        boundary_fail_batch=int(boundary_fail.batch) if boundary_fail is not None else None,
        boundary_fail_run_root=str(boundary_fail.run_root) if boundary_fail is not None else None,
        unusable=False,
        notes=notes,
        observations=obs,
    )


def _run_probe_once(
    *,
    repo_root: Path,
    probe_tool: Path,
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
) -> ProbeObservation:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        str(probe_tool),
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

    # Never infer PASS/FAIL from return code; only reject "hard" errors (rc=2).
    rc = subprocess.call(cmd, cwd=str(repo_root), env=env)
    if int(rc) == 2:
        raise PickBatchError(f"probe harness rc=2 (invalid args or cannot write): out_dir={out_dir}")

    metrics_path = out_dir / "metrics.json"
    env_path = out_dir / "env.json"
    if not metrics_path.exists() or not env_path.exists():
        raise PickBatchError(f"probe artifacts missing under out_dir={out_dir}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    env_obj = json.loads(env_path.read_text(encoding="utf-8"))

    reserved = metrics.get("peak_vram_reserved_bytes")
    total_vram = env_obj.get("total_vram_bytes")
    ratio: Optional[float] = None
    if isinstance(reserved, int) and isinstance(total_vram, int) and total_vram > 0:
        ratio = float(reserved) / float(total_vram)

    stability_pass = _is_pass(metrics)
    fail_reasons = list(metrics.get("fail_reasons") or [])

    return ProbeObservation(
        batch=int(batch),
        run_root=str(out_dir),
        vram_ratio_reserved=ratio,
        stability_pass=bool(stability_pass),
        fail_reasons=fail_reasons,
    )


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_csv_strs(s: str) -> List[str]:
    out: List[str] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part)
    return out


def _ant_preset_for_tier(tier: str) -> str:
    t = str(tier).strip().lower()
    if t == "small":
        return "OD1_CANON_SMALL"
    if t == "real":
        return "OD1_CANON_REAL"
    if t == "stress":
        return "OD1_CANON_STRESS"
    raise PickBatchError(f"unknown ant_tier: {tier!r} (expected small,real,stress)")


def _safe_tag(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pick batch targeting reserved VRAM ratio (VRA-77).")
    ap.add_argument("--ant-tiers", default="small,real,stress")
    ap.add_argument("--expert-heads", default="1,2,4,8,16")
    ap.add_argument("--colony", default="OD1_CANON_REAL")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--measure-steps", type=int, default=50)
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--accept-low", type=float, default=0.82)
    ap.add_argument("--accept-high", type=float, default=0.88)
    ap.add_argument("--max-calls", type=int, default=10, help="Hard cap on probe calls per config.")
    ap.add_argument("--force-device", default="", help="Optional VRX_FORCE_DEVICE override passed to probe (cpu/cuda).")
    ap.add_argument("--out-root", default="", help="Output root dir. Default: bench_vault/_tmp/vra77_batch_target_v0/<ts>/")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    repo_root = _repo_root()
    probe_tool = repo_root / "Golden Draft" / "tools" / "gpu_capacity_probe.py"
    if not probe_tool.exists():
        print(f"ERROR: probe harness missing: {probe_tool}", file=sys.stderr)
        return 2

    out_root = Path(args.out_root).resolve() if str(args.out_root).strip() else (repo_root / "bench_vault" / "_tmp" / "vra77_batch_target_v0" / _now_ts())
    out_root.mkdir(parents=True, exist_ok=True)

    ant_tiers = _parse_csv_strs(args.ant_tiers)
    expert_heads = _parse_csv_ints(args.expert_heads)

    rows: List[Dict[str, Any]] = []

    for tier in ant_tiers:
        ant_preset = _ant_preset_for_tier(tier)
        for eh in expert_heads:
            tag = _safe_tag(f"{tier}_E{eh}")
            runs_dir = out_root / "runs" / tag

            def _eval_at_batch(b: int) -> ProbeObservation:
                run_dir = runs_dir / _safe_tag(
                    f"{tier}_x_real_E{int(eh)}_B{int(b):04d}_{args.precision}_amp{int(args.amp)}_ms{int(args.measure_steps)}"
                )
                return _run_probe_once(
                    repo_root=repo_root,
                    probe_tool=probe_tool,
                    out_dir=run_dir,
                    ant=ant_preset,
                    colony=str(args.colony),
                    out_dim=int(eh),
                    batch=int(b),
                    warmup_steps=int(args.warmup_steps),
                    measure_steps=int(args.measure_steps),
                    precision=str(args.precision),
                    amp=int(args.amp),
                    force_device=str(args.force_device).strip(),
                )

            try:
                res = pick_batch_for_target(
                    eval_at_batch=_eval_at_batch,
                    target_ratio=float(args.target),
                    accept_low=float(args.accept_low),
                    accept_high=float(args.accept_high),
                    max_calls=int(args.max_calls),
                )
            except Exception as exc:
                print(f"WARN: config {tier}/E{eh} failed during pick: {exc}", file=sys.stderr)
                res = PickResult(
                    chosen_batch=None,
                    chosen_ratio=None,
                    chosen_run_root=None,
                    boundary_fail_batch=None,
                    boundary_fail_run_root=None,
                    unusable=True,
                    notes=[f"exception: {exc}"],
                    observations=[],
                )

            row: Dict[str, Any] = {
                "ant_tier": str(tier),
                "colony": str(args.colony),
                "expert_heads": int(eh),
                "precision": str(args.precision),
                "amp": int(args.amp),
                "target_ratio": float(args.target),
                "accept_band": [float(args.accept_low), float(args.accept_high)],
                "chosen_batch": res.chosen_batch,
                "chosen_ratio": res.chosen_ratio,
                "chosen_run_root": res.chosen_run_root,
                "boundary_fail_batch": res.boundary_fail_batch,
                "boundary_fail_run_root": res.boundary_fail_run_root,
                "unusable": bool(res.unusable),
                "notes": list(res.notes),
            }
            rows.append(row)

    out_obj: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime_now_utc(),
        "out_root": str(out_root),
        "rows": rows,
    }

    out_path = out_root / "ant_ratio_batch_targets_v0.json"
    out_path.write_text(_stable_json(out_obj), encoding="utf-8")
    print(f"[vra77] wrote: {out_path}")
    return 0


def datetime_now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
