"""Deterministic repulsion (diversity) sweep for PrismionLiquid on assoc_byte.

This is a "result-safe" runner:
- Runs are only considered valid if `report.json` exists under the run root.
- All runs are pinned to CPU + eval-disjoint so comparisons are meaningful.

It is intentionally placed in Golden Draft tooling.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SweepPoint:
    name: str
    env: Dict[str, str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _bench_entrypoint(repo_root: Path) -> Path:
    return repo_root / "Golden Draft" / "benchmarks" / "boot_synth_markov0" / "run_boot_synth_markov0.py"


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_tag(s: str) -> str:
    # Keep Windows-friendly filenames.
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run_one(*, repo_root: Path, run_root: Path, extra_env: Dict[str, str], args: Sequence[str]) -> Path:
    _ensure_dir(run_root)
    env = dict(os.environ)
    env.update(extra_env)
    env["VAR_PROJECT_ROOT"] = str(run_root)
    env["VAR_LOGGING_PATH"] = str(run_root / "vraxion.log")

    bench = _bench_entrypoint(repo_root)
    cmd = [sys.executable, "-u", str(bench), *args, "--respect-env"]
    # Avoid flooding the console: capture child output into per-run log files.
    out_path = run_root / "stdout.log"
    err_path = run_root / "stderr.log"
    with out_path.open("wb") as out_f, err_path.open("wb") as err_f:
        rc = subprocess.call(cmd, env=env, stdout=out_f, stderr=err_f)
    if rc != 0:
        raise RuntimeError(f"bench entrypoint exited rc={rc}")

    report = run_root / "report.json"
    if not report.exists():
        raise RuntimeError(f"missing report.json under {run_root}")
    return run_root


def _read_eval_acc(report_path: Path) -> float:
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    ev = rep.get("eval", {}) or {}
    return float(ev.get("eval_acc", 0.0))


def _fmt_pct(x: float) -> str:
    return f"{(100.0 * float(x)):.2f}%"


def _base_env(*, seed: int, val_range: int, n: int) -> Dict[str, str]:
    # Disable dashboard / autorun to keep this runner predictable.
    env: Dict[str, str] = {
        "VRX_LIVE": "0",
        "VRX_LIVE_DASH": "0",
        "VRX_LIVE_OPEN": "0",
        "VAR_RUN_SEED": str(int(seed)),
        "VAR_COMPUTE_DEVICE": "cpu",
        # "Tax collection" fix: diversity penalty is routed through move_penalty.
        "VRX_LMOVE": "1.0",
        # Assoc-byte knobs.
        "VRX_ASSOC_VAL_RANGE": str(int(val_range)),
        # Keep fixed unless explicitly sweeping them.
        "VRX_ASSOC_KEYS": "4",
        "VRX_ASSOC_PAIRS": "3",
        # Prismion mesh config (bank topology).
        "VRX_PRISMION_TOPOLOGY": "bank",
        "VRX_PRISMION_N": str(int(n)),
        "VRX_PRISMION_TOPK": str(int(n)),  # update all (no starvation in update rule)
        "VRX_PRISMION_ALPHA": "1.0",
        "VRX_PRISMION_ID_SCALE": "0.10",
        # Force off unrelated experimental couplings unless explicitly enabled.
        "PRISMN_BUSINJ": "0",
        "PRISMN_MIXING": "0",
        "PRISMN_TUNING": "0",
        "PRISMN_TGTDLT": "0",
        "VRX_PRISMION_ALPHA_ADAPT": "0",
        "VRX_THINK_RING": "0",
    }
    # Mosaic global integrator (global workspace lanes).
    env["PRISMN_INTEG"] = "1"
    env["PRISMN_IMODE"] = "mosaic"
    env["PRISMN_IALPHA"] = "1.0"
    return env


def _bench_args(*, steps: int, seq_len: int, batch_size: int, max_samples: int, eval_samples: int, lr: float) -> List[str]:
    return [
        "--arch",
        "prismion_liquid",
        "--synth-mode",
        "assoc_byte",
        "--eval-disjoint",
        "--steps",
        str(int(steps)),
        "--seq-len",
        str(int(seq_len)),
        "--batch-size",
        str(int(batch_size)),
        "--max-samples",
        str(int(max_samples)),
        "--eval-samples",
        str(int(eval_samples)),
        "--ring-len",
        str(int(256)),
        "--slot-dim",
        str(int(128)),
        "--device",
        "cpu",
        "--lr",
        str(float(lr)),
        "--tag",
        "assoc_byte_repulsion_sweep",
    ]


def _plan_dvlam_sweep(*, base: Dict[str, str], dvtaus: float, dvlams: Iterable[float]) -> List[SweepPoint]:
    pts: List[SweepPoint] = []
    for lam in dvlams:
        env = dict(base)
        env["PRISMN_DVTAU"] = str(float(dvtaus))
        env["PRISMN_DVLAM"] = str(float(lam))
        name = f"dvlam_{lam:g}_tau_{dvtaus:g}"
        pts.append(SweepPoint(name=name, env=env))
    return pts


def _plan_dvtau_sweep(*, base: Dict[str, str], dvlam: float, dvtaus: Iterable[float]) -> List[SweepPoint]:
    pts: List[SweepPoint] = []
    for tau in dvtaus:
        env = dict(base)
        env["PRISMN_DVLAM"] = str(float(dvlam))
        env["PRISMN_DVTAU"] = str(float(tau))
        name = f"dvlam_{dvlam:g}_tau_{tau:g}"
        pts.append(SweepPoint(name=name, env=env))
    return pts


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except Exception:
            continue
    return rows


def _key_for_row(r: Dict[str, object]) -> str:
    # Stable-ish dedupe key for resume.
    return "|".join(
        [
            str(r.get("stage", "")),
            str(r.get("name", "")),
            str(r.get("seed", "")),
            str(r.get("val_range", "")),
            str(r.get("n", "")),
            str(r.get("dvlambda", "")),
            str(r.get("dvtau", "")),
        ]
    )


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sweep_assoc_repulsion.py")
    p.add_argument("--out-jsonl", default="", help="Write/append results to this jsonl (default: timestamped under bench_vault/sweeps).")
    p.add_argument("--resume", action="store_true", help="If --out-jsonl exists, skip already-completed points.")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_cli = _parse_args(argv)
    repo_root = _repo_root()
    out_dir = repo_root / "bench_vault" / "sweeps"
    _ensure_dir(out_dir)

    # Frozen experiment settings (fast gate).
    seed0 = 126
    val_range = 16
    n = 3
    steps = 300
    seq_len = 128
    batch_size = 32
    max_samples = 2000
    eval_samples = 512
    lr = 1e-3

    base = _base_env(seed=seed0, val_range=val_range, n=n)
    bench_args = _bench_args(
        steps=steps,
        seq_len=seq_len,
        batch_size=batch_size,
        max_samples=max_samples,
        eval_samples=eval_samples,
        lr=lr,
    )

    # Stage 0: baseline (diversity off).
    if str(args_cli.out_jsonl).strip():
        results_jsonl = Path(str(args_cli.out_jsonl).strip())
        if not results_jsonl.is_absolute():
            results_jsonl = repo_root / results_jsonl
    else:
        results_jsonl = out_dir / f"assoc_byte_repulsion_sweep_{_now_ts()}.jsonl"

    done: set[str] = set()
    if bool(args_cli.resume) and results_jsonl.exists():
        for r in _read_jsonl(results_jsonl):
            done.add(_key_for_row(r))
        print(f"[sweep] resume enabled: {len(done)} completed points found in {results_jsonl}")
    rows: List[Dict[str, object]] = []

    def _run_point(pt: SweepPoint) -> Tuple[float, Path]:
        group = f"assoc_byte_v{val_range}_n{n}_mosaic"
        tag = _safe_tag(f"{group}_{pt.name}")
        run_root = repo_root / "bench_vault" / "benchmarks" / group / f"{_now_ts()}_{tag}"
        _run_one(repo_root=repo_root, run_root=run_root, extra_env=pt.env, args=bench_args)
        acc = _read_eval_acc(run_root / "report.json")
        return acc, run_root

    print("[sweep] Stage 0: baseline (diversity OFF)")
    base0 = dict(base)
    base0["PRISMN_DVLAM"] = "0.0"
    base0["PRISMN_DVTAU"] = "0.1"
    key0 = _key_for_row(
        {
            "stage": "baseline",
            "name": "baseline_div0",
            "seed": seed0,
            "val_range": val_range,
            "n": n,
            "dvlambda": 0.0,
            "dvtau": 0.1,
        }
    )
    if key0 in done:
        print("[sweep] baseline already present; loading as best_acc seed=126 from last row (best-effort)")
        acc0 = 0.0
        root0 = repo_root
        for r in reversed(_read_jsonl(results_jsonl)):
            if _key_for_row(r) == key0 and "eval_acc" in r:
                try:
                    acc0 = float(r.get("eval_acc", 0.0))
                except Exception:
                    acc0 = 0.0
                try:
                    root0 = Path(str(r.get("run_root", str(repo_root))))
                except Exception:
                    root0 = repo_root
                break
        print(f"[sweep] baseline disjoint eval acc (resumed): {_fmt_pct(acc0)}")
    else:
        try:
            acc0, root0 = _run_point(SweepPoint(name="baseline_div0", env=base0))
            row0: Dict[str, object] = {
                "stage": "baseline",
                "name": "baseline_div0",
                "seed": seed0,
                "val_range": val_range,
                "n": n,
                "dvlambda": 0.0,
                "dvtau": 0.1,
                "eval_acc": acc0,
                "eval_acc_pct": _fmt_pct(acc0),
                "run_root": str(root0),
            }
            rows.append(row0)
            _write_jsonl(results_jsonl, rows)
            done.add(_key_for_row(row0))
            rows.clear()
            print(f"[sweep] baseline disjoint eval acc: {_fmt_pct(acc0)}")
        except Exception as e:
            row0 = {
                "stage": "baseline",
                "name": "baseline_div0",
                "seed": seed0,
                "val_range": val_range,
                "n": n,
                "dvlambda": 0.0,
                "dvtau": 0.1,
                "error": repr(e),
            }
            rows.append(row0)
            _write_jsonl(results_jsonl, rows)
            rows.clear()
            raise

    # Stage 1: DVLAM ladder sweep.
    print("[sweep] Stage 1: DVLAM ladder (DVTAU fixed at 0.1)")
    # Baseline (diversity off) already ran in Stage 0, so omit 0.0 here.
    dvlams = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    best_lam = 0.0
    best_acc = acc0
    for pt in _plan_dvlam_sweep(base=base, dvtaus=0.1, dvlams=dvlams):
        key = _key_for_row(
            {
                "stage": "dvlam",
                "name": pt.name,
                "seed": seed0,
                "val_range": val_range,
                "n": n,
                "dvlambda": float(pt.env.get("PRISMN_DVLAM", "0.0")),
                "dvtau": float(pt.env.get("PRISMN_DVTAU", "0.1")),
            }
        )
        if key in done:
            print(f"[sweep] skip (resume): {pt.name}")
            continue

        try:
            acc, root = _run_point(pt)
        except Exception as e:
            lam = float(pt.env.get("PRISMN_DVLAM", "0.0"))
            tau = float(pt.env.get("PRISMN_DVTAU", "0.1"))
            row = {
                "stage": "dvlam",
                "name": pt.name,
                "seed": seed0,
                "val_range": val_range,
                "n": n,
                "dvlambda": lam,
                "dvtau": tau,
                "error": repr(e),
            }
            rows.append(row)
            _write_jsonl(results_jsonl, rows)
            done.add(_key_for_row(row))
            rows.clear()
            print(f"[sweep] {pt.name}: ERROR {e!r}")
            continue
        lam = float(pt.env.get("PRISMN_DVLAM", "0.0"))
        tau = float(pt.env.get("PRISMN_DVTAU", "0.1"))
        row = {
            "stage": "dvlam",
            "name": pt.name,
            "seed": seed0,
            "val_range": val_range,
            "n": n,
            "dvlambda": lam,
            "dvtau": tau,
            "eval_acc": acc,
            "eval_acc_pct": _fmt_pct(acc),
            "run_root": str(root),
        }
        rows.append(row)
        if acc > best_acc:
            best_acc = acc
            best_lam = lam
        print(f"[sweep] {pt.name}: {_fmt_pct(acc)}")
        _write_jsonl(results_jsonl, rows)
        done.add(_key_for_row(row))
        rows.clear()

    print(f"[sweep] best DVLAM so far: {best_lam:g} -> {_fmt_pct(best_acc)}")

    # Stage 2: DVTAU sweep at best DVLAM (if non-zero).
    print("[sweep] Stage 2: DVTAU sweep at best DVLAM")
    dvtaus = [0.0, 0.05, 0.10, 0.20, 0.30]
    best_tau = 0.1
    best_pair_acc = best_acc
    for pt in _plan_dvtau_sweep(base=base, dvlam=best_lam, dvtaus=dvtaus):
        key = _key_for_row(
            {
                "stage": "dvtau",
                "name": pt.name,
                "seed": seed0,
                "val_range": val_range,
                "n": n,
                "dvlambda": float(pt.env.get("PRISMN_DVLAM", "0.0")),
                "dvtau": float(pt.env.get("PRISMN_DVTAU", "0.1")),
            }
        )
        if key in done:
            print(f"[sweep] skip (resume): {pt.name}")
            continue

        try:
            acc, root = _run_point(pt)
        except Exception as e:
            lam = float(pt.env.get("PRISMN_DVLAM", "0.0"))
            tau = float(pt.env.get("PRISMN_DVTAU", "0.1"))
            row = {
                "stage": "dvtau",
                "name": pt.name,
                "seed": seed0,
                "val_range": val_range,
                "n": n,
                "dvlambda": lam,
                "dvtau": tau,
                "error": repr(e),
            }
            rows.append(row)
            _write_jsonl(results_jsonl, rows)
            done.add(_key_for_row(row))
            rows.clear()
            print(f"[sweep] {pt.name}: ERROR {e!r}")
            continue
        lam = float(pt.env.get("PRISMN_DVLAM", "0.0"))
        tau = float(pt.env.get("PRISMN_DVTAU", "0.1"))
        row = {
            "stage": "dvtau",
            "name": pt.name,
            "seed": seed0,
            "val_range": val_range,
            "n": n,
            "dvlambda": lam,
            "dvtau": tau,
            "eval_acc": acc,
            "eval_acc_pct": _fmt_pct(acc),
            "run_root": str(root),
        }
        rows.append(row)
        if acc > best_pair_acc:
            best_pair_acc = acc
            best_tau = tau
        print(f"[sweep] {pt.name}: {_fmt_pct(acc)}")
        _write_jsonl(results_jsonl, rows)
        done.add(_key_for_row(row))
        rows.clear()

    print(f"[sweep] best (DVLAM,DVTAU): ({best_lam:g},{best_tau:g}) -> {_fmt_pct(best_pair_acc)}")

    # Stage 3: multi-seed confirmation for the best pair.
    print("[sweep] Stage 3: confirm best pair across seeds 126/127/128")
    seeds = [126, 127, 128]
    for sd in seeds:
        env = _base_env(seed=sd, val_range=val_range, n=n)
        env["PRISMN_DVLAM"] = str(float(best_lam))
        env["PRISMN_DVTAU"] = str(float(best_tau))
        pt = SweepPoint(name=f"confirm_seed{sd}", env=env)
        key = _key_for_row(
            {
                "stage": "confirm",
                "name": pt.name,
                "seed": sd,
                "val_range": val_range,
                "n": n,
                "dvlambda": float(best_lam),
                "dvtau": float(best_tau),
            }
        )
        if key in done:
            print(f"[sweep] skip (resume): {pt.name}")
            continue

        try:
            acc, root = _run_point(pt)
        except Exception as e:
            row = {
                "stage": "confirm",
                "name": pt.name,
                "seed": sd,
                "val_range": val_range,
                "n": n,
                "dvlambda": float(best_lam),
                "dvtau": float(best_tau),
                "error": repr(e),
            }
            rows.append(row)
            _write_jsonl(results_jsonl, rows)
            done.add(_key_for_row(row))
            rows.clear()
            print(f"[sweep] seed={sd} -> ERROR {e!r}")
            continue

        row = {
            "stage": "confirm",
            "name": pt.name,
            "seed": sd,
            "val_range": val_range,
            "n": n,
            "dvlambda": float(best_lam),
            "dvtau": float(best_tau),
            "eval_acc": acc,
            "eval_acc_pct": _fmt_pct(acc),
            "run_root": str(root),
        }
        rows.append(row)
        print(f"[sweep] seed={sd} -> {_fmt_pct(acc)}")
        _write_jsonl(results_jsonl, rows)
        done.add(_key_for_row(row))
        rows.clear()

    print(f"[sweep] results written: {results_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
