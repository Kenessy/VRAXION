"""Run a 2-seed benchmark datapoint (same config) and ingest/gate results.

Design goals:
- Single command to produce a *rankable* datapoint pair (two seeds).
- No inline PowerShell logic; everything is driven from Python.
- Auto-launch the Streamlit results dashboard (detached) so the user can watch.
- Optional microprobe (default 20 steps) to estimate ETA before committing.

This is intended to be the lowest-friction "datapoint miner" for the
VRAXION multi-axis frontier (N, out_dim, slot_dim, ring_len, etc.).

Usage (example):
  python "Golden Draft/tools/datapoint_2seed_runner.py" ^
    --tag dev32_sd4_rl64_od1 ^
    --seeds 111,222 ^
    --steps 1000 ^
    --hard-eval-samples 4096
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


def _repo_root() -> Path:
    # .../Golden Draft/tools/datapoint_2seed_runner.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _parse_seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise SystemExit("no seeds provided (use --seeds 111,222)")
    return out


def _ensure_dashboard(repo_root: Path, port: int) -> None:
    launcher = repo_root / "Golden Draft" / "tools" / "_scratch" / "launch_results_dashboard.py"
    if not launcher.exists():
        raise SystemExit(f"dashboard launcher missing: {launcher}")
    subprocess.run(
        [sys.executable, str(launcher), "--port", str(int(port))],
        cwd=str(repo_root),
        check=True,
    )


def _ci95(p: float, n: int) -> float:
    # Normal approximation (good enough for PASS/FAIL gating at our eval_n).
    if n <= 0:
        return float("nan")
    import math

    p = max(0.0, min(1.0, float(p)))
    se = math.sqrt(max(0.0, p * (1.0 - p) / float(n)))
    return 1.96 * se


def _read_report(run_root: Path) -> dict:
    import json

    report_path = run_root / "report.json"
    if not report_path.exists():
        raise RuntimeError(f"missing report.json: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _gate_row(*, run_root: Path) -> dict:
    r = _read_report(run_root)
    settings = r.get("settings") or {}
    ev = r.get("eval") or {}
    train = r.get("train") or {}

    val_range = int(settings.get("val_range") or 0)
    chance = (1.0 / float(val_range)) if val_range else float("nan")
    acc = float(ev.get("eval_acc") or 0.0)
    n = int(ev.get("eval_n") or 0)
    ci = _ci95(acc, n)
    lower = acc - ci
    pass_gate = bool(val_range) and bool(n) and (lower > chance)

    return {
        "seed": settings.get("seed"),
        "steps": int(train.get("steps") or 0),
        "val_range": val_range,
        "chance": chance,
        "eval_acc": acc,
        "eval_n": n,
        "ci95": ci,
        "lower95": lower,
        "acc_delta": acc - chance if val_range else float("nan"),
        "pass": pass_gate,
    }


def _postmortem_eval(
    *,
    repo_root: Path,
    run_root: Path,
    checkpoint: Optional[Path],
    eval_samples: int,
    batch_size: int,
    device: str,
    prismn_id_scale: float,
) -> None:
    """Overwrite run_root/report.json with a high-precision eval from the latest checkpoint."""
    tool = repo_root / "Golden Draft" / "tools" / "eval_ckpt_assoc_byte.py"
    if not tool.exists():
        raise SystemExit(f"missing postmortem eval tool: {tool}")

    if checkpoint is None or not checkpoint.exists():
        print(f"[postmortem] skip (no checkpoint): {run_root}")
        return

    cmd = [
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
        "--prismn-id-scale",
        str(float(prismn_id_scale)),
    ]

    subprocess.run(
        cmd,
        cwd=str(repo_root / "Golden Draft"),
        check=True,
    )


def _find_checkpoint(run_root: Path) -> Optional[Path]:
    # Prefer common stable names; fall back to latest step checkpoint.
    for name in ("checkpoint_last_good.pt", "checkpoint.pt"):
        p = run_root / name
        if p.exists():
            return p

    ck_dir = run_root / "checkpoints"
    if ck_dir.exists():
        cands = list(ck_dir.glob("checkpoint_step_*.pt"))
        if cands:
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0]
    return None


def _run_boot(
    *,
    repo_root: Path,
    tag: str,
    seed: int,
    steps: int,
    seq_len: int,
    batch_size: int,
    max_samples: int,
    eval_samples: int,
    ring_len: int,
    slot_dim: int,
    device: str,
    lr: float,
    model: str,
    prismn_n: int,
    prismn_mode: str,
    prismn_shared: int,
    prismn_out_dim: int,
    prismn_id_scale: float,
    synth_mode: str,
    keys: int,
    pairs: int,
    val_range: int,
    assoc_unique_keys: bool,
    assoc_mq_dup: int,
    assoc_mq_grouped: int,
    eval_disjoint: bool,
    eval_ptr_deterministic: bool,
    abort_after: int,
    abort_acc: float,
) -> Tuple[Path, str]:
    boot = repo_root / "Golden Draft" / "benchmarks" / "boot_synth_assoc_byte" / "run_boot_synth_assoc_byte.py"
    if not boot.exists():
        raise SystemExit(f"boot runner missing: {boot}")

    cmd = [
        sys.executable,
        "-u",
        str(boot),
        "--tag",
        tag,
        "--seed",
        str(int(seed)),
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
        str(int(ring_len)),
        "--slot-dim",
        str(int(slot_dim)),
        "--device",
        str(device),
        "--lr",
        str(float(lr)),
        "--model",
        str(model),
        "--prismn-n",
        str(int(prismn_n)),
        "--prismn-mode",
        str(prismn_mode),
        "--prismn-shared",
        str(int(prismn_shared)),
        "--prismn-out-dim",
        str(int(prismn_out_dim)),
        "--prismn-id-scale",
        str(float(prismn_id_scale)),
        "--synth-mode",
        str(synth_mode),
        "--keys",
        str(int(keys)),
        "--pairs",
        str(int(pairs)),
        "--val-range",
        str(int(val_range)),
        "--assoc-mq-dup",
        str(int(assoc_mq_dup)),
        "--assoc-mq-grouped",
        str(int(assoc_mq_grouped)),
    ]

    if assoc_unique_keys:
        cmd.append("--assoc-unique-keys")
    if eval_disjoint:
        cmd.append("--eval-disjoint")
    if eval_ptr_deterministic:
        cmd.append("--eval-ptr-deterministic")
    if int(abort_after) > 0:
        cmd.extend(["--abort-after", str(int(abort_after)), "--abort-acc", str(float(abort_acc))])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    p = subprocess.run(cmd, cwd=str(repo_root), env=env, text=True, capture_output=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"boot runner failed rc={p.returncode}\n{out[-4000:]}")

    # Parse run_root from the runner log line:
    #   [bench] report saved: S:\...\report.json
    m = re.search(r"\[bench\]\s+report saved:\s+(?P<path>.+report\.json)\s*$", out, flags=re.M)
    if not m:
        raise RuntimeError(f"Could not locate report.json path in runner output.\n{out[-4000:]}")
    report_path = Path(m.group("path").strip()).resolve()
    run_root = report_path.parent
    return run_root, out


def _estimate_sec_per_step(run_root: Path, steps: int) -> Optional[float]:
    log_path = run_root / "vraxion.log"
    if not log_path.exists():
        return None
    # Example line:
    # [..] step 0020/0500 | loss ... | t=12.3s | ...
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"step\s+\d+/\d+\s+\|\s+loss\s+[\d\.]+\s+\|\s+t=(?P<t>[\d\.]+)s", txt)
    if not m:
        return None
    try:
        t_s = float(m.group("t"))
        if steps <= 0:
            return None
        return t_s / float(steps)
    except Exception:
        return None


def _ingest_and_gate(repo_root: Path) -> None:
    # Ingest into results_master.csv, then apply professor gate into derived/ tables.
    ingest = repo_root / "Golden Draft" / "tools" / "results_ingest.py"
    gate = repo_root / "Golden Draft" / "tools" / "results_professor_gate.py"
    if not ingest.exists():
        raise SystemExit(f"missing ingest tool: {ingest}")
    if not gate.exists():
        raise SystemExit(f"missing professor gate tool: {gate}")
    subprocess.run([sys.executable, str(ingest)], cwd=str(repo_root), check=True)
    subprocess.run([sys.executable, str(gate)], cwd=str(repo_root), check=True)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="dev32_sd4_rl64_od1")
    p.add_argument("--seeds", type=str, default="111,222")
    # Hard cap: fixed max length (never extend).
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=4096)
    # Training-time eval sample count (used for the soft-cap check + end-of-run smoke).
    p.add_argument("--eval-samples", type=int, default=512)
    # Postmortem (hard) eval sample count used for PASS/FAIL gating.
    p.add_argument("--hard-eval-samples", type=int, default=4096)
    p.add_argument("--ring-len", type=int, default=64)
    p.add_argument("--slot-dim", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model", type=str, default="prismion_hallway_bank")
    p.add_argument("--prismn-n", type=int, default=32)
    p.add_argument("--prismn-mode", type=str, default="mosaic", choices=["mean", "mosaic", "orb_sum"])
    p.add_argument("--prismn-shared", type=int, default=1, choices=[0, 1])
    p.add_argument("--prismn-out-dim", type=int, default=1)
    p.add_argument("--prismn-id-scale", type=float, default=0.02)
    p.add_argument("--synth-mode", type=str, default="assoc_byte", choices=["assoc_byte", "assoc_clean", "assoc_mix", "logic_nand"])
    p.add_argument("--keys", type=int, default=64)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--val-range", type=int, default=16)
    p.add_argument("--assoc-unique-keys", action="store_true", default=True)
    p.add_argument("--assoc-mq-dup", type=int, default=4)
    p.add_argument("--assoc-mq-grouped", type=int, default=1, choices=[0, 1])
    p.add_argument("--eval-disjoint", action="store_true", default=True)
    p.add_argument("--eval-ptr-deterministic", action="store_true", default=True)
    # Soft cap: abort early ONLY if clearly dead (time saver, not a ranking gate).
    p.add_argument("--soft-cap-steps", type=int, default=200)
    p.add_argument(
        "--soft-abort-margin",
        type=float,
        default=0.01,
        help="Abort if eval_acc < (chance - margin) at soft-cap.",
    )
    p.add_argument("--dashboard-port", type=int, default=8520)
    # Optional ETA probe; disabled by default for datapoint mining.
    p.add_argument("--microprobe-steps", type=int, default=0)
    p.add_argument("--microprobe-eval-samples", type=int, default=0)
    args = p.parse_args(argv)

    repo_root = _repo_root()

    # Always start the dashboard so the user can monitor live.
    _ensure_dashboard(repo_root, int(args.dashboard_port))

    seeds = _parse_seeds(str(args.seeds))
    if len(seeds) < 2:
        raise SystemExit("need at least 2 seeds (e.g., --seeds 111,222)")
    seeds2 = seeds[:2]

    # Microprobe (ETA guard). Optional; disabled by default for datapoint mining.
    if int(args.microprobe_steps) > 0 and int(args.microprobe_eval_samples) > 0:
        probe_tag = f"zprobe_{args.tag}_seed{seeds2[0]}"
        t0 = time.time()
        probe_root, _ = _run_boot(
            repo_root=repo_root,
            tag=probe_tag,
            seed=int(seeds2[0]),
            steps=int(args.microprobe_steps),
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            max_samples=int(max(args.max_samples, args.microprobe_eval_samples)),
            eval_samples=int(args.microprobe_eval_samples),
            ring_len=int(args.ring_len),
            slot_dim=int(args.slot_dim),
            device=str(args.device),
            lr=float(args.lr),
            model=str(args.model),
            prismn_n=int(args.prismn_n),
            prismn_mode=str(args.prismn_mode),
            prismn_shared=int(args.prismn_shared),
            prismn_out_dim=int(args.prismn_out_dim),
            prismn_id_scale=float(args.prismn_id_scale),
            synth_mode=str(args.synth_mode),
            keys=int(args.keys),
            pairs=int(args.pairs),
            val_range=int(args.val_range),
            assoc_unique_keys=bool(args.assoc_unique_keys),
            assoc_mq_dup=int(args.assoc_mq_dup),
            assoc_mq_grouped=int(args.assoc_mq_grouped),
            eval_disjoint=bool(args.eval_disjoint),
            eval_ptr_deterministic=bool(args.eval_ptr_deterministic),
            abort_after=0,
            abort_acc=0.0,
        )
        wall = time.time() - t0
        sps = _estimate_sec_per_step(probe_root, int(args.microprobe_steps))
        if sps is not None:
            eta_train_one = sps * float(args.steps)
            # crude eval scaling estimate from probe wall time minus training time
            eta_eval_probe = max(0.0, wall - (sps * float(args.microprobe_steps)))
            eta_eval_one = eta_eval_probe * (float(args.eval_samples) / float(args.microprobe_eval_samples))
            eta_one = eta_train_one + eta_eval_one
            eta_two = 2.0 * eta_one
            print(f"[microprobe] run_root={probe_root}")
            # ASCII-only output (Windows console encodings vary).
            print(f"[microprobe] sec_per_step~{sps:.3f} wall~{wall:.1f}s")
            print(
                f"[microprobe] ETA per seed~{eta_one/60.0:.1f} min "
                f"(train~{eta_train_one/60.0:.1f}, eval~{eta_eval_one/60.0:.1f})"
            )
            print(f"[microprobe] ETA two seeds~{eta_two/60.0:.1f} min")
        else:
            print(f"[microprobe] run_root={probe_root} (could not estimate sec/step from log)")

    # Full 2-seed datapoint.
    run_roots: List[Path] = []
    gate_rows: List[dict] = []
    for seed in seeds2:
        full_tag = f"{args.tag}_seed{int(seed)}"
        chance = 1.0 / float(int(args.val_range))
        abort_after = int(args.soft_cap_steps) if int(args.soft_cap_steps) > 0 else 0
        abort_acc = float(chance) - float(args.soft_abort_margin) if abort_after > 0 else 0.0
        run_root, _ = _run_boot(
            repo_root=repo_root,
            tag=full_tag,
            seed=int(seed),
            steps=int(args.steps),
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            max_samples=int(max(args.max_samples, args.eval_samples)),
            eval_samples=int(args.eval_samples),
            ring_len=int(args.ring_len),
            slot_dim=int(args.slot_dim),
            device=str(args.device),
            lr=float(args.lr),
            model=str(args.model),
            prismn_n=int(args.prismn_n),
            prismn_mode=str(args.prismn_mode),
            prismn_shared=int(args.prismn_shared),
            prismn_out_dim=int(args.prismn_out_dim),
            prismn_id_scale=float(args.prismn_id_scale),
            synth_mode=str(args.synth_mode),
            keys=int(args.keys),
            pairs=int(args.pairs),
            val_range=int(args.val_range),
            assoc_unique_keys=bool(args.assoc_unique_keys),
            assoc_mq_dup=int(args.assoc_mq_dup),
            assoc_mq_grouped=int(args.assoc_mq_grouped),
            eval_disjoint=bool(args.eval_disjoint),
            eval_ptr_deterministic=bool(args.eval_ptr_deterministic),
            abort_after=abort_after,
            abort_acc=abort_acc,
        )
        print(f"[run] seed={seed} run_root={run_root}")
        run_roots.append(run_root)

        # If the run was aborted early, don't spend time on high-precision eval.
        r = _read_report(run_root)
        steps_done = int((r.get("train") or {}).get("steps") or 0)
        if steps_done >= int(args.steps):
            _postmortem_eval(
                repo_root=repo_root,
                run_root=run_root,
                checkpoint=_find_checkpoint(run_root),
                eval_samples=int(args.hard_eval_samples),
                batch_size=int(args.batch_size),
                device=str(args.device),
                prismn_id_scale=float(args.prismn_id_scale),
            )
        gate_rows.append(_gate_row(run_root=run_root))

    _ingest_and_gate(repo_root)
    print("[done] ingested + professor-gated results.")
    for rr in run_roots:
        print(f"[done] {rr}")
    print(f"[done] dashboard: http://localhost:{int(args.dashboard_port)}")

    # Print the binary decision for this 2-seed datapoint (hard gate).
    passes = [bool(gr.get("pass")) for gr in gate_rows]
    overall = "PASS" if all(passes) else ("UNSTABLE" if any(passes) else "FAIL")
    print(f"[gate] overall={overall} hard_steps={int(args.steps)} hard_eval_n={int(args.hard_eval_samples)}")
    for gr in gate_rows:
        print(
            "[gate] seed={seed} steps={steps} acc={acc:.6f} n={n} ci95={ci:.6f} lower95={lo:.6f} chance={ch:.6f} pass={ps}".format(
                seed=gr.get("seed"),
                steps=gr.get("steps"),
                acc=float(gr.get("eval_acc") or 0.0),
                n=int(gr.get("eval_n") or 0),
                ci=float(gr.get("ci95") or 0.0),
                lo=float(gr.get("lower95") or 0.0),
                ch=float(gr.get("chance") or 0.0),
                ps=bool(gr.get("pass")),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
