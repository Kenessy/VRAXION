"""Boot benchmark: synthetic Markov0 sequence classification.

This is a small, deterministic probe intended to answer one question:
"Does the VRAXION goldenized training stack boot and learn anything at all?"

It deliberately avoids external datasets (torchvision downloads, etc.) by using
the existing synthetic mode in tools.instnct_data.

Outputs:
- Creates a run directory under `bench_vault/` (git-ignored)
- Writes `report.json` (settings snapshot + train/eval summaries)
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _set_env(name: str, value: str, *, respect_existing: bool) -> None:
    if respect_existing and os.environ.get(name) is not None:
        return
    os.environ[name] = value


def _try_git_rev(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def _bootstrap_import_paths(repo_root: Path) -> None:
    draftr = repo_root / "Golden Draft"
    gcode = repo_root / "Golden Code"
    if str(draftr) not in sys.path:
        sys.path.insert(0, str(draftr))
    if str(gcode) not in sys.path:
        sys.path.insert(0, str(gcode))


def _run_dir(repo_root: Path, tag: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return repo_root / "bench_vault" / "benchmarks" / tag / ts


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=512)
    p.add_argument("--ring-len", type=int, default=256)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tag", type=str, default="boot_synth_markov0")
    p.add_argument(
        "--respect-env",
        action="store_true",
        help="Do not override existing VRX_/VAR_ env vars inside this process.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    repo_root = Path(__file__).resolve().parents[3]
    run_root = _run_dir(repo_root, args.tag)
    run_root.mkdir(parents=True, exist_ok=True)

    # Route all runner artifacts into a unique, git-ignored directory.
    _set_env("VAR_PROJECT_ROOT", str(run_root), respect_existing=bool(args.respect_env))
    _set_env("VAR_LOGGING_PATH", str(run_root / "vraxion.log"), respect_existing=bool(args.respect_env))
    _set_env("VAR_COMPUTE_DEVICE", str(args.device), respect_existing=bool(args.respect_env))
    _set_env("VRX_PRECISION", "fp32", respect_existing=bool(args.respect_env))

    # Make the model small enough for fast CPU probes (unless already overridden).
    _set_env("VRX_RING_LEN", str(int(args.ring_len)), respect_existing=bool(args.respect_env))
    _set_env("VRX_SLOT_DIM", str(int(args.slot_dim)), respect_existing=bool(args.respect_env))
    _set_env("VRX_PTR_STRIDE", "1", respect_existing=bool(args.respect_env))

    # Synthetic dataset: Markov0 (predict last token).
    _set_env("VRX_SYNTH", "1", respect_existing=bool(args.respect_env))
    _set_env("VRX_SYNTH_MODE", "markov0", respect_existing=bool(args.respect_env))
    _set_env("VRX_SYNTH_LEN", str(int(args.seq_len)), respect_existing=bool(args.respect_env))
    _set_env("VRX_SYNTH_SHUFFLE", "1", respect_existing=bool(args.respect_env))

    _set_env("VRX_BATCH_SIZE", str(int(args.batch_size)), respect_existing=bool(args.respect_env))
    _set_env("VRX_MAX_SAMPLES", str(int(args.max_samples)), respect_existing=bool(args.respect_env))
    _set_env("VRX_EVAL_SAMPLES", str(int(args.eval_samples)), respect_existing=bool(args.respect_env))
    _set_env("VRX_LR", str(float(args.lr)), respect_existing=bool(args.respect_env))

    _bootstrap_import_paths(repo_root)

    from tools._checkpoint_io import atomic_json_dump  # type: ignore
    from tools import instnct_data, instnct_eval, instnct_train_steps, instnct_train_wallclock  # type: ignore
    from vraxion.settings import load_settings  # type: ignore
    from vraxion.instnct import infra  # type: ignore
    from vraxion.instnct.absolute_hallway import AbsoluteHallway  # type: ignore
    from vraxion.instnct.seed import set_seed  # type: ignore

    cfg = load_settings()
    infra.ROOT = str(cfg.root)
    infra.LOG_PATH = str(cfg.log_path)
    set_seed(int(cfg.seed))

    loader, num_classes, collate = instnct_data.get_seq_mnist_loader(
        train=True,
        batch_size=int(args.batch_size),
        max_samples=int(args.max_samples),
    )
    spec = instnct_eval.EvalLoaderSpec(eval_samples=int(args.eval_samples), batch_size=int(args.batch_size))
    eval_loader, eval_size = instnct_eval.build_eval_loader_from_subset(loader.dataset, spec=spec, input_collate=collate)
    instnct_eval.log_eval_overlap(loader.dataset, eval_loader.dataset, eval_size, "subset", log=infra.log)

    model = AbsoluteHallway(
        input_dim=1,
        num_classes=int(num_classes),
        ring_len=int(cfg.ring_len),
        slot_dim=int(cfg.slot_dim),
    )

    train_sum = instnct_train_steps.train_steps(model, loader, int(args.steps), "synth_markov0", "absolute_hallway")

    eval_deps = instnct_eval.EvalDeps(
        device=str(cfg.device),
        dtype=cfg.dtype,
        amp_autocast=instnct_train_wallclock.amp_autocast,
        log=infra.log,
        synth_mode=str(getattr(cfg, "synth_mode", "")),
        mi_shuffle=bool(getattr(cfg, "mi_shuffle", False)),
        mitosis_enabled=bool(getattr(cfg, "mitosis_enabled", False)),
    )
    eval_sum = instnct_eval.eval_model(model, eval_loader, "synth_markov0", "absolute_hallway", deps=eval_deps)

    report: Dict[str, Any] = {
        "benchmark": "boot_synth_markov0",
        "run_root": str(run_root),
        "git_rev": _try_git_rev(repo_root),
        "platform": {
            "python": sys.version.split()[0],
            "os": platform.platform(),
        },
        "settings": {
            "device": str(cfg.device),
            "dtype": str(cfg.dtype),
            "seed": int(cfg.seed),
            "ring_len": int(cfg.ring_len),
            "slot_dim": int(cfg.slot_dim),
            "batch_size": int(args.batch_size),
            "seq_len": int(args.seq_len),
            "max_samples": int(args.max_samples),
            "eval_samples": int(args.eval_samples),
            "lr": float(args.lr),
        },
        "train": train_sum,
        "eval": eval_sum,
    }

    atomic_json_dump(report, str(run_root / "report.json"), indent=2)
    infra.log(f"[bench] report saved: {run_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
