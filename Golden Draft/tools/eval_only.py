#!/usr/bin/env python
"""Evaluation-only runner for VRAXION checkpoints.

This tool is "Golden Draft" internal tooling.

Primary goal: load a checkpoint (monolithic .pt or modular dir) and run the
same evaluation loop used by the Golden Draft runner, without starting a full
training loop.

Usage examples:
  python -m tools.eval_only --checkpoint checkpoints/checkpoint.pt
  python -m tools.eval_only --checkpoint checkpoints/checkpoint.pt --dataset seq_mnist

Return codes (stable):
- 3: checkpoint path missing
- 4: missing required dependency (torch / vraxion)
- 5: runtime error
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager, NoReturn, Optional, Sequence, Tuple


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _reporoot() -> Path:
    """Return the repo root (parent of ``tools/``)."""

    return Path(__file__).resolve().parents[1]


def _bootstrap_paths(repdir: Path) -> None:
    """Ensure Golden Draft + Golden Code are importable for standalone runs."""

    draft_root = str(repdir)
    if draft_root not in sys.path:
        sys.path.insert(0, draft_root)

    candls: list[str] = []
    for keystr in ("VRAXION_GOLDEN_SRC", "GOLDEN_CODE_ROOT", "GOLDEN_CODE_PATH", "GOLDEN_CODE_DIR"):
        envval = os.environ.get(keystr)
        if envval:
            candls.append(envval)

    candls.append(str(repdir.parent / "Golden Code"))
    candls.append(r"S:\AI\Golden Code")
    candls.append(r"S:/AI/Golden Code")

    for candpt in candls:
        try:
            if candpt and os.path.isdir(candpt):
                if candpt not in sys.path:
                    sys.path.insert(0, candpt)
                break
        except OSError:
            continue


def _reexec(repdir: Path) -> NoReturn:
    """Re-exec this tool as a module with repo root on PYTHONPATH."""

    envmap = os.environ.copy()
    envmap["_VRAXION_EVAL_ONLY_REEXEC"] = "1"

    oldpth = envmap.get("PYTHONPATH", "")
    repstr = str(repdir)
    envmap["PYTHONPATH"] = repstr if not oldpth else repstr + os.pathsep + oldpth

    cmdlst = [sys.executable, "-m", "tools.eval_only", *sys.argv[1:]]
    os.execvpe(cmdlst[0], cmdlst, envmap)
    raise RuntimeError("exec failed")


def _parse(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation for a VRAXION checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt (or modular dir)")
    parser.add_argument("--dataset", default="seq_mnist", help="Dataset name (currently: seq_mnist)")
    parser.add_argument("--model-name", default="absolute_hallway", help="Model name tag for logging")
    parser.add_argument("--debug", action="store_true", help="Print tracebacks on error")
    return parser.parse_args(list(argv) if argv is not None else None)


def _amp_autocast_from_cfg(torch_mod: Any, *, device: str, use_amp: bool, dtype: Any) -> callable:
    """Return a function that yields an autocast context manager (best-effort)."""

    if not use_amp:
        return lambda: nullcontext()

    if str(device).lower() != "cuda":
        return lambda: nullcontext()

    def _ctx() -> ContextManager[Any]:
        try:
            return torch_mod.amp.autocast(device_type="cuda", dtype=dtype)  # type: ignore[attr-defined]
        except Exception:
            try:
                return torch_mod.cuda.amp.autocast(dtype=dtype)
            except Exception:
                return nullcontext()

    return _ctx


def main(argv: Optional[Sequence[str]] = None) -> int:
    argobj = _parse(argv)

    ckppth = Path(os.path.expanduser(argobj.checkpoint)).resolve()
    if not ckppth.exists():
        _eprint(f"[eval_only] ERROR: checkpoint not found: {ckppth}")
        return 3

    repdir = _reporoot()

    # If run as a file, re-exec as a module so relative imports behave.
    if os.environ.get("_VRAXION_EVAL_ONLY_REEXEC") != "1" and (__package__ is None or __package__ == ""):
        _reexec(repdir)

    _bootstrap_paths(repdir)

    try:
        import torch
        from vraxion.instnct.absolute_hallway import AbsoluteHallway
        from vraxion.instnct import infra
        from vraxion.instnct import modular_checkpoint as mckpt
        from vraxion.settings import load_settings

        from tools import instnct_data, instnct_eval
    except Exception as exc:
        _eprint(f"[eval_only] ERROR: missing required dependency: {exc}")
        if argobj.debug:
            traceback.print_exc()
        return 4

    # Seed for evaluation determinism (mirrors typical tooling).
    try:
        torch.manual_seed(1337)
    except Exception:
        pass

    try:
        cfg = load_settings()

        # Keep infra in sync with settings-driven paths.
        infra.ROOT = str(cfg.root)
        infra.LOG_PATH = str(cfg.log_path)

        # Build a deterministic eval loader. For now we always use a subset
        # of the training dataset (no torchvision requirement).
        train_loader, num_classes, collate = instnct_data.get_seq_mnist_loader(train=True)

        spec = instnct_eval.EvalLoaderSpec(eval_samples=int(cfg.eval_samples), batch_size=int(cfg.batch_size))
        eval_loader, eval_size = instnct_eval.build_eval_loader_from_subset(
            train_loader.dataset, spec=spec, input_collate=collate
        )
        instnct_eval.log_eval_overlap(train_loader.dataset, eval_loader.dataset, eval_size, "train_subset", log=infra.log)

        xb, _ = next(iter(eval_loader))
        if not hasattr(xb, "shape") or len(xb.shape) < 2:
            raise RuntimeError(f"unexpected batch shape: {getattr(xb, 'shape', None)}")
        input_dim = int(xb.shape[-1])

        model = AbsoluteHallway(
            input_dim=input_dim,
            num_classes=int(num_classes),
            ring_len=int(cfg.ring_len),
            slot_dim=int(cfg.slot_dim),
        )
    except Exception as exc:
        _eprint(f"[eval_only] ERROR: failed to initialize eval: {exc}")
        if argobj.debug:
            traceback.print_exc()
        return 5

    # Resolve modular dir (if any) and load checkpoint.
    try:
        mckpt.DEVICE = str(cfg.device)

        moddir = mckpt._resolve_modular_resume_dir(str(ckppth))
        if moddir:
            ckpt = mckpt._load_modular_checkpoint(model, optimizer=None, scaler=None, base_dir=str(moddir))
        else:
            ckpt = torch.load(str(ckppth), map_location=str(cfg.device))

        state: Any = ckpt
        if isinstance(ckpt, dict):
            state = ckpt.get("model", ckpt)

        if isinstance(state, dict) and moddir is None:
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                _eprint(f"[eval_only] Missing keys: {missing}")
            if unexpected:
                _eprint(f"[eval_only] Unexpected keys: {unexpected}")

        # Load checkpoint overrides (best-effort, behavior-compatible).
        if isinstance(ckpt, dict):
            for key in (
                "update_scale",
                "ptr_inertia",
                "ptr_inertia_ema",
                "ptr_inertia_floor",
                "agc_scale_max",
                "ground_speed_ema",
                "ground_speed_limit",
                "ground_speed",
            ):
                if key in ckpt:
                    try:
                        setattr(model, key, ckpt[key])
                    except Exception:
                        pass
    except Exception as exc:
        _eprint(f"[eval_only] ERROR: failed to load checkpoint: {exc}")
        if argobj.debug:
            traceback.print_exc()
        return 5

    # Env-var overrides match runtime knobs.
    envini = os.environ.get("VRX_SCALE_INIT")
    envmax = os.environ.get("VRX_SCALE_MAX")
    if envini is not None:
        try:
            model.update_scale = float(envini)
        except Exception:
            pass
    if envmax is not None:
        try:
            model.agc_scale_max = float(envmax)
        except Exception:
            pass

    envptr = os.environ.get("VRX_PTR_INERTIA_OVERRIDE")
    if envptr is not None:
        try:
            model.ptr_inertia = float(envptr)
            model.ptr_inertia_ema = model.ptr_inertia
        except Exception:
            pass

    # Hard resets used by legacy scripts.
    try:
        model.agc_scale_cap = getattr(model, "agc_scale_max", getattr(model, "agc_scale_cap", None))
        model.ground_speed_ema = None
        model.ground_speed_limit = None
        model.ground_speed = None
        model.debug_scale_out = getattr(model, "update_scale", getattr(model, "debug_scale_out", None))
    except Exception:
        pass

    # Run evaluation.
    try:
        deps = instnct_eval.EvalDeps(
            device=str(cfg.device),
            dtype=cfg.dtype,
            amp_autocast=_amp_autocast_from_cfg(torch, device=str(cfg.device), use_amp=bool(cfg.use_amp), dtype=cfg.dtype),
            log=infra.log,
            synth_mode=str(getattr(cfg, "synth_mode", "")),
            mi_shuffle=bool(getattr(cfg, "mi_shuffle", False)),
            mitosis_enabled=bool(getattr(cfg, "mitosis_enabled", False)),
        )

        model.eval()
        with torch.no_grad():
            stats = instnct_eval.eval_model(model, eval_loader, str(argobj.dataset), str(argobj.model_name), deps=deps)
    except Exception as exc:
        _eprint(f"[eval_only] ERROR: eval failed: {exc}")
        if argobj.debug:
            traceback.print_exc()
        return 5

    print(f"[eval_only] stats: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

