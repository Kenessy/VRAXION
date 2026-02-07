"""Postmortem eval for synth assoc_byte runs (load checkpoint, run eval, write report.json).

Problem this solves:
- Training runs are often stopped early (manual stop at step ~200, etc.).
- The boot benchmark writes report.json only when the run completes normally.
- We need a deterministic way to evaluate an EXISTING checkpoint and persist results
  into the SAME run_root so `tools/results_ingest.py` can pick it up.

Usage:
  python tools/eval_ckpt_assoc_byte.py --run-root S:\\...\\bench_vault\\benchmarks\\...\\<ts>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def _bootstrap_import_paths(repo_root: Path) -> None:
    draftr = repo_root / "Golden Draft"
    gcode = repo_root / "Golden Code"
    if str(draftr) not in sys.path:
        sys.path.insert(0, str(draftr))
    if str(gcode) not in sys.path:
        sys.path.insert(0, str(gcode))


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


def _env_set(name: str, value: str) -> None:
    os.environ[name] = value


def _read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    with path.open("rb") as f:
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace")


def _infer_seed_from_path(run_root: Path) -> Optional[int]:
    # Common run tags include "...seed222_..." somewhere in the path.
    m = re.search(r"\bseed(\d+)\b", str(run_root).replace("\\", "/"))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _infer_synth_from_log(log_txt: str) -> Dict[str, Any]:
    # Example:
    # [synth] mode=assoc_byte rows=2000 keys=64 vals=16 pairs=4 mq_dup=1 len=128
    m = re.search(
        r"\[synth\]\s+mode=(?P<mode>\S+)\s+rows=(?P<rows>\d+)\s+keys=(?P<keys>\d+)\s+vals=(?P<vals>\d+)\s+pairs=(?P<pairs>\d+)(?:\s+mq_dup=(?P<mq_dup>\d+))?\s+len=(?P<seq_len>\d+)",
        log_txt,
    )
    if not m:
        raise RuntimeError("Could not infer synth settings from vraxion.log (missing [synth] header).")
    out: Dict[str, Any] = dict(m.groupdict())
    for k in ("rows", "keys", "vals", "pairs", "seq_len"):
        out[k] = int(out[k])
    out["mq_dup"] = int(out["mq_dup"]) if out.get("mq_dup") is not None else 1
    return out


def _infer_eval_disjoint_from_log(log_txt: str) -> bool:
    return bool(re.search(r"\[eval\]\s+split=disjoint\b", log_txt))


def _infer_absolute_hallway_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Infer AbsoluteHallway ctor args from a checkpoint state_dict."""

    router = state.get("router_map")
    if not torch.is_tensor(router):
        raise RuntimeError("AbsoluteHallway checkpoint missing router_map")
    ring_len = int(router.numel())

    w_in = state.get("input_proj.weight")
    if not torch.is_tensor(w_in) or w_in.dim() != 2:
        raise RuntimeError("AbsoluteHallway checkpoint missing input_proj.weight")
    slot_dim = int(w_in.shape[0])

    # head: LocationExpertRouter
    num_classes = None
    expert_heads = None

    w_single = state.get("head.single.weight")
    if torch.is_tensor(w_single) and w_single.dim() == 2:
        num_classes = int(w_single.shape[0])
        expert_heads = 1
    else:
        max_idx = -1
        w0 = None
        for k in state.keys():
            if not isinstance(k, str) or not k.startswith("head.experts."):
                continue
            parts = k.split(".")
            if len(parts) < 4:
                continue
            try:
                idx = int(parts[2])
            except Exception:
                continue
            max_idx = max(max_idx, idx)
            if idx == 0 and k.endswith(".weight"):
                w0 = state.get(k)
        if torch.is_tensor(w0) and w0 is not None and w0.dim() == 2:
            num_classes = int(w0.shape[0])
            expert_heads = max_idx + 1

    if num_classes is None or expert_heads is None:
        raise RuntimeError("AbsoluteHallway checkpoint missing head weights (single or experts.*)")

    return {
        "ring_len": int(ring_len),
        "slot_dim": int(slot_dim),
        "num_classes": int(num_classes),
        "expert_heads": int(expert_heads),
    }


def _infer_prismion_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    # slot_dim via core.input_proj.weight: [slot_dim, input_dim]
    slot_dim = int(state["core.input_proj.weight"].shape[0])

    # ring_len via per-ring vectors.
    ring_len = None
    for k in ("core.theta_ptr_reduced", "core.theta_gate_reduced", "core.router_map"):
        v = state.get(k)
        if v is not None and hasattr(v, "shape") and len(v.shape) == 1:
            ring_len = int(v.shape[0])
            break
    if ring_len is None:
        raise RuntimeError("Could not infer ring_len from checkpoint tensors.")

    # msg projection: shared msg_proj vs per-prismion msg_w/msg_b.
    shared_msgproj = True
    n = None
    out_dim = None
    if "msg_w" in state:
        shared_msgproj = False
        n = int(state["msg_w"].shape[0])
        out_dim = int(state["msg_w"].shape[1])
    elif "msg_proj.weight" in state:
        shared_msgproj = True
        out_dim = int(state["msg_proj.weight"].shape[0])
        n = int(state["id_embed.weight"].shape[0]) if "id_embed.weight" in state else None
    if n is None or out_dim is None:
        raise RuntimeError("Could not infer prismion N/out_dim from checkpoint tensors.")

    # head: [num_classes, head_in]
    num_classes = int(state["head.weight"].shape[0])
    head_in = int(state["head.weight"].shape[1])
    if head_in == n * out_dim:
        mode = "mosaic"
    elif head_in == out_dim:
        mode = "mean"
    else:
        mode = f"unknown(head_in={head_in})"

    return {
        "ring_len": ring_len,
        "slot_dim": slot_dim,
        "num_classes": num_classes,
        "prismn_n": n,
        "prismn_out_dim": out_dim,
        "prismn_shared_msgproj": 1 if shared_msgproj else 0,
        "prismn_mode": mode,
    }


def _loss_slope(losses: list[float], window: int = 50) -> float:
    if not losses:
        return 0.0
    y = losses[-int(min(window, len(losses))) :]
    if len(y) < 2:
        return 0.0
    n = float(len(y))
    sx = (n - 1.0) * n / 2.0
    sxx = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0
    sy = float(sum(y))
    sxy = float(sum(i * yi for i, yi in enumerate(y)))
    denom = (n * sxx - sx * sx)
    if denom == 0.0:
        return 0.0
    return (n * sxy - sx * sy) / denom


def _last_debug_from_log(log_txt: str) -> Dict[str, Any]:
    last = None
    for line in log_txt.splitlines():
        if "| debug " in line:
            last = line
    if not last:
        return {}
    try:
        jtxt = last.split("| debug ", 1)[1].strip()
        return json.loads(jtxt)
    except Exception:
        return {}


def _count_params(model: Any) -> Dict[str, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if bool(getattr(p, "requires_grad", False)):
            trainable += n
    return {"total": int(total), "trainable": int(trainable)}


def _count_module_params(mod: Any) -> int:
    if mod is None:
        return 0
    try:
        return int(sum(int(p.numel()) for p in mod.parameters()))
    except Exception:
        return 0


def _build_absolute_hallway(shape: Dict[str, Any]):
    # Keep global expert-head topology aligned with checkpoint shape before
    # constructing the model, or strict state_dict load can mismatch.
    import vraxion.instnct.absolute_hallway as ah  # type: ignore

    ah.EXPERT_HEADS = int(shape["expert_heads"])
    return ah.AbsoluteHallway(
        input_dim=1,
        num_classes=int(shape["num_classes"]),
        ring_len=int(shape["ring_len"]),
        slot_dim=int(shape["slot_dim"]),
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--eval-samples", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-seed-offset", type=int, default=1000003)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--prismn-id-scale", type=float, default=0.02, help="Must match training if it was overridden.")
    p.add_argument(
        "--force-eval-disjoint",
        action="store_true",
        help="Force disjoint eval split (ignore any inference from vraxion.log).",
    )
    p.add_argument(
        "--force-eval-subset",
        action="store_true",
        help="Force subset eval split (ignore any inference from vraxion.log).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    run_root = Path(args.run_root).resolve()
    if not run_root.exists():
        raise SystemExit(f"run_root not found: {run_root}")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (run_root / "checkpoint_last_good.pt")
    if not ckpt_path.exists():
        ckpt_path = run_root / "checkpoint.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    log_path = run_root / "vraxion.log"
    if not log_path.exists():
        raise SystemExit(f"vraxion.log not found: {log_path}")

    repo_root = Path(__file__).resolve().parents[2]
    _bootstrap_import_paths(repo_root)

    from tools._checkpoint_io import atomic_json_dump, safe_torch_load  # type: ignore
    from tools import instnct_data, instnct_eval, instnct_train_wallclock  # type: ignore
    from vraxion.instnct import infra  # type: ignore
    from vraxion.instnct.seed import set_seed  # type: ignore

    log_txt = _read_text(log_path)
    synth = _infer_synth_from_log(log_txt)
    eval_disjoint_infer = _infer_eval_disjoint_from_log(log_txt)
    if bool(args.force_eval_disjoint) and bool(args.force_eval_subset):
        raise SystemExit("cannot pass both --force-eval-disjoint and --force-eval-subset")
    if bool(args.force_eval_disjoint):
        eval_disjoint = True
        eval_split_source = "forced"
    elif bool(args.force_eval_subset):
        eval_disjoint = False
        eval_split_source = "forced"
    else:
        eval_disjoint = bool(eval_disjoint_infer)
        eval_split_source = "inferred"

    ck = safe_torch_load(str(ckpt_path))
    state = ck.get("model") or {}
    shape: Dict[str, Any]
    model_kind: str
    if "router_map" in state:
        model_kind = "absolute_hallway"
        shape = _infer_absolute_hallway_from_state(state)
    else:
        model_kind = "prismion_hallway_bank"
        shape = _infer_prismion_from_state(state)

    seed = _infer_seed_from_path(run_root)
    if seed is None:
        seed = int(ck.get("step") or 0)

    # Route all eval logs into the run_root for traceability.
    _env_set("VAR_PROJECT_ROOT", str(run_root))
    _env_set("VAR_LOGGING_PATH", str(run_root / "vraxion_eval.log"))
    _env_set("VAR_COMPUTE_DEVICE", str(args.device))
    _env_set("VRX_PRECISION", "fp32")
    _env_set("VAR_RUN_SEED", str(int(seed)))

    _env_set("VRX_RING_LEN", str(int(shape["ring_len"])))
    _env_set("VRX_SLOT_DIM", str(int(shape["slot_dim"])))

    _env_set("VRX_SYNTH", "1")
    _env_set("VRX_SYNTH_MODE", str(synth["mode"]))
    _env_set("VRX_SYNTH_LEN", str(int(synth["seq_len"])))
    _env_set("VRX_ASSOC_KEYS", str(int(synth["keys"])))
    _env_set("VRX_ASSOC_PAIRS", str(int(synth["pairs"])))
    _env_set("VRX_ASSOC_VAL_RANGE", str(int(synth["vals"])))
    _env_set("VRX_ASSOC_MQ_DUP", str(int(synth.get("mq_dup", 1))))

    # Ensure the dataset is large enough for the requested eval sample count.
    _env_set("VRX_MAX_SAMPLES", str(int(max(int(args.eval_samples), int(synth["rows"])))))
    _env_set("VRX_EVAL_SAMPLES", str(int(args.eval_samples)))
    _env_set("VRX_BATCH_SIZE", str(int(args.batch_size)))
    _env_set("VRX_LR", "0.0")

    if model_kind == "absolute_hallway":
        _env_set("VRX_EXPERT_HEADS", str(int(shape["expert_heads"])))
    else:
        _env_set("VRX_MODEL", "prismion_hallway_bank")
        _env_set("VRX_PRISMN_N", str(int(shape["prismn_n"])))
        _env_set("VRX_PRISMN_MODE", str(shape["prismn_mode"]))
        _env_set("VRX_PRISMN_SHARED", str(int(shape["prismn_shared_msgproj"])))
        _env_set("VRX_PRISMN_OUT_DIM", str(int(shape["prismn_out_dim"])))
        _env_set("VRX_PRISMN_OUT_DTYPE", "fp64")
        _env_set("VRX_PRISMN_ID_SCALE", str(float(args.prismn_id_scale)))

    _env_set("VRX_EVAL_DISJOINT", "1" if eval_disjoint else "0")
    _env_set("VRX_EVAL_SEED_OFFSET", str(int(args.eval_seed_offset)))

    infra.ROOT = str(run_root)
    infra.LOG_PATH = str(run_root / "vraxion_eval.log")
    set_seed(int(seed))

    loader, num_classes, collate = instnct_data.get_seq_mnist_loader(
        train=True,
        batch_size=int(args.batch_size),
        max_samples=int(os.environ["VRX_MAX_SAMPLES"]),
    )
    if int(num_classes) != int(shape["num_classes"]):
        infra.log(
            f"[eval_ckpt] WARNING: num_classes mismatch ckpt={shape['num_classes']} loader={num_classes}"
        )

    spec = instnct_eval.EvalLoaderSpec(eval_samples=int(args.eval_samples), batch_size=int(args.batch_size))
    if eval_disjoint:
        set_seed(int(seed) + int(args.eval_seed_offset))
        eval_src, _, eval_collate = instnct_data.get_seq_mnist_loader(
            train=True,
            batch_size=int(args.batch_size),
            max_samples=int(os.environ["VRX_MAX_SAMPLES"]),
        )
        set_seed(int(seed))
        eval_loader, eval_size = instnct_eval.build_eval_loader_from_dataset(
            eval_src.dataset, spec=spec, input_collate=eval_collate
        )
        instnct_eval.log_eval_overlap(loader.dataset, eval_loader.dataset, eval_size, "disjoint", log=infra.log)
    else:
        eval_loader, eval_size = instnct_eval.build_eval_loader_from_subset(
            loader.dataset, spec=spec, input_collate=collate
        )
        instnct_eval.log_eval_overlap(loader.dataset, eval_loader.dataset, eval_size, "subset", log=infra.log)

    if model_kind == "absolute_hallway":
        model = _build_absolute_hallway(shape)
        model.load_state_dict(state, strict=True)
    else:
        # Legacy model support (optional): require external Golden Code to provide it.
        from vraxion.instnct.prismion_hallway_bank import PrismionHallwayBank  # type: ignore

        model = PrismionHallwayBank(
            input_dim=1,
            num_classes=int(shape["num_classes"]),
            ring_len=int(shape["ring_len"]),
            slot_dim=int(shape["slot_dim"]),
        )
        model.load_state_dict(state, strict=True)

    params = _count_params(model)
    params["head"] = _count_module_params(getattr(model, "head", None))
    params["core"] = _count_module_params(getattr(model, "core", None))

    eval_deps = instnct_eval.EvalDeps(
        device=str(args.device),
        dtype=torch.float32,
        amp_autocast=instnct_train_wallclock.amp_autocast,
        log=infra.log,
        synth_mode=str(synth["mode"]),
        mi_shuffle=False,
        mitosis_enabled=False,
    )
    eval_sum = instnct_eval.eval_model(
        model, eval_loader, "synth_assoc_byte", str(model_kind), deps=eval_deps
    )

    losses = [float(x) for x in (ck.get("losses") or [])] if isinstance(ck.get("losses"), list) else []
    debug_last = _last_debug_from_log(log_txt)

    report: Dict[str, Any] = {
        "benchmark": "eval_ckpt_assoc_byte",
        "run_root": str(run_root),
        "git_rev": _try_git_rev(repo_root),
        "platform": {"python": sys.version.split()[0], "os": platform.platform()},
        "settings": {
            "device": str(args.device),
            "dtype": "fp32",
            "seed": int(seed),
            "model": str(model_kind),
            "ring_len": int(shape["ring_len"]),
            "slot_dim": int(shape["slot_dim"]),
            "batch_size": int(args.batch_size),
            "seq_len": int(synth["seq_len"]),
            "max_samples": int(os.environ["VRX_MAX_SAMPLES"]),
            "eval_samples": int(args.eval_samples),
            "lr": 0.0,
            "keys": int(synth["keys"]),
            "pairs": int(synth["pairs"]),
            "val_range": int(synth["vals"]),
            "assoc_unique_keys": True,  # not inferable from logs; safe default for our current regime
            "assoc_mq_dup": int(synth.get("mq_dup", 1)),
            "assoc_mq_grouped": 1,
            "eval_disjoint": bool(eval_disjoint),
            "eval_split_source": str(eval_split_source),
            "eval_seed_offset": int(args.eval_seed_offset),
            "eval_ptr_deterministic": False,
            "abort_after": 0,
            "abort_acc": 0.0,
        },
        "model_shape": dict(shape),
        "params": params,
        "train": {
            "steps": int(ck.get("step") or 0),
            "loss_slope": float(_loss_slope(losses, window=50)),
            **{k: debug_last.get(k) for k in debug_last.keys() if k.startswith("lane_")},
        },
        "eval_mid": None,
        "eval": eval_sum,
        "notes": "postmortem eval from checkpoint (no additional training)",
    }

    atomic_json_dump(report, str(run_root / "report.json"), indent=2)
    infra.log(f"[eval_ckpt] report saved: {run_root / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
