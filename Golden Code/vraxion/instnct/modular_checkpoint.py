"""Modular checkpoint + expert lifecycle utilities.

Behavior-preserving extraction from the legacy monolithic training script.

Modular checkpoint layout (stable contract):
  <base_dir>/
    system/router.state      # core model state + optimizer/scaler + training metadata
    experts/expert_###.pt    # per-expert tensors (sliced from head.experts.*)
    experts/meta.json        # lightweight expert lifecycle metadata

Assumptions about the model object (minimal):
  - `model.state_dict()` contains keys like `head.experts.<idx>.<param_name>`.
  - `model.head.experts` is an iterable of nn.Modules (optional, for load).
  - `model.head.num_experts` is an int (optional, for save).

Contract (do not break when refactoring):
  - Keep file layout stable.
  - `router.state` excludes `head.experts.*` tensors; those live in `experts/`.
  - Load is lenient: strict load may fail, then falls back to strict=False.
  - Expert loads are strict=False.
  - State hashing is deterministic (sorted keys).

Hardening (non-breaking):
  - Atomic writes for router.state and meta.json (temp file + os.replace).
  - Best-effort torch.load compatibility for `weights_only=` across torch versions.
  - Guardrails for corrupt expert shards / malformed meta.json (resume continues).

NOTE: This module is intentionally model-architecture agnostic; avoid pulling in
GRU / multi-circle sync logic here.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


# Defaults copied from legacy conventions; callers may override at runtime.
USE_AMP = False
UPDATE_SCALE = 0.0005
PTR_INERTIA = 0.0
AGC_SCALE_MAX = 1.0
EXPERT_HEADS = 1

# The loader uses a global DEVICE for torch.load map_location.
DEVICE: Any = "cpu"


def log(msg: str) -> None:
    """Minimal drop-in logger (legacy compatible)."""

    print(msg, flush=True)


def _coerce_int(valobj: Any, defval: int = 0) -> int:
    try:
        return int(valobj)
    except Exception:
        return int(defval)


def _coerce_flt(valobj: Any, defval: float = 0.0) -> float:
    try:
        return float(valobj)
    except Exception:
        return float(defval)


def _coerce_bol(valobj: Any, defval: bool = False) -> bool:
    try:
        if isinstance(valobj, str):
            lowval = valobj.strip().lower()
            if lowval in {"true", "1", "yes", "y", "t"}:
                return True
            if lowval in {"false", "0", "no", "n", "f"}:
                return False
        return bool(valobj)
    except Exception:
        return bool(defval)


def _torch_load_compat(path: str, *, map_location: Any = "cpu", weights_only: Optional[bool] = None) -> Any:
    """torch.load wrapper that is compatible across torch versions.

    Newer torch versions accept `weights_only=`. Older versions raise TypeError.

    If `weights_only=True` fails on a legacy payload, this function retries with
    `weights_only=False` when supported.
    """

    if weights_only is None:
        return torch.load(path, map_location=map_location)

    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)  # type: ignore[call-arg]
    except TypeError:
        # Older torch: no weights_only kwarg.
        return torch.load(path, map_location=map_location)
    except Exception:
        if weights_only:
            try:
                return torch.load(path, map_location=map_location, weights_only=False)  # type: ignore[call-arg]
            except TypeError:
                return torch.load(path, map_location=map_location)
        raise


def _atomic_torch_save(obj: Any, path: str) -> None:
    """Atomically torch.save(obj) -> path (temp file + os.replace)."""

    dirpth = os.path.dirname(path) or "."
    os.makedirs(dirpth, exist_ok=True)

    basnam = os.path.basename(path)
    fdpair = tempfile.mkstemp(prefix=f".tmp.{basnam}.", dir=dirpth)
    os.close(fdpair[0])
    tmppth = fdpair[1]

    try:
        with open(tmppth, "wb") as filobj:
            torch.save(obj, filobj)
            filobj.flush()
            try:
                os.fsync(filobj.fileno())
            except OSError:
                # Some filesystems/environments do not support fsync.
                pass

        os.replace(tmppth, path)
        tmppth = ""
    finally:
        if tmppth:
            try:
                os.remove(tmppth)
            except OSError:
                pass


def _atomic_json_dump(obj: Any, path: str, *, indent: int = 2) -> None:
    """Atomically json.dump(obj) -> path (temp file + os.replace)."""

    dirpth = os.path.dirname(path) or "."
    os.makedirs(dirpth, exist_ok=True)

    basnam = os.path.basename(path)
    fdpair = tempfile.mkstemp(prefix=f".tmp.{basnam}.", dir=dirpth)
    os.close(fdpair[0])
    tmppth = fdpair[1]

    try:
        with open(tmppth, "w", encoding="utf-8") as filobj:
            json.dump(obj, filobj, indent=indent)
            filobj.flush()
            try:
                os.fsync(filobj.fileno())
            except OSError:
                pass

        os.replace(tmppth, path)
        tmppth = ""
    finally:
        if tmppth:
            try:
                os.remove(tmppth)
            except OSError:
                pass


def _compute_expert_similarity_stats(model: Any, sim_thresh: float):
    head = getattr(model, "head", None)
    experts = getattr(head, "experts", None) if head is not None else None
    if not experts:
        return None

    with torch.no_grad():
        flat = []
        for expert in experts:
            vec = expert.weight.detach().float().reshape(-1)
            if expert.bias is not None:
                vec = torch.cat([vec, expert.bias.detach().float().reshape(-1)])
            flat.append(vec)

        mat = torch.stack(flat, dim=0)
        mat = F.normalize(mat, dim=1)
        sim = mat @ mat.T

        upper = torch.triu(sim, diagonal=1)
        if upper.numel() == 0:
            return None

        max_sim = float(upper.max().item())
        count = int((upper > sim_thresh).sum().item())
        flat_idx = int(torch.argmax(upper).item())
        i = flat_idx // upper.size(1)
        j = flat_idx % upper.size(1)

    return max_sim, count, (int(i), int(j))


def _resolve_hibernate_dir(base_dir: str, root_dir: str) -> str:
    path = base_dir
    if not os.path.isabs(path):
        path = os.path.join(root_dir, base_dir)
    os.makedirs(path, exist_ok=True)
    return path


def _extract_expert_state(head: Any, expert_id: int) -> Optional[Dict[str, Any]]:
    if head is None:
        return None
    experts = getattr(head, "experts", None)
    if experts is None or expert_id >= len(experts):
        return None

    state: Dict[str, Any] = {}
    with torch.no_grad():
        exp_state = experts[expert_id].state_dict()
        for name, tensor in exp_state.items():
            state[name] = tensor.detach().cpu()
    return state


def _zero_expert_weights(head: Any, expert_id: int) -> bool:
    if head is None:
        return False
    experts = getattr(head, "experts", None)
    if experts is None or expert_id >= len(experts):
        return False

    with torch.no_grad():
        for param in experts[expert_id].parameters():
            param.zero_()
    return True


def _hash_state_dict(state: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not state:
        return None

    hasher = hashlib.sha256()
    for key in sorted(state.keys()):
        tensor = state[key]
        hasher.update(key.encode("utf-8"))
        if torch.is_tensor(tensor):
            tenval = tensor.detach().contiguous().cpu()
            try:
                arr = tenval.numpy()
            except Exception:
                # Some dtypes (e.g. bfloat16 on some builds) cannot convert to NumPy.
                arr = tenval.to(dtype=torch.float32).numpy()
            hasher.update(arr.tobytes())
        else:
            hasher.update(repr(tensor).encode("utf-8"))
    return hasher.hexdigest()


def _save_expert_snapshot(state: Dict[str, Any], path: str) -> Optional[str]:
    _atomic_torch_save(state, path)
    return _hash_state_dict(state)


def _load_expert_snapshot(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not os.path.exists(path):
        return None, None

    try:
        state = _torch_load_compat(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        log(f"Expert snapshot load failed: {path} ({exc})")
        return None, None

    if not isinstance(state, dict):
        return None, None

    return state, _hash_state_dict(state)


def _resolve_modular_dir(base_dir: Optional[str], root_dir: str, fallback_path: Optional[str]) -> str:
    path = base_dir or ""
    if not path:
        path = fallback_path or ""
    if not path:
        path = os.path.join(root_dir, "checkpoints", "modular")

    if path.endswith(".pt"):
        path = os.path.splitext(path)[0] + "_modular"
    if not os.path.isabs(path):
        path = os.path.join(root_dir, path)

    os.makedirs(path, exist_ok=True)
    return path


def _modular_paths(base_dir: str) -> Tuple[str, str, str]:
    system_dir = os.path.join(base_dir, "system")
    experts_dir = os.path.join(base_dir, "experts")
    os.makedirs(system_dir, exist_ok=True)
    os.makedirs(experts_dir, exist_ok=True)

    router_path = os.path.join(system_dir, "router.state")
    meta_path = os.path.join(experts_dir, "meta.json")
    return router_path, experts_dir, meta_path


def _split_model_state_dict(state_dict: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    core: Dict[str, Any] = {}
    experts: Dict[int, Dict[str, Any]] = {}

    prefix = "head.experts."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            rest = key[len(prefix) :]
            parts = rest.split(".", 1)
            if len(parts) != 2:
                core[key] = value.detach().cpu() if torch.is_tensor(value) else value
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                core[key] = value.detach().cpu() if torch.is_tensor(value) else value
                continue
            expert_state = experts.setdefault(idx, {})
            expert_state[parts[1]] = value.detach().cpu() if torch.is_tensor(value) else value
        else:
            core[key] = value.detach().cpu() if torch.is_tensor(value) else value

    return core, experts


def _ensure_expert_tracking(model: Any, num_experts: int, step: int) -> None:
    contrib = getattr(model, "expert_contrib", None)
    last_used = getattr(model, "expert_last_used", None)
    created = getattr(model, "expert_created_step", None)
    tenured = getattr(model, "expert_tenured", None)

    if contrib is None or last_used is None or created is None or tenured is None:
        model.expert_contrib = [0.0 for _ in range(num_experts)]
        model.expert_last_used = [int(step) for _ in range(num_experts)]
        model.expert_created_step = [int(step) for _ in range(num_experts)]
        model.expert_tenured = [False for _ in range(num_experts)]
        return

    if len(contrib) < num_experts:
        missing = num_experts - len(contrib)
        model.expert_contrib.extend([0.0 for _ in range(missing)])
        model.expert_last_used.extend([int(step) for _ in range(missing)])
        model.expert_created_step.extend([int(step) for _ in range(missing)])
        model.expert_tenured.extend([False for _ in range(missing)])


def _update_expert_usage(model: Any, num_experts: int, step: int) -> None:
    counts = getattr(model, "ptr_expert_counts", None)
    if counts is None or getattr(counts, "numel", lambda: 0)() != num_experts:
        return

    _ensure_expert_tracking(model, num_experts, step)
    counts_list = counts.detach().cpu().tolist()
    for idx, count in enumerate(counts_list):
        if count <= 0:
            continue
        model.expert_last_used[idx] = int(step)
        model.expert_contrib[idx] += float(count)


def _build_expert_meta(model: Any, step: int, num_experts: int, contrib_thresh: float, probation_steps: int):
    _ensure_expert_tracking(model, num_experts, step)

    meta = []
    for idx in range(num_experts):
        contrib = float(model.expert_contrib[idx])
        created = int(model.expert_created_step[idx])
        last_used = int(model.expert_last_used[idx])
        age = max(0, int(step) - created)
        probation = age < probation_steps
        tenured = contrib >= contrib_thresh
        model.expert_tenured[idx] = bool(tenured)

        meta.append(
            {
                "id": idx,
                "contrib": contrib,
                "created_step": created,
                "last_used_step": last_used,
                "age_steps": age,
                "probation": bool(probation),
                "tenured": bool(tenured),
            }
        )

    return meta


def _load_modular_meta(model: Any, meta_path: str) -> None:
    if not os.path.exists(meta_path):
        return

    try:
        with open(meta_path, "r", encoding="utf-8") as filobj:
            data = json.load(filobj)
    except Exception as exc:
        log(f"Invalid modular meta.json (ignored): {meta_path} ({exc})")
        return

    if not isinstance(data, dict):
        return

    experts = data.get("experts", [])
    if not isinstance(experts, list) or not experts:
        return

    # Defensive parsing: invalid entries should not crash resume.
    model.expert_contrib = [_coerce_flt(item.get("contrib", 0.0)) if isinstance(item, dict) else 0.0 for item in experts]
    model.expert_last_used = [
        _coerce_int(item.get("last_used_step", 0)) if isinstance(item, dict) else 0 for item in experts
    ]
    model.expert_created_step = [
        _coerce_int(item.get("created_step", 0)) if isinstance(item, dict) else 0 for item in experts
    ]
    model.expert_tenured = [_coerce_bol(item.get("tenured", False)) if isinstance(item, dict) else False for item in experts]


def _save_modular_checkpoint(
    model: Any,
    optimizer: Any,
    scaler: Any,
    step: int,
    losses: Any,
    base_dir: str,
    contrib_thresh: float,
    probation_steps: int,
    ttl_steps: int = 0,
    gc_enabled: bool = False,
) -> None:
    router_path, experts_dir, meta_path = _modular_paths(base_dir)
    state_dict = model.state_dict()
    core_state, expert_states = _split_model_state_dict(state_dict)

    payload = {
        "model": core_state,
        "optim": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if USE_AMP and scaler is not None else None,
        "step": int(step),
        "losses": list(losses),
        "update_scale": getattr(model, "update_scale", UPDATE_SCALE),
        "ptr_inertia": getattr(model, "ptr_inertia", PTR_INERTIA),
        "ptr_inertia_ema": getattr(model, "ptr_inertia_ema", getattr(model, "ptr_inertia", PTR_INERTIA)),
        "ptr_inertia_floor": getattr(model, "ptr_inertia_floor", 0.0),
        "agc_scale_max": getattr(model, "agc_scale_max", AGC_SCALE_MAX),
        "ground_speed_ema": getattr(model, "ground_speed_ema", None),
        "ground_speed_limit": getattr(model, "ground_speed_limit", None),
        "num_experts": getattr(getattr(model, "head", None), "num_experts", EXPERT_HEADS),
        "param_names": [name for name, _ in model.named_parameters()],
    }

    _atomic_torch_save(payload, router_path)

    num_experts = int(payload["num_experts"])
    expert_meta = _build_expert_meta(model, step, num_experts, contrib_thresh, probation_steps)

    deleted = []
    for idx, state in expert_states.items():
        if idx >= num_experts:
            continue

        if gc_enabled and ttl_steps > 0:
            tenured = bool(model.expert_tenured[idx])
            idle = int(step) - int(model.expert_last_used[idx])
            age = int(step) - int(model.expert_created_step[idx])
            probation = age < probation_steps
            if not tenured and not probation and idle >= ttl_steps:
                deleted.append(idx)
                continue

        path = os.path.join(experts_dir, f"expert_{idx:03d}.pt")
        _atomic_torch_save(state, path)

    meta = {
        "num_experts": num_experts,
        "step": int(step),
        "experts": expert_meta,
        "deleted": deleted,
    }
    _atomic_json_dump(meta, meta_path, indent=2)


def _load_modular_checkpoint(model: Any, optimizer: Any, scaler: Any, base_dir: str) -> Dict[str, Any]:
    router_path, experts_dir, meta_path = _modular_paths(base_dir)
    ckpt = _torch_load_compat(router_path, map_location=DEVICE, weights_only=False)

    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        log(f"Modular load strict failed: {exc}; retrying strict=False")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            log(f"Modular load missing keys: {missing}")
        if unexpected:
            log(f"Modular load unexpected keys: {unexpected}")

    head = getattr(model, "head", None)
    if head is not None and getattr(head, "experts", None):
        for idx, expert in enumerate(head.experts):
            path = os.path.join(experts_dir, f"expert_{idx:03d}.pt")
            if not os.path.exists(path):
                continue
            try:
                state = _torch_load_compat(path, map_location="cpu", weights_only=True)
            except Exception as exc:
                log(f"Expert load failed (ignored): {path} ({exc})")
                continue
            if not isinstance(state, dict):
                log(f"Expert snapshot is not a dict (ignored): {path}")
                continue
            try:
                expert.load_state_dict(state, strict=False)
            except Exception as exc:
                log(f"Expert load_state_dict failed (ignored): {path} ({exc})")

    if ckpt.get("optim") is not None and optimizer is not None:
        optimizer.load_state_dict(ckpt["optim"])
    if ckpt.get("scaler") is not None and USE_AMP and scaler is not None:
        scaler.load_state_dict(ckpt["scaler"])

    _load_modular_meta(model, meta_path)
    return ckpt


def _resolve_modular_resume_dir(resume_path: Optional[str]) -> Optional[str]:
    if resume_path and os.path.isdir(resume_path):
        router_path = os.path.join(resume_path, "system", "router.state")
        if os.path.exists(router_path):
            return resume_path

    if resume_path and resume_path.endswith(".pt"):
        candidate = os.path.splitext(resume_path)[0] + "_modular"
        router_path = os.path.join(candidate, "system", "router.state")
        if os.path.exists(router_path):
            return candidate

    return None
