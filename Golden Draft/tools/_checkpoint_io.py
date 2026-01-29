"""Shared helpers for Golden Draft checkpoint CLI tools.

These helpers intentionally live outside the end-user runtime package.

Design goals:
- Best-effort safer ``torch.load`` defaults (CPU + ``weights_only`` when available).
- Atomic writes (temp file + ``os.replace``) to avoid partially-written outputs.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch


# Expert state_dict key prefix convention.
EXPPRF = "head.experts."


def safe_torch_load(path: str | os.PathLike[str], *, map_location: Any = "cpu") -> Any:
    """Best-effort ``torch.load`` wrapper compatible across torch versions."""

    pthstr = os.fspath(path)

    try:
        return torch.load(pthstr, map_location=map_location, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        # Older torch: no weights_only kwarg.
        return torch.load(pthstr, map_location=map_location)
    except Exception:
        # Compatibility fallback: allow non-weight objects.
        try:
            return torch.load(pthstr, map_location=map_location, weights_only=False)  # type: ignore[call-arg]
        except TypeError:
            return torch.load(pthstr, map_location=map_location)


def atomic_torch_save(obj: Any, path: str | os.PathLike[str]) -> None:
    """Atomically write a torch payload to ``path``."""

    dstpth = Path(path)
    outdir = dstpth.parent
    outdir.mkdir(parents=True, exist_ok=True)

    fd, tmppth = tempfile.mkstemp(prefix=dstpth.name + ".", suffix=".tmp", dir=str(outdir))
    os.close(fd)

    try:
        torch.save(obj, tmppth)
        os.replace(tmppth, str(dstpth))
        tmppth = ""
    finally:
        if tmppth:
            try:
                os.remove(tmppth)
            except FileNotFoundError:
                pass


def atomic_json_dump(payload: Any, path: str | os.PathLike[str], *, indent: int = 2) -> None:
    """Atomically write JSON to ``path``."""

    dstpth = Path(path)
    outdir = dstpth.parent
    outdir.mkdir(parents=True, exist_ok=True)

    fd, tmppth = tempfile.mkstemp(prefix=dstpth.name + ".", suffix=".tmp", dir=str(outdir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as filobj:
            json.dump(payload, filobj, indent=indent)
        os.replace(tmppth, str(dstpth))
        tmppth = ""
    finally:
        if tmppth:
            try:
                os.remove(tmppth)
            except FileNotFoundError:
                pass


def to_cpu_detached(valobj: Any) -> Any:
    """Detach + move tensors to CPU; pass through non-tensors."""

    if torch.is_tensor(valobj):
        return valobj.detach().cpu()
    return valobj


def infer_num_experts(state: Mapping[str, Any], *, expert_prefix: str = EXPPRF) -> int:
    """Infer expert count from keys like ``head.experts.{idx}.<...>``."""

    maxidx = -1
    for keystr in state.keys():
        if not isinstance(keystr, str) or not keystr.startswith(expert_prefix):
            continue
        try:
            idxval = int(keystr[len(expert_prefix) :].split(".", 1)[0])
        except ValueError:
            continue
        if idxval > maxidx:
            maxidx = idxval
    return maxidx + 1


def split_model_state(
    state: Mapping[str, Any],
    *,
    expert_prefix: str = EXPPRF,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Split a monolithic state_dict into core params and per-expert params."""

    core6x: dict[str, Any] = {}
    expmap: dict[int, dict[str, Any]] = {}

    for keystr, valobj in state.items():
        if not isinstance(keystr, str) or not keystr.startswith(expert_prefix):
            core6x[keystr] = to_cpu_detached(valobj)
            continue

        rest6x = keystr[len(expert_prefix) :]
        parts6 = rest6x.split(".", 1)
        if len(parts6) != 2:
            core6x[keystr] = to_cpu_detached(valobj)
            continue
        try:
            idxval = int(parts6[0])
        except ValueError:
            core6x[keystr] = to_cpu_detached(valobj)
            continue

        expmap.setdefault(idxval, {})[parts6[1]] = to_cpu_detached(valobj)

    return core6x, expmap


def expert_param_keys(
    state: Mapping[str, Any],
    expert_id: int,
    *,
    expert_prefix: str = EXPPRF,
) -> list[str]:
    """Return all state_dict keys for a given expert id."""

    expprf = f"{expert_prefix}{int(expert_id)}."
    return [keystr for keystr in state.keys() if isinstance(keystr, str) and keystr.startswith(expprf)]
