"""INSTNCT evolution helpers.

This module is a behavior-preserving extraction of the *evolution mode* tail
from the legacy monolithic training script.

Scope is intentionally narrow:
- ``mutate_state_dict``
- ``save_evo_checkpoint``
- ``run_evolution``

Model internals and training/eval logic are treated as external dependencies
and must be injected by the caller.

The legacy behavior is intentionally kept "warts and all" (e.g. the mutation
noise is generated on CPU and returns CPU tensors) so this module can be
dropped into the monolith with minimal churn.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import torch

from ._checkpoint_io import atomic_torch_save


def _is_pointer_param(name: str) -> bool:
    """Heuristic used by the legacy script to identify pointer-like params."""

    return (
        name.startswith("theta_ptr_reduced")
        or name.startswith("theta_gate_reduced")
        or name.startswith("jump_score")
        or name.startswith("gate_head")
    )


def mutate_state_dict(
    parent_state: Mapping[str, torch.Tensor],
    std: float,
    pointer_only: bool = False,
    *,
    is_pointer_param: Callable[[str], bool] = _is_pointer_param,
) -> Dict[str, torch.Tensor]:
    """Return a mutated copy of a ``state_dict``.

    Behavior matches the legacy excerpt:
    - Non-floating tensors are cloned unchanged.
    - If ``pointer_only`` is True, non-pointer parameters are cloned unchanged.
    - Noise is sampled on CPU and added to a CPU copy of the parent tensor.

    The returned floating tensors live on CPU (matching the legacy code).
    """

    child: Dict[str, torch.Tensor] = {}
    for keystr, tenval in parent_state.items():
        if not torch.is_floating_point(tenval):
            child[keystr] = tenval.clone()
            continue
        if pointer_only and not is_pointer_param(keystr):
            child[keystr] = tenval.clone()
            continue
        noise6 = torch.randn_like(tenval, device="cpu") * std
        child[keystr] = (tenval.cpu() + noise6).to(tenval.dtype)
    return child


@dataclass(frozen=True)
class EvolutionConfig:
    """Configuration bundle mirroring the legacy ``EVO_*`` globals."""

    pop: int
    gens: int
    steps: int
    mut_std: float
    pointer_only: bool
    checkpoint_every: int
    resume: bool
    checkpoint_individual: bool
    progress: bool


def save_evo_checkpoint(
    gen: int,
    model: Any,
    train_stats: Any,
    eval_stats: Any,
    fitness: float,
    *,
    root: str,
    checkpoint_every: int,
    log: Callable[[str], None],
) -> None:
    """Save evolution checkpoints.

    Writes:
    - ``{root}/artifacts/evolution/evo_latest.pt``
    - Periodic snapshots when ``checkpoint_every > 0``.

    The periodic log string is kept identical to legacy.
    """

    evodir = os.path.join(root, "artifacts", "evolution")
    os.makedirs(evodir, exist_ok=True)

    payobj = {
        "gen": gen,
        "model": model.state_dict(),
        "train": train_stats,
        "eval": eval_stats,
        "fitness": fitness,
    }

    latpth = os.path.join(evodir, "evo_latest.pt")
    atomic_torch_save(payobj, latpth)

    if checkpoint_every > 0 and gen % checkpoint_every == 0:
        genpth = os.path.join(evodir, f"evo_gen_{gen:06d}.pt")
        atomic_torch_save(payobj, genpth)
        log(f"Evolution checkpoint saved @ gen {gen} -> {genpth}")


def run_evolution(
    dataset_name: str,
    loader: Any,
    eval_loader: Any,
    input_dim: int,
    num_classes: int,
    *,
    root: str,
    ring_len: int,
    slot_dim: int,
    config: EvolutionConfig,
    model_ctor: Callable[..., Any],
    train_steps: Callable[..., Any],
    eval_model: Callable[..., Mapping[str, Any]],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Run the simple evolutionary training loop.

    This mirrors the legacy implementation:
    - optional resume from ``artifacts/evolution/evo_latest.pt``
    - initialize population
    - per generation: train+eval each individual, keep elites, refill by
      mutating random elites.

    Returns a summary dict (matching the legacy shape).

    If ``config.gens <= 0`` the loop is infinite (legacy behavior).
    """

    evodir = os.path.join(root, "artifacts", "evolution")
    evochk = os.path.join(evodir, "evo_latest.pt")

    ressta: Optional[Mapping[str, torch.Tensor]] = None
    stagen = 0

    if config.resume and os.path.exists(evochk):
        try:
            payobj = torch.load(evochk, map_location="cpu")
            ressta = payobj.get("model")
            stagen = int(payobj.get("gen", -1)) + 1
            log(f"Evolution resume: loaded {evochk} (start_gen={stagen})")
        except Exception as exc:
            log(f"Evolution resume failed: {exc}; starting fresh.")

    log(
        "=== Evolution mode | dataset="
        f"{dataset_name} | pop={config.pop} gens={config.gens} steps/ind={config.steps} "
        f"pointer_only={int(config.pointer_only)} resume={int(config.resume)} start_gen={stagen} ==="
    )

    # init population
    poplst: list[Any] = []
    if ressta is not None:
        elite = model_ctor(input_dim=input_dim, num_classes=num_classes, ring_len=ring_len, slot_dim=slot_dim)
        elite.load_state_dict(ressta)
        poplst.append(elite)

        while len(poplst) < config.pop:
            child = model_ctor(
                input_dim=input_dim,
                num_classes=num_classes,
                ring_len=ring_len,
                slot_dim=slot_dim,
            )
            child.load_state_dict(
                mutate_state_dict(ressta, std=config.mut_std, pointer_only=config.pointer_only)
            )
            poplst.append(child)
    else:
        for _ in range(config.pop):
            modobj = model_ctor(
                input_dim=input_dim,
                num_classes=num_classes,
                ring_len=ring_len,
                slot_dim=slot_dim,
            )
            poplst.append(modobj)

    bestev: Optional[Tuple[float, Any, Any, Mapping[str, Any]]] = None

    if config.gens > 0:
        genitr: Iterable[int] = range(stagen, stagen + config.gens)
    else:
        genitr = count(stagen)

    for genval in genitr:
        fitlst: list[Tuple[float, Any, Any, Mapping[str, Any]]] = []
        for idxval, modobj in enumerate(poplst):
            trnst = train_steps(modobj, loader, config.steps, dataset_name, f"evo_{genval}_{idxval}")
            evlst = eval_model(modobj, eval_loader, dataset_name, f"evo_{genval}_{idxval}")
            fitval = 1.0 - evlst["eval_loss"]
            fitlst.append((fitval, modobj, trnst, evlst))

        fitlst.sort(key=lambda x: x[0], reverse=True)
        topk = max(1, config.pop // 3)
        elites = fitlst[:topk]

        if bestev is None or elites[0][0] > bestev[0]:
            bestev = elites[0]

        if config.progress:
            log(
                f"Gen {genval}: best_acc={elites[0][3]['eval_acc']:.4f}, "
                f"loss={elites[0][3]['eval_loss']:.4f}"
            )

        save_evo_checkpoint(
            genval,
            elites[0][1],
            elites[0][2],
            elites[0][3],
            elites[0][0],
            root=root,
            checkpoint_every=config.checkpoint_every,
            log=log,
        )

        # Refill population
        newpop: list[Any] = [elm[1] for elm in elites]  # keep elites
        while len(newpop) < config.pop:
            parobj = random.choice(elites)[1]
            child = model_ctor(
                input_dim=input_dim,
                num_classes=num_classes,
                ring_len=ring_len,
                slot_dim=slot_dim,
            )
            child.load_state_dict(
                mutate_state_dict(parobj.state_dict(), std=config.mut_std, pointer_only=config.pointer_only)
            )
            newpop.append(child)
        poplst = newpop

    # Unreachable for infinite runs; kept to mirror legacy structure.
    assert bestev is not None
    fitval, _bestm, trnst, evlst = bestev
    return {
        "mode": "evolution",
        "best_train": trnst,
        "best_eval": evlst,
        "best_fitness": fitval,
    }
