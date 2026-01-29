"""INSTNCT orchestration helpers.

This module coordinates training/evaluation loops for the Golden Draft runner.

It contains:
- ``run_phase``: one dataset phase (train + eval)
- ``run_lockout_test``: deterministic synth A->B probe
- ``main``: top-level orchestration (including synth short-circuit)

Design goals
- Make dependencies explicit via a context object (for auditability/testing).
- Use atomic JSON writes for summaries (avoid partial/corrupt files).

Model internals remain external; callers inject the model class and the
train/eval functions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch

from ._checkpoint_io import atomic_json_dump
from .instnct_evolution import EvolutionConfig, run_evolution


@dataclass(frozen=True)
class InstnctRunnerContext:
    """Explicit dependency bundle for INSTNCT orchestration."""

    # Paths / environment
    root: str
    data_dir: str
    checkpoint_path: str

    # Run controls
    resume: bool
    run_mode: str
    eval_split: str
    seed: int

    # Runtime
    device: str
    offline_only: bool
    disable_sync: bool

    # Model shape
    ring_len: int
    slot_dim: int

    # Optional synth metadata (may be filled earlier in the pipeline)
    synth_meta: Dict[str, Any]

    # Lockout config
    phase_a_steps: int
    phase_b_steps: int
    synth_len: int
    synth_shuffle: bool

    # Evolution config
    evo: EvolutionConfig

    # Callables
    log: Callable[[str], None]
    rotate_artifacts: Callable[[], None]
    set_seed: Callable[[int], None]
    maybe_override_expert_heads: Callable[[str], None]
    sync_current_to_last: Callable[[], None]

    # Data / loaders
    get_seq_mnist_loader: Callable[[], Tuple[Any, int, Any]]
    build_eval_loader_from_subset: Callable[..., Tuple[Any, int]]
    build_eval_loader_from_dataset: Callable[..., Tuple[Any, int]]
    build_synth_pair_loaders: Callable[[], Tuple[Any, Any, Any]]
    log_eval_overlap: Callable[..., None]

    # Training / eval
    model_ctor: Callable[..., Any]
    train_wallclock: Callable[..., Any]
    train_steps: Callable[..., Any]
    eval_model: Callable[..., Mapping[str, Any]]


def run_phase(
    dataset_name: str,
    loader: Any,
    eval_loader: Any,
    input_dim: int,
    num_classes: int,
    *,
    ctx: InstnctRunnerContext,
) -> Dict[str, Any]:
    """Train/eval the AbsoluteHallway model for a dataset and return a summary."""

    extra = ""
    if ctx.synth_meta.get("enabled"):
        extra = f" | synth_mode={ctx.synth_meta.get('mode')}"

    ctx.log(f"=== INSTNCT | dataset={dataset_name} | num_classes={num_classes}{extra} ===")

    hallway = ctx.model_ctor(
        input_dim=input_dim,
        num_classes=num_classes,
        ring_len=ctx.ring_len,
        slot_dim=ctx.slot_dim,
    )
    hall_train = ctx.train_wallclock(
        hallway,
        loader,
        dataset_name,
        "absolute_hallway",
        num_classes,
        eval_loader=eval_loader,
    )
    hall_eval = ctx.eval_model(hallway, eval_loader, dataset_name, "absolute_hallway")

    result: Dict[str, Any] = {
        "dataset": dataset_name,
        "absolute_hallway": {"train": hall_train, "eval": hall_eval},
    }
    if ctx.synth_meta:
        result["meta"] = dict(ctx.synth_meta)
    return result


def run_lockout_test(*, ctx: InstnctRunnerContext) -> Dict[str, Any]:
    """2-phase synthetic training to probe catastrophic forgetting."""

    ctx.log("=== Lockout test | deterministic synth A->B (label flip) ===")

    loader_a, loader_b, collate = ctx.build_synth_pair_loaders()
    eval_a, _ = ctx.build_eval_loader_from_subset(loader_a.dataset, input_collate=collate)
    eval_b, _ = ctx.build_eval_loader_from_subset(loader_b.dataset, input_collate=collate)

    model = ctx.model_ctor(input_dim=1, num_classes=2, ring_len=ctx.ring_len, slot_dim=ctx.slot_dim)

    train_a = ctx.train_steps(model, loader_a, ctx.phase_a_steps, "synthA", "absolute_hallway")
    eval_a_post = ctx.eval_model(model, eval_a, "synthA", "absolute_hallway")
    eval_b_pre = ctx.eval_model(model, eval_b, "synthB_pre", "absolute_hallway")

    train_b = ctx.train_steps(model, loader_b, ctx.phase_b_steps, "synthB", "absolute_hallway")
    eval_b_post = ctx.eval_model(model, eval_b, "synthB_post", "absolute_hallway")
    eval_a_post_b = ctx.eval_model(model, eval_a, "synthA_postB", "absolute_hallway")

    return {
        "mode": "lockout",
        "phase_a": {"train": train_a, "eval": eval_a_post},
        "phase_b": {"pre_eval": eval_b_pre, "train": train_b, "eval": eval_b_post},
        "forgetting_check": eval_a_post_b,
        "meta": {
            "phase_a_steps": ctx.phase_a_steps,
            "phase_b_steps": ctx.phase_b_steps,
            "synth_len": ctx.synth_len,
            "synth_shuffle": ctx.synth_shuffle,
        },
    }


def _build_mnist_eval_loader(ctx: InstnctRunnerContext, mnist_loader: Any, mnist_collate: Any) -> Tuple[Any, int, str]:
    """Match legacy evaluation split selection logic (subset vs MNIST test)."""

    eval_label = "train_subset"
    if ctx.synth_meta.get("enabled") or ctx.eval_split == "subset":
        mnist_eval_loader, eval_size = ctx.build_eval_loader_from_subset(
            mnist_loader.dataset,
            input_collate=mnist_collate,
        )
        eval_label = "train_subset"
        return mnist_eval_loader, eval_size, eval_label

    try:
        import torchvision.transforms as T
        from torchvision.datasets import MNIST

        transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
        eval_ds = MNIST(
            os.path.join(ctx.data_dir, "mnist_seq"),
            train=False,
            download=not ctx.offline_only,
            transform=transform,
        )
        mnist_eval_loader, eval_size = ctx.build_eval_loader_from_dataset(eval_ds, input_collate=mnist_collate)
        eval_label = "mnist_test"
        return mnist_eval_loader, eval_size, eval_label
    except Exception as exc:
        ctx.log(f"[eval] test split unavailable ({exc}); falling back to train subset")
        mnist_eval_loader, eval_size = ctx.build_eval_loader_from_subset(
            mnist_loader.dataset,
            input_collate=mnist_collate,
        )
        eval_label = "train_subset_fallback"
        return mnist_eval_loader, eval_size, eval_label


def main(*, ctx: Optional[InstnctRunnerContext] = None) -> None:
    """Entry point used by Golden Draft runner scripts.

    Environment variables (strict "1" parsing where applicable):
    - VRX_SYNTH: if "1", bypass MNIST and run synthetic loop
    - VRX_SYNTH_ONCE: if "1" with VRX_SYNTH=1, run one iteration and exit
    """

    if ctx is None:
        ctx = default_context()

    os.makedirs(ctx.data_dir, exist_ok=True)
    if not ctx.resume:
        ctx.rotate_artifacts()

    ctx.set_seed(ctx.seed)
    if ctx.resume:
        ctx.maybe_override_expert_heads(os.path.abspath(ctx.checkpoint_path))

    # Reduce kernel search overhead / variance.
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    ctx.log(f"INSTNCT start | device={ctx.device} | offline_only={ctx.offline_only}")

    summary = []

    # Synthetic short-circuit: if VRX_SYNTH=1, bypass MNIST entirely.
    synth_env = os.environ.get("VRX_SYNTH", "0").strip()
    synth_once = os.environ.get("VRX_SYNTH_ONCE") == "1"
    if synth_env == "1":
        synth_loader, synth_classes, synth_collate = ctx.get_seq_mnist_loader()
        synth_eval_loader, eval_size = ctx.build_eval_loader_from_subset(
            synth_loader.dataset,
            input_collate=synth_collate,
        )
        ctx.log_eval_overlap(synth_loader.dataset, synth_eval_loader.dataset, eval_size, "synth_subset")

        # Run once for probes; otherwise run forever until manually stopped.
        if synth_once:
            run_phase(
                "synth",
                synth_loader,
                synth_eval_loader,
                input_dim=1,
                num_classes=synth_classes,
                ctx=ctx,
            )
            return

        while True:
            run_phase(
                "synth",
                synth_loader,
                synth_eval_loader,
                input_dim=1,
                num_classes=synth_classes,
                ctx=ctx,
            )
        return

    if ctx.run_mode == "lockout":
        summary.append(run_lockout_test(ctx=ctx))
    else:
        mnist_loader, mnist_classes, mnist_collate = ctx.get_seq_mnist_loader()

        mnist_eval_loader, eval_size, eval_label = _build_mnist_eval_loader(ctx, mnist_loader, mnist_collate)
        ctx.log_eval_overlap(mnist_loader.dataset, mnist_eval_loader.dataset, eval_size, eval_label)

        if ctx.run_mode == "evolution":
            summary.append(
                run_evolution(
                    "seq_mnist",
                    mnist_loader,
                    mnist_eval_loader,
                    input_dim=1,
                    num_classes=mnist_classes,
                    root=ctx.root,
                    ring_len=ctx.ring_len,
                    slot_dim=ctx.slot_dim,
                    config=ctx.evo,
                    model_ctor=ctx.model_ctor,
                    train_steps=ctx.train_steps,
                    eval_model=ctx.eval_model,
                    log=ctx.log,
                )
            )
        else:
            summary.append(
                run_phase(
                    "seq_mnist",
                    mnist_loader,
                    mnist_eval_loader,
                    input_dim=1,
                    num_classes=mnist_classes,
                    ctx=ctx,
                )
            )

    # Ensure GPU work is complete before writing summary to avoid partial/hung writes.
    # For synthetic indefinite run we skip summary writing and syncing to keep process alive.
    if torch.cuda.is_available() and not ctx.disable_sync:
        torch.cuda.synchronize()

    if summary:
        sum_path = os.path.join(ctx.root, "summaries", "current", "vraxion_summary.json")
        try:
            atomic_json_dump(summary, sum_path, indent=2)
            ctx.log(f"Run done. Summary saved to {sum_path}")
        except Exception as exc:
            ctx.log(f"Summary write skipped: {exc}")

        try:
            ctx.sync_current_to_last()
        except Exception as exc:
            ctx.log(f"Summary sync skipped: {exc}")


def default_context() -> InstnctRunnerContext:
    """Build a self-contained runner context from Golden Code + Golden Draft."""

    # Ensure Golden Draft + Golden Code are importable for standalone runs.
    #
    # Tests bootstrap sys.path in tests/conftest.py, but end-user scripts should
    # not depend on that.
    import sys
    from pathlib import Path

    draftr = Path(__file__).resolve().parents[1]
    reproo = draftr.parent

    if str(draftr) not in sys.path:
        sys.path.insert(0, str(draftr))

    candls: list[str] = []
    for keystr in ("VRAXION_GOLDEN_SRC", "GOLDEN_CODE_ROOT", "GOLDEN_CODE_PATH", "GOLDEN_CODE_DIR"):
        envval = os.environ.get(keystr)
        if envval:
            candls.append(envval)

    candls.append(str(reproo / "Golden Code"))
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

    from vraxion.settings import load_settings
    from vraxion.instnct import infra
    from vraxion.instnct.absolute_hallway import AbsoluteHallway
    from vraxion.instnct.seed import _maybe_override_expert_heads, set_seed

    from . import instnct_data, instnct_eval, instnct_train_steps, instnct_train_wallclock

    cfg = load_settings()

    # Keep infra in sync with settings-driven paths.
    infra.ROOT = str(cfg.root)
    infra.LOG_PATH = str(cfg.log_path)

    eval_spec = instnct_eval.EvalLoaderSpec(eval_samples=int(cfg.eval_samples), batch_size=int(cfg.batch_size))
    eval_deps = instnct_eval.EvalDeps(
        device=str(cfg.device),
        dtype=cfg.dtype,
        amp_autocast=instnct_train_wallclock.amp_autocast,
        log=infra.log,
        synth_mode=str(getattr(cfg, "synth_mode", "")),
        mi_shuffle=bool(getattr(cfg, "mi_shuffle", False)),
        mitosis_enabled=bool(getattr(cfg, "mitosis_enabled", False)),
    )

    evo_cfg = EvolutionConfig(
        pop=int(cfg.evo_pop),
        gens=int(cfg.evo_gens),
        steps=int(cfg.evo_steps),
        mut_std=float(cfg.evo_mut_std),
        pointer_only=bool(cfg.evo_pointer_only),
        checkpoint_every=int(cfg.evo_checkpoint_every),
        resume=bool(cfg.evo_resume),
        checkpoint_individual=bool(cfg.evo_checkpoint_individual),
        progress=bool(cfg.evo_progress),
    )

    def _eval_model(model: Any, loader: Any, dataset_name: str, model_name: str) -> Mapping[str, Any]:
        return instnct_eval.eval_model(model, loader, dataset_name, model_name, deps=eval_deps)

    def _build_eval_subset(ds: Any, *, input_collate: Any = None) -> Tuple[Any, int]:
        return instnct_eval.build_eval_loader_from_subset(ds, spec=eval_spec, input_collate=input_collate)

    def _build_eval_dataset(ds: Any, *, input_collate: Any = None) -> Tuple[Any, int]:
        return instnct_eval.build_eval_loader_from_dataset(ds, spec=eval_spec, input_collate=input_collate)

    def _log_overlap(train_ds: Any, eval_ds: Any, eval_size: int, label: str) -> None:
        instnct_eval.log_eval_overlap(train_ds, eval_ds, eval_size, label, log=infra.log)

    return InstnctRunnerContext(
        root=str(cfg.root),
        data_dir=str(cfg.data_dir),
        checkpoint_path=str(cfg.checkpoint_path),
        resume=bool(cfg.resume),
        run_mode=str(cfg.run_mode),
        eval_split=str(cfg.eval_split),
        seed=int(cfg.seed),
        device=str(cfg.device),
        offline_only=bool(cfg.offline_only),
        disable_sync=False,
        ring_len=int(cfg.ring_len),
        slot_dim=int(cfg.slot_dim),
        # Keep a live reference so instnct_data can update it during loader builds.
        synth_meta=instnct_data.SYNTH_META,
        phase_a_steps=int(cfg.phase_a_steps),
        phase_b_steps=int(cfg.phase_b_steps),
        synth_len=int(cfg.synth_len),
        synth_shuffle=bool(cfg.synth_shuffle),
        evo=evo_cfg,
        log=infra.log,
        rotate_artifacts=infra.rotate_artifacts,
        set_seed=set_seed,
        maybe_override_expert_heads=_maybe_override_expert_heads,
        sync_current_to_last=infra.sync_current_to_last,
        get_seq_mnist_loader=instnct_data.get_seq_mnist_loader,
        build_eval_loader_from_subset=_build_eval_subset,
        build_eval_loader_from_dataset=_build_eval_dataset,
        build_synth_pair_loaders=instnct_data.build_synth_pair_loaders,
        log_eval_overlap=_log_overlap,
        model_ctor=AbsoluteHallway,
        train_wallclock=instnct_train_wallclock.train_wallclock,
        train_steps=instnct_train_steps.train_steps,
        eval_model=_eval_model,
    )


if __name__ == "__main__":
    main()
