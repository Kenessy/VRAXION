"""INSTNCT entrypoint orchestration helpers (refactor scaffold).

This module extracts *orchestration* into dependency-injected functions while
keeping domain logic (train/eval/evolve) in the original module.

Intended integration pattern (high level):

    from tools.instnct_entrypoint import main, EntrypointDeps

    def _train(ctx): ...
    def _eval(ctx): ...
    def _evolve(ctx): ...

    if __name__ == "__main__":
        raise SystemExit(main(deps=EntrypointDeps(train=_train, evaluate=_eval, evolve=_evolve)))

This keeps integration points explicit while allowing unit tests to validate:
- env var parsing semantics
- run-plan building
- action dispatch order
- header emission stability

Constraints:
- Stdlib only (no new dependencies)
- Header lines are treated as stable ASCII (pass-through by default)
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

from .env_utils import env_bool, env_str
from .log_headers import WriteLine, default_writer, emit_header


class Action(str, Enum):
    TRAIN = "train"
    EVAL = "eval"
    EVOLVE = "evolve"


@dataclass(frozen=True)
class EnvKeys:
    """Names of environment variables used by the entrypoint logic.

    IMPORTANT: keep these stable for callers. Defaults use the VRAXION prefix.
    """

    mode: str = "VRX_MODE"
    do_train: str = "VRX_DO_TRAIN"
    do_eval: str = "VRX_DO_EVAL"
    do_evolve: str = "VRX_DO_EVOLVE"
    dry_run: str = "VRX_DRY_RUN"
    run_id: str = "VRX_RUN_ID"
    no_header: str = "VRX_NO_HEADER"


@dataclass(frozen=True)
class RunPlan:
    actions: tuple[Action, ...]
    dry_run: bool
    run_id: str


@dataclass(frozen=True)
class EntrypointContext:
    """Context passed to action callables.

    Keep this lightweight and explicit. Domain code can attach additional objects
    via closures or by storing them in `extras`.
    """

    args: argparse.Namespace
    env: Mapping[str, str]
    run_id: str
    write_line: WriteLine
    extras: Mapping[str, Any] = field(default_factory=dict)


ActionFn = Callable[[EntrypointContext], Optional[int]]


@dataclass(frozen=True)
class EntrypointDeps:
    """Injectable dependencies for the entrypoint orchestration.

    Pass wrappers around existing runner globals (train/eval/evolve, etc.)
    to reduce global coupling without rewriting those functions.
    """

    train: Optional[ActionFn] = None
    evaluate: Optional[ActionFn] = None
    evolve: Optional[ActionFn] = None

    # Stable header lines: keep these ASCII and grep-friendly.
    header_lines: Sequence[str] = ()

    # IO
    write_line: Optional[WriteLine] = None

    # Env semantics
    env_keys: EnvKeys = field(default_factory=EnvKeys)

    # Deterministic run-id generation in tests
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)

    # Whether to sanitize header to ASCII (recommended if logs must be ASCII-only)
    ensure_header_ascii: bool = True

    # If true, unknown/invalid env values do NOT crash; they simply fall back to defaults.
    strict_env: bool = False


def _normalize_mode_tokens(mode: str) -> list[str]:
    # Accept flexible separators: commas, spaces, pipes, plus, and hyphens.
    tokens = re.split(r"[^A-Za-z0-9]+", mode.strip())
    return [tok.lower() for tok in tokens if tok.strip()]


def parse_mode(mode: str) -> tuple[Action, ...]:
    """Parse a mode string like 'train-eval-evolve' into ordered actions."""

    tokens = _normalize_mode_tokens(mode)
    if not tokens:
        return ()

    mapping = {
        "train": Action.TRAIN,
        "training": Action.TRAIN,
        "eval": Action.EVAL,
        "evaluate": Action.EVAL,
        "evaluation": Action.EVAL,
        "evolve": Action.EVOLVE,
        "evolution": Action.EVOLVE,
    }

    actions: list[Action] = []
    for tok in tokens:
        if tok not in mapping:
            raise ValueError(f"Unknown mode token: {tok!r}")
        act = mapping[tok]
        if act not in actions:
            actions.append(act)
    return tuple(actions)


def _default_run_id(now: datetime) -> str:
    # ISO-like, ASCII, no ':' to remain Windows/path friendly.
    return now.astimezone(timezone.utc).strftime("vrx_%Y%m%d_%H%M%S_utc")


def build_run_plan(
    args: argparse.Namespace,
    env: Mapping[str, str],
    *,
    env_keys: EnvKeys,
    strict_env: bool,
    now_fn: Callable[[], datetime],
) -> RunPlan:
    """Pure run-plan builder (easy to test)."""

    # Derive run_id (CLI > env > generated).
    run_id = args.run_id or env_str(env, env_keys.run_id)
    if not run_id:
        run_id = _default_run_id(now_fn())

    # dry-run can be set by CLI or env.
    dry_run = bool(args.dry_run)
    if not dry_run:
        dry_run, _ = env_bool(env, env_keys.dry_run, default=False, strict=strict_env)

    # Determine actions.
    actions: tuple[Action, ...] = ()
    if args.mode:
        actions = parse_mode(args.mode)
    elif args.train or args.eval or args.evolve:
        seq: list[Action] = []
        if args.train:
            seq.append(Action.TRAIN)
        if args.eval:
            seq.append(Action.EVAL)
        if args.evolve:
            seq.append(Action.EVOLVE)
        actions = tuple(seq)
    else:
        mode = env_str(env, env_keys.mode)
        if mode:
            actions = parse_mode(mode)
        else:
            # Fall back to individual env flags (common pattern in legacy scripts).
            seq2: list[Action] = []
            do_train, _ = env_bool(env, env_keys.do_train, default=False, strict=strict_env)
            do_eval, _ = env_bool(env, env_keys.do_eval, default=False, strict=strict_env)
            do_evolve, _ = env_bool(env, env_keys.do_evolve, default=False, strict=strict_env)
            if do_train:
                seq2.append(Action.TRAIN)
            if do_eval:
                seq2.append(Action.EVAL)
            if do_evolve:
                seq2.append(Action.EVOLVE)
            actions = tuple(seq2)

    if not actions:
        # Conservative default: training only.
        actions = (Action.TRAIN,)

    return RunPlan(actions=actions, dry_run=dry_run, run_id=run_id)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)

    # Preserve legacy usage patterns: both --mode and flags.
    p.add_argument("--mode", default=None, help="Run mode: train/eval/evolve or combos like train-eval-evolve")
    p.add_argument("--train", action="store_true", help="Run training")
    p.add_argument("--eval", action="store_true", help="Run evaluation")
    p.add_argument("--evolve", action="store_true", help="Run evolution")
    p.add_argument("--dry-run", action="store_true", help="Print plan but do not execute actions")
    p.add_argument("--run-id", default=None, help="Stable run identifier (overrides env)")
    p.add_argument("--no-header", action="store_true", help="Disable header emission")

    return p


def _resolve_write_line(deps: EntrypointDeps) -> WriteLine:
    return deps.write_line or default_writer()


def _maybe_emit_header(
    deps: EntrypointDeps,
    env: Mapping[str, str],
    args: argparse.Namespace,
    write_line: WriteLine,
) -> None:
    # CLI flag wins, then env var.
    if args.no_header:
        return
    no_header, _ = env_bool(env, deps.env_keys.no_header, default=False, strict=deps.strict_env)
    if no_header:
        return
    if deps.header_lines:
        emit_header(
            list(deps.header_lines),
            write_line=write_line,
            ensure_ascii=deps.ensure_header_ascii,
        )


def _dispatch_action(action: Action, deps: EntrypointDeps, ctx: EntrypointContext) -> int:
    fn: Optional[ActionFn]
    if action is Action.TRAIN:
        fn = deps.train
    elif action is Action.EVAL:
        fn = deps.evaluate
    elif action is Action.EVOLVE:
        fn = deps.evolve
    else:
        raise AssertionError(f"Unhandled action: {action}")

    if fn is None:
        raise RuntimeError(f"No callable provided for action: {action.value}")

    rc = fn(ctx)
    return int(rc) if rc is not None else 0


def run(plan: RunPlan, *, deps: EntrypointDeps, args: argparse.Namespace, env: Mapping[str, str]) -> int:
    """Execute a run plan using injected dependencies."""

    write_line = _resolve_write_line(deps)
    ctx = EntrypointContext(args=args, env=env, run_id=plan.run_id, write_line=write_line)

    if plan.dry_run:
        write_line(f"[VRX] dry-run run_id={plan.run_id} actions={[a.value for a in plan.actions]}")
        return 0

    exit_code = 0
    for action in plan.actions:
        rc = _dispatch_action(action, deps, ctx)
        if rc != 0:
            exit_code = rc
            break
    return exit_code


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    deps: Optional[EntrypointDeps] = None,
) -> int:
    """Entrypoint function (importable + testable).

    Side effects:
    - reads argv/env (unless passed)
    - optionally prints header (via write_line)
    - calls injected action callables
    """

    if env is None:
        env = os.environ
    if deps is None:
        deps = EntrypointDeps()

    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    plan = build_run_plan(args, env, env_keys=deps.env_keys, strict_env=deps.strict_env, now_fn=deps.now_fn)
    write_line = _resolve_write_line(deps)

    _maybe_emit_header(deps, env, args, write_line)

    # Keep orchestration minimal; domain code handles its own logging.
    return run(plan, deps=deps, args=args, env=env)
