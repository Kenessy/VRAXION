# Gap List: Eval/IQ ("Smartness") v0 (2026-02-07)

This is a 1-page snapshot of what we cannot answer yet, and what has to be
built/measured next. It is intentionally short.

Context (what we already have):
- VRA-77 established batch targets near a reserved VRAM ratio (~0.85) per config.
- VRA-78 produces an OD1-first packet frontier with (VRAM ratio, tokens/s, assoc-byte accuracy).
- Wiki doctrine already separates systems-first vs quality-second: see Chapter 09.

Goal:
- Make \"ant smartness\" measurable under declared budgets, with falsifiers and artifacts.

## Missing metric/task (evaluation)

1) We do not yet have a minimal eval ladder that reliably distinguishes:
- stable-but-dumb vs stable-and-improving
- bigger ant is better vs more ants is better

Proposed next:
- VRA-79 (issue #69): https://github.com/VRAXION/VRAXION/issues/69
- Register TOT-H005/TOT-H006 in `Theory-of-Thought.md`.

2) We do not have a slow anchor that resists Goodharting on a single synthetic task.

Proposed next:
- VRA-81 (issue #71): https://github.com/VRAXION/VRAXION/issues/71
- VRA-80 contract for labeling (issue #70): https://github.com/VRAXION/VRAXION/issues/70

3) We do not have a consistent quality score contract for sweeps:
- what fields exist in artifacts
- how to compare apples-to-apples
- what is Supported vs Confirmed

Proposed next:
- VRA-80 (issue #70): https://github.com/VRAXION/VRAXION/issues/70
- VRA-82 (issue #72): https://github.com/VRAXION/VRAXION/issues/72

## Missing implementation feature (tooling)

1) A single command that:
- loads a checkpoint/run-root
- runs the full eval ladder deterministically
- writes a single report artifact with per-task metrics + confounds

Proposed next: VRA-81 (issue #71): https://github.com/VRAXION/VRAXION/issues/71

2) A unified frontier report that makes the distinction explicit:
- systems frontier (VRAM, tok/s, stability)
- intelligence frontier (quality metrics at fixed budget)

Proposed next:
- VRA-80 (issue #70): https://github.com/VRAXION/VRAXION/issues/70
- Small wiki distillation in Chapter 09 (stable doctrine).

3) Participation/utilization is not yet a hard gate in sweeps:
- we log some usage metrics, but we don't enforce \"no collapse\" in eval claims

Proposed next: VRA-83 (issue #73): https://github.com/VRAXION/VRAXION/issues/73

## Missing data (runs)

1) We do not yet have replicated quality results (seeds) for any ant tier at fixed budgets.

Proposed next: VRA-84 (issue #74): https://github.com/VRAXION/VRAXION/issues/74

2) We do not yet have an evidence-backed answer for:
- preferred ant size for a \"civilian GPU\" envelope (e.g., 12GB/16GB/24GB)
- whether mixed-size ants are worth the complexity

Proposed next:
- Hold mixed-size design in Hypotheses-stage until Module A exists (no untestable architecture bets).

