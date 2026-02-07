# Deep Research: Eval/IQ Ladder v0 (Ant "Smartness" Under Fixed Budgets)

Date: 2026-02-07
Scope: Module A deliverable for the Deep-Research Pipeline v1 (Eval/IQ first).

This is not a paper. It is a decision-support document that produces:
- registered hypotheses (with falsifiers and artifacts), and
- a small ticket pack to build the missing tooling.

## TLDR

- VRAXION already has a strong systems frontier pipeline (VRA-76/77/78): VRAM ratio, tokens/s, stability gates, plus one capability metric (`assoc_byte` disjoint accuracy).
- That capability metric is a start, but it is not yet a robust "IQ" axis; we need an evaluation ladder with multiple tasks and a clear budget protocol.
- We will treat "smartness" as a quality frontier measured only after stability is established, under explicitly declared budgets (iso-VRAM / iso-FLOPs/step / iso-params).
- The fastest win is an `eval_suite_v0` runner that produces one deterministic report artifact with per-task metrics and confounds.

## What we're trying to decide

1) What is the minimum evaluation ladder (fast -> medium -> slow) that distinguishes meaningful capability changes without turning into a new research project?
2) Under fixed budget modes (especially iso-VRAM), where do we expect a repeatable Pareto region in (ant body size) x (expert_heads) x (batch)?
3) How do we stop ourselves from confusing:
- runs stable and fast with
- is actually better at the task?

## Prior art (practical pointers, not exhaustive)

We want tasks that are:
- deterministic to generate,
- cheap enough to run repeatedly,
- hard to game with trivial shortcuts,
- and provide a stable score surface (not pure noise).

Candidate families:
- Associative recall / variable binding tasks (already present as `assoc_byte`).
- Algorithmic sequence tasks (copy/reverse/parity/brackets) for generalization under length or distribution shift.
- Compositional generalization tasks (e.g., SCAN-style) if we later want a language-like compositional axis without relying on web-scale datasets.
- A frozen anchor evaluation that we explicitly do not optimize against directly (to reduce Goodhart pressure).

Reference links (optional):
- Long Range Arena (long-context benchmark): https://arxiv.org/abs/2011.04006
  - Code: https://github.com/google-research/long-range-arena
- SCAN (compositional generalization): https://arxiv.org/abs/1711.00350
  - Follow-up critique + NACS: https://arxiv.org/abs/1809.04640
- bAbI toy reasoning tasks: https://arxiv.org/abs/1502.05698
  - Task generation code: https://github.com/facebookarchive/bAbI-tasks

## What VRAXION already has (repo/wiki truth)

Evaluation doctrine exists and is correct:
- Wiki Chapter 09 explicitly distinguishes systems-first then quality-second.

Artifact and gate contracts exist:
- GPU Chapter 01 + probe harness `metrics.json` provides PASS/FAIL and stability gates as artifact-truth.

A joined datapoint schema already exists:
- `Golden Draft/tools/ant_ratio_packet_v0.py` joins probe artifacts and capability artifacts into `ant_ratio_packet_v0`.
- Capability currently uses assoc-byte disjoint evaluation from `report.json.eval.eval_acc`.

A first 3-axis frontier already exists:
- `Golden Draft/docs/ops/ant_ratio_frontier_v0.md` defines (VRAM ratio, tokens/s, assoc accuracy) and derived columns.

Implication:
- We do not need to invent a new pipeline. We need to extend the capability side from one metric to an eval ladder.

## Budget protocol (locked doctrine for eval claims)

Every evaluation claim MUST specify:
- Budget mode: iso-VRAM, iso-FLOPs/step, or iso-params.
- Token/step policy for capability runs (fixed token budget is currently the safest default for comparability across batch sizes).
- Fail gates: runs that fail gates are not evidence of progress.

Practical default for v0:
- Systems frontier: iso-VRAM near target ratio (already VRA-77).
- Capability: fixed token budget with clamp to [min_steps, max_steps] (already in ant-ratio packet/sweep metadata).

## Minimal eval ladder v0 (proposal)

The ladder is designed so each rung is:
- cheap,
- deterministic,
- and returns a rankable scalar with a known baseline.

### Fast rung (seconds-minutes): "Does it learn anything at all?"

1) `assoc_byte` disjoint accuracy (existing)
- Baseline: chance = 1/num_classes (e.g., 1/256).
- Confound: can saturate early and become non-informative; can also be solved by shallow shortcuts depending on generator.

2) `assoc_byte` shifted-length stress (new)
- Same task, but evaluate on a longer seq_len than seen in the short capability budget run.
- Goal: detect works-only-at-one-length brittleness.

### Medium rung (minutes): "Does it compose or only memorize?"

3) 2-hop associative chaining (new)
- Example: learn A->B and B->C, query A->C.
- Baseline: chance known, but the task forces a compositional step.
- Confound: if generator leaks chain structure too directly, it becomes another memorize task.

4) Bracket validity (Dyck-1 / Dyck-2) OR parity under distribution shift (new)
- Goal: detect a non-trivial rule generalization at fixed compute.

### Slow rung (optional at v0): "Anchor that resists Goodhart"

5) Frozen holdout generator seed ("do not optimize against") (new)
- Same families as above, but with a separately versioned seed/config and disjoint parameterization.
- Not perfect anti-Goodhart, but a practical first step before introducing external datasets.

## What to log (quality + systems + utilization)

Per run record (minimum):
- Budget mode declaration (explicit).
- Systems: tokens/s, samples/s, peak VRAM reserved/allocated, step-time median/p95.
- Gates: PASS/FAIL and fail reasons (artifact-truth).
- Capability: per-task metrics (accuracy, loss if needed) plus chance baselines.
- Utilization: expert usage entropy, max-share, active expert count when experts/routing matter.

## Hypotheses to register (TOT-H proposals)

These are intentionally framed so they can be tested as soon as the eval ladder exists.

- TOT-H005 (proposed): Systems frontier and intelligence frontier diverge.
  - Prediction: configs that maximize tokens/s at fixed VRAM are not reliably the best configs on eval ladder quality metrics.
  - Falsifier: top systems configs are consistently top quality configs across seeds under the same budget mode.
  - Minimum artifacts: matched sweep packets + eval suite reports + seed list.

- TOT-H006 (proposed): Under iso-VRAM, there is a repeatable ant size Pareto band.
  - Prediction: a stable Pareto region exists in (ant_body_cells, expert_heads, batch) when quality is measured with the ladder.
  - Falsifier: frontier is unstable across seeds or collapses to a single trivial extreme.
  - Minimum artifacts: coarse sweep + confirmation runs, unchanged budgets, full artifacts.

## Experiment tickets (v1 pack)

The goal is to build a small, repeatable implementation surface:
- one runner,
- one report format,
- one place to join into existing packets,
- and one place to visualize.

Tickets created (titles and acceptance criteria are in the GitHub issues):
- VRA-79 (issue #69): https://github.com/VRAXION/VRAXION/issues/69
- VRA-80 (issue #70): https://github.com/VRAXION/VRAXION/issues/70
- VRA-81 (issue #71): https://github.com/VRAXION/VRAXION/issues/71
- VRA-82 (issue #72): https://github.com/VRAXION/VRAXION/issues/72
- VRA-83 (issue #73): https://github.com/VRAXION/VRAXION/issues/73
- VRA-84 (issue #74): https://github.com/VRAXION/VRAXION/issues/74

## Wiki distillation (small, stable changes only)

1) Chapter 09:
- Add a short, stable note: systems frontier is not intelligence frontier, and link to eval ladder doctrine.

2) Workbench:
- Link the eval ladder tickets as Next actions so the consolidation line is explicit.

3) Theory of Thought:
- Register TOT-H005 and TOT-H006 with tracking links to the new tickets.

