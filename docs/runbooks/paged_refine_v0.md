# Paged Refinement v0 (Static Space + Arc Refinement) — Runbook (Doc-only)

This runbook describes the intended **checkpoint-time refinement loop** for VRAXION:

> static keyspace (ring addresses) + local arc refinement (router_map edits) + offline expert growth.

**Important:** this is **documentation only** for the v1.2 sprint. No training/sweeps are executed as part of this change-set.

## Guardrails
- No runtime SSD paging / demand paging in v0/v1.
- Structural edits happen **offline at checkpoint boundaries**.
- Address refinement must be **arc-safe** by default (contiguous arc; parent-owned addresses only).

## Prereqs
- Repo root: `VRAXION/VRAXION`
- Tools:
  - Meta generator: `Golden Draft/tools/mitosis_meta_from_eval.py`
  - Offline split: `Golden Draft/vraxion_mitosis_split.py`
- Checkpoint format:
  - **Recommended for v0:** monolithic `.pt` checkpoints with `head.experts.*` tensors.
  - Note: `vraxion_mitosis_split.py` currently clones `head.experts.<id>.*` and does not support `head.single.*` (1→2) yet.

## Environment (recommended defaults)
These are the minimal knobs to keep the refinement path deterministic and reproducible:

- `VRX_EXPERT_HEADS=2` (ensures multi-expert head layout)
- `VRX_MODULAR_SAVE=1` (optional; for modular checkpoints)
- `VRX_MODULAR_AUTO_EXPERTS=1` (resume derives expert count from `system/router.state`)
- `VRX_MITOSIS=1` (enables mitosis telemetry in eval tooling; optional for v0)

## Artifact layout
Put run artifacts under an ignored temp root:

- `bench_vault/_tmp/paged_refine_v0/<run_id>/`
  - `ckpt_pre_split.pt`
  - `mitosis_meta.json`
  - `ckpt_post_split.pt`

## Step-by-step loop

### 1) Train to a checkpoint (not executed in this sprint)
Run your normal training entrypoint long enough to produce a checkpoint:
- `ckpt_pre_split.pt`

### 2) Generate arc-safe meta
From repo root:

```bash
python "Golden Draft/tools/mitosis_meta_from_eval.py" \
  --checkpoint bench_vault/_tmp/paged_refine_v0/<run_id>/ckpt_pre_split.pt \
  --output bench_vault/_tmp/paged_refine_v0/<run_id>/mitosis_meta.json
```

This produces:
- `hot_arc` (contiguous arc) and
- `hot_addresses` (subset of the arc that are currently owned by `parent_expert`).

### 3) Offline split (checkpoint-time refinement)

```bash
python "Golden Draft/vraxion_mitosis_split.py" \
  --checkpoint bench_vault/_tmp/paged_refine_v0/<run_id>/ckpt_pre_split.pt \
  --output bench_vault/_tmp/paged_refine_v0/<run_id>/ckpt_post_split.pt \
  --meta bench_vault/_tmp/paged_refine_v0/<run_id>/mitosis_meta.json
```

This clones `head.experts.<parent>` into a new expert id and redirects the selected addresses in `router_map`.

### 4) Resume from the post-split checkpoint (not executed in this sprint)
Resume training/eval from `ckpt_post_split.pt` using your standard runner, keeping:
- `VRX_MODULAR_AUTO_EXPERTS=1` (when applicable)

## Success checks (what to look for)
These are the “signals of life” for the refinement loop:
- Parent-expert domination decreases (e.g., `ptr_expert_max_share` drops).
- Metrics do not catastrophically regress immediately after split/resume.
- No stability guardrails trip (step-time explosion / heartbeat stall / VRAM guard, per the objective contract).

## Notes / future work
- **1→2 growth** from `head.single.*` checkpoints is not supported by the current offline split tool.
  - Future ticket: add a compatibility transform to treat `head.single.*` as expert 0 and materialize expert shards for 1→2.
