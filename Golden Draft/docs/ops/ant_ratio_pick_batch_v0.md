# Ant Ratio Pick Batch v0 (VRA-77)

This tool selects a per-config batch size that targets a fixed reserved VRAM
ratio, while ensuring the probe run stays PASS.

## Why

If you compare configs at a fixed batch, you often compare different VRAM
budgets (and therefore different failure risk). VRA-77 makes the budget axis
explicit by targeting a fixed reserved VRAM ratio band.

## Tool

```
python "Golden Draft/tools/ant_ratio_pick_batch_v0.py"
```

Defaults:
- ant tiers: `small,real,stress`
- colony: `OD1_CANON_REAL`
- expert heads: `1,2,4,8,16` (maps to probe `--out-dim`)
- precision/amp: `fp16` / `1`
- warmup/measure: `5` / `50`
- target ratio: `0.85` (accept band `[0.82, 0.88]`)

Outputs (gitignored):
- `bench_vault/_tmp/vra77_batch_target_v0/<ts>/ant_ratio_batch_targets_v0.json`
- per-run probe artifacts under `<ts>/runs/...`

## PASS/FAIL Rule (Locked)

Never use process exit code to classify PASS/FAIL.

PASS iff:
- `metrics.json.stability_pass == true`
- and `had_oom/had_nan/had_inf == false`

## Selection Algorithm (Kill-fast)

Per config:
1. Start `B=1`. If FAIL, mark config unusable.
2. Double while PASS and `vram_ratio_reserved < accept_low`.
3. If still below target, step up once to try to bracket above target.
4. Binary refine up to 6 extra calls.
5. If non-monotonic behavior is observed, stop refinement and choose the best
   observed PASS closest to target.

Hard cap: `--max-calls` probe invocations per config (default 10).

