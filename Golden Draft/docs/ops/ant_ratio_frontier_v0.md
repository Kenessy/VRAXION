# Ant Ratio Frontier v0 (VRA-76 to VRA-78)

This is the first "optimum surface" dataset for VRAXION OD1: it joins three axes
into one packet per configuration:

- Cost: reserved VRAM ratio (from the probe harness artifacts)
- Speed: throughput tokens/sec (from the probe harness artifacts)
- Capability: assoc_byte disjoint accuracy (from assoc eval `report.json`)

The output is:

- JSONL packets: `ant_ratio_packets.jsonl`
- Summary table: `ant_ratio_summary.csv`
- Interactive plot: `ant_ratio_frontier_v0.html` (single HTML file)

Derived columns in `ant_ratio_summary.csv`:

- `ant_body_cells`: `ant_ring_len * ant_slot_dim` mirrored from packet fields.
- `ant_body_scale_vs_small`: normalized body size where small preset (`2048*256`) is `1.0`.
- `vram_target_abs_error`: `abs(vram_ratio_reserved - 0.85)` to rank closeness to target budget.
- `tokens_per_vram_ratio`: throughput efficiency `throughput_tokens_per_s / vram_ratio_reserved`.
- `assoc_acc_per_token_per_s`: capability per speed unit `assoc_byte_disjoint_accuracy / throughput_tokens_per_s`.
- `assoc_acc_per_vram_ratio`: capability per allocator-budget unit `assoc_byte_disjoint_accuracy / vram_ratio_reserved`.

Important invariants

- PASS/FAIL is read from artifacts (not process exit code):
  - Probe PASS iff `metrics.json.stability_pass == true` and no `had_oom/had_nan/had_inf`.
- Run artifacts always go under repo-root `bench_vault/_tmp/...` (gitignored).
- Capability runs use a fixed token budget, not a fixed step budget:
  - `steps = floor(TOKEN_BUDGET / (batch * seq_len))` clamped to `[min_steps, max_steps]`.

Workflow (v0)

1. Pick batches that land near a reserved VRAM target ratio (VRA-77):
   - This runs the probe harness many times and writes a batch-target JSON.
   - Command:
     - `python "Golden Draft/tools/ant_ratio_pick_batch_v0.py"`
   - Output:
     - `bench_vault/_tmp/vra77_batch_target_v0/<ts>/ant_ratio_batch_targets_v0.json`

2. Run one datapoint per configuration and emit packets + plot (VRA-78):
   - Provide the batch-target JSON from step 1.
   - Command (example):
     - `python "Golden Draft/tools/ant_ratio_sweep_v0.py" --batch-targets "bench_vault/_tmp/vra77_batch_target_v0/<ts>/ant_ratio_batch_targets_v0.json"`
   - Output root:
     - `bench_vault/_tmp/vra78_ant_ratio_sweep_v0/<ts>/`
     - `bench_vault/_tmp/vra78_ant_ratio_sweep_v0/<ts>/ant_ratio_packets.jsonl`
     - `bench_vault/_tmp/vra78_ant_ratio_sweep_v0/<ts>/ant_ratio_summary.csv`
     - `bench_vault/_tmp/vra78_ant_ratio_sweep_v0/<ts>/ant_ratio_frontier_v0.html`

3. Rebuild the plot from packets (optional):
   - Command:
     - `python "Golden Draft/tools/ant_ratio_plot_v0.py" --packets <jsonl> --out <html>`

Interpreting the plot

- X (lower is better): `vram_ratio_reserved`
- Y (higher is better): `throughput_tokens_per_s`
- Z (higher is better): `assoc_byte_disjoint_accuracy`
- Color: `ant_tier` (small/real/stress)
- Marker size: `expert_heads` (out_dim)
- Marker symbol: PASS circle, FAIL x

Notes

- This is an OD1-first v0. It is not a general benchmarking suite.
- Avoid comparing capability numbers across different token budgets or eval splits.
- For ranking candidates into a "next sweep shortlist", use:
  - primary filter: PASS configs near target VRAM (`vram_target_abs_error` small),
  - tie-break 1: higher `tokens_per_vram_ratio`,
  - tie-break 2: higher `assoc_acc_per_vram_ratio`.
