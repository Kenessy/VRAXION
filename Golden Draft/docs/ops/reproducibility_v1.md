# Reproducibility v1

VRAXION performance claims are only useful if they are reproducible. This doc defines the **minimum result packet** for any benchmark/probe claim.

## Result packet (required for any claim)

Always record:

- `git_commit`: exact commit hash (and branch name if relevant)
- `VERSION.json`: the repo cadence version (MAJOR.MINOR.BUILD)
- `env.json`: emitted by `Golden Draft/tools/gpu_env_dump.py`
- `workload_id`: emitted by `Golden Draft/tools/workload_id.py` or by the probe harness
- Full CLI args / mapping notes (prefer `run_cmd.txt` emitted by harnesses)
- Output directory path containing raw artifacts (`metrics.json`, `metrics.csv`, `summary.md`)
- Seed(s) if the run is stochastic

If you cannot provide the above, treat the number as anecdotal.

## Canonical output layout

Use `bench_vault/_tmp/` for local run artifacts (it should remain ignored by git):

- `bench_vault/_tmp/<ticket_or_experiment>/<probe_id_or_run_name>/`
  - `env.json`
  - `run_cmd.txt`
  - `metrics.json`
  - `metrics.csv`
  - `summary.md`

## GPU probe contract

The probe harness is contract-driven:

- Objective/stability contract: `Golden Draft/docs/gpu/objective_contract_v1.md`

If you change what is measured or what constitutes PASS/FAIL, you must:

1) update the contract version or add an explicit contract addendum, and
2) keep backward compatibility for existing metrics consumers.

## Notes on Windows WDDM

Windows WDDM can enter regimes where step-times explode and VRAM behavior becomes misleading (paging/overcommit).
The contract treats these as stability failures, and documentation should call out when a datapoint came from a degraded regime.

See:
- `Golden Draft/docs/gpu/env_lock_v1.md`
- `Golden Draft/docs/gpu/vram_breakdown_v1.md`
