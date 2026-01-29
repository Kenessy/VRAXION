# boot_synth_markov0

Goal: a minimal, deterministic "proof-of-boot" benchmark that exercises the
full training+eval stack without downloading datasets.

Task (synthetic):
- Input: a binary token sequence `x[t] in {0,1}` with shape `[B,T,1]`
- Target: the last token `y = x[T-1]` ("markov0")

What it validates:
- settings/env parsing
- data loader plumbing (synthetic mode)
- model forward/backward (`AbsoluteHallway`)
- bounded-step training loop (`train_steps`)
- evaluation loop + metrics (`eval_model`)
- stable run artifact layout in `bench_vault/` (git-ignored)

## How to run

From repo root (PowerShell):

```powershell
$env:PYTHONUNBUFFERED = 1
python -u \"Golden Draft/benchmarks/boot_synth_markov0/run_boot_synth_markov0.py\"
```

The script writes a self-contained run directory under:
`bench_vault/benchmarks/boot_synth_markov0/<timestamp>/`

To avoid overriding existing env vars inside the process:

```powershell
python -u \"Golden Draft/benchmarks/boot_synth_markov0/run_boot_synth_markov0.py\" --respect-env
```

## Defaults (can be overridden)

The script sets conservative defaults (in-process only; it does NOT modify your
shell permanently). You can override via CLI flags:
- `VAR_COMPUTE_DEVICE=cpu`
- `VRX_RING_LEN=256`, `VRX_SLOT_DIM=128`
- `VRX_SYNTH=1`, `VRX_SYNTH_MODE=markov0`
- `steps=600`, `batch=32`, `seq_len=128`

Pass signal (informal): `eval_acc` should rise well above chance (>0.90) on CPU
within the configured step budget.
