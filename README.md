# VRAXION Golden Targets

This repo root contains two curated targets:

- `Golden Code/`: end-user ("DVD") runtime library code only.
  - Primary package: `Golden Code/vraxion/`
- `Golden Draft/`: production-quality, non-DVD code (tools, tests, harness).

## Quick commands

From `Golden Draft/`:

```powershell
python vraxion_run.py
python VRAXION_INFINITE.py
python tools/eval_only.py
python -m unittest discover -s tests -v
```

Sanity compile gate:

```powershell
python -m compileall "Golden Code" "Golden Draft"
```

## Naming conventions

- Runtime env vars use the `VRX_` prefix.
- Legacy naming (`prime_c19`, `tournament_phase6`, `TP6_*`) is intentionally removed from the active code surface.
