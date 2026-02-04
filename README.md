# VRAXION Golden Targets

This repo root contains two curated targets:

- `Golden Code/`: end-user ("DVD") runtime library code only.
  - Primary package: `Golden Code/vraxion/`
- `Golden Draft/`: production-quality, non-DVD code (tools, tests, harness).

## Where to look

- Pages (landing): https://kenessy.github.io/VRAXION/
- Wiki (deep dives): https://github.com/Kenessy/VRAXION/wiki
- Roadmap (public): https://github.com/users/Kenessy/projects/4
- Releases (public proof): https://github.com/Kenessy/VRAXION/releases

## Versioning (MAJOR.MINOR.BUILD)

VRAXION uses a simple cadence tracker stored in `VERSION.json`:

- `BUILD` increments on every merged "ticket completion" PR (fast/beta cadence).
- `MINOR` increments only for curated public updates (BUILD unchanged).
- `MAJOR` increments only for lifetime milestones (MINOR resets to 0; BUILD unchanged).

This does not replace the historical release tag `v1.0.0`.

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
