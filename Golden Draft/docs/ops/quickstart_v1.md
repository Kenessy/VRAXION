# Quickstart v1

Goal: a new engineer can run **CPU tests** and understand the **GPU probe harness** in under ~10 minutes.

## Prereqs

- Python: **3.11** recommended.
- OS: Windows is the currently verified development surface.

## 1) Create a virtual environment

From repo root:

```powershell
python -m venv .venv
# If Activate.ps1 is blocked, you can use: Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

## 2) Install minimal dependencies

This repo intentionally does not ship a one-size-fits-all requirements lock yet.

Minimum for tooling/tests:

```powershell
pip install numpy
```

For GPU/torch tooling, install PyTorch for your platform. CPU-only example:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 3) Run the CPU test suite

```powershell
python -m unittest discover -s "Golden Draft/tests" -v
```

## 4) Sanity compile gate

```powershell
python -m compileall "Golden Code" "Golden Draft"
```

## 5) GPU tooling (safe to inspect)

The probe harness (VRA-32) prints help and enforces an overwrite guard for output dirs:

```powershell
python "Golden Draft/tools/gpu_capacity_probe.py" --help
```

The env dump tool (VRA-29) writes `env.json` and is best-effort (works without CUDA):

```powershell
python "Golden Draft/tools/gpu_env_dump.py" --out-dir bench_vault/_tmp/env_dump --precision unknown --amp 0
```

## Common failures (and where to look)

- CUDA missing: expected on CPU machines. GPU probe runs require a working CUDA torch install.
- Windows WDDM stalls / timeouts: see `Golden Draft/docs/gpu/env_lock_v1.md`.
- VRAM oversubscription / paging symptoms: see `Golden Draft/docs/gpu/vram_breakdown_v1.md` and the stability gates in `Golden Draft/docs/gpu/objective_contract_v1.md`.
- Logging sync traps (accidental `cuda.synchronize()` via `.item()` loops): see `Golden Draft/docs/gpu/env_lock_v1.md`.
