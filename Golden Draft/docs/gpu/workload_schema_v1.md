# GPU Workload Schema v1 (AntSpec + ColonySpec)

This document defines a **contract-first** workload abstraction for VRAXION GPU runs.

- **VRA-31 scope:** define **what workload** is being run (explicit knobs + stable ID).
- **VRA-30 scope:** define **how runs are measured and validated** (objective + stability gates + artifacts).

If you are running GPU capacity/throughput characterization, you must reference:
- Workload definition (this doc): `docs/gpu/workload_schema_v1.md`
- Objective/stability contract (VRA-30): `docs/gpu/objective_contract_v1.md`

> Paths above are relative to `Golden Draft/`. Repo paths are:
> - `Golden Draft/docs/gpu/workload_schema_v1.md`
> - `Golden Draft/docs/gpu/objective_contract_v1.md`

---

## 1) Purpose & Scope

This schema exists to prevent "random" / implicit knob choices (ring length, slot dim, dtype, cadence, batch size, etc.)
from silently changing between runs.

Every workload spec:
- is **strictly validated** (unknown keys are rejected),
- maps to a deterministic, stable **`workload_id`**, and
- can be used as an input to future probe/sweep tooling (VRA-32 / VRA-36+).

This ticket **does not** execute runs or collect GPU data.

---

## 2) Schema: `workload_schema_v1`

### 2.1 Top-level JSON shape

```json
{
  "schema_version": "workload_schema_v1",
  "ant_spec": { "..." : "..." },
  "colony_spec": { "..." : "..." },
  "name": "optional human label",
  "notes": "optional notes"
}
```

`name` / `notes` are allowed but **excluded from the workload ID**.

### 2.2 AntSpec (shape / footprint knobs)

AntSpec captures model/shape and precision choices that materially affect VRAM footprint and compute.

Required keys (and env mapping):

| Key | Type | Meaning | Maps to env |
|---|---:|---|---|
| `ring_len` | int (>=1) | ring length | `VRX_RING_LEN` |
| `slot_dim` | int (>=1) | slot dimension | `VRX_SLOT_DIM` |
| `ptr_dtype` | enum | pointer dtype | `VRX_PTR_DTYPE` |
| `precision` | enum | math precision mode | `VRX_PRECISION` |

Allowed `ptr_dtype` values (aligned with `Golden Code/vraxion/settings.py`): `fp64`, `fp32`, `fp16`, `bf16`.

Allowed `precision` values: `fp64`, `fp32`, `fp16`, `bf16`, `amp`.

**Note on `precision="amp"`:** this selects autocast behavior; the underlying math dtype can be device-dependent.
It is still part of the workload ID because it changes runtime behavior.

Optional keys (allowed but excluded from ID): `name`, `notes`.

### 2.3 ColonySpec (workload / cadence knobs)

ColonySpec captures workload and cadence parameters that materially affect compute.

Required keys (and env mapping):

| Key | Type | Meaning | Maps to env |
|---|---:|---|---|
| `seq_len` | int (>=1) | sequence length used for accounting | (semantic; often matches `synth_len`) |
| `synth_len` | int (>=1) | synthetic sequence length | `VRX_SYNTH_LEN` |
| `batch_size` | int (>=1) | batch size | `VRX_BATCH_SIZE` |
| `ptr_update_every` | int (>=1) | pointer update cadence | `VRX_PTR_UPDATE_EVERY` |
| `state_loop_samples` | int (>=0) | optional additional compute loop | `VRX_STATE_LOOP_SAMPLES` |

Optional keys (allowed but excluded from ID): `name`, `notes`.

---

## 3) Strict validation rules (important)

- Unknown keys at top-level, `ant_spec`, or `colony_spec` are **errors**.
- No implicit coercion:
  - `"8192"` (string) is **not** an int.
  - `" fp32 "` is **not** a valid enum value.
- Validation failures are treated as "invalid workload spec" and must block a run.

---

## 4) Stable `workload_id` computation

The stable ID is computed from:
- `schema_version`
- required keys of `ant_spec`
- required keys of `colony_spec`

Excluded (do not affect ID):
- any `name` / `notes` fields (top-level and nested)

Algorithm (implemented in `tools/workload_id.py`):
1) Canonicalize spec (drop optional fields; validate types/keys/enums).
2) Serialize canonical JSON with:
   - sorted keys
   - separators `(",", ":")`
   - ASCII only (`ensure_ascii=True`)
3) Hash: SHA-256 over UTF-8 bytes
4) ID: `wl_v1_` + first 12 hex chars of the digest

---

## 5) Canonical OD1 templates (small / real / stress)

Templates live in `workloads/` (relative to `Golden Draft/`):
- `workloads/od1_small_v1.json`
- `workloads/od1_real_v1.json` (matches current defaults where defaults exist)
- `workloads/od1_stress_v1.json`

Compute the ID (example):

```bash
cd "Golden Draft"
python tools/workload_id.py workloads/od1_real_v1.json
```

**Note:** `od1_stress_v1.json` may OOM on smaller GPUs. That is expected; it exists to probe the limiter/guardrails.

---

## 6) Relationship to the objective/stability contract (VRA-30)

- This schema standardizes **what workload** is being attempted.
- The objective/stability contract (`docs/gpu/objective_contract_v1.md`) standardizes:
  - what metrics are recorded,
  - what constitutes PASS/FAIL,
  - and which artifacts must be emitted per run.

Future harness/sweep code must use both:
- **Workload schema** → sets knobs + stable IDs
- **Objective contract** → ensures runs are valid and comparable

