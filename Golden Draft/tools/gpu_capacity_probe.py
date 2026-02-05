"""VRA-32 GPU capacity/throughput probe harness.

This tool executes a small, contract-compliant training-like loop to measure:
- throughput (samples/sec + tokens/sec),
- per-step latency stats (median/p95),
- peak VRAM reserved/allocated (CUDA),
and emits required artifacts even on FAIL.

Contracts:
- Objective/stability: docs/gpu/objective_contract_v1.md (relative to Golden Draft/)
- Workload schema:     docs/gpu/workload_schema_v1.md (relative to Golden Draft/)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


CONTRACT_PATH_REL_GOLDEN_DRAFT = "docs/gpu/objective_contract_v1.md"
CONTRACT_REPO_PATH = "Golden Draft/docs/gpu/objective_contract_v1.md"
WORKLOAD_SCHEMA_PATH_REL_GOLDEN_DRAFT = "docs/gpu/workload_schema_v1.md"
WORKLOAD_SCHEMA_REPO_PATH = "Golden Draft/docs/gpu/workload_schema_v1.md"

ART_RUN_CMD = "run_cmd.txt"
ART_ENV = "env.json"
ART_METRICS_JSON = "metrics.json"
ART_METRICS_CSV = "metrics.csv"
ART_SUMMARY = "summary.md"

V1_GUARD_FILES = (ART_ENV, ART_METRICS_JSON, ART_SUMMARY)


def _bootstrap_import_path() -> None:
    """Ensure Golden Draft + Golden Code are importable for standalone runs."""

    draftr = Path(__file__).resolve().parents[1]
    reproo = draftr.parent

    if str(draftr) not in sys.path:
        sys.path.insert(0, str(draftr))

    candls: list[str] = []
    for keystr in ("VRAXION_GOLDEN_SRC", "GOLDEN_CODE_ROOT", "GOLDEN_CODE_PATH", "GOLDEN_CODE_DIR"):
        envval = os.environ.get(keystr)
        if envval:
            candls.append(envval)

    candls.append(str(reproo / "Golden Code"))
    candls.append(r"S:\AI\Golden Code")
    candls.append(r"S:/AI/Golden Code")

    for candpt in candls:
        try:
            if candpt and os.path.isdir(candpt):
                if candpt not in sys.path:
                    sys.path.insert(0, candpt)
                break
        except OSError:
            continue


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmppth = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmppth, str(path))
        tmppth = ""
    finally:
        if tmppth:
            try:
                os.remove(tmppth)
            except FileNotFoundError:
                pass


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmppth = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=True, sort_keys=True, indent=2)
            f.write("\n")
        os.replace(tmppth, str(path))
        tmppth = ""
    finally:
        if tmppth:
            try:
                os.remove(tmppth)
            except FileNotFoundError:
                pass


def _csv_cell(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (bool, int, float, str)):
        return str(val)
    # Lists/dicts: JSON encode to keep schema round-trippable.
    return json.dumps(val, ensure_ascii=True, sort_keys=True)


def _atomic_write_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(row.keys())
    fd, tmppth = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            w = csv.writer(f)
            w.writerow(keys)
            w.writerow([_csv_cell(row[k]) for k in keys])
        os.replace(tmppth, str(path))
        tmppth = ""
    finally:
        if tmppth:
            try:
                os.remove(tmppth)
            except FileNotFoundError:
                pass


def compute_stall_threshold_s(median_step_time_s: float) -> float:
    """Contract stall threshold: max(60s, 10x median step time)."""

    try:
        med = float(median_step_time_s)
    except Exception:
        med = 0.0
    if not math.isfinite(med) or med <= 0.0:
        med = 0.0
    return max(60.0, 10.0 * med)


def _parse_json_str(raw: str) -> Any:
    return json.loads(raw)


def _load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _preset_to_path(preset: str, *, draft_root: Path) -> Optional[Path]:
    key = preset.strip().lower()
    presets = {
        "od1_canon_small": draft_root / "workloads" / "od1_small_v1.json",
        "od1_canon_real": draft_root / "workloads" / "od1_real_v1.json",
        "od1_canon_stress": draft_root / "workloads" / "od1_stress_v1.json",
    }
    return presets.get(key)


def _load_spec_obj(arg: str, *, draft_root: Path) -> Any:
    pth = _preset_to_path(arg, draft_root=draft_root)
    if pth is not None:
        return _load_json_file(pth)

    cand = Path(arg)
    try:
        if cand.exists() and cand.is_file():
            return _load_json_file(cand)
    except OSError:
        # Strings like inline JSON can be invalid as a filesystem path on Windows.
        pass

    return _parse_json_str(arg)


def _extract_subspec(obj: Any, *, kind: str) -> Mapping[str, Any]:
    """Return either a subspec (ant_spec/colony_spec) or obj itself if already a subspec."""

    if not isinstance(obj, dict):
        raise ValueError(f"--{kind} must decode to a JSON object.")

    # If a full workload object is provided, extract the relevant subobject.
    if "schema_version" in obj and "ant_spec" in obj and "colony_spec" in obj:
        sub = obj.get(f"{kind}_spec")
        if not isinstance(sub, dict):
            raise ValueError(f"Full workload JSON missing {kind}_spec object.")
        return sub

    # Otherwise treat as a subspec.
    return obj


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gpu_capacity_probe",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "VRA-32 GPU capacity/throughput probe harness.\n"
            f"Contract (relative to Golden Draft/): {CONTRACT_PATH_REL_GOLDEN_DRAFT}\n"
            f"Repo path: {CONTRACT_REPO_PATH}\n"
            f"Workload schema (relative to Golden Draft/): {WORKLOAD_SCHEMA_PATH_REL_GOLDEN_DRAFT}\n"
            f"Repo path: {WORKLOAD_SCHEMA_REPO_PATH}"
        ),
        epilog=f"See {CONTRACT_PATH_REL_GOLDEN_DRAFT}",
    )

    p.add_argument("--ant", required=True, help="Ant spec: preset|json file|inline json")
    p.add_argument("--colony", required=True, help="Colony spec: preset|json file|inline json")
    p.add_argument("--out-dim", required=True, type=int, help="Expert head count (EXPERT_HEADS).")
    p.add_argument("--batch", required=True, type=int, help="Batch size (overrides colony_spec.batch_size).")
    p.add_argument("--warmup-steps", required=True, type=int)
    p.add_argument("--measure-steps", required=True, type=int)
    p.add_argument("--precision", required=True, choices=("fp32", "bf16", "fp16"))
    p.add_argument("--amp", required=True, type=int, choices=(0, 1))
    p.add_argument("--output-dir", required=True, help="Output directory for artifacts.")

    # Debug-only watchdog self-test knobs (default: disabled).
    p.add_argument("--debug-stall-after-step", type=int, default=-1, help="Inject a stall after this measured step. -1 disables.")
    p.add_argument("--debug-stall-s", type=float, default=0.0, help="Seconds to sleep when debug stall triggers.")
    p.add_argument(
        "--debug-stall-threshold-s",
        type=float,
        default=0.0,
        help="Override heartbeat stall threshold for debug self-test (seconds). 0 uses contract threshold.",
    )

    return p.parse_args(list(argv) if argv is not None else None)


def _validate_precision_amp(precision: str, amp: int) -> None:
    if precision == "fp32" and amp != 0:
        raise ValueError("precision=fp32 requires --amp 0")
    if precision in ("fp16", "bf16") and amp != 1:
        raise ValueError(f"precision={precision} requires --amp 1")


def _resolve_device(torch_mod: Any) -> str:
    if os.environ.get("VRX_FORCE_DEVICE", "").strip().lower() == "cpu":
        return "cpu"
    try:
        if bool(torch_mod.cuda.is_available()):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _torch_dtype_from_precision(torch_mod: Any, precision: str) -> Any:
    if precision == "fp32":
        return torch_mod.float32
    if precision == "fp16":
        return torch_mod.float16
    if precision == "bf16":
        return torch_mod.bfloat16
    raise ValueError(f"Unknown precision: {precision!r}")


def _torch_dtype_from_ptr_dtype(torch_mod: Any, ptr_dtype: str) -> Any:
    mapping = {
        "fp64": torch_mod.float64,
        "fp32": torch_mod.float32,
        "fp16": torch_mod.float16,
        "bf16": torch_mod.bfloat16,
    }
    if ptr_dtype not in mapping:
        raise ValueError(f"Unknown ptr_dtype: {ptr_dtype!r}")
    return mapping[ptr_dtype]


class _Heartbeat:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.armed = False
        self.last_progress_monotonic = time.monotonic()
        self.stall_threshold_s = 60.0
        self.stall_detected = False

    def arm(self, *, stall_threshold_s: float) -> None:
        with self.lock:
            self.armed = True
            self.stall_threshold_s = float(stall_threshold_s)
            self.last_progress_monotonic = time.monotonic()

    def update_progress(self) -> None:
        with self.lock:
            self.last_progress_monotonic = time.monotonic()

    def snapshot(self) -> tuple[bool, float, float, bool]:
        with self.lock:
            age = time.monotonic() - self.last_progress_monotonic
            return self.armed, float(self.stall_threshold_s), float(age), bool(self.stall_detected)

    def mark_stalled(self) -> None:
        with self.lock:
            self.stall_detected = True


def main(argv: Optional[Sequence[str]] = None) -> int:
    _bootstrap_import_path()

    # Parse args (invalid args must exit 2, without writing artifacts).
    try:
        args = _parse_args(argv)
        _validate_precision_amp(args.precision, int(args.amp))
        if int(args.out_dim) < 1:
            raise ValueError("--out-dim must be >= 1")
        if int(args.batch) < 1:
            raise ValueError("--batch must be >= 1")
        if int(args.warmup_steps) < 0:
            raise ValueError("--warmup-steps must be >= 0")
        if int(args.measure_steps) < 1:
            raise ValueError("--measure-steps must be >= 1")
        if float(args.debug_stall_s) < 0.0:
            raise ValueError("--debug-stall-s must be >= 0")
        if float(args.debug_stall_threshold_s) < 0.0:
            raise ValueError("--debug-stall-threshold-s must be >= 0")
        if int(args.debug_stall_after_step) >= 0:
            if float(args.debug_stall_s) <= 0.0:
                raise ValueError("--debug-stall-after-step requires --debug-stall-s > 0")
            if float(args.debug_stall_threshold_s) <= 0.0:
                raise ValueError("--debug-stall-after-step requires --debug-stall-threshold-s > 0")
            if int(args.debug_stall_after_step) >= int(args.measure_steps):
                raise ValueError("--debug-stall-after-step must be < --measure-steps")
        else:
            # Avoid ambiguous debug settings (threshold override without an injected stall).
            if float(args.debug_stall_s) > 0.0 or float(args.debug_stall_threshold_s) > 0.0:
                raise ValueError("--debug-stall-s/--debug-stall-threshold-s require --debug-stall-after-step >= 0")
    except Exception as exc:
        print(f"ERROR: invalid args: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"ERROR: cannot create output dir: {exc}", file=sys.stderr)
        return 2

    # Overwrite guard.
    for name in V1_GUARD_FILES:
        if (out_dir / name).exists():
            print(f"ERROR: overwrite guard: {name} already exists in output dir", file=sys.stderr)
            return 2

    # Import stdlib-only dependencies for strict spec + env dump.
    from tools import gpu_env_dump, workload_id

    draft_root = Path(__file__).resolve().parents[1]

    # Resolve and validate workload spec.
    try:
        ant_obj = _extract_subspec(_load_spec_obj(args.ant, draft_root=draft_root), kind="ant")
        col_obj = _extract_subspec(_load_spec_obj(args.colony, draft_root=draft_root), kind="colony")

        # Build full workload spec and apply CLI overrides.
        spec = {
            "schema_version": workload_id.SCHEMA_VERSION,
            "ant_spec": dict(ant_obj),
            "colony_spec": dict(col_obj),
        }
        spec["colony_spec"]["batch_size"] = int(args.batch)
        spec["ant_spec"]["precision"] = str(args.precision)

        canon = workload_id.canonicalize_spec(spec)
        wl_id = workload_id.compute_workload_id(canon)
    except Exception as exc:
        print(f"ERROR: invalid workload spec: {exc}", file=sys.stderr)
        return 2

    probe_id = f"{wl_id}__E{int(args.out_dim)}__{args.precision}_amp{int(args.amp)}"
    debug_self_test = int(args.debug_stall_after_step) >= 0

    # Write pre-run artifacts (must happen before any compute).
    try:
        cmd_text = {
            "argv": list(sys.argv if argv is None else argv),
            "contract_rel_golden_draft": CONTRACT_PATH_REL_GOLDEN_DRAFT,
            "contract_repo_path": CONTRACT_REPO_PATH,
            "workload_schema_rel_golden_draft": WORKLOAD_SCHEMA_PATH_REL_GOLDEN_DRAFT,
            "workload_schema_repo_path": WORKLOAD_SCHEMA_REPO_PATH,
            "resolved_spec": spec,
            "canonical_spec": canon,
            "workload_id": wl_id,
            "probe_id": probe_id,
            "notes": [
                "seq_len is accounting (tokens/sec uses seq_len).",
                "synth_len is generated input length (x uses synth_len).",
                "out_dim is not part of workload_schema_v1; do not compare across out_dim without probe_id.",
            ],
        }
        if debug_self_test:
            cmd_text["notes"].append(
                "DEBUG: watchdog self-test enabled "
                f"(stall_after_step={int(args.debug_stall_after_step)}, "
                f"stall_s={float(args.debug_stall_s)}, "
                f"stall_threshold_s={float(args.debug_stall_threshold_s)}). "
                "Not a rankable datapoint."
            )
        _atomic_write_text(out_dir / ART_RUN_CMD, json.dumps(cmd_text, ensure_ascii=True, sort_keys=True, indent=2) + "\n")
        env_path = gpu_env_dump.write_env_json(out_dir=out_dir, precision=args.precision, amp=int(args.amp))
        env_obj = json.loads(env_path.read_text(encoding="utf-8"))
        total_vram_bytes = env_obj.get("total_vram_bytes")
        if not isinstance(total_vram_bytes, int):
            total_vram_bytes = None
    except Exception as exc:
        print(f"ERROR: failed to write pre-run artifacts: {exc}", file=sys.stderr)
        return 2

    # Initialize metrics with required contract keys (must always be present).
    metrics: dict[str, Any] = {
        "batch_size": int(canon["colony_spec"]["batch_size"]),
        "seq_len": int(canon["colony_spec"]["seq_len"]),
        "warmup_steps": int(args.warmup_steps),
        "measure_steps": int(args.measure_steps),
        "measure_wall_time_s": None,
        "median_step_time_s": None,
        "p95_step_time_s": None,
        "throughput_samples_per_s": None,
        "throughput_tokens_per_s": None,
        "peak_vram_reserved_bytes": None,
        "peak_vram_allocated_bytes": None,
        "had_oom": False,
        "had_nan": False,
        "had_inf": False,
        "stability_pass": True,
        "fail_reasons": [],
        # Extras (allowed)
        "workload_id": wl_id,
        "probe_id": probe_id,
        "synth_len": int(canon["colony_spec"]["synth_len"]),
        "out_dim": int(args.out_dim),
        "device": None,
        "precision": str(args.precision),
        "amp": int(args.amp),
        "vram_guard_ratio": 0.92,
        "heartbeat_stall_detected": False,
        "heartbeat_stall_threshold_s": None,
        "heartbeat_last_progress_age_s": None,
        "step_time_mode": None,
        "runtime_exception_msg": None,
        "debug_stall_after_step": int(args.debug_stall_after_step),
        "debug_stall_s": float(args.debug_stall_s),
        "debug_stall_threshold_s": float(args.debug_stall_threshold_s),
    }

    write_lock = threading.Lock()
    wrote_post = False

    hb = _Heartbeat()

    def write_post_artifacts() -> None:
        nonlocal wrote_post
        with write_lock:
            if wrote_post:
                return
            _atomic_write_json(out_dir / ART_METRICS_JSON, metrics)
            _atomic_write_csv(out_dir / ART_METRICS_CSV, metrics)

            status = "PASS" if bool(metrics.get("stability_pass")) else "FAIL"
            reasons = metrics.get("fail_reasons") or []
            reasons_str = ", ".join(str(x) for x in reasons) if reasons else "(none)"
            summ = (
                f"# GPU Capacity Probe Summary (VRA-32)\n\n"
                f"- status: {status}\n"
                f"- probe_id: {probe_id}\n"
                f"- workload_id: {wl_id}\n"
                f"- device: {metrics.get('device')}\n"
                f"- precision/amp: {args.precision} / {int(args.amp)}\n"
                f"- out_dim: {int(args.out_dim)}\n"
                f"- batch_size: {metrics.get('batch_size')}\n"
                f"- seq_len (accounting): {metrics.get('seq_len')}\n"
                f"- synth_len (generated): {metrics.get('synth_len')}\n"
                f"- throughput_samples_per_s: {metrics.get('throughput_samples_per_s')}\n"
                f"- throughput_tokens_per_s: {metrics.get('throughput_tokens_per_s')}\n"
                f"- median_step_time_s: {metrics.get('median_step_time_s')}\n"
                f"- p95_step_time_s: {metrics.get('p95_step_time_s')}\n"
                f"- peak_vram_reserved_bytes: {metrics.get('peak_vram_reserved_bytes')}\n"
                f"- peak_vram_allocated_bytes: {metrics.get('peak_vram_allocated_bytes')}\n"
                f"- fail_reasons: {reasons_str}\n\n"
                f"Contract: {CONTRACT_REPO_PATH}\n"
            )
            if debug_self_test:
                summ += (
                    "\nNOTE: debug watchdog self-test enabled "
                    f"(stall_after_step={int(args.debug_stall_after_step)}, "
                    f"stall_s={float(args.debug_stall_s)}, "
                    f"stall_threshold_s={float(args.debug_stall_threshold_s)}).\n"
                )
            _atomic_write_text(out_dir / ART_SUMMARY, summ)
            wrote_post = True

    def watchdog_loop() -> None:
        while True:
            armed, thresh, age, stalled = hb.snapshot()
            if stalled:
                return
            if armed and age > thresh:
                hb.mark_stalled()
                metrics["heartbeat_stall_detected"] = True
                metrics["heartbeat_stall_threshold_s"] = float(thresh)
                metrics["heartbeat_last_progress_age_s"] = float(age)
                metrics["stability_pass"] = False
                fr = metrics.get("fail_reasons")
                if isinstance(fr, list):
                    fr.append("heartbeat_stall")
                try:
                    write_post_artifacts()
                finally:
                    # Exit 0 to honor "artifacts written => success exit code" policy.
                    os._exit(0)
            time.sleep(0.5)

    thr = threading.Thread(target=watchdog_loop, name="vra32_watchdog", daemon=True)
    thr.start()

    # Main run (best-effort: always write post artifacts even on failure).
    try:
        import torch
        import torch.nn.functional as F

        from vraxion.instnct import absolute_hallway
        from vraxion.instnct.absolute_hallway import AbsoluteHallway

        device = _resolve_device(torch)
        metrics["device"] = device

        # "precision" is treated as compute precision (autocast dtype) for the probe run.
        # Keep weights in fp32 to match typical training behavior (stable Adam + GradScaler on fp16).
        compute_dtype = _torch_dtype_from_precision(torch, args.precision)
        param_dtype = torch.float32
        ptr_dtype = _torch_dtype_from_ptr_dtype(torch, str(canon["ant_spec"]["ptr_dtype"]))

        # Init-time hook: must be set before model construction.
        absolute_hallway.EXPERT_HEADS = int(args.out_dim)

        # Make state_loop_samples a real knob: AbsoluteHallway only uses it when STATE_LOOP_METRICS is enabled.
        # Enable it iff the workload requests it, and set globals before model construction.
        state_loop_samples = int(canon["colony_spec"]["state_loop_samples"])
        absolute_hallway.STATE_LOOP_METRICS = bool(state_loop_samples > 0)
        absolute_hallway.STATE_LOOP_SAMPLES = int(max(0, state_loop_samples))

        model = AbsoluteHallway(
            input_dim=1,
            num_classes=256,
            ring_len=int(canon["ant_spec"]["ring_len"]),
            slot_dim=int(canon["ant_spec"]["slot_dim"]),
        )
        model.train(True)
        model.to(device=device)

        # Keep weights in fp32 (stable optimizer math). If this fails, there's no meaningful probe to run.
        model.to(dtype=param_dtype)

        # Forward-time hook: pointer dtype consulted in forward.
        absolute_hallway.PTR_DTYPE = ptr_dtype

        model.ptr_update_every = int(canon["colony_spec"]["ptr_update_every"])
        model.state_loop_samples = int(max(0, state_loop_samples))

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        B = int(canon["colony_spec"]["batch_size"])
        synth_len = int(canon["colony_spec"]["synth_len"])
        x = torch.randn(B, synth_len, 1, device=device, dtype=param_dtype)
        y = torch.randint(0, 256, (B,), device=device, dtype=torch.long)

        use_cuda = device == "cuda"

        autocast_ctx = nullcontext()
        if use_cuda and int(args.amp) == 1:
            autocast_ctx = torch.autocast(device_type="cuda", enabled=True, dtype=compute_dtype)

        scaler = None
        if use_cuda and int(args.amp) == 1 and args.precision == "fp16":
            scaler = torch.cuda.amp.GradScaler()

        def train_step() -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                out = model(x)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                # Force fp32 loss math for stability under autocast.
                loss = F.cross_entropy(logits.float(), y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            return loss

        def step_times_s_warmup() -> list[float]:
            durs: list[float] = []
            if use_cuda:
                metrics["step_time_mode"] = "cuda_events"
                ev_pairs: list[tuple[Any, Any]] = []
                for _ in range(int(args.warmup_steps)):
                    st = torch.cuda.Event(enable_timing=True)
                    en = torch.cuda.Event(enable_timing=True)
                    st.record()
                    loss = train_step()
                    en.record()
                    hb.update_progress()
                    if not torch.isfinite(loss):
                        if torch.isnan(loss):
                            metrics["had_nan"] = True
                        if torch.isinf(loss):
                            metrics["had_inf"] = True
                        metrics["stability_pass"] = False
                        metrics["fail_reasons"].append("nan_or_inf")
                        break
                    ev_pairs.append((st, en))
                torch.cuda.synchronize()
                for st, en in ev_pairs:
                    durs.append(float(st.elapsed_time(en)) / 1000.0)
                return durs
            else:
                metrics["step_time_mode"] = "perf_counter"
                for _ in range(int(args.warmup_steps)):
                    t0 = time.perf_counter()
                    loss = train_step()
                    t1 = time.perf_counter()
                    hb.update_progress()
                    if not torch.isfinite(loss):
                        if torch.isnan(loss):
                            metrics["had_nan"] = True
                        if torch.isinf(loss):
                            metrics["had_inf"] = True
                        metrics["stability_pass"] = False
                        metrics["fail_reasons"].append("nan_or_inf")
                        break
                    durs.append(t1 - t0)
                return durs

        # Warmup.
        warm_durs = step_times_s_warmup()
        warm_med = float(sorted(warm_durs)[len(warm_durs) // 2]) if warm_durs else 0.1
        stall_thresh_s = compute_stall_threshold_s(warm_med)
        if debug_self_test and float(args.debug_stall_threshold_s) > 0.0:
            stall_thresh_s = float(args.debug_stall_threshold_s)
        hb.arm(stall_threshold_s=stall_thresh_s)
        metrics["heartbeat_stall_threshold_s"] = float(hb.stall_threshold_s)

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Measure window walltime.
        if use_cuda:
            torch.cuda.synchronize()
        wall_t0 = time.perf_counter()

        step_durs_s: list[float] = []
        ev_pairs: list[tuple[Any, Any]] = []
        measured_ok = True

        for i in range(int(args.measure_steps)):
            if use_cuda:
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record()
                loss = train_step()
                en.record()
                ev_pairs.append((st, en))
            else:
                t0 = time.perf_counter()
                loss = train_step()
                t1 = time.perf_counter()
                step_durs_s.append(t1 - t0)

            hb.update_progress()

            if debug_self_test and i == int(args.debug_stall_after_step):
                time.sleep(float(args.debug_stall_s))

            if not torch.isfinite(loss):
                if torch.isnan(loss):
                    metrics["had_nan"] = True
                if torch.isinf(loss):
                    metrics["had_inf"] = True
                metrics["stability_pass"] = False
                metrics["fail_reasons"].append("nan_or_inf")
                measured_ok = False
                break

            # Update stall threshold once we have some measured durations.
            if i == 4 and not debug_self_test:
                if use_cuda:
                    # will be computed after sync; leave warm threshold.
                    pass
                else:
                    med5 = float(sorted(step_durs_s)[len(step_durs_s) // 2])
                    hb.arm(stall_threshold_s=compute_stall_threshold_s(med5))
                    metrics["heartbeat_stall_threshold_s"] = float(hb.stall_threshold_s)

        if use_cuda:
            torch.cuda.synchronize()
        wall_t1 = time.perf_counter()

        if use_cuda:
            for st, en in ev_pairs:
                step_durs_s.append(float(st.elapsed_time(en)) / 1000.0)

        # Ensure timings are complete.
        if len(step_durs_s) != int(args.measure_steps) or not measured_ok:
            metrics["stability_pass"] = False
            metrics["fail_reasons"].append("timing_missing")

        # Compute timing stats.
        if step_durs_s:
            srt = sorted(step_durs_s)
            med = float(srt[len(srt) // 2])
            p95 = float(srt[int(0.95 * (len(srt) - 1))])
            metrics["median_step_time_s"] = med
            metrics["p95_step_time_s"] = p95

            if med > 0 and p95 > 2.5 * med:
                metrics["stability_pass"] = False
                metrics["fail_reasons"].append("step_time_explosion")

        # Walltime + throughput.
        if measured_ok and int(args.measure_steps) > 0:
            wall = float(wall_t1 - wall_t0)
            metrics["measure_wall_time_s"] = wall
            if wall > 0:
                sps = (int(args.measure_steps) * int(metrics["batch_size"])) / wall
                metrics["throughput_samples_per_s"] = float(sps)
                metrics["throughput_tokens_per_s"] = float(sps) * float(metrics["seq_len"])

        # VRAM peaks.
        if use_cuda:
            try:
                metrics["peak_vram_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
                metrics["peak_vram_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                metrics["stability_pass"] = False
                metrics["fail_reasons"].append("vram_stats_failed")

        # VRAM guard.
        if isinstance(total_vram_bytes, int) and isinstance(metrics["peak_vram_reserved_bytes"], int):
            if metrics["peak_vram_reserved_bytes"] > int(metrics["vram_guard_ratio"] * total_vram_bytes):
                metrics["stability_pass"] = False
                metrics["fail_reasons"].append("vram_guard")

    except Exception as exc:
        # Catch-all: still write artifacts and exit 0.
        msg = f"{type(exc).__name__}: {exc}"
        metrics["stability_pass"] = False
        metrics["fail_reasons"].append("runtime_exception")
        metrics["runtime_exception_msg"] = msg

        # Heuristic OOM classification.
        low = msg.lower()
        if "out of memory" in low or "cuda oom" in low or "cuda out of memory" in low:
            metrics["had_oom"] = True
            metrics["fail_reasons"].append("oom")

    # Heartbeat final snapshot.
    armed, thresh, age, stalled = hb.snapshot()
    metrics["heartbeat_stall_threshold_s"] = float(thresh) if armed else metrics["heartbeat_stall_threshold_s"]
    metrics["heartbeat_last_progress_age_s"] = float(age)
    metrics["heartbeat_stall_detected"] = bool(stalled) or bool(metrics.get("heartbeat_stall_detected"))

    # Final write.
    try:
        write_post_artifacts()
    except Exception as exc:
        print(f"ERROR: failed to write artifacts: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
