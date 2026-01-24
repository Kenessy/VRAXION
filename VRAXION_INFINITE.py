import os
import subprocess
import sys
import time

# --- VRAXION INFINITE ENGINE WRAPPER v1.0 ---
# Runs the kernel indefinitely and restarts on crash.

ROOT = os.path.dirname(os.path.abspath(__file__))


def _run_probe(batch_size: int, steps: int) -> tuple[bool, float, str]:
    env = os.environ.copy()
    env["TP6_BATCH_SIZE"] = str(batch_size)
    env["TP6_MAX_STEPS"] = str(steps)
    env["TP6_SAVE_EVERY_STEPS"] = "0"
    env["TP6_RESUME"] = "0"
    env["TP6_IGNORE_MAX_STEPS"] = "0"
    env["TP6_SYNTH_ONCE"] = "1"
    env["TP6_IGNORE_WALL_CLOCK"] = "0"
    env["TP6_WALL"] = str(max(60, steps * 30))
    env["TP6_CKPT"] = os.path.join("checkpoints", "autotune", f"auto_bs{batch_size}.pt")
    env["VAR_LOGGING_PATH"] = os.path.join("logs", f"autotune_bs{batch_size}.log")
    log_path = env["VAR_LOGGING_PATH"]
    abs_log = log_path if os.path.isabs(log_path) else os.path.join(ROOT, log_path)
    if os.path.exists(abs_log):
        os.remove(abs_log)
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "tournament_phase6.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        return True, elapsed, output
    if "out of memory" in output.lower():
        return False, elapsed, "oom"
    return False, elapsed, output


def autotune_batch_size() -> None:
    if os.environ.get("TP6_AUTO_TUNE") != "1":
        return
    base = int(os.environ.get("TP6_BATCH_SIZE", "152"))
    max_batch = int(os.environ.get("TP6_AUTO_TUNE_MAX", str(base * 4)))
    steps = int(os.environ.get("TP6_AUTO_TUNE_STEPS", "20"))
    os.makedirs(os.path.join(ROOT, "checkpoints", "autotune"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    candidates = []
    bs = base
    while bs <= max_batch:
        candidates.append(bs)
        next_bs = int(bs * 1.5)
        if next_bs <= bs:
            next_bs = bs + 1
        bs = next_bs
    best = base
    for bs in candidates:
        ok, _, info = _run_probe(bs, steps)
        if ok:
            best = bs
            continue
        if info == "oom":
            break
    os.environ["TP6_BATCH_SIZE"] = str(best)


def bootstrap_env():
    # Hardcode env to avoid shell-quoting failures.
    os.environ["TP6_RESUME"] = "0"
    os.environ["TP6_CKPT"] = "checkpoints/expert_infinite/expert_inf.pt"
    os.environ["VAR_LOGGING_PATH"] = "logs/expert_infinite.log"
    os.environ["TP6_EXPERT_HEADS"] = "16"
    os.environ["TP6_PRECISION"] = "bf16"
    os.environ["TP6_SYNTH"] = "1"
    os.environ["TP6_SYNTH_MODE"] = "assoc_byte"
    os.environ["TP6_SYNTH_LEN"] = "512"
    os.environ["TP6_ASSOC_KEYS"] = "64"
    os.environ["TP6_ASSOC_PAIRS"] = "4"
    os.environ["PILOT_OFFLINE"] = "1"
    os.environ["TP6_MAX_SAMPLES"] = "8192"
    # Manual fixed batch for max throughput (no auto-tune).
    os.environ["TP6_BATCH_SIZE"] = "448"
    os.environ["TP6_MAX_STEPS"] = "999999999"
    os.environ["TP6_WALL"] = "999999999"
    os.environ["TP6_IGNORE_WALL_CLOCK"] = "1"
    os.environ["TP6_IGNORE_MAX_STEPS"] = "1"
    os.environ["TP6_UPDATE_SCALE"] = "0.05"
    os.environ["TP6_SCALE_INIT"] = "0.05"
    os.environ["TP6_SCALE_MIN"] = "0.0001"
    os.environ["TP6_SCALE_WARMUP_STEPS"] = "100"
    os.environ["TP6_SCALE_WARMUP_INIT"] = "0.000001"
    os.environ["TP6_SCALE_MAX"] = "1.0"
    os.environ["TP6_THERMO"] = "1"
    os.environ["TP6_THERMO_EVERY"] = "5"
    os.environ["PARAM_POINTER_FORWARD_STEP_PROB"] = "0.1"
    os.environ["TP6_PTR_UPDATE_GOV"] = "1"
    os.environ["TP6_PTR_UPDATE_AUTO"] = "1"
    os.environ["TP6_PTR_UPDATE_EVERY"] = "1"
    os.environ["TP6_PTR_VEL"] = "0"
    os.environ["TP6_DISABLE_SYNC"] = "1"
    os.environ["TP6_AUTO_TUNE"] = "0"
    os.environ["TP6_SHARD_ADAPT"] = "1"
    os.environ["TP6_SHARD_ADAPT_EVERY"] = "1"
    os.environ["TP6_TRACTION_LOG"] = "1"
    os.environ["TP6_STATE_DECAY"] = "1.0"
    os.environ["TP6_AGC_PLATEAU_WINDOW"] = "0"
    os.environ["TP6_GRAD_CLIP"] = "0"
    os.environ["TP6_XRAY"] = "0"
    os.environ["TP6_SAVE_EVERY_STEPS"] = "100"


def main():
    print("=" * 60)
    print("VRAXION // INFINITE_ENGINE_v1.0")
    print("STATUS: KERNEL_PURIFIED | IGNITION_LOCKED")
    print("=" * 60)

    bootstrap_env()
    log_path = os.environ.get("VAR_LOGGING_PATH")
    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        if os.environ.get("TP6_RESUME") == "0" and os.path.exists(log_path):
            os.remove(log_path)
    autotune_batch_size()
    os.makedirs("checkpoints/expert_infinite", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    try:
        import tournament_phase6 as tp6
        run_fn = getattr(tp6, "tournament_phase6_main", getattr(tp6, "main", None))
        if not run_fn:
            print("[FATAL] Could not find 'main' or 'tournament_phase6_main' in the kernel.")
            return
    except ImportError as e:
        print(f"[FATAL] Failed to import tournament_phase6.py: {e}")
        return

    iteration = 0
    while True:
        iteration += 1
        try:
            if iteration > 1:
                print(f"\n[SYSTEM] Re-Ignition Sequence #{iteration} Initiated...")
                os.environ["TP6_RESUME"] = "1"
            run_fn()
        except KeyboardInterrupt:
            print("\n[USER_STOP] Manual shutdown detected. Powering down.")
            break
        except Exception as e:
            print(f"\n[KINETIC_SHOCK] Kernel Crash Detected: {e}")
            print("Engine restarting in 3 seconds...")
            time.sleep(3)
            continue


if __name__ == "__main__":
    main()
