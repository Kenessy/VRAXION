#!/usr/bin/env python
"""VRAXION infinite supervisor.

This is "Golden Draft" internal tooling.

The original draft implementation ran the training loop in-process inside an
infinite loop. This refactor keeps the same user-facing intent but runs the
loop in a *child process* to reduce the risk of orphaned workers and to make
restarts cleaner.

Defaults match the original profile:
- CPU-only (VAR_COMPUTE_DEVICE=cpu, CUDA_VISIBLE_DEVICES="")
- fp32 weights, fp64 pointers
- OMP/MKL threads: 24
- resumes from checkpoints/checkpoint.pt
- logs to logs/current/vraxion.log

Usage:
  python VRAXION_INFINITE.py
  python VRAXION_INFINITE.py --threads 8

Notes
- Logs are append-only; this supervisor never truncates them.
- Unknown args are forwarded to the child process (and thus to its argparse).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _setenv(envmap: Dict[str, str], keystr: str, valstr: str, respekt: bool) -> None:
    if respekt and keystr in envmap:
        return
    envmap[keystr] = valstr


def _abspth(path: str, wrkdir: Path) -> str:
    # Expand user/vars, then resolve relative to work dir.
    expnd6 = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(expnd6):
        return os.path.abspath(expnd6)
    return os.path.abspath(str(wrkdir / expnd6))


def _ensure_dirs(logpth: str, ckppth: str) -> None:
    try:
        os.makedirs(os.path.dirname(logpth), exist_ok=True)
    except Exception:
        pass
    try:
        os.makedirs(os.path.dirname(ckppth), exist_ok=True)
    except Exception:
        pass


def build_env(
    *,
    threads: int,
    log_path: str,
    ckpt_path: str,
    respekt: bool,
    extra_set: List[str],
    extra_unset: List[str],
) -> Dict[str, str]:
    envmap: Dict[str, str] = dict(os.environ)

    _setenv(envmap, "VAR_COMPUTE_DEVICE", "cpu", respekt)
    _setenv(envmap, "CUDA_VISIBLE_DEVICES", "", respekt)
    _setenv(envmap, "VRX_PRECISION", "fp32", respekt)
    _setenv(envmap, "VRX_PTR_DTYPE", "fp64", respekt)

    _setenv(envmap, "OMP_NUM_THREADS", str(int(threads)), respekt)
    _setenv(envmap, "MKL_NUM_THREADS", str(int(threads)), respekt)

    _setenv(envmap, "VRX_RESUME", "1", respekt)
    _setenv(envmap, "VRX_CKPT", ckpt_path, respekt)
    _setenv(envmap, "VRX_SAVE_EVERY_STEPS", "50", respekt)
    _setenv(envmap, "VRX_EVAL_EVERY_STEPS", "10", respekt)
    _setenv(envmap, "VRX_EVAL_AT_CHECKPOINT", "0", respekt)

    _setenv(envmap, "VRX_SYNTH", "1", respekt)
    _setenv(envmap, "VRX_SYNTH_MODE", "assoc_byte", respekt)
    _setenv(envmap, "VRX_SYNTH_LEN", "512", respekt)
    _setenv(envmap, "VRX_ASSOC_KEYS", "64", respekt)
    _setenv(envmap, "VRX_ASSOC_PAIRS", "4", respekt)
    _setenv(envmap, "VRX_MAX_SAMPLES", "8192", respekt)
    _setenv(envmap, "VRX_BATCH_SIZE", "152", respekt)
    _setenv(envmap, "VRX_OFFLINE_ONLY", "1", respekt)
    _setenv(envmap, "VRX_EXPERT_HEADS", "16", respekt)

    _setenv(envmap, "VAR_LOGGING_PATH", log_path, respekt)

    for kvstr in extra_set:
        if "=" not in kvstr:
            continue
        keystr, valstr = kvstr.split("=", 1)
        keystr = keystr.strip()
        if not keystr:
            continue
        envmap[keystr] = valstr

    for keystr in extra_unset:
        keystr = keystr.strip()
        if not keystr:
            continue
        envmap.pop(keystr, None)

    # Pass module name for the child wrapper.
    envmap.setdefault("VRXMOD", "vraxion_run")
    return envmap


def _child_code() -> str:
    # Keep this string stdlib-only and small.
    return (
        "import importlib, os, sys\n"
        "thr=os.environ.get('OMP_NUM_THREADS') or os.environ.get('MKL_NUM_THREADS') or '1'\n"
        "try:\n"
        "    import torch\n"
        "    torch.set_num_threads(int(thr))\n"
        "except Exception:\n"
        "    pass\n"
        "mod=os.environ.get('VRXMOD','vraxion_run')\n"
        "m=importlib.import_module(mod)\n"
        "fn=getattr(m,'main',None)\n"
        "if fn is None:\n"
        "    raise SystemExit(1)\n"
        "rc=fn()\n"
        "raise SystemExit(rc if isinstance(rc,int) else 0)\n"
    )


def _spawn(
    *,
    python_exe: str,
    work_dir: Path,
    env_map: Dict[str, str],
    pass_args: List[str],
) -> subprocess.Popen:
    cmdlst = [python_exe, "-u", "-c", _child_code(), *pass_args]

    kwmap6: Dict[str, object] = {
        "cwd": str(work_dir),
        "env": env_map,
    }

    if os.name == "posix":
        kwmap6["start_new_session"] = True
    else:
        flgval = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if flgval:
            kwmap6["creationflags"] = flgval

    return subprocess.Popen(cmdlst, **kwmap6)  # type: ignore[arg-type]


def _killpg(proc6x: subprocess.Popen, sigval: int) -> None:
    if proc6x.pid is None:
        return

    if os.name == "posix":
        try:
            os.killpg(proc6x.pid, sigval)
        except Exception:
            return
    else:
        try:
            proc6x.send_signal(sigval)
        except Exception:
            return


def _stop_child(proc6x: subprocess.Popen, tout_s: float) -> None:
    try:
        _killpg(proc6x, signal.SIGTERM)
    except Exception:
        pass

    try:
        proc6x.wait(timeout=float(tout_s))
        return
    except Exception:
        pass

    try:
        _killpg(proc6x, signal.SIGKILL)
    except Exception:
        pass


def _parse(argv: Optional[Sequence[str]]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Infinite VRAXION supervisor")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--restart-delay", type=float, default=3.0)
    parser.add_argument("--max-restarts", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--work-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python", dest="python_exe", default=sys.executable)
    parser.add_argument("--respect-existing-env", action="store_true")
    parser.add_argument("--log-path", default=os.path.join("logs", "current", "vraxion.log"))
    parser.add_argument("--ckpt-path", default=os.path.join("checkpoints", "checkpoint.pt"))
    parser.add_argument("--module", default="vraxion_run", help="Child module name")
    parser.add_argument("--set", dest="set_kv", action="append", default=[], help="KEY=VALUE (repeatable)")
    parser.add_argument("--unset", dest="unset_keys", action="append", default=[], help="KEY (repeatable)")
    parser.add_argument("--once", action="store_true", help="Run one child and exit")
    return parser.parse_known_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argobj, unkls6 = _parse(argv)

    wrkdir = Path(os.path.expanduser(argobj.work_dir)).resolve()

    logpth = _abspth(str(argobj.log_path), wrkdir)
    ckppth = _abspth(str(argobj.ckpt_path), wrkdir)

    _ensure_dirs(logpth, ckppth)

    envmap = build_env(
        threads=int(argobj.threads),
        log_path=str(argobj.log_path),
        ckpt_path=str(argobj.ckpt_path),
        respekt=bool(argobj.respect_existing_env),
        extra_set=list(argobj.set_kv),
        extra_unset=list(argobj.unset_keys),
    )
    envmap["VRXMOD"] = str(argobj.module)

    stopit = {"flag": False}
    child6: List[subprocess.Popen] = []

    def _onsig(sigval: int, _frm: object) -> None:
        stopit["flag"] = True
        if child6:
            _stop_child(child6[0], 5.0)

    try:
        signal.signal(signal.SIGINT, _onsig)
        signal.signal(signal.SIGTERM, _onsig)
    except Exception:
        # Some environments disallow custom signal handlers.
        pass

    rstcnt = 0

    while True:
        if stopit["flag"]:
            return 0

        if int(argobj.max_restarts) > 0 and rstcnt > int(argobj.max_restarts):
            _eprint(f"[VRAXION] Max restarts reached ({argobj.max_restarts}); exiting")
            return 5

        _eprint("[VRAXION] START TRAINING LOOP")

        proc6x = _spawn(
            python_exe=str(argobj.python_exe),
            work_dir=wrkdir,
            env_map=envmap,
            pass_args=list(unkls6),
        )
        child6[:] = [proc6x]

        try:
            retcd6 = int(proc6x.wait())
        except KeyboardInterrupt:
            _eprint("[VRAXION] Manual stop requested; terminating child")
            _stop_child(proc6x, 5.0)
            return 0
        finally:
            child6[:] = []

        if stopit["flag"]:
            return 0

        if retcd6 == 0:
            _eprint("[VRAXION] TRAINING RUN EXITED NORMALLY. Restarting...")
        else:
            _eprint(f"[VRAXION] TRAINING CRASHED. Restarting... exit={retcd6}")

        if bool(argobj.once):
            return retcd6

        rstcnt += 1
        time.sleep(max(0.0, float(argobj.restart_delay)))


if __name__ == "__main__":
    raise SystemExit(main())

