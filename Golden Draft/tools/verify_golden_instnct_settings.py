from __future__ import annotations

"""
Verifies that the production "Golden Code" settings module matches the locked
golden behavior contract.

This mirrors the locked contract used by the Golden Disk verifier, but keeps
the contract JSON in Golden Draft so verification does not depend on external
mirror folders.

Run from repo root:
  python tools/verify_golden_instnct_settings.py
"""

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path


# Locked contract JSON is kept in Golden Draft so verification does not depend on
# external mirror folders.
CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"
EXPECTED = CONTRACTS_DIR / "instnct_settings_golden_expected.json"

# By default, verify against the production "Golden Code" tree if present.
# Override with VRAXION_GOLDEN_SRC=<path>.
DEFAULT_GOLDEN_SRC = Path(r"S:\AI\Golden Code")
_override = os.environ.get("VRAXION_GOLDEN_SRC")
GOLDEN_SRC = Path(_override).expanduser().resolve() if _override else DEFAULT_GOLDEN_SRC
if not GOLDEN_SRC.exists():
    raise SystemExit(f"Golden source path not found: {GOLDEN_SRC!s}")
sys.path.insert(0, str(GOLDEN_SRC))

import vraxion.settings as settings_mod  # noqa: E402
from vraxion.settings import load_settings  # noqa: E402


def _dtype_name(dt) -> str:
    s = str(dt)
    if s.startswith("torch."):
        return s.split(".", 1)[1]
    return s


@contextmanager
def _with_env(env: dict):
    keys = list(env.keys())
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k in keys:
            if old[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old[k]


def _expected_default_root() -> str:
    return str(Path(settings_mod.__file__).resolve().parents[1])


def _expected_default_log(root: str) -> str:
    return os.path.join(root, "logs", "current", "vraxion.log")


def _assert_paths(cfg, exp: dict) -> None:
    root_kind = exp.get("root_kind", "default")
    if root_kind == "custom":
        assert cfg.root == exp["root"], (cfg.root, exp["root"])
    else:
        assert cfg.root == _expected_default_root(), (cfg.root, _expected_default_root())

    assert cfg.data_dir == os.path.join(cfg.root, "data")

    log_kind = exp.get("log_path_kind", "default")
    if log_kind == "custom":
        assert cfg.log_path == exp["log_path"], (cfg.log_path, exp["log_path"])
    else:
        assert cfg.log_path == _expected_default_log(cfg.root), (
            cfg.log_path,
            _expected_default_log(cfg.root),
        )


def _assert_types(cfg, exp: dict) -> None:
    if "dtype" in exp:
        assert _dtype_name(cfg.dtype) == exp["dtype"], _dtype_name(cfg.dtype)
    if "ptr_dtype" in exp:
        assert _dtype_name(cfg.ptr_dtype) == exp["ptr_dtype"], _dtype_name(cfg.ptr_dtype)


def _assert_flags(cfg, exp: dict) -> None:
    if "use_amp" in exp:
        assert bool(cfg.use_amp) == bool(exp["use_amp"]), (cfg.use_amp, exp["use_amp"])
    if "thermo_enabled" in exp:
        assert bool(cfg.thermo_enabled) == bool(exp["thermo_enabled"])
    if "panic_enabled" in exp:
        assert bool(cfg.panic_enabled) == bool(exp["panic_enabled"])
    if "offline_only" in exp:
        assert bool(cfg.offline_only) == bool(exp["offline_only"])


def main() -> int:
    blob = json.loads(EXPECTED.read_text(encoding="utf-8"))
    for case in blob["cases"]:
        env = case.get("env", {})
        with _with_env(env):
            cfg = load_settings()

        exp = case["expected"]
        _assert_paths(cfg, exp)
        _assert_types(cfg, exp)
        _assert_flags(cfg, exp)

        # Exact-match a few stable fields (protect common regressions).
        if "device" in exp:
            assert str(cfg.device) == str(exp["device"])
        if "ring_len" in exp:
            assert int(cfg.ring_len) == int(exp["ring_len"])
        if "slot_dim" in exp:
            assert int(cfg.slot_dim) == int(exp["slot_dim"])

    print("OK: golden behavior verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
