"""Shared bootstrap/helpers for Golden Draft tests.

Golden Draft lives alongside the production "Golden Code" tree, but Golden Code
may not be on ``PYTHONPATH`` for local runs.

This module is named ``conftest.py`` for compatibility with pytest auto-loading,
but it does **not** depend on pytest.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


DRAFTR = Path(__file__).resolve().parents[1]
REPROO = DRAFTR.parent
WINGLD = r"S:\AI\Golden Code"
WINGL2 = r"S:/AI/Golden Code"

# Optional env overrides used by local runners / CI.
ENVKEY = (
    "VRAXION_GOLDEN_SRC",
    "GOLDEN_CODE_ROOT",
    "GOLDEN_CODE_PATH",
    "GOLDEN_CODE_DIR",
)


def _isdir(pthstr: str) -> bool:
    """Return True if *pthstr* is an existing directory.

    Some platforms may treat Windows-style drive paths as malformed. Those
    checks should be treated as "does not exist".
    """

    try:
        return bool(pthstr) and os.path.isdir(pthstr)
    except OSError:
        return False


def _addpth(pthstr: str) -> None:
    """Prepend *pthstr* to sys.path (if not already present)."""

    if pthstr and pthstr not in sys.path:
        sys.path.insert(0, pthstr)


def bootstrap_import_path() -> None:
    """Ensure Golden Draft + Golden Code are importable for tests."""

    # Golden Draft itself (for tools modules, local scripts, etc.).
    _addpth(str(DRAFTR))

    # Prefer explicit env override, then sibling "Golden Code", then the
    # historical Windows location.
    candls: list[str] = []

    for keystr in ENVKEY:
        envval = os.environ.get(keystr)
        if envval:
            candls.append(envval)

    coddir = REPROO / "Golden Code"
    candls.append(str(coddir))
    candls.append(WINGLD)
    candls.append(WINGL2)

    for candpt in candls:
        if _isdir(candpt):
            _addpth(candpt)
            break


@contextmanager
def temporary_env(**ovrmap: Optional[str]) -> Iterator[None]:
    """Temporarily set/unset environment variables.

    Pass KEY="value" to set, or KEY=None to ensure the variable is unset.

    This keeps behavior-lock tests deterministic when a developer has VRAXION
    overrides exported in their shell.
    """

    misobj = object()
    oldmap: dict[str, object] = {keystr: os.environ.get(keystr, misobj) for keystr in ovrmap}

    try:
        for keystr, valstr in ovrmap.items():
            if valstr is None:
                os.environ.pop(keystr, None)
            else:
                os.environ[keystr] = str(valstr)
        yield
    finally:
        for keystr, oldval in oldmap.items():
            if oldval is misobj:
                os.environ.pop(keystr, None)
            else:
                os.environ[keystr] = str(oldval)


# Run at import time so both unittest (explicit import) and pytest (auto-load)
# get deterministic import behavior.
bootstrap_import_path()
