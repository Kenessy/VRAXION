"""Utilities for emitting stable, ASCII-safe log headers.

The original entrypoint emits recognizable multi-line headers.
This module provides a tiny, testable shim so callers can:
- keep header text stable (pass-through)
- centralize ASCII validation/sanitization
- centralize the output mechanism (print vs logger) without changing lines
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


WriteLine = Callable[[str], None]


@dataclass(frozen=True)
class HeaderIssue:
    line_index: int
    original: str
    message: str


def _is_ascii(sval: str) -> bool:
    try:
        sval.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def sanitize_ascii(sval: str, *, replacement: str = "?") -> tuple[str, bool]:
    """Return (ascii_string, was_modified)."""

    if _is_ascii(sval):
        return sval, False

    sanitized = sval.encode("ascii", errors="replace").decode("ascii")
    if replacement != "?":
        sanitized = sanitized.replace("?", replacement)
    return sanitized, True


def default_writer(*, stream=None) -> WriteLine:
    """Return a simple writer that prints to a given stream (default: stdout)."""

    if stream is None:
        stream = sys.stdout

    def _write(line: str) -> None:
        # Intentionally avoid extra formatting; keep header lines stable.
        print(line, file=stream)

    return _write


def emit_header(
    lines: Sequence[str],
    *,
    write_line: Optional[WriteLine] = None,
    ensure_ascii: bool = True,
    ascii_replacement: str = "?",
) -> list[HeaderIssue]:
    """Emit header lines.

    Returns a list of issues if non-ASCII lines were sanitized.
    """

    if write_line is None:
        write_line = default_writer()

    issues: list[HeaderIssue] = []
    for idx, line in enumerate(lines):
        out_line = line
        if ensure_ascii:
            out_line, modified = sanitize_ascii(out_line, replacement=ascii_replacement)
            if modified:
                issues.append(HeaderIssue(idx, line, "Non-ASCII characters were sanitized"))
        write_line(out_line)

    return issues
