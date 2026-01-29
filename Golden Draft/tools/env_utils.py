"""Environment variable parsing helpers.

Goals:
- Centralize and standardize env var parsing (bool/int/float/str).
- Make env semantics explicit and testable.
- Avoid surprising crashes on malformed env values (default is non-strict).

This module is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, TypeVar


T = TypeVar("T")


TRUE_STRINGS: tuple[str, ...] = ("1", "true", "t", "yes", "y", "on")
FALSE_STRINGS: tuple[str, ...] = ("0", "false", "f", "no", "n", "off")


@dataclass(frozen=True)
class ParseIssue:
    """Represents a non-fatal issue while parsing an env var."""

    key: str
    raw_value: str
    message: str


def _strip_or_none(value: Optional[str], *, treat_empty_as_none: bool = True) -> Optional[str]:
    if value is None:
        return None
    sval = value.strip()
    if treat_empty_as_none and sval == "":
        return None
    return sval


def env_str(
    env: Mapping[str, str],
    key: str,
    default: Optional[str] = None,
    *,
    treat_empty_as_none: bool = True,
) -> Optional[str]:
    """Return env[key] stripped, or default if missing/empty."""

    return _strip_or_none(env.get(key), treat_empty_as_none=treat_empty_as_none) or default


def parse_bool(
    raw: Optional[str],
    default: bool,
    *,
    strict: bool = False,
    truthy: Sequence[str] = TRUE_STRINGS,
    falsy: Sequence[str] = FALSE_STRINGS,
) -> tuple[bool, Optional[str]]:
    """Parse a boolean from a string.

    Returns:
        (value, issue_message)

    If strict=False, unknown values fall back to default and return an issue message.
    If strict=True, unknown values raise ValueError.
    """

    sval = _strip_or_none(raw)
    if sval is None:
        return default, None

    lowered = sval.lower()
    if lowered in (t.lower() for t in truthy):
        return True, None
    if lowered in (f.lower() for f in falsy):
        return False, None

    msg = f"Unrecognized boolean value: {sval!r} (expected one of {list(truthy) + list(falsy)})"
    if strict:
        raise ValueError(msg)
    return default, msg


def env_bool(
    env: Mapping[str, str],
    key: str,
    default: bool = False,
    *,
    strict: bool = False,
    truthy: Sequence[str] = TRUE_STRINGS,
    falsy: Sequence[str] = FALSE_STRINGS,
) -> tuple[bool, Optional[ParseIssue]]:
    """Parse a bool env var.

    Returns:
        (value, issue) where issue is None when parsing is clean.
    """

    raw = env.get(key)
    value, issue = parse_bool(raw, default, strict=strict, truthy=truthy, falsy=falsy)
    if issue is None or raw is None:
        return value, None
    return value, ParseIssue(key=key, raw_value=raw, message=issue)


def env_int(
    env: Mapping[str, str],
    key: str,
    default: int,
    *,
    strict: bool = False,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> tuple[int, Optional[ParseIssue]]:
    """Parse an int env var (safe by default).

    If strict=False, malformed values fall back to default and return an issue.
    """

    raw = _strip_or_none(env.get(key))
    if raw is None:
        return default, None

    try:
        value = int(raw)
    except ValueError:
        msg = f"Invalid int for {key}: {raw!r}"
        if strict:
            raise ValueError(msg)
        return default, ParseIssue(key=key, raw_value=raw, message=msg)

    if min_value is not None and value < min_value:
        msg = f"{key}={value} is < min_value={min_value}"
        if strict:
            raise ValueError(msg)
        return default, ParseIssue(key=key, raw_value=raw, message=msg)

    if max_value is not None and value > max_value:
        msg = f"{key}={value} is > max_value={max_value}"
        if strict:
            raise ValueError(msg)
        return default, ParseIssue(key=key, raw_value=raw, message=msg)

    return value, None


def env_float(
    env: Mapping[str, str],
    key: str,
    default: float,
    *,
    strict: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> tuple[float, Optional[ParseIssue]]:
    """Parse a float env var (safe by default)."""

    raw = _strip_or_none(env.get(key))
    if raw is None:
        return default, None

    try:
        value = float(raw)
    except ValueError:
        msg = f"Invalid float for {key}: {raw!r}"
        if strict:
            raise ValueError(msg)
        return default, ParseIssue(key=key, raw_value=raw, message=msg)

    if min_value is not None and value < min_value:
        msg = f"{key}={value} is < min_value={min_value}"
        if strict:
            raise ValueError(msg)
        return default, ParseIssue(key=key, raw_value=raw, message=msg)

    if max_value is not None and value > max_value:
        msg = f"{key}={value} is > max_value={max_value}"
        if strict:
            raise ValueError(msg)
        return default, ParseIssue(key=key, raw_value=raw, message=msg)

    return value, None

