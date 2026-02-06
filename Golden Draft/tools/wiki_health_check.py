#!/usr/bin/env python3
"""
VRAXION Wiki Health Check

Checks the GitHub Wiki for:
- banned legacy links (Kenessy repo/pages)
- embedded SVG URL validity (HTTP 200) and raw.githubusercontent.com usage
- broken internal [[Page]] links (must resolve to an existing *.md file)
- milestone SVG drift (time-sensitive "active status" phrases inside the phases diagram)

Stdlib-only by design (runs in CI without extra deps).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_WIKI_URL = "https://github.com/VRAXION/VRAXION.wiki.git"


@dataclass(frozen=True)
class Finding:
    kind: str
    file: str
    line: int
    message: str


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _iter_md_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.md") if p.is_file()])


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _find_banned_strings(md_files: list[Path], banned: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for md in md_files:
        text = _read_text(md)
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            for b in banned:
                if b in line:
                    findings.append(
                        Finding(
                            kind="banned_link",
                            file=str(md),
                            line=idx,
                            message=f'Found banned substring "{b}"',
                        )
                    )
    return findings


def _check_home_locked(wiki_root: Path) -> list[Finding]:
    home = wiki_root / "Home.md"
    if not home.exists():
        # Case-insensitive fallback (defensive on non-Windows runners)
        matches = [p for p in _iter_md_files(wiki_root) if p.name.lower() == "home.md"]
        if matches:
            home = matches[0]
        else:
            return [
                Finding(
                    kind="home_lock",
                    file=str(wiki_root),
                    line=1,
                    message="Missing Home.md in wiki root",
                )
            ]

    text = _read_text(home)
    lines = [ln.strip() for ln in text.splitlines()]

    required_sentinel = "HOME_LOCKED_V2"
    required_headings = [
        "## TLDR",
        "## Non-negotiables",
        "## Reader map",
        "## Evidence contract (how we prove progress)",
        "## Scaling contract (Resolution over reshuffling)",
        "## Navigation (everything)",
    ]

    findings: list[Finding] = []

    if required_sentinel not in text:
        findings.append(
            Finding(
                kind="home_lock",
                file=str(home),
                line=1,
                message=f'Missing sentinel "{required_sentinel}"',
            )
        )

    for h in required_headings:
        if h not in lines:
            findings.append(
                Finding(
                    kind="home_lock",
                    file=str(home),
                    line=1,
                    message=f'Missing required heading "{h}"',
                )
            )

    if "Epistemic boundary" not in text:
        findings.append(
            Finding(
                kind="home_lock",
                file=str(home),
                line=1,
                message='Missing required substring "Epistemic boundary"',
            )
        )

    return findings


SVG_URL_RE = re.compile(r"https?://[^\s<>\"\)]*\.svg(?:\?[^\s<>\"\)]*)?")


def _extract_svg_urls(md_files: list[Path]) -> list[str]:
    urls: set[str] = set()
    for md in md_files:
        for match in SVG_URL_RE.finditer(_read_text(md)):
            urls.add(match.group(0))
    return sorted(urls)


def _http_check(url: str, *, timeout_s: float) -> tuple[bool, str]:
    headers = {"User-Agent": "VRAXION wiki_health_check/1.0"}

    # Prefer HEAD (fast), but fall back to a tiny GET in case a host rejects HEAD.
    try:
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if status in (200, 206):
                return True, f"ok:{status}"
            return False, f"bad_status:{status}"
    except urllib.error.HTTPError as e:
        # Some hosts return 403/405 for HEAD; fall back to GET with Range.
        head_err = f"http_error:{e.code}"
    except Exception as e:  # noqa: BLE001 - want a short error string
        return False, f"head_exc:{type(e).__name__}"

    try:
        get_headers = dict(headers)
        get_headers["Range"] = "bytes=0-0"
        req = urllib.request.Request(url, method="GET", headers=get_headers)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if status in (200, 206):
                return True, f"ok:{status} ({head_err})"
            return False, f"bad_status:{status} ({head_err})"
    except urllib.error.HTTPError as e:
        return False, f"http_error:{e.code} ({head_err})"
    except Exception as e:  # noqa: BLE001
        return False, f"get_exc:{type(e).__name__} ({head_err})"


MILESTONE_SVG_RAW_URL = "https://raw.githubusercontent.com/VRAXION/VRAXION/main/docs/assets/vraxion_phases.svg"
MILESTONE_SVG_BANNED_PHRASES = [
    "System status",
    "Active milestone",
    "Active focus",
]


def _fetch_text_with_retries(
    url: str,
    *,
    timeout_s: float,
    retries: int = 2,
    sleep_s: float = 0.6,
) -> tuple[str | None, str]:
    headers = {"User-Agent": "VRAXION wiki_health_check/1.0"}
    last_err = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, method="GET", headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if status not in (200, 206):
                    return None, f"bad_status:{status}"
                data = resp.read()
            return data.decode("utf-8", errors="replace"), "ok"
        except urllib.error.HTTPError as e:
            last_err = f"http_error:{e.code}"
        except Exception as e:  # noqa: BLE001 - keep error string short
            last_err = f"get_exc:{type(e).__name__}"

        if attempt < retries:
            time.sleep(sleep_s * (attempt + 1))
    return None, last_err


WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _slugify_wiki_title(title: str) -> str:
    out: list[str] = []
    for ch in title:
        if ch.isalnum() or ch in "-_()":
            out.append(ch)
        elif ch.isspace():
            out.append("-")
        else:
            # GitHub wiki file naming tends to replace punctuation with '-'
            out.append("-")
    return "".join(out)


def _extract_wikilink_targets(md_files: list[Path]) -> set[str]:
    targets: set[str] = set()
    for md in md_files:
        for match in WIKILINK_RE.finditer(_read_text(md)):
            raw = match.group(1).strip()
            if not raw:
                continue
            targets.add(raw)
    return targets


def _resolve_wikilink_to_filename(target: str, existing_files_lower: dict[str, str]) -> str | None:
    # Strip anchor and split label syntax.
    main = target.split("#", 1)[0].strip()
    if not main:
        return None

    # Ignore things that look like URLs.
    if "://" in main or main.startswith("//"):
        return None

    candidates: list[str] = []
    if "|" in main:
        left, right = [p.strip() for p in main.split("|", 1)]
        if left:
            candidates.append(left)
        if right:
            candidates.append(right)
    else:
        candidates.append(main)

    for cand in candidates:
        fname = _slugify_wiki_title(cand) + ".md"
        if fname.lower() in existing_files_lower:
            return existing_files_lower[fname.lower()]

    return _slugify_wiki_title(candidates[0]) + ".md"


def _check_wikilinks(wiki_root: Path, md_files: list[Path]) -> list[str]:
    existing = {p.name.lower(): p.name for p in _iter_md_files(wiki_root)}
    missing: list[str] = []
    for raw in sorted(_extract_wikilink_targets(md_files)):
        resolved = _resolve_wikilink_to_filename(raw, existing)
        if not resolved:
            continue
        if resolved.lower() not in existing:
            missing.append(f"{raw} -> {resolved}")
    return missing


def _clone_wiki(url: str, dest_dir: Path, *, branch: str | None) -> None:
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(dest_dir)]
    _run(cmd)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="wiki_health_check.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="VRAXION wiki health check (banned links, SVGs, and [[Page]] targets).",
        epilog=textwrap.dedent(
            f"""\
            Examples:
              python wiki_health_check.py
              python wiki_health_check.py --wiki-dir path/to/wiki

            Default wiki URL:
              {DEFAULT_WIKI_URL}
            """
        ),
    )
    parser.add_argument("--wiki-url", default=DEFAULT_WIKI_URL, help="Git remote URL for the wiki.")
    parser.add_argument(
        "--wiki-dir",
        default=None,
        help="Use an existing local wiki checkout (skip git clone).",
    )
    parser.add_argument("--branch", default="master", help="Wiki branch to clone (default: master).")
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=20.0,
        help="HTTP timeout (seconds) for SVG URL checks.",
    )
    parser.add_argument(
        "--allow-non-raw-svg",
        action="store_true",
        help="Do not enforce raw.githubusercontent.com/VRAXION/VRAXION for SVG URLs.",
    )
    args = parser.parse_args(argv)

    banned = [
        "github.com/Kenessy/VRAXION",
        "kenessy.github.io/VRAXION",
        "github.com/users/Kenessy/projects/4",
    ]

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="vraxion_wiki_health_") as tmp:
        if args.wiki_dir:
            wiki_root = Path(args.wiki_dir).resolve()
        else:
            wiki_root = Path(tmp) / "wiki"
            _clone_wiki(args.wiki_url, wiki_root, branch=args.branch)

        md_files = _iter_md_files(wiki_root)
        if not md_files:
            print(f"ERROR: no markdown files found under {wiki_root}")
            return 2

        findings = _find_banned_strings(md_files, banned)
        home_findings = _check_home_locked(wiki_root)

        svg_urls = _extract_svg_urls(md_files)
        svg_failures: list[str] = []
        if svg_urls:
            for url in svg_urls:
                if (not args.allow_non_raw_svg) and ("raw.githubusercontent.com/VRAXION/VRAXION/" not in url):
                    svg_failures.append(f"non_raw_svg: {url}")
                    continue
                ok, info = _http_check(url, timeout_s=args.timeout_sec)
                if not ok:
                    svg_failures.append(f"{info}: {url}")

        milestone_svg_violations: list[str] = []
        milestone_source = ""
        milestone_text: str | None = None

        # Prefer scanning the locally checked-out asset so PRs gate correctly and deterministically.
        milestone_local = Path(__file__).resolve().parents[2] / "docs" / "assets" / "vraxion_phases.svg"
        if milestone_local.exists():
            milestone_source = str(milestone_local)
            milestone_text = milestone_local.read_text(encoding="utf-8", errors="replace")
        else:
            milestone_source = MILESTONE_SVG_RAW_URL
            milestone_text, milestone_info = _fetch_text_with_retries(
                MILESTONE_SVG_RAW_URL,
                timeout_s=args.timeout_sec,
                retries=2,
            )
            if milestone_text is None:
                milestone_svg_violations.append(f"{milestone_info}: {MILESTONE_SVG_RAW_URL}")

        if milestone_text is not None:
            for phrase in MILESTONE_SVG_BANNED_PHRASES:
                if phrase in milestone_text:
                    milestone_svg_violations.append(f'{milestone_source}: phrase_found:"{phrase}"')

        missing_wikilinks = _check_wikilinks(wiki_root, md_files)

    elapsed = time.time() - started

    ok = True
    if findings:
        ok = False
        print("\nBANNED LINK FINDINGS:")
        for f in findings:
            print(f"- {f.file}:{f.line}: {f.message}")

    if home_findings:
        ok = False
        print("\nHOME LOCK VIOLATIONS:")
        for f in home_findings:
            print(f"- {f.file}:{f.line}: {f.message}")

    if missing_wikilinks:
        ok = False
        print("\nBROKEN [[...]] WIKI LINKS:")
        for m in missing_wikilinks:
            print(f"- {m}")

    if svg_failures:
        ok = False
        print("\nSVG URL FAILURES:")
        for s in svg_failures:
            print(f"- {s}")

    if milestone_svg_violations:
        ok = False
        print("\nMILESTONE SVG DRIFT VIOLATIONS:")
        for v in milestone_svg_violations:
            print(f"- {v}")

    print("\nSUMMARY:")
    print(f"- markdown_files: {len(md_files)}")
    print(f"- banned_findings: {len(findings)}")
    print(f"- home_lock_violations: {len(home_findings)}")
    print(f"- svg_urls: {len(svg_urls)} (failures: {len(svg_failures)})")
    print(f"- milestone_svg_drift_violations: {len(milestone_svg_violations)}")
    print(f"- broken_wikilinks: {len(missing_wikilinks)}")
    print(f"- elapsed_s: {elapsed:.2f}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
