#!/usr/bin/env python3
"""
VRAXION Pages Health Check (deterministic, stdlib-only)

Guards against:
- stale Pages domain drift (kenessy.github.io -> vraxion.github.io)
- broken social/meta assets referenced by docs/index.html (favicon + og:image)

This script is intentionally no-network so CI is stable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


EXPECTED_PAGES_BASE = "https://vraxion.github.io/VRAXION/"
BANNED_SUBSTRINGS = [
    "kenessy.github.io/VRAXION",
]


@dataclass(frozen=True)
class Issue:
    kind: str
    file: str
    message: str


class _HeadTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.metas: dict[str, str] = {}
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict: dict[str, str] = {
            (k.lower() if k else ""): (v if v is not None else "") for k, v in attrs
        }
        if tag == "meta":
            prop = attrs_dict.get("property", "").strip()
            content = attrs_dict.get("content", "").strip()
            if prop and content:
                self.metas[prop] = content
        elif tag == "link":
            # Keep all link tags; we will filter later.
            self.links.append(attrs_dict)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _scan_banned_substrings(paths: Iterable[Path]) -> list[Issue]:
    issues: list[Issue] = []
    for p in paths:
        text = _read_text(p)
        for banned in BANNED_SUBSTRINGS:
            if banned in text:
                issues.append(Issue(kind="banned_substring", file=str(p), message=f'Found "{banned}"'))
    return issues


def _docs_asset_path(repo_root: Path, href_or_url: str) -> Path | None:
    """
    Map a Pages URL (or relative href) to a local path under docs/.

    Supported:
    - "assets/foo.svg" -> docs/assets/foo.svg
    - "/VRAXION/assets/foo.svg" -> docs/assets/foo.svg
    - "https://vraxion.github.io/VRAXION/assets/foo.svg" -> docs/assets/foo.svg

    Anything else returns None.
    """

    s = href_or_url.strip()
    if not s:
        return None

    parsed = urlparse(s)
    if parsed.scheme in ("http", "https"):
        if not s.startswith(EXPECTED_PAGES_BASE):
            return None
        rel = s.removeprefix(EXPECTED_PAGES_BASE)
        return repo_root / "docs" / rel

    # Relative URL (or path-only)
    rel = s
    if rel.startswith("/VRAXION/"):
        rel = rel.removeprefix("/VRAXION/")
    elif rel.startswith("/"):
        rel = rel.removeprefix("/")
    return repo_root / "docs" / rel


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    readme = repo_root / "README.md"
    docs_index = repo_root / "docs" / "index.html"

    issues: list[Issue] = []

    for must_exist in (readme, docs_index):
        if not must_exist.exists():
            issues.append(Issue(kind="missing_file", file=str(must_exist), message="Missing required file"))
            print("\nPAGES HEALTH CHECK: missing required inputs")
            for i in issues:
                print(f"- {i.file}: {i.message}")
            return 2

    issues.extend(_scan_banned_substrings([readme, docs_index]))

    parser = _HeadTagParser()
    parser.feed(_read_text(docs_index))

    og_url = parser.metas.get("og:url")
    if og_url != EXPECTED_PAGES_BASE:
        issues.append(
            Issue(
                kind="bad_og_url",
                file=str(docs_index),
                message=f'Expected og:url "{EXPECTED_PAGES_BASE}", got "{og_url}"',
            )
        )

    # Favicon(s)
    icon_hrefs: list[str] = []
    for link in parser.links:
        rel = link.get("rel", "").lower()
        href = link.get("href", "").strip()
        if not href:
            continue
        # rel may contain multiple tokens (e.g. "shortcut icon")
        if "icon" in rel.split():
            icon_hrefs.append(href)

    if not icon_hrefs:
        issues.append(Issue(kind="missing_icon", file=str(docs_index), message="No <link rel=\"icon\"> found"))
    else:
        for href in icon_hrefs:
            p = _docs_asset_path(repo_root, href)
            if p is None:
                continue
            if not p.exists():
                issues.append(
                    Issue(
                        kind="missing_asset",
                        file=str(docs_index),
                        message=f'Icon asset missing: href="{href}" -> {p}',
                    )
                )

    og_image = parser.metas.get("og:image", "").strip()
    if og_image.startswith(EXPECTED_PAGES_BASE):
        p = _docs_asset_path(repo_root, og_image)
        if p is None or (not p.exists()):
            issues.append(
                Issue(
                    kind="missing_asset",
                    file=str(docs_index),
                    message=f'OG image asset missing: content="{og_image}" -> {p}',
                )
            )

    if issues:
        print("\nPAGES HEALTH CHECK FAILURES:")
        for i in issues:
            print(f"- {i.kind}: {i.file}: {i.message}")
        print("\nSUMMARY:")
        print(f"- issues: {len(issues)}")
        return 1

    print("\nSUMMARY:")
    print("- issues: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

