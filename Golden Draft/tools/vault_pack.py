from __future__ import annotations

"""Vault packer for archiving legacy folders into a single compressed zip.

This is used to keep the top-level workspace clean while retaining an auditable
snapshot of artifacts (logs/checkpoints/tmp/etc.) in a single file.

Design goals:
- Stdlib-only (no external deps).
- Deterministic ordering (stable zip entry list).
- Optional manifest with SHA-256 + size for integrity checks.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class _Entry:
    fs_path: Path
    zip_path: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_zip_path(path: Path) -> str:
    # Zip paths must be forward-slash.
    return str(path).replace("\\", "/").lstrip("/")


def _iter_entries(src: Sequence[Path], *, base: Path) -> Iterator[_Entry]:
    base = base.resolve()
    for root in src:
        root = root.resolve()
        if root.is_file():
            try:
                rel = root.relative_to(base)
            except ValueError:
                rel = Path(root.name)
            yield _Entry(fs_path=root, zip_path=_norm_zip_path(rel))
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            # Deterministic ordering.
            dirnames.sort()
            filenames.sort()

            for name in filenames:
                fp = Path(dirpath) / name
                try:
                    rel = fp.resolve().relative_to(base)
                except ValueError:
                    rel = Path(fp.name)
                yield _Entry(fs_path=fp, zip_path=_norm_zip_path(rel))


def _match_any(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def pack_zip(
    *,
    out_zip: Path,
    base: Path,
    src: Sequence[Path],
    exclude: Sequence[str],
    compression: str,
    write_manifest: bool,
) -> Tuple[int, int]:
    out_zip = out_zip.resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    if compression == "stored":
        zcmp = 0
    elif compression == "deflated":
        zcmp = 8
    elif compression == "lzma":
        zcmp = 14
    else:
        raise ValueError(f"unknown compression: {compression}")

    # Deduplicate by zip path (last one wins, but we sort so it's stable).
    entries = sorted(_iter_entries(src, base=base), key=lambda e: e.zip_path)
    uniq: List[_Entry] = []
    last: Optional[str] = None
    for e in entries:
        if last == e.zip_path:
            uniq[-1] = e
        else:
            uniq.append(e)
            last = e.zip_path

    manifest: List[dict] = []
    kept = 0
    skipped = 0

    # Write zip.
    import zipfile

    mode = "w"
    with zipfile.ZipFile(out_zip, mode=mode, compression=zcmp, compresslevel=9) as zf:
        for e in uniq:
            zp = e.zip_path
            if exclude and _match_any(zp, exclude):
                skipped += 1
                continue

            try:
                st = e.fs_path.stat()
            except OSError:
                skipped += 1
                continue

            zf.write(e.fs_path, arcname=zp)
            kept += 1

            if write_manifest:
                try:
                    hx = _sha256_file(e.fs_path)
                except OSError:
                    hx = None
                manifest.append(
                    {
                        "zip_path": zp,
                        "src_path": str(e.fs_path),
                        "size": int(st.st_size),
                        "mtime_ns": int(st.st_mtime_ns),
                        "sha256": hx,
                    }
                )

    if write_manifest:
        mf_path = out_zip.with_suffix(out_zip.suffix + ".manifest.json")
        mf_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return kept, skipped


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pack legacy folders into a compressed zip + manifest.")
    ap.add_argument("--out", required=True, help="Output zip path.")
    ap.add_argument(
        "--base",
        required=True,
        help="Base directory used to compute zip entry paths (recommended: S:\\AI).",
    )
    ap.add_argument(
        "--src",
        action="append",
        default=[],
        help="Source path to include (repeatable).",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern to exclude by zip path (repeatable). Example: '*/__pycache__/*'",
    )
    ap.add_argument(
        "--compression",
        choices=["stored", "deflated", "lzma"],
        default="deflated",
        help="Zip compression method.",
    )
    ap.add_argument("--no-manifest", action="store_true", help="Do not write the manifest JSON.")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    out_zip = Path(args.out)
    base = Path(args.base)
    src = [Path(p) for p in args.src]
    if not src:
        print("vault_pack: no --src provided", file=sys.stderr)
        return 2

    kept, skipped = pack_zip(
        out_zip=out_zip,
        base=base,
        src=src,
        exclude=args.exclude,
        compression=args.compression,
        write_manifest=(not args.no_manifest),
    )
    print(f"vault_pack: wrote {out_zip} (kept={kept} skipped={skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

