from __future__ import annotations

"""VRAXION Cache→RAM→HDD sync job (P2).

This tool mirrors Linear tickets into the PRIVATE GitHub Project:
  "VRAXION Archive (Internal)"

Key rules (adversarial-hardened):
- Default mode is DRY-RUN (no writes unless --apply).
- Sync is non-destructive: only updates a single delimited block in the mirror body.
- Duplicate mirror items are fatal: if >1 DraftIssue items share the same Linear key, STOP.
- Public project changes are NEVER automatic (promotion is explicit and separate).

Linear auth:
- Preferred: LINEAR_API_KEY env var (Authorization header is the raw key, NOT "Bearer <key>").
- Fallback for first run: --linear-export <path.json>

Stdlib-only; uses `gh` CLI for GitHub Projects operations.
"""

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SYNC_BEGIN = "<!-- VRX_SYNC BEGIN -->"
SYNC_END = "<!-- VRX_SYNC END -->"

PUBLIC_UPDATE_BEGIN = "<!-- VRX_PUBLIC_UPDATE BEGIN -->"
PUBLIC_UPDATE_END = "<!-- VRX_PUBLIC_UPDATE END -->"

DEFAULT_OWNER = "Kenessy"
DEFAULT_REPO = "VRAXION/VRAXION"
DEFAULT_TEAM_NAME = "VRAXION"
DEFAULT_PROJECTS = ("VRAXION_WALL", "VRAXION_IDEAS")

PRIVATE_ARCHIVE_TITLE = "VRAXION Archive (Internal)"
PUBLIC_ROADMAP_TITLE = "VRAXION Roadmap"

STATE_FILE = ".vrx_sync_state.json"
PROJECT_ITEM_DISCOVERY_LIMIT = 5000


class SyncError(RuntimeError):
    pass


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _repo_root() -> Path:
    # Golden Draft/tools/vrx_sync_linear_projects.py -> repo root
    return Path(__file__).resolve().parents[2]


def _load_version_str(repo_root: Path) -> str:
    try:
        obj = json.loads((repo_root / "VERSION.json").read_text(encoding="utf-8"))
        return f"{obj['major']}.{obj['minor']}.{obj['build']}"
    except Exception:
        return "unknown"


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _run_checked(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> str:
    p = _run(cmd, cwd=cwd)
    if p.returncode != 0:
        raise SyncError(f"Command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout


def gh_json(cmd: Sequence[str]) -> Any:
    raw = _run_checked(cmd)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SyncError(f"Failed to parse JSON from: {' '.join(cmd)}\n{exc}\nRAW:\n{raw[:2000]}") from exc


def patch_delimited_block(existing: str, begin: str, end: str, content: str) -> str:
    """Non-destructively upsert a single delimited block, preserving manual notes."""

    content = content.rstrip("\n")

    begin_count = existing.count(begin)
    end_count = existing.count(end)

    if begin_count == 0 and end_count == 0:
        prefix = existing.rstrip("\n")
        out = prefix
        if out:
            out += "\n\n"
        out += f"{begin}\n{content}\n{end}\n"
        return out

    if begin_count != 1 or end_count != 1:
        raise SyncError(f"Ambiguous block markers (begin={begin_count}, end={end_count}).")

    begin_idx = existing.find(begin)
    end_idx = existing.find(end)
    if begin_idx < 0 or end_idx < 0 or end_idx <= begin_idx:
        raise SyncError("Malformed block markers (ordering).")

    head = existing[: begin_idx + len(begin)]
    tail = existing[end_idx:]
    return f"{head}\n{content}\n{tail}"


def patch_sync_block(existing: str, snapshot: str) -> str:
    return patch_delimited_block(existing, SYNC_BEGIN, SYNC_END, snapshot)


def _extract_delimited_block(existing: str, begin: str, end: str) -> str:
    """Return the inner content between markers (no surrounding newlines)."""

    begin_count = existing.count(begin)
    end_count = existing.count(end)
    if begin_count != 1 or end_count != 1:
        raise SyncError(f"Ambiguous block markers (begin={begin_count}, end={end_count}).")

    begin_idx = existing.find(begin)
    end_idx = existing.find(end)
    if begin_idx < 0 or end_idx < 0 or end_idx <= begin_idx:
        raise SyncError("Malformed block markers (ordering).")

    inner = existing[begin_idx + len(begin) : end_idx]
    return inner.lstrip("\n").rstrip("\n")


def _parse_iso_dt(value: str) -> _dt.datetime:
    raw = value.strip()
    if not raw:
        raise SyncError("Empty ISO datetime")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = _dt.datetime.fromisoformat(raw)
    except ValueError as exc:
        raise SyncError(f"Invalid ISO datetime: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def _format_list(items: Sequence[str]) -> str:
    return ", ".join(items) if items else "—"


_GH_URL_RE = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/(?:pull|issues)/\d+")


def _extract_github_urls(text: str) -> list[str]:
    return sorted(set(_GH_URL_RE.findall(text or "")))


def _merge_evidence_links(snapshot: str, urls: Sequence[str]) -> str:
    """Return snapshot with Evidence links list merged with urls (dedup + sorted)."""

    want = sorted(set(urls))
    if not want:
        return snapshot

    lines = snapshot.splitlines()
    for i, line in enumerate(lines):
        if line == "- Evidence links: —":
            lines[i] = "- Evidence links:"
            lines[i + 1 : i + 1] = [f"  - {u}" for u in want]
            return "\n".join(lines).rstrip("\n")
        if line == "- Evidence links:":
            j = i + 1
            existing_urls: list[str] = []
            while j < len(lines) and lines[j].startswith("  - "):
                existing_urls.append(lines[j][4:])
                j += 1
            merged = sorted(set(existing_urls).union(want))
            lines[i + 1 : j] = [f"  - {u}" for u in merged]
            return "\n".join(lines).rstrip("\n")

    out = snapshot.rstrip("\n") + "\n\n- Evidence links:\n"
    out += "\n".join(f"  - {u}" for u in want)
    return out.rstrip("\n")


@dataclass(frozen=True)
class LinearRelation:
    identifier: str
    title: str


@dataclass(frozen=True)
class LinearIssue:
    identifier: str
    title: str
    url: str
    internal_id: str
    description: str
    state: str
    priority: Optional[int]
    labels: tuple[str, ...]
    created_at: str
    updated_at: str
    project: str
    blocked_by: tuple[LinearRelation, ...]
    blocks: tuple[LinearRelation, ...]
    attachments: tuple[tuple[str, str], ...]  # (title, url)


class LinearClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise SyncError("LINEAR_API_KEY is empty.")

    def graphql(self, query: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        payload = json.dumps({"query": query, "variables": dict(variables)}).encode("utf-8")
        req = urllib.request.Request("https://api.linear.app/graphql", data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        # IMPORTANT: Linear expects raw key (not Bearer).
        req.add_header("Authorization", self.api_key)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise SyncError(f"Linear HTTP error: {exc.code} {exc.reason}\n{body[:2000]}") from exc
        except Exception as exc:
            raise SyncError(f"Linear request failed: {exc}") from exc

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SyncError(f"Linear returned invalid JSON: {exc}\nRAW:\n{raw[:2000]}") from exc

        if "errors" in obj and obj["errors"]:
            raise SyncError(f"Linear GraphQL errors: {obj['errors']}")
        data = obj.get("data")
        if not isinstance(data, dict):
            raise SyncError("Linear GraphQL response missing data.")
        return data

    def _team_id(self, team_name: str) -> str:
        q = """
        query Teams {
          teams { nodes { id name } }
        }
        """
        data = self.graphql(q, {})
        nodes = data.get("teams", {}).get("nodes", [])
        for n in nodes:
            if n.get("name") == team_name:
                return str(n.get("id"))
        raise SyncError(f"Could not find Linear team by name: {team_name!r}")

    def list_team_issues(self, team_name: str, *, limit: int = 500) -> list[LinearIssue]:
        team_id = self._team_id(team_name)

        q = """
        query Issues($teamId: ID!, $after: String) {
          issues(
            first: 50,
            after: $after,
            orderBy: updatedAt,
            filter: { team: { id: { eq: $teamId } } }
          ) {
            nodes {
              id
              identifier
              title
              url
              description
              createdAt
              updatedAt
              priority
              state { name }
              project { name }
              labels(first: 100) { nodes { name } }
              attachments(first: 50) { nodes { title url } }
              relations(first: 50) { nodes { type relatedIssue { identifier title } } }
              inverseRelations(first: 50) { nodes { type issue { identifier title } } }
            }
            pageInfo { hasNextPage endCursor }
          }
        }
        """

        out: list[LinearIssue] = []
        after: Optional[str] = None
        while True:
            data = self.graphql(q, {"teamId": team_id, "after": after})
            conn = data.get("issues", {})
            nodes = conn.get("nodes", []) or []
            for n in nodes:
                proj = (n.get("project") or {}).get("name") or ""
                blocks_pairs: set[tuple[str, str]] = set()
                for rel in ((n.get("relations") or {}).get("nodes", []) or []):
                    if str(rel.get("type") or "").lower() != "blocks":
                        continue
                    ri = rel.get("relatedIssue") or {}
                    ident = str(ri.get("identifier") or "")
                    if not ident:
                        continue
                    blocks_pairs.add((ident, str(ri.get("title") or "")))

                blocked_by_pairs: set[tuple[str, str]] = set()
                for rel in ((n.get("inverseRelations") or {}).get("nodes", []) or []):
                    if str(rel.get("type") or "").lower() != "blocks":
                        continue
                    src = rel.get("issue") or {}
                    ident = str(src.get("identifier") or "")
                    if not ident:
                        continue
                    blocked_by_pairs.add((ident, str(src.get("title") or "")))

                out.append(
                    LinearIssue(
                        internal_id=str(n.get("id")),
                        identifier=str(n.get("identifier")),
                        title=str(n.get("title")),
                        url=str(n.get("url")),
                        description=str(n.get("description") or ""),
                        created_at=str(n.get("createdAt") or ""),
                        updated_at=str(n.get("updatedAt") or ""),
                        priority=int(n["priority"]) if n.get("priority") is not None else None,
                        state=str((n.get("state") or {}).get("name") or ""),
                        project=proj,
                        labels=tuple(sorted({str(x.get("name")) for x in (n.get("labels") or {}).get("nodes", []) or []})),
                        attachments=tuple(
                            (str(a.get("title") or ""), str(a.get("url") or ""))
                            for a in ((n.get("attachments") or {}).get("nodes", []) or [])
                        ),
                        blocked_by=tuple(
                            LinearRelation(identifier=ident, title=title)
                            for ident, title in sorted(blocked_by_pairs, key=lambda x: (x[0], x[1]))
                        ),
                        blocks=tuple(
                            LinearRelation(identifier=ident, title=title)
                            for ident, title in sorted(blocks_pairs, key=lambda x: (x[0], x[1]))
                        ),
                    )
                )
                if len(out) >= limit:
                    return out

            page = conn.get("pageInfo") or {}
            if not page.get("hasNextPage"):
                return out
            after = str(page.get("endCursor"))


def _load_linear_export(path: Path) -> list[LinearIssue]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "issues" not in obj:
        raise SyncError("linear export must be a JSON object with key: issues")
    issues = obj["issues"]
    if not isinstance(issues, list):
        raise SyncError("linear export: issues must be a list")

    def _parse_attachments(raw: Any) -> tuple[tuple[str, str], ...]:
        if raw is None:
            return ()
        if isinstance(raw, list) and len(raw) == 2 and all(isinstance(x, str) for x in raw):
            # PowerShell ConvertTo-Json sometimes collapses 1-item list-of-pairs.
            return ((str(raw[0]), str(raw[1])),)
        if not isinstance(raw, list):
            return ()
        out: list[tuple[str, str]] = []
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                out.append((str(entry[0]), str(entry[1])))
            elif isinstance(entry, dict):
                out.append((str(entry.get("title") or ""), str(entry.get("url") or "")))
        return tuple(out)

    out: list[LinearIssue] = []
    for it in issues:
        if not isinstance(it, dict):
            continue
        out.append(
            LinearIssue(
                internal_id=str(it.get("internal_id") or it.get("id") or ""),
                identifier=str(it.get("identifier")),
                title=str(it.get("title")),
                url=str(it.get("url")),
                description=str(it.get("description") or ""),
                created_at=str(it.get("created_at") or it.get("createdAt") or ""),
                updated_at=str(it.get("updated_at") or it.get("updatedAt") or ""),
                priority=int(it["priority"]) if it.get("priority") is not None else None,
                state=str(it.get("state") or ""),
                project=str(it.get("project") or ""),
                labels=tuple(it.get("labels") or []),
                attachments=_parse_attachments(it.get("attachments")),
                blocked_by=tuple(LinearRelation(**r) for r in (it.get("blocked_by") or [])),
                blocks=tuple(LinearRelation(**r) for r in (it.get("blocks") or [])),
            )
        )
    return out


def _snapshot(issue: LinearIssue) -> str:
    pr_urls: list[str] = []
    for _t, u in issue.attachments:
        if u:
            pr_urls.extend(_extract_github_urls(u))
    pr_urls.extend(_extract_github_urls(issue.description))
    pr_urls = sorted(set(pr_urls))

    lines: list[str] = []
    lines.append(f"Linear: {issue.identifier}  |  {issue.title}")
    lines.append(f"- URL: {issue.url}")
    lines.append(f"- Internal ID: {issue.internal_id}")
    lines.append(f"- CreatedAt: {issue.created_at or '—'}")
    lines.append(f"- UpdatedAt: {issue.updated_at or '—'}")
    lines.append(f"- Project: {issue.project or '—'}")
    lines.append(f"- State: {issue.state or '—'}")
    lines.append(f"- Priority: {issue.priority if issue.priority is not None else '—'}")
    lines.append(f"- Labels: {_format_list(list(issue.labels))}")

    if issue.blocked_by:
        lines.append("- Blocked by:")
        for r in sorted(issue.blocked_by, key=lambda x: x.identifier):
            lines.append(f"  - {r.identifier}: {r.title}")
    else:
        lines.append("- Blocked by: —")

    if issue.blocks:
        lines.append("- Blocks:")
        for r in sorted(issue.blocks, key=lambda x: x.identifier):
            lines.append(f"  - {r.identifier}: {r.title}")
    else:
        lines.append("- Blocks: —")

    if issue.attachments:
        lines.append("- Attachments:")
        for t, u in sorted(issue.attachments, key=lambda x: (x[1], x[0])):
            if u:
                lines.append(f"  - {t}: {u}".rstrip())
            else:
                lines.append(f"  - {t}: —".rstrip())
    else:
        lines.append("- Attachments: —")

    if pr_urls:
        lines.append("- Evidence links:")
        for u in pr_urls:
            lines.append(f"  - {u}")
    else:
        lines.append("- Evidence links: —")

    lines.append("")
    lines.append("Description:")
    lines.append(issue.description.rstrip("\n"))
    lines.append("")
    lines.append(f"Sync: { _utc_now_iso() }  |  Linear updatedAt: {issue.updated_at or '—'}")

    return "\n".join(lines).rstrip("\n")


def _project_by_title(owner: str, title: str) -> dict[str, Any]:
    obj = gh_json(["gh", "project", "list", "--owner", owner, "--format", "json", "-L", "200"])
    projs = obj.get("projects") or []
    for p in projs:
        if p.get("title") == title:
            return p
    raise SyncError(f"Missing GitHub Project for owner={owner}: {title!r}")


def _field_map(project_num: int, owner: str) -> dict[str, dict[str, Any]]:
    obj = gh_json(["gh", "project", "field-list", str(project_num), "--owner", owner, "--format", "json", "-L", "200"])
    fields = obj.get("fields") or []
    return {str(f.get("name")): f for f in fields}


def _item_list(project_num: int, owner: str, *, limit: int = PROJECT_ITEM_DISCOVERY_LIMIT) -> list[dict[str, Any]]:
    obj = gh_json(
        ["gh", "project", "item-list", str(project_num), "--owner", owner, "--format", "json", "-L", str(limit)]
    )
    items = obj.get("items") or []
    # We can't know the total item count from gh output, but hitting the limit is a strong truncation signal.
    if len(items) >= limit:
        raise SyncError(
            f"Project item-list hit limit (limit={limit}). Refusing to proceed to avoid missed mirrors/duplicates. "
            "Increase PROJECT_ITEM_DISCOVERY_LIMIT or implement pagination (gh api graphql)."
        )
    return items


def _ci_get(item: Mapping[str, Any], name: str) -> Any:
    want = name.lower()
    for k, v in item.items():
        if str(k).lower() == want:
            return v
    return None


def _mirror_items_by_key(items: Sequence[Mapping[str, Any]], key_field: str) -> dict[str, list[Mapping[str, Any]]]:
    """Return mapping of LinearKey -> list of DraftIssue items."""

    out: dict[str, list[Mapping[str, Any]]] = {}
    for it in items:
        content = it.get("content") or {}
        if content.get("type") != "DraftIssue":
            continue
        key = _ci_get(it, key_field)
        if not key:
            continue
        out.setdefault(str(key), []).append(it)
    return out


def _status_option_id(field: Mapping[str, Any], name: str) -> Optional[str]:
    for opt in field.get("options") or []:
        if opt.get("name") == name:
            return str(opt.get("id"))
    return None


def _linear_to_project_status(linear_state: str) -> str:
    mapping = {
        "Backlog": "Backlog",
        "Todo": "Ready",
        "In Progress": "In progress",
        "In Review": "In review",
        "Done": "Done",
        "Canceled": "Done",
        "Duplicate": "Done",
    }
    return mapping.get(linear_state, "Backlog")


def _linear_to_archive_status(linear_state: str, *, has_merged_pr: bool) -> str:
    if linear_state in ("Backlog", "Todo"):
        return "Queued"
    if linear_state in ("In Progress", "In Review"):
        return "In Progress"
    if linear_state == "Done":
        return "Merged" if has_merged_pr else "Archived"
    if linear_state in ("Canceled", "Duplicate"):
        return "Archived"
    return "Queued"


def _linear_to_lifecycle(linear_state: str) -> Optional[str]:
    if linear_state == "Done":
        return "Completed"
    if linear_state == "Canceled":
        return "Deleted"
    if linear_state == "Duplicate":
        return "Superseded"
    return None


def _has_merged_pr(urls: Sequence[str]) -> bool:
    # Best-effort: if any PR URL is merged, treat as merged evidence.
    for u in urls:
        if "/pull/" not in u:
            continue
        # Use the URL directly so repo moves/redirects do not break the check.
        p = _run(["gh", "pr", "view", u, "--json", "state", "--jq", ".state"])
        if p.returncode == 0 and p.stdout.strip() == "MERGED":
            return True
    return False


def _ensure_projects(owner: str) -> dict[str, Any]:
    priv = _project_by_title(owner, PRIVATE_ARCHIVE_TITLE)
    # Best-effort visibility check (field name differs by gh version; treat missing as "unknown").
    if priv.get("visibility") not in (None, "", "PRIVATE"):
        raise SyncError(f"Expected private archive project to be PRIVATE; got {priv.get('visibility')!r}")

    # NOTE: The public roadmap project is intentionally NOT required by this tool.
    # - `sync` mirrors Linear tickets into the private archive only.
    # - `prune-pr-items` and `promote` operate on the private archive + (optionally) a public issue,
    #   but do not require a GitHub Projects "roadmap" project to exist under the same owner.
    #
    # This matters after repo ownership transfers where:
    # - private archive stays under a user, but
    # - public roadmap moves to an org (or vice versa).
    return priv


def _ensure_gh_projects_commands() -> None:
    for cmd in (
        ["gh", "project", "--help"],
        ["gh", "project", "item-create", "--help"],
        ["gh", "project", "item-edit", "--help"],
        ["gh", "project", "item-archive", "--help"],
    ):
        p = _run(cmd)
        if p.returncode != 0:
            raise SyncError(f"Required gh command is missing or failed: {' '.join(cmd)}\n{p.stderr.strip()}")


def _ensure_gh_issue_commands() -> None:
    for cmd in (
        ["gh", "issue", "view", "--help"],
        ["gh", "issue", "edit", "--help"],
    ):
        p = _run(cmd)
        if p.returncode != 0:
            raise SyncError(f"Required gh command is missing or failed: {' '.join(cmd)}\n{p.stderr.strip()}")


def _smoke_draft_edit(priv_num: int, owner: str, *, apply: bool) -> None:
    """Gate: ensure DraftIssue body edits work (requires DI_* id)."""

    if not apply:
        print("\n[DRY-RUN] DraftIssue edit smoke test commands (manual):")
        print(f"- gh project item-create {priv_num} --owner {owner} --title \"VRX_SYNC_SMOKE\" --body \"ok\" --format json")
        print(f"- gh project item-list {priv_num} --owner {owner} --format json -L 200  # find DraftIssue content.id (DI_...)")
        print("- gh project item-edit --id <DI_ID> --body \"ok2\"")
        print(f"- gh project item-archive {priv_num} --owner {owner} --id <PVTI_ID>")
        return

    title = f"VRX_SYNC_SMOKE {_utc_now_iso()}"
    item = gh_json(
        [
            "gh",
            "project",
            "item-create",
            str(priv_num),
            "--owner",
            owner,
            "--title",
            title,
            "--body",
            "ok",
            "--format",
            "json",
        ]
    )
    item_id = str(item.get("id") or "")
    if not item_id:
        raise SyncError("Smoke test failed: gh project item-create returned no id")

    archive_cmd = ["gh", "project", "item-archive", str(priv_num), "--owner", owner, "--id", item_id]
    try:
        found = None
        for _attempt in range(5):
            items = _item_list(priv_num, owner, limit=5000)
            found = next((it for it in items if str(it.get("id")) == item_id), None)
            if found:
                break
            time.sleep(0.5)
        if not found:
            raise SyncError("Smoke test failed: created item not found in item-list")
        content = found.get("content") or {}
        draft_id = str(content.get("id") or "")
        if not draft_id.startswith("DI_"):
            raise SyncError("Smoke test failed: DraftIssue content.id (DI_...) not available from item-list")
        _run_checked(["gh", "project", "item-edit", "--id", draft_id, "--title", title, "--body", "ok2"])
    except Exception as exc:
        try:
            _run_checked(archive_cmd)
        except Exception as arch_exc:
            raise SyncError(f"Smoke test failed and could not archive smoke item: {arch_exc}") from exc
        raise
    else:
        _run_checked(archive_cmd)


def _load_state(repo_root: Path) -> dict[str, dict[str, str]]:
    path = repo_root / STATE_FILE
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    keys = obj.get("keys")
    if isinstance(keys, dict):
        out: dict[str, dict[str, str]] = {}
        for k, v in keys.items():
            if isinstance(v, str):
                out[str(k)] = {"updatedAt": v, "state": ""}
            elif isinstance(v, dict):
                out[str(k)] = {
                    "updatedAt": str(v.get("updatedAt") or v.get("updated_at") or ""),
                    "state": str(v.get("state") or ""),
                }
        return out
    return {}


def _write_state(repo_root: Path, prev: dict[str, dict[str, str]], issues: Sequence[LinearIssue]) -> None:
    path = repo_root / STATE_FILE
    out: dict[str, dict[str, str]] = dict(prev)
    for it in issues:
        out[it.identifier] = {"updatedAt": it.updated_at or "", "state": it.state or ""}
    payload = {"syncedAt": _utc_now_iso(), "keys": out}
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        return


def cmd_sync(args: argparse.Namespace) -> int:
    owner = args.owner
    repo_root = _repo_root()
    version_str = _load_version_str(repo_root)

    _ensure_gh_projects_commands()
    _ensure_projects(owner)  # gate
    priv = _project_by_title(owner, PRIVATE_ARCHIVE_TITLE)
    priv_num = int(priv["number"])
    priv_id = str(priv["id"])

    fields = _field_map(priv_num, owner)
    if "VRX Linear Key" not in fields:
        raise SyncError("Private archive project missing required field: 'VRX Linear Key'")

    items = _item_list(priv_num, owner)
    mirrors = _mirror_items_by_key(items, "VRX Linear Key")
    dup = {k: v for k, v in mirrors.items() if len(v) > 1}
    if dup:
        msg = ["Duplicate mirror items detected (fatal):"]
        for k, lst in sorted(dup.items()):
            msg.append(f"- {k}: " + ", ".join(str(it.get('id')) for it in lst))
        raise SyncError("\n".join(msg))

    since_dt: Optional[_dt.datetime] = _parse_iso_dt(args.since) if getattr(args, "since", None) else None
    prev_state = _load_state(repo_root)

    projects = tuple(args.projects.split(",")) if args.projects else DEFAULT_PROJECTS
    projects = tuple(p.strip() for p in projects if p.strip())

    if args.linear_export:
        issues = _load_linear_export(Path(args.linear_export))
    else:
        key = os.environ.get("LINEAR_API_KEY", "")
        if not key:
            raise SyncError("LINEAR_API_KEY missing. Use --linear-export <path.json> for first run fallback.")
        client = LinearClient(key)
        issues = client.list_team_issues(DEFAULT_TEAM_NAME, limit=args.limit)

    in_scope = [i for i in issues if i.project in projects]
    if since_dt is not None:
        filtered: list[LinearIssue] = []
        for it in in_scope:
            if not it.updated_at:
                continue
            try:
                if _parse_iso_dt(it.updated_at) >= since_dt:
                    filtered.append(it)
            except SyncError:
                continue
        in_scope = filtered

    done_transitions: list[LinearIssue] = []
    if prev_state:
        for it in in_scope:
            prev = prev_state.get(it.identifier, {})
            if (prev.get("state") or "") != "Done" and it.state == "Done":
                done_transitions.append(it)
    elif since_dt is not None:
        for it in in_scope:
            if it.state == "Done":
                done_transitions.append(it)

    planned_creates = 0
    planned_updates = 0

    _smoke_draft_edit(priv_num, owner, apply=args.apply)
    for issue in sorted(in_scope, key=lambda x: x.identifier):
        snap = _snapshot(issue)
        urls = _extract_github_urls(issue.description)
        for _t, u in issue.attachments:
            urls.extend(_extract_github_urls(u))
        urls = sorted(set(urls))

        has_pr = _has_merged_pr(urls)
        mirror = mirrors.get(issue.identifier, [])
        action = "update" if mirror else "create"
        if action == "create":
            planned_creates += 1
        else:
            planned_updates += 1

        if not args.apply:
            print(f"[DRY-RUN] {action} mirror for {issue.identifier} ({issue.state})")
            continue

        # Create draft issue item (body includes sync block).
        title = f"{issue.identifier}: {issue.title}"
        body = f"{SYNC_BEGIN}\n{snap}\n{SYNC_END}\n"

        if action == "create":
            item = gh_json(
                [
                    "gh",
                    "project",
                    "item-create",
                    str(priv_num),
                    "--owner",
                    owner,
                    "--title",
                    title,
                    "--body",
                    body,
                    "--format",
                    "json",
                ]
            )
            item_id = str(item["id"])
            # Re-list items so we can get the DraftIssue content ID for future edits.
            items = _item_list(priv_num, owner)
            mirrors = _mirror_items_by_key(items, "VRX Linear Key")
        else:
            it = mirror[0]
            item_id = str(it["id"])
            content = it.get("content") or {}
            draft_id = str(content.get("id") or "")
            cur_body = str(content.get("body") or "")
            new_body = patch_sync_block(cur_body, snap)
            # Draft body edits require the DI_* ID, not the PVTI_* item ID.
            if not draft_id:
                raise SyncError(
                    f"Cannot update mirror for {issue.identifier}: DraftIssue content.id (DI_*) missing from item-list"
                )
            _run_checked(["gh", "project", "item-edit", "--id", draft_id, "--title", title, "--body", new_body])

        # Set required VRX Linear Key field (text) and best-effort other fields.
        _run_checked(
            [
                "gh",
                "project",
                "item-edit",
                "--id",
                item_id,
                "--project-id",
                priv_id,
                "--field-id",
                str(fields["VRX Linear Key"]["id"]),
                "--text",
                issue.identifier,
            ]
        )

        # Built-in Status (best-effort).
        if "Status" in fields:
            st_field = fields["Status"]
            opt_name = _linear_to_project_status(issue.state)
            opt_id = _status_option_id(st_field, opt_name)
            if opt_id:
                _run_checked(
                    [
                        "gh",
                        "project",
                        "item-edit",
                        "--id",
                        item_id,
                        "--project-id",
                        priv_id,
                        "--field-id",
                        str(st_field["id"]),
                        "--single-select-option-id",
                        opt_id,
                    ]
                )

        # VRX Archive Status (best-effort).
        if "VRX Archive Status" in fields:
            st_field = fields["VRX Archive Status"]
            name = _linear_to_archive_status(issue.state, has_merged_pr=has_pr)
            opt_id = _status_option_id(st_field, name)
            if opt_id:
                _run_checked(
                    [
                        "gh",
                        "project",
                        "item-edit",
                        "--id",
                        item_id,
                        "--project-id",
                        priv_id,
                        "--field-id",
                        str(st_field["id"]),
                        "--single-select-option-id",
                        opt_id,
                    ]
                )

        # VRX Lifecycle (best-effort).
        if "VRX Lifecycle" in fields:
            st_field = fields["VRX Lifecycle"]
            life = _linear_to_lifecycle(issue.state)
            if life:
                opt_id = _status_option_id(st_field, life)
                if opt_id:
                    _run_checked(
                        [
                            "gh",
                            "project",
                            "item-edit",
                            "--id",
                            item_id,
                            "--project-id",
                            priv_id,
                            "--field-id",
                            str(st_field["id"]),
                            "--single-select-option-id",
                            opt_id,
                        ]
                    )

        # Version impact (optional).
        if "VRX Version Impact" in fields and issue.project == "VRAXION_WALL":
            st_field = fields["VRX Version Impact"]
            opt_id = _status_option_id(st_field, "build")
            if opt_id:
                _run_checked(
                    [
                        "gh",
                        "project",
                        "item-edit",
                        "--id",
                        item_id,
                        "--project-id",
                        priv_id,
                        "--field-id",
                        str(st_field["id"]),
                        "--single-select-option-id",
                        opt_id,
                    ]
                )

        # Build version (optional; only if merged PR evidence exists).
        if "VRX Build Version" in fields and has_pr:
            _run_checked(
                [
                    "gh",
                    "project",
                    "item-edit",
                    "--id",
                    item_id,
                    "--project-id",
                    priv_id,
                    "--field-id",
                    str(fields["VRX Build Version"]["id"]),
                    "--text",
                    version_str,
                ]
            )

    print(f"Sync summary (projects={projects}): {len(in_scope)} issues in scope.")
    print(f"- creates: {planned_creates}")
    print(f"- updates: {planned_updates}")
    if done_transitions:
        print("\nDone transitions:")
        for it in sorted(done_transitions, key=lambda x: x.identifier):
            print(f"- {it.identifier}: {it.title}")
    else:
        print("\nDone transitions: none")
    if not args.apply:
        print("\nDRY-RUN: no writes performed. Re-run with --apply to modify the private archive project.")
    else:
        _write_state(repo_root, prev_state, in_scope)
    return 0


def cmd_prune_pr_items(args: argparse.Namespace) -> int:
    owner = args.owner

    _ensure_gh_projects_commands()
    _ensure_projects(owner)

    priv = _project_by_title(owner, PRIVATE_ARCHIVE_TITLE)
    priv_num = int(priv["number"])

    items = _item_list(priv_num, owner)
    mirrors = _mirror_items_by_key(items, "VRX Linear Key")
    dup = {k: v for k, v in mirrors.items() if len(v) > 1}
    if dup:
        msg = ["Duplicate mirror items detected (fatal):"]
        for k, lst in sorted(dup.items()):
            msg.append(f"- {k}: " + ", ".join(str(it.get("id")) for it in lst))
        raise SyncError("\n".join(msg))

    keys = [k.strip() for k in (args.linear_keys or "").split(",") if k.strip()]
    if not keys:
        raise SyncError("--linear-keys is required")

    pr_items_by_key: dict[str, list[Mapping[str, Any]]] = {}
    for it in items:
        content = it.get("content") or {}
        if content.get("type") != "PullRequest":
            continue
        key = _ci_get(it, "VRX Linear Key")
        if not key:
            continue
        pr_items_by_key.setdefault(str(key), []).append(it)

    for key in sorted(keys):
        mirror = mirrors.get(key)
        if not mirror:
            raise SyncError(f"Cannot prune PR items: mirror DraftIssue not found for {key}")
        mirror_it = mirror[0]
        content = mirror_it.get("content") or {}
        draft_id = str(content.get("id") or "")
        title = str(content.get("title") or "")
        body = str(content.get("body") or "")
        if not draft_id.startswith("DI_"):
            raise SyncError(f"Cannot prune PR items: DraftIssue DI_* id missing for {key}")

        prs = pr_items_by_key.get(key, [])
        pr_urls = sorted(
            {
                str((p.get("content") or {}).get("url") or "")
                for p in prs
                if (p.get("content") or {}).get("url")
            }
        )
        if not pr_urls:
            print(f"[DRY-RUN] no PR items to prune for {key}" if not args.apply else f"no PR items to prune for {key}")
            continue

        try:
            snap = _extract_delimited_block(body, SYNC_BEGIN, SYNC_END)
        except SyncError as exc:
            raise SyncError(f"{key}: mirror body missing/ambiguous sync block: {exc}") from exc

        new_snap = _merge_evidence_links(snap, pr_urls)
        new_body = patch_sync_block(body, new_snap)

        if not args.apply:
            print(
                f"[DRY-RUN] {key}: would merge {len(pr_urls)} PR url(s) into Evidence links and archive {len(prs)} PR item(s)"
            )
            continue

        _run_checked(["gh", "project", "item-edit", "--id", draft_id, "--title", title, "--body", new_body])
        for p in prs:
            _run_checked(["gh", "project", "item-archive", str(priv_num), "--owner", owner, "--id", str(p.get("id"))])

    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    owner = args.owner
    repo = args.repo
    issue_number = int(args.public_update_issue)
    keys = [k.strip() for k in (args.linear_keys or "").split(",") if k.strip()]
    if not keys:
        raise SyncError("--linear-keys is required")

    _ensure_gh_projects_commands()
    _ensure_gh_issue_commands()
    _ensure_projects(owner)

    priv = _project_by_title(owner, PRIVATE_ARCHIVE_TITLE)
    priv_num = int(priv["number"])

    items = _item_list(priv_num, owner)
    mirrors = _mirror_items_by_key(items, "VRX Linear Key")
    dup = {k: v for k, v in mirrors.items() if len(v) > 1}
    if dup:
        msg = ["Duplicate mirror items detected (fatal):"]
        for k, lst in sorted(dup.items()):
            msg.append(f"- {k}: " + ", ".join(str(it.get("id")) for it in lst))
        raise SyncError("\n".join(msg))

    lines: list[str] = []
    for key in sorted(keys):
        mirror = mirrors.get(key)
        if not mirror:
            raise SyncError(f"Cannot promote: mirror DraftIssue not found for {key}")
        it = mirror[0]
        content = it.get("content") or {}
        title = str(content.get("title") or it.get("title") or "")
        if title.startswith(f"{key}:"):
            short_title = title[len(key) + 1 :].strip()
        else:
            short_title = title

        body = str(content.get("body") or "")
        try:
            snap = _extract_delimited_block(body, SYNC_BEGIN, SYNC_END)
        except SyncError:
            snap = ""
        evidence_urls = _extract_github_urls(snap) or _extract_github_urls(body)
        evidence = ", ".join(evidence_urls) if evidence_urls else "—"
        lines.append(f"- {key}: {short_title} ({evidence})")

    block = "\n".join(lines).rstrip("\n")

    issue = gh_json(["gh", "issue", "view", str(issue_number), "-R", repo, "--json", "body,url,title"])
    cur_body = str(issue.get("body") or "")
    new_body = patch_delimited_block(cur_body, PUBLIC_UPDATE_BEGIN, PUBLIC_UPDATE_END, block)

    if not args.apply:
        print(
            f"[DRY-RUN] would update public issue #{issue_number} block ({PUBLIC_UPDATE_BEGIN}..{PUBLIC_UPDATE_END}) with {len(lines)} item(s)"
        )
        if not args.no_archive_private:
            print(f"[DRY-RUN] would archive {len(keys)} private mirror item(s) after promotion")
        return 0

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", newline="\n") as f:
        f.write(new_body)
        body_path = f.name

    try:
        _run_checked(["gh", "issue", "edit", str(issue_number), "-R", repo, "--body-file", body_path])
    finally:
        try:
            os.unlink(body_path)
        except OSError:
            pass

    if not args.no_archive_private:
        for key in keys:
            it = mirrors[key][0]
            _run_checked(["gh", "project", "item-archive", str(priv_num), "--owner", owner, "--id", str(it.get("id"))])

    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("sync", help="Mirror Linear tickets into the private archive project (draft items).")
    ps.add_argument("--owner", default=DEFAULT_OWNER)
    ps.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    ps.add_argument("--limit", type=int, default=500)
    ps.add_argument("--projects", default=",".join(DEFAULT_PROJECTS), help="Comma-separated Linear project names")
    ps.add_argument("--since", default=None, help="ISO timestamp filter for Linear updatedAt (also used for reporting when no state)")
    ps.add_argument("--linear-export", default=None, help="Fallback JSON export (first-run only)")
    ps.set_defaults(func=cmd_sync)

    pp = sub.add_parser("prune-pr-items", help="Archive PR items after mirrors exist (and merge PR links into mirror).")
    pp.add_argument("--owner", default=DEFAULT_OWNER)
    pp.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    pp.add_argument("--linear-keys", required=True, help="Comma-separated Linear keys to prune (e.g. VRA-30,VRA-31)")
    pp.set_defaults(func=cmd_prune_pr_items)

    pr = sub.add_parser("promote", help="Curated public update helper (explicit only).")
    pr.add_argument("--owner", default=DEFAULT_OWNER)
    pr.add_argument("--repo", default=DEFAULT_REPO)
    pr.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    pr.add_argument("--public-update-issue", required=True, help="Public update issue number (in --repo)")
    pr.add_argument("--linear-keys", required=True, help="Comma-separated Linear keys to include")
    pr.add_argument("--no-archive-private", action="store_true", help="Do not archive private mirror items after promotion")
    pr.set_defaults(func=cmd_promote)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except SyncError as exc:
        print(f"[vrx_sync] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
