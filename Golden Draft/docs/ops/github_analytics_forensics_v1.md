# VRAXION GitHub Analytics Forensics v1

Generated: 2026-02-04 (UTC)
Repo: `Kenessy/VRAXION` (public)
Repo (current): `VRAXION/VRAXION`
Data window: GitHub "Traffic" API (last ~14 days only; daily aggregates)

## Executive Summary

**No clear suspicious pattern detected** in the available GitHub traffic telemetry.

What *looks* alarming at first glance is the clone volume (`674` clones / `303` uniques) compared to modest web views (`26` / `18` uniques). However, the **largest clone spikes align with heavy bursts of GitHub Actions runs** (primarily "pages build and deployment"), which is a strong "self-inflicted automation" explanation.

Important limitation: GitHub traffic endpoints **do not provide identity, IP, UA, or per-request logs**, so this cannot conclusively prove intent or detect "reverse engineering." It can only flag anomalies worth deeper investigation.

## What We Pulled (Max Available via GitHub API)

Traffic (last 14 days):
- `repos/<repo>/traffic/views` (daily)
- `repos/<repo>/traffic/clones` (daily)
- `repos/<repo>/traffic/popular/referrers`
- `repos/<repo>/traffic/popular/paths`

Releases:
- `repos/<repo>/releases` (asset download counts, if assets exist)

Automation context:
- `repos/<repo>/actions/workflows`
- `repos/<repo>/actions/runs?per_page=100` (then counted per day)

Repo metadata:
- `repos/<repo>` (stars, forks, watchers, timestamps, open issues)

## Reality Snapshot

Repo basics (at time of report):
- Stars: 0
- Forks: 0
- Watchers/subscribers: 0
- Open issues: 1 (public heartbeat issue `#8`)

Releases:
- `v1.0.0` exists, but **no release assets**. So there is **no "download_count"** telemetry beyond clones/views.

## Joined Daily Table (Clones / Views / Actions)

Notes:
- `views` are web UI page views, not clones.
- `clones` are git clone operations as counted by GitHub traffic.
- `actions_runs` is the count of Actions workflow runs created that day (mostly GitHub Pages deploy workflow in this repo).
- A "spike score" > ~3 is notable.

Median daily clones (window): `16.5` (used for spike score).

```
date        clones  uniq  views  vuniq  actions  clones/views  clones/actions  spike
----------  -----  ----  -----  -----  -------  -----------  -------------  -----
2026-01-21     26    16      4      4        0         6.5            n/a   1.58
2026-01-22    225   105      5      2       31        45.0           7.26  13.64  <-- spike
2026-01-23     24    17      3      3        2         8.0          12.00  1.45
2026-01-24     19    12      6      3        2         3.2           9.50  1.15
2026-01-25     45    23      2      2        4        22.5          11.25  2.73
2026-01-26    117    42      0      0       12         n/a           9.75  7.09  <-- spike
2026-01-27      3     3      1      1        0         3.0            n/a   0.18
2026-01-28      4     4      4      2        0         1.0            n/a   0.24
2026-01-29    176    98      0      0       25         n/a           7.04  10.67 <-- spike
2026-01-30     12     9      0      0        1         n/a          12.00  0.73
2026-01-31     14    11      0      0        1         n/a          14.00  0.85
2026-02-01      6     5      0      0        0         n/a            n/a   0.36
2026-02-02      2     2      1      1        0         2.0            n/a   0.12
2026-02-03      1     1      0      0        0         n/a            n/a   0.06
```

### Interpretation

The three big spikes (`2026-01-22`, `2026-01-26`, `2026-01-29`) all coincide with **high Actions run volume**:
- 2026-01-22: 31 runs
- 2026-01-26: 12 runs
- 2026-01-29: 25 runs

This is consistent with "automation noise" rather than a stealthy, low-signal external exfiltration pattern.

## Referrers + Paths (Quick Signal)

Top referrers (web):
- `github.com` (6)
- `Google` (1)

Top paths:
- repo root `/VRAXION/VRAXION` dominates
- a few doc/code paths (README, DEFENSIVE_PUBLICATION, discussions, etc.)

### Interpretation

No unknown referrers and no single deep code path dominating suggests normal browsing/reading, not targeted scraping of a specific file.

## Automation Context (Actions)

Workflows present:
- `pages-build-deployment` (active)

The Actions history shows repeated "pages build and deployment" runs, including bursts on the same days as the clone spikes.

## Adversarial Heuristics (Flags)

We flag a day as suspicious if any of these hit:
1) `clones >= 50` AND `actions_runs <= 2`
2) referrers include unknown/high-volume sources (not github.com/google)
3) top paths show unusual targeting (one deep file dominates)
4) sustained step-change in clones for several days without matching pushes/actions/releases

### Results

- Heuristic (1): **No hits** (no high-clone day with low Actions volume).
- Heuristic (2): **No hits** (only github.com + Google).
- Heuristic (3): **No hits** (root path dominates; no single "sensitive" deep file dominates).
- Heuristic (4): **No hits** (spikes are bursty and align with automation).

## What This Does *Not* Tell Us

Even with "no flags," you should assume:
- Public repo code can be cloned by anyone at any time.
- GitHub traffic does not provide enough detail to attribute clones to specific actors or confirm malicious intent.

If you truly need to prevent copying/reverse engineering, analytics won’t do it; access control does (private repo / split private core).

## Recommendations

### Security posture (if you care about theft)
- Treat public repo as cloneable by definition.
- If you want secrecy:
  - make the repo private, OR
  - split the "secret sauce" into a private repo/submodule and publish only the public surface.
- LICENSE exists already (good).
- Add `SECURITY.md` (recommended): how to report vulnerabilities; sets expectations and gives you a standard intake path.

### Monitoring (reduce ambiguity)
GitHub traffic data only retains ~14 days, so **you must snapshot daily** if you want long-term trend detection.

Minimal approach:
- Once per day, store:
  - raw JSON from `views`, `clones`, `popular/referrers`, `popular/paths`
  - a derived CSV like the daily table above
- Alert only on the strongest signal:
  - `clones >= 50` AND `actions_runs <= 2`

Where to store snapshots:
- locally under `bench_vault/` (safe if that folder is already a local-only artifact store), OR
- in a private repo (best if you want history + durability).

### Optional follow-on ticket (recommended)
Create a small script + scheduler:
- Script: `tools/github_traffic_snapshot.py`
  - uses `gh api` to pull endpoints + writes JSON/CSV
- Scheduler: Windows Task Scheduler daily at a fixed time
- Output folder: `bench_vault/github_traffic/YYYY-MM-DD/`

## How To Regenerate (Commands)

These are the exact endpoints used:

```bash
gh api repos/VRAXION/VRAXION

gh api repos/VRAXION/VRAXION/traffic/views
gh api repos/VRAXION/VRAXION/traffic/clones
gh api repos/VRAXION/VRAXION/traffic/popular/referrers
gh api repos/VRAXION/VRAXION/traffic/popular/paths

gh api repos/VRAXION/VRAXION/releases --paginate

gh api repos/VRAXION/VRAXION/actions/workflows
gh api repos/VRAXION/VRAXION/actions/runs?per_page=100
```
