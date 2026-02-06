# VRAXION Wiki + Pages Red-Team Pass (2026-02-06)

Goal: a skeptical competitor dev (OpenAI/Gemini) should have a hard time finding “gotchas” via drift, broken surfaces, or misquotable ambiguity.

## Threat model checklist (what an adversary will try)

### Drift / contradictions
- Home says “status lives elsewhere”, but an embedded diagram claims *current* status.
- “Locked” Home still changes implicitly (via assets / meta / external links).

### Broken surfaces
- 404s (Pages domain drift, missing favicon/og:image, broken raw SVG URLs).
- stale owner links (`Kenessy/*`).

### Misquote vectors
- “You only optimize throughput” (if objective is presented as universal).
- “Consciousness claims” or capability claims without artifacts (violates epistemic boundary).

## Findings (before fixes)
- **Milestone SVG drift (high risk):** `docs/assets/vraxion_phases.svg` contained live-status phrasing:
  - `System status`
  - `Active milestone`
  - `Active focus`
- **Pages meta drift (high risk):** `docs/index.html` + `README.md` referenced `kenessy.github.io/VRAXION/` and missing assets:
  - favicon `assets/vraxion-mark.svg` (missing)
  - og:image `assets/banner-dark.svg` (missing)
- **Misquote risk (medium):** Home’s “Evidence contract” objective could be read as the *only* objective.

## Fixes applied
- **Milestone SVG made timeless:** `docs/assets/vraxion_phases.svg` no longer claims current status and no longer uses “complete/in-progress” legend text.
- **Pages landing hardened:** `README.md` + `docs/index.html` now use `https://vraxion.github.io/VRAXION/`, and `og:image` + favicon point to an existing asset (`assets/vraxion_logo.svg`).
- **Evidence-contract framing tightened (wiki):** Home now states the throughput objective is the **GPU Chapter 1 baseline**; other evaluations must define their own objective + gates + artifacts.

## Guardrails added (CI)
- `Golden Draft/tools/wiki_health_check.py`
  - existing: banned-link scan + raw SVG URL HTTP checks + wikilink resolution + locked-Home structure
  - **new:** milestone SVG drift scan (banned phrases) against the repo asset
- `Golden Draft/tools/pages_health_check.py`
  - deterministic (no-network) Pages checks:
    - no `kenessy.github.io/VRAXION` in `README.md` or `docs/index.html`
    - `og:url` must be `https://vraxion.github.io/VRAXION/`
    - favicon + og:image assets must exist under `docs/`
- `.github/workflows/ci.yml`
  - new job: `docs-health` runs `pages_health_check.py`

## Acceptance / regression tests
- `python "Golden Draft/tools/wiki_health_check.py"` passes.
- `python "Golden Draft/tools/pages_health_check.py"` passes.
- Negative tests (expected CI failure):
  - remove `HOME_LOCKED_V2` from wiki Home
  - reintroduce `kenessy.github.io/VRAXION` into `README.md` or `docs/index.html`
  - put `System status` back into `docs/assets/vraxion_phases.svg`
  - point favicon to a missing file

