# Ticket Pack: Eval/IQ Ladder v0 (2026-02-07)

This file records the GitHub issues created as part of the Deep-Research Pipeline v1 (Eval/IQ first).

Repo: VRAXION/VRAXION

## Created issues

- VRA-79 (issue #69): Eval ladder v0 (fast/medium/slow synthetic tasks for "smartness")
  - https://github.com/VRAXION/VRAXION/issues/69

- VRA-80 (issue #70): Intelligence frontier contract (quality_score_v0 + systems-vs-quality labeling)
  - https://github.com/VRAXION/VRAXION/issues/70

- VRA-81 (issue #71): eval_suite_v0 runner (deterministic report.json for IQ ladder)
  - https://github.com/VRAXION/VRAXION/issues/71

- VRA-82 (issue #72): Eval claim promotion gates (Supported vs Confirmed) + seed policy
  - https://github.com/VRAXION/VRAXION/issues/72

- VRA-83 (issue #73): Utilization/collapse metrics as first-class eval artifacts
  - https://github.com/VRAXION/VRAXION/issues/73

- VRA-84 (issue #74): Quality sweep posture v0 (coarse then confirm) scripted
  - https://github.com/VRAXION/VRAXION/issues/74

## Intended wiring (high-level)

1) VRA-79 defines task generators + scoring.
2) VRA-81 creates one deterministic runner that emits report.json.
3) VRA-80 defines how we summarize those reports into a stable `quality_score_v0` (and prevents “systems == IQ” misreads).
4) VRA-83 ensures utilization/collapse context is always available alongside quality.
5) VRA-82 defines Supported vs Confirmed and the seed posture.
6) VRA-84 makes the posture runnable and repeatable.

