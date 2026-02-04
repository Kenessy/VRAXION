# Ideas Inbox (DEV)

This file is a *notes-only* inbox for potentially-useful ideas discovered during
external brainstorming (e.g., Pulse items). Nothing here is assumed correct.

We only promote ideas into code after:
- deterministic A/B validation on our benchmarks, and
- adversarial review for hidden non-determinism / system-dependent behavior.

## Pulse triage (saved ideas)

### 2026-01-31 — "Phase-Lag Mixing and Jitter-Adaptive Stabilization"

**Keep (maybe later):**
- **Orthogonal micro-resets / re-orthogonalization of drifting subspaces.**
  - Concept: periodically re-orthogonalize a drifting sub-block to reset its
    eigenstructure without wiping the whole model.
  - Potential use-case: only if we see genuine instability (NaNs, exploding
    gradients, chaotic oscillations) that is *not* explainable by task difficulty
    or config mistakes.
  - Determinism note: any “power-iteration” / random-vector estimate would need
    fixed seeding or a deterministic proxy before we trust it in our scorecard.

**Reject for now (not aligned with current bottleneck):**
- **Phase-lagged fast/slow mixing:** likely increases blur/smoothing; our current
  hard retrieval/binding experiments appear to penalize smoothing.
- **Wall-clock-jitter adaptive grad clamp:** non-deterministic by design (depends
  on system load), and therefore unsuitable for our reproducible benchmark loop.

### 2026-01-31 — "Entropy Budgeting and Residual Cache Experiments"

**Keep (maybe later):**
- **Optimizer mismatch for routers/gates vs core weights.**
  - If/when we introduce *learned* routing/gating networks (vs today's mostly
    heuristic selection), consider separate optimizers or parameter groups:
    - core weights: AdamW
    - routing/gating weights: SGD+momentum (or lower-LR AdamW)
  - Goal: reduce router wobble/instability while letting core learn smoothly.

**Maybe (very experimental / careful gating required):**
- **Residual cache / retrieval cache (SSRC-like).**
  - Potential use-case: stabilization on near-repeats, or explicit episodic
    memory experiments.
  - High risk: can silently become a leakage path (train -> eval) unless the cache
    is scoped per-sequence or reset at eval boundaries; also adds KNN complexity.

**Reject for now (high risk of harming our current binding benchmarks):**
- **Per-layer entropy "budgeting" via temperature-cooling logits or channel micro-drop.**
  - This is effectively adding blur/noise at the output distribution level.
  - Our current failures are underfitting/binding-related (not runaway peaky
    collapse), so this is low-probability to help and can reduce the learning
    signal on discrete retrieval tasks.

### 2026-01-31 — "Prime-Length Epochs for Fractal Training Timing"

**Keep (maybe later / training hygiene):**
- **Prime-length micro-epochs / prime step spans** to reduce accidental resonance
  between multiple cadences (eval interval, checkpoint interval, curriculum
  schedules, etc.).
  - This is plausible as a *hygiene* tactic for long-running training loops that
    have several interacting periodic processes.
  - Determinism note: avoid `random.shuffle()` unless it is seeded and logged; a
    fixed prime-cycle is better for reproducible A/B.

**Maybe (depends on our checkpoint strategy):**
- **Adaptive checkpoint "density"** (save full state more often during instability,
  lighter/less often during calm phases).
  - Useful primarily for long runs with expensive optimizer state and heavy I/O.
  - Must be carefully designed so it does not become wallclock-dependent; use
    deterministic instability proxies (loss/grad deltas), not step-time jitter.

**Not relevant to our current bottleneck:**
- This will not directly address the current `assoc_byte --eval-disjoint`
  specialization/binding gap; treat as a later training-loop quality-of-life tool.

### 2026-01-31 — "Reweighted Ghost Block for Self-Sculpting Depth"

**Keep (later / if we formalize MoE-like gating):**
- **"Ghost" exploration mass on gates** (epsilon-uniform blend) to avoid hard
  expert collapse:
  - `p_tilde = (1-eps)*softmax(logits/T) + eps/K`
  - Useful if/when we have *learned* expert selection and we see persistent
    starvation/collapse.
- **Temperature schedule driven by a deterministic uncertainty proxy** (e.g.,
  validation entropy) to widen/narrow selection over training.
  - Caution: the schedule must be reproducible (same eval stream/seed), and it can
    easily become a feedback loop if eval is noisy.

**Reject / caution for now:**
- Adding validation-driven control loops during our core architecture debugging can
  hide the real failure mode (it “masks” issues with moving thresholds).

### 2026-01-31 — "Exploring the ReHU-Sigma Activation Curve"

**Keep (maybe later / only if we see true chatter/oscillation):**
- **Hysteretic activation / Schmitt-trigger-like gates** (separate engage/release
  thresholds) to reduce mode-flip chatter under noisy inputs.
  - This is potentially useful for *controller/gating* components where we want
    stable mode switching, not for core discrete retrieval dynamics.

**Reject for now (likely counterproductive for current binding tests):**
- The proposed formulation introduces an additional sticky state + EMA smoothing,
  which is effectively another low-pass filter. Our current retrieval/binding
  experiments tend to penalize blur/smoothing.
 - The “batch size swings” angle is not relevant to our deterministic benchmark
   setup (fixed batch size). Also, stateful activations can behave awkwardly with
   variable batch sizes if not carefully designed.

### 2026-01-31 — "Numerical Clamp for Sigmoid Stability"

**Keep (practical hardening):**
- **Clamp the pre-sigmoid input** (e.g., `z = clamp(z, -8, 8)`) and guard against
  divide-by-tiny scales when implementing sigmoid-style gates.
  - This is a safe, low-risk numeric stability hardening for controller-style
    gating (brainstem thresholds, mix weights, etc.).
  - Prefer vectorized/tensor-safe implementations (PyTorch clamp) and keep behavior
    deterministic.
