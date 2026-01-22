# Defensive Publication (Prior Art)

Publication date: 2026-01-16  
Author/contact: kenessy.dani@gmail.com

This document is a public disclosure of the technical design implemented in
`tournament_phase6.py` within this repository. It is intended to establish
prior art for the described architecture and methods.

## Title
PRIME C-19 — Phase-Recurring Infinite Manifold Engine for Sequential Tasks

## Summary
The system implements a continuous pointer that moves on a circular memory
ring. At each timestep, a local neighborhood around the pointer is read using
a circular kernel (Gaussian or Von Mises). A recurrent update (e.g., GRU) is
applied to the readout, and the result is written back into the same ring
neighborhood. Pointer motion is controlled by learned signals and optional
stabilizers (inertia, deadzone, walk probability, phantom hysteresis). The
pointer uses circular (wrap-aware) interpolation so motion across the ring
seam is smooth and does not "teleport" across the ring.

The codename **PRIME C-19** also includes a custom activation function
("Candidate-19"), defined below.

## Hypothesis (Claim, Not Proof): Recursive Self-Monitoring
We disclose the following research hypothesis as prior art.

Claim (hypothesis): A persistent recurrent loop that predicts its own next
state and updates from the prediction error can yield measurable self-monitoring.
We further hypothesize that sufficiently stable self-monitoring may be a pathway
to machine self-conscious behavior. This is not yet proven.

Testable predictions (falsifiable):
1) The system can learn to predict its own next-step error better than chance.
2) After perturbations, the loop returns to baseline within a bounded recovery
   window (tau).
3) Error-prediction accuracy improves with deeper or longer loops even when
   task accuracy is held constant.

## Core Components

### 1) Circular Memory Ring
- Ring length: `L` (e.g., 2048 or 4096).
- Each ring slot stores a `slot_dim`-length vector.
- State tensor shape: `[batch, L, slot_dim]`.

### 2) Pointer State
- Continuous pointer `p` with domain `[0, L)`.
- Pointer values are treated on a ring (circular topology).
- Pointer values are maintained in float32 for numerical stability.

### 3) Circular Kernel Read/Write
At each timestep:
1. Select a window of integer indices around the pointer.
2. Compute a circular distance from each integer index to the pointer.
3. Compute kernel weights (Gaussian or Von Mises) from that distance.
4. Read a weighted sum from the ring using those weights.

Circular distance function:
```
delta(a, b, L) = remainder(b - a + L/2, L) - L/2
```

Gaussian weights (example):
```
logits = -(delta^2) / tau
weights = softmax(logits)
```

### 4) Recurrent Update and Writeback
Let `read` be the weighted read vector and `x_t` the input at time `t`.
The update is produced by a recurrent unit (e.g., GRU) and written back:
```
upd = GRU([x_t, read], read)
state = scatter_add(state, indices, weights * upd)
```
Optional state decay and clipping can be applied for stability.

### 5) Pointer Motion and Stabilizers
Pointer motion combines:
- A learned target pointer (jump proposal).
- A local walk component (increment on the ring).
- Stabilizers: inertia, deadzone, and optional gate.

All pointer blends are performed as circular interpolation along the shortest
arc on the ring:
```
circ_lerp(a, b, w, L) = remainder(a + w * delta(a, b, L), L)
```

### 6) Optional Hysteresis ("Phantom" Quantizer)
An optional quantizer can reduce flip-flopping by comparing two quantizations
offset by 0.5 and holding the previous discrete bin if they disagree. This
affects discrete pointer history/metrics while continuous pointer motion
remains unchanged.

### 8) Candidate-19 Activation (formerly "c-14")
The Candidate-19 activation is a piecewise, phase-recurring function that
maps input `u` using a repeating quadratic shape within alternating phases.

Definitions:
```
l = 6*pi
scaled = u / pi
n = floor(scaled)
t = scaled - n
h = t * (1 - t)
sgn = +1 if n is even else -1
core = pi * (sgn * h + rho * h * h)
```

Activation:
```
if u >= l:  u - l
elif u <= -l: u + l
else: core
```

Where `rho` is a tunable scalar (default 4.0 in the reference implementation).

### 7) Early-Exit ("Satiety") Semantics
If confidence exceeds a threshold, a sample can be marked inactive. When
inactive, pointer and state updates are frozen for that sample to prevent
further modification of memory or pointer position.

## Reference Implementation
See `tournament_phase6.py` in this repository for the complete, runnable
implementation, including training, evaluation, and logging.

## Intended Use
This system is designed for sequential tasks where pointer movement and memory
routing are learned jointly with task objectives. The architecture supports
continuous pointer motion and circular wraparound behavior.
