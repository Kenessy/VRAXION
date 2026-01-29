"""Batch sharding helpers.

Only a single public function is exposed: :func:`calculate_adaptive_vasc`.

The surrounding repository uses this helper to choose a *shard count* that:
- cleanly divides the batch size (no remainder), and
- shifts between fewer/larger shards and more/smaller shards based on a
  scale-free cohesion signal.

Behavior is "golden" and locked by ``tests/verify_golden.py``.
"""

from __future__ import annotations

import math
from typing import Tuple


EPSVAL = 1e-8


def _clamp01(valsix: float) -> float:
    """Clamp a value to the inclusive range [0.0, 1.0].

    This uses the same min/max clamping pattern as the legacy implementation,
    including its handling of NaNs.
    """

    return max(0.0, min(1.0, valsix))


def _pick_divs(numsix: int, targsx: float) -> int:
    """Pick the divisor of ``numsix`` closest to ``targsx``.

    The legacy version built a list of all divisors in ascending order and
    returned ``min(divs, key=lambda d: abs(d - targsx))``. When multiple divisors
    are equally close, ``min`` returns the first one, i.e. the *smaller* divisor.

    This implementation reproduces that choice (including tie-breaking) while
    avoiding an O(n) scan over the full ``range(1, numsix + 1)``.
    """

    bestdv = 1
    bestdf = abs(1 - targsx)

    rootxx = math.isqrt(numsix)
    for divsix in range(1, rootxx + 1):
        if numsix % divsix != 0:
            continue

        lowdiv = divsix
        higdiv = numsix // divsix

        for candiv in (lowdiv, higdiv):
            difval = abs(candiv - targsx)
            if difval < bestdf or (difval == bestdf and candiv < bestdv):
                bestdf = difval
                bestdv = candiv

    return bestdv


def calculate_adaptive_vasc(
    batch_size: int,
    dwell: float,
    grad_norm: float,
    max_dwell_limit: float,
    ema_grad_norm: float,
    *,
    min_group_ratio: float = 0.02,
) -> Tuple[int, int, float, float, float]:
    """Scale-free VASC: choose shard count from ratio-based cohesion signals.

    Returns:
        (shard_count, group_size, focus, tension, cohesion)

    Notes:
        This function's numeric behavior is intentionally stable.
    """

    batint = int(batch_size)
    if batint <= 0:
        return 1, 0, 0.0, 0.0, 0.0

    dwlmax = max(EPSVAL, float(max_dwell_limit))
    emaval = max(EPSVAL, float(ema_grad_norm))

    focval = _clamp01(float(dwell) / dwlmax)
    tenval = _clamp01(float(grad_norm) / (emaval + EPSVAL))
    cohval = _clamp01(focval - tenval)

    ceilng = float(batint)
    ratval = float(min_group_ratio)
    floors = max(1.0, ceilng * ratval)

    targsz = floors + (ceilng - floors) * cohval
    rawshd = ceilng / max(EPSVAL, targsz)

    shrcnt = _pick_divs(batint, rawshd)
    shrcnt = max(1, min(shrcnt, batint))
    grpsiz = batint // shrcnt

    return shrcnt, grpsiz, focval, tenval, cohval
