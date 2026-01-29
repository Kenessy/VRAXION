"""Additional behavior locks for :mod:`vraxion.instnct.sharding`.

These tests complement the broader sharding coverage by pinning down a few
properties that are easy to break during refactors:

- tie-breaking (when two shard counts are equally close)
- NaN clamp semantics
- determinism (same inputs -> same outputs)
- input coercion (batch_size coerced via int)
"""

from __future__ import annotations

import math
import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from vraxion.instnct.sharding import calculate_adaptive_vasc


class ShardingAdditionalTests(unittest.TestCase):
    def test_vasc_negative_batch_size_short_circuit(self) -> None:
        outtup = calculate_adaptive_vasc(
            batch_size=-7,
            dwell=123.0,
            grad_norm=456.0,
            max_dwell_limit=10.0,
            ema_grad_norm=1.0,
            min_group_ratio=0.5,
        )
        self.assertEqual(outtup, (1, 0, 0.0, 0.0, 0.0))

    def test_vasc_tie_breaking_prefers_smaller_divisor(self) -> None:
        # Construct a case where the target shard count is exactly 4.5 for a
        # batch size of 18. Divisors 3 and 6 are equally close; legacy behavior
        # prefers the smaller divisor (3).
        outtup = calculate_adaptive_vasc(
            batch_size=18,
            dwell=0.0,
            grad_norm=1.0,
            max_dwell_limit=10.0,
            ema_grad_norm=1.0,
            min_group_ratio=0.2222222222222222,
        )
        self.assertEqual(int(outtup[0]), 3)
        self.assertEqual(int(outtup[1]), 6)

    def test_vasc_nan_clamp_order_is_legacy_compatible(self) -> None:
        # The legacy min/max clamping pattern turns NaNs into boundary values.
        outtup = calculate_adaptive_vasc(
            batch_size=10,
            dwell=float("nan"),
            grad_norm=0.0,
            max_dwell_limit=1.0,
            ema_grad_norm=1.0,
            min_group_ratio=0.02,
        )
        self.assertFalse(math.isnan(float(outtup[2])))
        self.assertEqual(outtup, (1, 10, 1.0, 0.0, 1.0))

    def test_vasc_is_deterministic_for_same_inputs(self) -> None:
        argmap = dict(
            batch_size=64,
            dwell=0.25,
            grad_norm=0.75,
            max_dwell_limit=2.0,
            ema_grad_norm=2.0,
            min_group_ratio=0.02,
        )
        outone = calculate_adaptive_vasc(**argmap)
        outtwo = calculate_adaptive_vasc(**argmap)
        self.assertEqual(outone, outtwo)

    def test_vasc_batch_size_is_coerced_via_int(self) -> None:
        outflt = calculate_adaptive_vasc(
            batch_size=16.9,  # intentionally non-integer
            dwell=0.25,
            grad_norm=0.75,
            max_dwell_limit=2.0,
            ema_grad_norm=2.0,
            min_group_ratio=0.02,
        )
        outint = calculate_adaptive_vasc(
            batch_size=16,
            dwell=0.25,
            grad_norm=0.75,
            max_dwell_limit=2.0,
            ema_grad_norm=2.0,
            min_group_ratio=0.02,
        )
        self.assertEqual(outflt, outint)

    def test_vasc_nan_tension_clamps_as_legacy(self) -> None:
        outtup = calculate_adaptive_vasc(
            batch_size=8,
            dwell=float("nan"),
            grad_norm=float("nan"),
            max_dwell_limit=1.0,
            ema_grad_norm=1.0,
            min_group_ratio=0.02,
        )
        self.assertEqual(outtup, (8, 1, 1.0, 1.0, 0.0))

    def test_vasc_invariants_hold_for_small_range(self) -> None:
        for bssizx in range(1, 65):
            with self.subTest(batch_size=bssizx):
                outtup = calculate_adaptive_vasc(
                    batch_size=bssizx,
                    dwell=0.1,
                    grad_norm=0.2,
                    max_dwell_limit=1.0,
                    ema_grad_norm=1.0,
                    min_group_ratio=0.02,
                )
                shdcnt = int(outtup[0])
                grpsiz = int(outtup[1])

                self.assertGreaterEqual(shdcnt, 1)
                self.assertLessEqual(shdcnt, bssizx)
                self.assertEqual(bssizx % shdcnt, 0)
                self.assertEqual(grpsiz * shdcnt, bssizx)
                self.assertGreaterEqual(grpsiz, 1)

                self.assertGreaterEqual(float(outtup[2]), 0.0)
                self.assertLessEqual(float(outtup[2]), 1.0)
                self.assertGreaterEqual(float(outtup[3]), 0.0)
                self.assertLessEqual(float(outtup[3]), 1.0)
                self.assertGreaterEqual(float(outtup[4]), 0.0)
                self.assertLessEqual(float(outtup[4]), 1.0)


if __name__ == "__main__":
    unittest.main()
