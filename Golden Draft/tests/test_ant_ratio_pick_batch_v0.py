import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)


class TestAntRatioPickBatchV0(unittest.TestCase):
    def test_pick_batch_hits_target_band(self) -> None:
        from tools.ant_ratio_pick_batch_v0 import ProbeObservation, pick_batch_for_target

        # Monotonic "oracle": ratio increases with batch; FAIL once ratio crosses 0.92.
        def oracle(b: int) -> ProbeObservation:
            ratio = 0.10 + 0.03 * float(b)
            stability = ratio < 0.92
            return ProbeObservation(
                batch=int(b),
                run_root=f"run/B{b}",
                vram_ratio_reserved=ratio,
                stability_pass=bool(stability),
                fail_reasons=[] if stability else ["vram_guard"],
            )

        res = pick_batch_for_target(
            eval_at_batch=oracle,
            target_ratio=0.85,
            accept_low=0.82,
            accept_high=0.88,
            max_calls=10,
        )

        self.assertFalse(res.unusable)
        self.assertIsNotNone(res.chosen_batch)
        self.assertIsNotNone(res.chosen_ratio)
        self.assertGreaterEqual(res.chosen_ratio, 0.82)
        self.assertLessEqual(res.chosen_ratio, 0.88)

    def test_non_monotonic_aborts_refine(self) -> None:
        from tools.ant_ratio_pick_batch_v0 import ProbeObservation, pick_batch_for_target

        # Construct a non-monotonic ratio sequence.
        ratios = {1: 0.10, 2: 0.50, 4: 0.40, 8: 0.60}

        def oracle(b: int) -> ProbeObservation:
            ratio = ratios.get(int(b), 0.60)
            return ProbeObservation(
                batch=int(b),
                run_root=f"run/B{b}",
                vram_ratio_reserved=float(ratio),
                stability_pass=True,
                fail_reasons=[],
            )

        res = pick_batch_for_target(
            eval_at_batch=oracle,
            target_ratio=0.55,
            accept_low=0.40,
            accept_high=0.70,
            max_calls=6,
        )
        self.assertIn("non_monotonic_ratio_detected; abort refinement", res.notes)
        self.assertIsNotNone(res.chosen_batch)


if __name__ == "__main__":
    unittest.main()

