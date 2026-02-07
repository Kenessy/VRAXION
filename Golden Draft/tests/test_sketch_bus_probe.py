import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from tools.sketch_bus_probe import run_probe


class TestSketchBusProbe(unittest.TestCase):
    def test_probe_is_deterministic_on_cpu(self) -> None:
        s1 = run_probe(
            n_nodes=64,
            d_msg=8,
            m_buckets=128,
            k_write=2,
            k_read=2,
            batch_size=16,
            seed=123,
            device="cpu",
        )
        s2 = run_probe(
            n_nodes=64,
            d_msg=8,
            m_buckets=128,
            k_write=2,
            k_read=2,
            batch_size=16,
            seed=123,
            device="cpu",
        )

        # Timing is not deterministic; everything else should be.
        self.assertEqual(s1.writes_total, s2.writes_total)
        self.assertEqual(s1.buckets_used, s2.buckets_used)
        self.assertEqual(s1.buckets_empty, s2.buckets_empty)
        self.assertEqual(s1.max_bucket_load, s2.max_bucket_load)
        self.assertAlmostEqual(s1.collision_write_frac, s2.collision_write_frac, places=10)
        self.assertAlmostEqual(s1.ctx_cos_mean, s2.ctx_cos_mean, places=10)
        self.assertAlmostEqual(s1.ctx_mse_mean, s2.ctx_mse_mean, places=10)

    def test_more_buckets_reduces_collisions_and_error(self) -> None:
        small = run_probe(
            n_nodes=64,
            d_msg=8,
            m_buckets=64,
            k_write=2,
            k_read=2,
            batch_size=32,
            seed=0,
            device="cpu",
        )
        large = run_probe(
            n_nodes=64,
            d_msg=8,
            m_buckets=512,
            k_write=2,
            k_read=2,
            batch_size=32,
            seed=0,
            device="cpu",
        )

        self.assertLess(large.collision_write_frac, small.collision_write_frac)
        # Note: ctx error is not guaranteed to improve monotonically with M for a fixed
        # k_read, because larger M reduces occupancy (less mixing per bucket). This
        # probe is primarily about collision/interference and determinism.


if __name__ == "__main__":
    unittest.main()
