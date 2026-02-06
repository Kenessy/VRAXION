import json
import tempfile
import unittest
from pathlib import Path

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)


class TestAntRatioPacketV0(unittest.TestCase):
    def test_build_packet_basic_join(self) -> None:
        # Local import to keep sys.path behavior consistent with other Golden Draft tests.
        from tools.ant_ratio_packet_v0 import TokenBudget, build_packet

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            probe = root / "probe"
            assoc = root / "assoc"
            probe.mkdir()
            assoc.mkdir()

            (probe / "metrics.json").write_text(
                json.dumps(
                    {
                        "batch_size": 16,
                        "seq_len": 256,
                        "out_dim": 4,
                        "precision": "fp16",
                        "amp": 1,
                        "throughput_tokens_per_s": 123.0,
                        "throughput_samples_per_s": 0.5,
                        "peak_vram_reserved_bytes": 8_500_000_000,
                        "stability_pass": True,
                        "had_oom": False,
                        "had_nan": False,
                        "had_inf": False,
                        "fail_reasons": [],
                        "workload_id": "wl_x",
                        "probe_id": "probe_x",
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (probe / "env.json").write_text(json.dumps({"total_vram_bytes": 10_000_000_000}), encoding="utf-8")
            (probe / "run_cmd.txt").write_text(
                json.dumps(
                    {
                        "canonical_spec": {
                            "ant_spec": {"ring_len": 2048, "slot_dim": 256},
                            "colony_spec": {"batch_size": 16, "seq_len": 256},
                        }
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            (assoc / "report.json").write_text(
                json.dumps(
                    {"settings": {"eval_disjoint": True}, "eval": {"eval_acc": 0.75, "eval_n": 1234}},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            tb = TokenBudget(token_budget=256 * 16 * 10, min_steps=1, max_steps=10_000)
            pkt = build_packet(probe_run_root=probe, assoc_run_root=assoc, token_budget=tb)

            self.assertEqual(pkt["schema_version"], "ant_ratio_packet_v0")
            self.assertEqual(pkt["ant_tier"], "small")
            self.assertEqual(pkt["ant_body_cells"], 2048 * 256)
            self.assertAlmostEqual(pkt["ant_body_scale_vs_small"], 1.0, places=9)
            self.assertEqual(pkt["expert_heads"], 4)
            self.assertEqual(pkt["batch_size"], 16)
            self.assertEqual(pkt["seq_len"], 256)
            self.assertAlmostEqual(pkt["vram_ratio_reserved"], 0.85, places=6)
            self.assertEqual(pkt["stability_pass"], True)
            self.assertEqual(pkt["assoc_eval_disjoint"], True)
            self.assertAlmostEqual(pkt["assoc_byte_disjoint_accuracy"], 0.75, places=9)
            self.assertEqual(pkt["assoc_eval_n"], 1234)
            # token budget metadata present
            self.assertEqual(pkt["token_budget"], tb.token_budget)
            self.assertIsInstance(pkt["token_budget_steps"], int)


if __name__ == "__main__":
    unittest.main()
