"""VRA-32 tests: gpu_capacity_probe harness (CPU-only, adversarial)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)
from conftest import temporary_env

from tools import gpu_capacity_probe


class TestGpuCapacityProbe(unittest.TestCase):
    def test_unknown_keys_fail_strict_validation(self) -> None:
        ant = {"ring_len": 32, "slot_dim": 16, "ptr_dtype": "fp32", "precision": "fp32", "unknown": 1}
        col = {"seq_len": 8, "synth_len": 8, "batch_size": 2, "ptr_update_every": 1, "state_loop_samples": 0}

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            with temporary_env(VRX_FORCE_DEVICE="cpu"):
                rc = gpu_capacity_probe.main(
                    [
                        "--ant",
                        json.dumps(ant),
                        "--colony",
                        json.dumps(col),
                        "--out-dim",
                        "1",
                        "--batch",
                        "2",
                        "--warmup-steps",
                        "0",
                        "--measure-steps",
                        "1",
                        "--precision",
                        "fp32",
                        "--amp",
                        "0",
                        "--output-dir",
                        str(out_dir),
                    ]
                )
        self.assertEqual(rc, 2)

    def test_precision_amp_validation(self) -> None:
        ant = {"ring_len": 32, "slot_dim": 16, "ptr_dtype": "fp32", "precision": "fp32"}
        col = {"seq_len": 8, "synth_len": 8, "batch_size": 2, "ptr_update_every": 1, "state_loop_samples": 0}

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            with temporary_env(VRX_FORCE_DEVICE="cpu"):
                rc = gpu_capacity_probe.main(
                    [
                        "--ant",
                        json.dumps(ant),
                        "--colony",
                        json.dumps(col),
                        "--out-dim",
                        "1",
                        "--batch",
                        "2",
                        "--warmup-steps",
                        "0",
                        "--measure-steps",
                        "1",
                        "--precision",
                        "fp32",
                        "--amp",
                        "1",
                        "--output-dir",
                        str(out_dir),
                    ]
                )
        self.assertEqual(rc, 2)

    def test_cpu_run_emits_artifacts_and_required_keys(self) -> None:
        ant = {"ring_len": 32, "slot_dim": 16, "ptr_dtype": "fp32", "precision": "fp32"}
        # Intentionally differ to verify accounting: seq_len != synth_len.
        col = {"seq_len": 8, "synth_len": 4, "batch_size": 2, "ptr_update_every": 1, "state_loop_samples": 0}

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "run"
            with temporary_env(VRX_FORCE_DEVICE="cpu"):
                rc = gpu_capacity_probe.main(
                    [
                        "--ant",
                        json.dumps(ant),
                        "--colony",
                        json.dumps(col),
                        "--out-dim",
                        "1",
                        "--batch",
                        "2",
                        "--warmup-steps",
                        "1",
                        "--measure-steps",
                        "2",
                        "--precision",
                        "fp32",
                        "--amp",
                        "0",
                        "--output-dir",
                        str(out_dir),
                    ]
                )

            self.assertEqual(rc, 0)
            for name in (
                gpu_capacity_probe.ART_RUN_CMD,
                gpu_capacity_probe.ART_ENV,
                gpu_capacity_probe.ART_METRICS_JSON,
                gpu_capacity_probe.ART_METRICS_CSV,
                gpu_capacity_probe.ART_SUMMARY,
            ):
                self.assertTrue((out_dir / name).exists(), msg=f"missing artifact: {name}")

            metrics = json.loads((out_dir / gpu_capacity_probe.ART_METRICS_JSON).read_text(encoding="utf-8"))
            required = {
                "batch_size",
                "seq_len",
                "warmup_steps",
                "measure_steps",
                "measure_wall_time_s",
                "median_step_time_s",
                "p95_step_time_s",
                "throughput_samples_per_s",
                "throughput_tokens_per_s",
                "peak_vram_reserved_bytes",
                "peak_vram_allocated_bytes",
                "had_oom",
                "had_nan",
                "had_inf",
                "stability_pass",
                "fail_reasons",
            }
            self.assertTrue(required.issubset(set(metrics.keys())))

            # Accounting correctness: x uses synth_len, tokens/sec uses seq_len.
            self.assertEqual(metrics["seq_len"], 8)
            self.assertEqual(metrics["synth_len"], 4)
            if metrics["throughput_samples_per_s"] is not None and metrics["throughput_tokens_per_s"] is not None:
                self.assertAlmostEqual(
                    float(metrics["throughput_tokens_per_s"]),
                    float(metrics["throughput_samples_per_s"]) * float(metrics["seq_len"]),
                    places=5,
                )

    def test_overwrite_guard(self) -> None:
        ant = {"ring_len": 32, "slot_dim": 16, "ptr_dtype": "fp32", "precision": "fp32"}
        col = {"seq_len": 8, "synth_len": 8, "batch_size": 2, "ptr_update_every": 1, "state_loop_samples": 0}

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "run"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "env.json").write_text("{}", encoding="utf-8")

            with temporary_env(VRX_FORCE_DEVICE="cpu"):
                rc = gpu_capacity_probe.main(
                    [
                        "--ant",
                        json.dumps(ant),
                        "--colony",
                        json.dumps(col),
                        "--out-dim",
                        "1",
                        "--batch",
                        "2",
                        "--warmup-steps",
                        "0",
                        "--measure-steps",
                        "1",
                        "--precision",
                        "fp32",
                        "--amp",
                        "0",
                        "--output-dir",
                        str(out_dir),
                    ]
                )
        self.assertEqual(rc, 2)

    def test_watchdog_threshold_function(self) -> None:
        self.assertEqual(gpu_capacity_probe.compute_stall_threshold_s(0.1), 60.0)
        self.assertEqual(gpu_capacity_probe.compute_stall_threshold_s(6.0), 60.0)
        self.assertEqual(gpu_capacity_probe.compute_stall_threshold_s(7.0), 70.0)


if __name__ == "__main__":
    unittest.main()

