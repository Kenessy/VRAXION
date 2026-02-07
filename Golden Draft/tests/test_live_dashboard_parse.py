import unittest
import tempfile
from pathlib import Path
import importlib.util


class TestLiveDashboardParse(unittest.TestCase):
    @staticmethod
    def _load_module():
        module_path = Path(__file__).resolve().parents[1] / "tools" / "live_dashboard.py"
        spec = importlib.util.spec_from_file_location("live_dashboard_under_test", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load module: {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_parse_log_lines_attaches_grad_to_next_step_only(self):
        mod = self._load_module()

        lines = [
            "grad_norm(theta_ptr)=2.0\n",
            "step 1 | loss 1.0 | raw_delta=0.5 shard=1/8, traction=0.1\n",
            "step 2 | loss 2.0 | raw_delta=0.6 shard=1/8\n",
        ]
        rows = mod.parse_log_lines(lines)
        self.assertEqual(len(rows), 2)

        self.assertEqual(rows[0]["step"], 1)
        self.assertAlmostEqual(rows[0]["grad_norm"], 2.0)

        # The grad should not carry over.
        self.assertEqual(rows[1]["step"], 2)
        self.assertIsNone(rows[1]["grad_norm"])

    def test_parse_log_file_missing_returns_empty(self):
        mod = self._load_module()

        self.assertEqual(mod.parse_log_file("/path/does/not/exist.log"), [])

    def test_parse_log_lines_supports_v_cog_rd_format(self):
        mod = self._load_module()

        lines = [
            "grad_norm(theta_ptr)=1.0152e+00\n",
            (
                "synth | absolute_hallway | step 0060 | loss 6.3859 | t=278.7s | "
                "V_COG[PRGRS:28.1% ORB:2 RD:1.18e+00 AC:1 VH:0.00] "
                "RAW[SCA:0.045 INR:0.91->0.91/F:0.00], shard=27/1, traction=18.12\n"
            ),
        ]
        rows = mod.parse_log_lines(lines)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 60)
        self.assertAlmostEqual(rows[0]["loss"], 6.3859, places=4)
        self.assertAlmostEqual(rows[0]["raw_delta"], 1.18, places=2)
        self.assertEqual(rows[0]["shard_count"], 27.0)
        self.assertEqual(rows[0]["shard_size"], 1.0)
        self.assertAlmostEqual(rows[0]["traction"], 18.12, places=2)
        self.assertAlmostEqual(rows[0]["grad_norm"], 1.0152, places=4)

    def test_collect_live_status_reads_sibling_tails(self):
        mod = self._load_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "child_stdout.log"
            err_path = root / "child_stderr.log"
            sup_path = root / "supervisor.log"
            log_path.write_text("", encoding="utf-8")
            err_path.write_text("probe warning\n", encoding="utf-8")
            sup_path.write_text("[supervisor] launch\n", encoding="utf-8")

            status = mod.collect_live_status(str(log_path))
            self.assertTrue(status["log_exists"])
            self.assertGreaterEqual(int(status["log_size_bytes"]), 0)
            self.assertIn("probe warning", "\n".join(status["stderr_tail"]))
            self.assertIn("[supervisor] launch", "\n".join(status["supervisor_tail"]))


if __name__ == "__main__":
    unittest.main()
