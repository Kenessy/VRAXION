import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.vcog_parse import OnlineStats, parse_line, parse_vcog_kv


class TestVcogParse(unittest.TestCase):
    def test_parse_vcog_kv_numeric_and_percent(self):
        blob = "PRGRS:100.0% ORB:2 RD:1.0e+00 AC:1 NOTE:abc"
        out = parse_vcog_kv(blob)

        self.assertIn("PRGRS", out)
        self.assertIn("ORB", out)
        self.assertIn("RD", out)
        self.assertIn("AC", out)
        self.assertIn("NOTE", out)

        self.assertEqual(out["PRGRS"], 100.0)
        self.assertEqual(out["ORB"], 2.0)
        self.assertEqual(out["RD"], 1.0)
        self.assertEqual(out["AC"], 1.0)
        self.assertEqual(out["NOTE"], "abc")

    def test_parse_line_step_loss_and_vcog(self):
        line = (
            "synth | demo | step 0001/0100 | loss 0.1234 | "
            "V_COG[PRGRS:100.0% ORB:2 RD:1.0e+00 AC:1]"
        )
        ev, vcog = parse_line(line)

        self.assertIsNotNone(ev)
        self.assertIsNotNone(vcog)
        assert ev is not None
        assert vcog is not None

        self.assertIn("ts_utc", ev)
        self.assertEqual(ev["step"], 1)
        self.assertAlmostEqual(ev["loss"], 0.1234, places=7)
        self.assertEqual(vcog["PRGRS"], 100.0)
        self.assertEqual(vcog["ORB"], 2.0)

    def test_parse_line_no_match(self):
        ev, vcog = parse_line("hello world")
        self.assertIsNone(ev)
        self.assertIsNone(vcog)

    def test_online_stats_ignores_nan_inf(self):
        st = OnlineStats()
        st.update(float("nan"))
        st.update(float("inf"))
        st.update(float("-inf"))
        st.update(1.0)
        st.update(3.0)

        dct = st.to_dict()
        self.assertEqual(dct["n"], 2)
        self.assertAlmostEqual(dct["mean"], 2.0, places=7)
        # sample std for [1,3] is sqrt(2)
        self.assertAlmostEqual(dct["std"], math.sqrt(2.0), places=7)
        self.assertEqual(dct["min"], 1.0)
        self.assertEqual(dct["max"], 3.0)

    def test_online_stats_empty(self):
        st = OnlineStats()
        dct = st.to_dict()
        self.assertEqual(dct["n"], 0)
        self.assertIsNone(dct["mean"])
        self.assertIsNone(dct["std"])
        self.assertIsNone(dct["min"])
        self.assertIsNone(dct["max"])


if __name__ == "__main__":
    unittest.main()
