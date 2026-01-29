import unittest


class TestLiveDashboardParse(unittest.TestCase):
    def test_parse_log_lines_attaches_grad_to_next_step_only(self):
        from tools.live_dashboard import parse_log_lines

        lines = [
            "grad_norm(theta_ptr)=2.0\n",
            "step 1 | loss 1.0 | raw_delta=0.5 shard=1/8, traction=0.1\n",
            "step 2 | loss 2.0 | raw_delta=0.6 shard=1/8\n",
        ]
        rows = parse_log_lines(lines)
        self.assertEqual(len(rows), 2)

        self.assertEqual(rows[0]["step"], 1)
        self.assertAlmostEqual(rows[0]["grad_norm"], 2.0)

        # The grad should not carry over.
        self.assertEqual(rows[1]["step"], 2)
        self.assertIsNone(rows[1]["grad_norm"])

    def test_parse_log_file_missing_returns_empty(self):
        from tools.live_dashboard import parse_log_file

        self.assertEqual(parse_log_file("/path/does/not/exist.log"), [])


if __name__ == "__main__":
    unittest.main()
