import unittest

from tools.env_utils import env_bool, env_float, env_int, env_str, parse_bool


class TestEnvUtils(unittest.TestCase):
    def test_env_str_missing(self) -> None:
        env = {}
        self.assertIsNone(env_str(env, "X"))
        self.assertEqual(env_str(env, "X", default="d"), "d")

    def test_env_str_empty_treated_as_none(self) -> None:
        env = {"X": "   "}
        self.assertIsNone(env_str(env, "X"))
        self.assertEqual(env_str(env, "X", default="d"), "d")

    def test_parse_bool_truthy_falsy(self) -> None:
        for val in ["1", "true", "TRUE", " yes ", "On", "y"]:
            got, issue = parse_bool(val, default=False)
            self.assertTrue(got)
            self.assertIsNone(issue)

        for val in ["0", "false", "FALSE", " no ", "Off", "n"]:
            got, issue = parse_bool(val, default=True)
            self.assertFalse(got)
            self.assertIsNone(issue)

    def test_parse_bool_unknown_non_strict(self) -> None:
        got, issue = parse_bool("maybe", default=True, strict=False)
        self.assertTrue(got)
        self.assertIsNotNone(issue)

    def test_env_bool_returns_issue(self) -> None:
        env = {"B": "maybe"}
        got, issue = env_bool(env, "B", default=False)
        self.assertFalse(got)
        self.assertIsNotNone(issue)
        self.assertEqual(issue.key, "B")

    def test_env_int_parses_and_bounds(self) -> None:
        env = {"I": "42"}
        got, issue = env_int(env, "I", default=0)
        self.assertEqual(got, 42)
        self.assertIsNone(issue)

        got, issue = env_int({"I": "oops"}, "I", default=7)
        self.assertEqual(got, 7)
        self.assertIsNotNone(issue)

        got, issue = env_int({"I": "-1"}, "I", default=7, min_value=0)
        self.assertEqual(got, 7)
        self.assertIsNotNone(issue)

    def test_env_float_parses(self) -> None:
        got, issue = env_float({"F": "0.5"}, "F", default=1.0)
        self.assertAlmostEqual(got, 0.5)
        self.assertIsNone(issue)

        got, issue = env_float({"F": "oops"}, "F", default=1.0)
        self.assertAlmostEqual(got, 1.0)
        self.assertIsNotNone(issue)


if __name__ == "__main__":
    unittest.main()

