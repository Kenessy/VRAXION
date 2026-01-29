import unittest
from datetime import datetime, timezone

from tools.instnct_entrypoint import Action, EntrypointDeps, EnvKeys, build_run_plan, main, parse_mode


class TestInstnctEntrypoint(unittest.TestCase):
    def test_parse_mode_variants(self) -> None:
        self.assertEqual(parse_mode("train"), (Action.TRAIN,))
        self.assertEqual(parse_mode("eval"), (Action.EVAL,))
        self.assertEqual(parse_mode("evolve"), (Action.EVOLVE,))
        self.assertEqual(parse_mode("train-eval"), (Action.TRAIN, Action.EVAL))
        self.assertEqual(parse_mode("train,eval,evolve"), (Action.TRAIN, Action.EVAL, Action.EVOLVE))
        self.assertEqual(parse_mode("evolution|training"), (Action.EVOLVE, Action.TRAIN))

    def test_build_run_plan_precedence_cli_over_env(self) -> None:
        env = {"VRX_MODE": "train-eval-evolve", "VRX_RUN_ID": "env_id"}
        args = type(
            "Args",
            (),
            {
                "run_id": "cli_id",
                "dry_run": False,
                "mode": "eval",
                "train": False,
                "eval": False,
                "evolve": False,
            },
        )()
        plan = build_run_plan(
            args,
            env,
            env_keys=EnvKeys(),
            strict_env=False,
            now_fn=lambda: datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(plan.run_id, "cli_id")
        self.assertEqual(plan.actions, (Action.EVAL,))

    def test_build_run_plan_env_fallback(self) -> None:
        env = {"VRX_MODE": "train-eval", "VRX_RUN_ID": "env_id"}
        args = type(
            "Args",
            (),
            {
                "run_id": None,
                "dry_run": False,
                "mode": None,
                "train": False,
                "eval": False,
                "evolve": False,
            },
        )()
        plan = build_run_plan(
            args,
            env,
            env_keys=EnvKeys(),
            strict_env=False,
            now_fn=lambda: datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(plan.run_id, "env_id")
        self.assertEqual(plan.actions, (Action.TRAIN, Action.EVAL))

    def test_main_dispatch_order_and_header(self) -> None:
        called = []
        out_lines = []

        def w(line: str) -> None:
            out_lines.append(line)

        def train(ctx):
            called.append("train")
            return 0

        def evaluate(ctx):
            called.append("eval")
            return 0

        def evolve(ctx):
            called.append("evolve")
            return 0

        deps = EntrypointDeps(
            train=train,
            evaluate=evaluate,
            evolve=evolve,
            header_lines=["=== VRX HEADER ==="],
            write_line=w,
            now_fn=lambda: datetime(2020, 1, 1, tzinfo=timezone.utc),
        )

        rc = main(["--mode", "train-eval-evolve", "--run-id", "X"], env={}, deps=deps)
        self.assertEqual(rc, 0)
        self.assertEqual(called, ["train", "eval", "evolve"])
        # Header emitted first.
        self.assertGreaterEqual(len(out_lines), 1)
        self.assertEqual(out_lines[0], "=== VRX HEADER ===")

    def test_dry_run_does_not_execute_actions(self) -> None:
        called = []

        def train(ctx):
            called.append("train")
            return 0

        deps = EntrypointDeps(
            train=train,
            header_lines=[],
            write_line=lambda _: None,
            now_fn=lambda: datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        rc = main(["--mode", "train", "--dry-run", "--run-id", "X"], env={}, deps=deps)
        self.assertEqual(rc, 0)
        self.assertEqual(called, [])

    def test_no_header_flag(self) -> None:
        out_lines = []

        def w(line: str) -> None:
            out_lines.append(line)

        deps = EntrypointDeps(
            train=lambda ctx: 0,
            header_lines=["HEADER"],
            write_line=w,
            now_fn=lambda: datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        rc = main(["--mode", "train", "--no-header", "--run-id", "X"], env={}, deps=deps)
        self.assertEqual(rc, 0)
        self.assertEqual(out_lines, [])  # no header emitted


if __name__ == "__main__":
    unittest.main()
