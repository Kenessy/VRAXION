import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


class TestRunDbCmdPassThrough(unittest.TestCase):
    def test_run_db_preserves_double_dash_in_child_args(self):
        """Regression test: only the leading '--' separator should be stripped.

        The child command may legitimately include its own '--' separator.
        """

        run_db = ROOT / "tools" / "run_db.py"
        self.assertTrue(run_db.exists(), run_db)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            db_root = td / "db"

            emit = td / "emit_args.py"
            emit.write_text(
                """
import sys
print('ARGV=' + repr(sys.argv))
""".lstrip(),
                encoding="utf-8",
            )

            # Important: we pass TWO '--' tokens:
            # 1) the run_db separator, 2) a literal argument for the child.
            proc = subprocess.run(
                [
                    PY,
                    "-B",
                    str(run_db),
                    "--db-root",
                    str(db_root),
                    "--run-name",
                    "passthrough",
                    "--",
                    PY,
                    "-B",
                    str(emit),
                    "--",
                    "child_arg",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )

            # run_db stdout must be exactly one line: the created run_dir.
            out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            self.assertEqual(len(out_lines), 1)
            run_dir = Path(out_lines[0])
            self.assertTrue(run_dir.exists(), run_dir)

            # Child stdout is captured into stdout.log
            stdout_log = run_dir / "stdout.log"
            self.assertTrue(stdout_log.exists(), stdout_log)
            txt = stdout_log.read_text(encoding="utf-8", errors="replace")
            self.assertIn("ARGV=", txt)

            # Parse the repr(...) safely.
            argv_repr = txt.split("ARGV=", 1)[1].strip().splitlines()[0]
            child_argv = ast.literal_eval(argv_repr)
            # sys.argv[0] is the script path; the remaining should include '--'.
            self.assertIn("--", child_argv[1:])
            self.assertIn("child_arg", child_argv[1:])

            # Metrics file should remain valid JSONL (even if empty)
            metrics = run_dir / "metrics.jsonl"
            self.assertTrue(metrics.exists(), metrics)
            for ln in metrics.read_text(encoding="utf-8").splitlines():
                json.loads(ln)


if __name__ == "__main__":
    unittest.main()
