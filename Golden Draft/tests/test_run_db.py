import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
RUN_DB = ROOT / "tools" / "run_db.py"


class TestRunDb(unittest.TestCase):
    def test_run_db_exit_code_passthrough(self):
        self.assertTrue(RUN_DB.exists(), RUN_DB)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            db_root = td / "db"

            emit = td / "emit.py"
            emit.write_text(
                "import sys\n"
                "print('hello')\n"
                "sys.exit(7)\n",
                encoding="utf-8",
            )

            r = subprocess.run(
                [
                    PY,
                    "-B",
                    str(RUN_DB),
                    "--db-root",
                    str(db_root),
                    "--run-name",
                    "exitcode",
                    "--",
                    PY,
                    "-B",
                    str(emit),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 7, msg=r.stderr)

            run_dir = Path(r.stdout.strip().splitlines()[-1])
            self.assertTrue(run_dir.exists(), run_dir)

            # Summary should reflect the child exit code.
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary.get("exit_code"), 7)

    def test_run_db_metrics_and_sqlite(self):
        self.assertTrue(RUN_DB.exists(), RUN_DB)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            db_root = td / "db"

            emit = td / "emit.py"
            emit.write_text(
                "\n".join(
                    [
                        "print('synth | demo | step 0001/0100 | loss 0.1234 | V_COG[PRGRS:100.0% ORB:2 RD:1.0e+00 AC:1]')",
                        "print('grad_norm(theta_ptr)=1.0e-03')",
                        "print('synth | demo | step 0002/0100 | loss 0.2345 | V_COG[PRGRS:100.0% ORB:2 RD:2.0e+00 AC:2]')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            r = subprocess.run(
                [
                    PY,
                    "-B",
                    str(RUN_DB),
                    "--db-root",
                    str(db_root),
                    "--run-name",
                    "smoke",
                    "--",
                    PY,
                    "-B",
                    str(emit),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr)

            run_dir = Path(r.stdout.strip().splitlines()[-1])
            self.assertTrue(run_dir.exists(), run_dir)

            # Validate metrics.jsonl lines are JSON and contain expected fields.
            metrics_path = run_dir / "metrics.jsonl"
            lines = metrics_path.read_text(encoding="utf-8").splitlines()
            events = [json.loads(ln) for ln in lines if ln.strip()]

            steps = sorted({ev.get("step") for ev in events if "step" in ev})
            self.assertEqual(steps, [1, 2])
            self.assertTrue(all(ev.get("stream") == "stdout" for ev in events))
            self.assertTrue(all("ts_utc" in ev for ev in events))

            # Summary should capture the last-seen V_COG.
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            vcog_last = summary.get("vcog_last")
            self.assertIsInstance(vcog_last, dict)
            self.assertEqual(vcog_last.get("RD"), 2.0)
            self.assertEqual(vcog_last.get("AC"), 2.0)

            # SQLite index should have a row for this run.
            db_path = db_root / "runs.sqlite"
            self.assertTrue(db_path.exists(), db_path)
            con = sqlite3.connect(str(db_path))
            try:
                row = con.execute(
                    "SELECT run_id, exit_code, n_loss, loss_min, loss_max FROM runs"
                ).fetchone()
            finally:
                con.close()

            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row[1], 0)
            self.assertEqual(row[2], 2)
            self.assertAlmostEqual(row[3], 0.1234, places=7)
            self.assertAlmostEqual(row[4], 0.2345, places=7)


if __name__ == "__main__":
    unittest.main()
