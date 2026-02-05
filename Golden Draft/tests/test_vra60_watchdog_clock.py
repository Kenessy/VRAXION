import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import tools.vraxion_lab_supervisor as vls


class TestVra60WatchdogClock(unittest.TestCase):
    def test_compute_watchdog_stage(self) -> None:
        breach_s = 10.0
        surge_s = 3.0
        spill_s = 7.0
        self.assertEqual(vls._compute_watchdog_stage(0.0, surge_s, spill_s, breach_s), vls.WatchdogStage.OK)
        self.assertEqual(vls._compute_watchdog_stage(3.0, surge_s, spill_s, breach_s), vls.WatchdogStage.SURGE)
        self.assertEqual(vls._compute_watchdog_stage(6.9, surge_s, spill_s, breach_s), vls.WatchdogStage.SURGE)
        self.assertEqual(vls._compute_watchdog_stage(7.0, surge_s, spill_s, breach_s), vls.WatchdogStage.SPILL)
        self.assertEqual(vls._compute_watchdog_stage(9.9, surge_s, spill_s, breach_s), vls.WatchdogStage.SPILL)
        self.assertEqual(vls._compute_watchdog_stage(10.0, surge_s, spill_s, breach_s), vls.WatchdogStage.BREACH)
        self.assertEqual(vls._compute_watchdog_stage(999.0, surge_s, spill_s, breach_s), vls.WatchdogStage.BREACH)

    def test_forced_stall_reaches_washout_and_writes_artifacts(self) -> None:
        # Spawn the supervisor as a subprocess so a watchdog kill cannot take down
        # the test runner process tree.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            job_root = tmp / "job"
            wake_trigger = tmp / "wake_trigger.json"
            status_path = tmp / "nightmode_status.md"

            sup = Path(vls.__file__).resolve()
            child_cmd = [
                sys.executable,
                "-u",
                "-c",
                "import sys,time; print('hello'); sys.stdout.flush(); time.sleep(30)",
            ]
            cmd = [
                sys.executable,
                "-u",
                str(sup),
                "--job-name",
                "vra60_forced_stall",
                "--job-root",
                str(job_root),
                "--wake-trigger",
                str(wake_trigger),
                "--watchdog-no-output-s",
                "2",
                "--watchdog-abort-after-kills",
                "1",
                "--status-path",
                str(status_path),
                "--max-restarts",
                "0",
                "--",
                *child_cmd,
            ]

            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            self.assertEqual(res.returncode, 3, msg=f"unexpected rc={res.returncode} stdout={res.stdout} stderr={res.stderr}")

            # Per-job artifacts
            self.assertTrue((job_root / "watchdog_state.json").exists())
            self.assertTrue((job_root / "watchdog_events.jsonl").exists())
            self.assertTrue((job_root / "failure_summary.md").exists())
            self.assertTrue((job_root / "child_log_tail.txt").exists())
            self.assertTrue((job_root / "supervisor.log").exists())

            # Global-ish status file (overridden to temp path for test isolation).
            self.assertTrue(status_path.exists())

            # Wake trigger should be produced on washout (best-effort).
            self.assertTrue(wake_trigger.exists())

            state = json.loads((job_root / "watchdog_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["stage"], "WASHOUT")

            events = (job_root / "watchdog_events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertTrue(any('"stage": "WASHOUT"' in ln for ln in events))

