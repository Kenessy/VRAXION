"""Tests for the VRA-31 workload ID contract (stdlib-only)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from tools import workload_id


class TestWorkloadId(unittest.TestCase):
    def test_same_id_different_key_order(self) -> None:
        spec_a = json.loads(
            """
            {
              "schema_version": "workload_schema_v1",
              "ant_spec": {
                "ring_len": 8192,
                "slot_dim": 576,
                "ptr_dtype": "fp64",
                "precision": "fp32"
              },
              "colony_spec": {
                "seq_len": 256,
                "synth_len": 256,
                "batch_size": 16,
                "ptr_update_every": 1,
                "state_loop_samples": 0
              }
            }
            """
        )
        spec_b = json.loads(
            """
            {
              "colony_spec": {
                "state_loop_samples": 0,
                "ptr_update_every": 1,
                "batch_size": 16,
                "synth_len": 256,
                "seq_len": 256
              },
              "ant_spec": {
                "precision": "fp32",
                "ptr_dtype": "fp64",
                "slot_dim": 576,
                "ring_len": 8192
              },
              "schema_version": "workload_schema_v1"
            }
            """
        )

        wid_a = workload_id.compute_workload_id(workload_id.canonicalize_spec(spec_a))
        wid_b = workload_id.compute_workload_id(workload_id.canonicalize_spec(spec_b))
        self.assertEqual(wid_a, wid_b)

    def test_required_field_change_changes_id(self) -> None:
        base = {
            "schema_version": "workload_schema_v1",
            "ant_spec": {"ring_len": 8192, "slot_dim": 576, "ptr_dtype": "fp64", "precision": "fp32"},
            "colony_spec": {
                "seq_len": 256,
                "synth_len": 256,
                "batch_size": 16,
                "ptr_update_every": 1,
                "state_loop_samples": 0,
            },
        }
        base_id = workload_id.compute_workload_id(workload_id.canonicalize_spec(dict(base)))

        changed = json.loads(json.dumps(base))
        changed["ant_spec"]["ring_len"] = 8193
        changed_id = workload_id.compute_workload_id(workload_id.canonicalize_spec(changed))
        self.assertNotEqual(base_id, changed_id)

    def test_optional_name_notes_excluded(self) -> None:
        base = {
            "schema_version": "workload_schema_v1",
            "ant_spec": {"ring_len": 8192, "slot_dim": 576, "ptr_dtype": "fp64", "precision": "fp32"},
            "colony_spec": {
                "seq_len": 256,
                "synth_len": 256,
                "batch_size": 16,
                "ptr_update_every": 1,
                "state_loop_samples": 0,
            },
        }
        with_notes = json.loads(json.dumps(base))
        with_notes["name"] = "example"
        with_notes["notes"] = "hello"
        with_notes["ant_spec"]["name"] = "ant"
        with_notes["colony_spec"]["notes"] = "col"

        wid_a = workload_id.compute_workload_id(workload_id.canonicalize_spec(base))
        wid_b = workload_id.compute_workload_id(workload_id.canonicalize_spec(with_notes))
        self.assertEqual(wid_a, wid_b)

    def test_unknown_key_rejected(self) -> None:
        bad = {
            "schema_version": "workload_schema_v1",
            "ant_spec": {"ring_len": 8192, "slot_dim": 576, "ptr_dtype": "fp64", "precision": "fp32"},
            "colony_spec": {
                "seq_len": 256,
                "synth_len": 256,
                "batch_size": 16,
                "ptr_update_every": 1,
                "state_loop_samples": 0,
            },
            "unexpected": 1,
        }
        with self.assertRaises(ValueError):
            workload_id.canonicalize_spec(bad)

    def test_cli_smoke_real_template(self) -> None:
        draft_root = Path(__file__).resolve().parents[1]
        script = draft_root / "tools" / "workload_id.py"
        tpl = draft_root / "workloads" / "od1_real_v1.json"

        proc = subprocess.run(
            [sys.executable, str(script), str(tpl)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        wid = proc.stdout.strip()
        self.assertTrue(wid.startswith("wl_v1_"))
        self.assertEqual(len(wid), len("wl_v1_") + 12)


if __name__ == "__main__":
    unittest.main()

