import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestAntRatioPublishDedupe(unittest.TestCase):
    def test_republishing_same_sweep_is_byte_stable(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        tool = repo_root / "Golden Draft" / "tools" / "ant_ratio_publish_v0.py"
        self.assertTrue(tool.exists(), f"missing tool: {tool}")

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            sweep = td_path / "20260207_082612-full_rankable"
            sweep.mkdir(parents=True, exist_ok=True)

            (sweep / "sweep_meta.json").write_text(
                json.dumps({"generated_utc": "2026-02-07T12:36:07Z"}, sort_keys=True),
                encoding="utf-8",
            )
            summary = (
                "status,error,ant_tier,expert_heads,batch,steps,cap_train_rc,cap_train_nonzero_rc,cap_eval_device,"
                "eval_heartbeat_seen,attempt_eval_count,probe_pass,vram_ratio_reserved,throughput_tokens_per_s,"
                "tokens_per_vram_ratio,assoc_byte_disjoint_accuracy,assoc_acc_per_token_per_s,assoc_acc_per_vram_ratio,"
                "probe_run_root,assoc_run_root\n"
                "ok,,small,1,27,36,0,False,cpu,True,1,True,0.8633,1287.68,1491.47,0.0078125,0.000006,0.0090,"
                "bench_vault/_tmp/a/probe,bench_vault/_tmp/a/seed123\n"
            )
            (sweep / "ant_ratio_summary.csv").write_text(summary, encoding="utf-8")
            packet = {
                "ant_tier": "small",
                "expert_heads": 1,
                "batch_size": 27,
                "assoc_eval_n": 512,
                "token_budget_steps": 36,
                "stability_pass": True,
                "probe_run_root": "bench_vault/_tmp/a/probe",
                "assoc_run_root": "bench_vault/_tmp/a/seed123",
            }
            (sweep / "ant_ratio_packets.jsonl").write_text(json.dumps(packet, sort_keys=True) + "\n", encoding="utf-8")

            out_dir = td_path / "out"
            cmd = [
                sys.executable,
                str(tool),
                "--sweep-root",
                str(sweep),
                "--out-dir",
                str(out_dir),
            ]
            subprocess.check_call(cmd, cwd=str(repo_root))
            first_csv = (out_dir / "db_v0.csv").read_bytes()
            first_jsonl = (out_dir / "db_v0.jsonl").read_bytes()

            subprocess.check_call(cmd, cwd=str(repo_root))
            second_csv = (out_dir / "db_v0.csv").read_bytes()
            second_jsonl = (out_dir / "db_v0.jsonl").read_bytes()

            self.assertEqual(first_csv, second_csv)
            self.assertEqual(first_jsonl, second_jsonl)


if __name__ == "__main__":
    unittest.main()
