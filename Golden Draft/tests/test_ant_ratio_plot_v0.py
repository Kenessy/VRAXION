import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestAntRatioPlotV0(unittest.TestCase):
    def test_cli_writes_html(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        plot_tool = repo_root / "Golden Draft" / "tools" / "ant_ratio_plot_v0.py"
        self.assertTrue(plot_tool.exists(), f"missing tool: {plot_tool}")

        packets = [
            {
                "ant_tier": "small",
                "expert_heads": 1,
                "batch_size": 8,
                "stability_pass": True,
                "fail_reasons": [],
                "vram_ratio_reserved": 0.50,
                "throughput_tokens_per_s": 123.4,
                "assoc_byte_disjoint_accuracy": 0.10,
                "probe_run_root": "bench_vault/_tmp/example/probe",
                "assoc_run_root": "bench_vault/_tmp/example/assoc",
            },
            {
                "ant_tier": "stress",
                "expert_heads": 16,
                "batch_size": 2,
                "stability_pass": False,
                "fail_reasons": ["vram_guard"],
                "vram_ratio_reserved": 0.92,
                "throughput_tokens_per_s": 10.0,
                "assoc_byte_disjoint_accuracy": 0.01,
                "probe_run_root": "bench_vault/_tmp/example2/probe",
                "assoc_run_root": "bench_vault/_tmp/example2/assoc",
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            packets_path = td_path / "packets.jsonl"
            out_path = td_path / "frontier.html"
            packets_path.write_text(
                "\n".join(json.dumps(p, sort_keys=True, ensure_ascii=True) for p in packets) + "\n",
                encoding="utf-8",
            )

            subprocess.check_call(
                [sys.executable, str(plot_tool), "--packets", str(packets_path), "--out", str(out_path)],
                cwd=str(repo_root),
            )

            html = out_path.read_text(encoding="utf-8", errors="replace")
            self.assertIn("Plotly.newPlot", html)
            self.assertIn("ant_ratio_packets.jsonl", html)
            self.assertIn("stress", html)
