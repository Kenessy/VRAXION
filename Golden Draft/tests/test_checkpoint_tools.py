"""Tests for Golden Draft checkpoint CLI tools.

These tools are internal scaffolding and intentionally live outside the
end-user runtime package.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        # Keep Golden Draft tree clean (no __pycache__/pyc artifacts).
        [sys.executable, "-B", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestModularizeCheckpoint(unittest.TestCase):
    def test_missing_checkpoint_returns_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpstr:
            tmpdir = Path(tmpstr)
            ckppth = tmpdir / "no_such_checkpoint.pt"
            outdir = tmpdir / "out"

            resobj = _run([
                "-m",
                "tools.modularize_checkpoint",
                "--checkpoint",
                str(ckppth),
                "--output",
                str(outdir),
            ])
            self.assertEqual(resobj.returncode, 2, msg=resobj.stderr)

    def test_non_empty_output_without_force_returns_3(self) -> None:
        with tempfile.TemporaryDirectory() as tmpstr:
            tmpdir = Path(tmpstr)
            ckppth = tmpdir / "ckpt.pt"
            outdir = tmpdir / "out"
            outdir.mkdir()
            (outdir / "existing.txt").write_text("x", encoding="utf-8")

            torch.save({"model": {"router_map": torch.tensor([0])}}, ckppth)

            resobj = _run([
                "-m",
                "tools.modularize_checkpoint",
                "--checkpoint",
                str(ckppth),
                "--output",
                str(outdir),
            ])
            self.assertEqual(resobj.returncode, 3, msg=resobj.stderr)

    def test_writes_router_and_experts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpstr:
            tmpdir = Path(tmpstr)
            ckppth = tmpdir / "ckpt.pt"
            outdir = tmpdir / "out"

            model = {
                "router_map": torch.tensor([0, 1, 2, 1, 0]),
                "head.experts.0.weight": torch.randn(2, 2),
                "head.experts.1.weight": torch.randn(2, 2),
                "head.experts.2.weight": torch.randn(2, 2),
                "head.experts.2.bias": torch.randn(2),
                "backbone.weight": torch.randn(2, 2),
            }
            ckpt = {
                "model": model,
                "num_experts": 3,
                "step": 7,
            }
            torch.save(ckpt, ckppth)

            resobj = _run([
                "-m",
                "tools.modularize_checkpoint",
                "--checkpoint",
                str(ckppth),
                "--output",
                str(outdir),
                "--tenure-all",
            ])
            self.assertEqual(resobj.returncode, 0, msg=resobj.stderr)

            router_path = outdir / "system" / "router.state"
            self.assertTrue(router_path.exists())
            router = torch.load(router_path, map_location="cpu")
            self.assertEqual(int(router["num_experts"]), 3)

            # Router payload must exclude expert tensors.
            for keystr in router["model"].keys():
                self.assertFalse(str(keystr).startswith("head.experts."))

            expdir = outdir / "experts"
            self.assertTrue((expdir / "expert_000.pt").exists())
            self.assertTrue((expdir / "expert_001.pt").exists())
            self.assertTrue((expdir / "expert_002.pt").exists())

            meta_path = expdir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["num_experts"], 3)
            self.assertEqual(len(meta["experts"]), 3)
            for ent in meta["experts"]:
                self.assertTrue(ent["tenured"])
                self.assertEqual(ent["created_step"], 7)


class TestVraxionPruneMerge(unittest.TestCase):
    def test_prune_merges_highest_expert_and_trims_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpstr:
            tmpdir = Path(tmpstr)
            inpth = tmpdir / "in.pt"
            outpt = tmpdir / "out.pt"

            state = {
                "router_map": torch.tensor([0, 1, 2, 2, 1, 0], dtype=torch.long),
                "head.experts.0.weight": torch.randn(2, 2),
                "head.experts.0.bias": torch.randn(2),
                "head.experts.1.weight": torch.randn(2, 2),
                "head.experts.1.bias": torch.randn(2),
                "head.experts.2.weight": torch.randn(2, 2),
                "head.experts.2.bias": torch.randn(2),
                "other.weight": torch.randn(1),
            }
            param_names = [
                "head.experts.0.weight",
                "head.experts.0.bias",
                "head.experts.1.weight",
                "head.experts.1.bias",
                "head.experts.2.weight",
                "head.experts.2.bias",
                "other.weight",
            ]
            optim = {
                "state": {
                    0: {"m": torch.tensor(0.0)},
                    1: {"m": torch.tensor(0.0)},
                    2: {"m": torch.tensor(0.0)},
                    3: {"m": torch.tensor(0.0)},
                    4: {"m": torch.tensor(0.0)},
                    5: {"m": torch.tensor(0.0)},
                    6: {"m": torch.tensor(0.0)},
                },
                "param_groups": [{"params": [0, 1, 2, 3, 4, 5, 6]}],
            }
            ckpt = {
                "model": state,
                "num_experts": 3,
                "optim": optim,
                "param_names": param_names,
            }
            torch.save(ckpt, inpth)

            resobj = _run([
                "-m",
                "tools.vraxion_prune_merge",
                "--checkpoint",
                str(inpth),
                "--output",
                str(outpt),
                "--merge-from",
                "2",
                "--merge-into",
                "1",
            ])
            self.assertEqual(resobj.returncode, 0, msg=resobj.stderr)

            outck = torch.load(outpt, map_location="cpu")
            self.assertEqual(outck["num_experts"], 2)

            outst = outck["model"]
            self.assertTrue(torch.all(outst["router_map"] != 2))
            self.assertTrue(torch.all(outst["router_map"] <= 1))

            # Expert 2 tensors removed.
            for keystr in outst.keys():
                self.assertFalse(str(keystr).startswith("head.experts.2."))

            outop = outck.get("optim")
            self.assertIsInstance(outop, dict)
            grp0 = outop["param_groups"][0]
            self.assertNotIn(4, grp0["params"])
            self.assertNotIn(5, grp0["params"])
            self.assertNotIn(4, outop["state"])
            self.assertNotIn(5, outop["state"])

            outnm = outck.get("param_names")
            self.assertIsInstance(outnm, list)
            self.assertNotIn("head.experts.2.weight", outnm)
            self.assertNotIn("head.experts.2.bias", outnm)

    def test_prune_trims_param_names_even_without_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpstr:
            tmpdir = Path(tmpstr)
            inpth = tmpdir / "in.pt"
            outpt = tmpdir / "out.pt"

            state = {
                "router_map": torch.tensor([0, 1, 2, 2]),
                "head.experts.0.weight": torch.randn(2, 2),
                "head.experts.1.weight": torch.randn(2, 2),
                "head.experts.2.weight": torch.randn(2, 2),
            }
            ckpt = {
                "model": state,
                "num_experts": 3,
                "param_names": list(state.keys()),
            }
            torch.save(ckpt, inpth)

            resobj = _run([
                "-m",
                "tools.vraxion_prune_merge",
                "--checkpoint",
                str(inpth),
                "--output",
                str(outpt),
                "--merge-from",
                "2",
                "--merge-into",
                "1",
            ])
            self.assertEqual(resobj.returncode, 0, msg=resobj.stderr)

            outck = torch.load(outpt, map_location="cpu")
            outnm = outck.get("param_names")
            self.assertIsInstance(outnm, list)
            self.assertNotIn("head.experts.2.weight", outnm)


if __name__ == "__main__":
    unittest.main()
