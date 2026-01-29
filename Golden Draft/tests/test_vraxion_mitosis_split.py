import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import vraxion_mitosis_split


class TestVraxionMitosisSplit(unittest.TestCase):
    def test_mitosis_clones_expert_updates_router_and_optimizer(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            inpt = tmpdir / "in.pt"
            outpt = tmpdir / "out.pt"

            w0 = torch.randn(2, 3)
            b0 = torch.randn(2)

            state = {
                "router_map": torch.zeros(6, dtype=torch.long),
                "head.experts.0.weight": w0,
                "head.experts.0.bias": b0,
                "other.weight": torch.randn(1),
            }

            # Lightweight optimizer layout (as referenced by the tool docstring).
            pnames = [
                "head.experts.0.weight",
                "head.experts.0.bias",
                "other.weight",
            ]
            optim = {
                "param_groups": [{"params": [0, 1, 2]}],
                "state": {
                    0: {"m": torch.tensor(1.0)},
                    1: {"m": torch.tensor(2.0)},
                    2: {"m": torch.tensor(3.0)},
                },
            }

            ckpt = {
                "model": state,
                "num_experts": 1,
                "optim": optim,
                "param_names": list(pnames),
            }
            torch.save(ckpt, inpt)

            argv = [
                "--checkpoint",
                str(inpt),
                "--output",
                str(outpt),
                "--parent",
                "0",
                "--addresses",
                "1,3",
                "--noise",
                "0.0",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(vraxion_mitosis_split.main(argv), 0)

            outck = torch.load(outpt, map_location="cpu")
            self.assertEqual(outck["num_experts"], 2)
            self.assertIn("mitosis", outck)
            self.assertEqual(outck["mitosis"]["parent"], 0)
            self.assertEqual(outck["mitosis"]["child"], 1)
            self.assertEqual(outck["mitosis"]["addresses"], [1, 3])
            self.assertEqual(outck["mitosis"]["noise"], 0.0)

            outst = outck["model"]
            self.assertIn("head.experts.1.weight", outst)
            self.assertIn("head.experts.1.bias", outst)
            self.assertTrue(torch.allclose(outst["head.experts.1.weight"], w0))
            self.assertTrue(torch.allclose(outst["head.experts.1.bias"], b0))

            # router_map entries updated at the requested addresses.
            self.assertEqual(int(outst["router_map"][1].item()), 1)
            self.assertEqual(int(outst["router_map"][3].item()), 1)

            outpn = outck["param_names"]
            self.assertEqual(
                outpn,
                [
                    "head.experts.0.weight",
                    "head.experts.0.bias",
                    "other.weight",
                    "head.experts.1.weight",
                    "head.experts.1.bias",
                ],
            )

            outop = outck["optim"]
            self.assertEqual(outop["param_groups"][0]["params"], [0, 1, 2, 3, 4])
            self.assertIn(3, outop["state"])
            self.assertIn(4, outop["state"])
            self.assertTrue(torch.allclose(outop["state"][3]["m"], torch.tensor(1.0)))
            self.assertTrue(torch.allclose(outop["state"][4]["m"], torch.tensor(2.0)))


if __name__ == "__main__":
    unittest.main()
