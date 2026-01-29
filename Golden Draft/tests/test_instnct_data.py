"""Behavior locks for :mod:`tools.instnct_data`.

These tests avoid optional deps (torchvision/torchaudio) and only exercise the
synthetic paths plus the deterministic pair loaders.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

import torch

from vraxion.instnct import infra


class InstnctDataTests(unittest.TestCase):
    def _set_clean_env(self) -> None:
        for key in list(os.environ.keys()):
            if key.startswith("VRX_") or key.startswith("VAR_") or key.startswith("PILOT_"):
                os.environ.pop(key, None)

    def test_synth_mode_does_not_import_torchvision(self) -> None:
        self._set_clean_env()
        with conftest.temporary_env(
            VAR_RUN_SEED="123",
            VRX_BATCH_SIZE="4",
            VRX_MAX_SAMPLES="32",
            VRX_SYNTH="1",
            VRX_SYNTH_MODE="assoc_byte",
            VRX_SYNTH_LEN="16",
            VRX_ASSOC_KEYS="8",
            VRX_ASSOC_PAIRS="2",
            VRX_ASSOC_VAL_RANGE="8",
        ):
            sentinel = object()
            old_tv = sys.modules.get("torchvision", sentinel)
            sys.modules["torchvision"] = None  # type: ignore
            try:
                import tools.instnct_data as D

                with tempfile.TemporaryDirectory() as td:
                    old_log = infra.LOG_PATH
                    infra.LOG_PATH = os.path.join(td, "vraxion.log")
                    try:
                        loader, num_classes, collate = D.get_seq_mnist_loader()
                    finally:
                        infra.LOG_PATH = old_log

                self.assertEqual(num_classes, 8)
                self.assertTrue(callable(collate))
                xb, yb = next(iter(loader))
                self.assertEqual(tuple(xb.shape), (4, 16, 1))
                self.assertEqual(xb.dtype, torch.float32)
                self.assertEqual(tuple(yb.shape), (4,))
                self.assertEqual(yb.dtype, torch.int64)
                self.assertTrue(torch.all(yb >= 0).item())
                self.assertTrue(torch.all(yb < 8).item())
            finally:
                if old_tv is sentinel:
                    sys.modules.pop("torchvision", None)
                else:
                    sys.modules["torchvision"] = old_tv  # type: ignore

    def test_real_mode_requires_torchvision(self) -> None:
        self._set_clean_env()
        with conftest.temporary_env(VAR_RUN_SEED="123", VRX_BATCH_SIZE="2", VRX_SYNTH="0"):
            sentinel = object()
            old_tv = sys.modules.get("torchvision", sentinel)
            sys.modules["torchvision"] = None  # type: ignore
            try:
                import tools.instnct_data as D

                with self.assertRaises(RuntimeError) as ctx:
                    D.get_seq_mnist_loader()
                self.assertIn("torchvision is required", str(ctx.exception))
            finally:
                if old_tv is sentinel:
                    sys.modules.pop("torchvision", None)
                else:
                    sys.modules["torchvision"] = old_tv  # type: ignore

    def test_build_synth_pair_loaders_label_flip(self) -> None:
        self._set_clean_env()
        with conftest.temporary_env(
            VAR_RUN_SEED="7",
            VRX_BATCH_SIZE="4",
            VRX_MAX_SAMPLES="32",
            VRX_SYNTH_LEN="16",
            VRX_SYNTH_SHUFFLE="0",
        ):
            import tools.instnct_data as D

            la, lb, _ = D.build_synth_pair_loaders()
            xa, ya = next(iter(la))
            xb, yb = next(iter(lb))
            self.assertTrue(torch.equal(xa, xb))
            self.assertTrue(torch.equal(yb, 1 - ya))
            self.assertEqual(tuple(xa.shape), (4, 16, 1))
            self.assertEqual(tuple(ya.shape), (4,))

    def test_build_synth_pair_loaders_deterministic(self) -> None:
        self._set_clean_env()
        with conftest.temporary_env(
            VAR_RUN_SEED="7",
            VRX_BATCH_SIZE="4",
            VRX_MAX_SAMPLES="32",
            VRX_SYNTH_LEN="16",
            VRX_SYNTH_SHUFFLE="0",
        ):
            import tools.instnct_data as D

            la1, lb1, _ = D.build_synth_pair_loaders()
            xa1, ya1 = next(iter(la1))
            xb1, yb1 = next(iter(lb1))

            la2, lb2, _ = D.build_synth_pair_loaders()
            xa2, ya2 = next(iter(la2))
            xb2, yb2 = next(iter(lb2))

            self.assertTrue(torch.equal(xa1, xa2))
            self.assertTrue(torch.equal(ya1, ya2))
            self.assertTrue(torch.equal(xb1, xb2))
            self.assertTrue(torch.equal(yb1, yb2))


if __name__ == "__main__":
    unittest.main()
