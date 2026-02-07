import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)


class TestAntRatioSweepNonzeroRcCheckpoint(unittest.TestCase):
    def test_nonzero_rc_uses_checkpoint(self) -> None:
        from tools.ant_ratio_sweep_v0 import _run_capability_train

        with tempfile.TemporaryDirectory() as td:
            run_root = Path(td) / "assoc_run"
            run_root.mkdir(parents=True, exist_ok=True)
            ckpt = run_root / "checkpoint_last_good.pt"
            ckpt.write_bytes(b"fake")

            with mock.patch("tools.ant_ratio_sweep_v0.subprocess.run", return_value=SimpleNamespace(returncode=-1)):
                got_ckpt, got_rc = _run_capability_train(
                    repo_root=Path(td),
                    run_root=run_root,
                    seed=123,
                    device="cpu",
                    precision="fp32",
                    ring_len=2048,
                    slot_dim=256,
                    expert_heads=1,
                    batch=1,
                    steps=5,
                    synth_len=32,
                    assoc_keys=4,
                    assoc_pairs=2,
                    assoc_val_range=8,
                    max_samples=16,
                    eval_samples=0,
                    ptr_dtype="fp64",
                    offline_only=True,
                    save_every=1,
                    timeout_s=1,
                )

            self.assertEqual(got_ckpt, ckpt)
            self.assertEqual(got_rc, -1)


if __name__ == "__main__":
    unittest.main()
