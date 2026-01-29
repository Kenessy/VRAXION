import json
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tools import instnct_train_steps


class TinyTP6Model(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, with_theta_ptr: bool = False):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

        # Telemetry attributes used by the training loop (keep minimal but present).
        self.register_buffer("pointer_hist", torch.zeros(num_classes, dtype=torch.float32))
        self.satiety_exits = 0

        self.ptr_flip_rate = 0.0
        self.ptr_mean_dwell = 2.0
        self.ptr_delta_abs_mean = 0.1
        self.ptr_max_dwell = 3
        self.ptr_delta_raw_mean = 0.05

        self.ptr_inertia = 0.3
        self.ptr_deadzone = 0.1
        self.ptr_walk_prob = 0.2
        self.ptr_update_every = 1

        if with_theta_ptr:
            # The loop checks for this attribute to enable heartbeat + trace.
            self.theta_ptr_reduced = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        logits = self.fc(x)

        # Update pointer_hist for trace entropy computation.
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            hist = torch.bincount(preds, minlength=self.pointer_hist.numel()).to(self.pointer_hist.dtype)
            self.pointer_hist.add_(hist)

            # Provide stable-ish synthetic metrics.
            if hist.numel() > 0:
                self.ptr_flip_rate = float((hist > 0).float().mean().item())
            else:
                self.ptr_flip_rate = 0.0

        move_pen = logits.new_tensor(0.0)
        return logits, move_pen


class TestInstnctTrainSteps(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_train_steps_basic_cpu(self):
        instnct_train_steps.log = lambda _msg: None  # keep unit test output clean

        x = torch.randn(32, 8)
        y = torch.randint(0, 3, (32,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

        model = TinyTP6Model(in_dim=8, num_classes=3)

        steps = 5
        with patch.dict(os.environ, {"VAR_COMPUTE_DEVICE": "cpu", "VAR_TRAINING_TRACE_ENABLED": "0"}, clear=False):
            out = instnct_train_steps.train_steps(model, loader, steps, "unit", "tiny")

        # Required keys (stable contract)
        for k in (
            "loss_slope",
            "steps",
            "ptr_flip_rate",
            "pointer_hist",
            "satiety_exits",
            "ptr_mean_dwell",
            "ptr_max_dwell",
            "ptr_delta_abs_mean",
            "ptr_delta_raw_mean",
        ):
            self.assertIn(k, out)

        self.assertEqual(out["steps"], steps)

        # Metric types are permissive, but should be sane.
        self.assertTrue(out["ptr_flip_rate"] is None or isinstance(out["ptr_flip_rate"], float))
        self.assertTrue(out["loss_slope"] is None or isinstance(out["loss_slope"], float))

        if out["pointer_hist"] is not None:
            self.assertIsInstance(out["pointer_hist"], list)
            self.assertEqual(len(out["pointer_hist"]), 3)

    def test_training_trace_writes_jsonl_on_cpu(self):
        instnct_train_steps.log = lambda _msg: None

        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)

        model = TinyTP6Model(in_dim=4, num_classes=2, with_theta_ptr=True)

        with tempfile.TemporaryDirectory() as td:
            trace_path = str(pathlib.Path(td) / "trace.jsonl")

            with patch.dict(
                os.environ,
                {
                    "VAR_COMPUTE_DEVICE": "cpu",
                    "VAR_TRAINING_TRACE_ENABLED": "1",
                    "VAR_TRAINING_TRACE_PATH": trace_path,
                },
                clear=False,
            ):
                out = instnct_train_steps.train_steps(model, loader, 1, "unit", "tiny")

            self.assertEqual(out["steps"], 1)
            self.assertTrue(os.path.exists(trace_path))

            with open(trace_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            self.assertGreaterEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload.get("dataset"), "unit")
            self.assertEqual(payload.get("model"), "tiny")
            self.assertIn("loss", payload)
            self.assertIn("ctrl", payload)


if __name__ == "__main__":
    unittest.main()
