import itertools
import os
import unittest
from unittest import mock
import inspect

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tools import instnct_train_wallclock as wall  # noqa: E402


class DummyModel(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

        # Attributes accessed directly by train_wallclock (keep minimal).
        self.ptr_walk_prob = 0.0
        self.ptr_deadzone = 0.0
        self.ptr_update_every = 5
        self.ptr_inertia = 0.0
        self.ptr_inertia_ema = 0.0

        # Disable optional subsystems.
        self.think_enabled = False
        self.think_alpha_adapt = False
        self.vault_adapt = False
        self.vault_enabled = False

        # Provide fields referenced by logging/trace payload.
        self.ptr_flip_rate = 0.0
        self.ptr_pingpong_rate = 0.0
        self.ptr_max_dwell = 0
        self.ptr_mean_dwell = 0.0
        self.ptr_delta_abs_mean = 0.0
        self.ptr_delta_raw_mean = 0.0
        self.ptr_orbit = 0
        self.ptr_residual_mean = 0.0
        self.ptr_anchor_clicks = 0
        self.vault_inj_rate = 0.0
        self.vault_updates = 0

        # Ensure eval hooks have something to read.
        self.ptr_update_allowed = 0
        self.ptr_update_blocked = 0
        self.ptr_lock = 0
        self.time_pointer = 0
        self.ptr_warmup_steps = 0

        # Xray tracking.
        self._last_return_xray = None

    def forward(self, x, return_xray: bool = False):
        self._last_return_xray = bool(return_xray)
        logits = self.linear(x)
        move_pen = logits.new_tensor(0.0)
        if return_xray:
            xray = {
                "attn_mass": 0.0,
                "gate_sat": 0.0,
                "h_mag": 0.0,
                "damp_ratio": 0.0,
            }
            return logits, move_pen, xray
        return logits, move_pen


class TestInstnctTrainWallclock(unittest.TestCase):
    def setUp(self):
        self._env_patch = mock.patch.dict(
            os.environ,
            {
                "VAR_COMPUTE_DEVICE": "cpu",
                "VRX_IGNORE_WALL_CLOCK": "0",
                "VRX_IGNORE_MAX_STEPS": "0",
                "VRX_XRAY": "0",
                "VRX_FORCE_CADENCE_1": "0",
            },
            clear=False,
        )
        self._env_patch.start()

        # Keep tests fast and side-effect free.
        wall.DEVICE = "cpu"
        wall.DTYPE = torch.float32
        wall.USE_AMP = False

        wall.SAVE_EVERY_STEPS = 0
        wall.SAVE_HISTORY = False
        wall.SAVE_BAD = False
        wall.SAVE_LAST_GOOD = False
        wall.LIVE_TRACE_PATH = ""
        wall.LIVE_TRACE_EVERY = 0

        wall.EVAL_EVERY_STEPS = 0
        wall.EVAL_AT_CHECKPOINT = False

        wall.HEARTBEAT_STEPS = 10**9
        wall.HEARTBEAT_SECS = 0.0

        wall.SHARD_ENABLED = False
        wall.SHARD_ADAPT = False
        wall.TRACTION_ENABLED = False

        wall.PANIC_ENABLED = False
        wall.PTR_UPDATE_GOV = False
        wall.INERTIA_SIGNAL_ENABLED = False
        wall.MITOSIS_ENABLED = False

        wall.EXPERT_HEADS = 1
        wall.EXPERT_BUDGET = 0

        wall.THERMO_ENABLED = False
        wall.STAIRCASE_ENABLED = False
        wall.STAIRCASE_ADAPT = False
        wall.METABOLIC_HUNGER = False
        wall.METABOLIC_TELEMETRY = False
        wall.HIBERNATE_ENABLED = False

        # Silence logs for unit tests.
        self._log_patch = mock.patch.object(wall, "log", autospec=True, side_effect=lambda *a, **k: None)
        self._log_patch.start()

    def tearDown(self):
        self._log_patch.stop()
        self._env_patch.stop()

    def _make_loader(self, n: int = 8, in_dim: int = 4, num_classes: int = 3, batch: int = 4):
        x = torch.randn(n, in_dim)
        y = torch.randint(low=0, high=num_classes, size=(n,))
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=batch, shuffle=False)

    def test_train_wallclock_returns_expected_keys(self):
        model = DummyModel(in_dim=4, num_classes=3)
        loader = self._make_loader()

        # Deterministic wall-clock exit: advance time monotonically.
        counter = itertools.count(start=0.0, step=0.01)

        with mock.patch.object(wall.time, "time", side_effect=lambda: next(counter)):
            stats = wall.train_wallclock(
                model=model,
                loader=loader,
                dataset_name="ds",
                model_name="m",
                num_classes=3,
                wall_clock=0.05,
                eval_loader=None,
            )

        expected_keys = {
            "loss_slope",
            "steps",
            "losses",
            "pointer_hist",
            "satiety_exits",
            "ptr_flip_rate",
            "ptr_pingpong_rate",
            "ptr_max_dwell",
            "ptr_mean_dwell",
            "ptr_delta_abs_mean",
            "ptr_delta_raw_mean",
            "state_loop_entropy",
            "state_loop_flip_rate",
            "state_loop_abab_rate",
            "state_loop_max_dwell",
            "state_loop_mean_dwell",
        }
        self.assertTrue(expected_keys.issubset(set(stats.keys())))
        self.assertIsInstance(stats["steps"], int)
        self.assertIsInstance(stats["losses"], list)
        self.assertGreaterEqual(stats["steps"], 0)

    def test_ignore_wall_clock_flag_is_strict(self):
        src = inspect.getsource(wall.train_wallclock)
        self.assertIn('os.environ.get("VRX_IGNORE_WALL_CLOCK") == "1"', src)

    def test_xray_flag_calls_model_with_return_xray(self):
        model = DummyModel(in_dim=4, num_classes=3)
        loader = self._make_loader()

        counter = itertools.count(start=0.0, step=0.01)

        with mock.patch.dict(os.environ, {"VRX_XRAY": "1"}), mock.patch.object(wall.time, "time", side_effect=lambda: next(counter)):
            wall.train_wallclock(
                model=model,
                loader=loader,
                dataset_name="ds",
                model_name="m",
                num_classes=3,
                wall_clock=0.05,
                eval_loader=None,
            )

        self.assertTrue(model._last_return_xray)

    def test_force_cadence_sets_ptr_update_every_to_one(self):
        model = DummyModel(in_dim=4, num_classes=3)
        model.ptr_update_every = 7
        loader = self._make_loader()

        counter = itertools.count(start=0.0, step=0.01)

        with mock.patch.dict(os.environ, {"VRX_FORCE_CADENCE_1": "1"}), mock.patch.object(wall.time, "time", side_effect=lambda: next(counter)):
            wall.train_wallclock(
                model=model,
                loader=loader,
                dataset_name="ds",
                model_name="m",
                num_classes=3,
                wall_clock=0.05,
                eval_loader=None,
            )

        self.assertEqual(model.ptr_update_every, 1)


if __name__ == "__main__":
    unittest.main()
