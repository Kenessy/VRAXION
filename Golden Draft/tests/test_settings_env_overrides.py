"""Behavior locks for :func:`vraxion.settings.load_settings` env overrides.

These tests are stdlib-only and avoid any dataset downloads. They focus on a
small but high-leverage contract: specific VRX env vars must override defaults.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from vraxion.settings import load_settings


class SettingsEnvOverridesTests(unittest.TestCase):
    def test_load_settings_respects_ring_len_and_slot_dim_overrides(self) -> None:
        # Use a fully clean environment to avoid developer shell overrides
        # breaking determinism (load_settings parses many VRX_* vars).
        with patch.dict(os.environ, {"VRX_RING_LEN": "123", "VRX_SLOT_DIM": "99"}, clear=True):
            cfg = load_settings()

        self.assertEqual(int(cfg.ring_len), 123)
        self.assertEqual(int(cfg.slot_dim), 99)


if __name__ == "__main__":
    unittest.main()


