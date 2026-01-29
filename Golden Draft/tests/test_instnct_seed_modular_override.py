"""Behavior locks for seeding + modular expert-head override helpers."""

from __future__ import annotations

import os
import random
import tempfile
import unittest

import torch

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from vraxion.instnct import seed as seed_mod


class InstnctSeedAndModularOverrideTests(unittest.TestCase):
    def _write_router_state(self, base_dir: str, num_experts: int) -> str:
        system_dir = os.path.join(base_dir, "system")
        os.makedirs(system_dir, exist_ok=True)
        router_path = os.path.join(system_dir, "router.state")
        torch.save({"num_experts": int(num_experts)}, router_path)
        return router_path

    def test_set_seed_is_deterministic_for_python_and_torch(self) -> None:
        with conftest.temporary_env(PYTHONHASHSEED=None):
            seed_mod.set_seed(123)
            a = random.randint(0, 1_000_000)
            x = torch.rand(4, dtype=torch.float32)

            seed_mod.set_seed(123)
            b = random.randint(0, 1_000_000)
            y = torch.rand(4, dtype=torch.float32)

            self.assertEqual(a, b)
            self.assertTrue(torch.equal(x, y))

            # NumPy is optional; only assert when available.
            np = getattr(seed_mod, "np", None)
            if np is not None:
                seed_mod.set_seed(123)
                n1 = np.random.randint(0, 1_000_000)
                seed_mod.set_seed(123)
                n2 = np.random.randint(0, 1_000_000)
                self.assertEqual(int(n1), int(n2))

    def test_overrides_expert_heads_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_router_state(td, 7)
            seed_mod.EXPERT_HEADS = 3

            with conftest.temporary_env(
                VRX_MODULAR_AUTO_EXPERTS="1",
                VRX_EXPERT_HEADS=None,
            ):
                seed_mod._maybe_override_expert_heads(td)
                self.assertEqual(seed_mod.EXPERT_HEADS, 7)

    def test_does_not_override_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write_router_state(td, 7)

            for val in ("0", "true", "01", ""):
                seed_mod.EXPERT_HEADS = 3
                with conftest.temporary_env(
                    VRX_MODULAR_AUTO_EXPERTS=val,
                    VRX_EXPERT_HEADS=None,
                ):
                    seed_mod._maybe_override_expert_heads(td)
                    self.assertEqual(seed_mod.EXPERT_HEADS, 3)


if __name__ == "__main__":
    unittest.main()
