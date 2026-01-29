"""Behavior locks for :class:`vraxion.instnct.experts.LocationExpertRouter`."""

from __future__ import annotations

import unittest

import torch

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from vraxion.instnct.experts import LocationExpertRouter


class LocationExpertRouterTests(unittest.TestCase):
    def test_location_expert_router_routes_by_pointer_modulo(self) -> None:
        router = LocationExpertRouter(d_model=2, vocab_size=1, num_experts=3)

        assert router.experts is not None
        for idx, expert in enumerate(router.experts):
            with torch.no_grad():
                expert.weight.zero_()
                expert.bias.fill_(float(idx))

        x = torch.zeros(6, 2, dtype=torch.float32)
        ptr = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        out = router(x, ptr)

        expected = torch.tensor([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(out.cpu(), expected))

    def test_location_expert_router_defaults_to_expert_zero_when_no_pointer(self) -> None:
        router = LocationExpertRouter(d_model=2, vocab_size=1, num_experts=3)

        assert router.experts is not None
        for idx, expert in enumerate(router.experts):
            with torch.no_grad():
                expert.weight.zero_()
                expert.bias.fill_(float(idx))

        x = torch.zeros(4, 2, dtype=torch.float32)
        out = router(x, pointer_addresses=None)

        expected = torch.zeros(4, 1, dtype=torch.float32)
        self.assertTrue(torch.allclose(out.cpu(), expected))

    def test_location_expert_router_single_expert_ignores_pointer(self) -> None:
        router = LocationExpertRouter(d_model=2, vocab_size=1, num_experts=1)

        assert router.single is not None
        with torch.no_grad():
            router.single.weight.zero_()
            router.single.bias.fill_(3.14)

        x = torch.zeros(3, 2, dtype=torch.float32)
        ptr = torch.tensor([0, 1, 2], dtype=torch.long)
        out = router(x, ptr)

        expected = torch.full((3, 1), 3.14, dtype=torch.float32)
        self.assertTrue(torch.allclose(out.cpu(), expected))


if __name__ == "__main__":
    unittest.main()

