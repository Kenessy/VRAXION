import os
import sys

import torch


def _assert_close(name, actual, expected, tol=1e-6):
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def test_circular_lerp_boundary():
    from tournament_phase6 import AbsoluteHallway

    ring = 10.0
    prev = torch.tensor([9.8])
    walk = torch.tensor([0.8])
    out = AbsoluteHallway.circ_lerp(prev, walk, torch.tensor([0.2]), ring)
    expected = torch.remainder(prev + 0.2, ring)
    _assert_close("circular_lerp_wrap", float(out.item()), float(expected.item()))


def test_kernel_fractional_shift():
    from tournament_phase6 import AbsoluteHallway

    model = AbsoluteHallway(input_dim=1, num_classes=2, ring_len=64, slot_dim=4)
    ptr_a = torch.tensor([10.1], dtype=torch.float32)
    ptr_b = torch.tensor([10.9], dtype=torch.float32)
    offsets = torch.arange(-2, 3, dtype=torch.float32)
    _, wa, _ = model._compute_kernel_weights(ptr_a, offsets, model.ring_len)
    _, wb, _ = model._compute_kernel_weights(ptr_b, offsets, model.ring_len)
    diff = (wa - wb).abs().max().item()
    if diff < 1e-6:
        raise AssertionError(f"kernel_fractional_shift: weights too similar (max diff {diff})")


def test_settings_overrides():
    from prime_c19.settings import load_settings

    os.environ["TP6_RING_LEN"] = "123"
    cfg = load_settings()
    if cfg.ring_len != 123:
        raise AssertionError(f"settings_override: expected ring_len=123, got {cfg.ring_len}")


def main():
    tests = [
        test_circular_lerp_boundary,
        test_kernel_fractional_shift,
        test_settings_overrides,
    ]
    failures = []
    for test in tests:
        try:
            test()
        except Exception as exc:
            failures.append(f"{test.__name__}: {exc}")
    if failures:
        print("SMOKE CHECKS FAILED")
        for msg in failures:
            print(f"- {msg}")
        sys.exit(1)
    print("SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
