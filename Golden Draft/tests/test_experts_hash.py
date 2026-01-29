import unittest

import torch

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from vraxion.instnct.experts import _hash_state_dict


class TestExpertsHash(unittest.TestCase):
    def test_hash_bfloat16_is_deterministic(self) -> None:
        # Regression test: bf16 tensors used to crash when converting to NumPy.
        state = {"w": torch.ones((2, 2), dtype=torch.bfloat16)}
        h1 = _hash_state_dict(state)
        h2 = _hash_state_dict(state)

        self.assertIsInstance(h1, str)
        assert h1 is not None
        self.assertEqual(len(h1), 64)
        self.assertEqual(h1, h2)

        # Hash should change when the tensor contents change.
        h3 = _hash_state_dict({"w": torch.zeros((2, 2), dtype=torch.bfloat16)})
        self.assertNotEqual(h1, h3)


if __name__ == "__main__":
    unittest.main()
