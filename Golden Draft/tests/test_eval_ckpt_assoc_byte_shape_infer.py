import unittest

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)


class TestEvalCkptAssocByteShapeInfer(unittest.TestCase):
    def test_infer_absolute_hallway_single_head(self) -> None:
        import torch

        from tools.eval_ckpt_assoc_byte import _infer_absolute_hallway_from_state

        state = {
            "router_map": torch.zeros(2048, dtype=torch.int64),
            "input_proj.weight": torch.zeros((256, 1), dtype=torch.float32),
            "head.single.weight": torch.zeros((256, 256), dtype=torch.float32),
        }
        shape = _infer_absolute_hallway_from_state(state)
        self.assertEqual(shape["ring_len"], 2048)
        self.assertEqual(shape["slot_dim"], 256)
        self.assertEqual(shape["num_classes"], 256)
        self.assertEqual(shape["expert_heads"], 1)

    def test_infer_absolute_hallway_multi_head(self) -> None:
        import torch

        from tools.eval_ckpt_assoc_byte import _infer_absolute_hallway_from_state

        state = {
            "router_map": torch.zeros(8192, dtype=torch.int64),
            "input_proj.weight": torch.zeros((576, 1), dtype=torch.float32),
            "head.experts.0.weight": torch.zeros((256, 576), dtype=torch.float32),
            "head.experts.1.weight": torch.zeros((256, 576), dtype=torch.float32),
        }
        shape = _infer_absolute_hallway_from_state(state)
        self.assertEqual(shape["ring_len"], 8192)
        self.assertEqual(shape["slot_dim"], 576)
        self.assertEqual(shape["num_classes"], 256)
        self.assertEqual(shape["expert_heads"], 2)


if __name__ == "__main__":
    unittest.main()

