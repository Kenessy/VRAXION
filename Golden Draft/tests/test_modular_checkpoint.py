import os
import tempfile
import unittest

import torch
import torch.nn as nn

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from vraxion.instnct import modular_checkpoint as mc


class ModularCheckpointUtilsTests(unittest.TestCase):
    def test_split_model_state_dict_splits_experts_and_moves_to_cpu(self):
        state = {
            "embed.weight": torch.randn(4, 3),
            "head.experts.0.weight": torch.randn(3, 3),
            "head.experts.0.bias": torch.randn(3),
            "head.experts.1.weight": torch.randn(3, 3),
        }

        core, experts = mc._split_model_state_dict(state)
        self.assertIn("embed.weight", core)
        self.assertIn(0, experts)
        self.assertIn(1, experts)
        self.assertIn("weight", experts[0])
        self.assertIn("bias", experts[0])

        self.assertEqual(core["embed.weight"].device.type, "cpu")
        self.assertEqual(experts[0]["weight"].device.type, "cpu")

    def test_modular_paths_creates_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            router_path, experts_dir, meta_path = mc._modular_paths(td)

            self.assertTrue(os.path.isdir(os.path.join(td, "system")))
            self.assertTrue(os.path.isdir(experts_dir))
            self.assertTrue(router_path.endswith(os.path.join("system", "router.state")))
            self.assertTrue(meta_path.endswith(os.path.join("experts", "meta.json")))

    def test_resolve_modular_dir_handles_pt_suffix(self):
        with tempfile.TemporaryDirectory() as td:
            out = mc._resolve_modular_dir("run.pt", td, None)
            self.assertTrue(out.endswith("run_modular"))
            self.assertTrue(os.path.isdir(out))

    def test_hash_state_dict_is_deterministic_sorted_keys(self):
        a = {
            "b": torch.tensor([2, 3, 4]),
            "a": torch.tensor([1]),
        }
        b = {
            "a": torch.tensor([1]),
            "b": torch.tensor([2, 3, 4]),
        }
        self.assertEqual(mc._hash_state_dict(a), mc._hash_state_dict(b))


class _TinyHead(nn.Module):
    def __init__(self, num_experts: int = 2):
        super().__init__()
        self.num_experts = int(num_experts)
        self.experts = nn.ModuleList([nn.Linear(2, 3) for _ in range(self.num_experts)])


class _TinyModel(nn.Module):
    def __init__(self, num_experts: int = 2):
        super().__init__()
        self.head = _TinyHead(num_experts=num_experts)
        # Optional metadata fields used by the checkpoint payload.
        self.update_scale = 0.123
        self.ptr_inertia = 0.456


class ModularCheckpointRoundTripTests(unittest.TestCase):
    def test_save_and_load_modular_checkpoint_roundtrip(self):
        torch.manual_seed(0)
        m1 = _TinyModel(num_experts=2)
        opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as td:
            base_dir = os.path.join(td, "ckpt_modular")
            mc._save_modular_checkpoint(
                m1,
                opt1,
                scaler=None,
                step=12,
                losses=[1.0, 0.9],
                base_dir=base_dir,
                contrib_thresh=1.0,
                probation_steps=0,
                ttl_steps=0,
                gc_enabled=False,
            )

            torch.manual_seed(123)
            m2 = _TinyModel(num_experts=2)
            opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)
            _ = mc._load_modular_checkpoint(m2, opt2, scaler=None, base_dir=base_dir)

            st1 = m1.state_dict()
            st2 = m2.state_dict()

            for key in sorted(st1.keys()):
                if key.startswith("head.experts."):
                    self.assertTrue(torch.allclose(st1[key], st2[key]))

            # Meta should be present and sized correctly.
            self.assertEqual(getattr(m2, "expert_contrib"), [0.0, 0.0])
            self.assertEqual(getattr(m2, "expert_tenured"), [False, False])

    def test_router_state_excludes_expert_tensors(self):
        torch.manual_seed(0)
        m1 = _TinyModel(num_experts=2)
        opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as td:
            base_dir = os.path.join(td, "ckpt_modular")
            mc._save_modular_checkpoint(
                m1,
                opt1,
                scaler=None,
                step=1,
                losses=[1.0],
                base_dir=base_dir,
                contrib_thresh=1.0,
                probation_steps=0,
            )

            router_path = os.path.join(base_dir, "system", "router.state")
            ckpt = mc._torch_load_compat(router_path, map_location="cpu", weights_only=False)
            self.assertIsInstance(ckpt, dict)
            self.assertIn("model", ckpt)
            self.assertTrue(all(not k.startswith("head.experts.") for k in ckpt["model"].keys()))

    def test_load_attempts_strict_then_falls_back_to_strict_false(self):
        class _RecordingModel(_TinyModel):
            def __init__(self):
                super().__init__(num_experts=2)
                self.calls = []

            def load_state_dict(self, state_dict, strict=True):
                self.calls.append(bool(strict))
                if strict:
                    raise RuntimeError("forced strict failure")
                return super().load_state_dict(state_dict, strict=False)

        with tempfile.TemporaryDirectory() as td:
            base_dir = os.path.join(td, "ckpt_modular")
            router_path, _experts_dir, _meta_path = mc._modular_paths(base_dir)
            torch.save({"model": {}, "optim": None, "scaler": None}, router_path)

            m = _RecordingModel()
            _ = mc._load_modular_checkpoint(m, optimizer=None, scaler=None, base_dir=base_dir)
            self.assertEqual(m.calls, [True, False])

    def test_load_ignores_invalid_meta_json_and_preserves_existing_tracking(self):
        torch.manual_seed(0)
        m1 = _TinyModel(num_experts=2)
        opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as td:
            base_dir = os.path.join(td, "ckpt_modular")
            mc._save_modular_checkpoint(
                m1,
                opt1,
                scaler=None,
                step=3,
                losses=[0.1],
                base_dir=base_dir,
                contrib_thresh=1.0,
                probation_steps=0,
                ttl_steps=0,
                gc_enabled=False,
            )

            # Corrupt meta.json.
            meta_path = os.path.join(base_dir, "experts", "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write("{ this is not valid json")

            torch.manual_seed(123)
            m2 = _TinyModel(num_experts=2)
            # Sentinel values that should survive invalid meta load.
            m2.expert_contrib = ["SENTINEL"]
            m2.expert_last_used = ["SENTINEL"]
            m2.expert_created_step = ["SENTINEL"]
            m2.expert_tenured = ["SENTINEL"]

            opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)
            _ = mc._load_modular_checkpoint(m2, opt2, scaler=None, base_dir=base_dir)

            self.assertEqual(getattr(m2, "expert_contrib"), ["SENTINEL"])
            self.assertEqual(getattr(m2, "expert_tenured"), ["SENTINEL"])

    def test_load_skips_corrupt_expert_snapshot_payload(self):
        """A corrupt expert_###.pt file should not crash the whole resume."""

        torch.manual_seed(0)
        m1 = _TinyModel(num_experts=2)
        opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as td:
            base_dir = os.path.join(td, "ckpt_modular")
            mc._save_modular_checkpoint(
                m1,
                opt1,
                scaler=None,
                step=1,
                losses=[1.0],
                base_dir=base_dir,
                contrib_thresh=1.0,
                probation_steps=0,
                ttl_steps=0,
                gc_enabled=False,
            )

            # Corrupt expert_000.pt with a non-dict payload.
            expert0_path = os.path.join(base_dir, "experts", "expert_000.pt")
            torch.save(12345, expert0_path)

            torch.manual_seed(123)
            m2 = _TinyModel(num_experts=2)
            opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)

            before = {k: v.clone() for k, v in m2.head.experts[0].state_dict().items()}
            _ = mc._load_modular_checkpoint(m2, opt2, scaler=None, base_dir=base_dir)

            # Expert 0 should be unchanged (load skipped).
            after = m2.head.experts[0].state_dict()
            for k, v in before.items():
                self.assertTrue(torch.allclose(v, after[k]))

            # Expert 1 should still have loaded correctly.
            st1 = m1.state_dict()
            st2 = m2.state_dict()
            for key in sorted(st1.keys()):
                if key.startswith("head.experts.1."):
                    self.assertTrue(torch.allclose(st1[key], st2[key]))

    def test_expert_load_is_strict_false(self):
        torch.manual_seed(0)
        m1 = _TinyModel(num_experts=2)
        opt1 = torch.optim.SGD(m1.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as td:
            base_dir = os.path.join(td, "ckpt_modular")
            mc._save_modular_checkpoint(
                m1,
                opt1,
                scaler=None,
                step=1,
                losses=[1.0],
                base_dir=base_dir,
                contrib_thresh=1.0,
                probation_steps=0,
            )

            # Corrupt expert_000.pt by removing the bias key.
            expert_path = os.path.join(base_dir, "experts", "expert_000.pt")
            state = mc._torch_load_compat(expert_path, map_location="cpu", weights_only=True)
            self.assertIsInstance(state, dict)
            if "bias" in state:
                del state["bias"]
            torch.save(state, expert_path)

            # Load should not raise, and weight should still match.
            torch.manual_seed(123)
            m2 = _TinyModel(num_experts=2)
            opt2 = torch.optim.SGD(m2.parameters(), lr=0.01)
            mc._load_modular_checkpoint(m2, opt2, scaler=None, base_dir=base_dir)

            self.assertTrue(torch.allclose(m1.head.experts[0].weight, m2.head.experts[0].weight))


if __name__ == "__main__":
    unittest.main()
