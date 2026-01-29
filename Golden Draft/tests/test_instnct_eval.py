import math
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from tools.instnct_eval import (
    EvalDeps,
    EvalLoaderSpec,
    build_eval_loader_from_dataset,
    build_eval_loader_from_subset,
    eval_model,
    log_eval_overlap,
)


def _deps(**overrides):
    log = overrides.pop("log", lambda _: None)
    return EvalDeps(
        device="cpu",
        dtype=torch.float32,
        amp_autocast=lambda: nullcontext(),
        log=log,
        **overrides,
    )


class _ToyXYDataset(Dataset):
    def __init__(self, xs: List[torch.Tensor], ys: List[int]) -> None:
        assert len(xs) == len(ys)
        self._xs = xs
        self._ys = ys

    def __len__(self) -> int:
        return len(self._xs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._xs[idx], self._ys[idx]


class _ZeroLogitsModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.head = SimpleNamespace(out_features=int(num_classes))

    def forward(self, x: torch.Tensor):
        bsz = int(x.size(0))
        return torch.zeros((bsz, int(self.head.out_features)), device=x.device, dtype=x.dtype), None


class _PtrBinsModel(nn.Module):
    def __init__(self, num_classes: int, pointer_hist_bins: int) -> None:
        super().__init__()
        self.head = SimpleNamespace(out_features=int(num_classes))
        self.pointer_hist_bins = int(pointer_hist_bins)
        self.ptr_flip_rate = 0.0

    def forward(self, x: torch.Tensor):
        # x is (B, 1) with ptr bin IDs stored as floats/ints
        self.last_ptr_bins = x.view(-1).to(torch.long).cpu()
        bsz = int(x.size(0))
        return torch.zeros((bsz, int(self.head.out_features)), device=x.device, dtype=x.dtype), None


class _PerfectPtrBinsModel(nn.Module):
    """Legacy-like (logits, aux) model with perfect accuracy and stable MI bins."""

    def __init__(self, num_classes: int = 3, mi_bins: int = 8, ptr_flip_rate: float = 0.2):
        super().__init__()
        self.head = nn.Linear(1, int(num_classes), bias=False)
        self.pointer_hist_bins = int(mi_bins)
        self._ptr_flip_rate = float(ptr_flip_rate)

    def forward(self, x: torch.Tensor):
        labels = x.view(-1).detach().to(torch.long).clamp(0, self.head.out_features - 1)
        self.last_ptr_bins = labels.cpu()

        logits = torch.zeros((x.size(0), self.head.out_features), device=x.device, dtype=x.dtype)
        ar = torch.arange(x.size(0), device=x.device)
        logits[ar, labels.to(x.device)] = 5.0

        self.ptr_flip_rate = self._ptr_flip_rate
        return logits, None


class _FlipSeqModel(nn.Module):
    def __init__(self, num_classes: int, flips: List[float]) -> None:
        super().__init__()
        self.head = SimpleNamespace(out_features=int(num_classes))
        self._flips = list(flips)
        self._calln = 0
        self.pointer_hist_bins = 8

    def forward(self, x: torch.Tensor):
        self.last_ptr_bins = x.view(-1).to(torch.long).cpu()
        self.ptr_flip_rate = float(self._flips[self._calln])
        self._calln += 1
        bsz = int(x.size(0))
        return torch.zeros((bsz, int(self.head.out_features)), device=x.device, dtype=x.dtype), None


class _NoOutFeaturesHead:
    def __init__(self, out_features: int) -> None:
        self.experts = [SimpleNamespace(out_features=int(out_features))]


class _PatchHeadModel(nn.Module):
    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.head = _NoOutFeaturesHead(int(out_features))

    def forward(self, x: torch.Tensor):
        bsz = int(x.size(0))
        return torch.zeros((bsz, int(self.head.experts[0].out_features)), device=x.device, dtype=x.dtype), None


class _MitosisModel(nn.Module):
    def __init__(self, ring_len: int, num_experts: int) -> None:
        super().__init__()
        self.ring_len = int(ring_len)
        self.head = SimpleNamespace(out_features=2, num_experts=int(num_experts))
        half = int(ring_len // 2)
        rmap = [0] * half + [1] * (int(ring_len) - half)
        self.router_map = torch.tensor(rmap, dtype=torch.long)

    def forward(self, x: torch.Tensor):
        # x is (B, 2): [scale, addr]
        scale = x[:, 0:1].to(torch.float32)
        addr = x[:, 1].to(torch.long)
        self.last_ptr_int = addr
        logits = torch.cat([scale, torch.zeros_like(scale)], dim=1).to(x.dtype)
        return logits, None


class _HotAddrMitosisModel(nn.Module):
    def __init__(self, *, num_classes: int = 2, ring_len: int = 8):
        super().__init__()
        self.head = nn.Linear(1, int(num_classes), bias=False)
        self.head.num_experts = 2
        self.ring_len = int(ring_len)
        self.router_map = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    def forward(self, x: torch.Tensor):
        addr = x.view(-1).detach().to(torch.long)
        self.last_ptr_int = addr.to(x.device)

        logits = torch.zeros((x.size(0), 2), device=x.device, dtype=x.dtype)
        for i, a in enumerate(addr.tolist()):
            if a < 4:
                logits[i, 0] = 5.0
            else:
                logits[i, 1] = float(a)
        return logits, None


class TestEvalLoaders(unittest.TestCase):
    def test_build_eval_loader_from_subset_plain_dataset(self) -> None:
        xs = [torch.tensor([i], dtype=torch.float32) for i in range(10)]
        ys = list(range(10))
        base = _ToyXYDataset(xs, ys)

        spec = EvalLoaderSpec(eval_samples=4, batch_size=2)
        loader, eval_size = build_eval_loader_from_subset(base, spec=spec)

        self.assertEqual(eval_size, 4)
        self.assertIsInstance(loader.dataset, Subset)
        self.assertEqual(list(loader.dataset.indices), [0, 1, 2, 3])

        got = []
        for xb, yb in loader:
            got.extend(yb.tolist())
        self.assertEqual(got, [0, 1, 2, 3])

    def test_build_eval_loader_from_subset_subset_dataset(self) -> None:
        xs = [torch.tensor([i], dtype=torch.float32) for i in range(10)]
        ys = list(range(10))
        base = _ToyXYDataset(xs, ys)
        sub = Subset(base, [9, 8, 7, 6, 5])

        spec = EvalLoaderSpec(eval_samples=3, batch_size=2)
        loader, eval_size = build_eval_loader_from_subset(sub, spec=spec)

        self.assertEqual(eval_size, 3)
        self.assertIsInstance(loader.dataset, Subset)
        self.assertIs(loader.dataset.dataset, base)
        self.assertEqual(list(loader.dataset.indices), [9, 8, 7])

        got = []
        for xb, yb in loader:
            got.extend(yb.tolist())
        self.assertEqual(got, [9, 8, 7])

    def test_build_eval_loader_from_dataset(self) -> None:
        xs = [torch.tensor([i], dtype=torch.float32) for i in range(6)]
        ys = list(range(6))
        base = _ToyXYDataset(xs, ys)

        spec = EvalLoaderSpec(eval_samples=5, batch_size=4)
        loader, eval_size = build_eval_loader_from_dataset(base, spec=spec)

        self.assertEqual(eval_size, 5)
        got = []
        for xb, yb in loader:
            got.extend(yb.tolist())
        self.assertEqual(got, [0, 1, 2, 3, 4])


class TestOverlapLog(unittest.TestCase):
    def test_overlap_shared_base(self) -> None:
        xs = [torch.tensor([i], dtype=torch.float32) for i in range(6)]
        ys = list(range(6))
        base = _ToyXYDataset(xs, ys)

        train = Subset(base, [0, 1, 2, 3])
        evald = Subset(base, [2, 3, 4])

        logs: List[str] = []
        log_eval_overlap(train, evald, eval_size=3, label="toy", log=logs.append)
        self.assertEqual(logs, ["[eval] split=toy overlap=2/3 (shared base dataset)"])

    def test_overlap_disjoint_base(self) -> None:
        xs = [torch.tensor([i], dtype=torch.float32) for i in range(3)]
        ys = [0, 1, 2]
        train = _ToyXYDataset(xs, ys)
        evald = _ToyXYDataset(xs, ys)

        logs: List[str] = []
        log_eval_overlap(train, evald, eval_size=3, label="toy", log=logs.append)
        self.assertEqual(logs, ["[eval] split=toy overlap=0/3 (disjoint datasets)"])


class TestEvalModel(unittest.TestCase):
    def test_eval_model_basic(self) -> None:
        xs = [torch.tensor([0.0], dtype=torch.float32) for _ in range(4)]
        ys = [0, 1, 2, 3]
        ds = _ToyXYDataset(xs, ys)

        spec = EvalLoaderSpec(eval_samples=4, batch_size=2)
        loader, _ = build_eval_loader_from_dataset(ds, spec=spec)

        logs: List[str] = []
        deps = _deps(log=logs.append)
        res = eval_model(_ZeroLogitsModel(num_classes=4), loader, "toy", "zero", deps=deps)

        self.assertEqual(res["eval_n"], 4)
        self.assertAlmostEqual(res["eval_acc"], 0.25, places=7)
        self.assertAlmostEqual(res["eval_loss"], math.log(4.0), places=7)
        self.assertIsNone(res["eval_acc_d0"])
        self.assertIsNone(res["eval_acc_d1"])

        self.assertEqual(logs, [f"toy | zero | eval_loss {math.log(4.0):.4f} | eval_acc {0.25:.4f} | eval_n 4"])

    def test_eval_model_assoc_mix_domain_split(self) -> None:
        xs = [torch.tensor([0.0], dtype=torch.float32) for _ in range(4)]
        ys = [0, 1, 2, 3]
        ds = _ToyXYDataset(xs, ys)

        spec = EvalLoaderSpec(eval_samples=4, batch_size=2)
        loader, _ = build_eval_loader_from_dataset(ds, spec=spec)

        logs: List[str] = []
        deps = _deps(log=logs.append, synth_mode="assoc_mix")
        res = eval_model(_ZeroLogitsModel(num_classes=4), loader, "toy", "zero", deps=deps)

        self.assertAlmostEqual(res["eval_acc"], 0.25, places=7)
        self.assertAlmostEqual(res["eval_acc_d0"], 0.5, places=7)
        self.assertAlmostEqual(res["eval_acc_d1"], 0.0, places=7)

        self.assertEqual(
            logs,
            [
                f"toy | zero | eval_loss {math.log(4.0):.4f} | eval_acc {0.25:.4f} | "
                f"eval_acc_d0 {0.5:.4f} | eval_acc_d1 {0.0:.4f} | eval_n 4"
            ],
        )

    def test_eval_model_mi_bits(self) -> None:
        ys = [0, 1, 2, 0, 1, 2]
        xs = [torch.tensor([float(y)], dtype=torch.float32) for y in ys]
        ds = _ToyXYDataset(xs, ys)

        spec = EvalLoaderSpec(eval_samples=len(ds), batch_size=3)
        loader, _ = build_eval_loader_from_dataset(ds, spec=spec)

        res = eval_model(_PtrBinsModel(num_classes=3, pointer_hist_bins=8), loader, "toy", "ptr", deps=_deps())

        self.assertAlmostEqual(res["eval_mi_bits"], math.log(3.0) / math.log(2.0), places=6)
        self.assertIsNone(res["eval_mi_bits_shuffled"])

    def test_eval_model_mi_shuffle_deterministic(self) -> None:
        ys = [0, 1, 2, 0, 1, 2]
        xs = [torch.tensor([float(y)], dtype=torch.float32) for y in ys]
        ds = _ToyXYDataset(xs, ys)

        spec = EvalLoaderSpec(eval_samples=len(ds), batch_size=2)
        loader, _ = build_eval_loader_from_dataset(ds, spec=spec)

        gen = torch.Generator().manual_seed(123)
        deps = _deps(mi_shuffle=True, mi_shuffle_generator=gen)

        model = _PtrBinsModel(num_classes=3, pointer_hist_bins=8)
        res = eval_model(model, loader, "toy", "ptr", deps=deps)
        got = res["eval_mi_bits_shuffled"]
        self.assertIsNotNone(got)

        gen2 = torch.Generator().manual_seed(123)
        mi_bins = int(model.pointer_hist_bins)
        joint_s = torch.zeros((3, mi_bins), dtype=torch.long)
        for i in range(0, len(ys), 2):
            labels = torch.tensor(ys[i : i + 2], dtype=torch.long)
            bins = labels.clone()
            perm = torch.randperm(labels.numel(), generator=gen2)
            shuf = labels[perm]
            joint_s += torch.bincount(shuf * mi_bins + bins, minlength=joint_s.numel()).view_as(joint_s)

        def mi_bits(j: torch.Tensor) -> float:
            p = j.to(torch.float32) / float(j.sum().item())
            pc = p.sum(dim=1, keepdim=True)
            pb = p.sum(dim=0, keepdim=True)
            mi = (p * (torch.log(p + 1e-12) - torch.log(pc * pb + 1e-12))).sum().item()
            return float(mi / math.log(2.0))

        self.assertAlmostEqual(float(got), mi_bits(joint_s), places=6)

    def test_ptr_flip_rate_is_unweighted_mean_per_step(self) -> None:
        flips = [0.1, 0.5, 0.9]
        xs = [torch.tensor([0.0], dtype=torch.float32) for _ in range(5)]
        ys = [0, 0, 0, 0, 0]
        ds = _ToyXYDataset(xs, ys)

        spec = EvalLoaderSpec(eval_samples=len(ds), batch_size=2)
        loader, _ = build_eval_loader_from_dataset(ds, spec=spec)

        res = eval_model(_FlipSeqModel(num_classes=2, flips=flips), loader, "toy", "flip", deps=_deps())
        self.assertAlmostEqual(res["eval_ptr_flip_rate"], sum(flips) / len(flips), places=7)

    def test_head_out_features_is_patched(self) -> None:
        xs = [torch.tensor([0.0], dtype=torch.float32) for _ in range(2)]
        ys = [0, 1]
        ds = _ToyXYDataset(xs, ys)

        spec = EvalLoaderSpec(eval_samples=2, batch_size=2)
        loader, _ = build_eval_loader_from_dataset(ds, spec=spec)

        model = _PatchHeadModel(out_features=5)
        _ = eval_model(model, loader, "toy", "patch", deps=_deps())
        self.assertTrue(hasattr(model.head, "out_features"))
        self.assertEqual(int(model.head.out_features), 5)

    def test_eval_model_assoc_mix_mi_tei_and_last_eval_acc(self) -> None:
        xs = [torch.tensor([float(i % 3)], dtype=torch.float32) for i in range(6)]
        ys = [int(i % 3) for i in range(6)]
        loader = torch.utils.data.DataLoader(list(zip(xs, ys)), batch_size=3, shuffle=False)

        model = _PerfectPtrBinsModel(num_classes=3, mi_bins=8, ptr_flip_rate=0.2)

        logs: List[str] = []
        deps = _deps(log=logs.append, synth_mode="assoc_mix")
        out = eval_model(model, loader, dataset_name="ds", model_name="m", deps=deps)

        self.assertEqual(out["eval_n"], 6)
        self.assertAlmostEqual(out["eval_acc"], 1.0, places=6)
        self.assertAlmostEqual(out["eval_acc_d0"], 1.0, places=6)
        self.assertAlmostEqual(out["eval_acc_d1"], 1.0, places=6)

        expected_mi = math.log(3.0, 2.0)
        self.assertIsNotNone(out["eval_mi_bits"])
        self.assertAlmostEqual(out["eval_mi_bits"], expected_mi, places=4)

        self.assertAlmostEqual(out["eval_ptr_flip_rate"], 0.2, places=6)
        self.assertAlmostEqual(out["eval_tei"], expected_mi * 0.8, places=4)
        self.assertAlmostEqual(getattr(model, "last_eval_acc", None), 1.0, places=6)

        self.assertEqual(len(logs), 1)
        self.assertIn("ds | m | eval_loss", logs[0])
        self.assertIn("eval_acc 1.0000", logs[0])
        self.assertIn("eval_n 6", logs[0])

    def test_mitosis_telemetry_parent_and_hot_addresses(self) -> None:
        xs = [
            torch.tensor([0.0, 0.0], dtype=torch.float32),
            torch.tensor([1.0, 1.0], dtype=torch.float32),
            torch.tensor([2.0, 2.0], dtype=torch.float32),
            torch.tensor([3.0, 3.0], dtype=torch.float32),
            torch.tensor([0.0, 0.0], dtype=torch.float32),
            torch.tensor([0.0, 0.0], dtype=torch.float32),
            torch.tensor([0.5, 4.0], dtype=torch.float32),
            torch.tensor([0.5, 5.0], dtype=torch.float32),
        ]
        ys = [0] * len(xs)
        ds = _ToyXYDataset(xs, ys)

        spec = EvalLoaderSpec(eval_samples=len(ds), batch_size=4)
        loader, _ = build_eval_loader_from_dataset(ds, spec=spec)

        model = _MitosisModel(ring_len=8, num_experts=2)
        out = eval_model(model, loader, "toy", "mito", deps=_deps(mitosis_enabled=True))

        self.assertEqual(out["mitosis_parent_expert"], 0)
        self.assertAlmostEqual(out["mitosis_expert_imbalance"], 0.75, places=6)

        hot = out["mitosis_hot_addresses"]
        self.assertIsInstance(hot, list)
        self.assertEqual(hot[:4], [0, 1, 2, 3])

    def test_eval_model_mitosis_hot_addresses(self) -> None:
        xs = [torch.tensor([float(v)], dtype=torch.float32) for v in [4, 5, 6, 7, 7]]
        ys = [0 for _ in xs]
        loader = torch.utils.data.DataLoader(list(zip(xs, ys)), batch_size=5, shuffle=False)

        model = _HotAddrMitosisModel(num_classes=2, ring_len=8)
        out = eval_model(model, loader, dataset_name="ds", model_name="m", deps=_deps(mitosis_enabled=True))

        self.assertEqual(out["mitosis_parent_expert"], 1)
        self.assertAlmostEqual(out["mitosis_expert_imbalance"], 1.0, places=6)
        self.assertIsInstance(out["mitosis_hot_addresses"], list)
        self.assertTrue(out["mitosis_hot_addresses"])
        self.assertEqual(out["mitosis_hot_addresses"][0], 7)
        self.assertTrue(set(out["mitosis_hot_addresses"]).issubset({4, 5, 6, 7}))


if __name__ == "__main__":
    unittest.main()
