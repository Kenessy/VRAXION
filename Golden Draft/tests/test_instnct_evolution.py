import os
import tempfile
import unittest

import torch

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

import tools.instnct_evolution as evo
from tools.instnct_evolution import EvolutionConfig, mutate_state_dict, run_evolution, save_evo_checkpoint


class _DummyHallway:
    """Minimal model stub implementing the state_dict APIs required by evolution."""

    def __init__(self, input_dim: int, num_classes: int, ring_len: int, slot_dim: int):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.ring_len = ring_len
        self.slot_dim = slot_dim
        self.w = torch.tensor(0.0)

    def state_dict(self):
        return {"w": self.w.clone()}

    def load_state_dict(self, sd):
        self.w = sd["w"].clone()


def _train_steps(model: _DummyHallway, loader, steps: int, dataset_name: str, tag: str):
    # Deterministic (and fast) "training": increase scalar weight.
    model.w = model.w + (0.1 * float(steps))
    return {"steps": steps, "tag": tag, "w": float(model.w.item())}


def _eval_model(model: _DummyHallway, eval_loader, dataset_name: str, tag: str):
    # Deterministic "eval": smaller loss is better.
    wval = float(model.w.item())
    loss = 1.0 / (1.0 + wval)
    acc = wval / (1.0 + wval)
    return {"eval_loss": loss, "eval_acc": acc, "tag": tag}


class TestInstnctEvolution(unittest.TestCase):
    def test_mutate_state_dict_pointer_only(self):
        torch.manual_seed(0)
        parobj = {
            "theta_ptr_reduced_0": torch.zeros(4),
            "not_pointer": torch.zeros(4),
            "an_int": torch.tensor([1, 2, 3], dtype=torch.int64),
        }

        chiobj = mutate_state_dict(parobj, std=0.5, pointer_only=True)

        # Non-float cloned.
        self.assertTrue(torch.equal(chiobj["an_int"], parobj["an_int"]))
        self.assertNotEqual(chiobj["an_int"].data_ptr(), parobj["an_int"].data_ptr())

        # Pointer param is mutated (very likely differs).
        self.assertFalse(torch.equal(chiobj["theta_ptr_reduced_0"], parobj["theta_ptr_reduced_0"]))

        # Non-pointer float is cloned without noise.
        self.assertTrue(torch.equal(chiobj["not_pointer"], parobj["not_pointer"]))
        self.assertNotEqual(chiobj["not_pointer"].data_ptr(), parobj["not_pointer"].data_ptr())

    def test_save_evo_checkpoint_writes_latest_and_periodic(self):
        loglst = []

        def logfn(msgstr: str):
            loglst.append(str(msgstr))

        calls = []

        def savfn(obj, path):
            calls.append((obj, str(path)))
            os.makedirs(os.path.dirname(str(path)), exist_ok=True)
            with open(str(path), "wb") as filobj:
                filobj.write(b"x")

        oldsv = evo.atomic_torch_save
        evo.atomic_torch_save = savfn
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                model = _DummyHallway(1, 2, 3, 4)
                save_evo_checkpoint(
                    0,
                    model,
                    {"ok": True},
                    {"eval_loss": 0.25, "eval_acc": 0.75},
                    0.75,
                    root=tmpdir,
                    checkpoint_every=1,
                    log=logfn,
                )

                evodir = os.path.join(tmpdir, "artifacts", "evolution")
                latpth = os.path.join(evodir, "evo_latest.pt")
                genpth = os.path.join(evodir, "evo_gen_000000.pt")
                self.assertTrue(os.path.exists(latpth))
                self.assertTrue(os.path.exists(genpth))

                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0][0]["gen"], 0)
                self.assertTrue(calls[0][1].endswith("evo_latest.pt"))
                self.assertTrue(calls[1][1].endswith("evo_gen_000000.pt"))

                self.assertIn("Evolution checkpoint saved @ gen 0", "\n".join(loglst))
        finally:
            evo.atomic_torch_save = oldsv

    def test_run_evolution_resume_start_gen(self):
        loglst = []

        def logfn(msgstr: str):
            loglst.append(str(msgstr))

        with tempfile.TemporaryDirectory() as tmpdir:
            evodir = os.path.join(tmpdir, "artifacts", "evolution")
            os.makedirs(evodir, exist_ok=True)
            latpth = os.path.join(evodir, "evo_latest.pt")
            with open(latpth, "wb"):
                pass

            cfgobj = EvolutionConfig(
                pop=2,
                gens=1,
                steps=2,
                mut_std=0.0,
                pointer_only=False,
                checkpoint_every=0,
                resume=True,
                checkpoint_individual=False,
                progress=False,
            )

            oldld = evo.torch.load
            oldck = evo.save_evo_checkpoint
            evo.torch.load = lambda *args, **kwargs: {"gen": 4, "model": {"w": torch.tensor(1.0)}}
            evo.save_evo_checkpoint = lambda *args, **kwargs: None
            try:
                outobj = run_evolution(
                    "seq_mnist",
                    loader=object(),
                    eval_loader=object(),
                    input_dim=1,
                    num_classes=2,
                    root=tmpdir,
                    ring_len=8,
                    slot_dim=4,
                    config=cfgobj,
                    model_ctor=_DummyHallway,
                    train_steps=_train_steps,
                    eval_model=_eval_model,
                    log=logfn,
                )
            finally:
                evo.torch.load = oldld
                evo.save_evo_checkpoint = oldck

            self.assertEqual(outobj["mode"], "evolution")
            joistr = "\n".join(loglst)
            self.assertIn("Evolution resume: loaded", joistr)
            self.assertIn("start_gen=5", joistr)

    def test_run_evolution_resume_corrupt_checkpoint(self):
        loglst = []

        def logfn(msgstr: str):
            loglst.append(str(msgstr))

        with tempfile.TemporaryDirectory() as tmpdir:
            evodir = os.path.join(tmpdir, "artifacts", "evolution")
            os.makedirs(evodir, exist_ok=True)
            latpth = os.path.join(evodir, "evo_latest.pt")
            with open(latpth, "wb"):
                pass

            cfgobj = EvolutionConfig(
                pop=2,
                gens=1,
                steps=1,
                mut_std=0.0,
                pointer_only=False,
                checkpoint_every=0,
                resume=True,
                checkpoint_individual=False,
                progress=False,
            )

            oldld = evo.torch.load
            oldck = evo.save_evo_checkpoint
            evo.torch.load = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
            evo.save_evo_checkpoint = lambda *args, **kwargs: None
            try:
                run_evolution(
                    "dummy",
                    loader=object(),
                    eval_loader=object(),
                    input_dim=1,
                    num_classes=2,
                    root=tmpdir,
                    ring_len=8,
                    slot_dim=4,
                    config=cfgobj,
                    model_ctor=_DummyHallway,
                    train_steps=_train_steps,
                    eval_model=_eval_model,
                    log=logfn,
                )
            finally:
                evo.torch.load = oldld
                evo.save_evo_checkpoint = oldck

            self.assertTrue(any(linesx.startswith("Evolution resume failed:") for linesx in loglst))


if __name__ == "__main__":
    unittest.main()
