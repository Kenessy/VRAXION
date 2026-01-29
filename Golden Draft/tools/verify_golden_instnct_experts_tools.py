"""Verifier for INSTNCT expert router + offline checkpoint tools.

This is a dependency-light contract script (stdlib + torch) that validates:
- `vraxion.instnct.experts.LocationExpertRouter` routing semantics + hibernation restore
- `vraxion_mitosis_split.py` (offline expert clone)
- `tools/vraxion_prune_merge.py` (offline prune/merge)
- `tools/modularize_checkpoint.py` (checkpoint modularization)

Run from repo root:
  python tools/verify_golden_instnct_experts_tools.py

NOTE:
The release tree is imported from `S:\\AI\\Golden Code\\` (keep it clean).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

GOLDEN_SRC = Path(r"S:\AI\Golden Code")
sys.path.insert(0, str(GOLDEN_SRC))

from vraxion.instnct.experts import LocationExpertRouter, _hash_state_dict  # noqa: E402


def _safe_torch_load(path: str):
    """torch.load wrapper that avoids FutureWarning on newer PyTorch."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _make_dummy_ckpt(
    *,
    num_experts: int = 3,
    ring_len: int = 32,
    d_model: int = 2,
    vocab: int = 3,
):
    state: dict[str, torch.Tensor] = {}
    for exp_id in range(num_experts):
        w = torch.full((vocab, d_model), float(exp_id), dtype=torch.float32)
        b = torch.full((vocab,), float(exp_id), dtype=torch.float32)
        state[f"head.experts.{exp_id}.weight"] = w
        state[f"head.experts.{exp_id}.bias"] = b

    state["router_map"] = (torch.arange(ring_len, dtype=torch.long) % num_experts).clone()
    state["core_scalar"] = torch.tensor([123.0], dtype=torch.float32)

    param_names: list[str] = []
    for exp_id in range(num_experts):
        param_names.append(f"head.experts.{exp_id}.weight")
        param_names.append(f"head.experts.{exp_id}.bias")

    params = list(range(len(param_names)))
    optim_state = {pid: {"mom": torch.tensor([float(pid)])} for pid in params}
    optim = {"param_groups": [{"params": params}], "state": optim_state}

    return {
        "model": state,
        "num_experts": num_experts,
        "optim": optim,
        "param_names": list(param_names),
        "step": 7,
    }


def test_router_basic() -> None:
    rtr = LocationExpertRouter(d_model=2, vocab_size=3, num_experts=2)
    with torch.no_grad():
        rtr.experts[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        rtr.experts[0].bias.zero_()
        rtr.experts[1].weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]))
        rtr.experts[1].bias.zero_()

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ptr = torch.tensor([0, 1])
    y = rtr(x, ptr)
    exp = torch.tensor([[1.0, 2.0, 3.0], [6.0, 8.0, 14.0]])
    _assert(torch.allclose(y, exp), f"router mismatch: got={y} exp={exp}")

    y2 = rtr(x, None)
    exp2 = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 7.0]])
    _assert(torch.allclose(y2, exp2), f"router(None) mismatch: got={y2} exp={exp2}")


def test_router_hibernation_jit_fetch() -> None:
    rtr = LocationExpertRouter(d_model=2, vocab_size=3, num_experts=2)
    with torch.no_grad():
        rtr.experts[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        rtr.experts[0].bias.zero_()
        rtr.experts[1].weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]))
        rtr.experts[1].bias.zero_()

    with tempfile.TemporaryDirectory() as td:
        snap = Path(td) / "expert_001.pt"
        torch.save(rtr.experts[1].state_dict(), snap)
        disk_hash = _hash_state_dict(rtr.experts[1].state_dict())

        with torch.no_grad():
            rtr.experts[1].weight.zero_()
            rtr.experts[1].bias.zero_()

        rtr.hibernation_enabled = True
        rtr.hibernation_state = {1: {"offloaded": True, "path": str(snap), "hash": disk_hash}}

        x = torch.tensor([[3.0, 4.0]])
        ptr = torch.tensor([1])
        y = rtr(x, ptr)
        exp = torch.tensor([[6.0, 8.0, 14.0]])
        _assert(torch.allclose(y, exp), f"hibernation restore failed: got={y} exp={exp}")
        _assert(getattr(rtr, "hibernation_fetched", 0) == 1, "expected hibernation_fetched==1")
        _assert(getattr(rtr, "hibernation_corrupt", 0) == 0, "expected no corruption")
        _assert(rtr.hibernation_state[1]["offloaded"] is False, "expected offloaded cleared")


def test_offline_tools() -> None:
    ckpt = _make_dummy_ckpt(num_experts=3, ring_len=32)

    with tempfile.TemporaryDirectory() as td:
        tdpth = Path(td)
        ckpt_in = tdpth / "ckpt_in.pt"
        torch.save(ckpt, ckpt_in)

        ckpt_mito = tdpth / "ckpt_mito.pt"
        mito_py = ROOT / "vraxion_mitosis_split.py"
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(mito_py),
                "--checkpoint",
                str(ckpt_in),
                "--output",
                str(ckpt_mito),
                "--parent",
                "1",
                "--addresses",
                "0,5,17",
                "--noise",
                "0",
            ],
            check=True,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        mito = _safe_torch_load(str(ckpt_mito))
        _assert(mito["num_experts"] == 4, "mitosis should increase experts")
        st = mito["model"]
        _assert(st["router_map"][0].item() == 3, "addr 0 should redirect")
        _assert(st["router_map"][5].item() == 3, "addr 5 should redirect")
        _assert(st["router_map"][17].item() == 3, "addr 17 should redirect")
        _assert(torch.allclose(st["head.experts.3.weight"], st["head.experts.1.weight"]), "child weight copy")
        _assert(torch.allclose(st["head.experts.3.bias"], st["head.experts.1.bias"]), "child bias copy")

        opt = mito.get("optim")
        pnm = mito.get("param_names")
        _assert(opt is not None and pnm is not None, "mitosis should preserve optim+param_names")
        pids = list(opt["param_groups"][0]["params"])
        _assert(len(pids) == 8, "expected 8 params after cloning")
        _assert(pnm[-2:] == ["head.experts.3.weight", "head.experts.3.bias"], "child names appended")
        _assert(torch.allclose(opt["state"][6]["mom"], opt["state"][2]["mom"]), "momentum copy (weight)")
        _assert(torch.allclose(opt["state"][7]["mom"], opt["state"][3]["mom"]), "momentum copy (bias)")

        ckpt_prn = tdpth / "ckpt_prn.pt"
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "tools.vraxion_prune_merge",
                "--checkpoint",
                str(ckpt_in),
                "--output",
                str(ckpt_prn),
                "--merge-from",
                "2",
                "--merge-into",
                "0",
            ],
            check=True,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        prn = _safe_torch_load(str(ckpt_prn))
        _assert(prn["num_experts"] == 2, "prune should drop experts")
        stp = prn["model"]
        _assert("head.experts.2.weight" not in stp, "expert 2 removed")
        _assert((stp["router_map"] == 2).sum().item() == 0, "router map remapped")

        optp = prn.get("optim")
        pnmp = prn.get("param_names")
        _assert(optp is not None and pnmp is not None, "prune should preserve optim+param_names")
        _assert(all("head.experts.2." not in n for n in pnmp), "removed names trimmed")
        _assert(len(optp["param_groups"][0]["params"]) == 4, "expected 4 params after prune")

        out_dir = tdpth / "modular"
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "tools.modularize_checkpoint",
                "--checkpoint",
                str(ckpt_in),
                "--output",
                str(out_dir),
                "--tenure-all",
                "--force",
            ],
            check=True,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        router_state = _safe_torch_load(str(out_dir / "system" / "router.state"))
        _assert(router_state["num_experts"] == 3, "router.state should carry num_experts")
        _assert("router_map" in router_state["model"], "router_map should live in core model")
        _assert((out_dir / "experts" / "expert_000.pt").exists(), "expert_000.pt exists")
        _assert((out_dir / "experts" / "expert_001.pt").exists(), "expert_001.pt exists")
        _assert((out_dir / "experts" / "expert_002.pt").exists(), "expert_002.pt exists")


def main() -> int:
    test_router_basic()
    test_router_hibernation_jit_fetch()
    test_offline_tools()
    print("verify_golden_instnct_experts_tools: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
