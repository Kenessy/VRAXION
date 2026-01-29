"""INSTNCT evaluation utilities (Golden Draft).

This module is a behavior-preserving extraction/refactor of evaluation-related
helpers from the legacy monolithic training script.

Design goals:
- Keep evaluation behavior identical (metrics, log strings, return dict keys).
- Make dependencies explicit (device, dtype, autocast context, logging).
- Keep the surface area small and easy to unit-test.

This module intentionally does *not* reach into unrelated runner state. Any
value that used to be a global in the monolith should be passed in via a config
object or injected callables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset


LogFn = Callable[[str], None]
AmpAutocastFn = Callable[[], ContextManager[Any]]


@dataclass(frozen=True)
class EvalLoaderSpec:
    """Parameters used to build deterministic evaluation DataLoaders."""

    eval_samples: int
    batch_size: int

    # Legacy defaults.
    num_workers: int = 0
    pin_memory: bool = True


@dataclass(frozen=True)
class EvalDeps:
    """Runtime dependencies that were globals in the legacy script."""

    device: torch.device | str
    dtype: torch.dtype
    amp_autocast: AmpAutocastFn
    log: LogFn

    # Behavior toggles.
    synth_mode: str = ""
    mi_shuffle: bool = False
    mitosis_enabled: bool = False

    # Optional deterministic RNG source for MI label shuffling.
    # If None, torch's global RNG is used (legacy behavior).
    mi_shuffle_generator: Optional[torch.Generator] = None


def _default_collate(batch: Sequence[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legacy fallback collate: stack xs and create int64 labels."""

    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def build_eval_loader_from_subset(
    train_ds: Any,
    *,
    spec: EvalLoaderSpec,
    input_collate: Optional[Callable[[Sequence[Any]], Any]] = None,
) -> Tuple[DataLoader, int]:
    """Build a deterministic eval DataLoader from the *front* of train_ds.

    Legacy behavior:
    - eval_size = min(eval_samples, len(train_ds))
    - If train_ds is already a Subset, reuse its first indices.
    - shuffle=False, num_workers=0, pin_memory=True by default.

    Returns (loader, eval_size).
    """

    eval_size = min(int(spec.eval_samples), len(train_ds))

    if isinstance(train_ds, Subset):
        indices = train_ds.indices[:eval_size]
        eval_subset = Subset(train_ds.dataset, indices)
    else:
        eval_subset = Subset(train_ds, list(range(eval_size)))

    def _collate(batch: Sequence[Any]) -> Any:
        if input_collate is not None:
            return input_collate(batch)
        return _default_collate(batch)  # type: ignore[arg-type]

    loader = DataLoader(
        eval_subset,
        batch_size=int(spec.batch_size),
        shuffle=False,
        num_workers=int(spec.num_workers),
        pin_memory=bool(spec.pin_memory),
        collate_fn=_collate,
    )
    return loader, eval_size


def build_eval_loader_from_dataset(
    eval_ds: Any,
    *,
    spec: EvalLoaderSpec,
    input_collate: Optional[Callable[[Sequence[Any]], Any]] = None,
) -> Tuple[DataLoader, int]:
    """Build a deterministic eval DataLoader from the *front* of eval_ds.

    Returns (loader, eval_size).
    """

    eval_size = min(int(spec.eval_samples), len(eval_ds))
    eval_subset = Subset(eval_ds, list(range(eval_size)))

    def _collate(batch: Sequence[Any]) -> Any:
        if input_collate is not None:
            return input_collate(batch)
        return _default_collate(batch)  # type: ignore[arg-type]

    loader = DataLoader(
        eval_subset,
        batch_size=int(spec.batch_size),
        shuffle=False,
        num_workers=int(spec.num_workers),
        pin_memory=bool(spec.pin_memory),
        collate_fn=_collate,
    )
    return loader, eval_size


def log_eval_overlap(
    train_ds: Any,
    eval_ds: Any,
    eval_size: int,
    label: str,
    *,
    log: LogFn,
) -> None:
    """Log overlap count if train/eval share a base dataset.

    The overlap calculation preserves legacy semantics:
    - If base datasets differ: overlap=0.
    - If eval_ds isn't a Subset: overlap=eval_size.
    - If train_ds isn't a Subset: overlap=len(eval_idx).
    - Else: overlap=len(intersection(train_idx, eval_idx)).
    """

    def base_and_indices(ds: Any) -> Tuple[Any, Optional[set[int]]]:
        if isinstance(ds, Subset):
            # Subset.indices may be a list/sequence of ints.
            return ds.dataset, set(int(i) for i in ds.indices)
        return ds, None

    train_base, train_idx = base_and_indices(train_ds)
    eval_base, eval_idx = base_and_indices(eval_ds)

    if train_base is eval_base:
        if eval_idx is None:
            overlap = int(eval_size)
        elif train_idx is None:
            overlap = len(eval_idx)
        else:
            overlap = len(train_idx.intersection(eval_idx))
        log(f"[eval] split={label} overlap={overlap}/{eval_size} (shared base dataset)")
    else:
        log(f"[eval] split={label} overlap=0/{eval_size} (disjoint datasets)")


def _ensure_head_out_features(model: nn.Module) -> None:
    """Legacy patch: populate model.head.out_features if missing.

    Some legacy model variants store the classifier head under a wrapper
    (experts/single). The monolith patched this on the fly.
    """

    if not hasattr(model, "head"):
        return

    head = getattr(model, "head")
    if hasattr(head, "out_features"):
        return

    # Match the legacy probing order.
    if hasattr(head, "experts") and getattr(head, "experts"):
        experts = getattr(head, "experts")
        head.out_features = experts[0].out_features
    elif hasattr(head, "single") and getattr(head, "single") is not None:
        head.out_features = head.single.out_features


def _mi_bits_from_joint(joint: torch.Tensor) -> Optional[float]:
    """Compute mutual information in bits for a (class x bin) joint histogram."""

    if joint.numel() == 0:
        return None
    if int(joint.sum().item()) <= 0:
        return None

    p = joint.float() / joint.sum()
    pc = p.sum(dim=1, keepdim=True)
    pb = p.sum(dim=0, keepdim=True)

    # Legacy numerical guard.
    mi = (p * (torch.log(p + 1e-12) - torch.log(pc * pb + 1e-12))).sum()
    return float(mi / math.log(2.0))


def eval_model(
    model: nn.Module,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    dataset_name: str,
    model_name: str,
    *,
    deps: EvalDeps,
) -> Dict[str, Any]:
    """Run an evaluation loop and compute telemetry.

    This is a near-direct refactor of the legacy eval loop. It preserves:
    - Autocast usage via deps.amp_autocast()
    - Input casting to deps.dtype (after moving to deps.device)
    - Metric computations and return dict keys
    - Side-effect: sets model.last_eval_acc

    Optional telemetry is read from attributes that may be produced by the model
    during forward passes:
    - model.last_ptr_bins (int tensor): pointer histogram bin for each sample
    - model.pointer_hist_bins (int): number of bins for last_ptr_bins
    - model.ptr_flip_rate (float): pointer flip rate for the last step

    Mitosis telemetry (when deps.mitosis_enabled):
    - model.last_ptr_int (int tensor): pointer address per sample
    - model.router_map (int tensor): address -> expert id map
    - model.ring_len or model.ring_range: number of addresses in ring

    Returns a dict matching the legacy keys.
    """

    # Preserve legacy behavior: cast the model to device + dtype.
    model = model.to(deps.device, dtype=deps.dtype)
    model.eval()

    _ensure_head_out_features(model)

    criterion = nn.CrossEntropyLoss()

    collect_mitosis = bool(deps.mitosis_enabled)
    addr_loss_sum: Optional[torch.Tensor] = None
    addr_count: Optional[torch.Tensor] = None
    expert_counts: Optional[torch.Tensor] = None

    ring_len_raw = getattr(model, "ring_range", getattr(model, "ring_len", 0))
    ring_len = int(ring_len_raw or 0)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    dom0_correct = 0
    dom0_seen = 0
    dom1_correct = 0
    dom1_seen = 0

    mi_bins = int(getattr(model, "pointer_hist_bins", 128))

    # Kept on CPU to match legacy code and reduce GPU memory.
    joint = torch.zeros((model.head.out_features, mi_bins), dtype=torch.long)
    joint_shuffle = torch.zeros_like(joint) if deps.mi_shuffle else None

    ptr_flip_sum = 0.0
    ptr_steps = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(deps.device, non_blocking=True)
            if inputs.dtype != deps.dtype:
                # Legacy pattern: dtype-only cast after device move.
                inputs = inputs.to(deps.dtype)

            targets = targets.to(deps.device, non_blocking=True)

            with deps.amp_autocast():
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                if collect_mitosis:
                    loss_vec = F.cross_entropy(outputs, targets, reduction="none")

            total_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_seen += inputs.size(0)

            if deps.synth_mode == "assoc_mix":
                dom0_mask = targets < 2
                dom1_mask = ~dom0_mask

                dom0_seen += dom0_mask.sum().item()
                dom1_seen += dom1_mask.sum().item()

                dom0_correct += ((preds == targets) & dom0_mask).sum().item()
                dom1_correct += ((preds == targets) & dom1_mask).sum().item()

            if collect_mitosis and ring_len > 0:
                addr = getattr(model, "last_ptr_int", None)
                router_map = getattr(model, "router_map", None)

                if addr is not None:
                    addr_cpu = addr.to(torch.long).cpu()
                    loss_cpu = loss_vec.detach().cpu().to(torch.float32)

                    if addr_loss_sum is None:
                        addr_loss_sum = torch.zeros(ring_len, dtype=torch.float32)
                        addr_count = torch.zeros(ring_len, dtype=torch.float32)

                    # Legacy aggregation: sum loss per address and count per address.
                    addr_loss_sum += torch.bincount(addr_cpu, weights=loss_cpu, minlength=ring_len)[:ring_len]
                    addr_count += torch.bincount(addr_cpu, minlength=ring_len)[:ring_len]

                    if router_map is not None and getattr(router_map, "numel", lambda: 0)() > 0:
                        mapped = router_map.detach().cpu()
                        mapped_ids = mapped[addr_cpu.clamp(0, mapped.numel() - 1)]

                        num_experts = int(getattr(model.head, "num_experts", 1))
                        if expert_counts is None:
                            expert_counts = torch.zeros(num_experts, dtype=torch.float32)
                        expert_counts += torch.bincount(mapped_ids, minlength=num_experts).float()[:num_experts]

            if hasattr(model, "last_ptr_bins"):
                # Keep bincount accumulation on CPU (joint is CPU).
                bins = getattr(model, "last_ptr_bins").detach().to(device="cpu", dtype=torch.long)
                labels = targets.detach().cpu().to(torch.long)

                idx = labels * mi_bins + bins
                joint += torch.bincount(idx, minlength=joint.numel()).view_as(joint)

                if deps.mi_shuffle:
                    gen = deps.mi_shuffle_generator
                    perm = torch.randperm(labels.numel(), generator=gen)
                    labels_shuf = labels[perm]

                    idx_shuf = labels_shuf * mi_bins + bins
                    assert joint_shuffle is not None  # for type-checkers
                    joint_shuffle += torch.bincount(idx_shuf, minlength=joint.numel()).view_as(joint)

            if hasattr(model, "ptr_flip_rate"):
                ptr_flip_sum += float(getattr(model, "ptr_flip_rate"))
                ptr_steps += 1

    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)

    dom0_acc = dom0_correct / max(dom0_seen, 1) if deps.synth_mode == "assoc_mix" else None
    dom1_acc = dom1_correct / max(dom1_seen, 1) if deps.synth_mode == "assoc_mix" else None

    mi_bits = _mi_bits_from_joint(joint)
    mi_bits_shuffle = _mi_bits_from_joint(joint_shuffle) if joint_shuffle is not None else None

    ptr_flip_rate = (ptr_flip_sum / ptr_steps) if ptr_steps else None

    tei = None
    if mi_bits is not None and ptr_flip_rate is not None:
        tei = acc * mi_bits * (1.0 - ptr_flip_rate)

    try:
        model.last_eval_acc = float(acc)
    except Exception:
        model.last_eval_acc = None

    mitosis_parent = None
    mitosis_hot = None
    mitosis_imbalance = None

    if collect_mitosis and expert_counts is not None and addr_loss_sum is not None:
        total = expert_counts.sum().item()
        if total > 0:
            shares = expert_counts / total
            mitosis_parent = int(torch.argmax(shares).item())
            mitosis_imbalance = float(shares[mitosis_parent].item())

            router_map = getattr(model, "router_map", None)
            if router_map is not None and getattr(router_map, "numel", lambda: 0)() > 0:
                assert addr_count is not None
                loss_mean = addr_loss_sum / addr_count.clamp(min=1.0)

                router_cpu = router_map.detach().cpu()
                parent_mask = router_cpu == mitosis_parent
                addr_idx = torch.nonzero(parent_mask, as_tuple=False).view(-1)

                if addr_idx.numel() > 0:
                    parent_losses = loss_mean[addr_idx]
                    top_k = min(5, parent_losses.numel())
                    _top_vals, top_idx = torch.topk(parent_losses, k=top_k, largest=True)

                    mitosis_hot = [int(addr_idx[i].item()) for i in top_idx]

    # Preserve legacy stable log line.
    if deps.synth_mode == "assoc_mix":
        deps.log(
            f"{dataset_name} | {model_name} | eval_loss {avg_loss:.4f} | eval_acc {acc:.4f} | "
            f"eval_acc_d0 {dom0_acc:.4f} | eval_acc_d1 {dom1_acc:.4f} | eval_n {total_seen}"
        )
    else:
        deps.log(f"{dataset_name} | {model_name} | eval_loss {avg_loss:.4f} | eval_acc {acc:.4f} | eval_n {total_seen}")

    return {
        "eval_loss": avg_loss,
        "eval_acc": acc,
        "eval_acc_d0": dom0_acc,
        "eval_acc_d1": dom1_acc,
        "eval_n": total_seen,
        "eval_mi_bits": mi_bits,
        "eval_mi_bits_shuffled": mi_bits_shuffle,
        "eval_ptr_flip_rate": ptr_flip_rate,
        "eval_tei": tei,
        "mitosis_parent_expert": mitosis_parent,
        "mitosis_hot_addresses": mitosis_hot,
        "mitosis_expert_imbalance": mitosis_imbalance,
    }
