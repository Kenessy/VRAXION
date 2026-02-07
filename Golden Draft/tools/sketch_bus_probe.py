"""
Hash/Sketch Bus probe (E1)

Purpose
-------
Test whether a fixed-size "sketch bus" can provide usable global context when:
- every node participates every step (no top-k exclusion),
- communication bandwidth stays fixed (M buckets), and
- collisions are the price.

This probe is intentionally "learning-free": it measures collision/interference and
how well a node can approximate the global mean message by reading a small number
of hashed buckets.

Notes
-----
- Default device is CPU for determinism. On CUDA, scatter_add uses atomics and can
  be non-deterministic across runs.
- This is a probe harness, not a production module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


_U64_MASK = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """Deterministic 64-bit mix (SplitMix64)."""
    x = (x + 0x9E3779B97F4A7C15) & _U64_MASK
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _U64_MASK
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _U64_MASK
    z = z ^ (z >> 31)
    return z & _U64_MASK


def _hash_indices_and_signs(
    *,
    node_ids: Sequence[int],
    m_buckets: int,
    k: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      idx:   [N, k] int64 bucket indices in [0, m_buckets)
      signs: [N, k] float32 in {-1.0, +1.0}
    """
    if m_buckets <= 0:
        raise ValueError("m_buckets must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")

    idx = torch.empty((len(node_ids), k), dtype=torch.int64)
    signs = torch.empty((len(node_ids), k), dtype=torch.float32)

    # Two degrees of freedom:
    # - node_id changes which buckets each node touches
    # - j changes which of the k buckets we use per node
    for n, node_id in enumerate(node_ids):
        base = int(node_id) & _U64_MASK
        for j in range(k):
            h = _splitmix64(base ^ (seed + 0xD1B54A32D192ED03 * (j + 1)))
            idx[n, j] = int(h % m_buckets)
            # Use the top bit as a stable sign hash.
            signs[n, j] = 1.0 if ((h >> 63) & 1) == 0 else -1.0
    return idx, signs


def _bucket_occupancy(idx: torch.Tensor, m_buckets: int) -> torch.Tensor:
    """idx: [N,k] -> counts: [M]"""
    flat = idx.reshape(-1)
    counts = torch.bincount(flat, minlength=m_buckets)
    return counts.to(torch.int64)


@dataclass(frozen=True)
class SketchBusStats:
    # Configuration
    n_nodes: int
    d_msg: int
    m_buckets: int
    k_write: int
    k_read: int
    seed: int
    batch_size: int
    signal_alpha: float
    noise_std: float
    signed: int
    # Collision / occupancy stats
    writes_total: int
    buckets_used: int
    buckets_empty: int
    max_bucket_load: int
    mean_bucket_load: float
    p95_bucket_load: float
    collision_write_frac: float
    # Context quality (approximate global mean message)
    ctx_cos_mean: float
    ctx_mse_mean: float
    ctx_var_across_nodes: float
    # Timing
    wall_time_s: float


def run_probe(
    *,
    n_nodes: int,
    d_msg: int,
    m_buckets: int,
    k_write: int,
    k_read: int,
    batch_size: int,
    seed: int,
    device: str,
    signal_alpha: float = 1.0,
    noise_std: float = 1.0,
    signed: bool = False,
) -> SketchBusStats:
    t0 = time.perf_counter()

    if n_nodes <= 0 or d_msg <= 0 or batch_size <= 0:
        raise ValueError("n_nodes, d_msg, and batch_size must be > 0")

    node_ids = list(range(n_nodes))
    write_idx, write_sign = _hash_indices_and_signs(
        node_ids=node_ids, m_buckets=m_buckets, k=k_write, seed=seed
    )
    read_idx, read_sign = _hash_indices_and_signs(
        node_ids=node_ids, m_buckets=m_buckets, k=k_read, seed=seed ^ 0xA5A5A5A5
    )
    if not signed:
        write_sign.fill_(1.0)
        read_sign.fill_(1.0)

    counts = _bucket_occupancy(write_idx, m_buckets)
    writes_total = int(n_nodes * k_write)
    buckets_used = int((counts > 0).sum().item())
    buckets_empty = int((counts == 0).sum().item())
    max_bucket_load = int(counts.max().item()) if counts.numel() else 0
    mean_bucket_load = float(counts.float().mean().item()) if counts.numel() else 0.0
    p95_bucket_load = float(torch.quantile(counts.float(), 0.95).item()) if counts.numel() else 0.0

    # Fraction of writes that land in a bucket that has >1 writes.
    # (Count each write equally, not each bucket equally.)
    collision_writes = counts[counts > 1].sum().item() - (counts > 1).sum().item()
    collision_write_frac = float(collision_writes / max(1, writes_total))

    # Messages: [B, N, D]
    #
    # A probe needs a global signal, otherwise the global mean is ~0 and cosine
    # similarity becomes meaningless. We inject a shared component that all
    # nodes should be able to recover via the bus.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    shared = signal_alpha * torch.randn((batch_size, 1, d_msg), generator=g, dtype=torch.float32)
    noise = noise_std * torch.randn((batch_size, n_nodes, d_msg), generator=g, dtype=torch.float32)
    msgs = shared.expand(batch_size, n_nodes, d_msg) + noise
    msgs = msgs.to(device=device)

    # Build bus: [B, M, D]
    bus = torch.zeros((batch_size, m_buckets, d_msg), dtype=msgs.dtype, device=msgs.device)

    # Scatter-add writes (k_write buckets per node).
    # Indices for scatter_add must match src shape.
    for j in range(k_write):
        idx_j = write_idx[:, j].to(device=msgs.device)
        sign_j = write_sign[:, j].to(device=msgs.device, dtype=msgs.dtype)
        src = msgs * sign_j.view(1, n_nodes, 1)
        # Broadcast indices across batch and channel dims.
        ind = idx_j.view(1, n_nodes, 1).expand(batch_size, n_nodes, d_msg)
        bus.scatter_add_(1, ind, src)

    # Normalize so bus magnitude doesn't scale with k_write.
    bus = bus / float(k_write)

    # Each node reads k_read buckets (content-addressed context).
    ctx = torch.zeros((batch_size, n_nodes, d_msg), dtype=msgs.dtype, device=msgs.device)
    for j in range(k_read):
        idx_j = read_idx[:, j].to(device=msgs.device)
        sign_j = read_sign[:, j].to(device=msgs.device, dtype=msgs.dtype)
        ind = idx_j.view(1, n_nodes, 1).expand(batch_size, n_nodes, d_msg)
        gathered = bus.gather(1, ind) * sign_j.view(1, n_nodes, 1)
        ctx = ctx + gathered
    ctx = ctx / float(k_read)

    # Compare ctx to global mean message (a simple "is the bus giving global context" proxy).
    global_mean = msgs.mean(dim=1, keepdim=True)  # [B, 1, D]
    # Cosine similarity per node, mean over batch and nodes.
    cos = torch.nn.functional.cosine_similarity(ctx, global_mean.expand_as(ctx), dim=-1)
    ctx_cos_mean = float(cos.mean().item())
    ctx_mse_mean = float(((ctx - global_mean) ** 2).mean().item())
    ctx_var_across_nodes = float(ctx.var(dim=1, unbiased=False).mean().item())

    t1 = time.perf_counter()
    return SketchBusStats(
        n_nodes=n_nodes,
        d_msg=d_msg,
        m_buckets=m_buckets,
        k_write=k_write,
        k_read=k_read,
        seed=seed,
        batch_size=batch_size,
        signal_alpha=float(signal_alpha),
        noise_std=float(noise_std),
        signed=1 if signed else 0,
        writes_total=writes_total,
        buckets_used=buckets_used,
        buckets_empty=buckets_empty,
        max_bucket_load=max_bucket_load,
        mean_bucket_load=mean_bucket_load,
        p95_bucket_load=p95_bucket_load,
        collision_write_frac=collision_write_frac,
        ctx_cos_mean=ctx_cos_mean,
        ctx_mse_mean=ctx_mse_mean,
        ctx_var_across_nodes=ctx_var_across_nodes,
        wall_time_s=float(t1 - t0),
    )


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Probe a hash/sketch bus for swarm-style inclusion.")
    p.add_argument("--n", type=int, default=256, help="Number of nodes (experts).")
    p.add_argument("--d", type=int, default=16, help="Message dimension per node.")
    p.add_argument("--m", type=str, default="256,512,1024", help="Comma list of bucket counts to sweep.")
    p.add_argument("--k-write", type=int, default=2, help="Buckets written per node.")
    p.add_argument("--k-read", type=int, default=2, help="Buckets read per node.")
    p.add_argument("--batch", type=int, default=64, help="Batch size for the probe.")
    p.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    p.add_argument(
        "--signal-alpha",
        type=float,
        default=1.0,
        help="Shared component strength added to every node message (probe needs non-zero global signal).",
    )
    p.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help="Per-node noise standard deviation.",
    )
    p.add_argument(
        "--signed",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, use a sign-hash (+/-) on writes/reads to reduce collision bias (can cancel global mean).",
    )
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Compute device.")
    p.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional path to write JSON results (if empty, only prints).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    m_list = _parse_int_list(args.m)
    if not m_list:
        raise SystemExit("--m must contain at least one integer")

    rows = []
    for m in m_list:
        stats = run_probe(
            n_nodes=args.n,
            d_msg=args.d,
            m_buckets=m,
            k_write=args.k_write,
            k_read=args.k_read,
            batch_size=args.batch,
            seed=args.seed,
            device=args.device,
            signal_alpha=args.signal_alpha,
            noise_std=args.noise_std,
            signed=bool(args.signed),
        )
        rows.append(stats)

    # Print a compact table.
    hdr = (
        "M  used/empty  max  p95  coll_w  cos(ctx,mean)  mse(ctx,mean)  wall_s"
    )
    print(hdr)
    for r in rows:
        print(
            f"{r.m_buckets:<4d}"
            f"{r.buckets_used:>5d}/{r.buckets_empty:<5d} "
            f"{r.max_bucket_load:>4d} "
            f"{r.p95_bucket_load:>4.0f} "
            f"{r.collision_write_frac:>6.3f} "
            f"{r.ctx_cos_mean:>13.4f} "
            f"{r.ctx_mse_mean:>13.6f} "
            f"{r.wall_time_s:>7.3f}"
        )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rows": [asdict(r) for r in rows]}
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
