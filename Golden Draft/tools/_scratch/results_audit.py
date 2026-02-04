"""Audit bench_vault/results/results_master.csv for misleading / low-confidence rows.

This exists to avoid "smoke" runs (tiny steps or tiny eval_n) showing up as
"best" in dashboards/frontiers and wasting time.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
MASTER = REPO_ROOT / "bench_vault" / "results" / "results_master.csv"


def _ci95(p: float, n: int) -> float:
    # Binomial normal approx is fine for quick gating.
    if n <= 0:
        return float("nan")
    p = max(0.0, min(1.0, float(p)))
    se = math.sqrt(max(0.0, p * (1.0 - p) / float(n)))
    return 1.96 * se


def main() -> int:
    if not MASTER.exists():
        raise SystemExit(f"missing: {MASTER}")

    df = pd.read_csv(MASTER)
    print(f"[audit] master={MASTER} rows={len(df)} cols={len(df.columns)}")

    # Basic missingness
    for col in ("eval_acc", "eval_n", "steps", "val_range", "seq_len", "run_root"):
        if col not in df.columns:
            print(f"[audit] WARNING missing column: {col}")

    # Compute chance + CI + delta where possible.
    df["val_range_num"] = pd.to_numeric(df.get("val_range"), errors="coerce")
    df["chance_acc"] = 1.0 / df["val_range_num"]
    df["eval_acc_num"] = pd.to_numeric(df.get("eval_acc"), errors="coerce")
    df["eval_n_num"] = pd.to_numeric(df.get("eval_n"), errors="coerce")
    df["steps_num"] = pd.to_numeric(df.get("steps"), errors="coerce")
    df["acc_delta"] = df["eval_acc_num"] - df["chance_acc"]
    df["ci95"] = [
        _ci95(p, int(n)) if pd.notna(p) and pd.notna(n) else float("nan")
        for p, n in zip(df["eval_acc_num"].tolist(), df["eval_n_num"].tolist())
    ]

    # Quick flags for misleading points.
    df["flag_tiny_steps"] = df["steps_num"].fillna(-1) < 10
    df["flag_tiny_eval"] = df["eval_n_num"].fillna(-1) < 256
    df["flag_missing_steps"] = df["steps_num"].isna()
    df["flag_missing_eval"] = df["eval_acc_num"].isna() | df["eval_n_num"].isna()

    def _cnt(name: str, mask: pd.Series) -> None:
        print(f"[audit] {name} = {int(mask.sum())}")

    _cnt("missing_eval", df["flag_missing_eval"])
    _cnt("missing_steps", df["flag_missing_steps"])
    _cnt("tiny_steps(<10)", df["flag_tiny_steps"])
    _cnt("tiny_eval_n(<256)", df["flag_tiny_eval"])

    # Show top eval_acc rows (this is where misleading smoke runs surface).
    view = df.dropna(subset=["eval_acc_num"]).copy().sort_values("eval_acc_num", ascending=False).head(20)
    cols = [
        "eval_acc_num",
        "ci95",
        "eval_n_num",
        "chance_acc",
        "acc_delta",
        "steps_num",
        "seq_len",
        "val_range",
        "prismn_n",
        "prismn_out_dim",
        "run_root",
    ]
    cols = [c for c in cols if c in view.columns]
    print("\n[audit] TOP eval_acc (watch for tiny steps/eval_n):")
    print(view[cols].to_string(index=False))

    # List the most misleading: high eval_acc but tiny eval_n or tiny steps.
    mis = df[
        (df["eval_acc_num"].notna())
        & (df["eval_acc_num"] >= 0.10)
        & (df["flag_tiny_steps"] | df["flag_tiny_eval"])
    ].copy()
    if len(mis) > 0:
        mis = mis.sort_values(["eval_acc_num", "eval_n_num"], ascending=[False, True]).head(25)
        cols2 = ["eval_acc_num", "ci95", "eval_n_num", "steps_num", "seq_len", "run_root"]
        cols2 = [c for c in cols2 if c in mis.columns]
        print("\n[audit] MISLEADING high-acc smoke rows (should be filtered from leaderboards):")
        print(mis[cols2].to_string(index=False))
    else:
        print("\n[audit] No high-acc smoke rows detected (good).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

