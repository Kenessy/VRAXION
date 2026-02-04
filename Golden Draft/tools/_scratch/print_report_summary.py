"""Print a concise summary from a VRAXION run_root/report.json (ASCII-only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _get(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=str, required=True)
    args = p.parse_args()

    run_root = Path(args.run_root)
    report_path = run_root / "report.json"
    if not report_path.exists():
        raise SystemExit(f"missing report.json: {report_path}")

    r = json.loads(report_path.read_text(encoding="utf-8"))

    settings = r.get("settings") or {}
    prism = r.get("prismion_bank") or {}
    train = r.get("train") or {}
    ev = r.get("eval") or {}

    val_range = int(settings.get("val_range") or 0)
    chance = (1.0 / float(val_range)) if val_range else None
    eval_acc = ev.get("eval_acc")
    acc_delta = (float(eval_acc) - float(chance)) if (chance is not None and eval_acc is not None) else None

    print(f"run_root={run_root}")
    print(f"model={settings.get('model')} device={settings.get('device')} seed={settings.get('seed')}")
    print(
        "N={n} out_dim={od} mode={mode} shared={sh} ring_len={rl} slot_dim={sd} seq_len={sl}".format(
            n=prism.get("n"),
            od=prism.get("out_dim"),
            mode=prism.get("mode"),
            sh=prism.get("shared"),
            rl=settings.get("ring_len"),
            sd=settings.get("slot_dim"),
            sl=settings.get("seq_len"),
        )
    )
    print(f"steps={train.get('steps')} loss_slope={train.get('loss_slope')}")
    print(
        "eval_acc={acc} eval_loss={loss} eval_n={n} val_range={vr} chance={ch} acc_delta={ad}".format(
            acc=ev.get("eval_acc"),
            loss=ev.get("eval_loss"),
            n=ev.get("eval_n"),
            vr=val_range,
            ch=chance,
            ad=acc_delta,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

