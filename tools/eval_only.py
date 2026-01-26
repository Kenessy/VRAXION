import argparse
import os
import sys

import torch


def _resolve_checkpoint(path: str) -> str:
    if not path:
        return ""
    return os.path.abspath(path)


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    parser = argparse.ArgumentParser(description="Run eval-only pass for VRAXION checkpoints.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file or modular directory")
    parser.add_argument("--dataset", default="synth", help="Dataset label for logging")
    parser.add_argument("--model-name", default="absolute_hallway", help="Model name for logging")
    args = parser.parse_args()

    ckpt_path = _resolve_checkpoint(args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"[eval_only] checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    import tournament_phase6 as tp6

    modular_dir = tp6._resolve_modular_resume_dir(ckpt_path)
    if modular_dir:
        router_state = torch.load(os.path.join(modular_dir, "system", "router.state"), map_location="cpu")
        num_experts = router_state.get("num_experts")
        if num_experts:
            tp6.EXPERT_HEADS = int(num_experts)

    tp6.set_seed(tp6.SEED)
    loader, num_classes, collate = tp6.get_seq_mnist_loader()
    eval_loader, eval_size = tp6.build_eval_loader_from_subset(
        loader.dataset, input_collate=collate
    )
    tp6.log_eval_overlap(loader.dataset, eval_loader.dataset, eval_size, "eval_only_subset")

    model = tp6.AbsoluteHallway(
        input_dim=1,
        num_classes=num_classes,
        ring_len=tp6.RING_LEN,
        slot_dim=tp6.SLOT_DIM,
    )
    if modular_dir:
        ckpt = tp6._load_modular_checkpoint(model, optimizer=None, scaler=None, base_dir=modular_dir)
    else:
        ckpt = torch.load(ckpt_path, map_location=tp6.DEVICE)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[eval_only] missing keys: {missing}")
            print(f"[eval_only] unexpected keys: {unexpected}")

    if "update_scale" in ckpt:
        model.update_scale = float(ckpt["update_scale"])
    if "ptr_inertia" in ckpt:
        model.ptr_inertia = float(ckpt["ptr_inertia"])
    if "ptr_inertia_ema" in ckpt:
        model.ptr_inertia_ema = float(ckpt["ptr_inertia_ema"])
    if "ptr_inertia_floor" in ckpt:
        model.ptr_inertia_floor = float(ckpt["ptr_inertia_floor"])
    if "agc_scale_max" in ckpt:
        model.agc_scale_max = float(ckpt["agc_scale_max"])
    if "ground_speed_ema" in ckpt:
        model.ground_speed_ema = ckpt["ground_speed_ema"]
    if "ground_speed_limit" in ckpt:
        model.ground_speed_limit = ckpt["ground_speed_limit"]

    env_scale_init = os.environ.get("TP6_SCALE_INIT")
    env_scale_max = os.environ.get("TP6_SCALE_MAX")
    if env_scale_init is not None:
        model.update_scale = float(env_scale_init)
    if env_scale_max is not None:
        model.agc_scale_max = float(env_scale_max)
    env_inertia = os.environ.get("TP6_PTR_INERTIA_OVERRIDE")
    if env_inertia is not None:
        model.ptr_inertia = float(env_inertia)
        model.ptr_inertia_ema = model.ptr_inertia
    model.agc_scale_cap = model.agc_scale_max
    model.ground_speed_ema = None
    model.ground_speed_limit = None
    model.ground_speed = None
    model.debug_scale_out = model.update_scale

    model.to(tp6.DEVICE)
    model.eval()
    with torch.no_grad():
        stats = tp6.eval_model(model, eval_loader, args.dataset, args.model_name)
    print(f"[eval_only] stats: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
