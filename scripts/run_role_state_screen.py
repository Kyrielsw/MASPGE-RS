#!/usr/bin/env python3
"""Experiment 040: validation-only per-unit role-state screen."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from run_mafs_adapted_screen import (
    ROOT, MAFSAdaptedPower, SmarteoleWindowDataset, atomic_json,
    choose_device, evenly_spaced, evaluate, load_frozen_smarteole, log,
    seed_everything, train_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/smarteole_role_state_screen_v1.json",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/role_state_v1_screen",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=ROOT / "checkpoints/role_state_v1_screen",
    )
    parser.add_argument("--base-checkpoint-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--ablation-mode",
        choices=["full", "shared_units", "unscaled_history"],
        default="full",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not preregistered")
    result_path = args.output_dir / f"seed_{args.seed}" / "role_state.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed seed={args.seed}")
        return 0
    loading = dict(config["data_loading"])
    stage = dict(config["role_state"])
    if args.smoke:
        loading.update({
            "train_windows": 64, "validation_windows": 32,
            "batch_size": 16, "num_workers": 0,
        })
        stage.update({"epochs": 1, "minimum_epochs": 1, "early_stopping_patience": 1})
    seed_everything(args.seed)
    device = choose_device(args.device)
    arrays = load_frozen_smarteole(
        args.data or ROOT / config["dataset"]["path"],
        expected_sha256=config["dataset"]["sha256"],
    )
    common = {
        "x_role": arrays["x_role"], "context": arrays["context"],
        "power": arrays["power"],
        "history_steps": int(config["dataset"]["history_steps"]),
        "horizon_steps": int(config["dataset"]["horizon_steps"]),
    }
    train = SmarteoleWindowDataset(
        **common,
        end_indices=evenly_spaced(
            arrays["train_end_indices"], int(loading["train_windows"])
        ),
    )
    validation = SmarteoleWindowDataset(
        **common,
        end_indices=evenly_spaced(
            arrays["validation_end_indices"], int(loading["validation_windows"])
        ),
    )
    counts = tuple(
        int(arrays["feature_slot_mask"][0, role].sum()) for role in range(4)
    )
    model_config = config["model"]
    model = MAFSAdaptedPower(
        history_scales=tuple(model_config["history_scales"]),
        horizon_scales=tuple(model_config["horizon_scales"]),
        unit_count=int(arrays["power"].shape[1]),
        full_horizon_steps=int(config["dataset"]["horizon_steps"]),
        d_model=int(model_config["d_model"]), n_heads=int(model_config["n_heads"]),
        layers=int(model_config["layers"]),
        feedforward_size=int(model_config["feedforward_size"]),
        dropout=float(model_config["dropout"]), topology="fully",
        role_feature_counts=counts,
        role_state_hidden_size=int(model_config["role_state_hidden_size"]),
    ).to(device)
    base_root = args.base_checkpoint_root or ROOT / config["base_checkpoint_root"]
    base_path = (
        base_root / f"seed_{args.seed}" / "finetune_best.pt"
    )
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(
        base_checkpoint["model_state"], strict=False
    )
    allowed = {key for key in missing if key.startswith("role_state_adapter.")}
    if set(missing) != allowed or unexpected:
        raise RuntimeError(
            f"base checkpoint mismatch missing={missing} unexpected={unexpected}"
        )
    target_scale = float(arrays["target_scale"][0])
    base_validation = evaluate(
        model, validation, finetune=True, loading=loading, device=device,
        target_scale=target_scale,
    )
    model.freeze_for_role_state()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    started = time.perf_counter()
    conditioned, curve = train_stage(
        model, train, validation, stage_name="role_state", finetune=True,
        stage=stage, loading=loading, seed=args.seed + 30000,
        agent_loss_weight=0.0, target_scale=target_scale, device=device,
        checkpoint_path=args.checkpoint_dir / f"seed_{args.seed}" / "role_state" / "best.pt",
        use_role_state=True,
        role_state_l2_weight=float(stage["residual_l2_weight"]),
        role_state_ablation_mode=args.ablation_mode,
    )
    base_mse = base_validation["metrics"]["mse_standardized"]
    new_mse = conditioned["validation"]["metrics"]["mse_standardized"]
    result = {
        "status": "complete",
        "phase": config.get("phase", "validation_only_role_state_screen"),
        "holdout_evaluated": False, "seed": args.seed, "smoke": args.smoke,
        "mode": f"role_state_{args.ablation_mode}",
        "role_state_ablation_mode": args.ablation_mode,
        "base_checkpoint": str(base_path),
        "trainable_parameters": trainable, "base_validation": base_validation,
        "validation": conditioned["validation"],
        "relative_validation_mse_gain": (base_mse - new_mse) / base_mse,
        "best_epoch": conditioned["best_epoch"], "learning_curve": curve,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    log(
        f"DONE seed={args.seed} base_val={base_mse:.6f} "
        f"role_state_val={new_mse:.6f} "
        f"gain={100*result['relative_validation_mse_gain']:.2f}% "
        "holdout=NOT_EVALUATED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
