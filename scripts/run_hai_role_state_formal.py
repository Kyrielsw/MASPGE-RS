#!/usr/bin/env python3
"""Train MAFS Base and MASPGE-RS on HAI without constructing holdout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from run_mafs_adapted_screen import (
    ROOT, MAFSAdaptedPower, SmarteoleWindowDataset, atomic_json,
    choose_device, evenly_spaced, log, seed_everything, train_stage,
)
from maspge.data import load_frozen_hai


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/hai_role_state_formal_v1.json",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--ablation-mode",
        choices=["full", "without_a4"],
        default="full",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not preregistered")
    output_dir = args.output_dir or ROOT / (
        "results/hai_role_state_smoke_v1"
        if args.smoke else "results/hai_role_state_formal_v1"
    )
    checkpoint_dir = args.checkpoint_dir or ROOT / (
        "checkpoints/hai_role_state_smoke_v1"
        if args.smoke else "checkpoints/hai_role_state_formal_v1"
    )
    result_path = output_dir / f"seed_{args.seed}" / "role_state.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed seed={args.seed}")
        return 0
    loading = dict(config["data_loading"])
    pretrain = dict(config["pretrain"])
    finetune = dict(config["finetune"])
    role_state = dict(config["role_state"])
    if args.smoke:
        loading.update({
            "train_windows": 64, "validation_windows": 32,
            "batch_size": 8, "num_workers": 0, "log_every_batches": 4,
        })
        for stage in (pretrain, finetune, role_state):
            stage.update({
                "epochs": 1, "minimum_epochs": 1,
                "early_stopping_patience": 1,
            })

    seed_everything(args.seed)
    device = choose_device(args.device)
    arrays = load_frozen_hai(
        args.data or ROOT / config["dataset"]["path"],
        expected_sha256=config["dataset"]["sha256"],
    )
    if str(arrays["dataset_id"].item()) != config["dataset"]["dataset_id"]:
        raise RuntimeError("HAI dataset identity mismatch")
    common = {
        "x_role": arrays["x_role"], "context": arrays["context"],
        "power": arrays["power"], "power_valid": arrays["power_valid"],
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
    model_config = config["model"]
    role_counts = tuple(int(value) for value in arrays["role_feature_counts"])
    model = MAFSAdaptedPower(
        history_scales=tuple(model_config["history_scales"]),
        horizon_scales=tuple(model_config["horizon_scales"]),
        unit_count=int(arrays["power"].shape[1]),
        full_horizon_steps=int(config["dataset"]["horizon_steps"]),
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        layers=int(model_config["layers"]),
        feedforward_size=int(model_config["feedforward_size"]),
        dropout=float(model_config["dropout"]),
        topology=str(model_config["topology"]),
        role_feature_counts=role_counts,
        context_size=int(arrays["context"].shape[1]),
        router_hidden_size=int(model_config["router_hidden_size"]),
        role_state_hidden_size=int(model_config["role_state_hidden_size"]),
    ).to(device)
    checkpoint = checkpoint_dir / f"seed_{args.seed}" / "mafs_maspge_rs"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    log(
        f"HAI seed={args.seed} units=2 train={len(train)} "
        f"validation={len(validation)} holdout=NOT_CONSTRUCTED"
    )
    # HAI combines targets with different physical scales.  Selection and all
    # joint comparisons therefore use standardized metrics; per-unit physical
    # metrics are computed only in the later reporting stage.
    metric_scale = 1.0
    base_pretrain, pretrain_curve = train_stage(
        model, train, validation, stage_name="pretrain", finetune=False,
        stage=pretrain, loading=loading, seed=args.seed,
        agent_loss_weight=float(model_config["agent_loss_weight"]),
        target_scale=metric_scale, device=device,
        checkpoint_path=checkpoint / "pretrain_best.pt",
    )
    model.freeze_specialist_agents()
    base_finetune, finetune_curve = train_stage(
        model, train, validation, stage_name="finetune", finetune=True,
        stage=finetune, loading=loading, seed=args.seed + 10000,
        agent_loss_weight=float(model_config["agent_loss_weight"]),
        target_scale=metric_scale, device=device,
        checkpoint_path=checkpoint / "finetune_best.pt",
    )
    model.freeze_for_role_state()
    role_result, role_curve = train_stage(
        model, train, validation, stage_name="role_state", finetune=True,
        stage=role_state, loading=loading, seed=args.seed + 40000,
        agent_loss_weight=0.0, target_scale=metric_scale, device=device,
        checkpoint_path=checkpoint / "role_state_best.pt",
        use_role_state=True,
        role_state_l2_weight=float(role_state["residual_l2_weight"]),
        role_state_ablation_mode=args.ablation_mode,
    )
    base_mse = base_finetune["validation"]["metrics"]["mse_standardized"]
    role_mse = role_result["validation"]["metrics"]["mse_standardized"]
    payload = {
        "status": "complete",
        "phase": "hai_role_state_smoke" if args.smoke else "hai_strict_validation_only",
        "seed": args.seed,
        "smoke": args.smoke,
        "holdout_constructed": False,
        "holdout_evaluated": False,
        "dataset_sha256": str(arrays["sha256"].item()),
        "all_four_roles_active": True,
        "role_state_ablation_mode": args.ablation_mode,
        "a4_conditioning_enabled": args.ablation_mode != "without_a4",
        "joint_metric_space": "per-unit training-standardized",
        "joint_physical_metrics_reportable": False,
        "base": {
            "pretrain": {**base_pretrain, "learning_curve": pretrain_curve},
            "finetune": {**base_finetune, "learning_curve": finetune_curve},
        },
        "role_state": {**role_result, "learning_curve": role_curve},
        "relative_validation_mse_gain": (base_mse - role_mse) / base_mse,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda" else 0.0
        ),
    }
    atomic_json(result_path, payload)
    log(
        f"DONE seed={args.seed} base_val={base_mse:.6f} "
        f"role_val={role_mse:.6f} gain={100*payload['relative_validation_mse_gain']:.2f}% "
        "holdout=NOT_CONSTRUCTED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
