#!/usr/bin/env python3
"""Frozen SDWPF confirmation of the per-unit role-state adapter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from run_mafs_adapted_screen import (
    ROOT, MAFSAdaptedPower, SmarteoleWindowDataset, atomic_json,
    choose_device, evenly_spaced, evaluate, log, seed_everything, train_stage,
)
from maspge.data import load_frozen_sdwpf, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/sdwpf_role_state_formal_v1.json",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--base-checkpoint-root", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--validation-only", action="store_true",
        help="Train/select on validation without constructing or evaluating holdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not preregistered")
    output_dir = args.output_dir or ROOT / (
        "results/sdwpf_role_state_smoke_v1"
        if args.smoke else "results/sdwpf_role_state_formal_v1"
    )
    checkpoint_dir = args.checkpoint_dir or ROOT / (
        "checkpoints/sdwpf_role_state_smoke_v1"
        if args.smoke else "checkpoints/sdwpf_role_state_formal_v1"
    )
    result_path = output_dir / f"seed_{args.seed}" / "role_state.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed seed={args.seed}")
        return 0
    loading = dict(config["data_loading"])
    stage = dict(config["role_state"])
    if args.smoke:
        loading.update({
            "train_windows": 64, "validation_windows": 32,
            "holdout_windows": 0, "batch_size": 4, "num_workers": 0,
            "log_every_batches": 4,
        })
        stage.update({"epochs": 1, "minimum_epochs": 1, "early_stopping_patience": 1})

    seed_everything(args.seed)
    device = choose_device(args.device)
    arrays = load_frozen_sdwpf(args.data or ROOT / config["dataset"]["path"])
    if str(arrays["dataset_id"].item()) != config["dataset"]["dataset_id"]:
        raise RuntimeError("role-state experiment requires the dense SDWPF v2 contract")
    common = {
        "x_role": arrays["x_role"], "context": arrays["context"],
        "power": arrays["power"], "power_valid": arrays["power_valid"],
        "history_steps": int(config["dataset"]["history_steps"]),
        "horizon_steps": int(config["dataset"]["horizon_steps"]),
    }
    train = SmarteoleWindowDataset(
        **common, end_indices=evenly_spaced(
            arrays["train_end_indices"], int(loading["train_windows"])
        ),
    )
    validation = SmarteoleWindowDataset(
        **common, end_indices=evenly_spaced(
            arrays["validation_end_indices"], int(loading["validation_windows"])
        ),
    )
    holdout = None if (args.smoke or args.validation_only) else SmarteoleWindowDataset(
        **common, end_indices=evenly_spaced(
            arrays["holdout_end_indices"], int(loading["holdout_windows"])
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
        dropout=float(model_config["dropout"]), topology=str(model_config["topology"]),
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
    missing, unexpected = model.load_state_dict(base_checkpoint["model_state"], strict=False)
    allowed = {key for key in missing if key.startswith("role_state_adapter.")}
    if set(missing) != allowed or unexpected:
        raise RuntimeError(f"base checkpoint mismatch missing={missing} unexpected={unexpected}")
    if base_checkpoint.get("stage") != "finetune":
        raise RuntimeError("SDWPF base is not a validation-selected finetune checkpoint")

    target_scale = float(arrays["target_scale"][0])
    base_validation = evaluate(
        model, validation, finetune=True, loading=loading, device=device,
        target_scale=target_scale,
    )
    model.freeze_for_role_state()
    started = time.perf_counter()
    trained, curve = train_stage(
        model, train, validation, stage_name="role_state", finetune=True,
        stage=stage, loading=loading, seed=args.seed + 40000,
        agent_loss_weight=0.0, target_scale=target_scale, device=device,
        checkpoint_path=(
            checkpoint_dir / f"seed_{args.seed}" / "role_state" / "best.pt"
        ),
        use_role_state=True,
        role_state_l2_weight=float(stage["residual_l2_weight"]),
    )
    base_holdout = role_holdout = None
    if holdout is not None:
        role_holdout = evaluate(
            model, holdout, finetune=True, loading=loading, device=device,
            target_scale=target_scale, use_role_state=True,
        )
        model.load_state_dict(base_checkpoint["model_state"], strict=False)
        base_holdout = evaluate(
            model, holdout, finetune=True, loading=loading, device=device,
            target_scale=target_scale, use_role_state=False,
        )
    base_val_mse = base_validation["metrics"]["mse_standardized"]
    role_val_mse = trained["validation"]["metrics"]["mse_standardized"]
    result = {
        "status": "complete",
        "phase": (
            "sdwpf_role_state_smoke" if args.smoke
            else "sdwpf_frozen_role_state_validation_only"
            if args.validation_only
            else "sdwpf_frozen_role_state_formal"
        ),
        "seed": args.seed, "smoke": args.smoke,
        "holdout_evaluated": holdout is not None,
        "training_performed_before_holdout": True,
        "no_sdwpf_hyperparameter_search": True,
        "eligible_as_strict_unseen_test": False,
        "dataset_sha256": str(arrays["sha256"].item()),
        "base_checkpoint_path": str(base_path),
        "base_checkpoint_sha256": sha256_file(base_path),
        "base_validation": base_validation,
        "role_state_validation": trained["validation"],
        "relative_validation_mse_gain": (base_val_mse - role_val_mse) / base_val_mse,
        "base_holdout": base_holdout, "role_state_holdout": role_holdout,
        "best_epoch": trained["best_epoch"], "learning_curve": curve,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda" else 0.0
        ),
    }
    atomic_json(result_path, result)
    log(
        f"DONE seed={args.seed} base_val={base_val_mse:.6f} "
        f"role_val={role_val_mse:.6f} "
        f"holdout={'EVALUATED_FROZEN' if holdout else 'NOT_EVALUATED'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
