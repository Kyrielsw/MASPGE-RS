#!/usr/bin/env python3
"""Validation-only MAFS and Generic Unit State training on XAI4HEAT."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from maspge.data import load_frozen_xai4heat
from maspge.mafs_adapted import MAFSAdaptedPower, PerUnitGenericStateAdapter
from run_mafs_adapted_screen import (
    ROOT, SmarteoleWindowDataset, atomic_json, choose_device, evenly_spaced,
    log, seed_everything, train_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/xai4heat_frozen_v1.json")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/xai4heat_main_validation_v1")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/xai4heat_main_validation_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not preregistered")
    result_path = args.output_dir / f"seed_{args.seed}" / "result.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed seed={args.seed}")
        return 0
    loading = dict(config["data_loading"])
    stages = {name: dict(config[name]) for name in ("pretrain", "finetune", "generic_state")}
    if args.smoke:
        loading.update({"train_windows": 16, "validation_windows": 8, "batch_size": 8,
                        "num_workers": 0, "log_every_batches": 1})
        for stage in stages.values():
            stage.update({"epochs": 1, "minimum_epochs": 1, "early_stopping_patience": 1})

    seed_everything(args.seed)
    device = choose_device(args.device)
    arrays = load_frozen_xai4heat(
        args.data or ROOT / config["dataset"]["path"],
        expected_sha256=config["dataset"]["sha256"],
    )
    common = {
        "x_role": arrays["x_role"], "context": arrays["context"],
        "power": arrays["power"], "power_valid": arrays["power_valid"],
        "history_steps": int(config["dataset"]["history_steps"]),
        "horizon_steps": int(config["dataset"]["horizon_steps"]),
    }
    train = SmarteoleWindowDataset(
        **common,
        end_indices=evenly_spaced(arrays["train_end_indices"], int(loading["train_windows"])),
    )
    validation = SmarteoleWindowDataset(
        **common,
        end_indices=evenly_spaced(arrays["validation_end_indices"], int(loading["validation_windows"])),
    )
    model_config = config["model"]
    model = MAFSAdaptedPower(
        history_scales=tuple(model_config["history_scales"]),
        horizon_scales=tuple(model_config["horizon_scales"]),
        unit_count=5,
        full_horizon_steps=int(config["dataset"]["horizon_steps"]),
        d_model=int(model_config["d_model"]), n_heads=int(model_config["n_heads"]),
        layers=int(model_config["layers"]), feedforward_size=int(model_config["feedforward_size"]),
        dropout=float(model_config["dropout"]), topology=str(model_config["topology"]),
    ).to(device)
    checkpoint = args.checkpoint_dir / f"seed_{args.seed}"
    started = time.perf_counter()
    log(
        f"XAI4HEAT seed={args.seed} units=5 train={len(train)} "
        f"validation={len(validation)} holdout=NOT_CONSTRUCTED"
    )
    base_pretrain, pretrain_curve = train_stage(
        model, train, validation, stage_name="pretrain", finetune=False,
        stage=stages["pretrain"], loading=loading, seed=args.seed,
        agent_loss_weight=float(model_config["agent_loss_weight"]),
        target_scale=1.0, device=device,
        checkpoint_path=checkpoint / "mafs_base" / "pretrain_best.pt",
    )
    model.freeze_specialist_agents()
    base_finetune, finetune_curve = train_stage(
        model, train, validation, stage_name="finetune", finetune=True,
        stage=stages["finetune"], loading=loading, seed=args.seed + 10000,
        agent_loss_weight=float(model_config["agent_loss_weight"]),
        target_scale=1.0, device=device,
        checkpoint_path=checkpoint / "mafs_base" / "finetune_best.pt",
    )
    model.role_state_adapter = PerUnitGenericStateAdapter(
        history_scales=tuple(model_config["history_scales"]),
        role_feature_counts=(7,),
        state_hidden_size=int(model_config["generic_state_hidden_size"]),
        d_model=int(model_config["d_model"]),
    ).to(device)
    model.freeze_for_role_state()
    generic_result, generic_curve = train_stage(
        model, train, validation, stage_name="generic_unit_state", finetune=True,
        stage=stages["generic_state"], loading=loading, seed=args.seed + 40000,
        agent_loss_weight=0.0, target_scale=1.0, device=device,
        checkpoint_path=checkpoint / "generic_unit_state" / "best.pt",
        use_role_state=True,
        role_state_l2_weight=float(stages["generic_state"]["residual_l2_weight"]),
    )
    base_mse = base_finetune["validation"]["metrics"]["mse_standardized"]
    generic_mse = generic_result["validation"]["metrics"]["mse_standardized"]
    payload = {
        "status": "complete",
        "phase": "xai4heat_five_seed_validation_only",
        "seed": args.seed,
        "smoke": args.smoke,
        "holdout_constructed": False,
        "holdout_evaluated": False,
        "dataset_sha256": str(arrays["sha256"].item()),
        "functional_roles_used": False,
        "main_model": "generic_unit_state_precommunication_zero_init_residual",
        "base": {
            "pretrain": {**base_pretrain, "learning_curve": pretrain_curve},
            "finetune": {**base_finetune, "learning_curve": finetune_curve},
        },
        "generic_unit_state": {**generic_result, "learning_curve": generic_curve},
        "relative_validation_mse_gain_vs_mafs": (base_mse - generic_mse) / base_mse,
        "generic_adapter_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        ),
    }
    atomic_json(result_path, payload)
    log(
        f"DONE seed={args.seed} mafs={base_mse:.6f} generic={generic_mse:.6f} "
        f"gain={100*payload['relative_validation_mse_gain_vs_mafs']:.2f}% "
        "holdout=NOT_CONSTRUCTED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
