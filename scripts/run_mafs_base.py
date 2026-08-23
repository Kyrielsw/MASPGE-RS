#!/usr/bin/env python3
"""Train validation-selected adapted-MAFS base checkpoints without holdout."""

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
from maspge.data import load_frozen_sdwpf, load_frozen_smarteole


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["smarteole", "sdwpf"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mafs_base_v1.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not frozen")
    dataset_config = dict(config["datasets"][args.dataset])
    loading = {
        key: dataset_config[key] for key in (
            "train_windows", "validation_windows", "batch_size",
            "num_workers", "log_every_batches",
        )
    }
    pretrain = dict(dataset_config["pretrain"])
    finetune = dict(dataset_config["finetune"])
    if args.smoke:
        loading.update({
            "train_windows": 64, "validation_windows": 32,
            "batch_size": 8, "num_workers": 0, "log_every_batches": 4,
        })
        for stage in (pretrain, finetune):
            stage.update({
                "epochs": 1, "minimum_epochs": 1,
                "early_stopping_patience": 1,
            })
    output_root = ROOT / (
        "results/mafs_base_smoke" if args.smoke else "results/mafs_base_v1"
    ) / args.dataset
    checkpoint_root = ROOT / (
        "checkpoints/mafs_base_smoke" if args.smoke else "checkpoints/mafs_base_v1"
    ) / args.dataset / f"seed_{args.seed}"
    result_path = output_root / f"seed_{args.seed}.json"
    if args.resume and result_path.is_file():
        log(f"SKIP dataset={args.dataset} seed={args.seed}")
        return 0

    seed_everything(args.seed)
    device = choose_device(args.device)
    path = ROOT / dataset_config["path"]
    if args.dataset == "smarteole":
        arrays = load_frozen_smarteole(
            path, expected_sha256=dataset_config["sha256"]
        )
        power_valid = None
        target_scale = float(arrays["target_scale"][0])
    else:
        arrays = load_frozen_sdwpf(
            path, expected_sha256=dataset_config["sha256"]
        )
        power_valid = arrays["power_valid"]
        target_scale = 1.0
    common = {
        "x_role": arrays["x_role"], "context": arrays["context"],
        "power": arrays["power"], "power_valid": power_valid,
        "history_steps": int(dataset_config["history_steps"]),
        "horizon_steps": int(dataset_config["horizon_steps"]),
    }
    train = SmarteoleWindowDataset(
        **common, end_indices=evenly_spaced(
            arrays["train_end_indices"], int(loading["train_windows"])
        )
    )
    validation = SmarteoleWindowDataset(
        **common, end_indices=evenly_spaced(
            arrays["validation_end_indices"], int(loading["validation_windows"])
        )
    )
    model_config = config["model"]
    model = MAFSAdaptedPower(
        history_scales=tuple(model_config["history_scales"]),
        horizon_scales=tuple(model_config["horizon_scales"]),
        unit_count=int(arrays["power"].shape[1]),
        full_horizon_steps=int(dataset_config["horizon_steps"]),
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        layers=int(model_config["layers"]),
        feedforward_size=int(model_config["feedforward_size"]),
        dropout=float(model_config["dropout"]),
        topology=str(model_config["topology"]),
    ).to(device)
    started = time.perf_counter()
    pretrain_result, pretrain_curve = train_stage(
        model, train, validation, stage_name="pretrain", finetune=False,
        stage=pretrain, loading=loading, seed=args.seed,
        agent_loss_weight=float(model_config["agent_loss_weight"]),
        target_scale=target_scale, device=device,
        checkpoint_path=checkpoint_root / "pretrain_best.pt",
    )
    model.freeze_specialist_agents()
    finetune_result, finetune_curve = train_stage(
        model, train, validation, stage_name="finetune", finetune=True,
        stage=finetune, loading=loading, seed=args.seed + 10000,
        agent_loss_weight=float(model_config["agent_loss_weight"]),
        target_scale=target_scale, device=device,
        checkpoint_path=checkpoint_root / "finetune_best.pt",
    )
    atomic_json(result_path, {
        "status": "complete", "phase": "validation_only_mafs_base",
        "dataset": args.dataset, "seed": args.seed, "smoke": args.smoke,
        "holdout_evaluated": False,
        "pretrain": {**pretrain_result, "learning_curve": pretrain_curve},
        "finetune": {**finetune_result, "learning_curve": finetune_curve},
        "wall_seconds": time.perf_counter() - started,
    })
    log(
        f"DONE dataset={args.dataset} seed={args.seed} "
        f"validation_mse={finetune_result['validation']['metrics']['mse_standardized']:.6f} "
        "holdout=NOT_EVALUATED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
