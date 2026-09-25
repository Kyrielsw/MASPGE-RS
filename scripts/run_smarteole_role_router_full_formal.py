#!/usr/bin/env python3
"""Retrain frozen MAFS/MASPGE on every SMARTEOLE training window."""

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
        default=ROOT / "configs/smarteole_role_router_full_formal_v1.json",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/unified_formal_v1/role_router",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=ROOT / "checkpoints/unified_formal_v1/role_router",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not preregistered")
    result_path = args.output_dir / f"seed_{args.seed}" / "mafs_maspge_full.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed seed={args.seed}")
        return 0

    loading = dict(config["data_loading"])
    pretrain = dict(config["pretrain"])
    finetune = dict(config["finetune"])
    router = dict(config["role_router"])
    if args.smoke:
        loading.update({
            "train_windows": 64, "validation_windows": 32,
            "holdout_windows": 32, "batch_size": 16, "num_workers": 0,
        })
        for stage in (pretrain, finetune, router):
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
    datasets = {
        name: SmarteoleWindowDataset(
            **common,
            end_indices=evenly_spaced(
                arrays[f"{name}_end_indices"], int(loading[f"{name}_windows"])
            ),
        )
        for name in ("train", "validation", "holdout")
    }
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
        role_feature_counts=counts, context_size=int(arrays["context"].shape[1]),
        router_hidden_size=int(model_config["router_hidden_size"]),
    ).to(device)
    checkpoint = args.checkpoint_dir / f"seed_{args.seed}" / "mafs_maspge_full"
    target_scale = float(arrays["target_scale"][0])
    parameter_count = sum(p.numel() for p in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    log(
        f"UNIFIED FORMAL seed={args.seed} train={len(datasets['train'])} "
        f"validation={len(datasets['validation'])} holdout={len(datasets['holdout'])}"
    )
    base_pretrain, pretrain_curve = train_stage(
        model, datasets["train"], datasets["validation"],
        stage_name="pretrain", finetune=False, stage=pretrain, loading=loading,
        seed=args.seed, agent_loss_weight=float(model_config["agent_loss_weight"]),
        target_scale=target_scale, device=device,
        checkpoint_path=checkpoint / "pretrain_best.pt",
    )
    model.freeze_specialist_agents()
    base_finetune, finetune_curve = train_stage(
        model, datasets["train"], datasets["validation"],
        stage_name="finetune", finetune=True, stage=finetune, loading=loading,
        seed=args.seed + 10000,
        agent_loss_weight=float(model_config["agent_loss_weight"]),
        target_scale=target_scale, device=device,
        checkpoint_path=checkpoint / "finetune_best.pt",
    )
    base_holdout = evaluate(
        model, datasets["holdout"], finetune=True, loading=loading,
        device=device, target_scale=target_scale,
    )
    model.freeze_for_role_routing()
    routed, router_curve = train_stage(
        model, datasets["train"], datasets["validation"],
        stage_name="role_router", finetune=True, stage=router, loading=loading,
        seed=args.seed + 20000, agent_loss_weight=0.0,
        target_scale=target_scale, device=device,
        checkpoint_path=checkpoint / "role_router_best.pt", use_role_router=True,
        route_logit_l2_weight=float(router["route_logit_l2_weight"]),
        router_input_mode="full",
    )
    routed_holdout = evaluate(
        model, datasets["holdout"], finetune=True, loading=loading,
        device=device, target_scale=target_scale, use_role_router=True,
        router_input_mode="full",
    )
    result = {
        "status": "complete", "phase": "unified_full_training_formal",
        "smoke": args.smoke, "holdout_evaluated": True, "seed": args.seed,
        "mode": "mafs_maspge_full", "parameter_count": parameter_count,
        "base": {
            "validation": base_finetune["validation"], "holdout": base_holdout,
            "pretrain": {**base_pretrain, "learning_curve": pretrain_curve},
            "finetune": {**base_finetune, "learning_curve": finetune_curve},
        },
        "full_role_router": {
            "validation": routed["validation"], "holdout": routed_holdout,
            "best_epoch": routed["best_epoch"], "learning_curve": router_curve,
        },
        "train_windows": len(datasets["train"]),
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda" else 0.0
        ),
    }
    atomic_json(result_path, result)
    log(
        f"DONE seed={args.seed} base_holdout="
        f"{base_holdout['metrics']['mse_standardized']:.6f} routed_holdout="
        f"{routed_holdout['metrics']['mse_standardized']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
