#!/usr/bin/env python3
"""Five-seed SDWPF confirmation of the frozen full role router."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from run_mafs_adapted_screen import (
    ROOT,
    MAFSAdaptedPower,
    SmarteoleWindowDataset,
    atomic_json,
    choose_device,
    evenly_spaced,
    evaluate,
    log,
    seed_everything,
    train_stage,
)

from maspge.data import load_frozen_sdwpf
from maspge.metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/sdwpf_role_router_formal_v1.json",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/sdwpf_role_router_formal_v1",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=ROOT / "checkpoints/sdwpf_role_router_formal_v1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def persistence_metrics(dataset: SmarteoleWindowDataset, target_scale: float) -> dict:
    prediction, target, valid = [], [], []
    for item in dataset:
        prediction.append(item["y_current"])
        target.append(item["y_future"])
        valid.append(item["y_future_valid"])
    return regression_metrics(
        torch.stack(prediction), torch.stack(target), target_scale=target_scale,
        valid_mask=torch.stack(valid),
    )


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not preregistered")
    result_path = args.output_dir / f"seed_{args.seed}" / "mafs_role_router_full.json"
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
            "holdout_windows": 0, "batch_size": 16, "num_workers": 0,
        })
        for stage in (pretrain, finetune, router):
            stage.update({"epochs": 1, "minimum_epochs": 1, "early_stopping_patience": 1})

    seed_everything(args.seed)
    device = choose_device(args.device)
    arrays = load_frozen_sdwpf(args.data or ROOT / config["dataset"]["path"])
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
    holdout = None
    if not args.smoke:
        holdout = SmarteoleWindowDataset(
            **common,
            end_indices=evenly_spaced(
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
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        layers=int(model_config["layers"]),
        feedforward_size=int(model_config["feedforward_size"]),
        dropout=float(model_config["dropout"]),
        topology=str(model_config["topology"]),
        role_feature_counts=counts,
        context_size=int(arrays["context"].shape[1]),
        router_hidden_size=int(model_config["router_hidden_size"]),
    ).to(device)
    target_scale = float(arrays["target_scale"][0])
    checkpoint = args.checkpoint_dir / f"seed_{args.seed}" / "mafs_role_router_full"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    log(
        f"SDWPF seed={args.seed} units=134 train={len(train)} "
        f"validation={len(validation)} holdout={'FORBIDDEN_SMOKE' if args.smoke else len(holdout)}"
    )
    base_pretrain, pretrain_curve = train_stage(
        model, train, validation, stage_name="pretrain", finetune=False,
        stage=pretrain, loading=loading, seed=args.seed,
        agent_loss_weight=float(model_config["agent_loss_weight"]),
        target_scale=target_scale, device=device,
        checkpoint_path=checkpoint / "pretrain_best.pt",
    )
    model.freeze_specialist_agents()
    base_finetune, finetune_curve = train_stage(
        model, train, validation, stage_name="finetune", finetune=True,
        stage=finetune, loading=loading, seed=args.seed + 10000,
        agent_loss_weight=float(model_config["agent_loss_weight"]),
        target_scale=target_scale, device=device,
        checkpoint_path=checkpoint / "finetune_best.pt",
    )
    model.freeze_for_role_routing()
    router_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    routed, router_curve = train_stage(
        model, train, validation, stage_name="role_router", finetune=True,
        stage=router, loading=loading, seed=args.seed + 20000,
        agent_loss_weight=0.0, target_scale=target_scale, device=device,
        checkpoint_path=checkpoint / "role_router_best.pt", use_role_router=True,
        route_logit_l2_weight=float(router["route_logit_l2_weight"]),
        router_input_mode="full",
    )
    base_holdout = None
    persistence_holdout = None
    routed_holdout = None
    if holdout is not None:
        # Both validation-selected checkpoints are now frozen.  Holdout is
        # opened only after every trainable stage has finished.
        routed_holdout = evaluate(
            model, holdout, finetune=True, loading=loading, device=device,
            target_scale=target_scale, use_role_router=True,
            router_input_mode="full",
        )
        base_checkpoint = torch.load(
            checkpoint / "finetune_best.pt", map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(base_checkpoint["model_state"], strict=True)
        base_holdout = evaluate(
            model, holdout, finetune=True, loading=loading, device=device,
            target_scale=target_scale,
        )
        persistence_holdout = persistence_metrics(holdout, target_scale)

    result = {
        "status": "complete",
        "phase": "sdwpf_cross_dataset_formal" if holdout is not None else "sdwpf_smoke",
        "smoke": args.smoke,
        "holdout_evaluated": holdout is not None,
        "seed": args.seed,
        "dataset_sha256": str(arrays["sha256"].item()),
        "parameter_count": parameter_count,
        "router_trainable_parameters": router_trainable,
        "persistence_holdout": persistence_holdout,
        "base": {
            "pretrain": {**base_pretrain, "learning_curve": pretrain_curve},
            "finetune": {**base_finetune, "learning_curve": finetune_curve},
            "holdout": base_holdout,
        },
        "full_role_router": {
            **routed, "learning_curve": router_curve, "holdout": routed_holdout,
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda" else 0.0
        ),
    }
    atomic_json(result_path, result)
    base_mse = base_finetune["validation"]["metrics"]["mse_standardized"]
    routed_mse = routed["validation"]["metrics"]["mse_standardized"]
    log(
        f"DONE seed={args.seed} base_val={base_mse:.6f} routed_val={routed_mse:.6f} "
        f"holdout={'EVALUATED_ONCE' if holdout is not None else 'NOT_EVALUATED'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
