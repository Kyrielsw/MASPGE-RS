#!/usr/bin/env python3
"""Validation-only multi-horizon SOFTS and MAFS-to-MASPGE-RS training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from run_external_baseline_screen import run_mode
from run_mafs_adapted_screen import (
    ROOT, MAFSAdaptedPower, SmarteoleWindowDataset, atomic_json,
    choose_device, evenly_spaced, log, seed_everything, train_stage,
)
from maspge.data import (
    horizon_safe_end_indices, load_frozen_sdwpf, load_frozen_smarteole,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/multihorizon_main_v1.json",
    )
    parser.add_argument("--dataset", choices=["smarteole", "sdwpf"], required=True)
    parser.add_argument("--horizon", choices=[5, 30], type=int, required=True)
    parser.add_argument("--method", choices=["softs", "maspge"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_arrays(dataset: str, profile: dict) -> dict:
    path = ROOT / profile["path"]
    if dataset == "sdwpf":
        return load_frozen_sdwpf(path, expected_sha256=profile["sha256"])
    return load_frozen_smarteole(path, expected_sha256=profile["sha256"])


def role_counts(arrays: dict) -> tuple[int, ...]:
    mask = arrays["feature_slot_mask"]
    return tuple(int(mask[0, role].sum()) for role in range(mask.shape[1]))


def make_datasets(arrays: dict, config: dict, profile: dict, horizon: int,
                  *, smoke: bool) -> tuple[dict, SmarteoleWindowDataset,
                                            SmarteoleWindowDataset]:
    loading = dict(profile["data_loading"])
    if smoke:
        loading.update({"train_windows": 64, "validation_windows": 32,
                        "batch_size": 4, "num_workers": 0,
                        "log_every_batches": 4})
    common = {
        "x_role": arrays["x_role"], "context": arrays["context"],
        "power": arrays["power"], "history_steps": int(config["history_steps"]),
        "horizon_steps": horizon,
    }
    if "power_valid" in arrays:
        common["power_valid"] = arrays["power_valid"]
    endpoints = {
        split: horizon_safe_end_indices(
            arrays, split, horizon_steps=horizon,
            reference_horizon_steps=int(config["reference_horizon_steps"]),
        )
        for split in ("train", "validation")
    }
    train = SmarteoleWindowDataset(
        **common, end_indices=evenly_spaced(
            endpoints["train"], int(loading["train_windows"])
        ),
    )
    validation = SmarteoleWindowDataset(
        **common, end_indices=evenly_spaced(
            endpoints["validation"], int(loading["validation_windows"])
        ),
    )
    return loading, train, validation


def run_softs(args: argparse.Namespace, config: dict, profile: dict,
              arrays: dict, loading: dict, train: SmarteoleWindowDataset,
              validation: SmarteoleWindowDataset, device: torch.device) -> None:
    stage = dict(config["softs_training"])
    if args.smoke:
        stage.update({"epochs": 1, "minimum_epochs": 1,
                      "early_stopping_patience": 1})
    stage.update(loading)
    external_config = {
        "dataset": {"history_steps": config["history_steps"],
                    "horizon_steps": args.horizon,
                    "sha256": profile["sha256"]},
        "model": {"softs": {"softs_power_compact": config["model"]["softs"]}},
        "training": stage,
    }
    root = "multihorizon_smoke_v1" if args.smoke else "multihorizon_validation_v1"
    run_mode(
        mode="softs_power_compact", seed=args.seed, config=external_config,
        arrays=arrays, train_dataset=train, validation_dataset=validation,
        phase=f"{args.dataset}_h{args.horizon}_multihorizon_validation_only"
        + ("_smoke" if args.smoke else ""),
        device=device,
        output_dir=ROOT / f"results/{root}/{args.dataset}/h{args.horizon}",
        checkpoint_dir=ROOT / f"checkpoints/{root}/{args.dataset}/h{args.horizon}",
        resume=args.resume,
    )


def run_maspge(args: argparse.Namespace, config: dict, profile: dict,
               arrays: dict, loading: dict, train: SmarteoleWindowDataset,
               validation: SmarteoleWindowDataset, device: torch.device) -> None:
    root = "multihorizon_smoke_v1" if args.smoke else "multihorizon_validation_v1"
    output = ROOT / f"results/{root}/{args.dataset}/h{args.horizon}"
    result_path = output / f"seed_{args.seed}/maspge_rs.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed {args.dataset} h={args.horizon} seed={args.seed} maspge")
        return
    pretrain = dict(profile["pretrain"])
    finetune = dict(profile["finetune"])
    role_state = dict(profile["role_state"])
    if args.smoke:
        for stage in (pretrain, finetune, role_state):
            stage.update({"epochs": 1, "minimum_epochs": 1,
                          "early_stopping_patience": 1})
    model_config = config["model"]
    model = MAFSAdaptedPower(
        history_scales=tuple(model_config["history_scales"]),
        horizon_scales=tuple(config["horizon_scales"][str(args.horizon)]),
        unit_count=int(arrays["power"].shape[1]),
        full_horizon_steps=args.horizon,
        d_model=int(model_config["d_model"]), n_heads=int(model_config["n_heads"]),
        layers=int(model_config["layers"]),
        feedforward_size=int(model_config["feedforward_size"]),
        dropout=float(model_config["dropout"]), topology=str(model_config["topology"]),
        role_feature_counts=role_counts(arrays),
        role_state_hidden_size=int(model_config["role_state_hidden_size"]),
    ).to(device)
    checkpoint = ROOT / (
        f"checkpoints/{root}/{args.dataset}/h{args.horizon}/"
        f"seed_{args.seed}/mafs_maspge_rs"
    )
    metric_scale = float(arrays["target_scale"][0])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
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
        checkpoint_path=checkpoint / "role_state_best.pt", use_role_state=True,
        role_state_l2_weight=float(role_state["residual_l2_weight"]),
    )
    base_mse = base_finetune["validation"]["metrics"]["mse_standardized"]
    role_mse = role_result["validation"]["metrics"]["mse_standardized"]
    payload = {
        "status": "complete", "phase": (
            f"{args.dataset}_h{args.horizon}_multihorizon_validation_only"
            + ("_smoke" if args.smoke else "")
        ),
        "dataset": args.dataset, "horizon_steps": args.horizon,
        "forecast_minutes": args.horizon * int(profile["sampling_minutes"]),
        "seed": args.seed, "holdout_constructed": False,
        "holdout_evaluated": False, "dataset_sha256": profile["sha256"],
        "window_counts": {"train": len(train), "validation": len(validation)},
        "mafs_base": {"pretrain": {**base_pretrain, "learning_curve": pretrain_curve},
                      "finetune": {**base_finetune, "learning_curve": finetune_curve}},
        "maspge_rs": {**role_result, "learning_curve": role_curve},
        "relative_validation_mse_gain": (base_mse - role_mse) / base_mse,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda" else 0.0
        ),
    }
    atomic_json(result_path, payload)
    log(f"DONE {args.dataset} h={args.horizon} seed={args.seed} "
        f"base_val={base_mse:.6f} rs_val={role_mse:.6f} "
        f"gain={100*payload['relative_validation_mse_gain']:.2f}% "
        "holdout=NOT_CONSTRUCTED")


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"] or args.horizon not in config["new_horizons"]:
        raise ValueError("seed or horizon is not preregistered")
    profile = config["datasets"][args.dataset]
    seed_everything(args.seed)
    device = choose_device(args.device)
    arrays = load_arrays(args.dataset, profile)
    loading, train, validation = make_datasets(
        arrays, config, profile, args.horizon, smoke=args.smoke
    )
    log(f"MULTIHORIZON dataset={args.dataset} horizon={args.horizon} "
        f"method={args.method} seed={args.seed} train={len(train)} "
        f"validation={len(validation)} holdout=NOT_CONSTRUCTED")
    if args.method == "softs":
        run_softs(args, config, profile, arrays, loading, train, validation, device)
    else:
        run_maspge(args, config, profile, arrays, loading, train, validation, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
