#!/usr/bin/env python3
"""Read-only holdout evaluation for frozen multi-horizon checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from run_external_baseline_screen import (
    build_external_baseline, evaluate as evaluate_external,
)
from run_mafs_adapted_screen import (
    ROOT, MAFSAdaptedPower, SmarteoleWindowDataset, atomic_json,
    choose_device, evaluate as evaluate_maspge, evenly_spaced,
)
from run_multihorizon_validation import load_arrays, role_counts
from maspge.data import horizon_safe_end_indices, sha256_file


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
    return parser.parse_args()


def make_holdout(arrays: dict, config: dict, profile: dict,
                 horizon: int) -> SmarteoleWindowDataset:
    indices = horizon_safe_end_indices(
        arrays, "holdout", horizon_steps=horizon,
        reference_horizon_steps=int(config["reference_horizon_steps"]),
    )
    indices = evenly_spaced(indices, int(profile["data_loading"]["holdout_windows"]))
    kwargs = {
        "x_role": arrays["x_role"], "context": arrays["context"],
        "power": arrays["power"], "end_indices": indices,
        "history_steps": int(config["history_steps"]), "horizon_steps": horizon,
    }
    if "power_valid" in arrays:
        kwargs["power_valid"] = arrays["power_valid"]
    return SmarteoleWindowDataset(**kwargs)


def build_maspge(config: dict, arrays: dict, horizon: int) -> MAFSAdaptedPower:
    model = config["model"]
    return MAFSAdaptedPower(
        history_scales=tuple(model["history_scales"]),
        horizon_scales=tuple(config["horizon_scales"][str(horizon)]),
        unit_count=int(arrays["power"].shape[1]), full_horizon_steps=horizon,
        d_model=int(model["d_model"]), n_heads=int(model["n_heads"]),
        layers=int(model["layers"]),
        feedforward_size=int(model["feedforward_size"]),
        dropout=float(model["dropout"]), topology=str(model["topology"]),
        role_feature_counts=role_counts(arrays),
        role_state_hidden_size=int(model["role_state_hidden_size"]),
    )


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"] or args.horizon not in config["new_horizons"]:
        raise ValueError("seed or horizon is not preregistered")
    profile = config["datasets"][args.dataset]
    validation_root = ROOT / f"results/multihorizon_validation_v1/{args.dataset}/h{args.horizon}"
    checkpoint_root = ROOT / f"checkpoints/multihorizon_validation_v1/{args.dataset}/h{args.horizon}"
    output_root = ROOT / f"results/multihorizon_holdout_v1/{args.dataset}/h{args.horizon}"
    mode = "softs_power_compact" if args.method == "softs" else "maspge_rs"
    result_path = output_root / f"seed_{args.seed}/{mode}.json"
    if args.resume and result_path.is_file():
        print(f"SKIP {args.dataset} h={args.horizon} seed={args.seed} {mode}")
        return 0
    validation_path = validation_root / f"seed_{args.seed}/{mode}.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("holdout_evaluated") is not False:
        raise RuntimeError("validation source accessed holdout")
    arrays = load_arrays(args.dataset, profile)
    holdout = make_holdout(arrays, config, profile, args.horizon)
    device = choose_device(args.device)
    loading = dict(profile["data_loading"])
    started = time.perf_counter()
    if args.method == "softs":
        external_config = {
            "dataset": {"history_steps": config["history_steps"]},
            "model": {"softs": {"softs_power_compact": config["model"]["softs"]}},
        }
        active = arrays["active_role_mask"]
        active_roles = tuple(
            bool(active[role]) if active.ndim == 1 else bool(active[:, role].all())
            for role in range(active.shape[-1])
        )
        model = build_external_baseline(
            mode, config=external_config, role_feature_counts=role_counts(arrays),
            active_roles=active_roles,
            context_size=int(arrays["context"].shape[1]),
        ).to(device)
        checkpoint_path = checkpoint_root / f"seed_{args.seed}/{mode}/best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        holdout_result = evaluate_external(
            model, holdout, training=loading, device=device,
            target_scale=float(arrays["target_scale"][0]),
        )
        payload = {
            "status": "complete", "phase": "frozen_multihorizon_holdout",
            "dataset": args.dataset, "horizon_steps": args.horizon,
            "forecast_minutes": args.horizon * int(profile["sampling_minutes"]),
            "seed": args.seed, "mode": mode, "training_performed": False,
            "holdout_evaluated": True, "post_holdout_tuning_permitted": False,
            "dataset_sha256": profile["sha256"], "validation": validation["validation"],
            "holdout": holdout_result, "checkpoint_sha256": sha256_file(checkpoint_path),
            "wall_seconds": time.perf_counter() - started,
        }
    else:
        model = build_maspge(config, arrays, args.horizon).to(device)
        checkpoint_dir = checkpoint_root / f"seed_{args.seed}/mafs_maspge_rs"
        base_path = checkpoint_dir / "finetune_best.pt"
        role_path = checkpoint_dir / "role_state_best.pt"
        base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
        model.load_state_dict(base_checkpoint["model_state"], strict=True)
        base_result = evaluate_maspge(
            model, holdout, finetune=True, loading=loading, device=device,
            target_scale=float(arrays["target_scale"][0]),
        )
        role_checkpoint = torch.load(role_path, map_location="cpu", weights_only=True)
        model.load_state_dict(role_checkpoint["model_state"], strict=True)
        role_result = evaluate_maspge(
            model, holdout, finetune=True, loading=loading, device=device,
            target_scale=float(arrays["target_scale"][0]), use_role_state=True,
        )
        payload = {
            "status": "complete", "phase": "frozen_multihorizon_holdout",
            "dataset": args.dataset, "horizon_steps": args.horizon,
            "forecast_minutes": args.horizon * int(profile["sampling_minutes"]),
            "seed": args.seed, "mode": mode, "training_performed": False,
            "holdout_evaluated": True, "post_holdout_tuning_permitted": False,
            "dataset_sha256": profile["sha256"],
            "validation": {
                "mafs_base": validation["mafs_base"]["finetune"]["validation"],
                "maspge_rs": validation["maspge_rs"]["validation"],
            },
            "holdout": {"mafs_base": base_result, "maspge_rs": role_result},
            "checkpoint_sha256": {
                "mafs_base": sha256_file(base_path), "maspge_rs": sha256_file(role_path),
            },
            "wall_seconds": time.perf_counter() - started,
        }
    atomic_json(result_path, payload)
    print(f"DONE {args.dataset} h={args.horizon} seed={args.seed} mode={mode} "
          "training=FORBIDDEN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
