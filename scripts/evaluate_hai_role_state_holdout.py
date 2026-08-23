#!/usr/bin/env python3
"""Evaluate frozen HAI MAFS Base and MASPGE-RS checkpoints exactly once."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_mafs_adapted_screen import (
    ROOT, MAFSAdaptedPower, SmarteoleWindowDataset, atomic_json,
    choose_device, evenly_spaced, evaluate, log, seed_everything,
)
from maspge.data import load_frozen_hai, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/hai_role_state_formal_v1.json",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--checkpoint-root", type=Path,
        default=ROOT / "checkpoints/hai_role_state_formal_v1",
    )
    parser.add_argument(
        "--validation-summary", type=Path,
        default=ROOT / "results/hai_role_state_formal_v1/summary/validation_summary.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/hai_role_state_holdout_v1",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def add_per_unit_physical_metrics(
    result: dict, unit_names: list[str], target_scales: np.ndarray
) -> dict:
    rows = []
    for index, unit in enumerate(unit_names):
        mse = float(result["per_unit_mse_standardized"][index])
        mae = float(result["per_unit_mae_standardized"][index])
        scale = float(target_scales[index])
        rows.append({
            "unit": unit,
            "mse_standardized": mse,
            "mae_standardized": mae,
            "rmse_physical": float(np.sqrt(mse) * scale),
            "mae_physical": mae * scale,
            "target_scale": scale,
        })
    result["per_unit_metrics"] = rows
    result["joint_physical_metrics_reportable"] = False
    return result


def load_checkpoint(path: Path, expected_stage: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != expected_stage or "model_state" not in checkpoint:
        raise RuntimeError(f"unexpected checkpoint stage: {path}")
    return checkpoint


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not preregistered")
    validation = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    if validation.get("accepted_for_strict_frozen_holdout") is not True:
        raise RuntimeError("HAI validation did not authorize holdout evaluation")
    if validation.get("holdout_constructed") or validation.get("holdout_evaluated"):
        raise RuntimeError("HAI validation summary already contains holdout access")
    if validation.get("dataset_sha256") != config["dataset"]["sha256"]:
        raise RuntimeError("HAI validation dataset hash mismatch")
    output_path = args.output_dir / f"seed_{args.seed}" / "holdout.json"
    if args.resume and output_path.is_file():
        log(f"SKIP completed HAI holdout seed={args.seed}")
        return 0

    seed_everything(args.seed)
    device = choose_device(args.device)
    arrays = load_frozen_hai(
        args.data or ROOT / config["dataset"]["path"],
        expected_sha256=config["dataset"]["sha256"],
    )
    loading = dict(config["data_loading"])
    loading["holdout_windows"] = len(arrays["holdout_end_indices"])
    holdout = SmarteoleWindowDataset(
        x_role=arrays["x_role"], context=arrays["context"],
        power=arrays["power"], power_valid=arrays["power_valid"],
        end_indices=evenly_spaced(
            arrays["holdout_end_indices"], int(loading["holdout_windows"])
        ),
        history_steps=int(config["dataset"]["history_steps"]),
        horizon_steps=int(config["dataset"]["horizon_steps"]),
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
        role_feature_counts=tuple(int(x) for x in arrays["role_feature_counts"]),
        role_state_hidden_size=int(model_config["role_state_hidden_size"]),
    ).to(device)
    checkpoint = args.checkpoint_root / f"seed_{args.seed}" / "mafs_maspge_rs"
    base_path = checkpoint / "finetune_best.pt"
    role_path = checkpoint / "role_state_best.pt"
    base_checkpoint = load_checkpoint(base_path, "finetune")
    role_checkpoint = load_checkpoint(role_path, "role_state")
    unit_names = [str(name) for name in arrays["unit_names"].tolist()]
    target_scales = arrays["target_scale"]
    started = time.perf_counter()
    log(
        f"STRICT HAI HOLDOUT seed={args.seed} windows={len(holdout)} "
        "training=FORBIDDEN"
    )
    model.load_state_dict(base_checkpoint["model_state"], strict=True)
    base = evaluate(
        model, holdout, finetune=True, loading=loading, device=device,
        target_scale=1.0, use_role_state=False,
    )
    add_per_unit_physical_metrics(base, unit_names, target_scales)
    model.load_state_dict(role_checkpoint["model_state"], strict=True)
    role = evaluate(
        model, holdout, finetune=True, loading=loading, device=device,
        target_scale=1.0, use_role_state=True,
    )
    add_per_unit_physical_metrics(role, unit_names, target_scales)
    base_mse = base["metrics"]["mse_standardized"]
    role_mse = role["metrics"]["mse_standardized"]
    payload = {
        "status": "complete",
        "phase": "hai_strict_frozen_holdout_report",
        "seed": args.seed,
        "training_performed": False,
        "holdout_evaluated": True,
        "strict_unseen_model_test": True,
        "post_holdout_tuning_permitted": False,
        "dataset_sha256": str(arrays["sha256"].item()),
        "validation_summary": str(args.validation_summary),
        "base_checkpoint_sha256": sha256_file(base_path),
        "role_state_checkpoint_sha256": sha256_file(role_path),
        "base_holdout": base,
        "role_state_holdout": role,
        "relative_holdout_mse_gain": (base_mse - role_mse) / base_mse,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(output_path, payload)
    log(
        f"DONE seed={args.seed} base_holdout={base_mse:.6f} "
        f"role_holdout={role_mse:.6f} "
        f"gain={100*payload['relative_holdout_mse_gain']:.2f}% training=FORBIDDEN"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
