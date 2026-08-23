#!/usr/bin/env python3
"""Read-only HAI holdout evaluation for frozen external baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_external_baseline_screen import (
    ROOT, build_external_baseline, choose_device, evaluate,
)
from maspge.data import SmarteoleWindowDataset, load_frozen_hai, sha256_file


MODES = {
    "dlinear_power", "fouriergnn_power_official_small",
    "itransformer_power", "softs_power_compact",
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def add_per_unit_metrics(result: dict, arrays: dict[str, np.ndarray]) -> None:
    result["per_unit_metrics"] = [
        {
            "unit": str(arrays["unit_names"][index]),
            "mse_standardized": float(result["per_unit_mse_standardized"][index]),
            "mae_standardized": float(result["per_unit_mae_standardized"][index]),
            "rmse_physical": float(
                np.sqrt(result["per_unit_mse_standardized"][index])
                * arrays["target_scale"][index]
            ),
            "mae_physical": float(
                result["per_unit_mae_standardized"][index]
                * arrays["target_scale"][index]
            ),
        }
        for index in range(len(arrays["unit_names"]))
    ]
    result["joint_physical_metrics_reportable"] = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/hai_external_baselines_v1.json",
    )
    parser.add_argument(
        "--validation-root", type=Path,
        default=ROOT / "results/hai_external_validation_v1",
    )
    parser.add_argument(
        "--checkpoint-root", type=Path,
        default=ROOT / "checkpoints/hai_external_validation_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/hai_external_holdout_v1",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not preregistered")
    result_path = args.output_dir / f"seed_{args.seed}" / f"{args.mode}.json"
    if args.resume and result_path.is_file():
        print(f"SKIP HAI holdout seed={args.seed} mode={args.mode}", flush=True)
        return 0
    validation_path = args.validation_root / f"seed_{args.seed}" / f"{args.mode}.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("phase") != "hai_external_validation_only":
        raise RuntimeError(f"unexpected validation phase: {validation_path}")
    if validation.get("holdout_evaluated") is not False:
        raise RuntimeError(f"source validation accessed holdout: {validation_path}")
    arrays = load_frozen_hai(
        ROOT / config["dataset"]["path"],
        expected_sha256=config["dataset"]["sha256"],
    )
    holdout = SmarteoleWindowDataset(
        x_role=arrays["x_role"], context=arrays["context"],
        power=arrays["power"], power_valid=arrays["power_valid"],
        end_indices=arrays["holdout_end_indices"],
        history_steps=int(config["dataset"]["history_steps"]),
        horizon_steps=int(config["dataset"]["horizon_steps"]),
    )
    role_counts = tuple(int(value) for value in arrays["role_feature_counts"])
    active_roles = tuple(bool(value) for value in arrays["active_role_mask"].all(axis=0))
    device = choose_device(args.device)
    model = build_external_baseline(
        args.mode, config=config, role_feature_counts=role_counts,
        active_roles=active_roles, context_size=int(arrays["context"].shape[1]),
    ).to(device)
    checkpoint_path = (
        args.checkpoint_root / f"seed_{args.seed}" / args.mode / "best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("mode") != args.mode or checkpoint.get("seed") != args.seed:
        raise RuntimeError(f"checkpoint identity mismatch: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    training = dict(config["training_by_mode"][args.mode])
    training.update(config["data_loading"])
    started = time.perf_counter()
    holdout_result = evaluate(
        model, holdout, training=training, device=device, target_scale=1.0,
    )
    add_per_unit_metrics(holdout_result, arrays)
    payload = {
        "status": "complete",
        "phase": "hai_frozen_external_holdout",
        "mode": args.mode,
        "seed": args.seed,
        "training_performed": False,
        "holdout_evaluated": True,
        "post_holdout_tuning_permitted": False,
        "hai_hyperparameter_search": False,
        "dataset_sha256": str(arrays["sha256"].item()),
        "parameter_count": validation["parameter_count"],
        "validation": validation["validation"],
        "holdout": holdout_result,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, payload)
    print(
        f"DONE seed={args.seed} mode={args.mode} holdout_mse="
        f"{holdout_result['metrics']['mse_standardized']:.6f} training=FORBIDDEN",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
