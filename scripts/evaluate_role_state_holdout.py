#!/usr/bin/env python3
"""Experiment 042: report frozen role-state checkpoints on SMARTEOLE holdout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from run_mafs_adapted_screen import (
    ROOT, MAFSAdaptedPower, SmarteoleWindowDataset, atomic_json,
    choose_device, evenly_spaced, evaluate, load_frozen_smarteole, log,
)
from maspge.data import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/smarteole_role_state_holdout_v1.json",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--base-checkpoint-root", type=Path)
    parser.add_argument("--role-state-checkpoint-root", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/role_state_holdout_v1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state" not in payload or "validation" not in payload:
        raise RuntimeError(f"unexpected checkpoint payload: {path}")
    return payload


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not preregistered")
    if config["protocol"]["training_permitted"] is not False:
        raise RuntimeError("holdout evaluator requires training_permitted=false")
    result_path = args.output_dir / f"seed_{args.seed}" / "holdout_evaluation.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed seed={args.seed}")
        return 0

    device = choose_device(args.device)
    arrays = load_frozen_smarteole(
        args.data or ROOT / config["dataset"]["path"],
        expected_sha256=config["dataset"]["sha256"],
    )
    holdout_windows = 32 if args.smoke else int(
        config["data_loading"]["holdout_windows"]
    )
    holdout = SmarteoleWindowDataset(
        x_role=arrays["x_role"], context=arrays["context"], power=arrays["power"],
        end_indices=evenly_spaced(arrays["holdout_end_indices"], holdout_windows),
        history_steps=int(config["dataset"]["history_steps"]),
        horizon_steps=int(config["dataset"]["horizon_steps"]),
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

    sources = config["checkpoint_sources"]
    base_root = args.base_checkpoint_root or ROOT / sources["base"]
    role_root = args.role_state_checkpoint_root or ROOT / sources["role_state"]
    base_path = base_root / f"seed_{args.seed}" / "finetune_best.pt"
    role_path = role_root / f"seed_{args.seed}" / "role_state" / "best.pt"
    base_checkpoint = load_checkpoint(base_path)
    role_checkpoint = load_checkpoint(role_path)
    if base_checkpoint.get("stage") != "finetune":
        raise RuntimeError("base checkpoint is not a selected finetune checkpoint")
    if role_checkpoint.get("stage") != "role_state":
        raise RuntimeError("role-state checkpoint is not validation-selected")

    loading = dict(config["data_loading"])
    if args.smoke:
        loading.update({"batch_size": 16, "num_workers": 0})
    target_scale = float(arrays["target_scale"][0])
    started = time.perf_counter()
    log(f"FROZEN HOLDOUT seed={args.seed} windows={len(holdout)} training=FORBIDDEN")
    missing, unexpected = model.load_state_dict(base_checkpoint["model_state"], strict=False)
    allowed = {key for key in missing if key.startswith("role_state_adapter.")}
    if set(missing) != allowed or unexpected:
        raise RuntimeError(f"base checkpoint mismatch missing={missing} unexpected={unexpected}")
    base_holdout = evaluate(
        model, holdout, finetune=True, loading=loading, device=device,
        target_scale=target_scale, use_role_state=False,
    )
    model.load_state_dict(role_checkpoint["model_state"], strict=True)
    role_holdout = evaluate(
        model, holdout, finetune=True, loading=loading, device=device,
        target_scale=target_scale, use_role_state=True,
    )
    base_mse = base_holdout["metrics"]["mse_standardized"]
    role_mse = role_holdout["metrics"]["mse_standardized"]
    result = {
        "status": "complete", "phase": "frozen_role_state_holdout_report",
        "holdout_evaluated": True, "training_performed": False,
        "smarteole_holdout_already_observed": True,
        "eligible_as_strict_unseen_test": False,
        "seed": args.seed, "smoke": args.smoke,
        "mode": "mafs_per_unit_role_state",
        "checkpoint_paths": {"base": str(base_path), "role_state": str(role_path)},
        "checkpoint_sha256": {
            "base": sha256_file(base_path), "role_state": sha256_file(role_path),
        },
        "validation": {
            "base": base_checkpoint["validation"],
            "role_state": role_checkpoint["validation"],
        },
        "holdout": {"base": base_holdout, "role_state": role_holdout},
        "relative_holdout_mse_gain": (base_mse - role_mse) / base_mse,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    log(
        f"DONE seed={args.seed} base={base_mse:.6f} role_state={role_mse:.6f} "
        f"gain={100*result['relative_holdout_mse_gain']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
