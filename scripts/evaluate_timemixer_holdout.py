#!/usr/bin/env python3
"""Read-only frozen holdout evaluation for the three TimeMixer runs."""

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
from maspge.data import (
    SmarteoleWindowDataset, load_frozen_hai, load_frozen_sdwpf,
    load_frozen_smarteole, sha256_file,
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["smarteole", "sdwpf", "hai"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--revision", default="v2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / f"configs/{args.dataset}_timemixer_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is not preregistered")
    validation_root = ROOT / f"results/{args.dataset}_timemixer_validation_{args.revision}"
    checkpoint_root = ROOT / f"checkpoints/{args.dataset}_timemixer_validation_{args.revision}"
    output_root = ROOT / f"results/{args.dataset}_timemixer_holdout_{args.revision}"
    result_path = output_root / f"seed_{args.seed}/timemixer_power.json"
    if args.resume and result_path.is_file():
        print(f"SKIP {args.dataset} TimeMixer holdout seed={args.seed}", flush=True)
        return 0
    validation_path = validation_root / f"seed_{args.seed}/timemixer_power.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    expected_phase = f"{args.dataset}_external_validation_only"
    if validation.get("phase") != expected_phase:
        raise RuntimeError(f"unexpected validation phase: {validation_path}")
    if validation.get("holdout_evaluated") is not False:
        raise RuntimeError(f"source validation accessed holdout: {validation_path}")

    data_path = ROOT / config["dataset"]["path"]
    if args.dataset == "hai":
        arrays = load_frozen_hai(data_path, expected_sha256=config["dataset"]["sha256"])
    elif args.dataset == "sdwpf":
        arrays = load_frozen_sdwpf(data_path, expected_sha256=config["dataset"]["sha256"])
    else:
        arrays = load_frozen_smarteole(data_path, expected_sha256=config["dataset"]["sha256"])
    dataset_kwargs = {
        "x_role": arrays["x_role"], "context": arrays["context"],
        "power": arrays["power"], "end_indices": arrays["holdout_end_indices"],
        "history_steps": int(config["dataset"]["history_steps"]),
        "horizon_steps": int(config["dataset"]["horizon_steps"]),
    }
    if "power_valid" in arrays:
        dataset_kwargs["power_valid"] = arrays["power_valid"]
    holdout = SmarteoleWindowDataset(**dataset_kwargs)
    if "role_feature_counts" in arrays:
        role_counts = tuple(int(value) for value in arrays["role_feature_counts"])
    else:
        role_counts = tuple(
            int(arrays["feature_slot_mask"][0, role].sum())
            for role in range(arrays["feature_slot_mask"].shape[1])
        )
    active = arrays["active_role_mask"]
    active_roles = tuple(
        bool(active[role]) if active.ndim == 1 else bool(active[:, role].all())
        for role in range(active.shape[-1])
    )
    device = choose_device(args.device)
    model = build_external_baseline(
        "timemixer_power", config=config, role_feature_counts=role_counts,
        active_roles=active_roles, context_size=int(arrays["context"].shape[1]),
    ).to(device)
    checkpoint_path = checkpoint_root / f"seed_{args.seed}/timemixer_power/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("mode") != "timemixer_power" or checkpoint.get("seed") != args.seed:
        raise RuntimeError(f"checkpoint identity mismatch: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    training = dict(config["training_by_mode"]["timemixer_power"])
    training.update(config["data_loading"])
    started = time.perf_counter()
    result = evaluate(model, holdout, training=training, device=device, target_scale=1.0)
    payload = {
        "status": "complete", "phase": f"{args.dataset}_frozen_timemixer_holdout",
        "integration_revision": args.revision,
        "mode": "timemixer_power", "seed": args.seed,
        "training_performed": False, "holdout_evaluated": True,
        "post_holdout_tuning_permitted": False,
        "dataset_sha256": config["dataset"]["sha256"],
        "parameter_count": validation["parameter_count"],
        "validation": validation["validation"], "holdout": result,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, payload)
    print(f"DONE {args.dataset} seed={args.seed} holdout_mse="
          f"{result['metrics']['mse_standardized']:.6f} training=FORBIDDEN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
