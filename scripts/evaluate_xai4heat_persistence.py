#!/usr/bin/env python3
"""Evaluate the parameter-free XAI4HEAT persistence reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from maspge.data import load_frozen_xai4heat
from run_external_baseline_screen import (
    ROOT, SmarteoleWindowDataset, atomic_json, choose_device, evaluate, seed_everything,
)


class Persistence(nn.Module):
    def forward(self, *, y_current: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return y_current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/xai4heat_external_baselines_v1.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/xai4heat_persistence_validation_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result_path = args.output_dir / f"seed_{args.seed}" / "persistence.json"
    if args.resume and result_path.is_file():
        print(f"SKIP XAI4HEAT persistence seed={args.seed}", flush=True)
        return 0
    arrays = load_frozen_xai4heat(
        ROOT / config["dataset"]["path"], expected_sha256=config["dataset"]["sha256"]
    )
    dataset = SmarteoleWindowDataset(
        x_role=arrays["x_role"], context=arrays["context"], power=arrays["power"],
        power_valid=arrays["power_valid"], end_indices=arrays["validation_end_indices"],
        history_steps=int(config["dataset"]["history_steps"]),
        horizon_steps=int(config["dataset"]["horizon_steps"]),
    )
    seed_everything(args.seed)
    device = choose_device(args.device)
    result = evaluate(
        Persistence().to(device), dataset,
        training={"batch_size": int(config["data_loading"]["batch_size"]), "num_workers": 0},
        device=device, target_scale=1.0,
    )
    atomic_json(result_path, {
        "status": "complete", "phase": "xai4heat_persistence_validation",
        "seed": args.seed, "mode": "persistence", "parameter_count": 0,
        "training_performed": False, "holdout_evaluated": False,
        "dataset_sha256": str(arrays["sha256"].item()), "validation": result,
    })
    print(f"DONE seed={args.seed} persistence validation_mse={result['metrics']['mse_standardized']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
