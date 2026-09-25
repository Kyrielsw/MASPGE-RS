#!/usr/bin/env python3
"""One-shot frozen XAI4HEAT holdout evaluation; training is forbidden."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from maspge.baselines import build_external_baseline
from maspge.data import load_frozen_xai4heat
from maspge.mafs_adapted import MAFSAdaptedPower, PerUnitGenericStateAdapter
from run_external_baseline_screen import (
    ROOT, SmarteoleWindowDataset, atomic_json, choose_device, evaluate, seed_everything,
)


EXTERNAL = (
    "dlinear_power", "fouriergnn_power_official_small", "itransformer_power",
    "softs_power_compact", "timemixer_power",
)
MODES = ("persistence", *EXTERNAL, "mafs_adapted_fully", "generic_unit_state")


class Persistence(nn.Module):
    def forward(self, *, y_current: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return y_current


def checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state" not in payload:
        raise RuntimeError(f"checkpoint lacks model_state: {path}")
    return payload


def build_mafs(config: dict, *, generic: bool) -> MAFSAdaptedPower:
    model = config["model"]
    instance = MAFSAdaptedPower(
        history_scales=tuple(model["history_scales"]),
        horizon_scales=tuple(model["horizon_scales"]),
        unit_count=5, full_horizon_steps=int(config["dataset"]["horizon_steps"]),
        d_model=int(model["d_model"]), n_heads=int(model["n_heads"]),
        layers=int(model["layers"]), feedforward_size=int(model["feedforward_size"]),
        dropout=float(model["dropout"]), topology=str(model["topology"]),
    )
    if generic:
        instance.role_state_adapter = PerUnitGenericStateAdapter(
            history_scales=tuple(model["history_scales"]), role_feature_counts=(7,),
            state_hidden_size=int(model["generic_state_hidden_size"]),
            d_model=int(model["d_model"]),
        )
    return instance


def add_physical_metrics(result: dict, arrays: dict) -> None:
    rows = []
    for index, unit in enumerate(arrays["unit_names"]):
        mse = float(result["per_unit_mse_standardized"][index])
        mae = float(result["per_unit_mae_standardized"][index])
        scale = float(arrays["target_scale"][index])
        rows.append({
            "unit": str(unit), "mse_standardized": mse, "mae_standardized": mae,
            "rmse_kwh": float(np.sqrt(mse) * scale), "mae_kwh": mae * scale,
        })
    result["per_unit_metrics"] = rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/xai4heat_holdout_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main_config = json.loads((ROOT / "configs/xai4heat_frozen_v1.json").read_text(encoding="utf-8"))
    external_config = json.loads((ROOT / "configs/xai4heat_external_baselines_v1.json").read_text(encoding="utf-8"))
    if args.seed not in main_config["seeds"]:
        raise ValueError("seed is not preregistered")
    result_path = args.output_dir / f"seed_{args.seed}" / f"{args.mode}.json"
    if args.resume and result_path.is_file():
        print(f"SKIP completed seed={args.seed} mode={args.mode}", flush=True)
        return 0
    arrays = load_frozen_xai4heat(ROOT / main_config["dataset"]["path"])
    dataset = SmarteoleWindowDataset(
        x_role=arrays["x_role"], context=arrays["context"], power=arrays["power"],
        power_valid=arrays["power_valid"], end_indices=arrays["holdout_end_indices"],
        history_steps=int(main_config["dataset"]["history_steps"]),
        horizon_steps=int(main_config["dataset"]["horizon_steps"]),
    )
    seed_everything(args.seed)
    device = choose_device(args.device)
    use_role_state = False
    source_checkpoint = None
    if args.mode == "persistence":
        model = Persistence()
    elif args.mode in EXTERNAL:
        model = build_external_baseline(
            args.mode, config=external_config, role_feature_counts=(7,),
            active_roles=(True,), context_size=int(arrays["context"].shape[1]),
        )
        source_checkpoint = (
            ROOT / "checkpoints/xai4heat_external_validation_v1"
            / f"seed_{args.seed}" / args.mode / "best.pt"
        )
        model.load_state_dict(checkpoint(source_checkpoint)["model_state"], strict=True)
    elif args.mode == "mafs_adapted_fully":
        model = build_mafs(main_config, generic=False)
        source_checkpoint = (
            ROOT / "checkpoints/xai4heat_main_validation_v1"
            / f"seed_{args.seed}" / "mafs_base/finetune_best.pt"
        )
        model.load_state_dict(checkpoint(source_checkpoint)["model_state"], strict=True)
    else:
        model = build_mafs(main_config, generic=True)
        source_checkpoint = (
            ROOT / "checkpoints/xai4heat_main_validation_v1"
            / f"seed_{args.seed}" / "generic_unit_state/best.pt"
        )
        model.load_state_dict(checkpoint(source_checkpoint)["model_state"], strict=True)
        use_role_state = True
    model = model.to(device)
    loading = {"batch_size": int(main_config["data_loading"]["batch_size"]), "num_workers": 2}
    if args.mode in ("mafs_adapted_fully", "generic_unit_state"):
        # MAFS has a different call signature from the external wrappers.
        from run_mafs_adapted_screen import evaluate as evaluate_mafs
        result = evaluate_mafs(
            model, dataset, finetune=True, loading=loading, device=device,
            target_scale=1.0, use_role_state=use_role_state,
        )
    else:
        result = evaluate(model, dataset, training=loading, device=device, target_scale=1.0)
    add_physical_metrics(result, arrays)
    payload = {
        "status": "complete", "phase": "xai4heat_frozen_one_shot_holdout",
        "seed": args.seed, "mode": args.mode, "training_performed": False,
        "holdout_evaluated": True, "post_holdout_tuning_permitted": False,
        "dataset_sha256": str(arrays["sha256"].item()),
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
        "holdout": result,
    }
    atomic_json(result_path, payload)
    print(
        f"DONE seed={args.seed} mode={args.mode} "
        f"holdout_mse={result['metrics']['mse_standardized']:.6f} training=FORBIDDEN",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
