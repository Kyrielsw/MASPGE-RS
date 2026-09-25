#!/usr/bin/env python3
"""Same-input generic per-unit state control; validation only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from maspge.mafs_adapted import PerUnitGenericStateAdapter
from run_mafs_adapted_screen import (
    ROOT,
    SmarteoleWindowDataset,
    atomic_json,
    choose_device,
    evenly_spaced,
    evaluate,
    log,
    seed_everything,
    train_stage,
)
from run_semantic_controls_validation import (
    build_model,
    checkpoint_state,
    load_arrays,
    load_model_state,
    role_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/generic_state_control_validation_v1.json",
    )
    parser.add_argument(
        "--dataset", choices=["smarteole", "sdwpf", "hai"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    control = json.loads(args.config.read_text(encoding="utf-8"))
    if control["holdout_policy"] != "forbidden_during_validation_control":
        raise RuntimeError("generic-state validation control must forbid holdout")
    if args.seed not in control["seeds"]:
        raise ValueError("seed is not preregistered")
    profile = control["datasets"][args.dataset]
    experiment = json.loads(
        (ROOT / profile["experiment_config"]).read_text(encoding="utf-8")
    )
    output_dir = args.output_dir or ROOT / (
        "results/generic_state_control_smoke_v1"
        if args.smoke
        else "results/generic_state_control_validation_v1"
    )
    checkpoint_dir = args.checkpoint_dir or ROOT / (
        "checkpoints/generic_state_control_smoke_v1"
        if args.smoke
        else "checkpoints/generic_state_control_validation_v1"
    )
    result_path = output_dir / args.dataset / f"seed_{args.seed}" / "result.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed dataset={args.dataset} seed={args.seed}")
        return 0

    loading = dict(experiment["data_loading"])
    stage = dict(experiment["role_state"])
    if args.smoke:
        loading.update(
            {
                "train_windows": 64,
                "validation_windows": 32,
                "batch_size": 8,
                "num_workers": 0,
                "log_every_batches": 4,
            }
        )
        stage.update(
            {"epochs": 1, "minimum_epochs": 1, "early_stopping_patience": 1}
        )
    seed_everything(args.seed)
    device = choose_device(args.device)
    arrays = load_arrays(args.dataset, experiment)
    counts = role_counts(args.dataset, arrays)
    common = {
        "x_role": arrays["x_role"],
        "context": arrays["context"],
        "power": arrays["power"],
        "power_valid": arrays.get("power_valid"),
        "history_steps": int(experiment["dataset"]["history_steps"]),
        "horizon_steps": int(experiment["dataset"]["horizon_steps"]),
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
    base_path = ROOT / profile["base_checkpoint"].format(seed=args.seed)
    full_path = ROOT / profile["full_checkpoint"].format(seed=args.seed)
    target_scale = (
        1.0 if args.dataset == "hai" else float(arrays["target_scale"][0])
    )
    started = time.perf_counter()

    base_checkpoint = checkpoint_state(base_path)
    generic = build_model(experiment, arrays, counts).to(device)
    load_model_state(
        generic,
        base_checkpoint,
        allow_missing_adapter=args.dataset != "hai",
    )
    base_validation = evaluate(
        generic,
        validation,
        finetune=True,
        loading=loading,
        device=device,
        target_scale=target_scale,
    )
    assert generic.role_state_adapter is not None
    semantic_adapter_parameters = sum(
        parameter.numel() for parameter in generic.role_state_adapter.parameters()
    )
    model_config = experiment["model"]
    generic.role_state_adapter = PerUnitGenericStateAdapter(
        history_scales=tuple(int(value) for value in model_config["history_scales"]),
        role_feature_counts=counts,
        state_hidden_size=int(profile["generic_state_hidden_size"]),
        d_model=int(model_config["d_model"]),
    ).to(device)
    generic.freeze_for_role_state()
    generic_adapter_parameters = sum(
        parameter.numel()
        for parameter in generic.role_state_adapter.parameters()
        if parameter.requires_grad
    )
    parameter_relative_difference = (
        generic_adapter_parameters - semantic_adapter_parameters
    ) / semantic_adapter_parameters
    if abs(parameter_relative_difference) >= 0.01:
        raise RuntimeError(
            "generic-state adapter differs from semantic adapter by at least 1%"
        )
    generic_result, generic_curve = train_stage(
        generic,
        train,
        validation,
        stage_name="same_input_generic_unit_state",
        finetune=True,
        stage=stage,
        loading=loading,
        seed=args.seed + int(profile["training_seed_offset"]),
        agent_loss_weight=0.0,
        target_scale=target_scale,
        device=device,
        checkpoint_path=(
            checkpoint_dir
            / args.dataset
            / f"seed_{args.seed}"
            / "generic_state_best.pt"
        ),
        use_role_state=True,
        role_state_l2_weight=float(stage["residual_l2_weight"]),
    )

    semantic = build_model(experiment, arrays, counts).to(device)
    load_model_state(
        semantic, checkpoint_state(full_path), allow_missing_adapter=False
    )
    semantic_validation = evaluate(
        semantic,
        validation,
        finetune=True,
        loading=loading,
        device=device,
        target_scale=target_scale,
        use_role_state=True,
    )
    base_mse = base_validation["metrics"]["mse_standardized"]
    generic_mse = generic_result["validation"]["metrics"]["mse_standardized"]
    semantic_mse = semantic_validation["metrics"]["mse_standardized"]
    payload = {
        "status": "complete",
        "phase": control["phase"],
        "dataset": args.dataset,
        "seed": args.seed,
        "smoke": args.smoke,
        "holdout_evaluated": False,
        "hyperparameter_search": False,
        "same_sensor_values": True,
        "same_unit_axis": True,
        "role_specific_encoders": False,
        "role_feature_counts_used_only_to_remove_padding": list(counts),
        "flat_sensor_count_per_unit": sum(counts),
        "generic_state_hidden_size": int(profile["generic_state_hidden_size"]),
        "training_seed": args.seed + int(profile["training_seed_offset"]),
        "semantic_adapter_parameters": semantic_adapter_parameters,
        "generic_adapter_parameters": generic_adapter_parameters,
        "parameter_relative_difference": parameter_relative_difference,
        "base_checkpoint": str(base_path),
        "semantic_checkpoint": str(full_path),
        "base_validation": base_validation,
        "generic_state_validation": generic_result["validation"],
        "semantic_validation": semantic_validation,
        "generic_best_epoch": generic_result["best_epoch"],
        "generic_learning_curve": generic_curve,
        "generic_gain_vs_base": (base_mse - generic_mse) / base_mse,
        "semantic_gain_vs_generic": (generic_mse - semantic_mse) / generic_mse,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, payload)
    log(
        f"DONE dataset={args.dataset} seed={args.seed} "
        f"base={base_mse:.6f} generic={generic_mse:.6f} "
        f"semantic={semantic_mse:.6f} "
        f"generic_gain={100*payload['generic_gain_vs_base']:.2f}% "
        f"semantic_gain={100*payload['semantic_gain_vs_generic']:.2f}% "
        f"params={generic_adapter_parameters}/{semantic_adapter_parameters} "
        "holdout=NOT_EVALUATED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
