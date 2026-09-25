#!/usr/bin/env python3
"""Evaluate frozen MAFS, generic-state, and MASPGE-RS checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from maspge.data import sha256_file
from maspge.mafs_adapted import PerUnitGenericStateAdapter
from run_mafs_adapted_screen import (
    ROOT,
    SmarteoleWindowDataset,
    atomic_json,
    choose_device,
    evaluate,
    log,
    seed_everything,
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
        default=ROOT / "configs/generic_state_control_holdout_v1.json",
    )
    parser.add_argument(
        "--dataset", choices=["smarteole", "sdwpf", "hai"], required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def require_stage(
    checkpoint: dict, path: Path, expected: str, *, allow_synthetic_smoke: bool = False
) -> None:
    actual = checkpoint.get("stage")
    if actual != expected and not (allow_synthetic_smoke and actual == "synthetic_smoke"):
        raise RuntimeError(
            f"unexpected checkpoint stage for {path}: {checkpoint.get('stage')}"
        )


def main() -> int:
    args = parse_args()
    holdout_config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = holdout_config["protocol"]
    forbidden = (
        "training_permitted",
        "checkpoint_selection_permitted",
        "hyperparameter_search_permitted",
        "post_holdout_tuning_permitted",
    )
    if any(protocol[key] is not False for key in forbidden):
        raise RuntimeError("frozen holdout protocol permits a forbidden operation")
    if args.seed not in holdout_config["seeds"]:
        raise ValueError("seed is not preregistered")

    control = json.loads(
        (ROOT / holdout_config["validation_control"]).read_text(encoding="utf-8")
    )
    profile = control["datasets"][args.dataset]
    experiment = json.loads(
        (ROOT / profile["experiment_config"]).read_text(encoding="utf-8")
    )
    validation_path = (
        ROOT
        / holdout_config["validation_result_root"]
        / args.dataset
        / f"seed_{args.seed}"
        / "result.json"
    )
    validation_record = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation_record.get("holdout_evaluated") is not False:
        raise RuntimeError("generic-state source run was not validation-only")
    if validation_record.get("smoke") and not args.smoke:
        raise RuntimeError("smoke checkpoint cannot authorize holdout evaluation")
    if not validation_record.get("same_sensor_values"):
        raise RuntimeError("validation source does not satisfy the same-input control")

    output_root = args.output_dir or ROOT / holdout_config["output_root"]
    result_path = output_root / args.dataset / f"seed_{args.seed}" / "holdout.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed dataset={args.dataset} seed={args.seed}")
        return 0

    seed_everything(args.seed)
    device = choose_device(args.device)
    arrays = load_arrays(args.dataset, experiment)
    counts = role_counts(args.dataset, arrays)
    holdout_indices = arrays["holdout_end_indices"]
    if args.smoke:
        holdout_indices = holdout_indices[:32]
    holdout = SmarteoleWindowDataset(
        x_role=arrays["x_role"],
        context=arrays["context"],
        power=arrays["power"],
        power_valid=arrays.get("power_valid"),
        end_indices=holdout_indices,
        history_steps=int(experiment["dataset"]["history_steps"]),
        horizon_steps=int(experiment["dataset"]["horizon_steps"]),
    )
    loading = dict(experiment["data_loading"])
    if args.smoke:
        loading.update({"batch_size": 16, "num_workers": 0})
    target_scale = (
        1.0 if args.dataset == "hai" else float(arrays["target_scale"][0])
    )

    base_path = ROOT / profile["base_checkpoint"].format(seed=args.seed)
    semantic_path = ROOT / profile["full_checkpoint"].format(seed=args.seed)
    generic_path = (
        ROOT
        / holdout_config["generic_checkpoint_root"]
        / args.dataset
        / f"seed_{args.seed}"
        / "generic_state_best.pt"
    )
    base_checkpoint = checkpoint_state(base_path)
    semantic_checkpoint = checkpoint_state(semantic_path)
    generic_checkpoint = checkpoint_state(generic_path)
    require_stage(
        base_checkpoint, base_path, "finetune", allow_synthetic_smoke=args.smoke
    )
    require_stage(
        semantic_checkpoint,
        semantic_path,
        "role_state",
        allow_synthetic_smoke=args.smoke,
    )
    require_stage(
        generic_checkpoint, generic_path, "same_input_generic_unit_state"
    )
    checkpoint_validation_mse = generic_checkpoint["validation"]["metrics"][
        "mse_standardized"
    ]
    recorded_validation_mse = validation_record["generic_state_validation"]["metrics"][
        "mse_standardized"
    ]
    if abs(checkpoint_validation_mse - recorded_validation_mse) > 1e-10:
        raise RuntimeError("generic checkpoint does not match frozen validation record")

    started = time.perf_counter()
    log(
        f"FROZEN HOLDOUT dataset={args.dataset} seed={args.seed} "
        f"windows={len(holdout)} training=FORBIDDEN"
    )
    base = build_model(experiment, arrays, counts).to(device)
    load_model_state(
        base, base_checkpoint, allow_missing_adapter=args.dataset != "hai"
    )
    base_holdout = evaluate(
        base,
        holdout,
        finetune=True,
        loading=loading,
        device=device,
        target_scale=target_scale,
        use_role_state=False,
    )

    semantic = build_model(experiment, arrays, counts).to(device)
    load_model_state(semantic, semantic_checkpoint, allow_missing_adapter=False)
    semantic_holdout = evaluate(
        semantic,
        holdout,
        finetune=True,
        loading=loading,
        device=device,
        target_scale=target_scale,
        use_role_state=True,
    )

    generic = build_model(experiment, arrays, counts).to(device)
    model_config = experiment["model"]
    generic.role_state_adapter = PerUnitGenericStateAdapter(
        history_scales=tuple(int(value) for value in model_config["history_scales"]),
        role_feature_counts=counts,
        state_hidden_size=int(profile["generic_state_hidden_size"]),
        d_model=int(model_config["d_model"]),
    ).to(device)
    missing, unexpected = generic.load_state_dict(
        generic_checkpoint["model_state"], strict=True
    )
    if missing or unexpected:
        raise RuntimeError(
            f"generic checkpoint mismatch missing={missing} unexpected={unexpected}"
        )
    generic_holdout = evaluate(
        generic,
        holdout,
        finetune=True,
        loading=loading,
        device=device,
        target_scale=target_scale,
        use_role_state=True,
    )

    base_mse = base_holdout["metrics"]["mse_standardized"]
    generic_mse = generic_holdout["metrics"]["mse_standardized"]
    semantic_mse = semantic_holdout["metrics"]["mse_standardized"]
    payload = {
        "status": "complete",
        "phase": holdout_config["phase"],
        "dataset": args.dataset,
        "seed": args.seed,
        "smoke": args.smoke,
        "training_performed": False,
        "holdout_evaluated": True,
        "post_holdout_tuning_permitted": False,
        "study_level_holdout_previously_observed": True,
        "eligible_as_strict_unseen_test": False,
        "same_sensor_values_generic_and_maspge": True,
        "validation_authorization": str(validation_path),
        "checkpoint_paths": {
            "mafs_base": str(base_path),
            "generic_unit_state": str(generic_path),
            "maspge_rs": str(semantic_path),
        },
        "checkpoint_sha256": {
            "mafs_base": sha256_file(base_path),
            "generic_unit_state": sha256_file(generic_path),
            "maspge_rs": sha256_file(semantic_path),
        },
        "mafs_base_holdout": base_holdout,
        "generic_unit_state_holdout": generic_holdout,
        "maspge_rs_holdout": semantic_holdout,
        "generic_gain_vs_base": (base_mse - generic_mse) / base_mse,
        "maspge_gain_vs_generic": (generic_mse - semantic_mse) / generic_mse,
        "maspge_gain_vs_base": (base_mse - semantic_mse) / base_mse,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, payload)
    log(
        f"DONE dataset={args.dataset} seed={args.seed} base={base_mse:.6f} "
        f"generic={generic_mse:.6f} maspge={semantic_mse:.6f} "
        f"generic_gain={100*payload['generic_gain_vs_base']:.2f}% "
        f"maspge_gain={100*payload['maspge_gain_vs_generic']:.2f}% "
        "training=FORBIDDEN"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
