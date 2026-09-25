#!/usr/bin/env python3
"""Validation-only semantic controls for MASPGE-RS."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from run_mafs_adapted_screen import (
    ROOT, MAFSAdaptedPower, SmarteoleWindowDataset, atomic_json,
    choose_device, evenly_spaced, evaluate, log, seed_everything, train_stage,
)
from maspge.data import load_frozen_hai, load_frozen_sdwpf, load_frozen_smarteole
from maspge.semantic_controls import RoleContractControlDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/semantic_controls_validation_v1.json",
    )
    parser.add_argument("--dataset", choices=["smarteole", "sdwpf", "hai"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--contract-seed", type=int)
    parser.add_argument("--skip-permutation-evaluation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def checkpoint_state(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state" not in payload:
        raise RuntimeError(f"checkpoint lacks model_state: {path}")
    return payload


def load_arrays(dataset: str, experiment: dict) -> dict:
    path = ROOT / experiment["dataset"]["path"]
    if dataset == "smarteole":
        return load_frozen_smarteole(path, expected_sha256=experiment["dataset"]["sha256"])
    if dataset == "sdwpf":
        return load_frozen_sdwpf(path)
    return load_frozen_hai(path, expected_sha256=experiment["dataset"]["sha256"])


def role_counts(dataset: str, arrays: dict) -> tuple[int, ...]:
    if dataset == "hai":
        return tuple(int(value) for value in arrays["role_feature_counts"])
    return tuple(int(arrays["feature_slot_mask"][0, role].sum()) for role in range(4))


def build_model(experiment: dict, arrays: dict, counts: tuple[int, ...]) -> MAFSAdaptedPower:
    model = experiment["model"]
    return MAFSAdaptedPower(
        history_scales=tuple(model["history_scales"]),
        horizon_scales=tuple(model["horizon_scales"]),
        unit_count=int(arrays["power"].shape[1]),
        full_horizon_steps=int(experiment["dataset"]["horizon_steps"]),
        d_model=int(model["d_model"]), n_heads=int(model["n_heads"]),
        layers=int(model["layers"]), feedforward_size=int(model["feedforward_size"]),
        dropout=float(model["dropout"]), topology=str(model.get("topology", "fully")),
        role_feature_counts=counts, context_size=int(arrays["context"].shape[1]),
        router_hidden_size=int(model["router_hidden_size"]),
        role_state_hidden_size=int(model["role_state_hidden_size"]),
    )


def load_model_state(model: MAFSAdaptedPower, checkpoint: dict, *, allow_missing_adapter: bool) -> None:
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    allowed = {key for key in missing if key.startswith("role_state_adapter.")}
    if unexpected or (set(missing) != allowed if allow_missing_adapter else bool(missing)):
        raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")


def main() -> int:
    args = parse_args()
    control = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in control["seeds"]:
        raise ValueError("seed is not preregistered")
    profile = control["datasets"][args.dataset]
    experiment = json.loads((ROOT / profile["experiment_config"]).read_text(encoding="utf-8"))
    output_dir = args.output_dir or ROOT / (
        "results/semantic_controls_smoke_v1"
        if args.smoke else "results/semantic_controls_validation_v1"
    )
    checkpoint_dir = args.checkpoint_dir or ROOT / (
        "checkpoints/semantic_controls_smoke_v1"
        if args.smoke else "checkpoints/semantic_controls_validation_v1"
    )
    contract_seed = int(
        control["randomized_contract_seed"]
        if args.contract_seed is None else args.contract_seed
    )
    result_root = output_dir / args.dataset
    checkpoint_root = checkpoint_dir / args.dataset
    if args.contract_seed is not None:
        result_root = result_root / f"contract_{contract_seed}"
        checkpoint_root = checkpoint_root / f"contract_{contract_seed}"
    result_path = result_root / f"seed_{args.seed}" / "result.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed dataset={args.dataset} seed={args.seed}")
        return 0

    loading = dict(experiment["data_loading"])
    stage = dict(experiment["role_state"])
    if args.smoke:
        loading.update({"train_windows": 64, "validation_windows": 32, "batch_size": 8, "num_workers": 0, "log_every_batches": 4})
        stage.update({"epochs": 1, "minimum_epochs": 1, "early_stopping_patience": 1})
    seed_everything(args.seed)
    device = choose_device(args.device)
    arrays = load_arrays(args.dataset, experiment)
    counts = role_counts(args.dataset, arrays)
    common = {
        "x_role": arrays["x_role"], "context": arrays["context"],
        "power": arrays["power"], "power_valid": arrays.get("power_valid"),
        "history_steps": int(experiment["dataset"]["history_steps"]),
        "horizon_steps": int(experiment["dataset"]["horizon_steps"]),
    }
    train = SmarteoleWindowDataset(
        **common, end_indices=evenly_spaced(arrays["train_end_indices"], int(loading["train_windows"]))
    )
    validation = SmarteoleWindowDataset(
        **common, end_indices=evenly_spaced(arrays["validation_end_indices"], int(loading["validation_windows"]))
    )
    pseudo_train = RoleContractControlDataset(train, role_feature_counts=counts, permutation_seed=contract_seed)
    pseudo_validation = RoleContractControlDataset(validation, role_feature_counts=counts, permutation_seed=contract_seed)
    base_path = ROOT / profile["base_checkpoint"].format(seed=args.seed)
    full_path = ROOT / profile["full_checkpoint"].format(seed=args.seed)
    target_scale = 1.0 if args.dataset == "hai" else float(arrays["target_scale"][0])
    started = time.perf_counter()

    base_checkpoint = checkpoint_state(base_path)
    role_free = build_model(experiment, arrays, counts).to(device)
    load_model_state(role_free, base_checkpoint, allow_missing_adapter=args.dataset != "hai")
    base_validation = evaluate(role_free, validation, finetune=True, loading=loading, device=device, target_scale=target_scale)
    role_free.freeze_for_role_state()
    trainable_parameters = sum(p.numel() for p in role_free.parameters() if p.requires_grad)
    role_free_result, role_free_curve = train_stage(
        role_free, pseudo_train, pseudo_validation,
        stage_name="parameter_matched_role_free", finetune=True,
        stage=stage, loading=loading, seed=args.seed + 50000,
        agent_loss_weight=0.0, target_scale=target_scale, device=device,
        checkpoint_path=checkpoint_root / f"seed_{args.seed}" / "role_free_best.pt",
        use_role_state=True, role_state_l2_weight=float(stage["residual_l2_weight"]),
    )

    semantic = build_model(experiment, arrays, counts).to(device)
    full_checkpoint = checkpoint_state(full_path)
    load_model_state(semantic, full_checkpoint, allow_missing_adapter=False)
    semantic_parameters = sum(p.numel() for p in semantic.role_state_adapter.parameters())
    if semantic_parameters != trainable_parameters:
        raise RuntimeError("parameter-matched control does not match MASPGE-RS adapter")
    semantic_validation = evaluate(
        semantic, validation, finetune=True, loading=loading, device=device,
        target_scale=target_scale, use_role_state=True,
    )
    permutation_rows = []
    if not args.skip_permutation_evaluation:
        for permutation_seed in control["frozen_permutation_seeds"]:
            wrong_contract = RoleContractControlDataset(
                validation, role_feature_counts=counts, permutation_seed=int(permutation_seed)
            )
            metrics = evaluate(
                semantic, wrong_contract, finetune=True, loading=loading, device=device,
                target_scale=target_scale, use_role_state=True,
            )
            permutation_rows.append({"permutation_seed": int(permutation_seed), "validation": metrics})

    base_mse = base_validation["metrics"]["mse_standardized"]
    semantic_mse = semantic_validation["metrics"]["mse_standardized"]
    role_free_mse = role_free_result["validation"]["metrics"]["mse_standardized"]
    permuted_mse = [row["validation"]["metrics"]["mse_standardized"] for row in permutation_rows]
    permuted_mean = sum(permuted_mse) / len(permuted_mse) if permuted_mse else None
    payload = {
        "status": "complete", "phase": control["phase"],
        "dataset": args.dataset, "seed": args.seed, "smoke": args.smoke,
        "holdout_evaluated": False, "hyperparameter_search": False,
        "randomized_contract_seed": contract_seed,
        "role_feature_counts": list(counts),
        "adapter_trainable_parameters": trainable_parameters,
        "semantic_adapter_parameters": semantic_parameters,
        "exact_parameter_match": trainable_parameters == semantic_parameters,
        "base_checkpoint": str(base_path), "full_checkpoint": str(full_path),
        "base_validation": base_validation,
        "semantic_validation": semantic_validation,
        "parameter_matched_role_free": {**role_free_result, "learning_curve": role_free_curve},
        "frozen_role_permutations": permutation_rows,
        "semantic_gain_vs_base": (base_mse - semantic_mse) / base_mse,
        "semantic_gain_vs_role_free": (role_free_mse - semantic_mse) / role_free_mse,
        "mean_permutation_degradation": (
            (permuted_mean - semantic_mse) / semantic_mse
            if permuted_mean is not None else None
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, payload)
    permutation_status = (
        f"permuted_mean={permuted_mean:.6f}"
        if permuted_mean is not None else "permuted=SKIPPED"
    )
    log(
        f"DONE dataset={args.dataset} seed={args.seed} contract={contract_seed} "
        f"base={base_mse:.6f} semantic={semantic_mse:.6f} "
        f"role_free={role_free_mse:.6f} {permutation_status} "
        f"params={trainable_parameters} holdout=NOT_EVALUATED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
