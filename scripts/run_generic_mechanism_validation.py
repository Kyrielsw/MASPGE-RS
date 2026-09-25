#!/usr/bin/env python3
"""Frozen validation-only controls for the Generic Unit State mechanism."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import torch

from maspge.data import load_frozen_smarteole, load_frozen_xai4heat
from maspge.mafs_adapted import MAFSAdaptedPower, PerUnitGenericStateAdapter
from run_mafs_adapted_screen import (
    ROOT,
    SmarteoleWindowDataset,
    atomic_json,
    choose_device,
    evenly_spaced,
    evaluate,
    log,
    make_loader,
    move_batch,
    seed_everything,
    train_stage,
)
from run_semantic_controls_validation import (
    build_model,
    checkpoint_state,
    load_model_state,
    role_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/generic_mechanism_validation_v1.json",
    )
    parser.add_argument("--dataset", choices=["smarteole", "xai4heat"], required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generic_mechanism_validation_v1")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/generic_mechanism_validation_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity(config_path: Path) -> dict[str, str | None]:
    critical = [
        ROOT / "src/maspge/mafs_adapted.py",
        ROOT / "scripts/run_mafs_adapted_screen.py",
        Path(__file__).resolve(),
        config_path.resolve(),
    ]
    digest = hashlib.sha256()
    for path in critical:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"critical_code_sha256": digest.hexdigest(), "git_commit": commit}


def build_xai_model(experiment: dict, unit_count: int) -> MAFSAdaptedPower:
    model = experiment["model"]
    return MAFSAdaptedPower(
        history_scales=tuple(int(value) for value in model["history_scales"]),
        horizon_scales=tuple(int(value) for value in model["horizon_scales"]),
        unit_count=unit_count,
        full_horizon_steps=int(experiment["dataset"]["horizon_steps"]),
        d_model=int(model["d_model"]),
        n_heads=int(model["n_heads"]),
        layers=int(model["layers"]),
        feedforward_size=int(model["feedforward_size"]),
        dropout=float(model["dropout"]),
        topology=str(model["topology"]),
    )


def gradient_norms(model: MAFSAdaptedPower) -> dict[str, float]:
    groups = {"residual_projection": [], "upstream_encoder_fusion": []}
    assert model.role_state_adapter is not None
    for name, parameter in model.role_state_adapter.named_parameters():
        if parameter.grad is None:
            value = 0.0
        else:
            value = float(parameter.grad.detach().norm())
        key = (
            "residual_projection"
            if name.startswith("residual_projections")
            else "upstream_encoder_fusion"
        )
        groups[key].append(value)
    return {
        key: float(sum(value * value for value in values) ** 0.5)
        for key, values in groups.items()
    }


def diagnostic_objective(
    model: MAFSAdaptedPower,
    batch: dict[str, torch.Tensor],
    *,
    variant: dict,
    residual_l2_weight: float,
) -> tuple[torch.Tensor, dict]:
    output = model(
        batch["power_history"],
        finetune=True,
        x_role=batch["x_role"],
        context=batch["context"],
        use_role_state=True,
        generic_state_mode=str(variant["unit_state_mode"]),
        role_state_injection=str(variant["injection"]),
    )
    valid = batch["power_future_valid"].bool()
    loss = (
        output["final_sequence"] - batch["power_future_sequence"]
    ).square()[valid].mean()
    objective = loss + residual_l2_weight * output["role_state_mean_square"]
    return objective, output


def initialization_diagnostics(
    model: MAFSAdaptedPower,
    batch: dict[str, torch.Tensor],
    *,
    variant: dict,
    stage: dict,
) -> dict:
    initial_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    model.eval()
    with torch.inference_mode():
        base = model(batch["power_history"], finetune=True)["final_sequence"]
        conditioned = model(
            batch["power_history"], finetune=True,
            x_role=batch["x_role"], context=batch["context"],
            use_role_state=True,
            generic_state_mode=str(variant["unit_state_mode"]),
            role_state_injection=str(variant["injection"]),
        )
        initial_max_difference = float(
            (conditioned["final_sequence"] - base).abs().max()
        )
        initial_residual_rms = float(conditioned["role_state_rms"])

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(stage["learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
    )
    gradient_steps = []
    model.train()
    for diagnostic_step in (0, 1):
        optimizer.zero_grad(set_to_none=True)
        objective, _ = diagnostic_objective(
            model,
            batch,
            variant=variant,
            residual_l2_weight=float(stage["residual_l2_weight"]),
        )
        objective.backward()
        gradient_steps.append(
            {
                "diagnostic_step": diagnostic_step,
                "objective": float(objective.detach()),
                "gradient_norms_before_clipping": gradient_norms(model),
            }
        )
        optimizer.step()
    model.load_state_dict(initial_state)
    return {
        "batch_size": int(batch["power_history"].shape[0]),
        "initial_max_abs_prediction_difference_from_frozen_backbone": initial_max_difference,
        "initial_residual_rms": initial_residual_rms,
        "gradient_steps": gradient_steps,
        "diagnostic_updates_discarded_before_training": True,
    }


def run(args: argparse.Namespace) -> int:
    control = json.loads(args.config.read_text(encoding="utf-8"))
    if control["holdout_policy"] != "forbidden":
        raise RuntimeError("mechanism controls must forbid holdout evaluation")
    if args.seed not in control["seeds"]:
        raise ValueError("seed is not preregistered")
    if args.variant not in control["variants"]:
        raise ValueError("variant is not preregistered")
    profile = control["datasets"][args.dataset]
    variant = control["variants"][args.variant]
    experiment_path = ROOT / profile["experiment_config"]
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    result_root = args.output_dir / args.dataset / args.variant / f"seed_{args.seed}"
    checkpoint_root = args.checkpoint_dir / args.dataset / args.variant / f"seed_{args.seed}"
    result_path = result_root / "result.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed dataset={args.dataset} variant={args.variant} seed={args.seed}")
        return 0

    loading = dict(experiment["data_loading"])
    stage = dict(experiment["role_state"] if args.dataset == "smarteole" else experiment["generic_state"])
    if args.smoke:
        loading.update({
            "train_windows": 64, "validation_windows": 32,
            "batch_size": 8, "num_workers": 0, "log_every_batches": 4,
        })
        stage.update({"epochs": 1, "minimum_epochs": 1, "early_stopping_patience": 1})

    seed_everything(args.seed)
    device = choose_device(args.device)
    data_path = ROOT / experiment["dataset"]["path"]
    actual_data_sha256 = sha256(data_path)
    accepted_hashes = profile.get(
        "accepted_platform_data_sha256", [profile["expected_data_sha256"]]
    )
    if actual_data_sha256 not in accepted_hashes:
        raise RuntimeError(f"frozen dataset hash mismatch: {data_path}")
    if args.dataset == "smarteole":
        arrays = load_frozen_smarteole(
            data_path, expected_sha256=profile["expected_data_sha256"]
        )
        counts = role_counts(args.dataset, arrays)
    else:
        arrays = load_frozen_xai4heat(data_path)
        counts = (int(arrays["x_role"].shape[-1]),)

    common = {
        "x_role": arrays["x_role"], "context": arrays["context"],
        "power": arrays["power"], "power_valid": arrays.get("power_valid"),
        "history_steps": int(experiment["dataset"]["history_steps"]),
        "horizon_steps": int(experiment["dataset"]["horizon_steps"]),
    }
    train = SmarteoleWindowDataset(
        **common,
        end_indices=evenly_spaced(arrays["train_end_indices"], int(loading["train_windows"])),
    )
    validation = SmarteoleWindowDataset(
        **common,
        end_indices=evenly_spaced(arrays["validation_end_indices"], int(loading["validation_windows"])),
    )
    base_path = ROOT / profile["base_checkpoint"].format(seed=args.seed)
    base_checkpoint = checkpoint_state(base_path)
    if args.dataset == "smarteole":
        model = build_model(experiment, arrays, counts).to(device)
        load_model_state(model, base_checkpoint, allow_missing_adapter=True)
    else:
        model = build_xai_model(experiment, int(arrays["power"].shape[1])).to(device)
        load_model_state(model, base_checkpoint, allow_missing_adapter=True)
    target_scale = (
        float(arrays["target_scale"][0])
        if args.dataset == "smarteole" else 1.0
    )
    base_validation = evaluate(
        model, validation, finetune=True, loading=loading,
        device=device, target_scale=target_scale,
    )

    adapter_seed = args.seed + int(profile["generic_state_training_seed_offset"])
    seed_everything(adapter_seed)
    model_config = experiment["model"]
    model.role_state_adapter = PerUnitGenericStateAdapter(
        history_scales=tuple(int(value) for value in model_config["history_scales"]),
        role_feature_counts=counts,
        state_hidden_size=int(profile["generic_state_hidden_size"]),
        d_model=int(model_config["d_model"]),
        residual_initialization=str(variant["initialization"]),
        residual_initialization_std=float(variant.get("initialization_std", 1e-3)),
    ).to(device)
    model.freeze_for_role_state()
    adapter_parameters = sum(
        parameter.numel() for parameter in model.role_state_adapter.parameters()
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if adapter_parameters != trainable_parameters:
        raise RuntimeError("only the generic-state adapter may be trainable")

    diagnostic_loader = make_loader(
        train, loading=loading, device=device, shuffle=False, seed=adapter_seed
    )
    diagnostic_batch = move_batch(next(iter(diagnostic_loader)), device)
    seed_everything(adapter_seed)
    diagnostics = initialization_diagnostics(
        model, diagnostic_batch, variant=variant, stage=stage
    )
    seed_everything(adapter_seed)
    started = time.perf_counter()
    result, curve = train_stage(
        model, train, validation,
        stage_name=f"generic_mechanism_{args.variant}",
        finetune=True, stage=stage, loading=loading, seed=adapter_seed,
        agent_loss_weight=0.0, target_scale=target_scale, device=device,
        checkpoint_path=checkpoint_root / "best.pt",
        use_role_state=True,
        role_state_l2_weight=float(stage["residual_l2_weight"]),
        generic_state_mode=str(variant["unit_state_mode"]),
        role_state_injection=str(variant["injection"]),
    )
    validation_metrics = result["validation"]["metrics"]
    base_metrics = base_validation["metrics"]
    payload = {
        "status": "complete",
        "phase": control["phase"],
        "dataset": args.dataset,
        "scenario": profile["scenario"],
        "variant": args.variant,
        "variant_definition": variant,
        "seed": args.seed,
        "training_seed": adapter_seed,
        "smoke": args.smoke,
        "holdout_constructed": False,
        "holdout_evaluated": False,
        "data_path": str(data_path),
        "data_sha256": actual_data_sha256,
        "expected_formal_workstation_data_sha256": profile["expected_data_sha256"],
        "config_sha256": sha256(args.config),
        "checkpoint_path": str(base_path),
        "checkpoint_sha256": sha256(base_path),
        "code_identity": code_identity(args.config),
        "adapter_parameters": adapter_parameters,
        "trainable_parameters": trainable_parameters,
        "initialization_diagnostics": diagnostics,
        "base_validation": base_validation,
        "variant_validation": result["validation"],
        "best_epoch": result["best_epoch"],
        "epochs_completed": len(curve),
        "early_stopped": len(curve) < int(stage["epochs"]),
        "learning_curve": curve,
        "paired_gain_vs_same_checkpoint_mafs": (
            base_metrics["mse_standardized"] - validation_metrics["mse_standardized"]
        ) / base_metrics["mse_standardized"],
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda" else 0.0
        ),
    }
    atomic_json(result_path, payload)
    log(
        f"DONE dataset={args.dataset} variant={args.variant} seed={args.seed} "
        f"mse={validation_metrics['mse_standardized']:.6f} "
        f"mae={validation_metrics['mae_standardized']:.6f} "
        f"best_epoch={result['best_epoch']} holdout=NOT_EVALUATED"
    )
    return 0


def failure_path(args: argparse.Namespace) -> Path:
    return args.output_dir / args.dataset / args.variant / f"seed_{args.seed}" / "failure.json"


if __name__ == "__main__":
    parsed = parse_args()
    try:
        raise SystemExit(run(parsed))
    except Exception as error:
        atomic_json(
            failure_path(parsed),
            {
                "status": "failed", "dataset": parsed.dataset,
                "variant": parsed.variant, "seed": parsed.seed,
                "smoke": parsed.smoke, "error_type": type(error).__name__,
                "error": str(error), "traceback": traceback.format_exc(),
            },
        )
        raise
