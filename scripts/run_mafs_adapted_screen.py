#!/usr/bin/env python3
"""Two-stage, validation-only MAFS-adapted screen for SMARTEOLE."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from maspge.data import SmarteoleWindowDataset, load_frozen_smarteole
from maspge.mafs_adapted import MAFSAdaptedPower
from maspge.metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/smarteole_mafs_adapted_screen_v1.json",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--mode", choices=["mafs_adapted_fully", "mafs_adapted_ring"], required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/mafs_adapted_v1_screen")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/mafs_adapted_v1_screen")
    parser.add_argument("--train-windows", type=int)
    parser.add_argument("--validation-windows", type=int)
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def evenly_spaced(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices.copy()
    positions = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
    return indices[positions]


def make_loader(
    dataset: SmarteoleWindowDataset,
    *,
    loading: dict,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    workers = int(loading["num_workers"])
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(loading["batch_size"]),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def evaluate(
    model: MAFSAdaptedPower,
    dataset: SmarteoleWindowDataset,
    *,
    finetune: bool,
    loading: dict,
    device: torch.device,
    target_scale: float,
    use_role_router: bool = False,
    router_input_mode: str = "full",
    use_role_state: bool = False,
    role_state_ablation_mode: str = "full",
    generic_state_mode: str = "unitwise",
    role_state_injection: str = "pre_communication",
) -> dict:
    loader = make_loader(dataset, loading=loading, device=device, shuffle=False, seed=0)
    predictions = []
    targets = []
    target_validity = []
    sequence_squared_sum = 0.0
    sequence_absolute_sum = 0.0
    sequence_count = 0
    vote_sum = None
    vote_count = 0
    last_adjacency = None
    route_square_sum = 0.0
    route_count = 0
    role_state_square_sum = 0.0
    role_state_count = 0
    model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = model(
                batch["power_history"],
                finetune=finetune,
                x_role=batch["x_role"],
                context=batch["context"],
                use_role_router=use_role_router,
                router_input_mode=router_input_mode,
                use_role_state=use_role_state,
                role_state_ablation_mode=role_state_ablation_mode,
                generic_state_mode=generic_state_mode,
                role_state_injection=role_state_injection,
            )
            sequence = output["final_sequence"]
            target_sequence = batch["power_future_sequence"]
            sequence_valid = batch["power_future_valid"].bool()
            valid_count = sequence_valid.sum(dim=1).clamp_min(1)
            predictions.append(
                ((sequence * sequence_valid).sum(dim=1) / valid_count).cpu()
            )
            targets.append(batch["y_future"].cpu())
            target_validity.append(batch["y_future_valid"].cpu())
            error = sequence - target_sequence
            sequence_squared_sum += float(error.square()[sequence_valid].sum())
            sequence_absolute_sum += float(error.abs()[sequence_valid].sum())
            sequence_count += int(sequence_valid.sum())
            votes = output["vote_weights"]
            current_sum = votes.sum(dim=0).cpu()
            vote_sum = current_sum if vote_sum is None else vote_sum + current_sum
            vote_count += votes.shape[0]
            last_adjacency = output["adjacency"].detach().cpu()
            route_logits = output["role_route_logits"]
            route_square_sum += float(route_logits.square().sum())
            route_count += route_logits.numel()
            role_state_square_sum += (
                float(output["role_state_mean_square"].detach())
                * votes.shape[0]
            )
            role_state_count += votes.shape[0]
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    target_valid = torch.cat(target_validity)
    per_unit_mse = []
    per_unit_mae = []
    absolute = (prediction - target).abs()
    squared = absolute.square()
    for unit in range(prediction.shape[1]):
        unit_valid = target_valid[:, unit]
        per_unit_mse.append(float(squared[unit_valid, unit].mean()))
        per_unit_mae.append(float(absolute[unit_valid, unit].mean()))
    return {
        "metrics": regression_metrics(
            prediction, target, target_scale=target_scale, valid_mask=target_valid
        ),
        "per_unit_mse_standardized": per_unit_mse,
        "per_unit_mae_standardized": per_unit_mae,
        "prediction_shape": list(prediction.shape),
        "sequence_mse_standardized": sequence_squared_sum / sequence_count,
        "sequence_mae_standardized": sequence_absolute_sum / sequence_count,
        "mean_agent_vote_weights": (vote_sum / vote_count).tolist(),
        "agent_adjacency": last_adjacency.tolist(),
        "role_route_rms": (
            (route_square_sum / route_count) ** 0.5 if route_count else 0.0
        ),
        "role_state_rms": (
            (role_state_square_sum / role_state_count) ** 0.5
            if role_state_count else 0.0
        ),
    }


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_stage(
    model: MAFSAdaptedPower,
    train_dataset: SmarteoleWindowDataset,
    validation_dataset: SmarteoleWindowDataset,
    *,
    stage_name: str,
    finetune: bool,
    stage: dict,
    loading: dict,
    seed: int,
    agent_loss_weight: float,
    target_scale: float,
    device: torch.device,
    checkpoint_path: Path,
    use_role_router: bool = False,
    route_logit_l2_weight: float = 0.0,
    router_input_mode: str = "full",
    use_role_state: bool = False,
    role_state_l2_weight: float = 0.0,
    role_state_ablation_mode: str = "full",
    generic_state_mode: str = "unitwise",
    role_state_injection: str = "pre_communication",
) -> tuple[dict, list[dict]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(stage["learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
    )
    loader = make_loader(train_dataset, loading=loading, device=device, shuffle=True, seed=seed)
    best_mse = float("inf")
    best_state = None
    best_validation = None
    best_epoch = 0
    patience = 0
    curve = []
    for epoch in range(1, int(stage["epochs"]) + 1):
        model.train()
        objective_sum = 0.0
        batch_count = 0
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["power_history"],
                finetune=finetune,
                x_role=batch["x_role"],
                context=batch["context"],
                use_role_router=use_role_router,
                router_input_mode=router_input_mode,
                use_role_state=use_role_state,
                role_state_ablation_mode=role_state_ablation_mode,
                generic_state_mode=generic_state_mode,
                role_state_injection=role_state_injection,
            )
            sequence_error = (
                output["final_sequence"] - batch["power_future_sequence"]
            ).square()
            sequence_valid = batch["power_future_valid"].bool()
            final_loss = sequence_error[sequence_valid].mean()
            agent_loss = torch.zeros((), device=device)
            for sequence, horizon in zip(output["agent_sequences"], model.horizon_scales):
                target = batch["power_future_sequence"][:, :horizon]
                valid = batch["power_future_valid"][:, :horizon].bool()
                agent_loss = agent_loss + (sequence - target).square()[valid].mean()
            route_penalty = output["role_route_logits"].square().mean()
            role_state_penalty = output["role_state_mean_square"]
            objective = (
                final_loss
                + agent_loss_weight * agent_loss
                + route_logit_l2_weight * route_penalty
                + role_state_l2_weight * role_state_penalty
            )
            if not torch.isfinite(objective):
                raise RuntimeError(f"non-finite {stage_name} objective")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(stage["gradient_clip_norm"]))
            optimizer.step()
            objective_sum += float(objective.detach())
            batch_count += 1
            if batch_index % int(loading["log_every_batches"]) == 0 or batch_index == len(loader):
                memory = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
                log(
                    f"stage={stage_name} epoch={epoch} batch={batch_index}/{len(loader)} "
                    f"objective={objective_sum/batch_count:.6f} gpu_mem={memory:.2f}GiB"
                )
        validation = evaluate(
            model,
            validation_dataset,
            finetune=finetune,
            loading=loading,
            device=device,
            target_scale=target_scale,
            use_role_router=use_role_router,
            router_input_mode=router_input_mode,
            use_role_state=use_role_state,
            role_state_ablation_mode=role_state_ablation_mode,
            generic_state_mode=generic_state_mode,
            role_state_injection=role_state_injection,
        )
        validation_mse = validation["metrics"]["mse_standardized"]
        curve.append(
            {
                "epoch": epoch,
                "train_objective": objective_sum / batch_count,
                "validation_mse_standardized": validation_mse,
                "validation_mae_standardized": validation["metrics"]["mae_standardized"],
                "validation_sequence_mse_standardized": validation["sequence_mse_standardized"],
            }
        )
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            best_validation = validation
            patience = 0
            atomic_save(
                checkpoint_path,
                {"model_state": best_state, "stage": stage_name, "epoch": epoch, "validation": validation},
            )
        else:
            patience += 1
        log(
            f"EPOCH stage={stage_name} epoch={epoch} val_mse={validation_mse:.6f} "
            f"best_epoch={best_epoch} patience={patience}/{stage['early_stopping_patience']}"
        )
        if epoch >= int(stage["minimum_epochs"]) and patience >= int(stage["early_stopping_patience"]):
            log(f"EARLY-STOP stage={stage_name} epoch={epoch}")
            break
    if best_state is None or best_validation is None:
        raise RuntimeError(f"no best state for {stage_name}")
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "validation": best_validation}, curve


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    loading = dict(config["data_loading"])
    pretrain = dict(config["pretrain"])
    finetune_config = dict(config["finetune"])
    if args.train_windows is not None:
        loading["train_windows"] = args.train_windows
    if args.validation_windows is not None:
        loading["validation_windows"] = args.validation_windows
    if args.num_workers is not None:
        loading["num_workers"] = args.num_workers
    if args.pretrain_epochs is not None:
        pretrain["epochs"] = args.pretrain_epochs
    if args.finetune_epochs is not None:
        finetune_config["epochs"] = args.finetune_epochs
    result_path = args.output_dir / f"seed_{config['seed']}" / f"{args.mode}.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed mode={args.mode}: {result_path}")
        return 0
    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device(args.device)
    data_path = args.data or ROOT / config["dataset"]["path"]
    arrays = load_frozen_smarteole(data_path, expected_sha256=config["dataset"]["sha256"])
    common = {
        "x_role": arrays["x_role"],
        "context": arrays["context"],
        "power": arrays["power"],
        "history_steps": int(config["dataset"]["history_steps"]),
        "horizon_steps": int(config["dataset"]["horizon_steps"]),
    }
    train_dataset = SmarteoleWindowDataset(
        **common,
        end_indices=evenly_spaced(arrays["train_end_indices"], int(loading["train_windows"])),
    )
    validation_dataset = SmarteoleWindowDataset(
        **common,
        end_indices=evenly_spaced(arrays["validation_end_indices"], int(loading["validation_windows"])),
    )
    topology = args.mode.removeprefix("mafs_adapted_")
    model_config = config["model"]
    model = MAFSAdaptedPower(
        history_scales=tuple(int(value) for value in model_config["history_scales"]),
        horizon_scales=tuple(int(value) for value in model_config["horizon_scales"]),
        unit_count=int(arrays["power"].shape[1]),
        full_horizon_steps=int(config["dataset"]["horizon_steps"]),
        d_model=int(model_config["d_model"]),
        n_heads=int(model_config["n_heads"]),
        layers=int(model_config["layers"]),
        feedforward_size=int(model_config["feedforward_size"]),
        dropout=float(model_config["dropout"]),
        topology=topology,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    target_scale = float(arrays["target_scale"][0])
    checkpoint_root = args.checkpoint_dir / f"seed_{seed}" / args.mode
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    log(
        f"device={device} mode={args.mode} params={parameter_count} "
        f"train_windows={len(train_dataset)} validation_windows={len(validation_dataset)} "
        "holdout=FORBIDDEN"
    )
    pretrain_result, pretrain_curve = train_stage(
        model,
        train_dataset,
        validation_dataset,
        stage_name="pretrain",
        finetune=False,
        stage=pretrain,
        loading=loading,
        seed=seed,
        agent_loss_weight=float(model_config["agent_loss_weight"]),
        target_scale=target_scale,
        device=device,
        checkpoint_path=checkpoint_root / "pretrain_best.pt",
    )
    model.freeze_specialist_agents()
    finetune_result, finetune_curve = train_stage(
        model,
        train_dataset,
        validation_dataset,
        stage_name="finetune",
        finetune=True,
        stage=finetune_config,
        loading=loading,
        seed=seed + 10000,
        agent_loss_weight=float(model_config["agent_loss_weight"]),
        target_scale=target_scale,
        device=device,
        checkpoint_path=checkpoint_root / "finetune_best.pt",
    )
    result = {
        "status": "complete",
        "phase": "validation_only_two_stage_screen",
        "holdout_evaluated": False,
        "seed": seed,
        "mode": args.mode,
        "parameter_count": parameter_count,
        "best_epoch": finetune_result["best_epoch"],
        "epochs_completed": len(pretrain_curve) + len(finetune_curve),
        "validation": finetune_result["validation"],
        "pretrain": {**pretrain_result, "learning_curve": pretrain_curve},
        "finetune": {**finetune_result, "learning_curve": finetune_curve},
        "input_protocol": "power_history_plus_training_only_future_sequence_auxiliary_labels",
        "adaptation": {
            "history_scales": model_config["history_scales"],
            "horizon_scales": model_config["horizon_scales"],
            "topology": topology,
            "final_evaluation_target": "mean_of_predicted_15_step_sequence",
        },
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0,
    }
    atomic_json(result_path, result)
    log(
        f"DONE mode={args.mode} pretrain_val_mse={pretrain_result['validation']['metrics']['mse_standardized']:.6f} "
        f"finetune_val_mse={finetune_result['validation']['metrics']['mse_standardized']:.6f} "
        "holdout=NOT_EVALUATED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
