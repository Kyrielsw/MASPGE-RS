#!/usr/bin/env python3
"""Validation-only Experiment 028A for adapted external baselines."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from maspge.baselines import build_external_baseline
from maspge.data import SmarteoleWindowDataset, load_frozen_smarteole
from maspge.metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/smarteole_external_baselines_v1.json",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/external_baselines_v1_screen",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints/external_baselines_v1_screen",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-windows", type=int)
    parser.add_argument("--validation-windows", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", type=int)
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
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
    training: dict,
    device: torch.device,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    workers = int(training["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def predict(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        x_role=batch["x_role"],
        context=batch["context"],
        power_history=batch["power_history"],
        y_current=batch["y_current"],
    )


def evaluate(
    model: torch.nn.Module,
    dataset: SmarteoleWindowDataset,
    *,
    training: dict,
    device: torch.device,
    target_scale: float,
) -> dict:
    loader = make_loader(
        dataset, training=training, device=device, shuffle=False, seed=0
    )
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            predictions.append(predict(model, batch).cpu())
            targets.append(batch["y_future"].cpu())
            valid_masks.append(batch["y_future_valid"].cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    valid = torch.cat(valid_masks).bool()
    squared_error = (prediction - target).square()
    absolute_error = (prediction - target).abs()
    valid_per_unit = valid.sum(dim=0).clamp_min(1)
    return {
        "metrics": regression_metrics(
            prediction, target, target_scale=target_scale, valid_mask=valid
        ),
        "per_unit_mse_standardized": (
            (squared_error * valid).sum(dim=0).div(valid_per_unit).tolist()
        ),
        "per_unit_mae_standardized": (
            (absolute_error * valid).sum(dim=0).div(valid_per_unit).tolist()
        ),
        "prediction_shape": list(prediction.shape),
    }


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def run_mode(
    *,
    mode: str,
    seed: int,
    config: dict,
    arrays: dict[str, np.ndarray],
    train_dataset: SmarteoleWindowDataset,
    validation_dataset: SmarteoleWindowDataset,
    holdout_dataset: SmarteoleWindowDataset | None = None,
    phase: str = "validation_only_screen",
    device: torch.device,
    output_dir: Path,
    checkpoint_dir: Path,
    resume: bool,
) -> None:
    result_path = output_dir / f"seed_{seed}" / f"{mode}.json"
    if resume and result_path.is_file():
        log(f"SKIP completed seed={seed} mode={mode}")
        return
    seed_everything(seed)
    role_feature_counts = (
        tuple(int(value) for value in arrays["role_feature_counts"])
        if "role_feature_counts" in arrays
        else tuple(
            int(arrays["feature_slot_mask"][0, role].sum())
            for role in range(arrays["feature_slot_mask"].shape[1])
        )
    )
    active_mask = arrays["active_role_mask"]
    active_roles = tuple(
        bool(active_mask[role]) if active_mask.ndim == 1
        else bool(active_mask[:, role].all())
        for role in range(active_mask.shape[-1])
    )
    model = build_external_baseline(
        mode,
        config=config,
        role_feature_counts=role_feature_counts,
        active_roles=active_roles,
        context_size=int(arrays["context"].shape[1]),
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    training = config["training"]
    optimizer_name = str(training.get("optimizer", "adamw")).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.0)),
            eps=1e-8,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    scheduler = None
    if "scheduler_gamma" in training:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(training["scheduler_gamma"])
        )
    loader = make_loader(
        train_dataset,
        training=training,
        device=device,
        shuffle=True,
        seed=seed,
    )
    best_mse = float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    curve: list[dict] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        error_sum = 0.0
        value_count = 0
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(model, batch)
            squared_error = (prediction - batch["y_future"]).square()
            valid = batch["y_future_valid"].bool()
            loss = squared_error[valid].mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for {mode}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            error_sum += float(squared_error.detach()[valid].sum())
            value_count += int(valid.sum())
            if (
                batch_index % int(training["log_every_batches"]) == 0
                or batch_index == len(loader)
            ):
                memory = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                )
                log(
                    f"seed={seed} mode={mode} epoch={epoch} "
                    f"batch={batch_index}/{len(loader)} "
                    f"train_mse={error_sum/value_count:.6f} "
                    f"gpu_mem={memory:.2f}GiB"
                )
        validation = evaluate(
            model,
            validation_dataset,
            training=training,
            device=device,
            target_scale=float(arrays["target_scale"][0]),
        )
        validation_mse = validation["metrics"]["mse_standardized"]
        if (
            scheduler is not None
            and epoch % int(training["scheduler_step_epochs"]) == 0
        ):
            scheduler.step()
        curve.append(
            {
                "epoch": epoch,
                "train_mse_standardized": error_sum / value_count,
                "validation_mse_standardized": validation_mse,
                "validation_mae_standardized": validation["metrics"][
                    "mae_standardized"
                ],
            }
        )
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            patience = 0
            atomic_save(
                checkpoint_dir / f"seed_{seed}" / mode / "best.pt",
                {
                    "model_state": best_state,
                    "mode": mode,
                    "seed": seed,
                    "epoch": epoch,
                    "validation": validation,
                    "config": config,
                },
            )
        else:
            patience += 1
        log(
            f"EPOCH seed={seed} mode={mode} epoch={epoch} "
            f"val_mse={validation_mse:.6f} best_epoch={best_epoch} "
            f"patience={patience}/{training['early_stopping_patience']}"
        )
        if (
            epoch >= int(training["minimum_epochs"])
            and patience >= int(training["early_stopping_patience"])
        ):
            log(f"EARLY-STOP seed={seed} mode={mode} epoch={epoch}")
            break
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for {mode}")
    model.load_state_dict(best_state)
    best_validation = evaluate(
        model,
        validation_dataset,
        training=training,
        device=device,
        target_scale=float(arrays["target_scale"][0]),
    )
    holdout = None
    if holdout_dataset is not None:
        holdout = evaluate(
            model,
            holdout_dataset,
            training=training,
            device=device,
            target_scale=float(arrays["target_scale"][0]),
        )
    result = {
        "status": "complete",
        "phase": phase,
        "dataset_sha256": (
            str(arrays["sha256"].item())
            if "sha256" in arrays else str(config["dataset"].get("sha256", ""))
        ),
        "smoke": phase.endswith("_smoke"),
        "holdout_evaluated": holdout is not None,
        "seed": seed,
        "mode": mode,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "epochs_completed": len(curve),
        "validation": best_validation,
        "holdout": holdout,
        "learning_curve": curve,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "input_protocol": (
            "power_history_only"
            if mode.endswith("_power")
            or mode.startswith("softs_power_")
            or mode.startswith("fouriergnn_power_")
            else "all_active_role_features_plus_historical_context"
        ),
    }
    atomic_json(result_path, result)
    log(
        f"DONE seed={seed} mode={mode} best_epoch={best_epoch} "
        f"validation_mse={best_validation['metrics']['mse_standardized']:.6f} "
        + (
            f"holdout_mse={holdout['metrics']['mse_standardized']:.6f}"
            if holdout is not None
            else "holdout=NOT_EVALUATED"
        )
    )


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    for argument, key in (
        (args.train_windows, "train_windows"),
        (args.validation_windows, "validation_windows"),
        (args.epochs, "epochs"),
        (args.num_workers, "num_workers"),
    ):
        if argument is not None:
            training[key] = argument
    device = choose_device(args.device)
    seed = int(args.seed if args.seed is not None else config["seed"])
    modes = args.modes or list(config["models"])
    unknown = set(modes).difference(config["models"])
    if unknown:
        raise ValueError(f"Unknown modes: {sorted(unknown)}")
    data_path = args.data or ROOT / config["dataset"]["path"]
    arrays = load_frozen_smarteole(
        data_path, expected_sha256=config["dataset"]["sha256"]
    )
    common = {
        "x_role": arrays["x_role"],
        "context": arrays["context"],
        "power": arrays["power"],
        "history_steps": int(config["dataset"]["history_steps"]),
        "horizon_steps": int(config["dataset"]["horizon_steps"]),
    }
    train_indices = evenly_spaced(
        arrays["train_end_indices"], int(training["train_windows"])
    )
    validation_indices = evenly_spaced(
        arrays["validation_end_indices"],
        int(training["validation_windows"]),
    )
    train_dataset = SmarteoleWindowDataset(
        **common, end_indices=train_indices
    )
    validation_dataset = SmarteoleWindowDataset(
        **common, end_indices=validation_indices
    )
    log(
        f"device={device} seed={seed} modes={modes} "
        f"train_windows={len(train_dataset)} "
        f"validation_windows={len(validation_dataset)} "
        "holdout=FORBIDDEN"
    )
    for mode in modes:
        run_mode(
            mode=mode,
            seed=seed,
            config=config,
            arrays=arrays,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            device=device,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
        )
    log("all requested validation-only baseline jobs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
