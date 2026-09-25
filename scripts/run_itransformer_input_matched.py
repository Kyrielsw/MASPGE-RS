#!/usr/bin/env python3
"""Train the frozen same-input adapted iTransformer on validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from maspge.baselines import ITransformerInputMatchedAdapted
from maspge.data import (
    SmarteoleWindowDataset, load_frozen_hai, load_frozen_sdwpf,
    load_frozen_smarteole, load_frozen_xai4heat,
)
from maspge.metrics import regression_metrics
from run_external_baseline_screen import (
    ROOT, atomic_json, atomic_save, choose_device, evenly_spaced, log,
    make_loader, move_batch, seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/itransformer_input_matched_v1.json")
    parser.add_argument("--dataset", choices=["smarteole", "sdwpf", "hai", "xai4heat"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/itransformer_input_matched_validation_v1")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/itransformer_input_matched_v1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_arrays(profile: dict, path: Path) -> dict:
    kwargs = {"expected_sha256": profile["sha256"]}
    return {
        "smarteole": load_frozen_smarteole,
        "sdwpf": load_frozen_sdwpf,
        "hai": load_frozen_hai,
        "xai4heat": load_frozen_xai4heat,
    }[profile["loader"]](path, **kwargs)


def role_counts(arrays: dict) -> tuple[int, ...]:
    if "role_feature_counts" in arrays:
        return tuple(int(v) for v in arrays["role_feature_counts"])
    mask = arrays["feature_slot_mask"]
    return tuple(int(mask[0, role].sum()) for role in range(mask.shape[1]))


def build_dataset(arrays: dict, indices, history: int, horizon: int) -> SmarteoleWindowDataset:
    kwargs = dict(
        x_role=arrays["x_role"], context=arrays["context"], power=arrays["power"],
        end_indices=indices, history_steps=history, horizon_steps=horizon,
    )
    if "power_valid" in arrays:
        kwargs["power_valid"] = arrays["power_valid"]
    return SmarteoleWindowDataset(**kwargs)


def evaluate(model, dataset, *, loading: dict, device: torch.device, target_scale: float) -> dict:
    loader = make_loader(dataset, training=loading, device=device, shuffle=False, seed=0)
    predictions, targets, valid_means = [], [], []
    seq_sse = seq_sae = 0.0
    seq_count = 0
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = move_batch(raw, device)
            sequence = model(x_role=batch["x_role"], power_history=batch["power_history"])
            valid = batch["power_future_valid"].bool()
            counts = valid.sum(dim=1).clamp_min(1)
            predictions.append(((sequence * valid).sum(dim=1) / counts).cpu())
            targets.append(batch["y_future"].cpu())
            valid_means.append(batch["y_future_valid"].cpu())
            error = sequence - batch["power_future_sequence"]
            seq_sse += float(error.square()[valid].sum())
            seq_sae += float(error.abs()[valid].sum())
            seq_count += int(valid.sum())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    valid_mean = torch.cat(valid_means).bool()
    return {
        "metrics": regression_metrics(prediction, target, target_scale=target_scale, valid_mask=valid_mean),
        "sequence_mse_standardized": seq_sse / seq_count,
        "sequence_mae_standardized": seq_sae / seq_count,
        "prediction_shape": list(prediction.shape),
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed is outside the frozen protocol")
    profile = dict(config["datasets"][args.dataset])
    history = int(config["data_contract"]["history_steps"])
    horizon = int(config["data_contract"]["horizon_steps"])
    arrays = load_arrays(profile, ROOT / profile["path"])
    train_n = 64 if args.smoke else int(profile["train_windows"])
    val_n = 32 if args.smoke else int(profile["validation_windows"])
    train = build_dataset(arrays, evenly_spaced(arrays["train_end_indices"], train_n), history, horizon)
    validation = build_dataset(arrays, evenly_spaced(arrays["validation_end_indices"], val_n), history, horizon)
    counts = role_counts(arrays)
    model_cfg = config["model"]
    device = choose_device(args.device)
    seed_everything(args.seed)
    model = ITransformerInputMatchedAdapted(
        history_steps=history, horizon_steps=horizon, role_feature_counts=counts,
        d_model=int(model_cfg["d_model"]), n_heads=int(model_cfg["n_heads"]),
        layers=int(model_cfg["layers"]), feedforward_size=int(model_cfg["feedforward_size"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    result_path = args.output_dir / args.dataset / f"seed_{args.seed}.json"
    if args.resume and result_path.is_file():
        log(f"SKIP completed dataset={args.dataset} seed={args.seed}")
        return 0
    training = dict(config["training"])
    if args.smoke:
        training.update({"epochs": 1, "minimum_epochs": 1, "early_stopping_patience": 1})
    loading = dict(profile)
    loading.update({"num_workers": 0 if args.smoke else profile["num_workers"]})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    loader = make_loader(train, training=loading, device=device, shuffle=True, seed=args.seed)
    best_mse, best_epoch, best_state, best_validation = float("inf"), 0, None, None
    patience, curve = 0, []
    started = time.perf_counter()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train(); total = 0.0; count = 0
        for batch_index, raw in enumerate(loader, 1):
            batch = move_batch(raw, device); optimizer.zero_grad(set_to_none=True)
            prediction = model(x_role=batch["x_role"], power_history=batch["power_history"])
            valid = batch["power_future_valid"].bool()
            error = (prediction - batch["power_future_sequence"]).square()
            loss = error[valid].mean()
            if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"])); optimizer.step()
            total += float(error.detach()[valid].sum()); count += int(valid.sum())
            if batch_index % int(profile["log_every_batches"]) == 0 or batch_index == len(loader):
                log(f"dataset={args.dataset} seed={args.seed} epoch={epoch} batch={batch_index}/{len(loader)} train_sequence_mse={total/count:.6f}")
        validation_result = evaluate(model, validation, loading=loading, device=device, target_scale=float(arrays["target_scale"][0]))
        mse = validation_result["metrics"]["mse_standardized"]
        curve.append({"epoch": epoch, "train_sequence_mse_standardized": total/count, "validation_mse_standardized": mse, "validation_mae_standardized": validation_result["metrics"]["mae_standardized"], "validation_sequence_mse_standardized": validation_result["sequence_mse_standardized"]})
        if mse < best_mse:
            best_mse, best_epoch, best_validation, patience = mse, epoch, validation_result, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_save(args.checkpoint_dir / args.dataset / f"seed_{args.seed}" / "best.pt", {"model_state": best_state, "epoch": epoch, "validation": validation_result, "config": config, "dataset": args.dataset, "seed": args.seed})
        else: patience += 1
        log(f"EPOCH dataset={args.dataset} seed={args.seed} epoch={epoch} val_mse={mse:.6f} best_epoch={best_epoch} patience={patience}/{training['early_stopping_patience']}")
        if epoch >= int(training["minimum_epochs"]) and patience >= int(training["early_stopping_patience"]):
            log(f"EARLY-STOP dataset={args.dataset} seed={args.seed} epoch={epoch}"); break
    if best_state is None: raise RuntimeError("no finite checkpoint")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception: commit = None
    critical = hashlib.sha256((Path(__file__).read_bytes() + (ROOT / "src/maspge/baselines.py").read_bytes() + args.config.read_bytes())).hexdigest()
    payload = {
        "status": "complete", "phase": "smoke" if args.smoke else "validation_only",
        "holdout_evaluated": False,
        "dataset": args.dataset, "seed": args.seed, "dataset_sha256": sha256(ROOT / profile["path"]),
        "git_commit": commit, "critical_code_and_config_sha256": critical,
        "input_protocol": "60-step target history plus generic flattened per-unit sensors; no future covariates; loader context excluded symmetrically",
        "role_semantics_used": False, "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "effective_batch_size": int(profile["batch_size"]), "gradient_accumulation_steps": 1,
        "best_epoch": best_epoch, "epochs_completed": len(curve), "validation": best_validation,
        "learning_curve": curve, "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0,
    }
    atomic_json(result_path, payload)
    log(f"DONE dataset={args.dataset} seed={args.seed} validation_mse={best_mse:.6f} holdout=NOT_EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
