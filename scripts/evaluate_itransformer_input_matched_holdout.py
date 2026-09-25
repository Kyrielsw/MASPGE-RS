#!/usr/bin/env python3
"""Read-only holdout evaluation after all input-matched validation jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from maspge.baselines import ITransformerInputMatchedAdapted
from run_external_baseline_screen import ROOT, atomic_json, choose_device, evenly_spaced, log
from run_itransformer_input_matched import (
    build_dataset, evaluate, load_arrays, role_counts, sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/itransformer_input_matched_v1.json")
    parser.add_argument("--dataset", choices=["smarteole", "sdwpf", "hai", "xai4heat"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--validation-dir", type=Path, default=ROOT / "results/itransformer_input_matched_validation_v1")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/itransformer_input_matched_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/itransformer_input_matched_holdout_v1")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    # Gate prevents piecemeal holdout access before the complete validation matrix.
    missing = [
        str(args.validation_dir / dataset / f"seed_{seed}.json")
        for dataset in config["datasets"] for seed in config["seeds"]
        if not (args.validation_dir / dataset / f"seed_{seed}.json").is_file()
    ]
    if missing:
        raise RuntimeError(f"holdout gate closed; {len(missing)} validation results missing")
    output = args.output_dir / args.dataset / f"seed_{args.seed}.json"
    if args.resume and output.is_file():
        log(f"SKIP completed holdout dataset={args.dataset} seed={args.seed}")
        return 0
    profile = config["datasets"][args.dataset]
    arrays = load_arrays(profile, ROOT / profile["path"])
    history = int(config["data_contract"]["history_steps"])
    horizon = int(config["data_contract"]["horizon_steps"])
    holdout = build_dataset(
        arrays,
        evenly_spaced(arrays["holdout_end_indices"], int(profile["holdout_windows"])),
        history, horizon,
    )
    model_cfg = config["model"]
    model = ITransformerInputMatchedAdapted(
        history_steps=history, horizon_steps=horizon,
        role_feature_counts=role_counts(arrays), d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]), layers=int(model_cfg["layers"]),
        feedforward_size=int(model_cfg["feedforward_size"]), dropout=float(model_cfg["dropout"]),
    )
    checkpoint_path = args.checkpoint_dir / args.dataset / f"seed_{args.seed}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = choose_device(args.device); model.to(device)
    loading = dict(profile)
    result = evaluate(model, holdout, loading=loading, device=device, target_scale=float(arrays["target_scale"][0]))
    validation_record = json.loads((args.validation_dir / args.dataset / f"seed_{args.seed}.json").read_text(encoding="utf-8"))
    payload = {
        "phase": "frozen_post_hoc_holdout_report", "training_performed": False,
        "holdout_evaluated": True, "not_a_new_blind_test": True,
        "dataset": args.dataset, "seed": args.seed,
        "dataset_sha256": sha256(ROOT / profile["path"]),
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "validation_best_epoch": validation_record["best_epoch"],
        "parameter_count": validation_record["parameter_count"], "holdout": result,
    }
    atomic_json(output, payload)
    log(f"DONE HOLDOUT dataset={args.dataset} seed={args.seed} mse={result['metrics']['mse_standardized']:.6f} training=FORBIDDEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
