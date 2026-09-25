#!/usr/bin/env python3
"""Five-seed full-data external baselines with one-shot holdout evaluation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from run_external_baseline_screen import (
    ROOT,
    SmarteoleWindowDataset,
    choose_device,
    evenly_spaced,
    load_frozen_smarteole,
    log,
    run_mode,
)
from maspge.data import load_frozen_hai, load_frozen_sdwpf, load_frozen_xai4heat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/smarteole_paper_external_formal_v1.json",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/paper_external_formal_v1"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=ROOT / "checkpoints/paper_external_formal_v1"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.mode not in config["models"]:
        raise ValueError(f"Mode is not frozen for formal evaluation: {args.mode}")
    if args.seed not in config["seeds"]:
        raise ValueError(f"Seed is not frozen for formal evaluation: {args.seed}")
    mode_config = copy.deepcopy(config)
    loading = dict(config["data_loading"])
    training = dict(config["training_by_mode"][args.mode])
    training.update(
        {
            "train_windows": loading["train_windows"],
            "validation_windows": loading["validation_windows"],
            "batch_size": loading["batch_size"],
            "num_workers": loading["num_workers"],
            "log_every_batches": loading["log_every_batches"],
        }
    )
    if args.smoke:
        training.update(
            {"train_windows": 64, "validation_windows": 32, "epochs": 1,
             "minimum_epochs": 1, "early_stopping_patience": 1, "num_workers": 0}
        )
        loading["holdout_windows"] = 32
    mode_config["training"] = training
    device = choose_device(args.device)
    data_path = args.data or ROOT / config["dataset"]["path"]
    loader_name = config["dataset"].get("loader")
    if loader_name == "sdwpf":
        arrays = load_frozen_sdwpf(
            data_path, expected_sha256=config["dataset"].get("sha256")
        )
    elif loader_name == "hai":
        arrays = load_frozen_hai(
            data_path, expected_sha256=config["dataset"].get("sha256")
        )
    elif loader_name == "xai4heat":
        arrays = load_frozen_xai4heat(
            data_path, expected_sha256=config["dataset"].get("sha256")
        )
    else:
        arrays = load_frozen_smarteole(
            data_path, expected_sha256=config["dataset"]["sha256"]
        )
    common = {
        "x_role": arrays["x_role"], "context": arrays["context"], "power": arrays["power"],
        "history_steps": int(config["dataset"]["history_steps"]),
        "horizon_steps": int(config["dataset"]["horizon_steps"]),
    }
    if "power_valid" in arrays:
        common["power_valid"] = arrays["power_valid"]
    train = SmarteoleWindowDataset(
        **common,
        end_indices=evenly_spaced(arrays["train_end_indices"], int(training["train_windows"])),
    )
    validation = SmarteoleWindowDataset(
        **common,
        end_indices=evenly_spaced(arrays["validation_end_indices"], int(training["validation_windows"])),
    )
    holdout = None if args.validation_only else SmarteoleWindowDataset(
        **common,
        end_indices=evenly_spaced(
            arrays["holdout_end_indices"], int(loading["holdout_windows"])
        ),
    )
    log(
        f"FORMAL device={device} seed={args.seed} mode={args.mode} "
        f"train={len(train)} validation={len(validation)} "
        f"holdout={'NOT_CONSTRUCTED' if holdout is None else len(holdout)}"
    )
    run_mode(
        mode=args.mode, seed=args.seed, config=mode_config, arrays=arrays,
        train_dataset=train, validation_dataset=validation, holdout_dataset=holdout,
        phase=(
            f"{loader_name or 'smarteole'}_external_validation_only"
            if args.validation_only
            else "paper_external_five_seed_formal"
        ) + ("_smoke" if args.smoke else ""),
        device=device, output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
