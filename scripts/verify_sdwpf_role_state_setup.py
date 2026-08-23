#!/usr/bin/env python3
"""Preflight checks for the frozen SDWPF role-state experiment."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from maspge.data import load_frozen_sdwpf, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config_path = ROOT / "configs/sdwpf_role_state_formal_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = ROOT / config["dataset"]["path"]
    arrays = load_frozen_sdwpf(data_path)
    dataset_id = str(arrays["dataset_id"].item())
    if dataset_id != config["dataset"]["dataset_id"]:
        raise RuntimeError(f"dataset mismatch: {dataset_id}")
    if arrays["x_role"].shape != (35280, 134, 4, 6):
        raise RuntimeError(f"unexpected dense role tensor: {arrays['x_role'].shape}")
    if len(arrays["train_end_indices"]) < config["data_loading"]["train_windows"]:
        raise RuntimeError("insufficient training windows")
    if len(arrays["validation_end_indices"]) < config["data_loading"]["validation_windows"]:
        raise RuntimeError("insufficient validation windows")
    if len(arrays["holdout_end_indices"]) < config["data_loading"]["holdout_windows"]:
        raise RuntimeError("insufficient holdout windows")

    base_root = ROOT / config["base_checkpoint_root"]
    for seed in config["seeds"]:
        path = base_root / f"seed_{seed}" / "finetune_best.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("stage") != "finetune" or "model_state" not in payload:
            raise RuntimeError(f"invalid validation-selected base checkpoint: {path}")
        print(f"BASE_OK seed={seed} sha256={sha256_file(path)[:12]}")

    print(f"DATASET_OK id={dataset_id} sha256={sha256_file(data_path)}")
    print(f"x_role={arrays['x_role'].shape} power={arrays['power'].shape}")
    print(
        "windows train/validation/holdout="
        f"{len(arrays['train_end_indices'])}/"
        f"{len(arrays['validation_end_indices'])}/"
        f"{len(arrays['holdout_end_indices'])}"
    )
    print("PREFLIGHT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
