#!/usr/bin/env python3
"""Verify software, GPUs, and the frozen dataset before training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from maspge.data import load_frozen_smarteole


def main() -> int:
    config_path = ROOT / "configs/smarteole_role_state_full_validation_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = ROOT / config["dataset"]["path"]
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {properties.name}, "
            f"VRAM={properties.total_memory / 1024**3:.1f} GiB"
        )
    arrays = load_frozen_smarteole(
        data_path, expected_sha256=config["dataset"]["sha256"]
    )
    print(f"Dataset: {data_path}")
    print(f"x_role: {arrays['x_role'].shape} {arrays['x_role'].dtype}")
    print(f"context: {arrays['context'].shape}")
    print(f"power: {arrays['power'].shape}")
    print(
        "windows train/validation/holdout: "
        f"{len(arrays['train_end_indices'])}/"
        f"{len(arrays['validation_end_indices'])}/"
        f"{len(arrays['holdout_end_indices'])}"
    )
    finite = all(
        np.isfinite(arrays[name]).all()
        for name in ("x_role", "context", "power")
    )
    print(f"Core arrays finite: {finite}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("WARNING: formal 2-GPU launcher expects two CUDA GPUs")
    if not finite:
        raise RuntimeError("Dataset contains non-finite values")
    print("SETUP_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
