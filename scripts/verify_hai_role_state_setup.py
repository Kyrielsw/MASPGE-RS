#!/usr/bin/env python3
"""Preflight checks for the frozen HAI role-state data contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from maspge.data import load_frozen_hai, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config_path = ROOT / "configs/hai_23_05_role_state_contract_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = ROOT / config["output"]
    audit_path = ROOT / config["output_audit"]
    arrays = load_frozen_hai(data_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["model_training_performed"] or audit["holdout_model_evaluated"]:
        raise RuntimeError("HAI preprocessing audit violates the frozen protocol")
    if audit["x_role_shape"] != list(arrays["x_role"].shape):
        raise RuntimeError("HAI processed audit does not match the NPZ")
    if not np.isfinite(arrays["x_role"]).all():
        raise RuntimeError("HAI role tensor contains non-finite values")
    if not np.isfinite(arrays["power"]).all():
        raise RuntimeError("HAI power tensor contains non-finite values")
    print(f"DATASET_OK id={arrays['dataset_id'].item()} sha256={sha256_file(data_path)}")
    print(f"x_role={arrays['x_role'].shape} power={arrays['power'].shape}")
    print(f"units={arrays['unit_names'].tolist()} active_roles=A1+A2+A3+A4")
    print(
        "windows train/validation/holdout="
        f"{len(arrays['train_end_indices'])}/"
        f"{len(arrays['validation_end_indices'])}/"
        f"{len(arrays['holdout_end_indices'])}"
    )
    print(f"target_valid_fraction={arrays['power_valid'].mean(axis=0).tolist()}")
    print("HAI_ROLE_STATE_SETUP_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
