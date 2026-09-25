#!/usr/bin/env python3
"""Static data/protocol audit before any input-matched training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_external_baseline_screen import ROOT, atomic_json
from run_itransformer_input_matched import load_arrays, role_counts, sha256


def main() -> int:
    config_path = ROOT / "configs/itransformer_input_matched_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = []
    for name, profile in config["datasets"].items():
        path = ROOT / profile["path"]
        arrays = load_arrays(profile, path)
        counts = role_counts(arrays)
        assert arrays["x_role"].shape[0] == arrays["power"].shape[0] == arrays["context"].shape[0]
        assert arrays["x_role"].shape[1] == arrays["power"].shape[1]
        # Frozen arrays are [time, unit, storage-slot, feature].  The history
        # window and batch axes are introduced later by Dataset/DataLoader.
        assert arrays["x_role"].shape[2] == len(counts)
        assert np.isfinite(arrays["x_role"]).all() and np.isfinite(arrays["power"]).all()
        for split in ("train", "validation", "holdout"):
            indices = arrays[f"{split}_end_indices"]
            assert len(indices) == int(profile[f"{split}_windows"])
            assert int(indices.min()) >= 59
            assert int(indices.max()) + 15 < len(arrays["power"])
        assert int(arrays["train_end_indices"].max()) < int(arrays["validation_end_indices"].min())
        assert int(arrays["validation_end_indices"].max()) < int(arrays["holdout_end_indices"].min())
        datasets.append({
            "dataset": name, "dataset_sha256": sha256(path),
            "time_steps": int(arrays["power"].shape[0]), "unit_count": int(arrays["power"].shape[1]),
            "x_role_shape": list(arrays["x_role"].shape), "context_size_present_but_not_consumed": int(arrays["context"].shape[1]),
            "generic_sensor_feature_count": int(sum(counts)), "storage_slot_counts": list(counts),
            "windows": {split: int(len(arrays[f"{split}_end_indices"])) for split in ("train", "validation", "holdout")},
            "power_valid_mask_present": "power_valid" in arrays,
        })
    payload = {
        "status": "PASS", "training_performed": False, "holdout_evaluated": False,
        "old_itransformer_fullinfo_reusable": False,
        "old_baseline_obstacles": ["A1-A4-aware slot activation", "loader context concatenation not used by current Generic Unit State", "single horizon-mean output", "no complete historical-target sequence in token"],
        "frozen_replacement": config["method_label"],
        "input_boundary": config["data_contract"], "datasets": datasets,
    }
    output = ROOT / "results/itransformer_input_matched_audit_v1/audit.json"
    atomic_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
