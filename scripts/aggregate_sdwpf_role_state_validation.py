#!/usr/bin/env python3
"""Aggregate five-seed SDWPF role-state validation without touching holdout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> int:
    root = Path("results/sdwpf_role_state_formal_v1")
    rows = []
    dataset_hashes = set()
    for seed in (42, 43, 44, 45, 46):
        path = root / f"seed_{seed}" / "role_state.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("phase") != "sdwpf_frozen_role_state_validation_only":
            raise RuntimeError(f"unexpected phase: {path}")
        if record.get("holdout_evaluated") is not False:
            raise RuntimeError(f"validation run accessed holdout: {path}")
        dataset_hashes.add(record["dataset_sha256"])
        rows.append({
            "seed": seed,
            "base_validation_mse": record["base_validation"]["metrics"]["mse_standardized"],
            "role_state_validation_mse": record["role_state_validation"]["metrics"]["mse_standardized"],
            "relative_validation_mse_gain": record["relative_validation_mse_gain"],
            "role_state_rms": record["role_state_validation"]["role_state_rms"],
            "best_epoch": record["best_epoch"],
            "peak_cuda_memory_gib": record["peak_cuda_memory_gib"],
            "wall_seconds": record["wall_seconds"],
        })
    if len(dataset_hashes) != 1:
        raise RuntimeError(f"inconsistent SDWPF artifacts: {dataset_hashes}")
    gains = np.asarray([row["relative_validation_mse_gain"] for row in rows])
    payload = {
        "phase": "five_seed_sdwpf_frozen_role_state_validation_only",
        "holdout_evaluated": False,
        "no_sdwpf_hyperparameter_search": True,
        "dataset_sha256": next(iter(dataset_hashes)),
        "mean_relative_validation_mse_gain": float(gains.mean()),
        "seeds_beating_same_checkpoint_base": int((gains > 0).sum()),
        "maximum_role_state_rms": max(row["role_state_rms"] for row in rows),
        "next_step": "run frozen holdout evaluator once; do not tune from validation outcome",
        "rows": rows,
    }
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"SDWPF validation gain={100*payload['mean_relative_validation_mse_gain']:.2f}% "
        f"wins={payload['seeds_beating_same_checkpoint_base']}/5 "
        f"rms={payload['maximum_role_state_rms']:.4f}"
    )
    print("HOLDOUT WAS NOT EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
