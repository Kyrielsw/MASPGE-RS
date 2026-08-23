#!/usr/bin/env python3
"""Aggregate Experiment 041 five-seed full-data validation results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> int:
    root = Path("results/role_state_v1_full_validation")
    rows = []
    for seed in (42, 43, 44, 45, 46):
        path = root / f"seed_{seed}" / "role_state.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("holdout_evaluated") is not False:
            raise RuntimeError(f"full validation accessed holdout: {path}")
        if record.get("phase") != "five_seed_full_training_validation_only_role_state":
            raise RuntimeError(f"unexpected experiment phase: {path}")
        rows.append({
            "seed": seed,
            "base_validation_mse": record["base_validation"]["metrics"]["mse_standardized"],
            "role_state_validation_mse": record["validation"]["metrics"]["mse_standardized"],
            "relative_validation_mse_gain": record["relative_validation_mse_gain"],
            "role_state_rms": record["validation"]["role_state_rms"],
            "best_epoch": record["best_epoch"],
            "wall_seconds": record["wall_seconds"],
        })
    gains = np.asarray([row["relative_validation_mse_gain"] for row in rows])
    wins = int((gains > 0).sum())
    maximum_rms = max(row["role_state_rms"] for row in rows)
    accepted = wins >= 4 and float(gains.mean()) >= 0.01 and maximum_rms <= 1.0
    payload = {
        "phase": "five_seed_full_training_validation_only_role_state",
        "holdout_evaluated": False,
        "training_windows": 81188,
        "validation_windows": 17946,
        "mean_relative_validation_mse_gain": float(gains.mean()),
        "seeds_beating_same_checkpoint_base": wins,
        "maximum_role_state_rms": maximum_rms,
        "accepted_for_frozen_test_evaluation": accepted,
        "acceptance_rule": "wins>=4/5, mean validation MSE gain>=1%, max role-state RMS<=1",
        "rows": rows,
    }
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "full_validation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"role-state full validation gain={100*payload['mean_relative_validation_mse_gain']:.2f}% "
        f"wins={wins}/5 rms={maximum_rms:.4f} accepted={accepted}"
    )
    print("HOLDOUT WAS NOT EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
