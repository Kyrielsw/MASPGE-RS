#!/usr/bin/env python3
"""Aggregate frozen HAI validation runs without touching holdout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> int:
    root = Path("results/hai_role_state_formal_v1")
    rows = []
    dataset_hashes = set()
    for seed in (42, 43, 44, 45, 46):
        path = root / f"seed_{seed}" / "role_state.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("phase") != "hai_strict_validation_only":
            raise RuntimeError(f"unexpected phase: {path}")
        if record.get("holdout_constructed") or record.get("holdout_evaluated"):
            raise RuntimeError(f"HAI validation accessed holdout: {path}")
        dataset_hashes.add(record["dataset_sha256"])
        base = record["base"]["finetune"]["validation"]
        role = record["role_state"]["validation"]
        rows.append({
            "seed": seed,
            "base_validation_mse": base["metrics"]["mse_standardized"],
            "role_state_validation_mse": role["metrics"]["mse_standardized"],
            "base_per_unit_mse": base["per_unit_mse_standardized"],
            "role_state_per_unit_mse": role["per_unit_mse_standardized"],
            "relative_validation_mse_gain": record["relative_validation_mse_gain"],
            "role_state_rms": role["role_state_rms"],
            "best_epoch": record["role_state"]["best_epoch"],
            "wall_seconds": record["wall_seconds"],
        })
    if len(dataset_hashes) != 1:
        raise RuntimeError(f"inconsistent HAI artifacts: {dataset_hashes}")
    gains = np.asarray([row["relative_validation_mse_gain"] for row in rows])
    wins = int((gains > 0).sum())
    maximum_rms = max(row["role_state_rms"] for row in rows)
    accepted = bool(wins >= 4 and gains.mean() >= 0.01 and maximum_rms <= 1.0)
    payload = {
        "phase": "five_seed_hai_strict_validation_only",
        "holdout_constructed": False,
        "holdout_evaluated": False,
        "hai_hyperparameter_search": False,
        "dataset_sha256": next(iter(dataset_hashes)),
        "mean_relative_validation_mse_gain": float(gains.mean()),
        "seeds_beating_same_seed_mafs": wins,
        "maximum_role_state_rms": maximum_rms,
        "accepted_for_strict_frozen_holdout": accepted,
        "acceptance_rule": "wins>=4/5, mean MSE gain>=1%, max role-state RMS<=1",
        "rows": rows,
    }
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"HAI validation gain={100*payload['mean_relative_validation_mse_gain']:.2f}% "
        f"wins={wins}/5 rms={maximum_rms:.4f} accepted={accepted}"
    )
    print("HOLDOUT WAS NOT CONSTRUCTED OR EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
