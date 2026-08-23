#!/usr/bin/env python3
"""Aggregate the one-shot frozen SDWPF role-state holdout report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def main() -> int:
    root = Path("results/sdwpf_role_state_holdout_v1")
    rows = []
    hashes = set()
    for seed in (42, 43, 44, 45, 46):
        path = root / f"seed_{seed}" / "holdout_evaluation.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("training_performed") is not False:
            raise RuntimeError(f"holdout evaluator performed training: {path}")
        hashes.add(record["dataset_sha256"])
        base = record["base_holdout"]
        role = record["role_state_holdout"]
        rows.append({
            "seed": seed,
            "base_holdout_mse": base["metrics"]["mse_standardized"],
            "role_state_holdout_mse": role["metrics"]["mse_standardized"],
            "base_holdout_mae": base["metrics"]["mae_standardized"],
            "role_state_holdout_mae": role["metrics"]["mae_standardized"],
            "base_holdout_rmse_physical": base["metrics"]["rmse_physical"],
            "role_state_holdout_rmse_physical": role["metrics"]["rmse_physical"],
            "relative_holdout_mse_gain": record["relative_holdout_mse_gain"],
            "units_improved": sum(
                new < old for new, old in zip(
                    role["per_unit_mse_standardized"],
                    base["per_unit_mse_standardized"],
                )
            ),
            "role_state_rms": role["role_state_rms"],
            "wall_seconds": record["wall_seconds"],
        })
    if len(hashes) != 1:
        raise RuntimeError(f"inconsistent SDWPF artifacts: {hashes}")
    base = np.asarray([row["base_holdout_mse"] for row in rows])
    role = np.asarray([row["role_state_holdout_mse"] for row in rows])
    gains = np.asarray([row["relative_holdout_mse_gain"] for row in rows])
    summary = {
        "phase": "five_seed_sdwpf_frozen_role_state_cross_dataset_report",
        "holdout_evaluated": True,
        "training_performed": False,
        "no_sdwpf_hyperparameter_search": True,
        "sdwpf_holdout_previously_observed_by_old_router": True,
        "eligible_as_strict_unseen_test": False,
        "dataset_sha256": next(iter(hashes)),
        "base_holdout_mse_mean": float(base.mean()),
        "base_holdout_mse_std": float(base.std(ddof=1)),
        "role_state_holdout_mse_mean": float(role.mean()),
        "role_state_holdout_mse_std": float(role.std(ddof=1)),
        "mean_paired_relative_mse_gain": float(gains.mean()),
        "seeds_beating_same_checkpoint_base": int((role < base).sum()),
        "maximum_role_state_rms": max(row["role_state_rms"] for row in rows),
        "decision_policy": "report_only_no_post_holdout_tuning",
        "rows": rows,
    }
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_seed_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "holdout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"SDWPF MAFS Base holdout_mse={summary['base_holdout_mse_mean']:.6f}"
        f"±{summary['base_holdout_mse_std']:.6f}"
    )
    print(
        f"SDWPF Role-State holdout_mse={summary['role_state_holdout_mse_mean']:.6f}"
        f"±{summary['role_state_holdout_mse_std']:.6f} "
        f"gain={100*summary['mean_paired_relative_mse_gain']:.2f}% "
        f"wins={summary['seeds_beating_same_checkpoint_base']}/5"
    )
    print("REPORT ONLY; NO POST-HOLDOUT TUNING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
