#!/usr/bin/env python3
"""Aggregate five-seed same-input generic-state validation controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("smarteole", "sdwpf", "hai")
SEEDS = (42, 43, 44, 45, 46)


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def main() -> int:
    summaries = []
    all_rows = []
    root = ROOT / "results/generic_state_control_validation_v1"
    for dataset in DATASETS:
        rows = []
        for seed in SEEDS:
            path = root / dataset / f"seed_{seed}" / "result.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["holdout_evaluated"]:
                raise RuntimeError(f"holdout leakage: {path}")
            if not record["same_sensor_values"] or record["role_specific_encoders"]:
                raise RuntimeError(f"invalid generic-state protocol: {path}")
            if abs(float(record["parameter_relative_difference"])) >= 0.01:
                raise RuntimeError(f"parameter mismatch exceeds one percent: {path}")
            base = record["base_validation"]["metrics"]["mse_standardized"]
            generic = record["generic_state_validation"]["metrics"]["mse_standardized"]
            semantic = record["semantic_validation"]["metrics"]["mse_standardized"]
            base_mae = record["base_validation"]["metrics"]["mae_standardized"]
            generic_mae = record["generic_state_validation"]["metrics"]["mae_standardized"]
            semantic_mae = record["semantic_validation"]["metrics"]["mae_standardized"]
            row = {
                "dataset": dataset,
                "seed": seed,
                "base_mse": base,
                "generic_state_mse": generic,
                "maspge_rs_mse": semantic,
                "base_mae": base_mae,
                "generic_state_mae": generic_mae,
                "maspge_rs_mae": semantic_mae,
                "generic_gain_vs_base": (base - generic) / base,
                "maspge_gain_vs_generic": (generic - semantic) / generic,
                "generic_parameters": int(record["generic_adapter_parameters"]),
                "maspge_parameters": int(record["semantic_adapter_parameters"]),
                "generic_best_epoch": int(record["generic_best_epoch"]),
            }
            rows.append(row)
            all_rows.append(row)
        generic_gains = [row["generic_gain_vs_base"] for row in rows]
        semantic_gains = [row["maspge_gain_vs_generic"] for row in rows]
        summary = {
            "dataset": dataset,
            "mafs_base_mse": mean_std([row["base_mse"] for row in rows]),
            "generic_unit_state_mse": mean_std(
                [row["generic_state_mse"] for row in rows]
            ),
            "maspge_rs_mse": mean_std([row["maspge_rs_mse"] for row in rows]),
            "mafs_base_mae": mean_std([row["base_mae"] for row in rows]),
            "generic_unit_state_mae": mean_std(
                [row["generic_state_mae"] for row in rows]
            ),
            "maspge_rs_mae": mean_std([row["maspge_rs_mae"] for row in rows]),
            "mean_paired_generic_gain_vs_base": float(np.mean(generic_gains)),
            "generic_wins_vs_base": sum(gain > 0 for gain in generic_gains),
            "mean_paired_maspge_gain_vs_generic": float(np.mean(semantic_gains)),
            "maspge_wins_vs_generic": sum(gain > 0 for gain in semantic_gains),
            "parameter_pair": {
                "generic": sorted({row["generic_parameters"] for row in rows}),
                "maspge_rs": sorted({row["maspge_parameters"] for row in rows}),
            },
            "rows": rows,
        }
        summaries.append(summary)
        print(
            f"{dataset:<10} "
            f"base={summary['mafs_base_mse']['mean']:.6f}±{summary['mafs_base_mse']['std']:.6f} "
            f"generic={summary['generic_unit_state_mse']['mean']:.6f}±{summary['generic_unit_state_mse']['std']:.6f} "
            f"maspge={summary['maspge_rs_mse']['mean']:.6f}±{summary['maspge_rs_mse']['std']:.6f} "
            f"generic_gain={100*summary['mean_paired_generic_gain_vs_base']:.2f}% "
            f"maspge_gain={100*summary['mean_paired_maspge_gain_vs_generic']:.2f}% "
            f"wins={summary['maspge_wins_vs_generic']}/5"
        )
    payload = {
        "phase": "five_seed_three_dataset_validation_only_same_input_generic_state_control",
        "holdout_evaluated": False,
        "model_selection_from_holdout": False,
        "standard_deviation": "sample standard deviation across five model seeds, ddof=1",
        "primary_comparison": "MASPGE-RS versus same-input generic per-unit state adapter",
        "outcome_policy": "report all datasets; decide manuscript positioning before any frozen test evaluation",
        "datasets": summaries,
        "rows": all_rows,
    }
    output = root / "summary/generic_state_control_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {output}")
    print("HOLDOUT WAS NOT EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
