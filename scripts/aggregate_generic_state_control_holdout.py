#!/usr/bin/env python3
"""Aggregate frozen same-input generic-state holdout reports."""

from __future__ import annotations

import csv
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
    root = ROOT / "results/generic_state_control_holdout_v1"
    summaries = []
    all_rows = []
    for dataset in DATASETS:
        rows = []
        for seed in SEEDS:
            path = root / dataset / f"seed_{seed}" / "holdout.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("training_performed") is not False:
                raise RuntimeError(f"training detected in holdout report: {path}")
            if record.get("post_holdout_tuning_permitted") is not False:
                raise RuntimeError(f"post-holdout tuning permitted: {path}")
            if not record.get("same_sensor_values_generic_and_maspge"):
                raise RuntimeError(f"same-input protocol violated: {path}")
            metrics = {
                name: record[key]["metrics"]
                for name, key in (
                    ("base", "mafs_base_holdout"),
                    ("generic", "generic_unit_state_holdout"),
                    ("maspge", "maspge_rs_holdout"),
                )
            }
            row = {
                "dataset": dataset,
                "seed": seed,
                "base_mse": metrics["base"]["mse_standardized"],
                "generic_mse": metrics["generic"]["mse_standardized"],
                "maspge_mse": metrics["maspge"]["mse_standardized"],
                "base_mae": metrics["base"]["mae_standardized"],
                "generic_mae": metrics["generic"]["mae_standardized"],
                "maspge_mae": metrics["maspge"]["mae_standardized"],
                "generic_gain_vs_base": record["generic_gain_vs_base"],
                "maspge_gain_vs_generic": record["maspge_gain_vs_generic"],
                "maspge_gain_vs_base": record["maspge_gain_vs_base"],
            }
            rows.append(row)
            all_rows.append(row)
        summary = {
            "dataset": dataset,
            "mafs_base_mse": mean_std([row["base_mse"] for row in rows]),
            "generic_unit_state_mse": mean_std(
                [row["generic_mse"] for row in rows]
            ),
            "maspge_rs_mse": mean_std([row["maspge_mse"] for row in rows]),
            "mafs_base_mae": mean_std([row["base_mae"] for row in rows]),
            "generic_unit_state_mae": mean_std(
                [row["generic_mae"] for row in rows]
            ),
            "maspge_rs_mae": mean_std([row["maspge_mae"] for row in rows]),
            "mean_paired_generic_gain_vs_base": float(
                np.mean([row["generic_gain_vs_base"] for row in rows])
            ),
            "generic_wins_vs_base": sum(
                row["generic_mse"] < row["base_mse"] for row in rows
            ),
            "mean_paired_maspge_gain_vs_generic": float(
                np.mean([row["maspge_gain_vs_generic"] for row in rows])
            ),
            "maspge_wins_vs_generic": sum(
                row["maspge_mse"] < row["generic_mse"] for row in rows
            ),
            "mean_paired_maspge_gain_vs_base": float(
                np.mean([row["maspge_gain_vs_base"] for row in rows])
            ),
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
        "phase": "frozen_three_dataset_same_input_generic_state_holdout_report",
        "holdout_evaluated": True,
        "training_performed": False,
        "post_holdout_tuning_permitted": False,
        "study_level_holdout_previously_observed": True,
        "eligible_as_strict_unseen_test": False,
        "standard_deviation": "sample standard deviation across five model seeds, ddof=1",
        "decision_policy": "report all outcomes without further tuning",
        "datasets": summaries,
        "rows": all_rows,
    }
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_seed_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = output / "holdout_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {summary_path}")
    print("FROZEN REPORT; TRAINING AND POST-HOLDOUT TUNING FORBIDDEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
