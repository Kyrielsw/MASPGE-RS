#!/usr/bin/env python3
"""Aggregate the frozen validation-only structural ablation matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from run_mafs_adapted_screen import ROOT, atomic_json


SEEDS = (42, 43, 44, 45, 46)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def main() -> int:
    sm_full = read_json(
        ROOT / "results/role_state_v1_full_validation/summary/"
        "full_validation_summary.json"
    )
    hai_full = read_json(
        ROOT / "results/hai_role_state_formal_v1/summary/validation_summary.json"
    )
    rows: list[dict] = []
    for row in sm_full["rows"]:
        rows.extend([
            {
                "dataset": "smarteole", "mode": "mafs_base",
                "seed": row["seed"], "validation_mse": row["base_validation_mse"],
            },
            {
                "dataset": "smarteole", "mode": "maspge_rs_full",
                "seed": row["seed"],
                "validation_mse": row["role_state_validation_mse"],
            },
        ])
    for mode in ("shared_units", "unscaled_history"):
        for seed in SEEDS:
            payload = read_json(
                ROOT / f"results/role_state_ablation_v1/smarteole_{mode}/"
                f"seed_{seed}/role_state.json"
            )
            if payload["role_state_ablation_mode"] != mode:
                raise RuntimeError(f"SMARTEOLE mode mismatch for seed {seed}")
            rows.append({
                "dataset": "smarteole", "mode": mode, "seed": seed,
                "validation_mse": payload["validation"]["metrics"][
                    "mse_standardized"
                ],
            })
    for row in hai_full["rows"]:
        rows.extend([
            {
                "dataset": "hai", "mode": "mafs_base", "seed": row["seed"],
                "validation_mse": row["base_validation_mse"],
            },
            {
                "dataset": "hai", "mode": "maspge_rs_full", "seed": row["seed"],
                "validation_mse": row["role_state_validation_mse"],
            },
        ])
    for seed in SEEDS:
        payload = read_json(
            ROOT / "results/role_state_ablation_v1/hai_without_a4/"
            f"seed_{seed}/role_state.json"
        )
        if payload["role_state_ablation_mode"] != "without_a4":
            raise RuntimeError(f"HAI mode mismatch for seed {seed}")
        rows.append({
            "dataset": "hai", "mode": "without_a4", "seed": seed,
            "validation_mse": payload["role_state"]["validation"]["metrics"][
                "mse_standardized"
            ],
        })

    summary_rows = []
    for dataset in ("smarteole", "hai"):
        modes = sorted({row["mode"] for row in rows if row["dataset"] == dataset})
        full_by_seed = {
            row["seed"]: row["validation_mse"] for row in rows
            if row["dataset"] == dataset and row["mode"] == "maspge_rs_full"
        }
        for mode in modes:
            selected = [
                row for row in rows
                if row["dataset"] == dataset and row["mode"] == mode
            ]
            values = [float(row["validation_mse"]) for row in selected]
            mean, std = mean_std(values)
            paired = [
                (float(row["validation_mse"]) - full_by_seed[row["seed"]])
                / float(row["validation_mse"])
                for row in selected
                if mode != "maspge_rs_full"
            ]
            summary_rows.append({
                "dataset": dataset, "mode": mode,
                "validation_mse_mean": mean, "validation_mse_std": std,
                "full_relative_gain_mean": (
                    float(np.mean(paired)) if paired else 0.0
                ),
                "full_wins": (
                    int(sum(value > 0 for value in paired)) if paired else 0
                ),
                "runs": len(selected),
            })

    output = ROOT / "results/role_state_ablation_v1/summary"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "validation_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output / "validation_summary.json", {
        "phase": "five_seed_validation_only_structural_ablation",
        "holdout_evaluated": False,
        "seeds": list(SEEDS),
        "parameter_matched_controls": True,
        "selection_or_retuning_from_ablation": False,
        "summary": summary_rows,
        "rows": rows,
    })
    for row in summary_rows:
        print(
            f"{row['dataset']:<10} {row['mode']:<20} "
            f"MSE={row['validation_mse_mean']:.6f}"
            f"±{row['validation_mse_std']:.6f} "
            f"full_gain={100*row['full_relative_gain_mean']:.2f}% "
            f"wins={row['full_wins']}/5"
        )
    print("HOLDOUT WAS NOT EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
