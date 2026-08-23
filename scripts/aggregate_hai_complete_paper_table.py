#!/usr/bin/env python3
"""Create the seven-method, five-seed HAI paper table."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SEEDS = (42, 43, 44, 45, 46)
EXTERNAL = (
    "dlinear_power", "fouriergnn_power_official_small",
    "itransformer_power", "softs_power_compact",
)
METHODS = ("persistence", *EXTERNAL, "mafs_adapted_fully", "maspge_rs")


def metric_row(mode: str, seed: int, result: dict, source: Path) -> dict:
    return {
        "mode": mode,
        "seed": seed,
        "holdout_mse": result["metrics"]["mse_standardized"],
        "holdout_mae": result["metrics"]["mae_standardized"],
        "thermal_mse": result["per_unit_metrics"][0]["mse_standardized"],
        "hydro_mse": result["per_unit_metrics"][1]["mse_standardized"],
        "thermal_rmse_physical": result["per_unit_metrics"][0]["rmse_physical"],
        "hydro_rmse_physical": result["per_unit_metrics"][1]["rmse_physical"],
        "source": str(source),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def main() -> int:
    rows = []
    hashes = set()
    for seed in SEEDS:
        for mode in EXTERNAL:
            path = Path("results/hai_external_holdout_v1") / f"seed_{seed}" / f"{mode}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("training_performed") is not False:
                raise RuntimeError(f"external holdout trained parameters: {path}")
            hashes.add(record["dataset_sha256"])
            rows.append(metric_row(mode, seed, record["holdout"], path))
        path = (
            Path("results/hai_persistence_holdout_v1")
            / f"seed_{seed}" / "persistence.json"
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("training_performed") is not False:
            raise RuntimeError(f"persistence holdout trained parameters: {path}")
        hashes.add(record["dataset_sha256"])
        rows.append(metric_row("persistence", seed, record["holdout"], path))
        path = Path("results/hai_role_state_holdout_v1") / f"seed_{seed}" / "holdout.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("training_performed") is not False:
            raise RuntimeError(f"MASPGE-RS holdout trained parameters: {path}")
        hashes.add(record["dataset_sha256"])
        rows.append(metric_row(
            "mafs_adapted_fully", seed, record["base_holdout"], path
        ))
        rows.append(metric_row(
            "maspge_rs", seed, record["role_state_holdout"], path
        ))
    if len(hashes) != 1:
        raise RuntimeError(f"HAI main-table dataset mismatch: {hashes}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)
    summary = []
    for mode in METHODS:
        group = grouped[mode]
        item = {"mode": mode, "runs": len(group)}
        for metric in (
            "holdout_mse", "holdout_mae", "thermal_mse", "hydro_mse",
            "thermal_rmse_physical", "hydro_rmse_physical",
        ):
            mean, std = mean_std([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        summary.append(item)
    summary.sort(key=lambda item: item["holdout_mse_mean"])
    out = Path("results/hai_complete_paper_table_v1/summary")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_seed_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "aggregate_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    payload = {
        "phase": "hai_seven_method_five_seed_complete_paper_table",
        "holdout_evaluated": True,
        "training_performed_during_holdout": False,
        "post_holdout_tuning_permitted": False,
        "hai_hyperparameter_search": False,
        "dataset_sha256": next(iter(hashes)),
        "rows": summary,
    }
    (out / "aggregate_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for row in summary:
        print(
            f"{row['mode']:<38} holdout_mse={row['holdout_mse_mean']:.6f}"
            f"±{row['holdout_mse_std']:.6f}"
        )
    print(f"wrote {out / 'aggregate_results.csv'}")
    print("NO TRAINING OR POST-HOLDOUT TUNING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
