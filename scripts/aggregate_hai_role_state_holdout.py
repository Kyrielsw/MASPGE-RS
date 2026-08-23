#!/usr/bin/env python3
"""Aggregate the one-shot strict HAI holdout report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def main() -> int:
    root = Path("results/hai_role_state_holdout_v1")
    rows = []
    hashes = set()
    for seed in (42, 43, 44, 45, 46):
        path = root / f"seed_{seed}" / "holdout.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("phase") != "hai_strict_frozen_holdout_report":
            raise RuntimeError(f"unexpected HAI holdout phase: {path}")
        if record.get("training_performed") or not record.get("holdout_evaluated"):
            raise RuntimeError(f"invalid frozen HAI evaluation: {path}")
        hashes.add(record["dataset_sha256"])
        base = record["base_holdout"]
        role = record["role_state_holdout"]
        rows.append({
            "seed": seed,
            "base_mse": base["metrics"]["mse_standardized"],
            "base_mae": base["metrics"]["mae_standardized"],
            "role_state_mse": role["metrics"]["mse_standardized"],
            "role_state_mae": role["metrics"]["mae_standardized"],
            "relative_mse_gain": record["relative_holdout_mse_gain"],
            "base_per_unit": base["per_unit_metrics"],
            "role_state_per_unit": role["per_unit_metrics"],
            "role_state_rms": role["role_state_rms"],
        })
    if len(hashes) != 1:
        raise RuntimeError(f"inconsistent HAI holdout datasets: {hashes}")
    gains = [row["relative_mse_gain"] for row in rows]
    per_unit = {}
    for unit_index, unit in enumerate(("thermal", "hydro")):
        base_mse = [row["base_per_unit"][unit_index]["mse_standardized"] for row in rows]
        role_mse = [row["role_state_per_unit"][unit_index]["mse_standardized"] for row in rows]
        base_rmse = [row["base_per_unit"][unit_index]["rmse_physical"] for row in rows]
        role_rmse = [row["role_state_per_unit"][unit_index]["rmse_physical"] for row in rows]
        per_unit[unit] = {
            "base_mse_standardized": mean_std(base_mse),
            "role_state_mse_standardized": mean_std(role_mse),
            "mean_paired_relative_mse_gain": float(np.mean([
                (base - proposed) / base for base, proposed in zip(base_mse, role_mse)
            ])),
            "base_rmse_physical": mean_std(base_rmse),
            "role_state_rmse_physical": mean_std(role_rmse),
            "seeds_improved": int(sum(proposed < base for base, proposed in zip(base_mse, role_mse))),
        }
    payload = {
        "phase": "five_seed_hai_strict_frozen_holdout_report",
        "holdout_evaluated": True,
        "training_performed": False,
        "strict_unseen_model_test": True,
        "post_holdout_tuning_permitted": False,
        "dataset_sha256": next(iter(hashes)),
        "base_mse_standardized": mean_std([row["base_mse"] for row in rows]),
        "role_state_mse_standardized": mean_std([row["role_state_mse"] for row in rows]),
        "base_mae_standardized": mean_std([row["base_mae"] for row in rows]),
        "role_state_mae_standardized": mean_std([row["role_state_mae"] for row in rows]),
        "mean_paired_relative_mse_gain": float(np.mean(gains)),
        "seeds_beating_same_seed_mafs": int(sum(gain > 0 for gain in gains)),
        "per_unit": per_unit,
        "rows": rows,
    }
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base = payload["base_mse_standardized"]
    role = payload["role_state_mse_standardized"]
    print(f"HAI MAFS Base MSE={base['mean']:.6f}±{base['std']:.6f}")
    print(f"HAI MASPGE-RS MSE={role['mean']:.6f}±{role['std']:.6f}")
    print(
        f"paired gain={100*payload['mean_paired_relative_mse_gain']:.2f}% "
        f"wins={payload['seeds_beating_same_seed_mafs']}/5"
    )
    for unit, values in per_unit.items():
        print(
            f"{unit} gain={100*values['mean_paired_relative_mse_gain']:.2f}% "
            f"wins={values['seeds_improved']}/5"
        )
    print("STRICT UNSEEN HOLDOUT REPORT; NO POST-HOLDOUT TUNING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
