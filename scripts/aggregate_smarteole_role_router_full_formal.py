#!/usr/bin/env python3
"""Early five-seed summary for the full-training-set proposed method."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def main() -> int:
    root = Path("results/unified_formal_v1/role_router")
    rows = []
    for seed in (42, 43, 44, 45, 46):
        path = root / f"seed_{seed}" / "mafs_maspge_full.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("train_windows") != 81188 or not record.get("holdout_evaluated"):
            raise RuntimeError(f"not a complete full-training result: {path}")
        base = record["base"]["holdout"]
        routed = record["full_role_router"]["holdout"]
        base_mse = float(base["metrics"]["mse_standardized"])
        routed_mse = float(routed["metrics"]["mse_standardized"])
        rows.append({
            "seed": seed,
            "base_holdout_mse": base_mse,
            "routed_holdout_mse": routed_mse,
            "base_holdout_mae": base["metrics"]["mae_standardized"],
            "routed_holdout_mae": routed["metrics"]["mae_standardized"],
            "relative_holdout_mse_gain": (base_mse - routed_mse) / base_mse,
            "units_improved": sum(
                new < old for new, old in zip(
                    routed["per_unit_mse_standardized"],
                    base["per_unit_mse_standardized"],
                )
            ),
            "maximum_mean_agent_vote": max(routed["mean_agent_vote_weights"]),
        })
    base = np.asarray([row["base_holdout_mse"] for row in rows])
    routed = np.asarray([row["routed_holdout_mse"] for row in rows])
    gains = np.asarray([row["relative_holdout_mse_gain"] for row in rows])
    wins = int((routed < base).sum())
    maximum_vote = max(row["maximum_mean_agent_vote"] for row in rows)
    summary = {
        "phase": "full_training_role_router_early_summary",
        "smarteole_holdout_previously_observed": True,
        "train_windows": 81188,
        "base_holdout_mse_mean": float(base.mean()),
        "base_holdout_mse_std": float(base.std(ddof=1)),
        "routed_holdout_mse_mean": float(routed.mean()),
        "routed_holdout_mse_std": float(routed.std(ddof=1)),
        "mean_paired_relative_mse_gain": float(gains.mean()),
        "seeds_beating_base": wins,
        "maximum_mean_agent_vote": maximum_vote,
        "descriptive_acceptance": (
            wins >= 4 and float(gains.mean()) >= 0.01 and maximum_vote <= 0.75
        ),
        "rows": rows,
    }
    out = Path("results/unified_formal_v1/summary")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "role_router_per_seed_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "role_router_early_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"full-data MAFS base holdout_mse={summary['base_holdout_mse_mean']:.6f}"
        f"±{summary['base_holdout_mse_std']:.6f}"
    )
    print(
        f"full-data MASPGE holdout_mse={summary['routed_holdout_mse_mean']:.6f}"
        f"±{summary['routed_holdout_mse_std']:.6f} "
        f"gain={100*summary['mean_paired_relative_mse_gain']:.2f}% wins={wins}/5"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
