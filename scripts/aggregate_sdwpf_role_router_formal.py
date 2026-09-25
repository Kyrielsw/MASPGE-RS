#!/usr/bin/env python3
"""Aggregate the five-seed paper-ready SDWPF confirmation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def main() -> int:
    root = Path("results/sdwpf_role_router_formal_v1")
    rows = []
    hashes = set()
    for seed in (42, 43, 44, 45, 46):
        path = root / f"seed_{seed}" / "mafs_role_router_full.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("holdout_evaluated") is not True:
            raise RuntimeError(f"formal run did not evaluate holdout once: {path}")
        hashes.add(record["dataset_sha256"])
        base_val = record["base"]["finetune"]["validation"]["metrics"]
        routed_val = record["full_role_router"]["validation"]["metrics"]
        base_test = record["base"]["holdout"]["metrics"]
        routed_test = record["full_role_router"]["holdout"]["metrics"]
        rows.append({
            "seed": seed,
            "base_validation_mse": base_val["mse_standardized"],
            "routed_validation_mse": routed_val["mse_standardized"],
            "base_holdout_mse": base_test["mse_standardized"],
            "routed_holdout_mse": routed_test["mse_standardized"],
            "base_holdout_mae": base_test["mae_standardized"],
            "routed_holdout_mae": routed_test["mae_standardized"],
            "holdout_relative_mse_gain": (
                base_test["mse_standardized"] - routed_test["mse_standardized"]
            ) / base_test["mse_standardized"],
            "maximum_mean_agent_vote": max(
                record["full_role_router"]["holdout"]["mean_agent_vote_weights"]
            ),
            "wall_seconds": record["wall_seconds"],
            "peak_cuda_memory_gib": record["peak_cuda_memory_gib"],
        })
    if len(hashes) != 1:
        raise RuntimeError(f"seeds used different processed datasets: {hashes}")
    base = np.asarray([row["base_holdout_mse"] for row in rows])
    routed = np.asarray([row["routed_holdout_mse"] for row in rows])
    gains = np.asarray([row["holdout_relative_mse_gain"] for row in rows])
    wins = int((routed < base).sum())
    maximum_vote = max(row["maximum_mean_agent_vote"] for row in rows)
    accepted = wins >= 4 and float(gains.mean()) >= 0.01 and maximum_vote <= 0.75
    summary = {
        "phase": "five_seed_full_scale_cross_dataset_formal",
        "dataset": "SDWPF KDD 245 days / 134 turbines",
        "processed_dataset_sha256": hashes.pop(),
        "holdout_evaluated": True,
        "seeds": [42, 43, 44, 45, 46],
        "base_holdout_mse_mean": float(base.mean()),
        "base_holdout_mse_std": float(base.std(ddof=1)),
        "routed_holdout_mse_mean": float(routed.mean()),
        "routed_holdout_mse_std": float(routed.std(ddof=1)),
        "mean_paired_relative_mse_gain": float(gains.mean()),
        "seeds_beating_same_run_mafs": wins,
        "maximum_mean_agent_vote": maximum_vote,
        "accepted_for_main_table": accepted,
        "acceptance_rule": "wins>=4/5, mean paired MSE gain>=1%, max mean vote<=0.75",
        "rows": rows,
    }
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_seed_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "formal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"base holdout_mse={summary['base_holdout_mse_mean']:.6f}"
        f"±{summary['base_holdout_mse_std']:.6f}"
    )
    print(
        f"full role router holdout_mse={summary['routed_holdout_mse_mean']:.6f}"
        f"±{summary['routed_holdout_mse_std']:.6f} "
        f"gain={100*summary['mean_paired_relative_mse_gain']:.2f}% "
        f"wins={summary['seeds_beating_same_run_mafs']}/5"
    )
    print("Paper table: results/sdwpf_role_router_formal_v1/summary/per_seed_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
