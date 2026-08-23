#!/usr/bin/env python3
"""Aggregate multi-horizon validation or holdout results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics_for(root: Path, split: str, dataset: str, horizon: int,
                seed: int, method: str) -> dict:
    if method == "softs_power_compact":
        payload = json.loads((root / dataset / f"h{horizon}" /
                              f"seed_{seed}/{method}.json").read_text(encoding="utf-8"))
        return payload[split]["metrics"]
    payload = json.loads((root / dataset / f"h{horizon}" /
                          f"seed_{seed}/maspge_rs.json").read_text(encoding="utf-8"))
    if split == "validation":
        key = "mafs_base" if method == "mafs_adapted_fully" else "maspge_rs"
        source = (payload["mafs_base"]["finetune"]["validation"]
                  if key == "mafs_base" else payload["maspge_rs"]["validation"])
    else:
        key = "mafs_base" if method == "mafs_adapted_fully" else "maspge_rs"
        source = payload["holdout"][key]
    return source["metrics"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "holdout"], required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    root = project / f"results/multihorizon_{args.split}_v1"
    methods = ("softs_power_compact", "mafs_adapted_fully", "maspge_rs")
    rows = []
    for dataset in ("smarteole", "sdwpf"):
        for horizon in (5, 30):
            for method in methods:
                values = [metrics_for(root, args.split, dataset, horizon, seed, method)
                          for seed in (42, 43, 44, 45, 46)]
                rows.append({
                    "dataset": dataset, "horizon_steps": horizon, "method": method,
                    "runs": 5,
                    "mse_mean": float(np.mean([v["mse_standardized"] for v in values])),
                    "mse_std": float(np.std([v["mse_standardized"] for v in values])),
                    "mae_mean": float(np.mean([v["mae_standardized"] for v in values])),
                    "mae_std": float(np.std([v["mae_standardized"] for v in values])),
                })
    for dataset in ("smarteole", "sdwpf"):
        for horizon in (5, 30):
            print(f"{dataset} horizon={horizon}")
            for row in rows:
                if row["dataset"] == dataset and row["horizon_steps"] == horizon:
                    print(f"  {row['method']:<24} MSE={row['mse_mean']:.6f}±{row['mse_std']:.6f}")
    output = root / "summary" / f"{args.split}_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"split": args.split, "rows": rows}, indent=2),
                      encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
