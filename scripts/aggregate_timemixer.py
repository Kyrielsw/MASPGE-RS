#!/usr/bin/env python3
"""Aggregate the frozen five-seed TimeMixer validation or holdout results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["smarteole", "sdwpf", "hai"], required=True)
    parser.add_argument("--split", choices=["validation", "holdout"], required=True)
    parser.add_argument("--revision", default="v2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    source = root / f"results/{args.dataset}_timemixer_{args.split}_{args.revision}"
    rows = []
    for seed in (42, 43, 44, 45, 46):
        payload = json.loads(
            (source / f"seed_{seed}/timemixer_power.json").read_text(encoding="utf-8")
        )
        metrics = payload[args.split]["metrics"]
        rows.append({"seed": seed, "mse": metrics["mse_standardized"],
                     "mae": metrics["mae_standardized"]})
    summary = {
        "dataset": args.dataset, "model": "timemixer_power", "split": args.split,
        "integration_revision": args.revision,
        "runs": 5, "mse_mean": float(np.mean([row["mse"] for row in rows])),
        "mse_std": float(np.std([row["mse"] for row in rows])),
        "mae_mean": float(np.mean([row["mae"] for row in rows])),
        "mae_std": float(np.std([row["mae"] for row in rows])), "rows": rows,
    }
    output = source / "summary" / f"{args.split}_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{args.dataset} TimeMixer {args.split} MSE="
          f"{summary['mse_mean']:.6f}±{summary['mse_std']:.6f}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
