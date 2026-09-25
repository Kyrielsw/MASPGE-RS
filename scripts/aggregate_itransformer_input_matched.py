#!/usr/bin/env python3
"""Aggregate all seeds for the input-matched fairness control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_external_baseline_screen import ROOT, atomic_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/itransformer_input_matched_v1.json")
    parser.add_argument("--validation-dir", type=Path, default=ROOT / "results/itransformer_input_matched_validation_v1")
    parser.add_argument("--holdout-dir", type=Path, default=ROOT / "results/itransformer_input_matched_holdout_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/itransformer_input_matched_summary_v1")
    parser.add_argument("--include-holdout", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
    rows, summary = [], []
    for dataset in config["datasets"]:
        dataset_rows = []
        for seed in config["seeds"]:
            validation = json.loads((args.validation_dir / dataset / f"seed_{seed}.json").read_text(encoding="utf-8"))
            vm = validation["validation"]["metrics"]
            row = {"dataset": dataset, "seed": seed, "parameter_count": validation["parameter_count"], "best_epoch": validation["best_epoch"], "epochs_completed": validation["epochs_completed"], "wall_seconds": validation["wall_seconds"], "validation_mse": vm["mse_standardized"], "validation_mae": vm["mae_standardized"]}
            if args.include_holdout:
                holdout = json.loads((args.holdout_dir / dataset / f"seed_{seed}.json").read_text(encoding="utf-8"))
                hm = holdout["holdout"]["metrics"]
                row.update({"holdout_mse": hm["mse_standardized"], "holdout_mae": hm["mae_standardized"], "checkpoint_sha256": holdout["checkpoint_sha256"]})
            rows.append(row); dataset_rows.append(row)
        item = {"dataset": dataset, "runs": len(dataset_rows), "parameter_count": dataset_rows[0]["parameter_count"]}
        for split in (["validation", "holdout"] if args.include_holdout else ["validation"]):
            for metric in ("mse", "mae"):
                values = np.asarray([r[f"{split}_{metric}"] for r in dataset_rows], dtype=float)
                item[f"{split}_{metric}_mean"] = float(values.mean())
                item[f"{split}_{metric}_std"] = float(values.std(ddof=1))
        summary.append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    payload = {"method": config["method_label"], "adapted_not_official": True, "holdout_included": args.include_holdout, "input_protocol": config["data_contract"], "summary": summary, "rows": rows}
    atomic_json(args.output_dir / "aggregate_summary.json", payload)
    for item in summary:
        split = "holdout" if args.include_holdout else "validation"
        print(f"{item['dataset']:<12} {split}_mse={item[f'{split}_mse_mean']:.6f}±{item[f'{split}_mse_std']:.6f} mae={item[f'{split}_mae_mean']:.6f}±{item[f'{split}_mae_std']:.6f}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
