#!/usr/bin/env python3
"""Aggregate the frozen eight-method XAI4HEAT validation comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (42, 43, 44, 45, 46)
EXTERNAL = (
    "dlinear_power", "fouriergnn_power_official_small", "itransformer_power",
    "softs_power_compact", "timemixer_power",
)


def main() -> int:
    rows = []
    hashes = set()
    for seed in SEEDS:
        for mode in EXTERNAL:
            path = Path("results/xai4heat_external_validation_v1") / f"seed_{seed}" / f"{mode}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("holdout_evaluated") is not False:
                raise RuntimeError(f"validation accessed holdout: {path}")
            hashes.add(record["dataset_sha256"])
            metrics = record["validation"]["metrics"]
            rows.append({"mode": mode, "seed": seed, "parameter_count": record["parameter_count"],
                         "validation_mse": metrics["mse_standardized"],
                         "validation_mae": metrics["mae_standardized"]})
        path = Path("results/xai4heat_persistence_validation_v1") / f"seed_{seed}" / "persistence.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        hashes.add(record["dataset_sha256"])
        metrics = record["validation"]["metrics"]
        rows.append({"mode": "persistence", "seed": seed, "parameter_count": 0,
                     "validation_mse": metrics["mse_standardized"],
                     "validation_mae": metrics["mae_standardized"]})
        path = Path("results/xai4heat_main_validation_v1") / f"seed_{seed}" / "result.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("holdout_evaluated") is not False:
            raise RuntimeError(f"main validation accessed holdout: {path}")
        hashes.add(record["dataset_sha256"])
        for mode, metrics in (
            ("mafs_adapted_fully", record["base"]["finetune"]["validation"]["metrics"]),
            ("generic_unit_state", record["generic_unit_state"]["validation"]["metrics"]),
        ):
            rows.append({"mode": mode, "seed": seed, "parameter_count": None,
                         "validation_mse": metrics["mse_standardized"],
                         "validation_mae": metrics["mae_standardized"]})
    if len(hashes) != 1:
        raise RuntimeError(f"dataset mismatch: {hashes}")
    ranking = []
    for mode in ("persistence", *EXTERNAL, "mafs_adapted_fully", "generic_unit_state"):
        selected = [row for row in rows if row["mode"] == mode]
        mse = np.asarray([row["validation_mse"] for row in selected])
        mae = np.asarray([row["validation_mae"] for row in selected])
        ranking.append({"mode": mode, "validation_mse_mean": float(mse.mean()),
                        "validation_mse_std": float(mse.std(ddof=1)),
                        "validation_mae_mean": float(mae.mean()),
                        "validation_mae_std": float(mae.std(ddof=1)), "runs": len(selected)})
    ranking.sort(key=lambda item: item["validation_mse_mean"])
    mafs = {row["seed"]: row for row in rows if row["mode"] == "mafs_adapted_fully"}
    generic = {row["seed"]: row for row in rows if row["mode"] == "generic_unit_state"}
    gains = [(mafs[s]["validation_mse"] - generic[s]["validation_mse"]) / mafs[s]["validation_mse"] for s in SEEDS]
    payload = {
        "phase": "xai4heat_eight_method_validation_only", "holdout_evaluated": False,
        "dataset_sha256": next(iter(hashes)), "rows": rows, "ranking": ranking,
        "generic_vs_mafs": {"paired_relative_mse_gains": gains,
                            "mean_paired_relative_mse_gain": float(np.mean(gains)),
                            "seeds_improved": int(sum(value > 0 for value in gains))},
    }
    out = Path("results/xai4heat_complete_validation_v1/summary")
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for row in ranking:
        print(f"{row['mode']:<36} validation_mse={row['validation_mse_mean']:.6f}±{row['validation_mse_std']:.6f}")
    print(f"Generic vs MAFS gain={100*np.mean(gains):.2f}% wins={sum(value > 0 for value in gains)}/5")
    print("HOLDOUT WAS NOT EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
