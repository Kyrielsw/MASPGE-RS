#!/usr/bin/env python3
"""Aggregate the frozen eight-method XAI4HEAT holdout report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from evaluate_xai4heat_holdout import MODES


SEEDS = (42, 43, 44, 45, 46)


def main() -> int:
    rows = []
    hashes = set()
    for seed in SEEDS:
        for mode in MODES:
            path = Path("results/xai4heat_holdout_v1") / f"seed_{seed}" / f"{mode}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("training_performed") is not False or record.get("holdout_evaluated") is not True:
                raise RuntimeError(f"invalid frozen report: {path}")
            hashes.add(record["dataset_sha256"])
            metrics = record["holdout"]["metrics"]
            rows.append({"mode": mode, "seed": seed,
                         "holdout_mse": metrics["mse_standardized"],
                         "holdout_mae": metrics["mae_standardized"]})
    if len(hashes) != 1:
        raise RuntimeError(f"dataset mismatch: {hashes}")
    ranking = []
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        mse = np.asarray([row["holdout_mse"] for row in selected])
        mae = np.asarray([row["holdout_mae"] for row in selected])
        ranking.append({"mode": mode, "holdout_mse_mean": float(mse.mean()),
                        "holdout_mse_std": float(mse.std(ddof=1)),
                        "holdout_mae_mean": float(mae.mean()),
                        "holdout_mae_std": float(mae.std(ddof=1)), "runs": len(selected)})
    ranking.sort(key=lambda item: item["holdout_mse_mean"])
    mafs = {row["seed"]: row for row in rows if row["mode"] == "mafs_adapted_fully"}
    generic = {row["seed"]: row for row in rows if row["mode"] == "generic_unit_state"}
    gains = [(mafs[s]["holdout_mse"] - generic[s]["holdout_mse"]) / mafs[s]["holdout_mse"] for s in SEEDS]
    payload = {
        "phase": "xai4heat_frozen_eight_method_holdout", "training_performed": False,
        "holdout_evaluated": True, "post_holdout_tuning_permitted": False,
        "dataset_sha256": next(iter(hashes)), "rows": rows, "ranking": ranking,
        "generic_vs_mafs": {"paired_relative_mse_gains": gains,
                            "mean_paired_relative_mse_gain": float(np.mean(gains)),
                            "seed_gain_std": float(np.std(gains, ddof=1)),
                            "seeds_improved": int(sum(value > 0 for value in gains))},
    }
    out = Path("results/xai4heat_complete_holdout_v1/summary")
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for row in ranking:
        print(f"{row['mode']:<36} holdout_mse={row['holdout_mse_mean']:.6f}±{row['holdout_mse_std']:.6f}")
    print(f"Generic vs MAFS gain={100*np.mean(gains):.2f}% wins={sum(value > 0 for value in gains)}/5")
    print("FROZEN HOLDOUT REPORT; NO TRAINING OR POST-HOLDOUT TUNING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
