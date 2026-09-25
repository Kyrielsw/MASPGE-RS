#!/usr/bin/env python3
"""Aggregate frozen Generic Unit State validation controls and write handoff."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from run_mafs_adapted_screen import ROOT, atomic_json


CONFIG = ROOT / "configs/generic_mechanism_validation_v1.json"
RESULT_ROOT = ROOT / "results/generic_mechanism_validation_v1"
SUMMARY_ROOT = RESULT_ROOT / "summary"


def load_rows(control: dict) -> tuple[list[dict], list[dict]]:
    complete = []
    failures = []
    for dataset in control["datasets"]:
        for variant in control["variants"]:
            for seed in control["seeds"]:
                root = RESULT_ROOT / dataset / variant / f"seed_{seed}"
                result_path = root / "result.json"
                failure_path = root / "failure.json"
                if result_path.is_file():
                    complete.append(json.loads(result_path.read_text(encoding="utf-8")))
                elif failure_path.is_file():
                    failures.append(json.loads(failure_path.read_text(encoding="utf-8")))
                else:
                    failures.append({
                        "status": "missing", "dataset": dataset,
                        "variant": variant, "seed": seed,
                        "error": "no result.json or failure.json",
                    })
    return complete, failures


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def variant_summary(rows: list[dict]) -> dict:
    mse = [row["variant_validation"]["metrics"]["mse_standardized"] for row in rows]
    mae = [row["variant_validation"]["metrics"]["mae_standardized"] for row in rows]
    gain = [row["paired_gain_vs_same_checkpoint_mafs"] for row in rows]
    best = [row["best_epoch"] for row in rows]
    completed = [row["epochs_completed"] for row in rows]
    initial_difference = [
        row["initialization_diagnostics"]
        ["initial_max_abs_prediction_difference_from_frozen_backbone"]
        for row in rows
    ]
    first_epoch = [row["learning_curve"][0]["validation_mse_standardized"] for row in rows]
    return {
        "runs": len(rows),
        "mse_mean": mean_std(mse)[0], "mse_std": mean_std(mse)[1],
        "mae_mean": mean_std(mae)[0], "mae_std": mean_std(mae)[1],
        "paired_gain_vs_mafs_mean": mean_std(gain)[0],
        "seeds_beating_mafs": sum(value > 0 for value in gain),
        "best_epoch_mean": mean_std(best)[0], "best_epoch_std": mean_std(best)[1],
        "epochs_completed_mean": mean_std(completed)[0],
        "first_epoch_mse_mean": mean_std(first_epoch)[0],
        "first_epoch_mse_std": mean_std(first_epoch)[1],
        "maximum_initial_prediction_difference": max(initial_difference),
        "adapter_parameters": sorted({row["adapter_parameters"] for row in rows}),
    }


def paired(control_rows: list[dict], preferred: str, control: str) -> dict:
    by_key = {(row["seed"], row["variant"]): row for row in control_rows}
    rows = []
    for seed in sorted({row["seed"] for row in control_rows}):
        left = by_key[(seed, preferred)]
        right = by_key[(seed, control)]
        left_mse = left["variant_validation"]["metrics"]["mse_standardized"]
        right_mse = right["variant_validation"]["metrics"]["mse_standardized"]
        rows.append({
            "seed": seed, "preferred_mse": left_mse, "control_mse": right_mse,
            "relative_mse_gain": (right_mse - left_mse) / right_mse,
        })
    gains = [row["relative_mse_gain"] for row in rows]
    return {
        "preferred": preferred, "control": control,
        "mean_paired_relative_mse_gain": mean_std(gains)[0],
        "std_paired_relative_mse_gain": mean_std(gains)[1],
        "preferred_wins": sum(value > 0 for value in gains),
        "rows": rows,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "variant", "seed", "mse", "mae", "gain_vs_mafs",
        "best_epoch", "epochs_completed", "initial_prediction_difference",
        "gradient_step0_projection", "gradient_step0_upstream",
        "gradient_step1_upstream", "adapter_parameters", "wall_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            diagnostics = row["initialization_diagnostics"]
            writer.writerow({
                "dataset": row["dataset"], "variant": row["variant"],
                "seed": row["seed"],
                "mse": row["variant_validation"]["metrics"]["mse_standardized"],
                "mae": row["variant_validation"]["metrics"]["mae_standardized"],
                "gain_vs_mafs": row["paired_gain_vs_same_checkpoint_mafs"],
                "best_epoch": row["best_epoch"],
                "epochs_completed": row["epochs_completed"],
                "initial_prediction_difference": diagnostics[
                    "initial_max_abs_prediction_difference_from_frozen_backbone"
                ],
                "gradient_step0_projection": diagnostics["gradient_steps"][0]
                    ["gradient_norms_before_clipping"]["residual_projection"],
                "gradient_step0_upstream": diagnostics["gradient_steps"][0]
                    ["gradient_norms_before_clipping"]["upstream_encoder_fusion"],
                "gradient_step1_upstream": diagnostics["gradient_steps"][1]
                    ["gradient_norms_before_clipping"]["upstream_encoder_fusion"],
                "adapter_parameters": row["adapter_parameters"],
                "wall_seconds": row["wall_seconds"],
            })


def verdict(comparison: dict) -> str:
    gain = 100 * comparison["mean_paired_relative_mse_gain"]
    wins = comparison["preferred_wins"]
    return f"{gain:.2f}% mean paired gain, {wins}/5 seed wins"


def write_handoff(summary: dict, path: Path) -> None:
    sections = [
        "# Generic Unit State mechanism-control paper handoff",
        "",
        "## Protocol status",
        "",
        "This handoff reports validation-only controls on SMARTEOLE and "
        "XAI4HEAT. No holdout/test evaluation was performed, and these "
        "controls must not be described as new blind-test evidence.",
        "",
    ]
    for dataset, item in summary["datasets"].items():
        sections.extend([
            f"## {dataset}", "",
            f"- Independent unit state versus pooled/broadcast: {verdict(item['comparisons']['unit_specific_vs_pooled'])}.",
            f"- Pre- versus post-communication injection: {verdict(item['comparisons']['pre_vs_post'])}.",
            f"- Zero versus small-random initialization: {verdict(item['comparisons']['zero_vs_small_random'])}.",
            "",
        ])
        for name, values in item["variants"].items():
            sections.append(
                f"- `{name}`: MSE {values['mse_mean']:.6f} ± {values['mse_std']:.6f}; "
                f"MAE {values['mae_mean']:.6f} ± {values['mae_std']:.6f}; "
                f"best epoch {values['best_epoch_mean']:.1f} ± {values['best_epoch_std']:.1f}."
            )
        sections.append("")

    zero_identity = all(
        item["variants"][name]["maximum_initial_prediction_difference"] <= 1e-6
        for item in summary["datasets"].values()
        for name in ("full_zero", "pooled_broadcast_zero", "post_communication_zero")
    )
    sections.extend([
        "## Claim audit", "",
        f"- Supported: zero-initialized variants preserve the frozen backbone "
        f"prediction at step 0 within a preregistered 1e-6 numerical tolerance: {zero_identity}.",
        "- The unit-state, injection-position, and initialization accuracy claims "
        "are supported only where the dataset-specific paired result above is "
        "positive and consistent; mixed results must be written as mixed evidence.",
        "- Initialization stability must be discussed using first-epoch MSE "
        "dispersion, stopping epochs, full curves, and final MSE dispersion—not "
        "inferred from deterministic identity alone.",
        "- Not supported by this experiment: claims about A1--A4 semantics, "
        "test-set generalization of the new controls, or universal superiority.",
        "",
        "## Reproducibility", "",
        "Per-seed JSON files contain learning curves, initial prediction gaps, "
        "two-step gradient diagnostics, checkpoint/data/config/code hashes, "
        "parameter counts, and wall time. `per_seed_results.csv` is the compact "
        "paper-analysis table; `summary.json` is the authoritative aggregate.",
        "",
    ])
    path.write_text("\n".join(sections), encoding="utf-8")


def main() -> int:
    control = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows, failures = load_rows(control)
    expected = len(control["datasets"]) * len(control["variants"]) * len(control["seeds"])
    if failures or len(rows) != expected:
        SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
        atomic_json(SUMMARY_ROOT / "failures.json", {"expected": expected, "failures": failures})
        raise RuntimeError(f"mechanism suite incomplete: {len(rows)}/{expected}; failures={len(failures)}")

    summary = {
        "phase": control["phase"], "holdout_evaluated": False,
        "complete_runs": len(rows), "failed_runs": 0, "datasets": {},
    }
    for dataset in control["datasets"]:
        selected = [row for row in rows if row["dataset"] == dataset]
        summary["datasets"][dataset] = {
            "variants": {
                variant: variant_summary(
                    [row for row in selected if row["variant"] == variant]
                )
                for variant in control["variants"]
            },
            "comparisons": {
                "unit_specific_vs_pooled": paired(selected, "full_zero", "pooled_broadcast_zero"),
                "pre_vs_post": paired(selected, "full_zero", "post_communication_zero"),
                "zero_vs_small_random": paired(selected, "full_zero", "full_small_random"),
            },
        }
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(rows, SUMMARY_ROOT / "per_seed_results.csv")
    atomic_json(SUMMARY_ROOT / "summary.json", summary)
    write_handoff(summary, SUMMARY_ROOT / "paper_handoff.md")
    for dataset, item in summary["datasets"].items():
        print(f"{dataset}")
        for comparison in item["comparisons"].values():
            print(f"  {comparison['preferred']} vs {comparison['control']}: {verdict(comparison)}")
    print("HOLDOUT WAS NOT EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
