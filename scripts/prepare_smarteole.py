#!/usr/bin/env python3
"""Prepare and accept the SMARTEOLE S1 role-aware data layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from maspge.smarteole import prepare_smarteole_s1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/smarteole_role_contract_v1.json",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=ROOT / "data/s1_complete_case_v0.1.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/smarteole_split_manifest_v1.json",
    )
    parser.add_argument(
        "--acceptance-output",
        type=Path,
        default=ROOT
        / "data/smarteole_acceptance_v1.json",
    )
    return parser.parse_args()


def assert_windows_stay_in_runs(
    indices: np.ndarray,
    run_ids: np.ndarray,
    timestamps_minute: np.ndarray,
    *,
    history_steps: int,
    horizon_steps: int,
) -> None:
    for end in indices:
        start = int(end) - history_steps + 1
        target_end = int(end) + horizon_steps
        assert start >= 0
        assert target_end < len(run_ids)
        assert run_ids[start] == run_ids[end] == run_ids[target_end]
        timestamp_slice = timestamps_minute[start : target_end + 1]
        assert np.all(np.diff(timestamp_slice) == 1)


def main() -> int:
    args = parse_args()
    manifest = prepare_smarteole_s1(
        args.config, args.output_npz, args.manifest
    )

    with np.load(args.output_npz, allow_pickle=False) as data:
        x_role = data["x_role"]
        context = data["context"]
        power = data["power"]
        candidate_edges = data["candidate_edges"]
        run_ids = data["run_ids"]
        timestamps = data["timestamps_minute"]
        active_role_mask = data["active_role_mask"]
        feature_slot_mask = data["feature_slot_mask"]
        role_zscore_mask = data["role_zscore_mask"]
        context_zscore_mask = data["context_zscore_mask"].astype(bool)
        split_indices = {
            "train": data["train_end_indices"],
            "validation": data["validation_end_indices"],
            "holdout": data["holdout_end_indices"],
        }

        history_steps = int(manifest["window"]["history_steps"])
        horizon_steps = int(manifest["window"]["horizon_steps"])
        for indices in split_indices.values():
            assert_windows_stay_in_runs(
                indices,
                run_ids,
                timestamps,
                history_steps=history_steps,
                horizon_steps=horizon_steps,
            )

        split_run_sets = {
            split: set(run_ids[indices].tolist())
            for split, indices in split_indices.items()
        }
        assert not split_run_sets["train"] & split_run_sets["validation"]
        assert not split_run_sets["train"] & split_run_sets["holdout"]
        assert not split_run_sets["validation"] & split_run_sets["holdout"]
        split_time_ordered = bool(
            timestamps[split_indices["train"]].max()
            < timestamps[split_indices["validation"]].min()
            and timestamps[split_indices["validation"]].max()
            < timestamps[split_indices["holdout"]].min()
        )

        train_run_ids = manifest["split"]["run_ids"]["train"]
        train_rows = np.isin(run_ids, train_run_ids)
        role_train_means = {}
        role_zscore_train_means = []
        role_passthrough_values = []
        role_names = manifest["tensor_contract"]["roles"]
        for role_index, role_name in enumerate(role_names):
            slot_means = []
            for feature_index in range(x_role.shape[-1]):
                if not feature_slot_mask[:, role_index, feature_index].any():
                    continue
                slot_means.append(
                    float(
                        np.mean(
                            x_role[
                                train_rows, :, role_index, feature_index
                            ],
                            dtype=np.float64,
                        )
                    )
                )
                if role_zscore_mask[role_index, feature_index]:
                    role_zscore_train_means.append(slot_means[-1])
                else:
                    role_passthrough_values.append(
                        x_role[:, :, role_index, feature_index].reshape(-1)
                    )
            role_train_means[role_name] = slot_means

        if role_passthrough_values:
            role_passthrough = np.concatenate(role_passthrough_values)
        else:
            role_passthrough = np.array([], dtype=np.float32)
        context_passthrough = context[:, ~context_zscore_mask]
        context_train_mean = np.mean(
            context[train_rows][:, context_zscore_mask],
            axis=0,
            dtype=np.float64,
        )

        checks = {
            "finite_role_tensor": bool(np.isfinite(x_role).all()),
            "finite_context_tensor": bool(np.isfinite(context).all()),
            "finite_power_tensor": bool(np.isfinite(power).all()),
            "a4_mask_all_zero": bool((active_role_mask[:, 3] == 0).all()),
            "a4_values_all_zero": bool((x_role[:, :, 3, :] == 0).all()),
            "candidate_diagonal_all_zero": bool(
                np.trace(candidate_edges, axis1=1, axis2=2).sum() == 0
            ),
            "role_normalized_absmax_below_50": bool(
                np.abs(x_role).max() < 50.0
            ),
            "context_normalized_absmax_below_50": bool(
                np.abs(context).max() < 50.0
            ),
            "power_normalized_absmax_below_20": bool(
                np.abs(power).max() < 20.0
            ),
            "train_context_mean_near_zero": bool(
                np.abs(context_train_mean).max() < 1e-4
            ),
            "train_power_mean_near_zero": bool(
                abs(float(np.mean(power[train_rows], dtype=np.float64)))
                < 1e-4
            ),
            "train_role_means_near_zero": bool(
                all(
                    abs(value) < 1e-4
                    for value in role_zscore_train_means
                )
            ),
            "role_passthrough_values_in_zero_one": bool(
                len(role_passthrough) == 0
                or (
                    role_passthrough.min() >= 0.0
                    and role_passthrough.max() <= 1.0
                )
            ),
            "context_passthrough_values_in_zero_one": bool(
                context_passthrough.size == 0
                or (
                    context_passthrough.min() >= 0.0
                    and context_passthrough.max() <= 1.0
                )
            ),
            "split_run_sets_disjoint": True,
            "splits_strictly_chronological": split_time_ordered,
            "all_windows_run_safe": True,
            "all_splits_nonempty": bool(
                all(len(indices) > 0 for indices in split_indices.values())
            ),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise AssertionError(f"S1 acceptance checks failed: {failed}")

        acceptance = {
            "dataset": manifest["dataset"],
            "status": "S1_accepted",
            "config": str(args.config),
            "processed_file": manifest["processed_file"],
            "manifest": str(args.manifest),
            "checks": checks,
            "shapes": {
                "x_role": list(x_role.shape),
                "context": list(context.shape),
                "power": list(power.shape),
                "candidate_edges": list(candidate_edges.shape),
            },
            "window_counts": {
                split: int(len(indices))
                for split, indices in split_indices.items()
            },
            "role_train_means_after_normalization": role_train_means,
            "context_train_mean_max_abs": float(
                np.abs(context_train_mean).max()
            ),
            "power_train_mean_abs": abs(
                float(np.mean(power[train_rows], dtype=np.float64))
            ),
            "normalized_absolute_maxima": {
                "x_role": float(np.abs(x_role).max()),
                "context": float(np.abs(context).max()),
                "power": float(np.abs(power).max()),
            },
        }

    manifest["status"] = "S1_accepted"
    temporary_manifest = args.manifest.with_suffix(
        args.manifest.suffix + ".tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_manifest.replace(args.manifest)

    args.acceptance_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.acceptance_output.with_suffix(
        args.acceptance_output.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(args.acceptance_output)

    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
