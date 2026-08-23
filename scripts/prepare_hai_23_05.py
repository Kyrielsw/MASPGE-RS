#!/usr/bin/env python3
"""Create the frozen two-unit HAI 23.05 forecasting contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROLE_NAMES = (
    "A1_energy_input", "A2_conversion",
    "A3_generation", "A4_auxiliary",
)


def end_indices(start: int, stop: int, history: int, horizon: int) -> np.ndarray:
    first = start + history - 1
    last = stop - horizon - 1
    if last < first:
        raise RuntimeError("HAI source segment is too short for the frozen window")
    return np.arange(first, last + 1, dtype=np.int64)


def required_columns(config: dict) -> list[str]:
    names = [*config["targets"].values(), *config["context_columns"]]
    for roles in config["role_slots"].values():
        for slots in roles.values():
            names.extend(name for name in slots if name is not None)
    return list(dict.fromkeys(names))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resample_file(
    path: Path, columns: list[str], factor: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, usecols=["timestamp", *columns])
    if len(frame) % factor:
        raise RuntimeError(
            f"{path.name} rows={len(frame)} is not divisible by {factor}"
        )
    timestamps = pd.to_datetime(frame.pop("timestamp"), errors="coerce")
    if bool(timestamps.isna().any()):
        raise RuntimeError(f"{path.name} contains invalid timestamps")
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    blocks = values.reshape(-1, factor, len(columns))
    finite_count = np.isfinite(blocks).sum(axis=1)
    totals = np.nansum(blocks, axis=1)
    means = np.divide(
        totals, finite_count, out=np.full_like(totals, np.nan),
        where=finite_count > 0,
    )
    block_timestamps = timestamps.iloc[factor - 1::factor].to_numpy()
    return means, finite_count == factor, block_timestamps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/hai_23_05_role_state_contract_v1.json",
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--admission-audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_dir = args.raw_dir or ROOT / config["raw_dir"]
    admission_path = args.admission_audit or ROOT / config["admission_audit"]
    output = args.output or ROOT / config["output"]
    audit_output = args.audit or ROOT / config["output_audit"]
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if not admission.get("admitted_for_contract_freeze", False):
        raise RuntimeError("HAI admission did not pass; preprocessing is forbidden")
    if admission.get("model_training_performed") or admission.get("holdout_model_evaluated"):
        raise RuntimeError("HAI admission audit violates the preparation protocol")

    factor = int(config["resample_seconds"])
    if factor <= 0:
        raise ValueError("resample_seconds must be positive")
    columns = required_columns(config)
    column_index = {name: index for index, name in enumerate(columns)}
    ordered_files = [
        name for split in ("train", "validation", "holdout")
        for name in config["split_files"][split]
    ]
    admitted_hashes = {
        Path(item["path"]).name: item["sha256"]
        for item in admission["file_audits"]
    }
    segments: dict[str, np.ndarray] = {}
    complete: dict[str, np.ndarray] = {}
    segment_timestamps: dict[str, np.ndarray] = {}
    for name in ordered_files:
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != admitted_hashes.get(name):
            raise RuntimeError(f"{name} differs from the admitted raw file")
        segments[name], complete[name], segment_timestamps[name] = resample_file(
            path, columns, factor
        )

    train_names = config["split_files"]["train"]
    training_raw = np.concatenate([segments[name] for name in train_names], axis=0)
    training_medians = np.nanmedian(training_raw, axis=0)
    training_medians = np.where(np.isfinite(training_medians), training_medians, 0.0)
    filled: dict[str, np.ndarray] = {}
    for name in ordered_files:
        frame = pd.DataFrame(segments[name]).ffill()
        frame = frame.fillna(pd.Series(training_medians, index=frame.columns))
        filled[name] = frame.to_numpy(np.float64)
    training_filled = np.concatenate([filled[name] for name in train_names], axis=0)
    feature_mean = training_filled.mean(axis=0, dtype=np.float64)
    feature_scale = training_filled.std(axis=0, dtype=np.float64)
    feature_scale = np.where(feature_scale > 1e-6, feature_scale, 1.0)
    normalized = {
        name: ((filled[name] - feature_mean) / feature_scale).astype(np.float32)
        for name in ordered_files
    }

    values = np.concatenate([normalized[name] for name in ordered_files], axis=0)
    validity = np.concatenate([complete[name] for name in ordered_files], axis=0)
    units = config["unit_order"]
    time_steps = len(values)
    x_role = np.zeros((time_steps, len(units), 4, 6), dtype=np.float32)
    feature_slot_mask = np.zeros((len(units), 4, 6), dtype=np.bool_)
    for unit_index, unit in enumerate(units):
        for role_index, role in enumerate(ROLE_NAMES):
            slots = config["role_slots"][unit][role]
            if len(slots) != 6:
                raise RuntimeError(f"{unit}/{role} must define exactly six slots")
            for slot_index, field in enumerate(slots):
                if field is not None:
                    x_role[:, unit_index, role_index, slot_index] = values[
                        :, column_index[field]
                    ]
                    feature_slot_mask[unit_index, role_index, slot_index] = True

    power = np.stack(
        [values[:, column_index[config["targets"][unit]]] for unit in units],
        axis=1,
    ).astype(np.float32)
    power_valid = np.stack(
        [validity[:, column_index[config["targets"][unit]]] for unit in units],
        axis=1,
    ).astype(np.bool_)
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name in ordered_files:
        offsets[name] = (cursor, cursor + len(normalized[name]))
        cursor += len(normalized[name])
    history = int(config["history_steps"])
    horizon = int(config["horizon_steps"])
    split_indices = {}
    for split, names in config["split_files"].items():
        split_indices[split] = np.concatenate([
            end_indices(*offsets[name], history, horizon) for name in names
        ])

    timeline = np.concatenate([segment_timestamps[name] for name in ordered_files])
    seconds = pd.DatetimeIndex(timeline).hour * 3600
    seconds += pd.DatetimeIndex(timeline).minute * 60
    seconds += pd.DatetimeIndex(timeline).second
    context = np.stack(
        [
            np.sin(2 * np.pi * seconds / 86400),
            np.cos(2 * np.pi * seconds / 86400),
            values[:, column_index[config["context_columns"][0]]],
        ], axis=1,
    ).astype(np.float32)
    target_indices = [column_index[config["targets"][unit]] for unit in units]
    payload = {
        "dataset_id": np.asarray(config["dataset_id"]),
        "x_role": x_role,
        "context": context,
        "power": power,
        "power_valid": power_valid,
        "train_end_indices": split_indices["train"],
        "validation_end_indices": split_indices["validation"],
        "holdout_end_indices": split_indices["holdout"],
        "feature_slot_mask": feature_slot_mask,
        "role_feature_counts": np.asarray([6, 6, 6, 6], dtype=np.int64),
        "active_role_mask": np.ones((len(units), 4), dtype=np.bool_),
        "target_mean": feature_mean[target_indices].astype(np.float32),
        "target_scale": feature_scale[target_indices].astype(np.float32),
        "coordinate_xy": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "unit_names": np.asarray(units),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "training_medians": training_medians.astype(np.float32),
    }
    if not all(np.isfinite(value).all() for value in (x_role, context, power)):
        raise RuntimeError("processed HAI contains non-finite model inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    audit = {
        "dataset_id": config["dataset_id"],
        "source_admission_audit": str(admission_path),
        "model_training_performed": False,
        "holdout_model_evaluated": False,
        "raw_sha256": admitted_hashes,
        "time_steps": time_steps,
        "unit_names": units,
        "x_role_shape": list(x_role.shape),
        "sampling_seconds": factor,
        "history_steps": history,
        "horizon_steps": horizon,
        "history_minutes": history * factor / 60,
        "horizon_minutes": horizon * factor / 60,
        "file_resampled_steps": {name: len(normalized[name]) for name in ordered_files},
        "split_windows": {name: len(indices) for name, indices in split_indices.items()},
        "target_valid_fraction": {
            unit: float(power_valid[:, index].mean())
            for index, unit in enumerate(units)
        },
        "active_roles": list(ROLE_NAMES),
        "a4_slots": {
            unit: [name for name in config["role_slots"][unit]["A4_auxiliary"] if name]
            for unit in units
        },
        "role_slots": config["role_slots"],
        "imputation": config["protocol"]["causal_imputation"],
        "normalization": config["protocol"]["normalization"],
        "window_boundary": config["protocol"]["window_boundary"],
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
