#!/usr/bin/env python3
"""Build the frozen 134-turbine SDWPF contract for Experiment 037.

The transformation is causal: missing history is forward-filled only, then
filled with a training-period median.  Original invalid targets remain masked
and are never scored as observed ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_RAW_MD5 = "8f0cc58c4b0d0fb809824035d05b4e76"
EXPECTED_LOCATION_MD5 = "08bccca55c6cb2c7066cf3adb4f7aae9"
UNITS = 134
DAYS = 245
STEPS_PER_DAY = 144
TRAIN_DAYS = 171
VALIDATION_DAYS = 37
HISTORY_STEPS = 60
HORIZON_STEPS = 15
SENSOR_COLUMNS = (
    "Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3",
    "Prtv", "Patv",
)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_minute(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def split_bounds() -> tuple[tuple[int, int], ...]:
    train_end = TRAIN_DAYS * STEPS_PER_DAY
    validation_end = (TRAIN_DAYS + VALIDATION_DAYS) * STEPS_PER_DAY
    total = DAYS * STEPS_PER_DAY
    return ((0, train_end), (train_end, validation_end), (validation_end, total))


def end_indices(start: int, stop: int) -> np.ndarray:
    first = start + HISTORY_STEPS - 1
    last = stop - HORIZON_STEPS - 1
    return np.arange(first, last + 1, dtype=np.int64)


def causal_fill(values: np.ndarray, train_stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward-fill each turbine inside each split, then use train medians."""

    filled = values.astype(np.float32, copy=True)
    medians = np.nanmedian(filled[:train_stop], axis=(0, 1))
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    for start, stop in split_bounds():
        frame = pd.DataFrame(filled[start:stop].reshape(stop - start, -1))
        frame = frame.ffill().fillna(
            pd.Series(np.tile(medians, UNITS), index=frame.columns)
        )
        filled[start:stop] = frame.to_numpy(np.float32).reshape(
            stop - start, UNITS, len(SENSOR_COLUMNS)
        )
    return filled, medians


def standardize(features: np.ndarray, train_stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[:train_stop].mean(axis=(0, 1), dtype=np.float64)
    scale = features[:train_stop].std(axis=(0, 1), dtype=np.float64)
    scale = np.where(scale > 1e-6, scale, 1.0)
    normalized = ((features - mean) / scale).astype(np.float32)
    return normalized, mean.astype(np.float32), scale.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--locations", type=Path, required=True)
    parser.add_argument(
        "--contract", choices=["compact_v1", "dense_role_v2"],
        default="compact_v1",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--skip-md5", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(
            "data/sdwpf_kdd_245days_v1.npz"
            if args.contract == "compact_v1"
            else "data/sdwpf_kdd_245days_role_state_v2.npz"
        )
    if args.audit is None:
        args.audit = Path(
            "data/sdwpf_kdd_245days_v1.audit.json"
            if args.contract == "compact_v1"
            else "data/sdwpf_kdd_245days_role_state_v2.audit.json"
        )

    if not args.skip_md5:
        if md5(args.csv) != EXPECTED_RAW_MD5:
            raise RuntimeError("SDWPF CSV MD5 mismatch")
        if md5(args.locations) != EXPECTED_LOCATION_MD5:
            raise RuntimeError("SDWPF location MD5 mismatch")

    frame = pd.read_csv(args.csv)
    required = {"TurbID", "Day", "Tmstamp", *SENSOR_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"raw SDWPF columns missing: {sorted(missing)}")
    frame = frame.loc[:, ["TurbID", "Day", "Tmstamp", *SENSOR_COLUMNS]].copy()
    frame["minute"] = frame["Tmstamp"].map(parse_minute)
    if bool((frame["minute"] % 10 != 0).any()):
        raise RuntimeError("SDWPF contains non-10-minute timestamps")
    frame["position"] = (frame["Day"].astype(int) - 1) * STEPS_PER_DAY + frame["minute"] // 10
    frame.sort_values(["TurbID", "position"], inplace=True)
    expected_rows = UNITS * DAYS * STEPS_PER_DAY
    if len(frame) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, found {len(frame)}")
    turbine_ids = np.sort(frame["TurbID"].unique())
    if len(turbine_ids) != UNITS:
        raise RuntimeError(f"expected {UNITS} turbines, found {len(turbine_ids)}")
    positions = frame["position"].to_numpy().reshape(UNITS, -1)
    expected_positions = np.arange(DAYS * STEPS_PER_DAY)
    if not bool(np.all(positions == expected_positions[None, :])):
        raise RuntimeError("each turbine must contain a complete aligned 245-day grid")

    raw = frame.loc[:, list(SENSOR_COLUMNS)].to_numpy(np.float32).reshape(
        UNITS, DAYS * STEPS_PER_DAY, len(SENSOR_COLUMNS)
    ).transpose(1, 0, 2)
    column = {name: index for index, name in enumerate(SENSOR_COLUMNS)}
    patv = raw[:, :, column["Patv"]]
    wspd = raw[:, :, column["Wspd"]]
    target_valid = np.isfinite(patv) & (patv >= 0)
    target_valid &= ~(np.isfinite(wspd) & (wspd > 2.5) & (patv <= 0))
    target_valid &= ~(
        (raw[:, :, column["Pab1"]] > 89)
        | (raw[:, :, column["Pab2"]] > 89)
        | (raw[:, :, column["Pab3"]] > 89)
    )
    target_valid &= np.isfinite(raw[:, :, column["Wdir"]])
    target_valid &= np.abs(raw[:, :, column["Wdir"]]) <= 180
    target_valid &= np.isfinite(raw[:, :, column["Ndir"]])
    target_valid &= np.abs(raw[:, :, column["Ndir"]]) <= 720

    # Do not feed known-bad target/sensor values back into later histories.
    raw[:, :, column["Patv"]][~target_valid] = np.nan
    raw[:, :, column["Wspd"]][raw[:, :, column["Wspd"]] < 0] = np.nan
    raw[:, :, column["Wdir"]][np.abs(raw[:, :, column["Wdir"]]) > 180] = np.nan
    raw[:, :, column["Ndir"]][np.abs(raw[:, :, column["Ndir"]]) > 720] = np.nan
    for name in ("Pab1", "Pab2", "Pab3"):
        raw[:, :, column[name]][raw[:, :, column[name]] > 89] = np.nan

    train_stop = TRAIN_DAYS * STEPS_PER_DAY
    filled, train_medians = causal_fill(raw, train_stop)
    normalized, feature_mean, feature_scale = standardize(filled, train_stop)

    wind_direction = np.deg2rad(filled[:, :, column["Wdir"]])
    nacelle_direction = np.deg2rad(filled[:, :, column["Ndir"]])
    derived = {
        "Wspd": normalized[:, :, column["Wspd"]],
        "WdirSin": np.sin(wind_direction),
        "WdirCos": np.cos(wind_direction),
        "Etmp": normalized[:, :, column["Etmp"]],
        "Itmp": normalized[:, :, column["Itmp"]],
        "NdirSin": np.sin(nacelle_direction),
        "NdirCos": np.cos(nacelle_direction),
        "Pab1": normalized[:, :, column["Pab1"]],
        "Pab2": normalized[:, :, column["Pab2"]],
        "Pab3": normalized[:, :, column["Pab3"]],
        "Prtv": normalized[:, :, column["Prtv"]],
        "Patv": normalized[:, :, column["Patv"]],
    }
    roles = (
        ("Wspd", "WdirSin", "WdirCos", "Etmp"),
        ("Itmp", "NdirSin", "NdirCos", "Pab1", "Pab2", "Pab3"),
        ("Prtv", "Patv"),
        (),
    )
    if args.contract == "compact_v1":
        x_role = np.zeros((DAYS * STEPS_PER_DAY, 4), dtype=np.float32)
        for role_index, names in enumerate(roles):
            if names:
                x_role[:, role_index] = np.stack(
                    [derived[name] for name in names], axis=-1
                ).mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
        dataset_id = "sdwpf_kdd_245days_v1"
    else:
        maximum_role_features = max(len(names) for names in roles)
        x_role = np.zeros(
            (DAYS * STEPS_PER_DAY, UNITS, 4, maximum_role_features),
            dtype=np.float32,
        )
        for role_index, names in enumerate(roles):
            if names:
                x_role[:, :, role_index, : len(names)] = np.stack(
                    [derived[name] for name in names], axis=-1
                ).astype(np.float32)
        dataset_id = "sdwpf_kdd_245days_role_state_v2"

    timeline = np.arange(DAYS * STEPS_PER_DAY)
    step_of_day = timeline % STEPS_PER_DAY
    day = timeline // STEPS_PER_DAY
    context = np.stack(
        [
            np.sin(2 * np.pi * step_of_day / STEPS_PER_DAY),
            np.cos(2 * np.pi * step_of_day / STEPS_PER_DAY),
            np.sin(2 * np.pi * day / 7),
            np.cos(2 * np.pi * day / 7),
            derived["Wspd"].mean(axis=1),
            derived["Etmp"].mean(axis=1),
            derived["Patv"].mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)

    power = normalized[:, :, column["Patv"]].astype(np.float32)
    feature_slot_mask = np.zeros((UNITS, 4, 6), dtype=np.bool_)
    for role_index, names in enumerate(roles):
        feature_slot_mask[:, role_index, : len(names)] = True
    locations = pd.read_csv(args.locations).sort_values("TurbID")
    if locations.shape[0] != UNITS or not {"x", "y"}.issubset(locations.columns):
        raise RuntimeError("unexpected SDWPF location table")
    coordinates = locations[["x", "y"]].to_numpy(np.float32)

    train_bounds, validation_bounds, holdout_bounds = split_bounds()
    payload = {
        "dataset_id": np.asarray(dataset_id),
        "x_role": x_role,
        "context": context,
        "power": power,
        "power_valid": target_valid.astype(np.bool_),
        "train_end_indices": end_indices(*train_bounds),
        "validation_end_indices": end_indices(*validation_bounds),
        "holdout_end_indices": end_indices(*holdout_bounds),
        "feature_slot_mask": feature_slot_mask,
        "active_role_mask": np.asarray([True, True, True, False]),
        "target_mean": np.asarray([feature_mean[column["Patv"]]], dtype=np.float32),
        "target_scale": np.asarray([feature_scale[column["Patv"]]], dtype=np.float32),
        "coordinate_xy": coordinates,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "train_medians": train_medians,
    }
    if not all(np.isfinite(value).all() for value in (x_role, context, power)):
        raise RuntimeError("processed SDWPF contains non-finite model inputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    audit = {
        "dataset_id": dataset_id,
        "role_contract": args.contract,
        "x_role_shape": list(x_role.shape),
        "raw_csv_md5": md5(args.csv),
        "locations_md5": md5(args.locations),
        "rows": len(frame),
        "time_steps": DAYS * STEPS_PER_DAY,
        "unit_count": UNITS,
        "sampling_minutes": 10,
        "history_steps": HISTORY_STEPS,
        "horizon_steps": HORIZON_STEPS,
        "split_days": {"train": 171, "validation": 37, "holdout": 37},
        "split_windows": {
            "train": len(payload["train_end_indices"]),
            "validation": len(payload["validation_end_indices"]),
            "holdout": len(payload["holdout_end_indices"]),
        },
        "valid_target_fraction": float(target_valid.mean()),
        "active_roles": ["A1_energy_input", "A2_conversion", "A3_generation"],
        "inactive_roles": ["A4_auxiliary"],
        "imputation": "split-local causal forward fill then training median",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
