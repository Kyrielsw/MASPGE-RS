# -*- coding: utf-8 -*-
"""SMARTEOLE S1 role-aware, run-safe data preparation.

This module is intentionally separate from the legacy flat-agent loader used
by HAI and TEP.  It prepares a fixed role tensor for homogeneous generation
units without changing the already-validated legacy data semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROLE_ORDER = ("A1", "A2", "A3", "A4")
SCADA_AVERAGE_VARIABLES = (
    "active_power",
    "blade_1_pitch_angle",
    "wind_speed",
    "nacelle_position",
    "wind_vane",
    "wind_direction",
    "generator_speed",
    "temperature",
)


@dataclass(frozen=True)
class RunRecord:
    run_id: int
    start_index: int
    end_index_exclusive: int
    start_time: str
    end_time: str
    steps: int
    window_capacity: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def angle_sin_degrees(values: np.ndarray) -> np.ndarray:
    return np.sin(np.deg2rad(values))


def angle_cos_degrees(values: np.ndarray) -> np.ndarray:
    return np.cos(np.deg2rad(values))


def circular_mean_degrees(values: np.ndarray) -> np.ndarray:
    """Circular mean over the last axis, returned in [0, 360)."""
    radians = np.deg2rad(values)
    with np.errstate(invalid="ignore"):
        sin_mean = np.nanmean(np.sin(radians), axis=-1)
        cos_mean = np.nanmean(np.cos(radians), axis=-1)
    result = np.mod(np.rad2deg(np.arctan2(sin_mean, cos_mean)), 360.0)
    result[np.isnan(values).all(axis=-1)] = np.nan
    return result


def angular_difference_degrees(left: np.ndarray, right: float) -> np.ndarray:
    return np.abs((left - right + 180.0) % 360.0 - 180.0)


def build_wake_candidate_edges(
    wind_from_degrees: np.ndarray,
    coordinates_xy: np.ndarray,
    *,
    rotor_diameter_m: float,
    maximum_distance_rotor_diameters: float,
    half_angle_degrees: float,
) -> np.ndarray:
    """Return boolean adjacency with convention ``[time, receiver, sender]``.

    Wind direction uses the meteorological convention: it indicates where the
    wind comes from.  The downwind bearing is therefore ``wind + 180°``.
    X is east and Y is north; bearings are clockwise from north.
    """
    wind_from = np.asarray(wind_from_degrees, dtype=np.float64)
    coordinates = np.asarray(coordinates_xy, dtype=np.float64)
    if wind_from.ndim != 1:
        raise ValueError("wind_from_degrees must have shape [time]")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates_xy must have shape [unit, 2]")

    n_rows = len(wind_from)
    n_units = len(coordinates)
    adjacency = np.zeros((n_rows, n_units, n_units), dtype=np.uint8)
    downwind = np.mod(wind_from + 180.0, 360.0)
    maximum_distance = rotor_diameter_m * maximum_distance_rotor_diameters
    valid_direction = np.isfinite(wind_from)

    for sender in range(n_units):
        for receiver in range(n_units):
            if sender == receiver:
                continue
            dx = coordinates[receiver, 0] - coordinates[sender, 0]
            dy = coordinates[receiver, 1] - coordinates[sender, 1]
            distance = math.hypot(dx, dy)
            if distance > maximum_distance:
                continue
            bearing = math.degrees(math.atan2(dx, dy)) % 360.0
            active = (
                valid_direction
                & (
                    angular_difference_degrees(downwind, bearing)
                    <= half_angle_degrees
                )
            )
            adjacency[:, receiver, sender] = active.astype(np.uint8)

    return adjacency


def assign_contiguous_run_ids(
    timestamps: np.ndarray,
    *,
    sampling_seconds: int,
) -> np.ndarray:
    """Assign a new run whenever adjacent timestamps are not exactly sampled."""
    values = np.asarray(timestamps).astype("datetime64[s]").astype(np.int64)
    if values.ndim != 1:
        raise ValueError("timestamps must have shape [time]")
    if len(values) == 0:
        return np.empty(0, dtype=np.int32)
    differences = np.diff(values)
    breaks = np.concatenate(
        [np.array([False]), differences != int(sampling_seconds)]
    )
    return np.cumsum(breaks, dtype=np.int32)


def enumerate_runs(
    run_ids: np.ndarray,
    timestamps: np.ndarray,
    *,
    history_steps: int,
    horizon_steps: int,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    if len(run_ids) == 0:
        return records
    unique_ids, starts = np.unique(run_ids, return_index=True)
    ends = np.concatenate([starts[1:], np.array([len(run_ids)])])
    for run_id, start, end in zip(unique_ids, starts, ends):
        steps = int(end - start)
        capacity = max(0, steps - history_steps - horizon_steps + 1)
        records.append(
            RunRecord(
                run_id=int(run_id),
                start_index=int(start),
                end_index_exclusive=int(end),
                start_time=str(
                    np.asarray(timestamps)[start].astype("datetime64[m]")
                ),
                end_time=str(
                    np.asarray(timestamps)[end - 1].astype("datetime64[m]")
                ),
                steps=steps,
                window_capacity=capacity,
            )
        )
    return records


def _nearest_boundary(
    cumulative: np.ndarray,
    target: float,
    *,
    minimum_boundary: int,
    maximum_boundary: int,
) -> int:
    """Choose a number of leading runs, keeping all three splits non-empty."""
    candidates = np.arange(minimum_boundary, maximum_boundary + 1)
    achieved = cumulative[candidates - 1]
    return int(candidates[np.argmin(np.abs(achieved - target))])


def split_runs_chronologically(
    records: list[RunRecord],
    ratios: Iterable[float],
) -> dict[str, tuple[int, ...]]:
    """Split whole chronological runs to approximate window-count ratios."""
    ratios_tuple = tuple(float(value) for value in ratios)
    if len(ratios_tuple) != 3 or not np.isclose(sum(ratios_tuple), 1.0):
        raise ValueError("ratios must contain three values summing to 1")
    if len(records) < 3:
        raise ValueError("At least three runs are required for train/val/holdout")

    capacities = np.array([record.window_capacity for record in records], dtype=np.int64)
    total = int(capacities.sum())
    if total <= 0:
        raise ValueError("No valid windows are available")
    cumulative = np.cumsum(capacities)
    first = _nearest_boundary(
        cumulative,
        total * ratios_tuple[0],
        minimum_boundary=1,
        maximum_boundary=len(records) - 2,
    )
    second = _nearest_boundary(
        cumulative,
        total * (ratios_tuple[0] + ratios_tuple[1]),
        minimum_boundary=first + 1,
        maximum_boundary=len(records) - 1,
    )

    result = {
        "train": tuple(record.run_id for record in records[:first]),
        "validation": tuple(record.run_id for record in records[first:second]),
        "holdout": tuple(record.run_id for record in records[second:]),
    }
    for split_name, run_id_values in result.items():
        split_capacity = sum(
            record.window_capacity
            for record in records
            if record.run_id in run_id_values
        )
        if split_capacity <= 0:
            raise ValueError(f"{split_name} contains no valid windows")
    return result


def build_window_end_indices(
    records: list[RunRecord],
    selected_run_ids: Iterable[int],
    *,
    history_steps: int,
    horizon_steps: int,
) -> np.ndarray:
    selected = set(int(run_id) for run_id in selected_run_ids)
    parts = []
    for record in records:
        if record.run_id not in selected or record.window_capacity == 0:
            continue
        first_end = record.start_index + history_steps - 1
        stop_exclusive = record.end_index_exclusive - horizon_steps
        parts.append(np.arange(first_end, stop_exclusive, dtype=np.int64))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def fit_mean_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _transform(values: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "identity":
        return values.astype(np.float32, copy=False)
    if transform_name == "sin_degrees":
        return angle_sin_degrees(values).astype(np.float32)
    if transform_name == "cos_degrees":
        return angle_cos_degrees(values).astype(np.float32)
    raise ValueError(f"Unsupported transform: {transform_name}")


def _timestamp_feature(
    timestamps: pd.Series,
    transform_name: str,
) -> np.ndarray:
    if transform_name.startswith("minute_of_day_"):
        phase = (
            timestamps.dt.hour.to_numpy() * 60
            + timestamps.dt.minute.to_numpy()
        ) / 1440.0 * (2.0 * np.pi)
    elif transform_name.startswith("day_of_year_"):
        phase = (
            timestamps.dt.dayofyear.to_numpy() - 1
        ) / 366.0 * (2.0 * np.pi)
    else:
        raise ValueError(f"Unsupported timestamp transform: {transform_name}")
    if transform_name.endswith("_sin"):
        return np.sin(phase).astype(np.float32)
    if transform_name.endswith("_cos"):
        return np.cos(phase).astype(np.float32)
    raise ValueError(f"Unsupported timestamp transform: {transform_name}")


def _raw_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _load_merged_frame(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Path], list[str]]:
    raw_paths = {
        name: Path(path) for name, path in config["raw_files"].items()
    }
    units = config["units"]
    unit_numbers = list(range(1, len(units) + 1))

    complete_average_columns = [
        f"{variable}_{number}_avg"
        for variable in SCADA_AVERAGE_VARIABLES
        for number in unit_numbers
    ]
    role_source_columns = []
    for role in config["roles"].values():
        for feature in role.get("features", []):
            role_source_columns.extend(
                feature["source"].format(unit_number=number)
                for number in unit_numbers
            )
    scada_columns = list(
        dict.fromkeys(["time"] + complete_average_columns + role_source_columns)
    )
    scada = pd.read_csv(
        raw_paths["scada"],
        usecols=scada_columns,
        na_values=["NA"],
    )

    control_sources = [
        item["source"]
        for item in config["shared_context"]
        if item["source_file"] == "control_log"
    ]
    control = pd.read_csv(
        raw_paths["control_log"],
        usecols=["time"] + control_sources,
        na_values=["NA"],
    )
    windcube_sources = [
        item["source"]
        for item in config["shared_context"]
        if item["source_file"] == "windcube"
    ]
    windcube = pd.read_csv(
        raw_paths["windcube"],
        usecols=["time"] + windcube_sources,
        na_values=["NA"],
    )

    for frame in (scada, control, windcube):
        frame["time"] = pd.to_datetime(frame["time"])
        if frame["time"].duplicated().any():
            raise ValueError("A temporal source contains duplicate timestamps")

    merged = scada.merge(
        control, on="time", how="left", validate="one_to_one"
    ).merge(
        windcube, on="time", how="left", validate="one_to_one"
    )
    merged = merged.sort_values("time").reset_index(drop=True)
    return merged, raw_paths, complete_average_columns


def prepare_smarteole_s1(
    config_path: Path,
    output_npz: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Prepare normalized role tensors and leak-free split indices."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    merged, raw_paths, complete_average_columns = _load_merged_frame(config)

    units = tuple(config["units"])
    roles = ROLE_ORDER
    n_rows = len(merged)
    n_units = len(units)
    max_features = max(
        len(config["roles"][role].get("features", [])) for role in roles
    )
    x_role = np.zeros(
        (n_rows, n_units, len(roles), max_features), dtype=np.float32
    )
    feature_slot_mask = np.zeros(
        (n_units, len(roles), max_features), dtype=np.uint8
    )
    role_zscore_mask = np.zeros(
        (len(roles), max_features), dtype=np.uint8
    )
    active_role_mask = np.zeros((n_units, len(roles)), dtype=np.uint8)
    role_feature_names: dict[str, list[str]] = {}
    physical_range_mask = np.ones(n_rows, dtype=bool)
    physical_range_violations: dict[str, int] = {}

    for role_index, role_id in enumerate(roles):
        role_config = config["roles"][role_id]
        features = role_config.get("features", [])
        role_feature_names[role_id] = [feature["name"] for feature in features]
        if role_config.get("active", False):
            active_role_mask[:, role_index] = 1
        for feature_index, feature in enumerate(features):
            if feature.get("normalization", "zscore") == "zscore":
                role_zscore_mask[role_index, feature_index] = 1
            for unit_index in range(n_units):
                source = feature["source"].format(
                    unit_number=unit_index + 1
                )
                values = merged[source].to_numpy(dtype=np.float64)
                if "valid_range" in feature:
                    lower, upper = map(float, feature["valid_range"])
                    finite = np.isfinite(values)
                    within = finite & (values >= lower) & (values <= upper)
                    physical_range_mask &= within
                    physical_range_violations[
                        f"{role_id}:{source}"
                    ] = int((finite & ~within).sum())
                x_role[:, unit_index, role_index, feature_index] = _transform(
                    values, feature["transform"]
                )
                feature_slot_mask[
                    unit_index, role_index, feature_index
                ] = 1

    timestamps_series = merged["time"]
    context_columns = []
    context_parts = []
    context_normalization = []
    raw_context_sources = []
    for item in config["shared_context"]:
        context_columns.append(item["name"])
        context_normalization.append(
            item.get("normalization", "zscore")
        )
        if item["source_file"] == "timestamp":
            values = _timestamp_feature(
                timestamps_series, item["transform"]
            )
        else:
            raw_context_sources.append(item["source"])
            raw_values = merged[item["source"]].to_numpy(dtype=np.float64)
            if "valid_range" in item:
                lower, upper = map(float, item["valid_range"])
                finite = np.isfinite(raw_values)
                within = (
                    finite
                    & (raw_values >= lower)
                    & (raw_values <= upper)
                )
                physical_range_mask &= within
                physical_range_violations[
                    f"{item['source_file']}:{item['source']}"
                ] = int((finite & ~within).sum())
            values = _transform(raw_values, item["transform"])
        context_parts.append(values)
    context = np.stack(context_parts, axis=1).astype(np.float32)

    power = np.stack(
        [
            merged[
                config["target"]["source"].format(unit_number=unit_index + 1)
            ].to_numpy(dtype=np.float32)
            for unit_index in range(n_units)
        ],
        axis=1,
    )
    direction_values = np.stack(
        [
            merged[f"wind_direction_{unit_index + 1}_avg"].to_numpy(
                dtype=np.float64
            )
            for unit_index in range(n_units)
        ],
        axis=1,
    )

    base_complete_mask = (
        merged[complete_average_columns].notna().all(axis=1).to_numpy()
    )
    if raw_context_sources:
        base_complete_mask &= (
            merged[raw_context_sources].notna().all(axis=1).to_numpy()
        )
    base_complete_mask &= np.isfinite(x_role[:, :, :3, :]).all(
        axis=(1, 2, 3)
    )
    base_complete_mask &= np.isfinite(context).all(axis=1)
    base_complete_mask &= np.isfinite(power).all(axis=1)
    complete_mask = base_complete_mask & physical_range_mask
    rows_excluded_by_physical_range = int(
        (base_complete_mask & ~physical_range_mask).sum()
    )

    original_rows = len(merged)
    original_start = str(merged["time"].iloc[0])
    original_end = str(merged["time"].iloc[-1])
    timestamps = (
        merged.loc[complete_mask, "time"]
        .to_numpy(dtype="datetime64[ns]")
    )
    x_role = x_role[complete_mask]
    context = context[complete_mask]
    power = power[complete_mask]
    direction_values = direction_values[complete_mask]

    if not (
        np.isfinite(x_role).all()
        and np.isfinite(context).all()
        and np.isfinite(power).all()
    ):
        raise ValueError("Complete-case tensors still contain non-finite values")

    run_ids = assign_contiguous_run_ids(
        timestamps, sampling_seconds=int(config["sampling_seconds"])
    )
    history_steps = int(config["window"]["history_steps"])
    horizon_steps = int(config["window"]["horizon_steps"])
    records = enumerate_runs(
        run_ids,
        timestamps,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
    )
    split_run_ids = split_runs_chronologically(
        records, config["split"]["ratios"]
    )
    split_indices = {
        split_name: build_window_end_indices(
            records,
            ids,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
        )
        for split_name, ids in split_run_ids.items()
    }

    train_row_mask = np.isin(run_ids, split_run_ids["train"])
    role_mean = np.zeros((len(roles), max_features), dtype=np.float32)
    role_scale = np.ones((len(roles), max_features), dtype=np.float32)
    normalized_role = np.zeros_like(x_role)
    for role_index in range(len(roles)):
        for feature_index in range(max_features):
            if not feature_slot_mask[:, role_index, feature_index].any():
                continue
            if not role_zscore_mask[role_index, feature_index]:
                normalized_role[:, :, role_index, feature_index] = x_role[
                    :, :, role_index, feature_index
                ]
                continue
            train_values = x_role[
                train_row_mask, :, role_index, feature_index
            ].reshape(-1)
            mean, scale = fit_mean_scale(train_values[:, None])
            role_mean[role_index, feature_index] = mean[0]
            role_scale[role_index, feature_index] = scale[0]
            normalized_role[:, :, role_index, feature_index] = (
                x_role[:, :, role_index, feature_index] - mean[0]
            ) / scale[0]

    context_mean = np.zeros(context.shape[1], dtype=np.float32)
    context_scale = np.ones(context.shape[1], dtype=np.float32)
    normalized_context = context.copy()
    context_zscore_mask = np.array(
        [policy == "zscore" for policy in context_normalization],
        dtype=np.uint8,
    )
    for context_index, use_zscore in enumerate(context_zscore_mask):
        if not use_zscore:
            continue
        mean, scale = fit_mean_scale(
            context[train_row_mask, context_index : context_index + 1]
        )
        context_mean[context_index] = mean[0]
        context_scale[context_index] = scale[0]
        normalized_context[:, context_index] = (
            context[:, context_index] - mean[0]
        ) / scale[0]
    target_mean, target_scale = fit_mean_scale(
        power[train_row_mask].reshape(-1, 1)
    )
    normalized_power = (
        (power - target_mean[0]) / target_scale[0]
    ).astype(np.float32)

    coordinates = pd.read_csv(raw_paths["coordinates"]).set_index("Turbine")
    coordinate_xy = np.stack(
        [
            coordinates.loc[list(units), "X_RGF93"].to_numpy(dtype=np.float64),
            coordinates.loc[list(units), "Y_RGF93"].to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    farm_direction = circular_mean_degrees(direction_values)
    graph_config = config["dynamic_candidate_graph"]
    candidate_edges = build_wake_candidate_edges(
        farm_direction,
        coordinate_xy,
        rotor_diameter_m=float(graph_config["rotor_diameter_m"]),
        maximum_distance_rotor_diameters=float(
            graph_config["maximum_distance_rotor_diameters"]
        ),
        half_angle_degrees=float(graph_config["half_angle_degrees"]),
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = output_npz.with_suffix(output_npz.suffix + ".tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(
            handle,
            x_role=normalized_role,
            context=normalized_context,
            power=normalized_power,
            candidate_edges=candidate_edges,
            timestamps_minute=timestamps.astype("datetime64[m]").astype(np.int64),
            run_ids=run_ids,
            train_end_indices=split_indices["train"],
            validation_end_indices=split_indices["validation"],
            holdout_end_indices=split_indices["holdout"],
            feature_slot_mask=feature_slot_mask,
            role_zscore_mask=role_zscore_mask,
            active_role_mask=active_role_mask,
            unit_mask=np.ones(n_units, dtype=np.uint8),
            role_mean=role_mean,
            role_scale=role_scale,
            context_mean=context_mean,
            context_scale=context_scale,
            context_zscore_mask=context_zscore_mask,
            target_mean=target_mean,
            target_scale=target_scale,
            coordinate_xy=coordinate_xy.astype(np.float32),
        )
    temporary_npz.replace(output_npz)

    total_windows = sum(record.window_capacity for record in records)
    manifest = {
        "dataset": config["dataset_name"],
        "schema_version": config["schema_version"],
        "status": "S1_prepared_pending_acceptance",
        "config_path": str(config_path),
        "raw_files": {
            name: _raw_file_record(path) for name, path in raw_paths.items()
        },
        "source_table": {
            "rows": original_rows,
            "start": original_start,
            "end": original_end,
        },
        "complete_case": {
            "rows": len(timestamps),
            "fraction_of_scada_rows": len(timestamps) / original_rows,
            "excluded_rows": original_rows - len(timestamps),
            "rows_before_physical_range_filter": int(
                base_complete_mask.sum()
            ),
            "rows_excluded_by_physical_range": rows_excluded_by_physical_range,
            "require_all_56_scada_average_fields": True,
            "require_all_context_fields": True,
        },
        "data_quality": {
            "physical_range_violations_by_source": {
                key: value
                for key, value in physical_range_violations.items()
                if value > 0
            },
            "physical_ranges_from_config": True,
        },
        "tensor_contract": {
            "x_role": list(normalized_role.shape),
            "context": list(normalized_context.shape),
            "power": list(normalized_power.shape),
            "candidate_edges": list(candidate_edges.shape),
            "candidate_edge_convention": "[time, receiver, sender]",
            "roles": list(roles),
            "units": list(units),
            "role_feature_names": role_feature_names,
            "active_role_mask": active_role_mask.tolist(),
            "feature_slot_mask": feature_slot_mask.tolist(),
            "role_zscore_mask": role_zscore_mask.tolist(),
        },
        "window": {
            "history_steps": history_steps,
            "horizon_steps": horizon_steps,
            "target_aggregation": config["window"]["target_aggregation"],
            "total_window_capacity": total_windows,
        },
        "runs": {
            "count": len(records),
            "records": [record.__dict__ for record in records],
        },
        "split": {
            "strategy": config["split"]["strategy"],
            "requested_ratios": config["split"]["ratios"],
            "run_ids": {
                key: list(value) for key, value in split_run_ids.items()
            },
            "window_counts": {
                key: int(len(value)) for key, value in split_indices.items()
            },
            "actual_window_ratios": {
                key: float(len(value) / total_windows)
                for key, value in split_indices.items()
            },
            "run_sets_disjoint": (
                not (
                    set(split_run_ids["train"])
                    & set(split_run_ids["validation"])
                    or set(split_run_ids["train"])
                    & set(split_run_ids["holdout"])
                    or set(split_run_ids["validation"])
                    & set(split_run_ids["holdout"])
                )
            ),
        },
        "normalization": {
            "fit_rows": int(train_row_mask.sum()),
            "fit_run_ids": list(split_run_ids["train"]),
            "role_mean": role_mean.tolist(),
            "role_scale": role_scale.tolist(),
            "context_columns": context_columns,
            "context_normalization": context_normalization,
            "context_zscore_mask": context_zscore_mask.tolist(),
            "context_mean": context_mean.tolist(),
            "context_scale": context_scale.tolist(),
            "target_mean": float(target_mean[0]),
            "target_scale": float(target_scale[0]),
        },
        "dynamic_candidate_graph": {
            "configuration": graph_config,
            "rows_with_any_edge": int(
                (candidate_edges.sum(axis=(1, 2)) > 0).sum()
            ),
            "fraction_rows_with_any_edge": float(
                (candidate_edges.sum(axis=(1, 2)) > 0).mean()
            ),
            "mean_edges_per_row": float(
                candidate_edges.sum(axis=(1, 2)).mean()
            ),
            "maximum_edges_per_row": int(
                candidate_edges.sum(axis=(1, 2)).max()
            ),
            "self_edges": int(
                np.trace(candidate_edges, axis1=1, axis2=2).sum()
            ),
        },
        "processed_file": {
            "path": str(output_npz),
            "bytes": output_npz.stat().st_size,
            "sha256": sha256(output_npz),
        },
        "acceptance_assertions": {
            "finite_role_tensor": bool(np.isfinite(normalized_role).all()),
            "finite_context_tensor": bool(np.isfinite(normalized_context).all()),
            "finite_power_tensor": bool(np.isfinite(normalized_power).all()),
            "a4_inactive_for_all_units": bool(
                (active_role_mask[:, roles.index("A4")] == 0).all()
            ),
            "no_self_candidate_edges": bool(
                np.trace(candidate_edges, axis1=1, axis2=2).sum() == 0
            ),
            "split_run_sets_disjoint": bool(
                not (
                    set(split_run_ids["train"])
                    & set(split_run_ids["validation"])
                    or set(split_run_ids["train"])
                    & set(split_run_ids["holdout"])
                    or set(split_run_ids["validation"])
                    & set(split_run_ids["holdout"])
                )
            ),
            "all_splits_have_windows": bool(
                all(len(indices) > 0 for indices in split_indices.values())
            ),
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(
        manifest_path.suffix + ".tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    return manifest
