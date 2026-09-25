#!/usr/bin/env python3
"""Build the frozen five-unit XAI4HEAT generic-state contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
STATIONS = ("L4", "L8", "L12", "L17", "L22")
SENSORS = ("t_amb", "t_ref", "t_sup_prim", "t_ret_prim", "t_sup_sec", "t_ret_sec")
HISTORY = 60
HORIZON = 15
SPLITS = {"2021-2022": "train", "2022-2023": "validation", "2023-2024": "holdout"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def season_id(timestamp: pd.Timestamp) -> str:
    year = timestamp.year if timestamp.month >= 11 else timestamp.year - 1
    return f"{year}-{year + 1}"


def locate_csv_root(root: Path) -> Path:
    candidates = list(root.rglob("xai4heat_scada_L4_processed.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one extracted L4 CSV, found {len(candidates)}")
    return candidates[0].parent


def valid_endpoints(
    indices: np.ndarray,
    season_ids: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    output = []
    index_set = set(int(value) for value in indices)
    for end in indices:
        end = int(end)
        start = end - HISTORY + 1
        future_end = end + HORIZON
        if start < 0 or future_end >= len(season_ids):
            continue
        if start not in index_set or future_end not in index_set:
            continue
        if season_ids[start] != season_ids[future_end]:
            continue
        if not bool(valid[start : future_end + 1].all()):
            continue
        output.append(end)
    return np.asarray(output, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip", type=Path,
        default=ROOT / "data/raw/xai4heat_v1/xai4heat_scada_dataset_2024_v1.zip",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/xai4heat_scada_2024_generic_v1.npz",
    )
    args = parser.parse_args()
    if not args.zip.is_file():
        raise FileNotFoundError(args.zip)

    with tempfile.TemporaryDirectory(prefix="xai4heat_") as temporary:
        with zipfile.ZipFile(args.zip) as archive:
            archive.extractall(temporary)
        csv_root = locate_csv_root(Path(temporary))
        frames = {}
        hashes = {}
        for station in STATIONS:
            path = csv_root / f"xai4heat_scada_{station}_processed.csv"
            hashes[path.name] = sha256(path)
            frame = pd.read_csv(path)
            expected = ["datetime", *SENSORS, "delta_e"]
            if list(frame.columns) != expected:
                raise RuntimeError(f"unexpected schema for {station}: {list(frame.columns)}")
            frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
            if frame["datetime"].duplicated().any():
                raise RuntimeError(f"duplicate timestamps in {station}")
            frames[station] = frame.set_index("datetime").sort_index()

        common = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
        timestamps = pd.DatetimeIndex(common)
        seasons = np.asarray([season_id(value) for value in timestamps], dtype="U9")
        if set(seasons) != set(SPLITS):
            raise RuntimeError(f"unexpected common seasons: {sorted(set(seasons))}")

        raw = np.stack(
            [frames[station].loc[timestamps, [*SENSORS, "delta_e"]].to_numpy(float)
             for station in STATIONS],
            axis=1,
        )
        if not np.isfinite(raw).all():
            raise RuntimeError("published common data contains non-finite values")

        quality_valid = np.ones((len(timestamps), len(STATIONS)), dtype=bool)
        boundary = np.r_[True, seasons[1:] != seasons[:-1]]
        # The first retained energy difference after an off-season is not an
        # hourly target.  The 2021 common grid starts one hour later and is not
        # marked; later season starts contain the documented cumulative jump.
        for index in np.flatnonzero(boundary):
            if index > 0:
                quality_valid[index, :] = False

        train_rows = seasons == "2021-2022"
        train_power = raw[train_rows, :, -1]
        train_q99 = np.quantile(train_power, 0.99, axis=0)
        gross = raw[:, :, -1] > (10.0 * train_q99[None, :])
        synchronized = (raw[:, :, -1] > (4.0 * train_q99[None, :])).sum(axis=1) >= 3
        quality_valid[gross] = False
        quality_valid[synchronized, :] = False

        # Fixed physical range; causal fill is performed independently inside
        # each station and season.  Training medians are the final fallback.
        sensor = raw[:, :, :-1].copy()
        sensor[:, STATIONS.index("L22"), 0][
            (sensor[:, STATIONS.index("L22"), 0] < -40)
            | (sensor[:, STATIONS.index("L22"), 0] > 50)
        ] = np.nan
        training_sensor_median = np.nanmedian(sensor[train_rows], axis=0)
        for station_index in range(len(STATIONS)):
            for feature_index in range(len(SENSORS)):
                series = pd.Series(sensor[:, station_index, feature_index])
                for season in SPLITS:
                    rows = seasons == season
                    series.loc[rows] = series.loc[rows].ffill()
                sensor[:, station_index, feature_index] = series.fillna(
                    training_sensor_median[station_index, feature_index]
                ).to_numpy()

        power_raw = raw[:, :, -1].copy()
        feature_power = power_raw.copy()
        train_valid = quality_valid & train_rows[:, None]
        target_mean = np.asarray(
            [power_raw[train_valid[:, unit], unit].mean() for unit in range(5)],
            dtype=np.float64,
        )
        target_scale = np.asarray(
            [power_raw[train_valid[:, unit], unit].std() for unit in range(5)],
            dtype=np.float64,
        )
        target_scale = np.maximum(target_scale, 1e-6)
        for unit in range(5):
            feature_power[~quality_valid[:, unit], unit] = target_mean[unit]
        power = ((feature_power - target_mean) / target_scale).astype(np.float32)

        sensor_mean = sensor[train_rows].mean(axis=0)
        sensor_scale = sensor[train_rows].std(axis=0)
        sensor_scale = np.maximum(sensor_scale, 1e-6)
        sensor_standardized = ((sensor - sensor_mean) / sensor_scale).astype(np.float32)
        state = np.concatenate([sensor_standardized, power[:, :, None]], axis=-1)
        x_role = state[:, :, None, :]

        hour = timestamps.hour.to_numpy()
        day = timestamps.dayofyear.to_numpy()
        context = np.stack(
            [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
             np.sin(2 * np.pi * day / 366), np.cos(2 * np.pi * day / 366)],
            axis=1,
        ).astype(np.float32)

        split_indices = {}
        for season, split in SPLITS.items():
            rows = np.flatnonzero(seasons == season)
            split_indices[split] = valid_endpoints(rows, seasons, quality_valid)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            dataset_id=np.asarray("xai4heat_scada_2024_generic_v1"),
            x_role=x_role,
            context=context,
            power=power,
            power_valid=quality_valid,
            train_end_indices=split_indices["train"],
            validation_end_indices=split_indices["validation"],
            holdout_end_indices=split_indices["holdout"],
            feature_slot_mask=np.ones((5, 1, 7), dtype=bool),
            role_feature_counts=np.asarray([7], dtype=np.int64),
            active_role_mask=np.ones((5, 1), dtype=bool),
            target_scale=target_scale.astype(np.float32),
            target_mean=target_mean.astype(np.float32),
            sensor_mean=sensor_mean.astype(np.float32),
            sensor_scale=sensor_scale.astype(np.float32),
            coordinate_xy=np.stack([np.arange(5), np.zeros(5)], axis=1).astype(np.float32),
            unit_names=np.asarray(STATIONS, dtype="U3"),
            timestamps=np.asarray(
                timestamps.strftime("%Y-%m-%dT%H:%M:%S").tolist(), dtype="U19"
            ),
            season_ids=seasons,
            quality_valid=quality_valid,
        )

    audit = {
        "dataset_id": "xai4heat_scada_2024_generic_v1",
        "source_zip_sha256": sha256(args.zip),
        "source_csv_sha256": hashes,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "time_steps": int(len(timestamps)),
        "unit_names": list(STATIONS),
        "state_shape": list(x_role.shape),
        "history_steps": HISTORY,
        "horizon_steps": HORIZON,
        "splits_by_season": SPLITS,
        "split_windows": {key: int(len(value)) for key, value in split_indices.items()},
        "invalid_target_values": int((~quality_valid).sum()),
        "anomaly_policy": {
            "season_boundary": "exclude post-off-season cumulative differences",
            "individual_gross_spike": "exclude above 10x station training q99",
            "synchronized_spike": "exclude timestamp when >=3 units exceed 4x station training q99",
            "l22_ambient": "outside [-40,50] -> missing -> season-local causal ffill -> training median",
        },
        "normalization": "per-unit/per-variable training-season mean and std only",
        "future_covariates": False,
        "model_training_performed": False,
        "validation_or_holdout_evaluated": False,
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
