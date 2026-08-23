#!/usr/bin/env python3
"""Audit the four normal HAI 23.05 files before freezing a forecast contract.

This script performs schema, time-continuity, missingness and variability checks.
It never reads attack files, constructs model windows or evaluates a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/hai_23_05_admission_v1.json"
DEFAULT_RAW_DIR = ROOT / "data/raw/hai_23_05"
DEFAULT_OUTPUT = ROOT / "data/hai_23_05_admission_v1.audit.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RunningStats:
    count: int = 0
    missing: int = 0
    total: float = 0.0
    total_square: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    changes: int = 0
    previous: float | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        self.missing += int((~finite).sum())
        clean = values[finite]
        if not clean.size:
            return
        self.count += int(clean.size)
        self.total += float(clean.sum(dtype=np.float64))
        self.total_square += float(np.square(clean).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(clean.min()))
        self.maximum = max(self.maximum, float(clean.max()))
        if self.previous is not None:
            self.changes += int(clean[0] != self.previous)
        if clean.size > 1:
            self.changes += int(np.count_nonzero(np.diff(clean)))
        self.previous = float(clean[-1])

    def summary(self) -> dict[str, float | int | bool | None]:
        denominator = self.count + self.missing
        if self.count == 0:
            return {
                "count": 0,
                "missing": self.missing,
                "missing_fraction": 1.0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "changes": 0,
                "varying": False,
            }
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "missing": self.missing,
            "missing_fraction": self.missing / max(denominator, 1),
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
            "changes": self.changes,
            "varying": bool(self.maximum > self.minimum and self.changes > 0),
        }


def ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def required_columns(config: dict) -> list[str]:
    role_columns = [
        name
        for unit_roles in config["candidate_roles"].values()
        for names in unit_roles.values()
        for name in names
    ]
    return ordered_unique(
        [config["timestamp_column"], *config["targets"].values(), *role_columns]
    )


def audit_file(
    path: Path,
    *,
    timestamp_column: str,
    numeric_columns: list[str],
    expected_sampling_seconds: float,
    chunksize: int,
) -> dict:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing_columns = sorted(set([timestamp_column, *numeric_columns]) - set(header))
    if missing_columns:
        raise RuntimeError(f"{path.name} is missing columns: {missing_columns}")
    attack_columns = [name for name in header if "attack" in name.lower()]
    stats = {name: RunningStats() for name in numeric_columns}
    rows = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    previous_ns: int | None = None
    non_one_second_gaps = 0
    duplicate_or_backward = 0
    minimum_delta_seconds: float | None = None
    maximum_delta_seconds: float | None = None

    usecols = [timestamp_column, *numeric_columns]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        timestamps = pd.to_datetime(chunk[timestamp_column], errors="coerce")
        if bool(timestamps.isna().any()):
            raise RuntimeError(f"{path.name} contains invalid timestamps")
        ns = timestamps.astype("int64").to_numpy()
        if first_timestamp is None:
            first_timestamp = str(timestamps.iloc[0])
        if previous_ns is not None:
            ns_with_previous = np.concatenate([[previous_ns], ns])
        else:
            ns_with_previous = ns
        deltas = np.diff(ns_with_previous) / 1_000_000_000
        if deltas.size:
            non_one_second_gaps += int(
                np.count_nonzero(deltas != expected_sampling_seconds)
            )
            duplicate_or_backward += int(np.count_nonzero(deltas <= 0.0))
            current_min = float(deltas.min())
            current_max = float(deltas.max())
            minimum_delta_seconds = (
                current_min if minimum_delta_seconds is None
                else min(minimum_delta_seconds, current_min)
            )
            maximum_delta_seconds = (
                current_max if maximum_delta_seconds is None
                else max(maximum_delta_seconds, current_max)
            )
        previous_ns = int(ns[-1])
        last_timestamp = str(timestamps.iloc[-1])
        rows += len(chunk)
        for name in numeric_columns:
            stats[name].update(pd.to_numeric(chunk[name], errors="coerce").to_numpy())

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": rows,
        "columns": len(header),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "minimum_delta_seconds": minimum_delta_seconds,
        "maximum_delta_seconds": maximum_delta_seconds,
        "non_one_second_gaps": non_one_second_gaps,
        "duplicate_or_backward_timestamps": duplicate_or_backward,
        "attack_columns": attack_columns,
        "statistics": {name: value.summary() for name, value in stats.items()},
    }


def combine_statistics(file_audits: list[dict], columns: list[str]) -> dict[str, dict]:
    combined: dict[str, dict] = {}
    for name in columns:
        rows = [item["statistics"][name] for item in file_audits]
        count = sum(int(row["count"]) for row in rows)
        missing = sum(int(row["missing"]) for row in rows)
        if count == 0:
            combined[name] = {
                "count": 0, "missing": missing, "missing_fraction": 1.0,
                "mean": None, "std": None, "min": None, "max": None,
                "changes": 0, "varying": False,
            }
            continue
        total = sum(float(row["mean"]) * int(row["count"]) for row in rows)
        mean = total / count
        second = sum(
            (float(row["std"]) ** 2 + float(row["mean"]) ** 2)
            * int(row["count"])
            for row in rows
        ) / count
        minimum = min(float(row["min"]) for row in rows if row["min"] is not None)
        maximum = max(float(row["max"]) for row in rows if row["max"] is not None)
        changes = sum(int(row["changes"]) for row in rows)
        combined[name] = {
            "count": count,
            "missing": missing,
            "missing_fraction": missing / max(count + missing, 1),
            "mean": mean,
            "std": math.sqrt(max(second - mean * mean, 0.0)),
            "min": minimum,
            "max": maximum,
            "changes": changes,
            "varying": bool(maximum > minimum and changes > 0),
        }
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--minimum-rows-per-file", type=int, default=100_000)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    columns = required_columns(config)
    timestamp_column = config["timestamp_column"]
    numeric_columns = [name for name in columns if name != timestamp_column]
    paths = [args.raw_dir / name for name in config["normal_files"]]
    missing_files = [str(path) for path in paths if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"HAI normal files not found: {missing_files}")

    file_audits = [
        audit_file(
            path,
            timestamp_column=timestamp_column,
            numeric_columns=numeric_columns,
            expected_sampling_seconds=float(
                config["admission"]["expected_sampling_seconds"]
            ),
            chunksize=args.chunksize,
        )
        for path in paths
    ]
    combined = combine_statistics(file_audits, numeric_columns)
    rules = config["admission"]
    failures: list[str] = []
    for item in file_audits:
        if item["rows"] < args.minimum_rows_per_file:
            failures.append(f"{Path(item['path']).name}: insufficient rows")
        if item["duplicate_or_backward_timestamps"]:
            failures.append(f"{Path(item['path']).name}: non-monotonic timestamps")
        if item["non_one_second_gaps"]:
            failures.append(f"{Path(item['path']).name}: non-1-second gaps")
        if rules["prohibit_attack_columns"] and item["attack_columns"]:
            failures.append(f"{Path(item['path']).name}: attack columns present")
    for unit, target in config["targets"].items():
        summary = combined[target]
        if summary["std"] is None or summary["std"] < rules["minimum_target_std"]:
            failures.append(f"{unit} target {target} is degenerate")
        if summary["missing_fraction"] > rules["maximum_missing_fraction"]:
            failures.append(f"{unit} target {target} missing fraction too high")
    varying_a4: dict[str, list[str]] = {}
    for unit, unit_roles in config["candidate_roles"].items():
        names = unit_roles["A4_auxiliary"]
        varying_a4[unit] = [
            name for name in names
            if combined[name]["varying"]
            and combined[name]["missing_fraction"]
            <= rules["maximum_missing_fraction"]
        ]
        minimum = int(rules["minimum_varying_a4_tags"][unit])
        if len(varying_a4[unit]) < minimum:
            failures.append(
                f"{unit} A4 has {len(varying_a4[unit])} varying tags, requires {minimum}"
            )
    audit = {
        "phase": "HAI_23_05_schema_and_distribution_admission_only",
        "dataset_id": config["dataset_id"],
        "model_training_performed": False,
        "holdout_model_evaluated": False,
        "normal_files_only": True,
        "config": str(args.config),
        "file_audits": file_audits,
        "combined_statistics": combined,
        "varying_a4_tags": varying_a4,
        "split_files": config["split_files"],
        "admitted_for_contract_freeze": not failures,
        "admission_failures": failures,
        "next_step": (
            "freeze resampling and exact role slots; then implement deterministic preprocessing"
            if not failures else
            "stop before preprocessing and resolve the listed data-admission failures"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "admitted_for_contract_freeze": audit["admitted_for_contract_freeze"],
        "admission_failures": failures,
        "varying_a4_tags": varying_a4,
        "rows_by_file": {
            Path(item["path"]).name: item["rows"] for item in file_audits
        },
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
