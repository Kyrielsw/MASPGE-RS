"""Frozen power-generation datasets and leakage-safe window views."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_smarteole(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. See data/README.md."
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Dataset SHA256 mismatch. "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "x_role",
        "context",
        "power",
        "train_end_indices",
        "validation_end_indices",
        "holdout_end_indices",
        "feature_slot_mask",
        "active_role_mask",
        "target_scale",
        "coordinate_xy",
    }
    missing = required.difference(arrays)
    if missing:
        raise RuntimeError(f"Processed dataset is missing arrays: {missing}")
    if arrays["x_role"].shape != (120575, 7, 4, 6):
        raise RuntimeError(
            f"Unexpected x_role shape: {arrays['x_role'].shape}"
        )
    expected_split_sizes = {
        "train_end_indices": 81188,
        "validation_end_indices": 17946,
        "holdout_end_indices": 16419,
    }
    for name, expected_size in expected_split_sizes.items():
        if len(arrays[name]) != expected_size:
            raise RuntimeError(
                f"Unexpected {name} size: {len(arrays[name])}"
            )
    return arrays


def load_frozen_sdwpf(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    """Load the processed SDWPF contract without touching the raw holdout."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Processed SDWPF dataset not found: {path}. See data/SDWPF_README.md."
        )
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Dataset SHA256 mismatch. "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "x_role",
        "context",
        "power",
        "power_valid",
        "train_end_indices",
        "validation_end_indices",
        "holdout_end_indices",
        "feature_slot_mask",
        "active_role_mask",
        "target_scale",
        "target_mean",
        "coordinate_xy",
        "dataset_id",
    }
    missing = required.difference(arrays)
    if missing:
        raise RuntimeError(f"Processed SDWPF dataset is missing arrays: {missing}")
    dataset_id = str(arrays["dataset_id"].item())
    if dataset_id not in {
        "sdwpf_kdd_245days_v1", "sdwpf_kdd_245days_role_state_v2"
    }:
        raise RuntimeError(f"Unexpected SDWPF dataset_id: {arrays['dataset_id']}")
    time_steps, unit_count = arrays["power"].shape
    if (time_steps, unit_count) != (245 * 144, 134):
        raise RuntimeError(f"Unexpected SDWPF power shape: {arrays['power'].shape}")
    if arrays["power_valid"].shape != arrays["power"].shape:
        raise RuntimeError("SDWPF power_valid must align with power")
    expected_role_shape = (
        (time_steps, 4)
        if dataset_id == "sdwpf_kdd_245days_v1"
        else (time_steps, unit_count, 4, 6)
    )
    if arrays["x_role"].shape != expected_role_shape:
        raise RuntimeError(
            f"Unexpected {dataset_id} role shape: {arrays['x_role'].shape}"
        )
    if arrays["coordinate_xy"].shape != (unit_count, 2):
        raise RuntimeError("SDWPF coordinates must have shape [134, 2]")
    for split in ("train_end_indices", "validation_end_indices", "holdout_end_indices"):
        indices = arrays[split]
        if indices.ndim != 1 or len(indices) == 0:
            raise RuntimeError(f"SDWPF split is empty or malformed: {split}")
    arrays["sha256"] = np.asarray(actual_sha256)
    return arrays


def load_frozen_hai(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    """Load the frozen two-unit HAI 23.05 normal-operation contract."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Processed HAI dataset not found: {path}. See data/HAI_README.md."
        )
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            "HAI dataset SHA256 mismatch. "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "dataset_id", "x_role", "context", "power", "power_valid",
        "train_end_indices", "validation_end_indices", "holdout_end_indices",
        "feature_slot_mask", "role_feature_counts", "active_role_mask",
        "target_scale", "target_mean", "coordinate_xy", "unit_names",
    }
    missing = required.difference(arrays)
    if missing:
        raise RuntimeError(f"Processed HAI dataset is missing arrays: {missing}")
    if str(arrays["dataset_id"].item()) != "hai_23_05_role_state_v1":
        raise RuntimeError(f"Unexpected HAI dataset_id: {arrays['dataset_id']}")
    time_steps = arrays["power"].shape[0]
    if arrays["power"].shape != (time_steps, 2):
        raise RuntimeError(f"Unexpected HAI power shape: {arrays['power'].shape}")
    if arrays["x_role"].shape != (time_steps, 2, 4, 6):
        raise RuntimeError(f"Unexpected HAI role shape: {arrays['x_role'].shape}")
    if arrays["power_valid"].shape != arrays["power"].shape:
        raise RuntimeError("HAI power_valid must align with power")
    if arrays["feature_slot_mask"].shape != (2, 4, 6):
        raise RuntimeError("HAI feature_slot_mask must have shape [2,4,6]")
    if arrays["role_feature_counts"].tolist() != [6, 6, 6, 6]:
        raise RuntimeError("HAI role feature counts must preserve six-slot roles")
    if arrays["active_role_mask"].shape != (2, 4):
        raise RuntimeError("HAI active role mask must have shape [2,4]")
    if not bool(arrays["active_role_mask"].all()):
        raise RuntimeError("HAI contract requires A1--A4 active for both units")
    if arrays["target_scale"].shape != (2,):
        raise RuntimeError("HAI target_scale must be per unit")
    previous_last = -1
    for split in ("train_end_indices", "validation_end_indices", "holdout_end_indices"):
        indices = arrays[split]
        if indices.ndim != 1 or len(indices) == 0:
            raise RuntimeError(f"HAI split is empty or malformed: {split}")
        if int(indices[0]) <= previous_last:
            raise RuntimeError("HAI file-level splits must be chronological and disjoint")
        previous_last = int(indices[-1])
    arrays["sha256"] = np.asarray(actual_sha256)
    return arrays


class SmarteoleWindowDataset(Dataset):
    """Run-safe views over arrays produced by the accepted S1 pipeline."""

    def __init__(
        self,
        x_role: np.ndarray,
        context: np.ndarray,
        power: np.ndarray,
        end_indices: np.ndarray,
        *,
        history_steps: int,
        horizon_steps: int,
        power_valid: np.ndarray | None = None,
    ) -> None:
        self.x_role = x_role
        self.context = context
        self.power = power
        self.end_indices = np.asarray(end_indices, dtype=np.int64)
        self.history_steps = int(history_steps)
        self.horizon_steps = int(horizon_steps)
        self.power_valid = (
            np.ones_like(power, dtype=np.bool_)
            if power_valid is None
            else np.asarray(power_valid, dtype=np.bool_)
        )
        if self.power_valid.shape != self.power.shape:
            raise ValueError("power_valid must align with power")

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        end = int(self.end_indices[item])
        start = end - self.history_steps + 1
        future_start = end + 1
        future_end = end + self.horizon_steps + 1
        future_power = self.power[future_start:future_end]
        future_valid = self.power_valid[future_start:future_end]
        valid_count = future_valid.sum(axis=0)
        future_mean = np.divide(
            (future_power * future_valid).sum(axis=0, dtype=np.float64),
            np.maximum(valid_count, 1),
        ).astype(np.float32)
        return {
            "x_role": torch.from_numpy(
                self.x_role[start : end + 1]
            ).float(),
            "context": torch.from_numpy(
                self.context[start : end + 1]
            ).float(),
            "y_current": torch.from_numpy(self.power[end]).float(),
            "power_history": torch.from_numpy(
                self.power[start : end + 1]
            ).float(),
            "y_future": torch.from_numpy(future_mean),
            "y_future_valid": torch.from_numpy(valid_count > 0),
            "power_future_sequence": torch.from_numpy(
                future_power
            ).float(),
            "power_future_valid": torch.from_numpy(future_valid),
        }


PowerWindowDataset = SmarteoleWindowDataset


def horizon_safe_end_indices(
    arrays: dict[str, np.ndarray],
    split: str,
    *,
    horizon_steps: int,
    reference_horizon_steps: int = 15,
) -> np.ndarray:
    """Filter frozen endpoints so a new horizon cannot cross a boundary.

    The function deliberately starts from the original frozen endpoint set.
    Shorter horizons therefore use the same anchors as the 15-step study,
    while longer horizons only remove unsafe anchors.  SMARTEOLE run IDs
    enforce run boundaries; contiguous datasets use the frozen split stop.
    """
    name = f"{split}_end_indices"
    if name not in arrays:
        raise KeyError(name)
    indices = np.asarray(arrays[name], dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError(f"empty or malformed split endpoints: {name}")
    horizon = int(horizon_steps)
    if horizon <= 0:
        raise ValueError("horizon_steps must be positive")
    safe = indices + horizon < arrays["power"].shape[0]
    if "run_ids" in arrays:
        candidates = indices[safe]
        safe_indices = candidates[
            arrays["run_ids"][candidates]
            == arrays["run_ids"][candidates + horizon]
        ]
    else:
        split_stop = int(indices[-1]) + int(reference_horizon_steps) + 1
        safe_indices = indices[safe & (indices + horizon < split_stop)]
    if len(safe_indices) == 0:
        raise ValueError(f"no {split} endpoints support horizon={horizon}")
    return safe_indices
