#!/usr/bin/env python3
"""Preflight audit for the same-input generic-state validation control."""

from __future__ import annotations

import json

import torch

from maspge.mafs_adapted import PerUnitGenericStateAdapter
from run_semantic_controls_validation import (
    ROOT,
    build_model,
    checkpoint_state,
    load_arrays,
    role_counts,
)


def main() -> int:
    control = json.loads(
        (ROOT / "configs/generic_state_control_validation_v1.json").read_text(
            encoding="utf-8"
        )
    )
    if control["holdout_policy"] != "forbidden_during_validation_control":
        raise RuntimeError("generic-state control must forbid holdout evaluation")
    for dataset, profile in control["datasets"].items():
        experiment = json.loads(
            (ROOT / profile["experiment_config"]).read_text(encoding="utf-8")
        )
        arrays = load_arrays(dataset, experiment)
        counts = role_counts(dataset, arrays)
        model = build_model(experiment, arrays, counts)
        assert model.role_state_adapter is not None
        semantic_parameters = sum(
            parameter.numel() for parameter in model.role_state_adapter.parameters()
        )
        generic = PerUnitGenericStateAdapter(
            history_scales=tuple(experiment["model"]["history_scales"]),
            role_feature_counts=counts,
            state_hidden_size=int(profile["generic_state_hidden_size"]),
            d_model=int(experiment["model"]["d_model"]),
        )
        generic_parameters = sum(parameter.numel() for parameter in generic.parameters())
        relative_difference = (generic_parameters - semantic_parameters) / semantic_parameters
        if abs(relative_difference) >= 0.01:
            raise RuntimeError(f"{dataset}: adapter parameter mismatch exceeds 1%")
        sample = torch.randn(2, max(experiment["model"]["history_scales"]), 3, 4, 6)
        outputs = generic(sample)
        if any(torch.count_nonzero(output).item() for output in outputs):
            raise RuntimeError("generic residual projection is not zero initialized")
        if generic.total_feature_count != sum(counts):
            raise RuntimeError("generic adapter dropped a valid sensor slot")
        for seed in control["seeds"]:
            checkpoint_state(ROOT / profile["base_checkpoint"].format(seed=seed))
            checkpoint_state(ROOT / profile["full_checkpoint"].format(seed=seed))
        print(
            f"{dataset:<10} units={arrays['power'].shape[1]} counts={counts} "
            f"flat_sensors={sum(counts)} params={generic_parameters}/{semantic_parameters} "
            f"difference={100*relative_difference:.2f}% checkpoints=10/10"
        )
    print("GENERIC_STATE_CONTROL_SETUP_OK")
    print("HOLDOUT WAS NOT CONSTRUCTED OR EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
