from __future__ import annotations

import torch

from maspge.mafs_adapted import MAFSAdaptedPower, topology_mask


def build_model() -> MAFSAdaptedPower:
    return MAFSAdaptedPower(
        history_scales=(2, 4), horizon_scales=(1, 2), unit_count=3,
        full_horizon_steps=2, d_model=8, n_heads=2, layers=1,
        feedforward_size=16, dropout=0.0, topology="fully",
        role_feature_counts=(2, 2, 2, 2), role_state_hidden_size=3,
    )


def test_topologies_are_symmetric_and_have_no_raw_self_edges() -> None:
    for name in ("fully", "ring", "chain", "star"):
        mask = topology_mask(4, name)
        assert torch.equal(mask, mask.T)
        assert torch.equal(mask.diag(), torch.zeros(4))


def test_zero_initialized_adapter_exactly_preserves_backbone() -> None:
    model = build_model().eval()
    power = torch.randn(2, 4, 3)
    roles = torch.randn(2, 4, 3, 4, 2)
    base = model(power, finetune=True)["final_sequence"]
    conditioned = model(
        power, finetune=True, x_role=roles, use_role_state=True
    )["final_sequence"]
    assert torch.allclose(base, conditioned)


def test_all_frozen_ablation_modes_preserve_output_contract() -> None:
    model = build_model().eval()
    power = torch.randn(2, 4, 3)
    roles = torch.randn(2, 4, 3, 4, 2)
    for mode in ("full", "shared_units", "unscaled_history", "without_a4"):
        output = model(
            power, finetune=True, x_role=roles, use_role_state=True,
            role_state_ablation_mode=mode,
        )
        assert output["final_sequence"].shape == (2, 2, 3)
        assert torch.isfinite(output["final_sequence"]).all()


def test_role_state_freeze_trains_only_adapter() -> None:
    model = build_model()
    model.freeze_for_role_state()
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable
    assert all(name.startswith("role_state_adapter.") for name in trainable)
