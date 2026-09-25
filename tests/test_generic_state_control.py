import torch

from maspge.mafs_adapted import PerUnitGenericStateAdapter


def make_adapter(counts=(3, 2, 1, 0), hidden=7):
    return PerUnitGenericStateAdapter(
        history_scales=(2, 4),
        role_feature_counts=counts,
        state_hidden_size=hidden,
        d_model=8,
    )


def test_generic_state_adapter_preserves_unit_axis_and_starts_at_zero() -> None:
    adapter = make_adapter()
    outputs = adapter(torch.randn(3, 4, 5, 4, 3))
    assert len(outputs) == 2
    assert all(output.shape == (3, 5, 8) for output in outputs)
    assert all(torch.count_nonzero(output).item() == 0 for output in outputs)


def test_generic_state_adapter_flattens_every_valid_slot_and_ignores_padding() -> None:
    counts = (3, 2, 1, 0)
    adapter = make_adapter(counts=counts)
    x_role = torch.arange(2 * 4 * 1 * 4 * 3, dtype=torch.float32).reshape(2, 4, 1, 4, 3)
    flattened = adapter.flatten_valid_sensors(x_role)
    expected = torch.cat(
        [x_role[..., role, :count] for role, count in enumerate(counts) if count],
        dim=-1,
    )
    assert flattened.shape[-1] == sum(counts)
    assert torch.equal(flattened, expected)


def test_generic_state_adapter_parameter_match_is_within_one_percent() -> None:
    smarteole = PerUnitGenericStateAdapter(
        history_scales=(15, 30, 45, 60),
        role_feature_counts=(4, 4, 4, 0),
        state_hidden_size=31,
        d_model=64,
    )
    hai = PerUnitGenericStateAdapter(
        history_scales=(15, 30, 45, 60),
        role_feature_counts=(6, 6, 6, 6),
        state_hidden_size=35,
        d_model=64,
    )
    smarteole_parameters = sum(p.numel() for p in smarteole.parameters())
    hai_parameters = sum(p.numel() for p in hai.parameters())
    assert smarteole_parameters == 42084
    assert hai_parameters == 51988
    assert abs(smarteole_parameters - 42368) / 42368 < 0.01
    assert abs(hai_parameters - 52224) / 52224 < 0.01
