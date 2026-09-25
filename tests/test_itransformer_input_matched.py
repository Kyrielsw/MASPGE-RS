import torch

from maspge.baselines import ITransformerInputMatchedAdapted


def build_model():
    return ITransformerInputMatchedAdapted(
        history_steps=60, horizon_steps=15, role_feature_counts=(2, 1),
        d_model=16, n_heads=4, layers=1, feedforward_size=32, dropout=0.0,
    )


def test_input_matched_outputs_complete_horizon_and_is_finite():
    model = build_model()
    x = torch.randn(3, 60, 5, 2, 2)
    power = torch.randn(3, 60, 5)
    output = model(x_role=x, power_history=power)
    assert output.shape == (3, 15, 5)
    assert torch.isfinite(output).all()


def test_flattening_is_generic_and_ignores_unused_padding():
    model = build_model()
    x = torch.randn(2, 60, 4, 2, 2)
    x_changed = x.clone(); x_changed[:, :, :, 1, 1] = 1e6
    assert torch.equal(model.flatten_sensors(x), model.flatten_sensors(x_changed))


def test_context_is_not_an_effective_input():
    model = build_model().eval()
    x = torch.randn(2, 60, 4, 2, 2); power = torch.randn(2, 60, 4)
    first = model(x_role=x, power_history=power, context=torch.zeros(2, 60, 9))
    second = model(x_role=x, power_history=power, context=torch.randn(2, 60, 9))
    assert torch.equal(first, second)
