from __future__ import annotations

import pytest
import torch

from sentinel_v3.calibration import calibrate_amplitude_scale
from sentinel_v3.losses import (
    deterministic_detail_target,
    high_frequency_loss,
    highpass,
    low_frequency_loss,
    robust_rms,
)
from sentinel_v3.model import ModelConfig, SentinelV3
from sentinel_v3.sensors import SENTINEL1, SENTINEL2
from sentinel_v3.training import (
    EMA,
    JointObjective,
    _load_compatible_state,
    _stage_learning_rates,
)


def _batch(delta_days: int = 0, eligible: bool = True) -> dict[str, object]:
    batch_size = 4
    return {
        "s2": torch.rand(batch_size, 10, 32, 32),
        "sar": torch.randn(batch_size, 2, 32, 32) * 4 - 15,
        "s2_view": torch.rand(batch_size, 10, 32, 32),
        "sar_view": torch.randn(batch_size, 2, 32, 32) * 4 - 15,
        "s2_target": torch.rand(batch_size, 10, 32, 32),
        "sar_target": torch.randn(batch_size, 2, 32, 32) * 4 - 15,
        "valid": torch.ones(batch_size, 1, 32, 32),
        "metadata": torch.zeros(batch_size, 8),
        "input_gsd": torch.full((batch_size,), 10.0),
        "target_gsd": torch.full((batch_size,), 10.0),
        "delta_days": torch.full((batch_size,), delta_days, dtype=torch.long),
        "hf_eligible": torch.full((batch_size,), eligible),
    }


def test_highpass_removes_constant_low_frequency() -> None:
    values = torch.ones(2, 3, 32, 32) * 4.0
    residual = highpass(values)
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    assert float(low_frequency_loss(residual, torch.ones(2, 1, 32, 32))) == 0.0


def test_sar_frequency_loss_is_finite() -> None:
    prediction = highpass(torch.randn(2, 2, 32, 32))
    target = highpass(torch.randn(2, 2, 32, 32))
    loss, metrics = high_frequency_loss(prediction, target, torch.ones(2, 1, 32, 32), "sar")
    assert torch.isfinite(loss)
    assert "speckle_scale" in metrics


def test_sar_deterministic_target_rejects_isolated_speckle() -> None:
    target = torch.zeros(1, 2, 8, 8)
    target[..., 4, 4] = 20.0
    mask = torch.ones(1, 1, 8, 8)
    detail = deterministic_detail_target(target, torch.zeros_like(target), mask, "sar")
    torch.testing.assert_close(detail, torch.zeros_like(detail))


def test_multiscale_conditions_match_codec_grid(tiny_model: SentinelV3) -> None:
    pyramid = tiny_model.encode(torch.randn(2, 2, 32, 32), SENTINEL1, torch.ones(2, 1, 32, 32))
    conditions = tiny_model.residual_dit.multiscale_condition(pyramid, (8, 8))
    assert len(conditions) == 4
    assert all(
        condition.shape == (2, tiny_model.config.dit_hidden, 8, 8) for condition in conditions
    )
    torch.testing.assert_close(
        tiny_model.residual_dit.condition_gates,
        torch.zeros_like(tiny_model.residual_dit.condition_gates),
    )


def test_codec_latent_standardization_roundtrip(tiny_model: SentinelV3) -> None:
    codec = tiny_model.codec
    mean = torch.linspace(-1, 1, codec.latent_channels)
    std = torch.linspace(0.5, 1.5, codec.latent_channels)
    codec.set_statistics("optical", mean, std)
    latent = torch.randn(2, codec.latent_channels, 8, 8)
    torch.testing.assert_close(
        codec.denormalize(codec.normalize(latent, "optical"), "optical"), latent
    )


def test_deterministic_stochastic_decomposition(tiny_model: SentinelV3) -> None:
    target = torch.rand(2, 3, 32, 32)
    physical = torch.rand(2, 3, 32, 32)
    detail = highpass(torch.randn_like(target) * 0.01)
    target_detail = highpass(target - physical)
    texture = target_detail - detail.detach()
    torch.testing.assert_close(detail + texture, target_detail)
    assert not texture.requires_grad


def test_sar_deterministic_target_excludes_pixel_speckle() -> None:
    torch.manual_seed(4)
    target = torch.randn(2, 2, 32, 32) * 3.0
    valid = torch.ones(2, 1, 32, 32)
    deterministic = JointObjective._deterministic_target(
        target, torch.zeros_like(target), valid, SENTINEL1
    )
    raw = highpass(target)
    assert deterministic.abs().mean() < raw.abs().mean()


@pytest.mark.parametrize(
    ("stage", "module_name"),
    (("detail", "detail_head"), ("codec", "codec"), ("flow", "residual_dit")),
)
def test_delta_greater_than_one_has_exact_zero_residual_gradient(
    tiny_model: SentinelV3, stage: str, module_name: str
) -> None:
    objective = JointObjective(tiny_model, [0.5, 0.5])
    loss, _ = objective(_batch(delta_days=2), stage)
    loss.backward()
    assert float(loss.detach()) == 0.0
    gradients = [
        parameter.grad
        for parameter in getattr(tiny_model, module_name).parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert sum(float(gradient.abs().sum()) for gradient in gradients) == 0.0


def test_ineligible_samples_do_not_update_codec_statistics(tiny_model: SentinelV3) -> None:
    objective = JointObjective(tiny_model, [0.5, 0.5])
    before = {
        modality: tuple(value.clone() for value in tiny_model.codec.statistics(modality))
        for modality in ("optical", "sar")
    }
    objective(_batch(delta_days=2), "codec")
    for modality in ("optical", "sar"):
        after = tiny_model.codec.statistics(modality)
        for actual, expected in zip(after, before[modality], strict=True):
            torch.testing.assert_close(actual, expected)


def test_robust_amplitude_is_blockwise() -> None:
    values = torch.ones(2, 3, 32, 32)
    amplitude = robust_rms(values, torch.ones(2, 1, 32, 32))
    assert amplitude.shape == (2, 3, 8, 8)
    torch.testing.assert_close(amplitude, torch.ones_like(amplitude))


def test_optical_logit_composition_is_bounded_and_reports_violation() -> None:
    physical = torch.tensor([[[[0.01, 0.99]]]])
    detail = torch.tensor([[[[-1.0, 1.0]]]])
    composed, violation = SentinelV3.compose_visual(
        physical, detail, torch.zeros_like(detail), "optical", return_violation=True
    )
    assert bool(((composed >= 0) & (composed <= 1)).all())
    assert float(violation) == 1.0


def test_amplitude_calibration_obeys_rmse_guardrail() -> None:
    physical = torch.zeros(1, 3, 8, 8) + 0.5
    target = physical.clone()
    texture = torch.ones_like(physical) * 0.2
    alpha, metrics = calibrate_amplitude_scale(
        physical,
        torch.zeros_like(physical),
        texture,
        target,
        torch.ones(1, 1, 8, 8),
        "optical",
    )
    assert alpha == 0.0
    assert metrics["rmse_ratio"] == 0.0


def test_scene_conditioned_amplitude_shape(tiny_model: SentinelV3) -> None:
    pyramid = tiny_model.encode(torch.randn(2, 2, 32, 32), SENTINEL1, torch.ones(2, 1, 32, 32))
    amplitude = tiny_model.residual_amplitude(pyramid, SENTINEL2, 3, (32, 32))
    assert amplitude.shape == (2, 3, 8, 8)
    assert bool((amplitude <= tiny_model.config.optical_residual_limit).all())


def test_stage_learning_rates_match_v32_schedule() -> None:
    config = {"learning_rate": 1e-4, "encoder_learning_rate": 2e-5}
    assert _stage_learning_rates(config, "flow") == pytest.approx((0.0, 0.0, 1e-4))
    assert _stage_learning_rates(config, "balance") == pytest.approx((2e-6, 1e-5, 1e-5))


def test_v31_state_is_compatible_initialization_only(tiny_model: SentinelV3) -> None:
    state = tiny_model.state_dict()
    first_name = next(iter(state))
    loaded, initialized = _load_compatible_state(
        tiny_model, {first_name: state[first_name].clone(), "legacy.weight": torch.ones(1)}
    )
    assert loaded == 1
    assert initialized == len(state) - 1


def test_zero_legacy_adapter_gate_is_not_imported() -> None:
    model = SentinelV3(
        ModelConfig(
            width=8,
            hidden=32,
            encoder_depth=3,
            heads=4,
            adapter_rank=8,
            dit_hidden=32,
            dit_depth=1,
            dit_heads=4,
            codec_width=8,
            codec_latent_channels=4,
            flow_steps=2,
        )
    )
    name = "encoder.adapters.3.optical.scale"
    before = model.state_dict()[name].clone()
    loaded, initialized = _load_compatible_state(model, {name: torch.zeros(())})
    assert loaded == 0
    assert initialized == len(model.state_dict())
    torch.testing.assert_close(model.state_dict()[name], before)


def test_ema_validation_weights_are_restored(tiny_model: SentinelV3) -> None:
    ema = EMA(tiny_model, 0.9)
    name, parameter = next(iter(tiny_model.named_parameters()))
    original = parameter.detach().clone()
    ema.state[name].fill_(3.0)
    with ema.apply_to(tiny_model):
        torch.testing.assert_close(parameter, torch.full_like(parameter, 3.0))
    torch.testing.assert_close(parameter, original)
