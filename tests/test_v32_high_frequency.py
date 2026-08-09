from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sentinel_v3.calibration import (
    _calibration_anchor_detail,
    calibrate_amplitude_scale,
    select_texture_release_candidate,
)
from sentinel_v3.data import StatefulIndexSampler, StatefulShardSampler, V2ShardDataset
from sentinel_v3.evaluation import tail_quantile_error
from sentinel_v3.losses import (
    deterministic_detail_target,
    frequency_bands,
    high_frequency_loss,
    highpass,
    low_frequency_loss,
    robust_rms,
    texture_reliability_gate,
)
from sentinel_v3.model import ModelConfig, SentinelV3
from sentinel_v3.sensors import SENTINEL1, SENTINEL2
from sentinel_v3.training import (
    EMA,
    JointObjective,
    _load_compatible_state,
    _set_trainable,
    _stable_clip_grad_norm_,
    _stage_learning_rates,
    texture_benefit_target,
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


def test_laplacian_bands_reconstruct_highpass() -> None:
    values = torch.randn(2, 3, 32, 32)
    bands = frequency_bands(values, levels=3)
    assert len(bands) == 3
    assert all(band.shape == values.shape for band in bands)
    torch.testing.assert_close(sum(bands, torch.zeros_like(values)), highpass(values))


def test_texture_reliability_gate_rejects_unsupported_optical_texture() -> None:
    texture = torch.zeros(1, 3, 32, 32)
    texture[..., 8:24, 15:17] = 0.1
    mask = torch.ones(1, 1, 32, 32)
    _, unsupported = texture_reliability_gate(
        torch.zeros(1, 2, 32, 32), texture, mask, threshold=0.15
    )
    _, supported = texture_reliability_gate(texture, texture, mask, threshold=0.15)
    assert int(torch.count_nonzero(unsupported)) == 0
    assert int(torch.count_nonzero(supported)) > 0


def test_sar_frequency_loss_is_finite() -> None:
    prediction = highpass(torch.randn(2, 2, 32, 32))
    target = highpass(torch.randn(2, 2, 32, 32))
    loss, metrics = high_frequency_loss(prediction, target, torch.ones(2, 1, 32, 32), "sar")
    assert torch.isfinite(loss)
    assert "speckle_scale" in metrics


def test_frequency_loss_has_finite_gradient_at_zero_spectrum() -> None:
    prediction = torch.zeros(2, 3, 32, 32, requires_grad=True)
    target = highpass(torch.randn_like(prediction))
    loss, _ = high_frequency_loss(
        prediction, target, torch.ones(2, 1, 32, 32), "optical"
    )
    loss.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


def test_sar_tail_quantile_error_tracks_bright_and_dark_tails() -> None:
    target = torch.arange(100, dtype=torch.float32).reshape(1, 1, 10, 10)
    prediction = target.clone()
    mask = torch.ones(1, 1, 10, 10)
    assert float(tail_quantile_error(prediction, target, mask, 0.01)) == 0.0
    assert float(tail_quantile_error(prediction, target, mask, 0.99)) == 0.0
    prediction[..., 0, 0] -= 20.0
    prediction[..., -1, -1] += 20.0
    assert float(tail_quantile_error(prediction, target, mask, 0.01)) > 0.0
    assert float(tail_quantile_error(prediction, target, mask, 0.99)) > 0.0


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


def test_detail_head_is_conditioned_on_physical_base(tiny_model: SentinelV3) -> None:
    pyramid = tiny_model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32))
    base_head = tiny_model.detail_head.base_heads["optical"]
    output_head = tiny_model.detail_head.output_heads["optical"]
    with torch.no_grad():
        base_head.weight.fill_(0.01)
        output_head.weight.fill_(0.01)
    tiny_model.set_detail_confidence_threshold("optical", 0.0)
    without_base = tiny_model.deterministic_detail(
        pyramid, SENTINEL1, SENTINEL2, (32, 32), base=torch.zeros(1, 3, 32, 32)
    )
    with_base = tiny_model.deterministic_detail(
        pyramid, SENTINEL1, SENTINEL2, (32, 32), base=torch.ones(1, 3, 32, 32)
    )
    assert not torch.equal(with_base, without_base)
    torch.testing.assert_close(
        tiny_model.residual_dit.condition_gates,
        torch.zeros_like(tiny_model.residual_dit.condition_gates),
    )


def test_detail_confidence_has_one_map_per_frequency_band(tiny_model: SentinelV3) -> None:
    pyramid = tiny_model.encode(torch.randn(2, 2, 32, 32), SENTINEL1, torch.ones(2, 1, 32, 32))
    detail, bands, confidence = tiny_model.deterministic_detail_with_confidence(
        pyramid, SENTINEL1, SENTINEL2, (32, 32), torch.zeros(2, 3, 32, 32)
    )
    assert detail.shape == (2, 3, 32, 32)
    assert len(bands) == 3
    assert confidence.shape == (2, 3, 8, 8)
    assert bool(((confidence > 0) & (confidence < 1)).all())


def test_eval_detail_falls_back_when_confidence_is_below_threshold(
    tiny_model: SentinelV3,
) -> None:
    pyramid = tiny_model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32))
    with torch.no_grad():
        tiny_model.detail_head.output_heads["optical"].weight.fill_(0.01)
    tiny_model.eval()
    detail = tiny_model.deterministic_detail(
        pyramid, SENTINEL1, SENTINEL2, (32, 32), torch.zeros(1, 3, 32, 32)
    )
    torch.testing.assert_close(detail, torch.zeros_like(detail))
    tiny_model.train()


def test_eval_detail_uses_calibrated_modality_threshold(tiny_model: SentinelV3) -> None:
    pyramid = tiny_model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32))
    with torch.no_grad():
        tiny_model.detail_head.output_heads["optical"].weight.fill_(0.01)
        tiny_model.detail_head.confidence_heads["optical"].bias.fill_(2.0)
    tiny_model.eval()
    tiny_model.set_detail_confidence_threshold("optical", 1.01)
    suppressed = tiny_model.deterministic_detail(
        pyramid, SENTINEL1, SENTINEL2, (32, 32), torch.zeros(1, 3, 32, 32)
    )
    tiny_model.set_detail_confidence_threshold("optical", 0.0)
    released = tiny_model.deterministic_detail(
        pyramid, SENTINEL1, SENTINEL2, (32, 32), torch.zeros(1, 3, 32, 32)
    )
    torch.testing.assert_close(suppressed, torch.zeros_like(suppressed))
    assert int(torch.count_nonzero(released)) > 0


def test_production_detail_uses_hard_gate_during_flow_training(
    tiny_model: SentinelV3,
) -> None:
    pyramid = tiny_model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    with torch.no_grad():
        tiny_model.detail_head.output_heads["optical"].weight.fill_(0.01)
    tiny_model.train()
    tiny_model.set_detail_confidence_threshold("optical", 1.01)
    detail = tiny_model.deterministic_detail(
        pyramid,
        SENTINEL1,
        SENTINEL2,
        (32, 32),
        torch.zeros(1, 3, 32, 32),
    )
    torch.testing.assert_close(detail, torch.zeros_like(detail))


def test_optical_anchor_detail_is_physical_aligned_and_checkpoint_persistent(
    tiny_model: SentinelV3,
) -> None:
    pyramid = tiny_model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32))
    base = torch.rand(1, 3, 32, 32)
    tiny_model.eval()
    tiny_model.set_detail_confidence_threshold("optical", 1.01)
    tiny_model.set_optical_anchor_band_scales((0.1, 0.2, 0.3))
    detail = tiny_model.deterministic_detail(pyramid, SENTINEL1, SENTINEL2, (32, 32), base)
    bands = frequency_bands(base, levels=3)
    expected = highpass(0.1 * bands[0] + 0.2 * bands[1] + 0.3 * bands[2])
    torch.testing.assert_close(detail, expected)
    torch.testing.assert_close(
        tiny_model.state_dict()["optical_anchor_band_scales"],
        torch.tensor([0.1, 0.2, 0.3]),
    )


def test_optical_anchor_density_only_increases_supported_fine_blocks(
    tiny_model: SentinelV3,
) -> None:
    pyramid = tiny_model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32))
    base = torch.zeros(1, 3, 32, 32)
    base[..., 8:24, 15:17] = 0.5
    tiny_model.eval()
    tiny_model.set_detail_confidence_threshold("optical", 1.01)
    tiny_model.set_optical_anchor_band_scales((0.2, 0.0, 0.0))
    uniform = tiny_model.deterministic_detail(pyramid, SENTINEL1, SENTINEL2, (32, 32), base)
    tiny_model.set_optical_anchor_density(0.4, 1.5)
    adaptive = tiny_model.deterministic_detail(pyramid, SENTINEL1, SENTINEL2, (32, 32), base)
    assert adaptive[..., 8:24, 12:20].abs().mean() > uniform[..., 8:24, 12:20].abs().mean()
    gain = (adaptive - uniform).abs()
    assert gain[..., 8:24, 12:20].mean() > 10.0 * gain[..., :4, :4].mean()


def test_optical_source_density_only_modulates_supported_anchor_regions(
    tiny_model: SentinelV3,
) -> None:
    encoded = tiny_model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    source = torch.zeros_like(encoded[0])
    source[..., 8:24, 15:17] = 1.0
    pyramid = (source, encoded[1], encoded[2], encoded[3])
    base = torch.rand(1, 3, 32, 32)
    tiny_model.set_detail_confidence_threshold("optical", 1.01)
    tiny_model.set_optical_anchor_band_scales((0.2, 0.0, 0.0))
    uniform = tiny_model.deterministic_detail(
        pyramid, SENTINEL1, SENTINEL2, (32, 32), base
    )
    tiny_model.set_optical_anchor_source_density(0.4, 1.5)
    adaptive = tiny_model.deterministic_detail(
        pyramid, SENTINEL1, SENTINEL2, (32, 32), base
    )
    gain = (adaptive - uniform).abs()
    assert gain[..., 8:24, 12:20].mean() > 5.0 * gain[..., :4, :4].mean()


def test_source_aware_anchor_calibration_matches_runtime_highpass(
    tiny_model: SentinelV3,
) -> None:
    fine_band = torch.ones(1, 3, 32, 32)
    empty_band = torch.zeros_like(fine_band)
    source_density = torch.ones(1, 1, 8, 8)
    source_density[..., 2:6, 2:6] = 8.0
    raw_anchor = SentinelV3.source_aware_optical_anchor(
        (fine_band, empty_band, empty_band),
        source_density,
        torch.tensor((0.2, 0.0, 0.0)),
        0.0,
        1.0,
        0.4,
        1.0,
    )
    projected_anchor = highpass(raw_anchor)
    assert raw_anchor.amax() > raw_anchor.amin()
    assert not torch.allclose(raw_anchor, projected_anchor)

    encoded = tiny_model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    source = torch.zeros_like(encoded[0])
    source[..., 8:24, 12:20] = 1.0
    pyramid = (source, encoded[1], encoded[2], encoded[3])
    base = torch.rand(1, 3, 32, 32)
    valid = torch.ones(1, 1, 32, 32)
    valid[..., :8, :8] = 0.0
    tiny_model.eval()
    tiny_model.set_detail_confidence_threshold("optical", 1.01)
    tiny_model.set_optical_anchor_band_scales((0.2, 0.1, 0.05))
    tiny_model.set_optical_anchor_density(0.2, 1.0)
    tiny_model.set_optical_anchor_source_density(0.4, 1.0)
    runtime_anchor = tiny_model.deterministic_detail(
        pyramid, SENTINEL1, SENTINEL2, (32, 32), base
    )
    calibration_source_density = F.avg_pool2d(
        highpass(pyramid[0]).abs().mean(dim=1, keepdim=True), 4, stride=4
    )
    calibration_anchor = _calibration_anchor_detail(
        torch.stack(frequency_bands(base, levels=3), dim=1),
        calibration_source_density,
        valid,
        {
            "band_scales": tuple(float(value) for value in tiny_model.optical_anchor_band_scales),
            "density_gain": float(tiny_model.optical_anchor_density_gain),
            "density_threshold": float(tiny_model.optical_anchor_density_threshold),
            "source_gain": float(tiny_model.optical_anchor_source_gain),
            "source_threshold": float(tiny_model.optical_anchor_source_threshold),
        },
    )
    assert int(torch.count_nonzero(calibration_anchor[..., :8, :8])) == 0
    torch.testing.assert_close(calibration_anchor, runtime_anchor * valid)


def test_high_frequency_sampler_excludes_long_gap_shards() -> None:
    dataset = V2ShardDataset.__new__(V2ShardDataset)
    dataset.shards = [{"count": 2}, {"count": 2}]
    dataset.ends = [2, 4]
    dataset.prior_shards = [
        {"pair_id": "2017:tile:2017-01-01:ascending:2017-01-01"},
        {"pair_id": "2017:tile:2017-01-01:ascending:2017-01-04"},
    ]
    sampler = StatefulShardSampler(dataset, high_frequency_only=True)
    assert sorted(iter(sampler)) == [0, 1]


def test_audited_index_sampler_partitions_without_rank_overlap() -> None:
    left = list(StatefulIndexSampler([10, 20, 30, 40], replicas=2, rank=0, seed=3))
    right = list(StatefulIndexSampler([10, 20, 30, 40], replicas=2, rank=1, seed=3))
    assert not set(left).intersection(right)
    assert sorted(left + right) == [10, 20, 30, 40]


def test_frequency_adapter_is_identity_at_initialization(tiny_model: SentinelV3) -> None:
    adapter = tiny_model.residual_dit.frequency_adapter
    values = torch.randn(2, tiny_model.config.dit_hidden, 8, 8)
    torch.testing.assert_close(adapter(values), values)


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
    (
        ("detail", "detail_head"),
        ("codec", "codec"),
        ("flow", "residual_dit"),
        ("risk", "residual_dit"),
        ("bridge", "residual_dit"),
    ),
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


def test_codec_modality_filter_only_updates_selected_head(tiny_model: SentinelV3) -> None:
    tiny_model.codec.requires_grad_(False)
    tiny_model.codec.input_heads["sar"].requires_grad_(True)
    tiny_model.codec.output_heads["sar"].requires_grad_(True)
    objective = JointObjective(tiny_model, [0.5, 0.5], codec_train_modality="sar")
    loss, metrics = objective(_batch(), "codec")
    loss.backward()
    assert metrics
    assert all(name.startswith("opt2sar/") or name == "loss" for name in metrics)
    assert any(
        parameter.grad is not None
        for parameter in tiny_model.codec.input_heads["sar"].parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in tiny_model.codec.input_heads["optical"].parameters()
    )


def test_codec_perceptual_loss_is_scheduled(tiny_model: SentinelV3) -> None:
    objective = JointObjective(tiny_model, codec_perceptual_every=8)
    objective.set_progress(1, 10)
    assert objective.current_step == 1
    assert objective.codec_perceptual_every == 8


def test_flow_optimizes_the_composed_optical_endpoint(tiny_model: SentinelV3) -> None:
    objective = JointObjective(tiny_model, flow_rollout_every=8)
    objective.set_progress(1, 10)
    loss, metrics = objective(_batch(), "flow")
    assert "sar2opt/composed_hf_reconstruction" in metrics
    assert "sar2opt/composed_hf_gradient" in metrics
    assert "sar2opt/composed_pixel" in metrics
    assert 0.0 <= float(metrics["sar2opt/endpoint_source_weight"]) <= 1.0
    loss.backward()
    assert tiny_model.residual_dit.amplitude_head.weight.grad is not None
    assert float(tiny_model.residual_dit.amplitude_head.weight.grad.abs().sum()) > 0.0


def test_flow_composed_perceptual_loss_is_scheduled(
    tiny_model: SentinelV3, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Distance(torch.nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return (left - right).abs().mean(dim=(1, 2, 3), keepdim=True)

    monkeypatch.setattr(
        "sentinel_v3.evaluation.perceptual_evaluators",
        lambda _device: (Distance(), Distance()),
    )
    objective = JointObjective(tiny_model, flow_rollout_every=1, flow_rollout_steps=2)
    objective.set_progress(0, 10)
    loss, metrics = objective(_batch(), "flow")
    assert "sar2opt/rollout_lpips" in metrics
    assert "sar2opt/rollout_dists" in metrics
    assert "sar2opt/rollout_pixel" in metrics
    assert torch.isfinite(loss)


def test_flow_integrator_is_differentiable(tiny_model: SentinelV3) -> None:
    pyramid = tiny_model.encode(torch.randn(2, 2, 32, 32), SENTINEL1, torch.ones(2, 1, 32, 32))
    latent = torch.randn(2, tiny_model.config.codec_latent_channels, 8, 8)
    integrated = tiny_model.integrate_flow(latent, pyramid, SENTINEL2, 3, steps=2)
    assert integrated.shape == latent.shape
    integrated.square().mean().backward()
    assert tiny_model.residual_dit.output[-1].weight.grad is not None


def test_risk_stage_uses_target_free_candidate_and_updates_only_risk_head(
    tiny_model: SentinelV3,
) -> None:
    objective = JointObjective(tiny_model, risk_flow_steps=1)
    loss, metrics = objective(_batch(), "risk")
    assert torch.isfinite(loss)
    assert "sar2opt/positive_fraction" in metrics
    assert "opt2sar/positive_fraction" not in metrics
    loss.backward()
    assert tiny_model.residual_dit.texture_risk_head[-1].weight.grad is not None
    assert float(tiny_model.residual_dit.texture_risk_head[-1].weight.grad.abs().sum()) > 0.0
    assert tiny_model.residual_dit.output[-1].weight.grad is None


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


def test_texture_release_selection_falls_back_to_anchor() -> None:
    physical = {
        "detail_enabled": False,
        "alpha": 0.0,
        "visual_beneficial": False,
        "release_beneficial": False,
        "lpips_improvement": 0.0,
        "dists_improvement": 0.0,
    }
    anchor = {
        "detail_enabled": True,
        "alpha": 0.0,
        "visual_beneficial": True,
        "release_beneficial": False,
        "lpips_improvement": 0.03,
        "dists_improvement": 0.04,
    }
    unsafe_texture = {
        "detail_enabled": True,
        "alpha": 0.2,
        "visual_beneficial": True,
        "release_beneficial": False,
        "lpips_improvement": 0.04,
        "dists_improvement": 0.04,
    }
    selected = select_texture_release_candidate(
        [physical, anchor, unsafe_texture]
    )
    assert selected is anchor


def test_scene_conditioned_amplitude_shape(tiny_model: SentinelV3) -> None:
    pyramid = tiny_model.encode(torch.randn(2, 2, 32, 32), SENTINEL1, torch.ones(2, 1, 32, 32))
    amplitude = tiny_model.residual_amplitude(pyramid, SENTINEL2, 3, (32, 32))
    assert amplitude.shape == (2, 3, 8, 8)
    assert bool((amplitude <= tiny_model.config.optical_residual_limit).all())


def test_optical_bridge_has_independent_amplitude_calibration(
    tiny_model: SentinelV3,
) -> None:
    pyramid = tiny_model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    tiny_model.config.optical_bridge_enabled = True
    tiny_model.optical_alpha_scale.zero_()
    active = tiny_model.residual_amplitude(pyramid, SENTINEL2, 3, (32, 32))
    assert int(torch.count_nonzero(active)) > 0
    tiny_model.set_amplitude_scale("optical", 0.0)
    suppressed = tiny_model.residual_amplitude(pyramid, SENTINEL2, 3, (32, 32))
    torch.testing.assert_close(suppressed, torch.zeros_like(suppressed))
    assert float(tiny_model.optical_alpha_scale) == 0.0


def test_optical_texture_release_gate_is_blockwise_and_sar_is_unchanged(
    tiny_model: SentinelV3,
) -> None:
    amplitude = torch.tensor([[[[0.01, 0.03]], [[0.02, 0.04]], [[0.03, 0.05]]]])
    tiny_model.set_optical_texture_amplitude_floor(0.035)
    optical_gate = tiny_model.texture_release_gate(amplitude, SENTINEL2)
    sar_gate = tiny_model.texture_release_gate(amplitude[:, :2], SENTINEL1)
    torch.testing.assert_close(optical_gate, torch.tensor([[[[0.0, 1.0]]]]))
    torch.testing.assert_close(sar_gate, torch.ones_like(sar_gate))


def test_optical_texture_risk_threshold_is_validated(tiny_model: SentinelV3) -> None:
    tiny_model.set_optical_texture_risk_threshold(0.65)
    assert float(tiny_model.optical_texture_risk_threshold) == pytest.approx(0.65)
    with pytest.raises(ValueError):
        tiny_model.set_optical_texture_risk_threshold(1.1)


def test_optical_texture_floor_suppresses_low_amplitude_blocks(
    tiny_model: SentinelV3,
) -> None:
    pyramid = tiny_model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32))
    residual = torch.randn(1, 3, 32, 32)
    amplitude = torch.full((1, 3, 8, 8), 0.02)
    tiny_model.set_optical_texture_amplitude_floor(0.03)
    shaped = tiny_model.shape_residual_texture(
        residual, pyramid, SENTINEL2, amplitude=amplitude
    )
    torch.testing.assert_close(shaped, torch.zeros_like(shaped))


def test_optical_bridge_uses_independent_floor_and_skips_legacy_risk(
    tiny_model: SentinelV3,
) -> None:
    pyramid = tiny_model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    residual = torch.randn(1, 3, 32, 32)
    amplitude = torch.full((1, 3, 8, 8), 0.02)
    tiny_model.set_optical_texture_risk_threshold(1.0)
    tiny_model.optical_texture_amplitude_floor.fill_(0.16)
    tiny_model.config.optical_bridge_enabled = True
    shaped = tiny_model.shape_residual_texture(
        residual, pyramid, SENTINEL2, amplitude=amplitude
    )
    assert int(torch.count_nonzero(shaped)) > 0
    tiny_model.set_optical_texture_amplitude_floor(0.03)
    suppressed = tiny_model.shape_residual_texture(
        residual, pyramid, SENTINEL2, amplitude=amplitude
    )
    torch.testing.assert_close(suppressed, torch.zeros_like(suppressed))


def test_texture_risk_head_predicts_one_probability_per_block(
    tiny_model: SentinelV3,
) -> None:
    pyramid = tiny_model.encode(torch.randn(2, 2, 32, 32), SENTINEL1, torch.ones(2, 1, 32, 32))
    texture = torch.randn(2, 3, 32, 32) * 0.02
    probability = tiny_model.texture_release_probability(pyramid, SENTINEL2, texture)
    assert probability.shape == (2, 1, 8, 8)
    assert bool(((probability > 0.0) & (probability < 1.0)).all())


def test_texture_benefit_target_prefers_candidate_closer_to_reference() -> None:
    target = torch.rand(1, 3, 32, 32)
    physical = torch.zeros_like(target)
    valid = torch.ones(1, 1, 32, 32)
    good_probability, good_benefit, support = texture_benefit_target(
        physical, target, target, valid
    )
    bad_probability, bad_benefit, _ = texture_benefit_target(
        physical, physical * 2.0, target, valid
    )
    assert bool((good_benefit > 0).all())
    assert bool((good_probability > 0.5).all())
    assert bool((bad_benefit == 0).all())
    assert float(good_probability.mean()) > float(bad_probability.mean())
    torch.testing.assert_close(support, torch.ones_like(support))


def test_modality_specific_flow_noise_preserves_legacy_fallback(
    tiny_model: SentinelV3,
) -> None:
    tiny_model.config.flow_noise_scale = 0.35
    tiny_model.config.optical_flow_noise_scale = 0.05
    tiny_model.config.sar_flow_noise_scale = None
    assert tiny_model.flow_noise_scale(SENTINEL2) == pytest.approx(0.05)
    assert tiny_model.flow_noise_scale(SENTINEL1) == pytest.approx(0.35)


def test_sampled_sar_texture_preserves_each_block_mean(tiny_model: SentinelV3) -> None:
    pyramid = tiny_model.encode(torch.rand(1, 10, 32, 32), SENTINEL2, torch.ones(1, 1, 32, 32))
    texture = tiny_model.sample_residual(pyramid, SENTINEL1, (1, 2, 32, 32), seed=7)
    block_mean = torch.nn.functional.avg_pool2d(texture, 4, stride=4)
    torch.testing.assert_close(block_mean, torch.zeros_like(block_mean), atol=2e-6, rtol=0)


def test_shaped_optical_texture_uses_amplitude_and_limits_chroma(
    tiny_model: SentinelV3,
) -> None:
    pyramid = tiny_model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32))
    residual = torch.randn(1, 3, 32, 32)
    amplitude = torch.full((1, 3, 8, 8), 0.02, requires_grad=True)
    shaped = tiny_model.shape_residual_texture(
        residual, pyramid, SENTINEL2, amplitude=amplitude
    )
    assert shaped.shape == residual.shape
    chroma = shaped - shaped.mean(dim=1, keepdim=True)
    assert float(chroma.detach().abs().max()) <= 0.03001
    shaped.abs().mean().backward()
    assert amplitude.grad is not None
    assert int(torch.count_nonzero(amplitude.grad)) > 0


def test_shaped_texture_has_finite_gradient_at_zero_block_rms(
    tiny_model: SentinelV3,
) -> None:
    pyramid = tiny_model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    residual = torch.zeros(1, 3, 32, 32, requires_grad=True)
    amplitude = torch.full((1, 3, 8, 8), 0.02)
    shaped = tiny_model.shape_residual_texture(
        residual, pyramid, SENTINEL2, amplitude=amplitude
    )
    shaped.sum().backward()
    assert residual.grad is not None
    assert bool(torch.isfinite(residual.grad).all())


def test_stage_learning_rates_match_v32_schedule() -> None:
    config = {"learning_rate": 1e-4, "encoder_learning_rate": 2e-5}
    assert _stage_learning_rates(config, "flow") == pytest.approx((0.0, 0.0, 1e-4))
    assert _stage_learning_rates(config, "risk") == pytest.approx((0.0, 0.0, 1e-4))
    assert _stage_learning_rates(config, "bridge") == pytest.approx((0.0, 0.0, 1e-4))
    assert _stage_learning_rates(config, "balance") == pytest.approx((2e-6, 1e-5, 1e-5))


def test_optical_bridge_requires_matching_anchor_latent(tiny_model: SentinelV3) -> None:
    pyramid = tiny_model.encode(
        torch.randn(2, 2, 32, 32), SENTINEL1, torch.ones(2, 1, 32, 32)
    )
    latent = torch.randn(2, tiny_model.config.codec_latent_channels, 8, 8)
    time = torch.rand(2)
    with pytest.raises(ValueError, match="anchor latent"):
        tiny_model.flow_velocity(
            latent,
            time,
            pyramid,
            SENTINEL2,
            3,
            use_optical_bridge=True,
        )


def test_bridge_stage_only_unfreezes_optical_bridge_parameters(
    tiny_model: SentinelV3,
) -> None:
    _set_trainable(tiny_model, "bridge")
    trainable = {
        name for name, parameter in tiny_model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("residual_dit.optical_bridge_") for name in trainable)
    assert not tiny_model.residual_dit.output[-1].weight.requires_grad
    assert not any(parameter.requires_grad for parameter in tiny_model.codec.parameters())
    assert not any(parameter.requires_grad for parameter in tiny_model.encoder.parameters())


def test_bridge_objective_updates_only_optical_bridge(tiny_model: SentinelV3) -> None:
    tiny_model.config.optical_bridge_enabled = True
    tiny_model.set_optical_anchor_band_scales((0.2, 0.0, 0.0))
    _set_trainable(tiny_model, "bridge")
    objective = JointObjective(tiny_model, [0.5, 0.5], flow_perceptual_every=8)
    objective.set_progress(1, 10)
    loss, metrics = objective(_batch(), "bridge")
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics
    assert all(name.startswith("sar2opt/") or name == "loss" for name in metrics)
    bridge_output = tiny_model.residual_dit.optical_bridge_output[-1].weight
    assert bridge_output.grad is not None
    assert int(torch.count_nonzero(bridge_output.grad)) > 0
    assert tiny_model.residual_dit.output[-1].weight.grad is None
    assert all(parameter.grad is None for parameter in tiny_model.codec.parameters())


def test_zero_bridge_perceptual_weight_skips_evaluator(tiny_model: SentinelV3) -> None:
    tiny_model.config.optical_bridge_enabled = True
    tiny_model.set_optical_anchor_band_scales((0.2, 0.0, 0.0))
    objective = JointObjective(
        tiny_model,
        [0.5, 0.5],
        flow_visual_perceptual_weight=0.0,
    )
    objective._optical_visual_perceptual = lambda *args: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("perceptual evaluator should be skipped")
    )
    objective.set_progress(0, 10)
    loss, _ = objective(_batch(), "bridge")
    assert torch.isfinite(loss)


def test_stable_gradient_clipping_handles_overflowing_naive_norm() -> None:
    parameter = torch.nn.Parameter(torch.ones(8))
    parameter.grad = torch.full_like(parameter, 1e30)
    norm, maximum = _stable_clip_grad_norm_([parameter], 1.0)
    assert norm == pytest.approx(8**0.5 * 1e30, rel=1e-6)
    assert maximum == pytest.approx(1e30, rel=1e-6)
    assert torch.isfinite(parameter.grad).all()
    assert float(parameter.grad.norm()) == pytest.approx(1.0)


def test_enabling_optical_bridge_does_not_change_sar_sampling(
    tiny_model: SentinelV3,
) -> None:
    pyramid = tiny_model.encode(
        torch.rand(1, 10, 32, 32), SENTINEL2, torch.ones(1, 1, 32, 32)
    )
    tiny_model.config.optical_bridge_enabled = False
    baseline = tiny_model.sample_residual(
        pyramid, SENTINEL1, (1, 2, 32, 32), seed=19
    )
    tiny_model.config.optical_bridge_enabled = True
    actual = tiny_model.sample_residual(
        pyramid, SENTINEL1, (1, 2, 32, 32), seed=19
    )
    torch.testing.assert_close(actual, baseline)


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
