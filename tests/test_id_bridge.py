from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sentinel_v3.api import Observation, TargetRequest, translate
from sentinel_v3.config import load_config
from sentinel_v3.losses import (
    cross_modal_identifiability_target,
    haar_dwt2,
    haar_idwt2,
    haar_packet_dwt2,
    haar_packet_idwt2,
    high_frequency_loss,
    highpass,
)
from sentinel_v3.model import ModelConfig, SentinelV3
from sentinel_v3.sensors import SENTINEL1, SENTINEL2
from sentinel_v3.training import (
    EMA,
    JointObjective,
    _checkpoint_payload,
    _optimizer,
    _scheduler,
    _set_trainable,
)


def _batch(delta_days: int = 0) -> dict[str, object]:
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
        "hf_eligible": torch.ones(batch_size, dtype=torch.bool),
    }


def _haar_model(
    *,
    anchor_origin: bool = False,
    optical_innovation_scale: float = 1.0,
    sar_innovation_scale: float = 1.0,
) -> SentinelV3:
    return SentinelV3(
        ModelConfig(
            width=8,
            hidden=32,
            encoder_depth=1,
            heads=4,
            adapter_rank=8,
            dit_hidden=32,
            dit_depth=1,
            dit_heads=4,
            codec_width=8,
            codec_latent_channels=16,
            flow_steps=2,
            id_bridge_enabled=True,
            id_bridge_state="haar_packet",
            id_bridge_state_channels=48,
            id_bridge_optical_state_scale=0.03,
            id_bridge_sar_state_scale=4.0,
            id_bridge_anchor_origin=anchor_origin,
            id_bridge_optical_innovation_scale=optical_innovation_scale,
            id_bridge_sar_innovation_scale=sar_innovation_scale,
        )
    )


def test_haar_roundtrip_is_orthonormal_and_rejects_odd_dimensions() -> None:
    values = torch.randn(2, 3, 32, 24)
    coefficients = haar_dwt2(values)
    assert coefficients.shape == (2, 3, 4, 16, 12)
    torch.testing.assert_close(haar_idwt2(coefficients), values, atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError, match="even"):
        haar_dwt2(torch.randn(1, 1, 31, 32))


def test_haar_packet_roundtrip_parseval_and_dimension_requirements() -> None:
    values = torch.randn(2, 3, 32, 24)
    coefficients = haar_packet_dwt2(values)
    assert coefficients.shape == (2, 48, 8, 6)
    torch.testing.assert_close(haar_packet_idwt2(coefficients), values, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(coefficients.square().sum(), values.square().sum(), rtol=2e-6, atol=2e-4)
    with pytest.raises(ValueError, match="divisible by four"):
        haar_packet_dwt2(torch.randn(1, 1, 31, 32))
    with pytest.raises(ValueError, match="divisible by four"):
        haar_packet_dwt2(torch.randn(1, 1, 30, 32))
    with pytest.raises(ValueError, match="divisible by sixteen"):
        haar_packet_idwt2(torch.randn(1, 15, 8, 8))


def test_haar_id_bridge_config_validation() -> None:
    with pytest.raises(ValueError, match="codec or haar_packet"):
        ModelConfig(id_bridge_state="unknown")
    with pytest.raises(ValueError, match="must be 48"):
        ModelConfig(id_bridge_state="haar_packet", id_bridge_state_channels=47)
    with pytest.raises(ValueError, match="positive"):
        ModelConfig(id_bridge_optical_state_scale=0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ModelConfig(id_bridge_optical_innovation_scale=-0.01)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ModelConfig(id_bridge_sar_innovation_scale=1.01)


def test_haar_id_bridge_smoke_config_selects_packet_state() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_id_bridge_haar.yaml")
    assert config["model"]["id_bridge_state"] == "haar_packet"
    assert config["model"]["id_bridge_state_channels"] == 48
    assert config["model"]["optical_flow_noise_scale"] == pytest.approx(0.35)
    assert config["model"]["sar_flow_noise_scale"] == pytest.approx(0.35)


def test_haar_anchor_config_selects_observable_optical_origin() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "id_bridge_haar_anchor.yaml")
    assert config["model"]["id_bridge_anchor_origin"] is True
    assert config["model"]["id_bridge_optical_innovation_scale"] == 0.0
    assert config["model"]["id_bridge_sar_innovation_scale"] == 1.0


def test_haar_id_bridge_uses_exact_packet_state_without_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model()
    assert model.codec.latent_channels == 16
    assert model.id_bridge_origin.latent_channels == 48
    assert model.residual_dit.latent_channels == 48
    assert int(torch.count_nonzero(model.residual_dit.output[-1].weight)) == 0
    assert int(torch.count_nonzero(model.residual_dit.output[-1].bias)) == 0

    def codec_called(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("Haar id bridge must not call the residual codec")

    monkeypatch.setattr(model.codec, "encode", codec_called)
    monkeypatch.setattr(model.codec, "decode", codec_called)
    optical = torch.randn(2, 3, 32, 32)
    state = model.encode_id_bridge_residual(optical, SENTINEL2)
    assert state.shape == (2, 48, 8, 8)
    assert int(torch.count_nonzero(state[:, [0, 16, 32]])) == 0
    assert model.decode_id_bridge_residual(state, SENTINEL2).shape == optical.shape


def test_haar_id_bridge_projection_and_packet_decode_constraints() -> None:
    model = _haar_model()
    optical = torch.randn(2, 3, 32, 32)
    projected_optical = model.project_id_bridge_residual(optical, SENTINEL2)
    optical_state = model.encode_id_bridge_residual(projected_optical, SENTINEL2)
    assert int(torch.count_nonzero(optical_state[:, [0, 16, 32]])) == 0
    torch.testing.assert_close(
        model.decode_id_bridge_residual(optical_state, SENTINEL2),
        projected_optical,
        atol=3e-6,
        rtol=3e-6,
    )
    constant = model.project_id_bridge_residual(torch.ones(1, 3, 32, 32), SENTINEL2)
    assert int(torch.count_nonzero(constant)) == 0

    sar = torch.randn(2, 2, 32, 32)
    projected_sar = model.project_id_bridge_residual(sar, SENTINEL1)
    sar_state = model.encode_id_bridge_residual(projected_sar, SENTINEL1)
    assert sar_state.shape == (2, 48, 8, 8)
    assert int(torch.count_nonzero(sar_state[:, [0, 16]])) == 0
    assert int(torch.count_nonzero(sar_state[:, 32:])) == 0
    torch.testing.assert_close(
        model.decode_id_bridge_residual(sar_state, SENTINEL1),
        projected_sar,
        atol=3e-6,
        rtol=3e-6,
    )

    unconstrained = torch.randn_like(sar_state)
    unconstrained[:, [0, 16]] = 1.0
    unconstrained[:, 32:] = 1.0
    decoded = model.decode_id_bridge_residual(unconstrained, SENTINEL1)
    reencoded = model.encode_id_bridge_residual(decoded, SENTINEL1)
    expected = unconstrained.clone()
    expected[:, [0, 16]] = 0.0
    expected[:, 32:] = 0.0
    torch.testing.assert_close(reencoded, expected, atol=3e-6, rtol=3e-6)


def test_observable_anchor_origin_matches_frozen_detail_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(anchor_origin=True, optical_innovation_scale=0.0)
    pyramid = model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    physical = torch.rand(1, 3, 32, 32)
    detail = torch.randn_like(physical)
    expected = detail.detach()
    calls = 0

    def deterministic_detail(
        _pyramid: object,
        source: object,
        target: object,
        output_size: tuple[int, int],
        *,
        base: torch.Tensor | None,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        assert source == SENTINEL1 and target == SENTINEL2
        assert output_size == (32, 32) and base is physical
        return detail

    monkeypatch.setattr(model, "deterministic_detail", deterministic_detail)
    mu, correction, anchor_detail, _, _ = model.predict_id_bridge_origin_components(
        pyramid, physical, SENTINEL2
    )
    assert calls == 1
    assert int(torch.count_nonzero(correction)) == 0
    assert not anchor_detail.requires_grad
    torch.testing.assert_close(anchor_detail, expected)
    torch.testing.assert_close(mu, torch.zeros_like(mu))

    default_model = _haar_model()
    default_pyramid = default_model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    default_mu, _, _ = default_model.predict_id_bridge_origin(
        default_pyramid, physical, SENTINEL2
    )
    assert int(torch.count_nonzero(default_mu)) == 0


def test_observable_anchor_reliability_gates_raw_correction_and_detaches_logits() -> None:
    model = _haar_model(anchor_origin=True, optical_innovation_scale=0.0)
    valid = torch.ones(1, 1, 32, 32)
    pyramid = tuple(level.detach() for level in model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid))
    physical = torch.rand(1, 3, 32, 32)
    head = model.id_bridge_origin.output_heads["optical"][-1]

    with torch.no_grad():
        head.bias.zero_()
        head.bias[:48].fill_(2.0)
        head.bias[-3:].fill_(-100.0)
    mu_low, correction_low, anchor_low, _, _ = model.predict_id_bridge_origin_components(
        pyramid, physical, SENTINEL2
    )
    assert anchor_low.shape == physical.shape
    torch.testing.assert_close(mu_low, torch.zeros_like(mu_low))
    assert int(torch.count_nonzero(correction_low)) > 0

    with torch.no_grad():
        head.bias[-3:].fill_(100.0)
    mu_high, correction_high, _anchor_high, _, _ = model.predict_id_bridge_origin_components(
        pyramid, physical, SENTINEL2
    )
    torch.testing.assert_close(mu_high, correction_high)

    with torch.no_grad():
        head.bias[-3:].zero_()
    mu, correction, _, _, reliability_logits = model.predict_id_bridge_origin_components(
        pyramid, physical, SENTINEL2
    )
    correction.retain_grad()
    reliability_logits.retain_grad()
    model.decode_id_bridge_residual(mu, SENTINEL2).square().mean().backward()
    assert correction.grad is not None and int(torch.count_nonzero(correction.grad)) > 0
    assert reliability_logits.grad is None or int(torch.count_nonzero(reliability_logits.grad)) == 0

    _, _, _, _, bce_logits = model.predict_id_bridge_origin_components(pyramid, physical, SENTINEL2)
    bce_logits.retain_grad()
    F.binary_cross_entropy_with_logits(bce_logits, torch.zeros_like(bce_logits)).backward()
    assert bce_logits.grad is not None and int(torch.count_nonzero(bce_logits.grad)) > 0

    legacy_model = _haar_model()
    legacy_head = legacy_model.id_bridge_origin.output_heads["optical"][-1]
    with torch.no_grad():
        legacy_head.bias.zero_()
        legacy_head.bias[:48].fill_(2.0)
        legacy_head.bias[-3:].fill_(-100.0)
    legacy_pyramid = legacy_model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    legacy_mu, legacy_correction, legacy_anchor, _, _ = (
        legacy_model.predict_id_bridge_origin_components(legacy_pyramid, physical, SENTINEL2)
    )
    assert int(torch.count_nonzero(legacy_anchor)) == 0
    torch.testing.assert_close(legacy_mu, legacy_correction)


def test_observable_anchor_innovation_gate_obeys_reliability_and_scale() -> None:
    model = _haar_model(
        anchor_origin=True,
        optical_innovation_scale=0.25,
        sar_innovation_scale=0.5,
    )
    latent = torch.randn(2, 48, 8, 8)
    mu = torch.randn_like(latent)
    q_one_logits = torch.full((2, 3, 8, 8), 100.0)
    q_zero_logits = torch.full((2, 3, 8, 8), -100.0)
    torch.testing.assert_close(
        model.gate_id_bridge_innovation(latent, mu, q_one_logits, SENTINEL2), mu
    )
    torch.testing.assert_close(
        model.gate_id_bridge_innovation(latent, mu, q_zero_logits, SENTINEL2),
        mu + 0.25 * (latent - mu),
    )
    torch.testing.assert_close(
        model.gate_id_bridge_innovation(latent, mu, q_zero_logits, SENTINEL1),
        mu + 0.5 * (latent - mu),
    )
    with pytest.raises(ValueError, match="B3HW"):
        model.gate_id_bridge_innovation(latent, mu, q_zero_logits[:, :2], SENTINEL2)

    legacy_model = _haar_model()
    torch.testing.assert_close(
        legacy_model.gate_id_bridge_innovation(latent, mu, q_zero_logits, SENTINEL2), latent
    )


def test_observable_anchor_optical_sampling_drops_transport_innovation() -> None:
    model = _haar_model(anchor_origin=True, optical_innovation_scale=0.0)
    valid = torch.ones(1, 1, 32, 32)
    optical_pyramid = model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    physical_optical = torch.rand(1, 3, 32, 32)
    optical_first, diagnostics = model.sample_id_bridge_residual(
        optical_pyramid, physical_optical, SENTINEL2, seed=17, return_origin=True
    )
    optical_second = model.sample_id_bridge_residual(
        optical_pyramid, physical_optical, SENTINEL2, seed=18
    )
    torch.testing.assert_close(optical_first, optical_second)
    assert diagnostics.keys() >= {"anchor_detail", "correction_gate", "innovation_gate"}
    assert int(torch.count_nonzero(diagnostics["innovation_gate"])) == 0
    torch.testing.assert_close(
        diagnostics["correction_gate"],
        torch.full_like(diagnostics["correction_gate"], 0.5),
    )

    sar_pyramid = model.encode(torch.rand(1, 10, 32, 32), SENTINEL2, valid)
    physical_sar = torch.randn(1, 2, 32, 32)
    sar_first = model.sample_id_bridge_residual(sar_pyramid, physical_sar, SENTINEL1, seed=17)
    sar_second = model.sample_id_bridge_residual(sar_pyramid, physical_sar, SENTINEL1, seed=18)
    assert not torch.equal(sar_first, sar_second)

    legacy_model = _haar_model()
    legacy_pyramid = legacy_model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    _, legacy_diagnostics = legacy_model.sample_id_bridge_residual(
        legacy_pyramid, physical_optical, SENTINEL2, seed=17, return_origin=True
    )
    torch.testing.assert_close(
        legacy_diagnostics["innovation_gate"],
        torch.ones_like(legacy_diagnostics["innovation_gate"]),
    )
    torch.testing.assert_close(
        legacy_diagnostics["correction_gate"],
        torch.ones_like(legacy_diagnostics["correction_gate"]),
    )


def test_observable_anchor_step_zero_preserves_pixel_detail_without_double_add(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(anchor_origin=True, optical_innovation_scale=0.0).eval()
    valid = torch.ones(1, 1, 32, 32)
    pyramid = model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    base = torch.rand(1, 3, 32, 32)
    anchor = torch.full_like(base, 0.02)

    def deterministic_detail(*_args: object, **_kwargs: object) -> torch.Tensor:
        return anchor

    monkeypatch.setattr(model, "deterministic_detail", deterministic_detail)
    detail = model.visual_detail(pyramid, SENTINEL1, SENTINEL2, (32, 32), base)
    first, diagnostics = model.sample_id_bridge_residual(
        pyramid, base, SENTINEL2, seed=17, return_origin=True
    )
    second = model.sample_id_bridge_residual(pyramid, base, SENTINEL2, seed=18)
    zeros = torch.zeros_like(first)
    torch.testing.assert_close(detail, anchor)
    torch.testing.assert_close(diagnostics["anchor_detail"], anchor)
    torch.testing.assert_close(first, zeros, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(second, zeros, atol=1e-7, rtol=0.0)
    expected = model.compose_visual(base, anchor, zeros, "optical")
    actual = model.compose_visual(base, detail, first, "optical")
    assert isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected)


def test_observable_anchor_shrink_penalizes_correction_not_anchor() -> None:
    anchor = torch.randn(1, 48, 8, 8, requires_grad=True)
    correction = torch.randn(1, 48, 8, 8, requires_grad=True)
    mu = anchor + correction
    endpoint = torch.randn_like(mu)
    q_oracle = torch.zeros(1, 1, 8, 8)
    values = JointObjective._id_bridge_anchor_values(mu, correction, endpoint, q_oracle)
    values.mean().backward()
    assert anchor.grad is None or int(torch.count_nonzero(anchor.grad)) == 0
    assert correction.grad is not None and int(torch.count_nonzero(correction.grad)) > 0


def test_observable_anchor_optical_gate_blocks_transport_visual_gradient() -> None:
    model = _haar_model(anchor_origin=True, optical_innovation_scale=0.0)
    latent = torch.randn(1, 48, 8, 8, requires_grad=True)
    mu = torch.randn_like(latent, requires_grad=True)
    logits = torch.zeros(1, 3, 8, 8)
    residual = model.decode_id_bridge_residual(
        model.gate_id_bridge_innovation(latent, mu, logits, SENTINEL2), SENTINEL2
    )
    residual.square().mean().backward()
    assert mu.grad is not None and int(torch.count_nonzero(mu.grad)) > 0
    assert latent.grad is None or int(torch.count_nonzero(latent.grad)) == 0


def test_observable_anchor_optical_endpoint_keeps_dit_velocity_gradient() -> None:
    model = _haar_model(anchor_origin=True, optical_innovation_scale=0.0)
    _set_trainable(model, "id_bridge")
    objective = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=1,
        flow_visual_perceptual_weight=0.0,
    )
    loss, metrics = objective._id_bridge_direction(
        _batch(),
        torch.tensor([0]),
        "sar_view",
        "s2_target",
        SENTINEL1,
        SENTINEL2,
        torch.ones(4),
    )
    assert {
        "origin_hf_reconstruction",
        "origin_hf_gradient",
        "origin_hf_spectrum",
        "origin_low_frequency",
    } <= metrics.keys()
    loss.backward()
    gradient = model.residual_dit.output[-1].weight.grad
    assert gradient is not None and int(torch.count_nonzero(gradient)) > 0


def test_observable_anchor_origin_hf_supervision_updates_raw_correction() -> None:
    model = _haar_model(anchor_origin=True, optical_innovation_scale=0.0)
    objective = JointObjective(model, [0.5, 0.5])
    indices = torch.tensor([0])
    target, base, valid, pyramid = objective._physical_context(
        _batch(),
        indices,
        "sar_view",
        "s2_target",
        SENTINEL1,
        SENTINEL2,
        joint=False,
    )
    full_residual = highpass((target - base.detach()) * valid) * valid
    mu, correction, anchor_detail, _, _ = model.predict_id_bridge_origin_components(
        pyramid, base, SENTINEL2
    )
    assert not anchor_detail.requires_grad
    correction.retain_grad()
    origin_loss, _ = high_frequency_loss(
        anchor_detail + model.decode_id_bridge_residual(mu, SENTINEL2),
        full_residual,
        valid,
        "optical",
        sample_weight=torch.ones(1),
    )
    origin_loss.backward()
    assert correction.grad is not None and int(torch.count_nonzero(correction.grad)) > 0


def test_observable_anchor_sar_skips_origin_hf_supervision() -> None:
    model = _haar_model(anchor_origin=True)
    objective = JointObjective(model, [0.5, 0.5], flow_rollout_every=8)
    objective.set_progress(1, 10)
    loss, metrics = objective._id_bridge_direction(
        _batch(),
        torch.tensor([0]),
        "s2_view",
        "sar_target",
        SENTINEL2,
        SENTINEL1,
        torch.ones(4),
    )
    assert torch.isfinite(loss)
    assert not any(name.startswith("origin_") for name in metrics)


def test_id_bridge_rollout_perceptual_uses_schedule_compensation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model().eval()
    schedule = 4
    perceptual_weight = 0.125
    baseline = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=schedule,
        flow_visual_perceptual_weight=0.0,
    )
    weighted = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=schedule,
        flow_visual_perceptual_weight=perceptual_weight,
    )

    def perceptual(prediction: torch.Tensor, *_: object) -> tuple[torch.Tensor, torch.Tensor]:
        zero = prediction.sum() * 0.0
        return zero + 2.0, zero + 3.0

    monkeypatch.setattr(weighted, "_optical_visual_perceptual", perceptual)
    baseline.set_progress(schedule, 10)
    weighted.set_progress(schedule, 10)
    arguments = (
        _batch(),
        torch.tensor([0]),
        "sar_view",
        "s2_target",
        SENTINEL1,
        SENTINEL2,
        torch.ones(4),
    )
    torch.manual_seed(29)
    baseline_loss, _ = baseline._id_bridge_direction(*arguments)
    torch.manual_seed(29)
    weighted_loss, metrics = weighted._id_bridge_direction(*arguments)
    expected = schedule * perceptual_weight * (2.0 + 3.0)
    assert float((weighted_loss - baseline_loss).detach()) == pytest.approx(expected, abs=1e-5)
    assert float(metrics["rollout_lpips"]) == pytest.approx(2.0)
    assert float(metrics["rollout_dists"]) == pytest.approx(3.0)


def test_cross_modal_identifiability_rejects_shifted_structure() -> None:
    source = torch.zeros(1, 2, 32, 32)
    source[..., 8:24, 14:18] = 1.0
    energy = highpass(source).abs().mean(dim=1, keepdim=True)
    aligned = cross_modal_identifiability_target(
        source, (energy, energy, energy), torch.ones(1, 1, 32, 32)
    )
    shifted = cross_modal_identifiability_target(
        source,
        tuple(torch.roll(energy, shifts=8, dims=-1) for _ in range(3)),
        torch.ones(1, 1, 32, 32),
    )
    assert aligned.shape == (1, 3, 8, 8)
    assert float(aligned.mean()) > float(shifted.mean()) + 0.05


def test_id_bridge_origin_shapes_and_reliability_are_finite(tiny_model: SentinelV3) -> None:
    pyramid = tiny_model.encode(
        torch.randn(2, 2, 32, 32), SENTINEL1, torch.ones(2, 1, 32, 32)
    )
    mu, log_sigma, reliability_logits = tiny_model.predict_id_bridge_origin(
        pyramid, torch.rand(2, 3, 32, 32), SENTINEL2
    )
    assert mu.shape == (2, tiny_model.config.codec_latent_channels, 8, 8)
    assert log_sigma.shape == mu.shape
    assert reliability_logits.shape == (2, 3, 8, 8)
    assert bool(torch.isfinite(log_sigma).all())
    q = torch.sigmoid(reliability_logits)
    assert bool(((q > 0.0) & (q < 1.0)).all())


def test_id_bridge_residual_is_reproducible_for_a_fixed_seed(tiny_model: SentinelV3) -> None:
    pyramid = tiny_model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    physical = torch.rand(1, 3, 32, 32)
    first = tiny_model.sample_id_bridge_residual(pyramid, physical, SENTINEL2, seed=17)
    second = tiny_model.sample_id_bridge_residual(pyramid, physical, SENTINEL2, seed=17)
    third = tiny_model.sample_id_bridge_residual(pyramid, physical, SENTINEL2, seed=18)
    assert isinstance(first, torch.Tensor)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, third)


def test_haar_id_bridge_residual_is_reproducible_for_a_fixed_seed() -> None:
    model = _haar_model()
    pyramid = model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    physical = torch.rand(1, 3, 32, 32)
    first = model.sample_id_bridge_residual(pyramid, physical, SENTINEL2, seed=17)
    second = model.sample_id_bridge_residual(pyramid, physical, SENTINEL2, seed=17)
    third = model.sample_id_bridge_residual(pyramid, physical, SENTINEL2, seed=18)
    assert isinstance(first, torch.Tensor)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, third)


def test_visual_detail_and_residual_dispatch_id_and_legacy_paths(
    tiny_model: SentinelV3, monkeypatch: pytest.MonkeyPatch
) -> None:
    pyramid = tiny_model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    base = torch.rand(1, 3, 32, 32)
    calls: list[str] = []

    def legacy_detail(*_args: object, **_kwargs: object) -> torch.Tensor:
        calls.append("legacy_detail")
        return torch.ones_like(base)

    def legacy_sample(
        _pyramid: object,
        _target: object,
        shape: tuple[int, int, int, int],
        *,
        seed: int,
        steps: int | None,
        bridge_anchor: torch.Tensor | None,
    ) -> torch.Tensor:
        calls.append("legacy_sample")
        assert shape == tuple(base.shape)
        assert seed == 3 and steps is None and bridge_anchor is not None
        return torch.full_like(base, 0.1)

    monkeypatch.setattr(tiny_model, "deterministic_detail", legacy_detail)
    monkeypatch.setattr(tiny_model, "sample_residual", legacy_sample)
    detail = tiny_model.visual_detail(pyramid, SENTINEL1, SENTINEL2, (32, 32), base)
    residual = tiny_model.sample_visual_residual(pyramid, SENTINEL2, base, detail, seed=3)
    torch.testing.assert_close(detail, torch.ones_like(base))
    torch.testing.assert_close(residual, torch.full_like(base, 0.1))
    assert calls == ["legacy_detail", "legacy_sample"]

    def id_sample(
        _pyramid: object,
        physical: torch.Tensor,
        _target: object,
        *,
        seed: int,
        steps: int | None,
        return_origin: bool = False,
    ) -> torch.Tensor:
        calls.append("id_sample")
        assert physical is base and seed == 5 and steps is None and not return_origin
        return torch.full_like(base, 0.2)

    tiny_model.config.id_bridge_enabled = True
    monkeypatch.setattr(tiny_model, "sample_id_bridge_residual", id_sample)
    id_detail = tiny_model.visual_detail(pyramid, SENTINEL1, SENTINEL2, (32, 32), base)
    id_residual = tiny_model.sample_visual_residual(pyramid, SENTINEL2, base, id_detail, seed=5)
    torch.testing.assert_close(id_detail, torch.zeros_like(base))
    torch.testing.assert_close(id_residual, torch.full_like(base, 0.2))
    assert calls == ["legacy_detail", "legacy_sample", "id_sample"]


def test_id_bridge_only_opens_origin_and_generic_flow_parameters(tiny_model: SentinelV3) -> None:
    _set_trainable(tiny_model, "id_bridge")
    trainable = {
        name for name, parameter in tiny_model.named_parameters() if parameter.requires_grad
    }
    allowed = (
        "id_bridge_origin.",
        "residual_dit.input.",
        "residual_dit.scene_projections.",
        "residual_dit.condition_gates",
        "residual_dit.frequency_adapter.",
        "residual_dit.time.",
        "residual_dit.target.",
        "residual_dit.blocks.",
        "residual_dit.output.",
        "residual_dit.origin_projection.",
    )
    assert trainable
    assert all(name.startswith(allowed) for name in trainable)
    assert not any(name.startswith("residual_dit.amplitude_head") for name in trainable)
    assert not any(name.startswith("residual_dit.optical_bridge_") for name in trainable)
    assert not any(parameter.requires_grad for parameter in tiny_model.codec.parameters())
    assert not any(parameter.requires_grad for parameter in tiny_model.encoder.parameters())


def test_id_bridge_assignments_cover_directions_across_steps_and_ranks(
    tiny_model: SentinelV3,
) -> None:
    objective = JointObjective(tiny_model)
    objective.set_progress(0, 10)
    assert objective._id_bridge_assignments(1, torch.device("cpu")).tolist() == [0]
    assert objective._id_bridge_assignments(1, torch.device("cpu"), rank=1).tolist() == [1]

    objective.set_progress(1, 10)
    assert objective._id_bridge_assignments(1, torch.device("cpu")).tolist() == [1]
    assert objective._id_bridge_assignments(1, torch.device("cpu"), rank=1).tolist() == [0]

    for step, rank in ((0, 0), (0, 1), (1, 0), (1, 1)):
        objective.set_progress(step, 10)
        tasks = objective._id_bridge_assignments(2, torch.device("cpu"), rank=rank)
        expected = (torch.arange(2) + step + rank) % 2
        torch.testing.assert_close(tasks, expected)
        torch.testing.assert_close(torch.bincount(tasks, minlength=2), torch.ones(2, dtype=torch.long))


def test_id_bridge_start_isolates_distribution_gradients() -> None:
    mu = torch.randn(2, 4, 8, 8, requires_grad=True)
    log_sigma = torch.randn_like(mu, requires_grad=True)
    reliability_logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    z0, _, _ = JointObjective._id_bridge_start(
        mu, log_sigma, reliability_logits, 0.35, torch.randn_like(mu)
    )
    z0.square().mean().backward()
    assert mu.grad is not None and int(torch.count_nonzero(mu.grad)) > 0
    assert log_sigma.grad is None
    assert reliability_logits.grad is None

    logits = torch.zeros(2, 3, 8, 8, requires_grad=True)
    F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
    assert logits.grad is not None and int(torch.count_nonzero(logits.grad)) > 0

    calibrated_log_sigma = torch.zeros(2, 4, 8, 8, requires_grad=True)
    F.smooth_l1_loss(torch.sigmoid(calibrated_log_sigma), torch.zeros_like(calibrated_log_sigma)).backward()
    assert calibrated_log_sigma.grad is not None
    assert int(torch.count_nonzero(calibrated_log_sigma.grad)) > 0


def test_id_bridge_optimizer_updates_origin_and_checkpoint_ema(tiny_model: SentinelV3) -> None:
    _set_trainable(tiny_model, "id_bridge")
    optimizer = _optimizer(
        tiny_model,
        {"learning_rate": 1e-2, "encoder_learning_rate": 1e-3, "weight_decay": 0.0},
        "id_bridge",
        torch.device("cpu"),
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    trainable_ids = {id(parameter) for parameter in tiny_model.parameters() if parameter.requires_grad}
    assert optimizer_ids == trainable_ids
    assert {id(parameter) for parameter in tiny_model.id_bridge_origin.parameters()} <= optimizer_ids
    assert {id(parameter) for parameter in tiny_model.residual_dit.origin_projection.parameters()} <= optimizer_ids
    assert not any(
        id(parameter) in optimizer_ids
        for parameter in tiny_model.residual_dit.amplitude_head.parameters()
    )
    assert not any(
        id(parameter) in optimizer_ids
        for parameter in tiny_model.residual_dit.optical_bridge_anchor.parameters()
    )

    pyramid = tiny_model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    head = tiny_model.id_bridge_origin.output_heads["optical"][-1]
    weight_before = head.weight.detach().clone()
    bias_before = head.bias.detach().clone()
    ema = EMA(tiny_model, decay=0.5)
    mu, log_sigma, reliability_logits = tiny_model.predict_id_bridge_origin(
        pyramid, torch.rand(1, 3, 32, 32), SENTINEL2
    )
    (mu.mean() + log_sigma.mean() + reliability_logits.mean()).backward()
    optimizer.step()
    ema.update(tiny_model)
    assert not torch.equal(head.weight, weight_before)
    assert not torch.equal(head.bias, bias_before)

    scheduler = _scheduler(optimizer, warmup=0, maximum=1)
    payload = _checkpoint_payload(
        model=tiny_model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        stage="id_bridge",
        step=1,
        rank_states=[],
        config={},
        validation_protocol_hash="test",
        best_metrics={},
        quality_gates={},
    )
    weight_name = "id_bridge_origin.output_heads.optical.2.weight"
    bias_name = "id_bridge_origin.output_heads.optical.2.bias"
    assert int(torch.count_nonzero(payload["model"][weight_name])) > 0  # type: ignore[index]
    assert int(torch.count_nonzero(payload["model"][bias_name])) > 0  # type: ignore[index]
    assert int(torch.count_nonzero(payload["ema"]["state"][weight_name])) > 0  # type: ignore[index]
    assert int(torch.count_nonzero(payload["ema"]["state"][bias_name])) > 0  # type: ignore[index]
    assert payload["residual_state"] == {  # type: ignore[index]
        "kind": "codec",
        "channels": tiny_model.config.codec_latent_channels,
        "optical_scale": 0.03,
        "sar_scale": 4.0,
        "anchor_origin": False,
        "optical_innovation_scale": 1.0,
        "sar_innovation_scale": 1.0,
    }


def test_haar_id_bridge_optimizer_and_checkpoint_metadata() -> None:
    model = _haar_model(anchor_origin=True, optical_innovation_scale=0.0)
    _set_trainable(model, "id_bridge")
    optimizer = _optimizer(
        model,
        {"learning_rate": 1e-3, "encoder_learning_rate": 1e-3, "weight_decay": 0.0},
        "id_bridge",
        torch.device("cpu"),
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert optimizer_ids == trainable_ids
    assert not any(id(parameter) in optimizer_ids for parameter in model.codec.parameters())
    payload = _checkpoint_payload(
        model=model,
        ema=EMA(model, decay=0.5),
        optimizer=optimizer,
        scheduler=_scheduler(optimizer, warmup=0, maximum=1),
        stage="id_bridge",
        step=0,
        rank_states=[],
        config={},
        validation_protocol_hash="test",
        best_metrics={},
        quality_gates={},
    )
    assert payload["residual_state"] == {  # type: ignore[index]
        "kind": "haar_packet",
        "channels": 48,
        "optical_scale": 0.03,
        "sar_scale": 4.0,
        "anchor_origin": True,
        "optical_innovation_scale": 0.0,
        "sar_innovation_scale": 1.0,
    }


def test_id_bridge_long_gap_has_exact_zero_residual_gradient(tiny_model: SentinelV3) -> None:
    _set_trainable(tiny_model, "id_bridge")
    objective = JointObjective(
        tiny_model,
        [0.5, 0.5],
        flow_rollout_every=1,
        flow_visual_perceptual_weight=0.0,
    )
    loss, _ = objective(_batch(delta_days=2), "id_bridge")
    loss.backward()
    assert float(loss.detach()) == 0.0
    gradients = [
        parameter.grad for parameter in tiny_model.parameters() if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) == 0.0


def test_haar_id_bridge_long_gap_has_exact_zero_residual_gradient() -> None:
    model = _haar_model()
    _set_trainable(model, "id_bridge")
    objective = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=1,
        flow_visual_perceptual_weight=0.0,
    )
    loss, _ = objective(_batch(delta_days=2), "id_bridge")
    loss.backward()
    assert float(loss.detach()) == 0.0
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) == 0.0


def test_id_bridge_objective_updates_origin_and_generic_flow(tiny_model: SentinelV3) -> None:
    _set_trainable(tiny_model, "id_bridge")
    objective = JointObjective(
        tiny_model,
        [0.5, 0.5],
        flow_rollout_every=1,
        flow_visual_perceptual_weight=0.0,
    )
    objective.set_progress(1, 10)
    loss, metrics = objective(_batch(), "id_bridge")
    loss.backward()
    assert torch.isfinite(loss)
    assert "sar2opt/sigma_calibration" in metrics
    assert "opt2sar/rollout_speckle_scale" in metrics
    assert tiny_model.id_bridge_origin.output_heads["optical"][-1].weight.grad is not None
    assert tiny_model.residual_dit.output[-1].weight.grad is not None


def test_haar_id_bridge_objective_updates_both_origin_heads_and_flow() -> None:
    model = _haar_model()
    _set_trainable(model, "id_bridge")
    objective = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=1,
        flow_visual_perceptual_weight=0.0,
    )
    objective.set_progress(1, 10)
    loss, metrics = objective(_batch(), "id_bridge")
    loss.backward()
    assert torch.isfinite(loss)
    assert "sar2opt/sigma_calibration" in metrics
    assert "opt2sar/rollout_speckle_scale" in metrics
    for modality in ("optical", "sar"):
        gradient = model.id_bridge_origin.output_heads[modality][-1].weight.grad
        assert gradient is not None and int(torch.count_nonzero(gradient)) > 0
    assert model.residual_dit.output[-1].weight.grad is not None
    assert int(torch.count_nonzero(model.residual_dit.output[-1].weight.grad)) > 0


def test_visual_translation_routes_to_id_bridge_without_detail(tiny_model: SentinelV3) -> None:
    tiny_model.config.id_bridge_enabled = True
    observation = Observation(
        torch.randn(2, 32, 32), SENTINEL1, dt.date(2020, 1, 2), orbit="ascending"
    )
    result = translate(
        tiny_model.eval(), [observation], TargetRequest(SENTINEL2), "visual", seed=5
    )
    assert result.deterministic_detail is not None
    torch.testing.assert_close(result.deterministic_detail, torch.zeros_like(result.deterministic_detail))
    assert result.stochastic_residual is not None
    assert result.residual_amplitude is None


def test_observable_anchor_translation_composes_pixel_detail_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(anchor_origin=True, optical_innovation_scale=0.0).eval()

    def deterministic_detail(
        _pyramid: object,
        source: object,
        target: object,
        _output_size: tuple[int, int],
        *,
        base: torch.Tensor | None,
    ) -> torch.Tensor:
        assert source == SENTINEL1 and target == SENTINEL2 and base is not None
        return torch.full_like(base, 0.02)

    monkeypatch.setattr(model, "deterministic_detail", deterministic_detail)
    observation = Observation(
        torch.randn(2, 32, 32), SENTINEL1, dt.date(2020, 1, 2), orbit="ascending"
    )
    result = translate(model, [observation], TargetRequest(SENTINEL2), "visual", seed=5)
    assert result.deterministic_detail is not None
    assert result.stochastic_residual is not None
    assert int(torch.count_nonzero(result.deterministic_detail)) > 0
    zeros = torch.zeros_like(result.stochastic_residual)
    torch.testing.assert_close(result.stochastic_residual, zeros, atol=1e-7, rtol=0.0)
    base = result.physical[:, [2, 1, 0]]
    expected = model.compose_visual(base, result.deterministic_detail, zeros, "optical")
    assert isinstance(expected, torch.Tensor)
    torch.testing.assert_close(result.samples[0], expected)
