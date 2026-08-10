from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import sentinel_v3.training as training_module
from sentinel_v3.api import Observation, TargetRequest, translate
from sentinel_v3.config import load_config, validate_config
from sentinel_v3.losses import (
    anchor_gain_target,
    cross_modal_identifiability_target,
    frequency_bands,
    haar_dwt2,
    haar_idwt2,
    haar_packet_dwt2,
    haar_packet_idwt2,
    high_frequency_loss,
    highpass,
    phase_alignment_loss,
    phase_identifiability_target,
    phase_transport_gain_target,
    phase_transport_signed_coefficient_target,
    signed_phase_alignment_loss,
)
from sentinel_v3.model import ModelConfig, ObservablePhaseTransportHead, SentinelV3
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
    utility: bool = False,
    phase: bool = False,
    optical_only: bool = False,
    optical_innovation_scale: float = 1.0,
    sar_innovation_scale: float = 1.0,
    optical_innovation_band_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
    optical_correction_scale: float = 1.0,
    phase_transport: bool = False,
    phase_transport_gain_caps: tuple[float, float, float] = (0.5, 0.25, 0.1),
    phase_transport_carrier_gain_caps: tuple[float, float, float] = (0.5, 0.25, 0.1),
    phase_transport_offset_caps_px: tuple[float, float, float] = (0.5, 0.5, 0.5),
    phase_transport_initial_gate: float = 0.02,
    phase_transport_null_calibrated: bool = False,
    phase_transport_null_quantile: float = 0.75,
    phase_transport_support_epsilon: float = 0.01,
    phase_transport_carrier_mode: str = "physical_gain",
    phase_transport_carrier_support_mode: str = "continuous",
    phase_transport_carrier_basis_trainable: bool = True,
    phase_transport_detail_utility_enabled: bool = False,
    phase_transport_detail_scale_cap: float = 2.0,
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
            id_bridge_anchor_utility=utility,
            id_bridge_phase_identifiability=phase,
            id_bridge_optical_only=optical_only,
            id_bridge_optical_innovation_scale=optical_innovation_scale,
            id_bridge_sar_innovation_scale=sar_innovation_scale,
            id_bridge_optical_innovation_band_scales=optical_innovation_band_scales,
            id_bridge_optical_correction_scale=optical_correction_scale,
            phase_transport_enabled=phase_transport,
            phase_transport_gain_caps=phase_transport_gain_caps,
            phase_transport_carrier_gain_caps=phase_transport_carrier_gain_caps,
            phase_transport_offset_caps_px=phase_transport_offset_caps_px,
            phase_transport_initial_gate=phase_transport_initial_gate,
            phase_transport_null_calibrated=phase_transport_null_calibrated,
            phase_transport_null_quantile=phase_transport_null_quantile,
            phase_transport_support_epsilon=phase_transport_support_epsilon,
            phase_transport_carrier_mode=phase_transport_carrier_mode,
            phase_transport_carrier_support_mode=phase_transport_carrier_support_mode,
            phase_transport_carrier_basis_trainable=phase_transport_carrier_basis_trainable,
            phase_transport_detail_utility_enabled=phase_transport_detail_utility_enabled,
            phase_transport_detail_scale_cap=phase_transport_detail_scale_cap,
        )
    )


def _utility_codec_model() -> SentinelV3:
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
            id_bridge_enabled=False,
            id_bridge_state="codec",
            id_bridge_anchor_utility=True,
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
    with pytest.raises(ValueError, match="non-negative"):
        ModelConfig(id_bridge_optical_mid_basis_scale=-0.01)
    with pytest.raises(ValueError, match="non-negative"):
        ModelConfig(id_bridge_optical_coarse_basis_scale=float("nan"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ModelConfig(id_bridge_optical_correction_scale=1.01)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ModelConfig(id_bridge_sar_correction_scale=-0.01)


def test_id_bridge_utility_config_ranges_are_validated() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_id_bridge_haar.yaml")
    config["model"]["id_bridge_optical_mid_basis_scale"] = -0.01
    with pytest.raises(ValueError, match="non-negative"):
        validate_config(config)
    config["model"]["id_bridge_optical_mid_basis_scale"] = 0.15
    config["model"]["id_bridge_sar_correction_scale"] = float("inf")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_config(config)


def test_phase_bridge_config_ranges_and_configs() -> None:
    with pytest.raises(ValueError, match="three values"):
        ModelConfig(id_bridge_optical_innovation_band_scales=(1.0, 1.0))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ModelConfig(id_bridge_optical_innovation_band_scales=(1.0, -0.1, 1.0))
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_id_bridge_phase.yaml")
    config["train"]["id_bridge_antithetic_weight"] = -0.1
    with pytest.raises(ValueError, match="antithetic"):
        validate_config(config)

    root = Path(__file__).parents[1] / "configs"
    names = (
        "smoke_id_bridge_phase.yaml",
        "id_bridge_phase_connectivity.yaml",
        "id_bridge_phase_pilot.yaml",
    )
    expected_steps = (2, 100, 1000)
    for name, steps in zip(names, expected_steps, strict=True):
        phase = load_config(root / name)
        assert phase["train"]["stage"] == "id_bridge"
        assert phase["train"]["max_steps"] == steps
        assert phase["train"]["init_use_ema"] is True
        assert phase["train"]["ema_decay"] == pytest.approx(0.99)
        assert phase["train"]["id_bridge_antithetic_weight"] == pytest.approx(0.05)
        assert phase["train"]["flow_rollout_steps"] == 1
        assert phase["model"]["id_bridge_phase_identifiability"] is True
        assert phase["model"]["id_bridge_optical_only"] is True
        assert phase["model"]["id_bridge_anchor_origin"] is True
        assert phase["model"]["id_bridge_anchor_utility"] is False
        assert phase["model"]["id_bridge_optical_innovation_band_scales"] == [0.0, 0.0, 0.0]
    assert load_config(root / names[0])["validation"]["enabled"] is False
    pilot = load_config(root / names[2])
    assert pilot["train"]["batch_size"] == 1
    assert pilot["train"]["gradient_accumulation"] == 2


def test_phase_transport_config_ranges_and_configs() -> None:
    with pytest.raises(ValueError, match="three values"):
        ModelConfig(phase_transport_gain_caps=(0.5, 0.25))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ModelConfig(phase_transport_gain_caps=(0.5, -0.25, 0.1))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ModelConfig(phase_transport_offset_caps_px=(0.5, 0.5, float("inf")))
    with pytest.raises(ValueError, match="positive"):
        ModelConfig(phase_transport_hidden=0)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        ModelConfig(phase_transport_initial_gate=0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        ModelConfig(phase_transport_initial_gate=float("inf"))
    with pytest.raises(TypeError, match="null_calibrated"):
        ModelConfig(phase_transport_null_calibrated=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="null_quantile"):
        ModelConfig(phase_transport_null_quantile=1.0)
    with pytest.raises(ValueError, match="support_epsilon"):
        ModelConfig(phase_transport_support_epsilon=0.0)
    with pytest.raises(ValueError, match="carrier_mode"):
        ModelConfig(phase_transport_carrier_mode="unknown")
    assert ModelConfig(phase_transport_carrier_gain_caps=(0.5, 0.25, 0.0)).phase_transport_carrier_gain_caps == (
        0.5,
        0.25,
        0.0,
    )
    with pytest.raises(ValueError, match="carrier_gain_caps"):
        ModelConfig(phase_transport_carrier_gain_caps=(0.5, -0.25, 0.1))
    with pytest.raises(ValueError, match="carrier_gain_caps"):
        ModelConfig(phase_transport_carrier_gain_caps=(0.5, 0.25, 1.01))
    with pytest.raises(ValueError, match="carrier_support_mode"):
        ModelConfig(phase_transport_carrier_support_mode="unknown")
    assert ModelConfig().phase_transport_carrier_basis_trainable is True
    with pytest.raises(TypeError, match="carrier_basis_trainable"):
        ModelConfig(phase_transport_carrier_basis_trainable=1)  # type: ignore[arg-type]
    assert ModelConfig().phase_transport_detail_utility_enabled is False
    assert ModelConfig().phase_transport_detail_scale_cap == pytest.approx(2.0)
    with pytest.raises(TypeError, match="detail_utility_enabled"):
        ModelConfig(phase_transport_detail_utility_enabled=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="detail_scale_cap"):
        ModelConfig(phase_transport_detail_scale_cap=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="detail_scale_cap"):
        ModelConfig(phase_transport_detail_scale_cap=float("inf"))
    with pytest.raises(ValueError, match="detail_scale_cap"):
        ModelConfig(
            phase_transport_detail_utility_enabled=True,
            phase_transport_detail_scale_cap=1.0,
        )
    with pytest.raises(ValueError, match="detail_scale_cap"):
        ModelConfig(
            phase_transport_detail_utility_enabled=True,
            phase_transport_detail_scale_cap=4.01,
        )
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["train"]["phase_transport_hf_weight"] = -0.01
    with pytest.raises(ValueError, match="phase_transport_hf_weight"):
        validate_config(config)
    config["train"]["phase_transport_hf_weight"] = float("inf")
    with pytest.raises(ValueError, match="phase_transport_hf_weight"):
        validate_config(config)
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["train"]["phase_transport_utility_weight"] = -0.01
    with pytest.raises(ValueError, match="phase_transport_utility_weight"):
        validate_config(config)
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["model"]["phase_transport_initial_gate"] = 1.0
    with pytest.raises(ValueError, match="phase_transport_initial_gate"):
        validate_config(config)
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["model"]["phase_transport_null_quantile"] = 0.0
    with pytest.raises(ValueError, match="phase_transport_null_quantile"):
        validate_config(config)
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["model"]["phase_transport_carrier_mode"] = "unknown"
    with pytest.raises(ValueError, match="phase_transport_carrier_mode"):
        validate_config(config)
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["model"]["phase_transport_carrier_gain_caps"] = [0.5, 0.25, -0.1]
    with pytest.raises(ValueError, match="phase_transport_carrier_gain_caps"):
        validate_config(config)
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["model"]["phase_transport_carrier_support_mode"] = "unknown"
    with pytest.raises(ValueError, match="phase_transport_carrier_support_mode"):
        validate_config(config)
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["model"]["phase_transport_carrier_basis_trainable"] = 1
    with pytest.raises(TypeError, match="phase_transport_carrier_basis_trainable"):
        validate_config(config)
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["model"]["phase_transport_detail_utility_enabled"] = 1
    with pytest.raises(TypeError, match="phase_transport_detail_utility_enabled"):
        validate_config(config)
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["model"]["phase_transport_detail_utility_enabled"] = True
    config["model"]["phase_transport_detail_scale_cap"] = 1.0
    with pytest.raises(ValueError, match="phase_transport_detail_scale_cap"):
        validate_config(config)
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["train"]["phase_transport_detail_utility_kernel"] = 4
    with pytest.raises(ValueError, match="phase_transport_detail_utility_kernel"):
        validate_config(config)
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["train"]["phase_transport_detail_utility_kernel"] = True
    with pytest.raises(TypeError, match="phase_transport_detail_utility_kernel"):
        validate_config(config)
    legacy_config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    legacy_config["model"].pop("phase_transport_carrier_gain_caps")
    legacy_config["model"].pop("phase_transport_carrier_support_mode")
    legacy_config["model"].pop("phase_transport_carrier_basis_trainable")
    legacy_config["model"].pop("phase_transport_detail_utility_enabled")
    legacy_config["model"].pop("phase_transport_detail_scale_cap")
    validate_config(legacy_config)
    assert legacy_config["model"]["phase_transport_carrier_gain_caps"] == [0.5, 0.25, 0.1]
    assert legacy_config["model"]["phase_transport_carrier_support_mode"] == "continuous"
    assert legacy_config["model"]["phase_transport_carrier_basis_trainable"] is True
    assert legacy_config["model"]["phase_transport_detail_utility_enabled"] is False
    assert legacy_config["model"]["phase_transport_detail_scale_cap"] == pytest.approx(2.0)
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke_phase_transport.yaml")
    config["train"]["phase_transport_signed_alignment_weight"] = -0.01
    with pytest.raises(ValueError, match="phase_transport_signed_alignment_weight"):
        validate_config(config)

    root = Path(__file__).parents[1] / "configs"
    names = (
        "smoke_phase_transport.yaml",
        "phase_transport_connectivity.yaml",
        "phase_transport_pilot.yaml",
    )
    expected_steps = (2, 100, 1000)
    for name, steps in zip(names, expected_steps, strict=True):
        config = load_config(root / name)
        model = config["model"]
        train = config["train"]
        assert train["stage"] == "phase_transport"
        assert train["max_steps"] == steps
        assert train["batch_size"] == 1
        assert train["gradient_accumulation"] == 2
        assert train["flow_rollout_every"] == 4
        assert train["init_use_ema"] is True
        assert train["phase_transport_hf_weight"] == pytest.approx(0.05)
        assert train["phase_transport_utility_weight"] == pytest.approx(0.10)
        assert train["flow_visual_perceptual_weight"] == pytest.approx(0.10)
        assert model["phase_transport_enabled"] is True
        assert model["phase_transport_gain_caps"] == [0.5, 0.25, 0.1]
        assert model["phase_transport_offset_caps_px"] == [0.5, 0.5, 0.5]
        assert model["phase_transport_initial_gate"] == pytest.approx(0.02)
        assert model["id_bridge_phase_identifiability"] is True
        assert model["id_bridge_optical_only"] is True
        assert model["id_bridge_anchor_origin"] is True
        assert model["id_bridge_anchor_utility"] is False
        assert model["id_bridge_optical_correction_scale"] == 0.0
        assert model["id_bridge_optical_innovation_band_scales"] == [0.0, 0.0, 0.0]
        assert config["paths"]["output"].endswith("_v3")
        assert config["paths"]["reports"].endswith("_v3")
    assert load_config(root / names[0])["validation"]["enabled"] is False
    pilot = load_config(root / names[-1])
    assert pilot["train"]["validate_every"] == 250
    assert pilot["train"]["save_every"] == 250
    assert pilot["validation"]["full_steps"] == []


def test_null_calibrated_phase_transport_configs() -> None:
    root = Path(__file__).parents[1] / "configs"
    names = (
        "smoke_phase_transport_nc.yaml",
        "phase_transport_nc_connectivity.yaml",
        "phase_transport_nc_pilot.yaml",
    )
    expected_steps = (2, 100, 1000)
    for name, steps in zip(names, expected_steps, strict=True):
        config = load_config(root / name)
        model = config["model"]
        train = config["train"]
        assert train["stage"] == "phase_transport"
        assert train["max_steps"] == steps
        assert train["phase_transport_utility_weight"] == pytest.approx(0.25)
        assert train["phase_transport_hf_weight"] == pytest.approx(0.05)
        assert train["flow_visual_perceptual_weight"] == pytest.approx(0.10)
        assert train["find_unused_parameters"] is False
        assert model["phase_transport_null_calibrated"] is True
        assert model["phase_transport_null_quantile"] == pytest.approx(0.75)
        assert model["phase_transport_support_epsilon"] == pytest.approx(0.01)
        assert config["paths"]["output"].endswith("_v4")
        assert config["paths"]["reports"].endswith("_v4")
    assert load_config(root / names[0])["validation"]["enabled"] is False
    assert load_config(root / names[-1])["validation"]["full_steps"] == []


def test_canonical_ncopc_phase_transport_configs() -> None:
    root = Path(__file__).parents[1] / "configs"
    names = (
        "canonical_2017_2024_phase_transport_ncopc_connectivity.yaml",
        "canonical_2017_2024_phase_transport_ncopc_pilot.yaml",
    )
    expected_steps = (100, 1000)
    expected_validation = (100, 250)
    for name, steps, validation_every in zip(
        names, expected_steps, expected_validation, strict=True
    ):
        config = load_config(root / name)
        model = config["model"]
        train = config["train"]
        assert train["stage"] == "phase_transport"
        assert train["max_steps"] == steps
        assert train["batch_size"] == 1
        assert train["gradient_accumulation"] == 2
        assert train["validate_every"] == validation_every
        assert train["flow_rollout_steps"] == 1
        assert train["phase_transport_signed_alignment_weight"] == pytest.approx(0.05)
        assert model["phase_transport_null_calibrated"] is True
        assert model["phase_transport_carrier_mode"] == "orthogonal_source"
        assert "2017_2024" in config["paths"]["train_shards"]
        assert "ncopc" in config["paths"]["output"]
        assert "ncopc" in config["paths"]["reports"]
        validate_config(config)
    assert load_config(root / names[-1])["validation"]["full_steps"] == []


def test_canonical_ncopc_bnes_phase_transport_configs() -> None:
    root = Path(__file__).parents[1] / "configs"
    names = (
        "canonical_2017_2024_phase_transport_ncopc_bnes_connectivity.yaml",
        "canonical_2017_2024_phase_transport_ncopc_bnes_pilot.yaml",
    )
    expected_steps = (100, 1000)
    expected_validation = (100, 250)
    for name, steps, validation_every in zip(
        names, expected_steps, expected_validation, strict=True
    ):
        config = load_config(root / name)
        model = config["model"]
        train = config["train"]
        assert train["stage"] == "phase_transport"
        assert train["max_steps"] == steps
        assert train["batch_size"] == 1
        assert train["gradient_accumulation"] == 2
        assert train["validate_every"] == validation_every
        assert train["flow_rollout_steps"] == 1
        assert train["phase_transport_signed_alignment_weight"] == pytest.approx(0.05)
        assert model["phase_transport_carrier_mode"] == "orthogonal_source"
        assert model["phase_transport_carrier_support_mode"] == "binary_exceedance"
        assert model["phase_transport_carrier_gain_caps"] == [0.125, 0.0625, 0.025]
        assert "ncopc_bnes" in config["paths"]["output"]
        assert "ncopc_bnes" in config["paths"]["reports"]
        validate_config(config)
    assert load_config(root / names[-1])["validation"]["full_steps"] == []


def test_canonical_ncopc_stationary_phase_transport_configs() -> None:
    root = Path(__file__).parents[1] / "configs"
    names = (
        "canonical_2017_2024_phase_transport_ncopc_stationary_connectivity.yaml",
        "canonical_2017_2024_phase_transport_ncopc_stationary_pilot.yaml",
    )
    expected_steps = (100, 1000)
    expected_validation = (100, 250)
    for name, steps, validation_every in zip(
        names, expected_steps, expected_validation, strict=True
    ):
        config = load_config(root / name)
        model = config["model"]
        train = config["train"]
        assert train["stage"] == "phase_transport"
        assert train["max_steps"] == steps
        assert train["batch_size"] == 1
        assert train["gradient_accumulation"] == 2
        assert train["validate_every"] == validation_every
        assert train["flow_rollout_steps"] == 1
        assert train["phase_transport_utility_weight"] == pytest.approx(1.0)
        assert train["phase_transport_signed_alignment_weight"] == pytest.approx(0.0)
        assert model["phase_transport_carrier_mode"] == "orthogonal_source"
        assert model["phase_transport_carrier_support_mode"] == "binary_exceedance"
        assert model["phase_transport_carrier_gain_caps"] == [1.0, 0.5, 0.2]
        assert model["phase_transport_carrier_basis_trainable"] is False
        assert "ncopc_stationary" in config["paths"]["output"]
        assert "ncopc_stationary" in config["paths"]["reports"]
        validate_config(config)
    assert load_config(root / names[-1])["validation"]["full_steps"] == []


def test_canonical_spuf_phase_transport_configs() -> None:
    root = Path(__file__).parents[1] / "configs"
    names = (
        "canonical_2017_2024_phase_transport_spuf_connectivity.yaml",
        "canonical_2017_2024_phase_transport_spuf_pilot.yaml",
    )
    expected_steps = (100, 1000)
    expected_validation = (100, 250)
    for name, steps, validation_every in zip(
        names, expected_steps, expected_validation, strict=True
    ):
        config = load_config(root / name)
        model = config["model"]
        train = config["train"]
        assert train["stage"] == "phase_transport"
        assert train["max_steps"] == steps
        assert train["batch_size"] == 1
        assert train["gradient_accumulation"] == 2
        assert train["validate_every"] == validation_every
        assert train["flow_rollout_steps"] == 1
        assert train["init_use_ema"] is True
        assert train["phase_transport_utility_weight"] == pytest.approx(1.0)
        assert train["phase_transport_detail_utility_kernel"] == 5
        assert model["phase_transport_carrier_mode"] == "physical_gain"
        assert model["phase_transport_detail_utility_enabled"] is True
        assert model["phase_transport_detail_scale_cap"] == pytest.approx(2.0)
        assert "spuf" in config["paths"]["output"]
        assert "spuf" in config["paths"]["reports"]
        validate_config(config)
    assert load_config(root / names[-1])["validation"]["full_steps"] == []


def test_phase_identifiability_target_is_oriented_masked_and_scale_invariant() -> None:
    height = width = 32
    horizontal = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width).expand(
        1, 1, height, width
    )
    vertical = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1).expand(
        1, 1, height, width
    )
    source = torch.cat((horizontal, horizontal), dim=1)
    parallel = horizontal.expand(1, 3, height, width)
    perpendicular = vertical.expand(1, 3, height, width)
    valid = torch.ones(1, 1, height, width)
    aligned = phase_identifiability_target(source, (parallel, parallel, parallel), valid)
    orthogonal = phase_identifiability_target(
        source, (perpendicular, perpendicular, perpendicular), valid
    )
    inverted = phase_identifiability_target(
        source * 100.0,
        tuple(-0.01 * band for band in (parallel, parallel, parallel)),
        valid,
    )
    assert aligned.shape == (1, 3, 8, 8)
    assert bool(torch.isfinite(aligned).all())
    assert float(aligned.min()) >= 0.0 and float(aligned.max()) <= 1.0
    assert float(aligned.mean()) > float(orthogonal.mean()) + 0.5
    torch.testing.assert_close(aligned, inverted, atol=1e-4, rtol=1e-4)

    partial = valid.clone()
    partial[..., :4, :4] = 0.0
    masked = phase_identifiability_target(source, (parallel, parallel, parallel), partial)
    assert int(torch.count_nonzero(masked[..., 0, 0])) == 0
    constant = phase_identifiability_target(
        torch.ones_like(source),
        (torch.ones_like(parallel),) * 3,
        valid,
    )
    assert bool(torch.isfinite(constant).all())
    assert int(torch.count_nonzero(constant)) == 0


def test_phase_band_fields_map_to_legal_haar_packet_slots() -> None:
    model = _haar_model(
        phase=True,
        optical_innovation_band_scales=(0.25, 0.5, 0.75),
    )
    fields = torch.stack(
        (
            torch.full((1, 8, 8), 0.1),
            torch.full((1, 8, 8), 0.2),
            torch.full((1, 8, 8), 0.3),
        ),
        dim=1,
    )
    optical = model.id_bridge_band_fields_to_state(fields, SENTINEL2)
    assert optical.shape == (1, 48, 8, 8)
    fine_indices = [first * 4 + second for first in range(1, 4) for second in range(1, 4)]
    mid_indices = [first * 4 for first in range(1, 4)]
    coarse_indices = [second for second in range(1, 4)]
    for channel in range(3):
        offset = channel * 16
        assert int(torch.count_nonzero(optical[:, offset])) == 0
        torch.testing.assert_close(
            optical[:, [offset + value for value in fine_indices]],
            fields[:, :1].expand(-1, len(fine_indices), -1, -1),
        )
        torch.testing.assert_close(
            optical[:, [offset + value for value in mid_indices]],
            fields[:, 1:2].expand(-1, len(mid_indices), -1, -1),
        )
        torch.testing.assert_close(
            optical[:, [offset + value for value in coarse_indices]],
            fields[:, 2:3].expand(-1, len(coarse_indices), -1, -1),
        )
    sar = model.id_bridge_band_fields_to_state(fields, SENTINEL1)
    assert int(torch.count_nonzero(sar[:, 32:])) == 0
    assert int(torch.count_nonzero(sar[:, [0, 16]])) == 0
    release = model.id_bridge_innovation_release_state(torch.zeros_like(fields), SENTINEL2)
    torch.testing.assert_close(release[:, 1:4], torch.full_like(release[:, 1:4], 0.75))
    torch.testing.assert_close(release[:, [4, 8, 12]], torch.full_like(release[:, [4, 8, 12]], 0.5))
    torch.testing.assert_close(
        release[:, fine_indices], torch.full_like(release[:, fine_indices], 0.25)
    )
    metadata = model.residual_state_metadata()
    assert metadata["phase_identifiability"] is True
    assert metadata["optical_only"] is False
    assert metadata["optical_innovation_band_scales"] == (0.25, 0.5, 0.75)


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


def test_haar_utility_configs_enable_observable_anchor_gains() -> None:
    root = Path(__file__).parents[1] / "configs"
    names = (
        "id_bridge_haar_utility.yaml",
        "id_bridge_haar_utility_connectivity.yaml",
        "smoke_id_bridge_haar_utility.yaml",
    )
    for name in names:
        config = load_config(root / name)
        model = config["model"]
        assert model["id_bridge_anchor_origin"] is True
        assert model["id_bridge_anchor_utility"] is True
        assert model["id_bridge_optical_innovation_scale"] == 0.0
        assert model["id_bridge_sar_innovation_scale"] == 1.0
        assert model["id_bridge_optical_correction_scale"] == 0.0
        assert model["id_bridge_sar_correction_scale"] == 1.0
        assert model["id_bridge_optical_mid_basis_scale"] == pytest.approx(0.15)
        assert model["id_bridge_optical_coarse_basis_scale"] == pytest.approx(0.05)
    assert load_config(root / names[0])["train"]["max_steps"] == 5000
    assert load_config(root / names[1])["train"]["max_steps"] == 100
    smoke = load_config(root / names[2])
    assert smoke["train"]["max_steps"] == 2
    assert smoke["validation"]["enabled"] is False


def test_id_utility_configs_pretrain_only_the_optical_anchor() -> None:
    root = Path(__file__).parents[1] / "configs"
    names = (
        "smoke_id_utility.yaml",
        "id_utility_connectivity.yaml",
        "id_utility_pilot.yaml",
    )
    for name in names:
        config = load_config(root / name)
        assert config["train"]["stage"] == "id_utility"
        assert config["train"]["init_use_ema"] is True
        assert config["train"]["find_unused_parameters"] is True
        assert config["model"]["id_bridge_enabled"] is False
        assert config["model"]["id_bridge_anchor_utility"] is True
        assert config["model"]["id_bridge_optical_mid_basis_scale"] == pytest.approx(0.15)
        assert config["model"]["id_bridge_optical_coarse_basis_scale"] == pytest.approx(0.05)
    assert load_config(root / names[0])["validation"]["enabled"] is False
    assert load_config(root / names[1])["train"]["max_steps"] == 100
    pilot = load_config(root / names[2])
    assert pilot["train"]["max_steps"] == 1000
    assert pilot["train"]["validate_every"] == 250
    assert pilot["train"]["full_validate_every"] == 5000


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


def test_phase_origin_uses_state_q_for_mu_sigma_release_and_preserves_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
    )
    correction = torch.ones(1, 48, 8, 8, requires_grad=True)
    log_sigma = torch.zeros_like(correction, requires_grad=True)
    logits = torch.zeros(1, 3, 8, 8, requires_grad=True)
    protected_anchor = torch.randn(1, 3, 32, 32)
    pyramid = model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    physical = torch.rand(1, 3, 32, 32)

    monkeypatch.setattr(
        model.id_bridge_origin,
        "forward",
        lambda *_args: (correction, log_sigma, logits),
    )
    monkeypatch.setattr(
        model,
        "id_bridge_anchor_detail",
        lambda *_args, **_kwargs: protected_anchor,
    )
    mu, raw, anchor, returned_sigma, returned_logits = model.predict_id_bridge_origin_components(
        pyramid, physical, SENTINEL2
    )
    q_state = model.id_bridge_q_state(logits, SENTINEL2)
    torch.testing.assert_close(mu, q_state * correction)
    torch.testing.assert_close(raw, correction)
    torch.testing.assert_close(anchor, protected_anchor)
    torch.testing.assert_close(returned_sigma, log_sigma)
    torch.testing.assert_close(returned_logits, logits)
    mu.square().mean().backward(retain_graph=True)
    assert correction.grad is not None and int(torch.count_nonzero(correction.grad)) > 0
    assert logits.grad is None

    _z0, q, sigma = JointObjective._id_bridge_start(
        mu.detach(),
        log_sigma,
        logits,
        0.35,
        torch.ones_like(mu),
        q_state=q_state.detach(),
    )
    torch.testing.assert_close(q, q_state.detach())
    torch.testing.assert_close(sigma, 0.175 * (1.0 - q_state.detach()))
    gated = model.gate_id_bridge_innovation(
        torch.randn_like(mu), mu.detach(), logits, SENTINEL2, q_state=q_state.detach()
    )
    torch.testing.assert_close(gated, mu.detach())
    torch.testing.assert_close(
        model.id_bridge_anchor_detail(pyramid, physical, SENTINEL2, reliability_logits=logits),
        protected_anchor,
    )
    F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
    assert logits.grad is not None and int(torch.count_nonzero(logits.grad)) > 0


def test_phase_zero_release_keeps_mu_and_removes_seeded_innovation() -> None:
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
    ).eval()
    with torch.no_grad():
        model.id_bridge_origin.output_heads["optical"][-1].bias[:48].fill_(1.0)
    valid = torch.ones(1, 1, 32, 32)
    pyramid = model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    base = torch.rand(1, 3, 32, 32)
    first = model.sample_id_bridge_residual(pyramid, base, SENTINEL2, seed=11)
    second = model.sample_id_bridge_residual(pyramid, base, SENTINEL2, seed=12)
    assert isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    assert int(torch.count_nonzero(first)) > 0


def test_phase_transport_and_anchor_conditions_are_zero_init_and_trainable() -> None:
    model = _haar_model(anchor_origin=True, phase=True).eval()
    assert int(torch.count_nonzero(model.residual_dit.id_bridge_field_projection.weight)) == 0
    assert int(torch.count_nonzero(model.residual_dit.id_bridge_anchor_projection.weight)) == 0
    valid = torch.ones(1, 1, 32, 32)
    pyramid = model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    latent = torch.randn(1, 48, 8, 8)
    descriptors = model.descriptors(SENTINEL2.channels[:3], latent.device)
    baseline = model.residual_dit(
        latent, torch.zeros(1), pyramid, descriptors, origin_latent=torch.zeros_like(latent)
    )
    conditioned = model.residual_dit(
        latent,
        torch.zeros(1),
        pyramid,
        descriptors,
        origin_latent=torch.zeros_like(latent),
        transport_field=torch.randn(1, 4, 8, 8),
        id_bridge_anchor_state=torch.randn_like(latent),
    )
    torch.testing.assert_close(conditioned, baseline, atol=0.0, rtol=0.0)

    model.train()
    _set_trainable(model, "id_bridge")
    with torch.no_grad():
        model.residual_dit.output[-1].weight.normal_(std=0.01)
    velocity = model.flow_velocity(
        latent,
        torch.zeros(1),
        pyramid,
        SENTINEL2,
        3,
        origin_latent=torch.zeros_like(latent),
        transport_field=torch.randn(1, 4, 8, 8),
        id_bridge_anchor_state=torch.randn_like(latent),
        use_optical_bridge=False,
    )
    velocity.square().mean().backward()
    gradient = model.residual_dit.id_bridge_anchor_projection.weight.grad
    assert gradient is not None and int(torch.count_nonzero(gradient)) > 0


def test_observable_phase_transport_is_bounded_and_preserves_the_protected_anchor() -> None:
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
        phase_transport_gain_caps=(0.5, 0.25, 0.1),
        phase_transport_offset_caps_px=(0.5, 0.25, 0.125),
    ).eval()
    valid = torch.ones(1, 1, 32, 32)
    pyramid = model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    base = torch.rand(1, 3, 32, 32)
    metadata = model.residual_state_metadata()
    assert metadata["phase_transport_enabled"] is True
    assert metadata["phase_transport_gain_caps"] == (0.5, 0.25, 0.1)
    assert metadata["phase_transport_offset_caps_px"] == (0.5, 0.25, 0.125)
    assert metadata["phase_transport_initial_gate"] == pytest.approx(0.02)
    assert metadata["phase_transport_null_calibrated"] is False
    assert metadata["phase_transport_null_quantile"] == pytest.approx(0.75)
    assert metadata["phase_transport_support_epsilon"] == pytest.approx(0.01)
    anchor = model.id_bridge_anchor_detail(pyramid, base, SENTINEL2)
    delta, diagnostics = model.phase_transport_delta(pyramid, base, SENTINEL2)
    assert delta.shape == base.shape
    assert diagnostics["gain"].shape == (1, 3, 8, 8)
    assert diagnostics["gate"].shape == (1, 3, 8, 8)
    assert diagnostics["offset_px"].shape == (1, 3, 2, 8, 8)
    assert diagnostics["coherence"].shape == (1, 3, 8, 8)
    assert diagnostics["source_phase"].shape == (1, 3, 32, 32)
    assert bool(torch.isfinite(delta).all())
    assert float(delta.detach().abs().amax()) < 0.1
    assert bool((diagnostics["gain"] >= 0.0).all())
    torch.testing.assert_close(
        diagnostics["gate"],
        torch.full_like(diagnostics["gate"], 0.02),
        atol=1e-7,
        rtol=1e-7,
    )
    head = model.phase_transport_head
    assert head.output[-1].out_channels == 9
    assert int(torch.count_nonzero(head.output[-1].weight)) == 0
    torch.testing.assert_close(
        head.output[-1].bias[:3],
        torch.full_like(head.output[-1].bias[:3], math.log(0.02 / 0.98)),
    )
    torch.testing.assert_close(head.output[-1].bias[3:], torch.zeros_like(head.output[-1].bias[3:]))
    detail = model.visual_detail(pyramid, SENTINEL1, SENTINEL2, (32, 32), base)
    torch.testing.assert_close(detail - anchor, delta, atol=1e-7, rtol=1e-7)
    first = model.sample_visual_residual(pyramid, SENTINEL2, base, detail, seed=11)
    second = model.sample_visual_residual(pyramid, SENTINEL2, base, detail, seed=12)
    torch.testing.assert_close(first, torch.zeros_like(first), atol=0.0, rtol=0.0)
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)

    with torch.no_grad():
        head.output[-1].bias.fill_(12.0)
    delta, diagnostics = model.phase_transport_delta(pyramid, base, SENTINEL2)
    assert bool(torch.isfinite(delta).all())
    assert bool(torch.isfinite(diagnostics["coherence"]).all())
    gain_caps = torch.tensor((0.5, 0.25, 0.1)).view(1, 3, 1, 1)
    offset_caps = torch.tensor((0.5, 0.25, 0.125)).view(1, 3, 1, 1, 1)
    assert bool((diagnostics["gain"].abs() <= gain_caps + 1e-6).all())
    assert bool((diagnostics["gain"] >= 0.0).all())
    assert bool((diagnostics["gate"] > 0.0).all())
    assert bool((diagnostics["gate"] < 1.0).all())
    assert bool((diagnostics["offset_px"].abs() <= offset_caps + 1e-6).all())
    leakage = F.avg_pool2d(delta.detach(), 8, stride=8).abs().mean()
    assert float(leakage / delta.detach().abs().mean().clamp_min(1e-8)) < 0.1
    bands = torch.randn(1, 3, 3, 32, 32)
    torch.testing.assert_close(
        head.warp_bands(bands, torch.zeros(1, 3, 2, 32, 32)), bands, atol=1e-6, rtol=1e-6
    )

    with torch.no_grad():
        head.output[-1].bias[:3].fill_(-12.0)
    _, negative_diagnostics = model.phase_transport_delta(pyramid, base, SENTINEL2)
    assert bool((negative_diagnostics["gain"] >= 0.0).all())
    assert float(negative_diagnostics["gate"].detach().amax()) < 1e-5


def test_null_calibrated_phase_transport_uses_roll_nulls_without_offsets() -> None:
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
        phase_transport_null_calibrated=True,
    ).eval()
    head = model.phase_transport_head
    assert head.output[-1].out_channels == 3
    valid = torch.ones(1, 1, 32, 32)
    pyramid = model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    base = torch.rand(1, 3, 32, 32)
    first_delta, diagnostics = model.phase_transport_delta(pyramid, base, SENTINEL2)
    second_delta, second_diagnostics = model.phase_transport_delta(pyramid, base, SENTINEL2)
    torch.testing.assert_close(first_delta, second_delta, atol=0.0, rtol=0.0)
    torch.testing.assert_close(diagnostics["null_level"], second_diagnostics["null_level"])
    assert "offset_px" not in diagnostics
    assert {
        "gain",
        "gate",
        "effective_gate",
        "gain_support",
        "coherence",
        "null_level",
        "null_coherence",
        "source_phase",
    } <= diagnostics.keys()
    assert bool(torch.isfinite(first_delta).all())
    assert bool((diagnostics["gain"] >= 0.0).all())
    assert bool((diagnostics["effective_gate"] <= diagnostics["gate"]).all())
    assert bool((diagnostics["gain_support"] >= 0.0).all())
    assert bool((diagnostics["gain_support"] <= 1.0).all())
    metadata = model.residual_state_metadata()
    assert metadata["phase_transport_null_calibrated"] is True
    assert metadata["phase_transport_null_quantile"] == pytest.approx(0.75)
    assert metadata["phase_transport_support_epsilon"] == pytest.approx(0.01)

    physical_bands = torch.stack(frequency_bands(base, levels=3), dim=1)
    source_phase = diagnostics["source_phase"]
    null_height = head.phase_coherence(source_phase.roll(16, dims=-2), physical_bands)
    null_width = head.phase_coherence(source_phase.roll(16, dims=-1), physical_bands)
    torch.testing.assert_close(
        diagnostics["null_coherence"], 0.5 * (null_height + null_width)
    )

    detail = model.visual_detail(pyramid, SENTINEL1, SENTINEL2, (32, 32), base)
    sample_a = model.sample_visual_residual(pyramid, SENTINEL2, base, detail, seed=17)
    sample_b = model.sample_visual_residual(pyramid, SENTINEL2, base, detail, seed=17)
    torch.testing.assert_close(sample_a, sample_b, atol=0.0, rtol=0.0)


def test_null_calibrated_support_rejects_roll_controls_and_has_finite_zero_energy_gradients() -> None:
    model = _haar_model(phase_transport=True, phase_transport_null_calibrated=True)
    head = model.phase_transport_head
    height = width = 32
    signal = torch.zeros(1, 1, height, width)
    signal[..., 4:12, 5:13] = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    source = signal.expand(1, 3, -1, -1).clone()
    physical = signal.unsqueeze(1).expand(1, 3, 3, -1, -1).clone()
    aligned = head.phase_coherence(source, physical)
    rolled_height = head.phase_coherence(source.roll(height // 2, dims=-2), physical)
    rolled_width = head.phase_coherence(source.roll(width // 2, dims=-1), physical)
    _, support, _ = head._null_calibrated_support(source, physical, aligned)
    assert float(aligned.mean()) > float(rolled_height.mean()) + 0.05
    assert float(aligned.mean()) > float(rolled_width.mean()) + 0.05
    assert float(support.max()) > 0.5

    constant_source = torch.ones(1, 3, height, width, requires_grad=True)
    constant_physical = torch.ones(1, 3, 3, height, width)
    zero_coherence = head.phase_coherence(constant_source, constant_physical)
    _, zero_support, _ = head._null_calibrated_support(
        constant_source, constant_physical, zero_coherence
    )
    assert int(torch.count_nonzero(zero_support)) == 0
    (zero_coherence.mean() + zero_support.mean()).backward()
    assert constant_source.grad is not None
    assert bool(torch.isfinite(constant_source.grad).all())


def test_null_calibrated_phase_transport_migrates_v3_gain_rows_after_ema_overlay() -> None:
    legacy = _haar_model(phase_transport=True)
    initialized = _haar_model(phase_transport=True, phase_transport_null_calibrated=True)
    weight_name = "phase_transport_head.output.2.weight"
    bias_name = "phase_transport_head.output.2.bias"
    legacy_output = legacy.phase_transport_head.output[-1]
    with torch.no_grad():
        legacy_output.weight.copy_(
            torch.arange(legacy_output.weight.numel(), dtype=legacy_output.weight.dtype).reshape_as(
                legacy_output.weight
            )
        )
        legacy_output.bias.copy_(
            torch.arange(legacy_output.bias.numel(), dtype=legacy_output.bias.dtype)
        )
    initial_state = {name: value.detach().clone() for name, value in legacy.state_dict().items()}
    # This mirrors init_use_ema, where EMA values overlay the raw checkpoint state.
    initial_state[weight_name] = initial_state[weight_name] + 17.0
    initial_state[bias_name] = initial_state[bias_name] + 23.0

    loaded, _ = training_module._load_compatible_state(initialized, initial_state)

    assert loaded > 0
    target_output = initialized.phase_transport_head.output[-1]
    torch.testing.assert_close(target_output.weight, initial_state[weight_name][:3])
    torch.testing.assert_close(target_output.bias, initial_state[bias_name][:3])


def test_orthogonal_carrier_loads_legacy_projection_and_starts_at_parallel_delta() -> None:
    legacy = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        phase_transport_null_calibrated=True,
    ).eval()
    carrier = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        phase_transport_null_calibrated=True,
        phase_transport_carrier_mode="orthogonal_source",
    ).eval()
    legacy_state = {name: value.detach().clone() for name, value in legacy.state_dict().items()}
    loaded, missing = training_module._load_compatible_state(carrier, legacy_state)
    assert loaded > 0 and missing > 0
    head = carrier.phase_transport_head
    torch.testing.assert_close(
        head.carrier_source_phase_projection.weight,
        legacy.phase_transport_head.source_phase_projection.weight,
    )
    torch.testing.assert_close(
        head.carrier_source_phase_projection.bias,
        legacy.phase_transport_head.source_phase_projection.bias,
    )
    torch.testing.assert_close(head.carrier_head.weight, torch.zeros_like(head.carrier_head.weight))
    torch.testing.assert_close(head.carrier_head.bias, torch.zeros_like(head.carrier_head.bias))

    valid = torch.ones(1, 1, 32, 32)
    pyramid = legacy.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    base = torch.rand(1, 3, 32, 32)
    parallel_delta, _ = legacy.phase_transport_delta(pyramid, base, SENTINEL2)
    total_delta, diagnostics = carrier.phase_transport_delta(pyramid, base, SENTINEL2)
    torch.testing.assert_close(diagnostics["carrier_delta"], torch.zeros_like(total_delta), atol=0.0, rtol=0.0)
    torch.testing.assert_close(diagnostics["parallel_delta"], parallel_delta, atol=0.0, rtol=0.0)
    torch.testing.assert_close(total_delta, parallel_delta, atol=0.0, rtol=0.0)
    legacy_detail = legacy.visual_detail(pyramid, SENTINEL1, SENTINEL2, base.shape[-2:], base)
    carrier_detail = carrier.visual_detail(pyramid, SENTINEL1, SENTINEL2, base.shape[-2:], base)
    torch.testing.assert_close(carrier_detail, legacy_detail, atol=0.0, rtol=0.0)

    with torch.no_grad():
        head.carrier_head.bias.fill_(0.25)
    resumed = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        phase_transport_null_calibrated=True,
        phase_transport_carrier_mode="orthogonal_source",
    )
    resumed_loaded, resumed_missing = training_module._load_compatible_state(
        resumed, {name: value.detach().clone() for name, value in carrier.state_dict().items()}
    )
    assert resumed_loaded > 0 and resumed_missing < len(resumed.state_dict())
    resumed_state = resumed.state_dict()
    for name, value in carrier.state_dict().items():
        if name.startswith("phase_transport_head.carrier_"):
            torch.testing.assert_close(resumed_state[name], value)


def test_bnes_loads_existing_ncopc_state_and_starts_at_parallel_delta() -> None:
    continuous = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        phase_transport_null_calibrated=True,
        phase_transport_carrier_mode="orthogonal_source",
    ).eval()
    binary = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        phase_transport_null_calibrated=True,
        phase_transport_carrier_mode="orthogonal_source",
        phase_transport_carrier_gain_caps=(0.125, 0.0625, 0.025),
        phase_transport_carrier_support_mode="binary_exceedance",
    ).eval()
    loaded, missing = training_module._load_compatible_state(
        binary, {name: value.detach().clone() for name, value in continuous.state_dict().items()}
    )
    assert loaded > 0 and missing == 0
    valid = torch.ones(1, 1, 32, 32)
    pyramid = continuous.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    base = torch.rand(1, 3, 32, 32)
    continuous_delta, _ = continuous.phase_transport_delta(pyramid, base, SENTINEL2)
    binary_delta, diagnostics = binary.phase_transport_delta(pyramid, base, SENTINEL2)
    torch.testing.assert_close(diagnostics["carrier_delta"], torch.zeros_like(binary_delta))
    torch.testing.assert_close(binary_delta, continuous_delta, atol=0.0, rtol=0.0)
    metadata = binary.residual_state_metadata()
    assert metadata["phase_transport_carrier_gain_caps"] == (0.125, 0.0625, 0.025)
    assert metadata["phase_transport_carrier_support_mode"] == "binary_exceedance"


def test_observable_phase_transport_sigmoid_gate_is_smooth_and_bounded() -> None:
    raw_gain = torch.tensor(((-30.0, 0.0, 30.0),), requires_grad=True)
    gate = ObservablePhaseTransportHead.gain_gate(raw_gain)
    assert float(gate[0, 0].detach()) < 1e-10
    assert float(gate[0, 1].detach()) == pytest.approx(0.5)
    assert float(gate[0, 2].detach()) > 1.0 - 1e-6
    gate.sum().backward()
    assert raw_gain.grad is not None
    assert bool(torch.isfinite(raw_gain.grad).all())


def test_phase_transport_gain_target_matches_positive_constrained_blockwise_least_squares() -> None:
    height = width = 16
    coordinates = torch.arange(height).view(height, 1) + torch.arange(width).view(1, width)
    component = (coordinates.remainder(2).mul(2).sub(1)).float().view(1, 1, height, width)
    physical_bands = torch.zeros(1, 3, 1, height, width)
    physical_bands[:, :1] = component
    valid = torch.ones(1, 1, height, width)
    caps = (0.5, 0.25, 0.1)

    target = phase_transport_gain_target(physical_bands, 0.25 * component, valid, caps)
    assert target.shape == (1, 3, height // 4, width // 4)
    torch.testing.assert_close(target[:, :1], torch.full_like(target[:, :1], 0.5))
    assert int(torch.count_nonzero(target[:, 1:])) == 0

    capped = phase_transport_gain_target(physical_bands, 2.0 * component, valid, caps)
    torch.testing.assert_close(capped[:, :1], torch.ones_like(capped[:, :1]))
    negative = phase_transport_gain_target(physical_bands, -component, valid, caps)
    assert int(torch.count_nonzero(negative)) == 0
    empty = phase_transport_gain_target(
        torch.zeros_like(physical_bands), torch.zeros_like(component), valid, caps
    )
    assert int(torch.count_nonzero(empty)) == 0

    partial_valid = valid.clone()
    partial_valid[..., :4, :4] = 0.0
    partial = phase_transport_gain_target(
        physical_bands, 0.25 * component, partial_valid, caps
    )
    assert int(torch.count_nonzero(partial[..., :1, :1])) == 0
    torch.testing.assert_close(partial[:, :1, 1:, 1:], torch.full_like(partial[:, :1, 1:, 1:], 0.5))

    with pytest.raises(ValueError, match="three"):
        phase_transport_gain_target(physical_bands[:, :2], component, valid, caps)
    with pytest.raises(ValueError, match="gain_caps"):
        phase_transport_gain_target(physical_bands, component, valid, (0.5, -0.25, 0.1))
    with pytest.raises(ValueError, match="block_size"):
        phase_transport_gain_target(physical_bands, component, valid, caps, block_size=0)


def test_observable_phase_transport_coherence_and_alignment_are_sign_invariant() -> None:
    model = _haar_model(anchor_origin=True, phase=True, optical_only=True, phase_transport=True)
    head = model.phase_transport_head
    height = width = 32
    horizontal = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width).expand(
        1, 1, height, width
    )
    vertical = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1).expand(
        1, 1, height, width
    )
    source = horizontal.expand(1, 3, height, width)
    parallel = source.unsqueeze(2).expand(-1, -1, 3, -1, -1)
    orthogonal = vertical.expand(1, 3, height, width).unsqueeze(2).expand(-1, -1, 3, -1, -1)
    coherence = head.phase_coherence(source, parallel)
    inverted = head.phase_coherence(-source, parallel)
    perpendicular = head.phase_coherence(source, orthogonal)
    assert float(coherence.mean()) > float(perpendicular.mean()) + 0.5
    torch.testing.assert_close(coherence, inverted, atol=1e-6, rtol=1e-6)

    valid = torch.ones(1, 1, height, width)
    target_parallel = horizontal.expand(1, 3, height, width)
    target_perpendicular = vertical.expand(1, 3, height, width)
    aligned_loss = phase_alignment_loss(source, (target_parallel,) * 3, valid)
    perpendicular_loss = phase_alignment_loss(source, (target_perpendicular,) * 3, valid)
    inverted_loss = phase_alignment_loss(-source, (target_parallel,) * 3, valid)
    assert bool(torch.isfinite(aligned_loss))
    assert float(aligned_loss) + 0.5 < float(perpendicular_loss)
    torch.testing.assert_close(aligned_loss, inverted_loss, atol=1e-6, rtol=1e-6)


def test_signed_phase_alignment_prefers_matching_polarity_and_zero_energy_is_finite() -> None:
    source = torch.randn(2, 3, 32, 32, requires_grad=True)
    target_bands = tuple(
        source[:, band : band + 1].detach().repeat(1, 3, 1, 1) for band in range(3)
    )
    valid = torch.ones(2, 1, 32, 32)
    aligned = signed_phase_alignment_loss(source, target_bands, valid)
    inverted = signed_phase_alignment_loss(-source, target_bands, valid)
    assert bool(torch.isfinite(aligned))
    assert float(aligned.detach()) + 1.0 < float(inverted.detach())

    zero_source = torch.zeros_like(source, requires_grad=True)
    zero_loss = signed_phase_alignment_loss(
        zero_source,
        tuple(torch.zeros_like(band) for band in target_bands),
        valid,
    )
    assert bool(torch.isfinite(zero_loss))
    zero_loss.backward()
    assert zero_source.grad is not None
    assert bool(torch.isfinite(zero_source.grad).all())


def test_orthogonal_source_phase_carrier_injects_luminance_without_low_frequency_leakage() -> None:
    torch.manual_seed(31)
    head = ObservablePhaseTransportHead(
        (8, 16, 32, 32),
        hidden=32,
        null_calibrated=True,
        carrier_mode="orthogonal_source",
    )
    height = width = 32
    physical = torch.randn(2, 3, height, width)
    physical_bands = torch.stack(frequency_bands(physical, levels=3), dim=1)
    full_gains = torch.full((2, 3, height, width), 0.25)
    source = torch.randn(2, 3, height, width, requires_grad=True)

    carriers, carrier_rms, carrier_orthogonality = head.orthogonal_source_carriers(
        source, physical_bands
    )
    delta, diagnostics = head._orthogonal_source_delta(source, physical_bands, full_gains)
    inverted_delta, _ = head._orthogonal_source_delta(-source, physical_bands, full_gains)
    assert bool(torch.isfinite(carriers).all())
    assert bool(torch.isfinite(carrier_rms).all())
    assert bool(torch.isfinite(carrier_orthogonality).all())
    assert float(carrier_orthogonality.detach().abs().mean()) < 0.15
    assert float(carrier_rms.detach().mean()) > 0.0
    assert diagnostics["carrier_components"].requires_grad
    assert diagnostics["carrier_components"].shape == (2, 3, 3, height, width)
    torch.testing.assert_close(carriers[:, :, 0], carriers[:, :, 1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(carriers[:, :, 1], carriers[:, :, 2], atol=0.0, rtol=0.0)
    torch.testing.assert_close(delta[:, 0], delta[:, 1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(delta[:, 1], delta[:, 2], atol=0.0, rtol=0.0)
    torch.testing.assert_close(inverted_delta, -delta, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(diagnostics["carrier_components"], carriers)
    torch.testing.assert_close(diagnostics["carrier_rms"], carrier_rms)
    torch.testing.assert_close(diagnostics["carrier_orthogonality"], carrier_orthogonality)
    leakage = F.avg_pool2d(delta, 8, stride=8).abs().mean()
    assert float((leakage / delta.abs().mean().clamp_min(1e-8)).detach()) < 0.1


def test_orthogonal_source_phase_carrier_forward_uses_source_phase_projection() -> None:
    torch.manual_seed(33)
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        phase_transport_null_calibrated=True,
        phase_transport_carrier_mode="orthogonal_source",
    )
    head = model.phase_transport_head
    with torch.no_grad():
        head.output[-1].weight.zero_()
        head.output[-1].bias.fill_(12.0)
        head.carrier_source_phase_projection.weight.zero_()
        head.carrier_source_phase_projection.bias.zero_()
        for band in range(3):
            head.carrier_source_phase_projection.weight[band, band, 0, 0] = 1.0
        head.carrier_head.weight.zero_()
        head.carrier_head.bias.fill_(1.0)
    pyramid = (
        torch.randn(1, 8, 32, 32),
        torch.randn(1, 16, 16, 16),
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 32, 4, 4),
    )
    physical = torch.randn(1, 3, 32, 32)
    delta, diagnostics = head(pyramid, physical)
    inverted_delta, inverted_diagnostics = head((-pyramid[0], *pyramid[1:]), physical)
    assert int(torch.count_nonzero(delta)) > 0
    assert {
        "parallel_delta",
        "carrier_delta",
        "carrier_source_phase",
        "carrier_signed_gate",
        "carrier_effective_signed_coeff",
        "carrier_support",
    } <= diagnostics.keys()
    torch.testing.assert_close(
        inverted_diagnostics["carrier_delta"],
        -diagnostics["carrier_delta"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        inverted_delta,
        inverted_diagnostics["parallel_delta"] + inverted_diagnostics["carrier_delta"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_orthogonal_source_phase_carrier_respects_null_support_and_zero_physical_rms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(37)
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        phase_transport_null_calibrated=True,
        phase_transport_carrier_mode="orthogonal_source",
    )
    head = model.phase_transport_head
    assert head.carrier_mode == "orthogonal_source"
    height = width = 32
    source = torch.randn(1, 3, height, width, requires_grad=True)
    zero_physical = torch.zeros(1, 3, height, width, requires_grad=True)
    zero_bands = torch.stack(frequency_bands(zero_physical, levels=3), dim=1)
    zero_delta, _ = head._orthogonal_source_delta(
        source, zero_bands, torch.ones(1, 3, height, width)
    )
    torch.testing.assert_close(zero_delta, torch.zeros_like(zero_delta), atol=0.0, rtol=0.0)
    zero_delta.square().mean().backward()
    assert source.grad is not None and bool(torch.isfinite(source.grad).all())
    assert zero_physical.grad is not None and bool(torch.isfinite(zero_physical.grad).all())

    valid = torch.ones(1, 1, height, width)
    pyramid = model.encode(torch.randn(1, 2, height, width), SENTINEL1, valid)
    physical = torch.randn(1, 3, height, width)

    def no_carrier_support(
        _source_phase: torch.Tensor,
        _physical_bands: torch.Tensor,
        coherence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(coherence)
        return zeros, zeros, zeros

    with torch.no_grad():
        head.output[-1].bias.fill_(12.0)
        head.carrier_head.bias.fill_(1.0)
    monkeypatch.setattr(head, "_carrier_null_calibrated_support", no_carrier_support)
    delta, diagnostics = head(pyramid, physical)
    torch.testing.assert_close(
        diagnostics["carrier_delta"],
        torch.zeros_like(diagnostics["carrier_delta"]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        delta,
        diagnostics["parallel_delta"],
        atol=0.0,
        rtol=0.0,
    )
    assert {"carrier_rms", "carrier_orthogonality", "carrier_support"} <= diagnostics.keys()


def test_carrier_support_modes_keep_continuous_formula_and_binary_is_detached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.randn(1, 3, 8, 8)
    physical_bands = torch.randn(1, 3, 3, 8, 8)

    def fixed_null(*_args: object) -> torch.Tensor:
        return torch.full((1, 3, 2, 2), 0.5)

    continuous = ObservablePhaseTransportHead(
        (8, 16, 32, 32),
        hidden=32,
        carrier_mode="orthogonal_source",
        support_epsilon=0.1,
    )
    monkeypatch.setattr(continuous, "phase_coherence", fixed_null)
    continuous_coherence = torch.tensor(
        [[[[0.2, 0.5], [0.7, 0.9]]] * 3], requires_grad=True
    )
    continuous_null, continuous_support, _ = continuous._carrier_null_calibrated_support(
        source, physical_bands, continuous_coherence
    )
    continuous_excess = F.relu(continuous_coherence - continuous_null)
    continuous_expected = continuous_excess / (
        continuous_excess + continuous_null + continuous.support_epsilon
    )
    torch.testing.assert_close(continuous_support, continuous_expected, atol=0.0, rtol=0.0)
    continuous_support.sum().backward()
    assert continuous_coherence.grad is not None
    assert int(torch.count_nonzero(continuous_coherence.grad)) > 0

    binary = ObservablePhaseTransportHead(
        (8, 16, 32, 32),
        hidden=32,
        carrier_mode="orthogonal_source",
        carrier_support_mode="binary_exceedance",
    )
    monkeypatch.setattr(binary, "phase_coherence", fixed_null)
    binary_coherence = torch.tensor(
        [[[[float("nan"), 0.5], [0.5001, 0.2]]] * 3], requires_grad=True
    )
    binary_null, binary_support, _ = binary._carrier_null_calibrated_support(
        source, physical_bands, binary_coherence
    )
    expected_binary = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]] * 3])
    torch.testing.assert_close(binary_null, torch.full_like(binary_null, 0.5))
    torch.testing.assert_close(binary_support, expected_binary, atol=0.0, rtol=0.0)
    assert not binary_support.requires_grad
    assert set(binary_support.unique().tolist()) <= {0.0, 1.0}


def test_bnes_carrier_caps_and_support_control_only_the_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        phase_transport_null_calibrated=True,
        phase_transport_carrier_mode="orthogonal_source",
        phase_transport_carrier_gain_caps=(0.125, 0.0625, 0.025),
        phase_transport_carrier_support_mode="binary_exceedance",
    ).eval()
    wide = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        phase_transport_null_calibrated=True,
        phase_transport_carrier_mode="orthogonal_source",
        phase_transport_carrier_support_mode="binary_exceedance",
    ).eval()
    wide.load_state_dict(small.state_dict())

    def support_one(
        _source_phase: torch.Tensor,
        _physical_bands: torch.Tensor,
        coherence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(coherence)
        return zeros, torch.ones_like(coherence), zeros

    for head in (small.phase_transport_head, wide.phase_transport_head):
        with torch.no_grad():
            head.output[-1].bias.fill_(12.0)
            head.carrier_head.bias.fill_(0.7)
        monkeypatch.setattr(head, "_carrier_null_calibrated_support", support_one)
    valid = torch.ones(1, 1, 32, 32)
    pyramid = small.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    physical = torch.randn(1, 3, 32, 32)
    small_delta, small_diagnostics = small.phase_transport_delta(pyramid, physical, SENTINEL2)
    wide_delta, wide_diagnostics = wide.phase_transport_delta(pyramid, physical, SENTINEL2)
    torch.testing.assert_close(
        small_diagnostics["parallel_delta"], wide_diagnostics["parallel_delta"], atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        small_diagnostics["carrier_delta"],
        0.25 * wide_diagnostics["carrier_delta"],
        atol=1e-7,
        rtol=1e-6,
    )
    assert int(torch.count_nonzero(small_diagnostics["carrier_delta"])) > 0
    torch.testing.assert_close(small_diagnostics["carrier_support"], torch.ones_like(small_diagnostics["carrier_support"]))

    with torch.no_grad():
        small.phase_transport_head.carrier_head.bias.fill_(-0.7)
    negative_delta, negative_diagnostics = small.phase_transport_delta(pyramid, physical, SENTINEL2)
    torch.testing.assert_close(
        negative_diagnostics["carrier_delta"],
        -small_diagnostics["carrier_delta"],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        negative_delta,
        negative_diagnostics["parallel_delta"] + negative_diagnostics["carrier_delta"],
        atol=0.0,
        rtol=0.0,
    )

    def support_zero(
        _source_phase: torch.Tensor,
        _physical_bands: torch.Tensor,
        coherence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(coherence)
        return zeros, zeros, zeros

    monkeypatch.setattr(small.phase_transport_head, "_carrier_null_calibrated_support", support_zero)
    zero_delta, zero_diagnostics = small.phase_transport_delta(pyramid, physical, SENTINEL2)
    torch.testing.assert_close(zero_diagnostics["carrier_delta"], torch.zeros_like(zero_delta))
    torch.testing.assert_close(zero_delta, zero_diagnostics["parallel_delta"], atol=0.0, rtol=0.0)
    assert not torch.equal(small_delta, wide_delta)


def test_bnes_alignment_mask_excludes_null_regions_while_continuous_keeps_legacy_loss() -> None:
    valid = torch.ones(1, 1, 8, 8)
    support = torch.zeros(1, 3, 2, 2, requires_grad=True)
    with torch.no_grad():
        support[:, 0, 0, 1] = 0.001
        support[:, 2, 1, 0] = 0.75
    continuous_objective = JointObjective(
        _haar_model(
            phase_transport=True,
            phase_transport_carrier_mode="orthogonal_source",
        ),
        [0.5, 0.5],
    )
    continuous_mask = continuous_objective._carrier_signed_alignment_mask(valid, support)
    assert continuous_mask is valid
    source = torch.randn(1, 3, 8, 8, requires_grad=True)
    target_bands = tuple(torch.randn(1, 3, 8, 8) for _ in range(3))
    legacy_loss = signed_phase_alignment_loss(source, target_bands, valid)
    continuous_loss = signed_phase_alignment_loss(source, target_bands, continuous_mask)
    torch.testing.assert_close(continuous_loss, legacy_loss, atol=0.0, rtol=0.0)

    binary_objective = JointObjective(
        _haar_model(
            phase_transport=True,
            phase_transport_carrier_mode="orthogonal_source",
            phase_transport_carrier_support_mode="binary_exceedance",
        ),
        [0.5, 0.5],
    )
    mask = binary_objective._carrier_signed_alignment_mask(valid, support)
    expected = torch.zeros_like(valid)
    expected[..., :4, 4:] = 1.0
    expected[..., 4:, :4] = 1.0
    torch.testing.assert_close(mask, expected, atol=0.0, rtol=0.0)
    assert not mask.requires_grad
    empty_mask = binary_objective._carrier_signed_alignment_mask(
        valid, torch.zeros_like(support)
    )
    torch.testing.assert_close(empty_mask, torch.zeros_like(valid), atol=0.0, rtol=0.0)
    empty_loss = signed_phase_alignment_loss(
        source,
        target_bands,
        empty_mask,
    )
    assert bool(torch.isfinite(empty_loss))
    assert float(empty_loss.detach()) == 0.0
    empty_loss.backward()
    assert source.grad is not None
    assert int(torch.count_nonzero(source.grad)) == 0


def test_physical_gain_carrier_mode_preserves_legacy_phase_transport_formula() -> None:
    torch.manual_seed(41)
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        phase_transport_null_calibrated=True,
    )
    assert model.config.phase_transport_carrier_mode == "physical_gain"
    head = model.phase_transport_head
    valid = torch.ones(1, 1, 32, 32)
    pyramid = model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    physical = torch.randn(1, 3, 32, 32)
    delta, diagnostics = head(pyramid, physical)
    physical_bands = torch.stack(frequency_bands(physical, levels=3), dim=1)
    full_gains = F.interpolate(
        diagnostics["gain"], size=physical.shape[-2:], mode="bilinear", align_corners=False
    )
    expected = highpass((full_gains.unsqueeze(2) * physical_bands).sum(dim=1))
    torch.testing.assert_close(delta, expected, atol=0.0, rtol=0.0)
    assert "carrier_rms" not in diagnostics
    assert "carrier_orthogonality" not in diagnostics


def test_phase_transport_gain_oracle_prefers_aligned_carriers_over_orthogonal_physical() -> None:
    height = width = 16
    columns = (
        torch.arange(width).remainder(2).mul(2).sub(1).float().view(1, 1, 1, width)
    ).expand(1, 3, height, width)
    rows = (
        torch.arange(height).remainder(2).mul(2).sub(1).float().view(1, 1, height, 1)
    ).expand(1, 3, height, width)
    physical_bands = torch.zeros(1, 3, 3, height, width)
    physical_bands[:, 0] = columns
    aligned_carriers = torch.zeros_like(physical_bands)
    aligned_carriers[:, 0] = rows
    residual = 0.1 * rows
    valid = torch.ones(1, 1, height, width)
    caps = (0.5, 0.25, 0.1)

    physical_oracle = phase_transport_gain_target(physical_bands, residual, valid, caps)
    carrier_oracle = phase_transport_gain_target(aligned_carriers, residual, valid, caps)
    unhelpful_carriers = physical_bands.clone()
    unhelpful_oracle = phase_transport_gain_target(unhelpful_carriers, residual, valid, caps)
    torch.testing.assert_close(physical_oracle, torch.zeros_like(physical_oracle))
    assert float(carrier_oracle[:, 0].mean()) > 0.0
    torch.testing.assert_close(unhelpful_oracle, torch.zeros_like(unhelpful_oracle))


def test_phase_transport_signed_coefficient_oracle_preserves_polarity() -> None:
    height = width = 16
    component = (
        torch.arange(width).remainder(2).mul(2).sub(1).float().view(1, 1, 1, width)
    ).expand(1, 3, height, width)
    carriers = torch.zeros(1, 3, 3, height, width)
    carriers[:, 0] = component
    valid = torch.ones(1, 1, height, width)
    caps = (0.5, 0.25, 0.1)
    positive = phase_transport_signed_coefficient_target(
        carriers, 0.1 * component, valid, caps
    )
    empty = phase_transport_signed_coefficient_target(
        carriers, torch.zeros_like(component), valid, caps
    )
    negative = phase_transport_signed_coefficient_target(
        carriers, -0.1 * component, valid, caps
    )
    assert float(positive[:, 0].mean()) > 0.0
    torch.testing.assert_close(empty, torch.zeros_like(empty))
    assert float(negative[:, 0].mean()) < 0.0
    torch.testing.assert_close(negative, -positive, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("constant_source", "constant_physical"),
    ((True, False), (False, True), (True, True)),
)
def test_observable_phase_transport_zero_energy_inputs_have_finite_gradients(
    constant_source: bool,
    constant_physical: bool,
) -> None:
    model = _haar_model(anchor_origin=True, phase=True, optical_only=True, phase_transport=True)
    head = model.phase_transport_head
    height = width = 32
    source = (
        torch.ones(1, 3, height, width)
        if constant_source
        else torch.randn(1, 3, height, width)
    ).requires_grad_()
    physical_bands = (
        torch.ones(1, 3, 3, height, width)
        if constant_physical
        else torch.randn(1, 3, 3, height, width)
    )
    valid = torch.ones(1, 1, height, width)

    coherence = head.phase_coherence(source, physical_bands)
    alignment = phase_alignment_loss(
        source,
        tuple(physical_bands[:, index] for index in range(3)),
        valid,
    )
    total = coherence.mean() + alignment
    assert bool(torch.isfinite(coherence).all())
    assert bool(torch.isfinite(alignment))
    assert bool(torch.isfinite(total))
    total.backward()
    assert source.grad is not None
    assert bool(torch.isfinite(source.grad).all())


def test_phase_antithetic_starts_center_exactly_on_mu() -> None:
    mu = torch.randn(2, 48, 8, 8)
    log_sigma = torch.randn_like(mu)
    q_state = torch.rand_like(mu)
    epsilon = torch.randn_like(mu)
    z_plus, _, sigma = JointObjective._id_bridge_start(
        mu,
        log_sigma,
        torch.zeros(2, 3, 8, 8),
        0.35,
        epsilon,
        q_state=q_state,
    )
    z_minus = mu - sigma.detach() * epsilon
    torch.testing.assert_close(0.5 * (z_plus + z_minus), mu, atol=1e-6, rtol=1e-6)


def test_phase_optical_only_keeps_sar_detail_amplitude_and_seeded_sampling_legacy() -> None:
    phase = _haar_model(
        anchor_origin=True, phase=True, optical_only=True, phase_transport=True
    ).eval()
    legacy = _utility_codec_model().eval()
    assert phase.legacy_residual_dit is not None
    phase.legacy_residual_dit.load_state_dict(legacy.residual_dit.state_dict())
    legacy.detail_head.load_state_dict(phase.detail_head.state_dict())
    legacy.codec.load_state_dict(phase.codec.state_dict())
    valid = torch.ones(1, 1, 32, 32)
    pyramid = phase.encode(torch.randn(1, 10, 32, 32), SENTINEL2, valid)
    base = torch.randn(1, 2, 32, 32)
    phase_detail = phase.visual_detail(pyramid, SENTINEL2, SENTINEL1, (32, 32), base)
    legacy_detail = legacy.visual_detail(pyramid, SENTINEL2, SENTINEL1, (32, 32), base)
    torch.testing.assert_close(phase_detail, legacy_detail, atol=0.0, rtol=0.0)
    phase_amplitude = phase.residual_amplitude(pyramid, SENTINEL1, 2, (32, 32))
    legacy_amplitude = legacy.residual_amplitude(pyramid, SENTINEL1, 2, (32, 32))
    torch.testing.assert_close(phase_amplitude, legacy_amplitude, atol=0.0, rtol=0.0)
    phase_sample = phase.sample_visual_residual(
        pyramid, SENTINEL1, base, phase_detail, seed=23
    )
    legacy_sample = legacy.sample_visual_residual(
        pyramid, SENTINEL1, base, legacy_detail, seed=23
    )
    torch.testing.assert_close(phase_sample, legacy_sample, atol=0.0, rtol=0.0)
    assignments = JointObjective(phase)._id_bridge_assignments(3, torch.device("cpu"))
    torch.testing.assert_close(assignments, torch.zeros_like(assignments))

    initialized = _haar_model(anchor_origin=True, phase=True, optical_only=True)
    assert initialized.legacy_residual_dit is not None
    training_module._load_compatible_state(initialized, legacy.state_dict())
    torch.testing.assert_close(
        initialized.legacy_residual_dit.input.weight,
        legacy.residual_dit.input.weight,
    )


def test_phase_optical_visual_zero_residual_skips_id_bridge_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
    ).eval()
    model.set_amplitude_scale("optical", 1.0)
    valid = torch.ones(1, 1, 32, 32)
    pyramid = model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    base = torch.rand(1, 3, 32, 32)
    detail = torch.zeros_like(base)
    calls: list[str] = []

    def unexpected_flow(*_args: object, **_kwargs: object) -> torch.Tensor:
        calls.append("flow")
        raise AssertionError("zero phase Optical residual must bypass the 48-channel flow")

    monkeypatch.setattr(model, "sample_id_bridge_residual", unexpected_flow)
    residual = model.sample_visual_residual(pyramid, SENTINEL2, base, detail, seed=29)
    torch.testing.assert_close(residual, torch.zeros_like(base), atol=0.0, rtol=0.0)
    assert calls == []

    model.config.id_bridge_optical_innovation_band_scales = (0.1, 0.0, 0.0)

    def sampled_flow(*_args: object, **_kwargs: object) -> torch.Tensor:
        calls.append("flow")
        return torch.full_like(base, 0.125)

    monkeypatch.setattr(model, "sample_id_bridge_residual", sampled_flow)
    residual = model.sample_visual_residual(pyramid, SENTINEL2, base, detail, seed=29)
    torch.testing.assert_close(residual, torch.full_like(base, 0.125), atol=0.0, rtol=0.0)
    assert calls == ["flow"]


def test_phase_id_bridge_training_uses_optical_oracle_and_antithetic_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
    )
    _set_trainable(model, "id_bridge")
    objective = JointObjective(
        model,
        flow_rollout_every=1,
        flow_rollout_steps=1,
        flow_visual_perceptual_weight=0.0,
        id_bridge_antithetic_weight=0.05,
    )
    original_integrate = model.integrate_flow
    rollout_steps: list[int] = []

    def capture_integrate(*args: object, **kwargs: object) -> torch.Tensor:
        rollout_steps.append(int(kwargs["steps"]))
        return original_integrate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(model, "integrate_flow", capture_integrate)
    loss, metrics = objective(_batch(), "id_bridge")
    assert bool(torch.isfinite(loss))
    for name in (
        "sar2opt/q_fine",
        "sar2opt/q_mid",
        "sar2opt/q_coarse",
        "sar2opt/oracle_q_fine",
        "sar2opt/oracle_q_mid",
        "sar2opt/oracle_q_coarse",
        "sar2opt/q_mae",
        "sar2opt/q_corr",
        "sar2opt/q_corr_fine",
        "sar2opt/q_corr_mid",
        "sar2opt/q_corr_coarse",
        "sar2opt/release_fine",
        "sar2opt/release_mid",
        "sar2opt/release_coarse",
        "sar2opt/antithetic_center",
    ):
        assert name in metrics and bool(torch.isfinite(metrics[name]))
    torch.testing.assert_close(
        metrics["sar2opt/q_corr"],
        torch.stack(
            (
                metrics["sar2opt/q_corr_fine"],
                metrics["sar2opt/q_corr_mid"],
                metrics["sar2opt/q_corr_coarse"],
            )
        ).mean(),
    )
    assert rollout_steps == [1, 1]
    loss.backward()
    q_bias = model.id_bridge_origin.output_heads["optical"][-1].bias
    assert q_bias.grad is not None and int(torch.count_nonzero(q_bias.grad[-3:])) > 0
    assert not any(parameter.grad is not None for parameter in model.decoder.parameters())
    assert model.legacy_residual_dit is not None
    assert not any(parameter.grad is not None for parameter in model.legacy_residual_dit.parameters())

    model.zero_grad(set_to_none=True)
    zero_loss, _ = objective(_batch(delta_days=2), "id_bridge")
    zero_loss.backward()
    assert float(zero_loss.detach()) == 0.0
    assert all(
        parameter.grad is not None and int(torch.count_nonzero(parameter.grad)) == 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_phase_zero_antithetic_weight_skips_minus_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(anchor_origin=True, phase=True, optical_only=True)
    _set_trainable(model, "id_bridge")
    objective = JointObjective(
        model,
        flow_rollout_every=1,
        flow_rollout_steps=1,
        flow_visual_perceptual_weight=0.0,
        id_bridge_antithetic_weight=0.0,
    )
    original_integrate = model.integrate_flow
    rollout_steps: list[int] = []

    def capture_integrate(*args: object, **kwargs: object) -> torch.Tensor:
        rollout_steps.append(int(kwargs["steps"]))
        return original_integrate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(model, "integrate_flow", capture_integrate)
    _, metrics = objective(_batch(), "id_bridge")
    assert rollout_steps == [1]
    assert "sar2opt/antithetic_center" not in metrics


def test_nonphase_id_bridge_keeps_the_default_two_step_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(anchor_origin=True)
    _set_trainable(model, "id_bridge")
    objective = JointObjective(
        model,
        flow_rollout_every=1,
        flow_rollout_steps=2,
        flow_visual_perceptual_weight=0.0,
    )
    original_integrate = model.integrate_flow
    rollout_steps: list[int] = []

    def capture_integrate(*args: object, **kwargs: object) -> torch.Tensor:
        rollout_steps.append(int(kwargs["steps"]))
        return original_integrate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(model, "integrate_flow", capture_integrate)
    _, metrics = objective(_batch(), "id_bridge")
    assert rollout_steps == [2, 2]
    assert "sar2opt/antithetic_center" not in metrics
    for name in (
        "sar2opt/oracle_q_fine",
        "sar2opt/oracle_q_mid",
        "sar2opt/oracle_q_coarse",
        "sar2opt/q_mae",
        "sar2opt/q_corr",
        "sar2opt/q_corr_fine",
        "sar2opt/q_corr_mid",
        "sar2opt/q_corr_coarse",
    ):
        assert name not in metrics


def test_utility_anchor_initialization_matches_source_anchor() -> None:
    utility_model = _haar_model(anchor_origin=True, utility=True).eval()
    optical_head = utility_model.id_bridge_origin.output_heads["optical"][-1]
    sar_head = utility_model.id_bridge_origin.output_heads["sar"][-1]
    utility_head = utility_model.id_bridge_origin.anchor_utility_head[-1]
    assert int(torch.count_nonzero(optical_head.weight)) == 0
    assert int(torch.count_nonzero(sar_head.weight)) == 0
    torch.testing.assert_close(
        utility_head.bias,
        torch.tensor((0.0, -4.0, -4.0), dtype=utility_head.bias.dtype),
    )
    assert int(torch.count_nonzero(optical_head.bias)) == 0
    assert int(torch.count_nonzero(sar_head.bias)) == 0
    gains = utility_model.id_bridge_anchor_gains(utility_head.bias.view(1, 3, 1, 1))
    torch.testing.assert_close(gains[:, :1], torch.ones_like(gains[:, :1]))
    torch.testing.assert_close(
        gains[:, 1:], torch.full_like(gains[:, 1:], math.exp(-4.0))
    )

    legacy_model = _haar_model(anchor_origin=True).eval()
    legacy_model.load_state_dict(utility_model.state_dict())
    for model in (utility_model, legacy_model):
        model.set_detail_confidence_threshold("optical", 1.01)
        model.set_optical_anchor_band_scales((0.2, 0.0, 0.0))
        model.set_optical_anchor_density(0.3, 1.0)
        model.set_optical_anchor_source_density(0.4, 1.0)
    valid = torch.ones(1, 1, 32, 32)
    pyramid = utility_model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    physical = torch.rand(1, 3, 32, 32)
    _, _, logits = utility_model.id_bridge_origin(pyramid, physical, SENTINEL2)
    utility_anchor = utility_model.id_bridge_anchor_detail(
        pyramid, physical, SENTINEL2, reliability_logits=logits
    )
    legacy_anchor = legacy_model.id_bridge_anchor_detail(pyramid, physical, SENTINEL2)
    fine_component, _, _ = utility_model.id_bridge_anchor_components(
        pyramid, physical, SENTINEL2
    )
    torch.testing.assert_close(highpass(fine_component), legacy_anchor, atol=1e-6, rtol=1e-6)
    assert int(torch.count_nonzero(utility_anchor)) > 0
    torch.testing.assert_close(
        utility_model.visual_detail(pyramid, SENTINEL1, SENTINEL2, (32, 32), physical),
        utility_anchor,
        atol=1e-6,
        rtol=1e-6,
    )
    metadata = utility_model.residual_state_metadata()
    assert metadata["anchor_utility"] is True
    assert metadata["optical_mid_basis_scale"] == pytest.approx(0.15)
    assert metadata["optical_coarse_basis_scale"] == pytest.approx(0.05)
    assert metadata["optical_correction_scale"] == pytest.approx(1.0)
    assert metadata["sar_correction_scale"] == pytest.approx(1.0)


def test_anchor_utility_head_is_independent_of_flow_state_channels() -> None:
    codec_model = _utility_codec_model().eval()
    haar_model = _haar_model(anchor_origin=True, utility=True).eval()
    codec_head = codec_model.id_bridge_origin.anchor_utility_head
    haar_head = haar_model.id_bridge_origin.anchor_utility_head
    assert codec_model.id_bridge_origin.latent_channels == 16
    assert haar_model.id_bridge_origin.latent_channels == 48
    assert {
        name: tuple(value.shape) for name, value in codec_head.state_dict().items()
    } == {name: tuple(value.shape) for name, value in haar_head.state_dict().items()}
    codec_head.load_state_dict(haar_head.state_dict())

    valid = torch.ones(1, 1, 32, 32)
    pyramid = codec_model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    _, _, logits = codec_model.id_bridge_origin(pyramid, torch.rand(1, 3, 32, 32), SENTINEL2)
    expected_logits = torch.tensor((0.0, -4.0, -4.0), dtype=logits.dtype).view(1, 3, 1, 1)
    torch.testing.assert_close(logits, expected_logits.expand_as(logits))
    gains = codec_model.id_bridge_anchor_gains(logits)
    expected = torch.tensor((1.0, math.exp(-4.0), math.exp(-4.0)), dtype=gains.dtype)
    torch.testing.assert_close(gains, expected.view(1, 3, 1, 1).expand_as(gains))


def test_disabled_bridge_utility_detail_keeps_legacy_residual_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _utility_codec_model().eval()
    model.set_optical_anchor_band_scales((0.2, 0.0, 0.0))
    valid = torch.ones(1, 1, 32, 32)
    pyramid = model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    base = torch.rand(1, 3, 32, 32)

    def deterministic_detail(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("utility visual detail must bypass the legacy detail head")

    monkeypatch.setattr(model, "deterministic_detail", deterministic_detail)
    detail = model.visual_detail(pyramid, SENTINEL1, SENTINEL2, (32, 32), base)
    _, _, logits = model.id_bridge_origin(pyramid, base, SENTINEL2)
    expected_detail = model.id_bridge_anchor_detail(
        pyramid, base, SENTINEL2, reliability_logits=logits
    )
    torch.testing.assert_close(detail, expected_detail)
    assert int(torch.count_nonzero(detail)) > 0

    calls: list[str] = []

    def legacy_sample(
        _pyramid: object,
        _target: object,
        shape: tuple[int, int, int, int],
        *,
        seed: int,
        steps: int | None,
        bridge_anchor: torch.Tensor | None,
    ) -> torch.Tensor:
        calls.append("legacy")
        assert shape == tuple(base.shape)
        assert seed == 17 and steps is None and bridge_anchor is detail
        return torch.full_like(base, 0.1)

    def id_bridge_sample(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("disabled id bridge must retain legacy residual sampling")

    monkeypatch.setattr(model, "sample_residual", legacy_sample)
    monkeypatch.setattr(model, "sample_id_bridge_residual", id_bridge_sample)
    residual = model.sample_visual_residual(pyramid, SENTINEL2, base, detail, seed=17)
    torch.testing.assert_close(residual, torch.full_like(base, 0.1))
    assert calls == ["legacy"]


def test_utility_anchor_requires_logits_and_samples_only_innovation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(
        anchor_origin=True,
        utility=True,
        optical_innovation_scale=0.0,
        optical_correction_scale=0.0,
    ).eval()
    model.set_optical_anchor_band_scales((0.2, 0.0, 0.0))
    valid = torch.ones(1, 1, 32, 32)
    pyramid = model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    base = torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError, match="requires reliability logits"):
        model.id_bridge_anchor_detail(pyramid, base, SENTINEL2)
    with pytest.raises(ValueError, match="must be B3HW"):
        model.id_bridge_anchor_gains(torch.zeros(1, 2, 8, 8))
    with pytest.raises(ValueError, match="match the latent grid"):
        model.id_bridge_anchor_detail(
            pyramid, base, SENTINEL2, reliability_logits=torch.zeros(1, 3, 7, 8)
        )

    with torch.no_grad():
        model.id_bridge_origin.output_heads["optical"][-1].bias[:48].fill_(2.0)
    mu, correction, _, _, logits = model.predict_id_bridge_origin_components(
        pyramid, base, SENTINEL2
    )
    assert int(torch.count_nonzero(correction)) > 0
    torch.testing.assert_close(mu, torch.zeros_like(mu))
    utility_anchor = model.id_bridge_anchor_detail(
        pyramid, base, SENTINEL2, reliability_logits=logits
    )
    assert int(torch.count_nonzero(utility_anchor)) > 0
    logits_for_gradient = logits.detach().requires_grad_()
    model.id_bridge_anchor_detail(
        pyramid, base, SENTINEL2, reliability_logits=logits_for_gradient
    ).square().mean().backward()
    assert logits_for_gradient.grad is not None
    assert int(torch.count_nonzero(logits_for_gradient.grad)) > 0

    first = model.sample_id_bridge_residual(pyramid, base, SENTINEL2, seed=17)
    second = model.sample_id_bridge_residual(pyramid, base, SENTINEL2, seed=18)
    assert isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
    torch.testing.assert_close(first, torch.zeros_like(first), atol=1e-7, rtol=0.0)
    torch.testing.assert_close(second, torch.zeros_like(second), atol=1e-7, rtol=0.0)

    sar_pyramid = model.encode(torch.rand(1, 10, 32, 32), SENTINEL2, valid)
    sar_base = torch.randn(1, 2, 32, 32)
    sar_first = model.sample_id_bridge_residual(sar_pyramid, sar_base, SENTINEL1, seed=17)
    sar_second = model.sample_id_bridge_residual(sar_pyramid, sar_base, SENTINEL1, seed=18)
    assert isinstance(sar_first, torch.Tensor) and isinstance(sar_second, torch.Tensor)
    assert not torch.equal(sar_first, sar_second)

    physical = torch.zeros(1, 10, 32, 32)
    physical[:, [2, 1, 0]] = base
    log_variance = torch.zeros_like(physical)

    def physical_stub(*_args: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor, object]:
        return physical, log_variance, pyramid

    monkeypatch.setattr(model, "physical", physical_stub)
    observation = Observation(
        torch.randn(2, 32, 32), SENTINEL1, dt.date(2020, 1, 2), orbit="ascending"
    )
    result = translate(
        model, [observation], TargetRequest(SENTINEL2), "visual", num_samples=2, seed=17
    )
    assert result.deterministic_detail is not None
    assert result.stochastic_residual is not None
    assert int(torch.count_nonzero(result.deterministic_detail)) > 0
    torch.testing.assert_close(
        result.stochastic_residual,
        torch.zeros_like(result.stochastic_residual),
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(result.samples[0], result.samples[1])
    expected = model.compose_visual(
        physical[:, [2, 1, 0]],
        result.deterministic_detail,
        torch.zeros_like(result.stochastic_residual),
        "optical",
    )
    assert isinstance(expected, torch.Tensor)
    torch.testing.assert_close(result.samples[0], expected)


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


def test_anchor_gain_target_recovers_single_basis_gain_and_fallbacks() -> None:
    height = width = 32
    coordinates = torch.arange(height).view(height, 1) + torch.arange(width).view(1, width)
    checkerboard = (coordinates.remainder(2).mul(2).sub(1)).float()
    fine = checkerboard.view(1, 1, height, width).expand(1, 3, -1, -1).clone()
    zeros = torch.zeros_like(fine)
    valid = torch.ones(1, 1, height, width)
    full_residual = 3.0 * highpass(fine)

    target = anchor_gain_target((fine, zeros, zeros), full_residual, valid)
    assert target.shape == (1, 3, height // 4, width // 4)
    torch.testing.assert_close(
        target[:, :1, 1:-1, 1:-1],
        torch.full_like(target[:, :1, 1:-1, 1:-1], 0.75),
        atol=2e-5,
        rtol=2e-5,
    )
    assert int(torch.count_nonzero(target[:, 1:])) == 0

    default = anchor_gain_target((zeros, zeros, zeros), zeros, valid)
    torch.testing.assert_close(default[:, :1], torch.full_like(default[:, :1], 0.5))
    assert int(torch.count_nonzero(default[:, 1:])) == 0

    partial_valid = valid.clone()
    partial_valid[..., :4, :4] = 0.0
    partial = anchor_gain_target((fine, zeros, zeros), full_residual, partial_valid)
    torch.testing.assert_close(partial[:, :1, :1, :1], torch.full_like(partial[:, :1, :1, :1], 0.5))
    assert int(torch.count_nonzero(partial[:, 1:, :1, :1])) == 0

    with pytest.raises(ValueError, match="maximum_gain"):
        anchor_gain_target((fine, zeros, zeros), full_residual, valid, maximum_gain=0.0)
    with pytest.raises(ValueError, match="exactly three"):
        anchor_gain_target((fine, zeros), full_residual, valid)  # type: ignore[arg-type]


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


def test_id_bridge_visual_residual_applies_one_amplitude_scale(
    tiny_model: SentinelV3, monkeypatch: pytest.MonkeyPatch
) -> None:
    pyramid = tiny_model.encode(
        torch.randn(1, 2, 32, 32), SENTINEL1, torch.ones(1, 1, 32, 32)
    )
    base = torch.rand(1, 3, 32, 32)
    detail = torch.zeros_like(base)

    def id_sample(
        _pyramid: object,
        physical: torch.Tensor,
        _target: object,
        *,
        seed: int,
        steps: int | None,
        return_origin: bool = False,
    ) -> torch.Tensor:
        assert physical is base and steps is None and not return_origin
        generator = torch.Generator(device=physical.device).manual_seed(seed)
        return torch.randn(physical.shape, generator=generator, device=physical.device)

    tiny_model.config.id_bridge_enabled = True
    monkeypatch.setattr(tiny_model, "sample_id_bridge_residual", id_sample)
    tiny_model.set_amplitude_scale("optical", 1.0)
    reference = tiny_model.sample_visual_residual(pyramid, SENTINEL2, base, detail, seed=19)
    tiny_model.set_amplitude_scale("optical", 0.0)
    suppressed = tiny_model.sample_visual_residual(pyramid, SENTINEL2, base, detail, seed=19)
    tiny_model.set_amplitude_scale("optical", 0.37)
    scaled = tiny_model.sample_visual_residual(pyramid, SENTINEL2, base, detail, seed=19)
    torch.testing.assert_close(suppressed, torch.zeros_like(reference), atol=0.0, rtol=0.0)
    torch.testing.assert_close(scaled, reference * tiny_model.optical_alpha_scale)

    def legacy_sample(
        _pyramid: object,
        _target: object,
        shape: tuple[int, int, int, int],
        *,
        seed: int,
        steps: int | None,
        bridge_anchor: torch.Tensor | None,
    ) -> torch.Tensor:
        assert shape == tuple(base.shape)
        assert seed == 19 and steps is None and bridge_anchor is detail
        return torch.full_like(base, 0.2)

    tiny_model.config.id_bridge_enabled = False
    monkeypatch.setattr(tiny_model, "sample_residual", legacy_sample)
    legacy = tiny_model.sample_visual_residual(pyramid, SENTINEL2, base, detail, seed=19)
    torch.testing.assert_close(legacy, torch.full_like(base, 0.2))


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
        "residual_dit.id_bridge_field_projection.",
        "residual_dit.id_bridge_anchor_projection.",
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
        "anchor_utility": False,
        "phase_identifiability": False,
        "optical_only": False,
        "optical_mid_basis_scale": 0.15,
        "optical_coarse_basis_scale": 0.05,
        "optical_innovation_scale": 1.0,
        "sar_innovation_scale": 1.0,
        "optical_innovation_band_scales": (1.0, 1.0, 1.0),
        "optical_correction_scale": 1.0,
        "sar_correction_scale": 1.0,
        "phase_transport_enabled": False,
        "phase_transport_hidden": 128,
        "phase_transport_gain_caps": (0.5, 0.25, 0.1),
        "phase_transport_carrier_gain_caps": (0.5, 0.25, 0.1),
        "phase_transport_offset_caps_px": (0.5, 0.5, 0.5),
        "phase_transport_initial_gate": 0.02,
        "phase_transport_null_calibrated": False,
        "phase_transport_null_quantile": 0.75,
        "phase_transport_support_epsilon": 0.01,
        "phase_transport_carrier_mode": "physical_gain",
        "phase_transport_carrier_support_mode": "continuous",
        "phase_transport_carrier_basis_trainable": True,
        "phase_transport_detail_utility_enabled": False,
        "phase_transport_detail_scale_cap": 2.0,
        "antithetic_weight": 0.0,
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
        "anchor_utility": False,
        "phase_identifiability": False,
        "optical_only": False,
        "optical_mid_basis_scale": 0.15,
        "optical_coarse_basis_scale": 0.05,
        "optical_innovation_scale": 0.0,
        "sar_innovation_scale": 1.0,
        "optical_innovation_band_scales": (1.0, 1.0, 1.0),
        "optical_correction_scale": 1.0,
        "sar_correction_scale": 1.0,
        "phase_transport_enabled": False,
        "phase_transport_hidden": 128,
        "phase_transport_gain_caps": (0.5, 0.25, 0.1),
        "phase_transport_carrier_gain_caps": (0.5, 0.25, 0.1),
        "phase_transport_offset_caps_px": (0.5, 0.5, 0.5),
        "phase_transport_initial_gate": 0.02,
        "phase_transport_null_calibrated": False,
        "phase_transport_null_quantile": 0.75,
        "phase_transport_support_epsilon": 0.01,
        "phase_transport_carrier_mode": "physical_gain",
        "phase_transport_carrier_support_mode": "continuous",
        "phase_transport_carrier_basis_trainable": True,
        "phase_transport_detail_utility_enabled": False,
        "phase_transport_detail_scale_cap": 2.0,
        "antithetic_weight": 0.0,
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


def test_utility_id_bridge_long_gap_has_exact_zero_residual_gradient() -> None:
    model = _haar_model(anchor_origin=True, utility=True)
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


def test_id_utility_trains_only_optical_origin_parameters() -> None:
    model = _utility_codec_model()
    model.set_optical_anchor_band_scales((0.2, 0.0, 0.0))
    _set_trainable(model, "id_utility")
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("id_bridge_origin.") for name in trainable)
    optimizer = _optimizer(
        model,
        {"learning_rate": 1e-3, "encoder_learning_rate": 1e-3, "weight_decay": 0.0},
        "id_utility",
        torch.device("cpu"),
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert optimizer_ids == trainable_ids

    objective = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=4,
        flow_visual_perceptual_weight=0.0,
    )
    torch.testing.assert_close(
        objective._id_utility_assignments(3, torch.device("cpu")), torch.zeros(3, dtype=torch.long)
    )
    loss, metrics = objective(_batch(), "id_utility")
    assert torch.isfinite(loss)
    assert "sar2opt/q" in metrics
    assert "sar2opt/oracle_q" in metrics
    assert "sar2opt/anchor_gain_fine" in metrics
    assert not any(name.startswith("opt2sar/") for name in metrics)
    assert float(metrics["sar2opt/anchor_gain_fine"]) == pytest.approx(1.0)
    assert float(metrics["sar2opt/anchor_gain_mid"]) == pytest.approx(math.exp(-4.0))
    assert float(metrics["sar2opt/anchor_gain_coarse"]) == pytest.approx(math.exp(-4.0))
    loss.backward()
    utility_head = model.id_bridge_origin.anchor_utility_head[-1]
    assert utility_head.weight.grad is not None
    assert utility_head.bias.grad is not None
    assert int(torch.count_nonzero(utility_head.weight.grad)) > 0
    assert int(torch.count_nonzero(utility_head.bias.grad)) > 0
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    assert all(parameter.grad is None for parameter in model.detail_head.parameters())
    assert all(parameter.grad is None for parameter in model.residual_dit.parameters())


def test_id_utility_long_gap_has_exact_zero_origin_gradient() -> None:
    model = _utility_codec_model()
    _set_trainable(model, "id_utility")
    objective = JointObjective(model, [0.5, 0.5], flow_rollout_every=1)
    loss, _ = objective(_batch(delta_days=2), "id_utility")
    loss.backward()
    assert float(loss.detach()) == 0.0
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) == 0.0


def test_phase_transport_trains_only_the_observable_head_and_keeps_long_gaps_zero() -> None:
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
    )
    _set_trainable(model, "phase_transport")
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable and all(name.startswith("phase_transport_head.") for name in trainable)
    optimizer = _optimizer(
        model,
        {"learning_rate": 1e-3, "encoder_learning_rate": 1e-3, "weight_decay": 0.0},
        "phase_transport",
        torch.device("cpu"),
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert optimizer_ids == trainable_ids
    objective = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=4,
        flow_visual_perceptual_weight=0.0,
    )
    torch.testing.assert_close(
        objective._phase_transport_assignments(3, torch.device("cpu")),
        torch.zeros(3, dtype=torch.long),
    )
    objective.set_progress(1, 100)
    loss, metrics = objective(_batch(), "phase_transport")
    assert bool(torch.isfinite(loss))
    for name in (
        "sar2opt/hf_reconstruction",
        "sar2opt/hf_gradient",
        "sar2opt/hf_spectrum",
        "sar2opt/low_frequency",
        "sar2opt/rmse_hinge",
        "sar2opt/phase_alignment",
        "sar2opt/gain_signed_mean",
        "sar2opt/gain_utility",
        "sar2opt/gain_fine",
        "sar2opt/gain_mid",
        "sar2opt/gain_coarse",
        "sar2opt/gate_fine",
        "sar2opt/gate_mid",
        "sar2opt/gate_coarse",
        "sar2opt/oracle_gate_fine",
        "sar2opt/oracle_gate_mid",
        "sar2opt/oracle_gate_coarse",
        "sar2opt/oracle_active_fraction",
        "sar2opt/oracle_supported_fraction",
        "sar2opt/offset_px_abs_mean",
        "sar2opt/coherence_fine",
        "sar2opt/coherence_mid",
        "sar2opt/coherence_coarse",
        "sar2opt/low_frequency_leakage",
    ):
        assert name in metrics and bool(torch.isfinite(metrics[name]))
    assert float(metrics["sar2opt/gain_signed_mean"]) >= 0.0
    assert float(metrics["sar2opt/gate_fine"]) == pytest.approx(0.02, abs=1e-7)
    assert not any(name.startswith("opt2sar/") for name in metrics)
    loss.backward()
    output = model.phase_transport_head.output[-1]
    assert output.weight.grad is not None and int(torch.count_nonzero(output.weight.grad)) > 0
    assert output.bias.grad is not None and int(torch.count_nonzero(output.bias.grad)) > 0
    source_projection = model.phase_transport_head.source_phase_projection
    assert source_projection.weight.grad is not None
    assert source_projection.bias.grad is not None
    for parameter in model.phase_transport_head.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("phase_transport_head.")
    )

    model.zero_grad(set_to_none=True)
    zero_loss, _ = objective(_batch(delta_days=2), "phase_transport")
    zero_loss.backward()
    assert float(zero_loss.detach()) == 0.0
    assert all(
        parameter.grad is not None and int(torch.count_nonzero(parameter.grad)) == 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_null_calibrated_phase_transport_trains_without_offsets_and_keeps_long_gaps_zero() -> None:
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
        phase_transport_null_calibrated=True,
    )
    _set_trainable(model, "phase_transport")
    assert all(
        name.startswith("phase_transport_head.")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    assert model.phase_transport_head.output[-1].out_channels == 3
    objective = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=4,
        flow_visual_perceptual_weight=0.0,
        phase_transport_utility_weight=0.25,
    )
    objective.set_progress(1, 100)
    loss, metrics = objective(_batch(), "phase_transport")
    assert bool(torch.isfinite(loss))
    required = (
        "gain_utility",
        "null_level_fine",
        "null_level_mid",
        "null_level_coarse",
        "gain_support_fine",
        "gain_support_mid",
        "gain_support_coarse",
        "effective_gate_fine",
        "effective_gate_mid",
        "effective_gate_coarse",
        "support_active_fraction",
    )
    for name in required:
        value = metrics[f"sar2opt/{name}"]
        assert bool(torch.isfinite(value))
    assert not any("offset" in name for name in metrics)
    loss.backward()
    output = model.phase_transport_head.output[-1]
    assert output.weight.grad is not None and bool(torch.isfinite(output.weight.grad).all())
    assert output.bias.grad is not None and bool(torch.isfinite(output.bias.grad).all())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("phase_transport_head.")
    )

    model.zero_grad(set_to_none=True)
    zero_loss, _ = objective(_batch(delta_days=2), "phase_transport")
    zero_loss.backward()
    assert float(zero_loss.detach()) == 0.0
    assert all(
        parameter.grad is not None and int(torch.count_nonzero(parameter.grad)) == 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_spuf_zero_init_is_compatible_with_the_frozen_phase_baseline() -> None:
    legacy = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
        phase_transport_null_calibrated=True,
    ).eval()
    spuf = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
        phase_transport_null_calibrated=True,
        phase_transport_detail_utility_enabled=True,
        phase_transport_detail_scale_cap=2.0,
    ).eval()
    initial_head = {
        name: parameter.detach().clone()
        for name, parameter in spuf.phase_transport_head.detail_utility_head.named_parameters()
    }
    loaded, missing = training_module._load_compatible_state(
        spuf, {name: value.detach().clone() for name, value in legacy.state_dict().items()}
    )
    assert loaded > 0 and missing > 0
    assert not hasattr(legacy.phase_transport_head, "detail_utility_adapter")
    assert torch.equal(
        spuf.phase_transport_head.detail_utility_head.weight,
        initial_head["weight"],
    )
    assert torch.equal(spuf.phase_transport_head.detail_utility_head.bias, initial_head["bias"])

    valid = torch.ones(1, 1, 32, 32)
    pyramid = legacy.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    base = torch.rand(1, 3, 32, 32)
    legacy_delta, _ = legacy.phase_transport_delta(pyramid, base, SENTINEL2)
    spuf_delta, diagnostics = spuf.phase_transport_delta(pyramid, base, SENTINEL2)
    torch.testing.assert_close(diagnostics["detail_scale"], torch.ones_like(diagnostics["detail_scale"]))
    torch.testing.assert_close(diagnostics["detail_scale_raw"], torch.zeros_like(diagnostics["detail_scale_raw"]))
    torch.testing.assert_close(spuf_delta, legacy_delta, atol=0.0, rtol=0.0)
    legacy_detail = legacy.phase_transport_detail(pyramid, base, SENTINEL2)
    spuf_detail = spuf.phase_transport_detail(pyramid, base, SENTINEL2)
    spuf_visual = spuf.visual_detail(pyramid, SENTINEL1, SENTINEL2, (32, 32), base)
    torch.testing.assert_close(spuf_detail, legacy_detail, atol=0.0, rtol=0.0)
    torch.testing.assert_close(spuf_visual, legacy_detail, atol=0.0, rtol=0.0)
    metadata = spuf.residual_state_metadata()
    assert metadata["phase_transport_detail_utility_enabled"] is True
    assert metadata["phase_transport_detail_scale_cap"] == pytest.approx(2.0)


def test_spuf_composition_scales_existing_detail_bands_without_low_frequency_leakage() -> None:
    height = width = 32
    rows = torch.arange(height).view(height, 1)
    columns = torch.arange(width).view(1, width)
    checkerboard = rows.add(columns).remainder(2).mul(2).sub(1).float()
    base_detail = checkerboard.view(1, 1, height, width).expand(1, 3, -1, -1).clone()
    scales_one = torch.ones(1, 3, 8, 8)
    detail_one, correction_one = SentinelV3.compose_phase_transport_detail_utility(
        base_detail, scales_one
    )
    torch.testing.assert_close(detail_one, base_detail, atol=0.0, rtol=0.0)
    torch.testing.assert_close(correction_one, torch.zeros_like(correction_one), atol=0.0, rtol=0.0)

    scales_zero = torch.zeros_like(scales_one)
    scales_two = torch.full_like(scales_one, 2.0)
    _, correction_zero = SentinelV3.compose_phase_transport_detail_utility(
        base_detail, scales_zero
    )
    _, correction_two = SentinelV3.compose_phase_transport_detail_utility(
        base_detail, scales_two
    )
    expected_positive = highpass(sum(frequency_bands(base_detail, levels=3)))
    torch.testing.assert_close(correction_two, expected_positive, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(correction_zero, -expected_positive, atol=1e-6, rtol=1e-6)
    assert float((correction_two * base_detail).mean()) > 0.0
    assert float((correction_zero * base_detail).mean()) < 0.0
    assert float(F.avg_pool2d(correction_two, 4, stride=4).abs().amax()) < 1e-6
    assert float(F.avg_pool2d(correction_zero, 4, stride=4).abs().amax()) < 1e-6


def test_spuf_oracle_smoothing_and_scale_diagnostics_are_finite() -> None:
    oracle = torch.zeros(1, 3, 5, 5)
    oracle[:, :, 2, 2] = torch.tensor((0.5, 1.0, 1.5)).view(1, 3)
    strict_valid = torch.ones(1, 1, 5, 5)
    smoothed = JointObjective._smoothed_detail_utility_oracle(oracle, strict_valid, 5)
    expected = F.avg_pool2d(oracle, 5, stride=1, padding=2, count_include_pad=False)
    torch.testing.assert_close(smoothed, expected)
    assert not smoothed.requires_grad
    torch.testing.assert_close(
        JointObjective._smoothed_detail_utility_oracle(oracle, torch.zeros_like(strict_valid), 5),
        torch.zeros_like(oracle),
    )

    oracle = torch.tensor(
        [
            [
                [[0.2, 0.4], [0.6, 0.8]],
                [[0.3, 0.5], [0.7, 0.9]],
                [[0.1, 0.6], [1.1, 1.6]],
            ]
        ]
    )
    strict_valid = torch.ones(1, 1, 2, 2)
    weights = torch.ones(1)
    perfect = oracle.clone().requires_grad_()
    anti = 2.0 - oracle
    perfect_metrics = JointObjective._detail_scale_diagnostics(
        perfect, oracle, strict_valid, weights
    )
    anti_metrics = JointObjective._detail_scale_diagnostics(anti, oracle, strict_valid, weights)
    empty_metrics = JointObjective._detail_scale_diagnostics(
        perfect, oracle, torch.zeros_like(strict_valid), weights
    )
    for name in ("fine", "mid", "coarse"):
        assert float(perfect_metrics[f"detail_scale_mae_{name}"]) == pytest.approx(0.0)
        assert float(perfect_metrics[f"detail_scale_corr_{name}"]) == pytest.approx(1.0)
        assert float(anti_metrics[f"detail_scale_mae_{name}"]) > 0.0
        assert float(anti_metrics[f"detail_scale_corr_{name}"]) == pytest.approx(-1.0)
        assert float(empty_metrics[f"detail_scale_predicted_mean_{name}"]) == pytest.approx(0.0)
        assert float(empty_metrics[f"detail_scale_oracle_mean_{name}"]) == pytest.approx(0.0)
        assert float(empty_metrics[f"detail_scale_mae_{name}"]) == pytest.approx(0.0)
        assert float(empty_metrics[f"detail_scale_corr_{name}"]) == pytest.approx(0.0)
    assert all(bool(torch.isfinite(value)) for value in empty_metrics.values())
    assert not any(value.requires_grad for value in perfect_metrics.values())


def test_spuf_phase_transport_trains_only_the_scale_head_and_keeps_long_gaps_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
        phase_transport_null_calibrated=True,
        phase_transport_detail_utility_enabled=True,
        phase_transport_detail_scale_cap=2.0,
    )
    _set_trainable(model, "phase_transport")
    prefixes = (
        "phase_transport_head.detail_utility_adapter.",
        "phase_transport_head.detail_utility_head.",
    )
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all(name.startswith(prefixes) for name in trainable)
    assert not model.phase_transport_head.output[-1].weight.requires_grad
    assert not model.phase_transport_head.source_phase_projection.weight.requires_grad
    optimizer = _optimizer(
        model,
        {"learning_rate": 1e-3, "encoder_learning_rate": 1e-3, "weight_decay": 0.0},
        "phase_transport",
        torch.device("cpu"),
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_ids == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }

    valid = torch.ones(1, 1, 32, 32)
    pyramid = model.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    base = torch.rand(1, 3, 32, 32)
    composition_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_composition = SentinelV3.compose_phase_transport_detail_utility

    def capture_composition(
        base_detail: torch.Tensor, detail_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        composition_calls.append((base_detail, detail_scale))
        return original_composition(base_detail, detail_scale)

    monkeypatch.setattr(
        SentinelV3,
        "compose_phase_transport_detail_utility",
        staticmethod(capture_composition),
    )
    ema = EMA(model, decay=0.9)
    with ema.apply_to(model):
        detail, anchor, delta, diagnostics = model.phase_transport_detail(
            pyramid, base, SENTINEL2, return_diagnostics=True
        )
    torch.testing.assert_close(diagnostics["detail_scale"], torch.ones_like(diagnostics["detail_scale"]))
    torch.testing.assert_close(detail, anchor + delta, atol=0.0, rtol=0.0)

    before_step = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    objective = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=4,
        flow_visual_perceptual_weight=0.0,
        phase_transport_utility_weight=1.0,
        phase_transport_detail_utility_kernel=5,
    )
    objective.set_progress(1, 100)
    loss, metrics = objective(_batch(), "phase_transport")
    assert bool(torch.isfinite(loss))
    assert len(composition_calls) >= 2
    for name in (
        "detail_utility",
        "detail_scale_predicted_mean_fine",
        "detail_scale_oracle_mean_fine",
        "detail_scale_mae_fine",
        "detail_scale_corr_fine",
        "detail_utility_correction_rms",
        "detail_utility_low_frequency_leakage",
    ):
        metric_name = f"sar2opt/{name}"
        assert metric_name in metrics and bool(torch.isfinite(metrics[metric_name]))
    loss.backward()
    assert model.phase_transport_head.detail_utility_head.bias.grad is not None
    assert int(torch.count_nonzero(model.phase_transport_head.detail_utility_head.bias.grad)) > 0
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith(prefixes)
    )
    optimizer.step()
    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), before_step[name])
    }
    assert changed and all(name.startswith(prefixes) for name in changed)
    payload = _checkpoint_payload(
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=_scheduler(optimizer, warmup=0, maximum=1),
        stage="phase_transport",
        step=1,
        rank_states=[],
        config={},
        validation_protocol_hash="test",
        best_metrics={},
        quality_gates={},
    )
    residual_state = payload["residual_state"]  # type: ignore[index]
    assert residual_state["phase_transport_detail_utility_enabled"] is True  # type: ignore[index]
    assert residual_state["phase_transport_detail_scale_cap"] == pytest.approx(2.0)  # type: ignore[index]

    model.zero_grad(set_to_none=True)
    zero_loss, _ = objective(_batch(delta_days=2), "phase_transport")
    zero_loss.backward()
    assert float(zero_loss.detach()) == 0.0
    assert all(
        parameter.grad is not None and int(torch.count_nonzero(parameter.grad)) == 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_orthogonal_source_phase_transport_trains_only_head_and_keeps_long_gaps_zero() -> None:
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
        phase_transport_null_calibrated=True,
        phase_transport_carrier_mode="orthogonal_source",
        phase_transport_carrier_gain_caps=(0.125, 0.0625, 0.025),
        phase_transport_carrier_support_mode="binary_exceedance",
    )
    _set_trainable(model, "phase_transport")
    assert model.config.phase_transport_carrier_basis_trainable is True
    carrier_prefixes = (
        "phase_transport_head.carrier_source_phase_projection.",
        "phase_transport_head.carrier_adapter.",
        "phase_transport_head.carrier_head.",
    )
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable and all(name.startswith(carrier_prefixes) for name in trainable)
    assert not model.phase_transport_head.source_phase_projection.weight.requires_grad
    assert not model.phase_transport_head.output[-1].weight.requires_grad
    optimizer = _optimizer(
        model,
        {"learning_rate": 1e-3, "encoder_learning_rate": 1e-3, "weight_decay": 0.0},
        "phase_transport",
        torch.device("cpu"),
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_ids == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    before_step = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    objective = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=4,
        flow_visual_perceptual_weight=0.0,
        phase_transport_utility_weight=0.25,
        phase_transport_signed_alignment_weight=0.05,
    )
    objective.set_progress(1, 100)
    loss, metrics = objective(_batch(), "phase_transport")
    assert bool(torch.isfinite(loss))
    for name in (
        "signed_phase_alignment",
        "carrier_rms",
        "carrier_orthogonality",
        "carrier_oracle_fine",
        "carrier_oracle_abs_fine",
        "carrier_oracle_active_fraction",
        "carrier_oracle_supported_fraction",
        "carrier_gate_fine",
        "carrier_gate_abs_fine",
        "carrier_effective_fine",
        "carrier_effective_abs_fine",
        "carrier_support_fine",
        "carrier_alignment_active_fraction",
        "carrier_delta_rms",
    ):
        assert f"sar2opt/{name}" in metrics
        assert bool(torch.isfinite(metrics[f"sar2opt/{name}"]))
    loss.backward()
    source_projection = model.phase_transport_head.carrier_source_phase_projection
    assert source_projection.weight.grad is not None
    assert int(torch.count_nonzero(source_projection.weight.grad)) > 0
    carrier_head = model.phase_transport_head.carrier_head
    assert carrier_head.bias.grad is not None
    assert int(torch.count_nonzero(carrier_head.bias.grad)) > 0
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith(carrier_prefixes)
    )
    optimizer.step()
    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), before_step[name])
    }
    assert changed and all(name.startswith(carrier_prefixes) for name in changed)

    model.zero_grad(set_to_none=True)
    zero_loss, _ = objective(_batch(delta_days=2), "phase_transport")
    zero_loss.backward()
    assert float(zero_loss.detach()) == 0.0
    assert all(
        parameter.grad is not None and int(torch.count_nonzero(parameter.grad)) == 0
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_stationary_carrier_phase_transport_freezes_basis_and_preserves_zero_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bnes = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
        phase_transport_null_calibrated=True,
        phase_transport_carrier_mode="orthogonal_source",
        phase_transport_carrier_gain_caps=(1.0, 0.5, 0.2),
        phase_transport_carrier_support_mode="binary_exceedance",
    ).eval()
    stationary = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
        phase_transport_null_calibrated=True,
        phase_transport_carrier_mode="orthogonal_source",
        phase_transport_carrier_gain_caps=(1.0, 0.5, 0.2),
        phase_transport_carrier_support_mode="binary_exceedance",
        phase_transport_carrier_basis_trainable=False,
    ).eval()
    stationary.load_state_dict(bnes.state_dict())
    _set_trainable(stationary, "phase_transport")
    prefixes = (
        "phase_transport_head.carrier_adapter.",
        "phase_transport_head.carrier_head.",
    )
    trainable = {
        name for name, parameter in stationary.named_parameters() if parameter.requires_grad
    }
    assert trainable and all(name.startswith(prefixes) for name in trainable)
    assert not stationary.phase_transport_head.carrier_source_phase_projection.weight.requires_grad
    optimizer = _optimizer(
        stationary,
        {"learning_rate": 1e-3, "encoder_learning_rate": 1e-3, "weight_decay": 0.0},
        "phase_transport",
        torch.device("cpu"),
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_ids == {
        id(parameter) for parameter in stationary.parameters() if parameter.requires_grad
    }

    valid = torch.ones(1, 1, 32, 32)
    pyramid = bnes.encode(torch.randn(1, 2, 32, 32), SENTINEL1, valid)
    base = torch.rand(1, 3, 32, 32)
    baseline_delta, _ = bnes.phase_transport_delta(pyramid, base, SENTINEL2)
    ema = EMA(stationary, decay=0.9)
    with ema.apply_to(stationary):
        stationary_delta, diagnostics = stationary.phase_transport_delta(pyramid, base, SENTINEL2)
    torch.testing.assert_close(diagnostics["carrier_delta"], torch.zeros_like(stationary_delta))
    torch.testing.assert_close(stationary_delta, diagnostics["parallel_delta"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(stationary_delta, baseline_delta, atol=0.0, rtol=0.0)
    assert stationary.residual_state_metadata()["phase_transport_carrier_basis_trainable"] is False

    def support_one(
        _source_phase: torch.Tensor,
        _physical_bands: torch.Tensor,
        coherence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(coherence)
        return zeros, torch.ones_like(coherence), zeros

    monkeypatch.setattr(stationary.phase_transport_head, "_carrier_null_calibrated_support", support_one)
    before_step = {
        name: parameter.detach().clone() for name, parameter in stationary.named_parameters()
    }
    objective = JointObjective(
        stationary,
        [0.5, 0.5],
        flow_rollout_every=4,
        flow_visual_perceptual_weight=0.0,
        phase_transport_utility_weight=1.0,
        phase_transport_signed_alignment_weight=0.0,
    )
    objective.set_progress(1, 100)
    loss, metrics = objective(_batch(), "phase_transport")
    assert bool(torch.isfinite(loss))
    assert "sar2opt/carrier_coeff_corr_fine" in metrics
    loss.backward()
    assert stationary.phase_transport_head.carrier_source_phase_projection.weight.grad is None
    assert all(
        parameter.grad is None
        for name, parameter in stationary.named_parameters()
        if not name.startswith(prefixes)
    )
    optimizer.step()
    changed = {
        name
        for name, parameter in stationary.named_parameters()
        if not torch.equal(parameter.detach(), before_step[name])
    }
    assert changed and all(name.startswith(prefixes) for name in changed)

    stationary.zero_grad(set_to_none=True)
    zero_loss, _ = objective(_batch(delta_days=2), "phase_transport")
    zero_loss.backward()
    assert float(zero_loss.detach()) == 0.0
    assert all(
        parameter.grad is not None and int(torch.count_nonzero(parameter.grad)) == 0
        for parameter in stationary.parameters()
        if parameter.requires_grad
    )


def test_carrier_coefficient_diagnostics_cover_perfect_anti_and_empty_support() -> None:
    oracle = torch.tensor(
        [
            [
                [[0.2, -0.4], [0.6, -0.8]],
                [[0.1, -0.3], [0.5, -0.7]],
                [[0.15, -0.25], [0.45, -0.65]],
            ]
        ]
    )
    support = torch.ones_like(oracle)
    support[..., 0, 0] = 0.0
    strict_valid = torch.ones(1, 1, 2, 2)
    strict_valid[..., 0, 1] = 0.0
    weights = torch.ones(1)
    target = oracle * support
    perfect = target.clone()
    perfect[..., 0, 0] = 99.0
    perfect[..., 0, 1] = -99.0
    perfect_metrics = JointObjective._carrier_coefficient_diagnostics(
        perfect, oracle, support, strict_valid, weights
    )
    anti_metrics = JointObjective._carrier_coefficient_diagnostics(
        -target, oracle, support, strict_valid, weights
    )
    empty_metrics = JointObjective._carrier_coefficient_diagnostics(
        perfect, oracle, torch.zeros_like(support), strict_valid, weights
    )
    for name in ("fine", "mid", "coarse"):
        assert float(perfect_metrics[f"carrier_coeff_mae_{name}"]) == pytest.approx(0.0)
        assert float(perfect_metrics[f"carrier_coeff_corr_{name}"]) == pytest.approx(1.0)
        assert float(perfect_metrics[f"carrier_sign_accuracy_{name}"]) == pytest.approx(1.0)
        assert float(anti_metrics[f"carrier_coeff_mae_{name}"]) > 0.0
        assert float(anti_metrics[f"carrier_coeff_corr_{name}"]) == pytest.approx(-1.0)
        assert float(anti_metrics[f"carrier_sign_accuracy_{name}"]) == pytest.approx(0.0)
        assert float(empty_metrics[f"carrier_coeff_mae_{name}"]) == pytest.approx(0.0)
        assert float(empty_metrics[f"carrier_coeff_corr_{name}"]) == pytest.approx(0.0)
        assert float(empty_metrics[f"carrier_sign_accuracy_{name}"]) == pytest.approx(0.0)
    assert all(bool(torch.isfinite(value)) for value in empty_metrics.values())


def test_orthogonal_source_phase_transport_uses_detached_carrier_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
        phase_transport_null_calibrated=True,
        phase_transport_carrier_mode="orthogonal_source",
        phase_transport_carrier_gain_caps=(0.125, 0.0625, 0.025),
        phase_transport_carrier_support_mode="binary_exceedance",
    )
    _set_trainable(model, "phase_transport")
    objective = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=4,
        flow_visual_perceptual_weight=0.0,
        phase_transport_utility_weight=0.25,
        phase_transport_signed_alignment_weight=0.05,
    )
    captured: dict[str, torch.Tensor] = {}
    original_delta = model.phase_transport_delta
    original_context = objective._physical_context
    original_anchor = model.id_bridge_anchor_detail

    def capture_delta(*args: object, **kwargs: object) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        delta, diagnostics = original_delta(*args, **kwargs)  # type: ignore[arg-type]
        captured["carrier_components"] = diagnostics["carrier_components"]
        captured["parallel_delta"] = diagnostics["parallel_delta"]
        return delta, diagnostics

    def capture_context(*args: object, **kwargs: object) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]
    ]:
        context = original_context(*args, **kwargs)  # type: ignore[arg-type]
        captured["target"] = context[0]
        captured["base"] = context[1]
        captured["valid"] = context[2]
        return context

    def capture_anchor(*args: object, **kwargs: object) -> torch.Tensor:
        anchor = original_anchor(*args, **kwargs)  # type: ignore[arg-type]
        captured["protected_anchor"] = anchor
        return anchor

    oracle_inputs: list[torch.Tensor] = []
    oracle_residuals: list[torch.Tensor] = []
    oracle_caps: list[tuple[float, float, float]] = []
    original_oracle = training_module.phase_transport_signed_coefficient_target

    def capture_oracle(*args: object, **kwargs: object) -> torch.Tensor:
        oracle_inputs.append(args[0])  # type: ignore[arg-type]
        oracle_residuals.append(args[1])  # type: ignore[arg-type]
        oracle_caps.append(tuple(args[3]))  # type: ignore[arg-type]
        return original_oracle(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(model, "phase_transport_delta", capture_delta)
    monkeypatch.setattr(objective, "_physical_context", capture_context)
    monkeypatch.setattr(model, "id_bridge_anchor_detail", capture_anchor)
    monkeypatch.setattr(training_module, "phase_transport_signed_coefficient_target", capture_oracle)
    objective.set_progress(1, 100)
    loss, metrics = objective(_batch(), "phase_transport")
    assert bool(torch.isfinite(loss))
    assert captured["carrier_components"].requires_grad
    assert len(oracle_inputs) == 1
    assert oracle_caps == [(0.125, 0.0625, 0.025)]
    assert not oracle_inputs[0].requires_grad
    torch.testing.assert_close(oracle_inputs[0], captured["carrier_components"].detach())
    expected_residual = (
        highpass(captured["target"] - captured["base"].detach()) * captured["valid"]
        - captured["protected_anchor"]
        - captured["parallel_delta"].detach()
    )
    torch.testing.assert_close(oracle_residuals[0], expected_residual)
    assert float(metrics["sar2opt/carrier_oracle_active"]) == 1.0
    assert not any("carrier_components" in name for name in metrics)


def test_physical_gain_phase_transport_keeps_physical_oracle_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(
        anchor_origin=True,
        phase=True,
        optical_only=True,
        phase_transport=True,
        optical_correction_scale=0.0,
        optical_innovation_band_scales=(0.0, 0.0, 0.0),
        phase_transport_null_calibrated=True,
    )
    _set_trainable(model, "phase_transport")
    objective = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=4,
        flow_visual_perceptual_weight=0.0,
        phase_transport_utility_weight=0.25,
    )
    captured: dict[str, torch.Tensor] = {}
    original_context = objective._physical_context

    def capture_context(*args: object, **kwargs: object) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]
    ]:
        context = original_context(*args, **kwargs)  # type: ignore[arg-type]
        captured["base"] = context[1]
        return context

    oracle_inputs: list[torch.Tensor] = []
    original_oracle = training_module.phase_transport_gain_target

    def capture_oracle(*args: object, **kwargs: object) -> torch.Tensor:
        oracle_inputs.append(args[0])  # type: ignore[arg-type]
        return original_oracle(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(objective, "_physical_context", capture_context)
    monkeypatch.setattr(training_module, "phase_transport_gain_target", capture_oracle)
    objective.set_progress(1, 100)
    _, metrics = objective(_batch(), "phase_transport")
    expected = torch.stack(frequency_bands(captured["base"].detach(), levels=3), dim=1)
    assert len(oracle_inputs) == 1
    torch.testing.assert_close(oracle_inputs[0], expected)
    assert "sar2opt/carrier_oracle_active" not in metrics


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


def test_utility_optical_direction_trains_gain_logits_and_detaches_anchor_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(
        anchor_origin=True,
        utility=True,
        optical_innovation_scale=0.0,
        optical_correction_scale=0.0,
    )
    model.set_optical_anchor_band_scales((0.2, 0.0, 0.0))
    _set_trainable(model, "id_bridge")
    objective = JointObjective(
        model,
        [0.5, 0.5],
        flow_rollout_every=4,
        flow_visual_perceptual_weight=0.0,
    )
    objective.set_progress(1, 10)
    captured_projection_inputs: list[torch.Tensor] = []
    project = model.project_id_bridge_residual

    def record_project(values: torch.Tensor, target: object) -> torch.Tensor:
        captured_projection_inputs.append(values)
        assert target is SENTINEL2
        return project(values, target)  # type: ignore[arg-type]

    monkeypatch.setattr(model, "project_id_bridge_residual", record_project)
    loss, metrics = objective._id_bridge_direction(
        _batch(),
        torch.tensor([0]),
        "sar_view",
        "s2_target",
        SENTINEL1,
        SENTINEL2,
        torch.ones(4),
    )
    assert len(captured_projection_inputs) == 1
    assert not captured_projection_inputs[0].requires_grad
    assert {"anchor_gain", "anchor_gain_fine", "anchor_gain_mid", "anchor_gain_coarse"} <= (
        metrics.keys()
    )
    assert float(metrics["anchor_gain_fine"]) == pytest.approx(1.0)
    assert float(metrics["anchor_gain_mid"]) == pytest.approx(math.exp(-4.0))
    assert float(metrics["anchor_gain_coarse"]) == pytest.approx(math.exp(-4.0))
    loss.backward()
    head = model.id_bridge_origin.anchor_utility_head[-1]
    for gradient in (head.weight.grad, head.bias.grad):
        assert bool(torch.isfinite(gradient).all())
        assert int(torch.count_nonzero(gradient)) > 0
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.decoder.parameters())


def test_utility_sar_direction_keeps_cross_modal_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _haar_model(anchor_origin=True, utility=True)
    _set_trainable(model, "id_bridge")
    objective = JointObjective(model, [0.5, 0.5], flow_rollout_every=4)

    def utility_oracle_called(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("SAR must retain the cross-modal identifiability oracle")

    monkeypatch.setattr(training_module, "anchor_gain_target", utility_oracle_called)
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
    assert "anchor_gain" not in metrics


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
