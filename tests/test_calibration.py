from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from sentinel_v3 import calibration
from sentinel_v3.model import ModelConfig, SentinelV3
from sentinel_v3.sensors import SensorSpec


class _PriorAwareCalibrationModel:
    def __init__(self) -> None:
        self.detail_bases: list[Tensor] = []
        self.visual_calls: list[tuple[str, str]] = []
        self.residual_calls: list[tuple[str, int]] = []
        self.amplitude_scales: list[tuple[str, float]] = []

    def set_amplitude_scale(self, modality: str, value: float) -> None:
        self.amplitude_scales.append((modality, value))

    def physical(
        self,
        source: Tensor,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        valid: Tensor,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor, object]:
        del source_spec, valid, kwargs
        channels = 3 if target_spec.modality == "optical" else 2
        value = 0.2 if target_spec.modality == "optical" else -20.0
        physical = source.new_full((source.shape[0], channels, *source.shape[-2:]), value)
        return physical, torch.zeros_like(physical), object()

    def visual_detail(
        self,
        pyramid: object,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        output_size: tuple[int, int],
        base: Tensor,
    ) -> Tensor:
        del pyramid
        assert output_size == tuple(base.shape[-2:])
        self.detail_bases.append(base.detach().clone())
        self.visual_calls.append((source_spec.modality, target_spec.modality))
        return torch.zeros_like(base)

    def sample_visual_residual(
        self,
        pyramid: object,
        target_spec: SensorSpec,
        base: Tensor,
        detail: Tensor,
        *,
        seed: int,
    ) -> Tensor:
        del pyramid
        assert detail.shape == base.shape
        self.residual_calls.append((target_spec.modality, seed))
        return torch.zeros_like(base)

    def deterministic_detail(self, *args: object, **kwargs: object) -> Tensor:
        del args, kwargs
        raise AssertionError("calibration must use visual_detail")

    def sample_residual(self, *args: object, **kwargs: object) -> Tensor:
        del args, kwargs
        raise AssertionError("calibration must use sample_visual_residual")

    @staticmethod
    def amplitude_scale_name(modality: str) -> str:
        assert modality in {"optical", "sar"}
        return f"calibrated_{modality}_scale"


class _SarBiasCalibrationModel:
    def __init__(self, scene_biases: list[float]) -> None:
        self.scene_biases = scene_biases
        self.sar_physical_calls = 0

    def set_amplitude_scale(self, modality: str, value: float) -> None:
        del modality, value

    def physical(
        self,
        source: Tensor,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        valid: Tensor,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor, object]:
        del source_spec, valid, kwargs
        channels = 3 if target_spec.modality == "optical" else 2
        if target_spec.modality == "optical":
            value = 0.5
        else:
            value = self.scene_biases[self.sar_physical_calls]
            self.sar_physical_calls += 1
        physical = source.new_full((source.shape[0], channels, *source.shape[-2:]), value)
        return physical, torch.zeros_like(physical), object()

    def visual_detail(
        self,
        pyramid: object,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        output_size: tuple[int, int],
        base: Tensor,
    ) -> Tensor:
        del pyramid, source_spec, target_spec
        assert output_size == tuple(base.shape[-2:])
        return torch.zeros_like(base)

    def sample_visual_residual(
        self,
        pyramid: object,
        target_spec: SensorSpec,
        base: Tensor,
        detail: Tensor,
        *,
        seed: int,
    ) -> Tensor:
        del pyramid, target_spec, detail, seed
        return torch.zeros_like(base)

    @staticmethod
    def amplitude_scale_name(modality: str) -> str:
        return f"calibrated_{modality}_scale"


class _HeterogeneousOpticalRmseCalibrationModel(_SarBiasCalibrationModel):
    def __init__(self) -> None:
        super().__init__([0.0, 0.0])
        self.optical_bases = [0.5, 0.9]
        self.optical_physical_calls = 0

    def physical(
        self,
        source: Tensor,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        valid: Tensor,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor, object]:
        if target_spec.modality != "optical":
            return super().physical(source, source_spec, target_spec, valid, **kwargs)
        value = self.optical_bases[self.optical_physical_calls]
        self.optical_physical_calls += 1
        physical = source.new_full((source.shape[0], 3, *source.shape[-2:]), value)
        return physical, torch.zeros_like(physical), object()

    def sample_visual_residual(
        self,
        pyramid: object,
        target_spec: SensorSpec,
        base: Tensor,
        detail: Tensor,
        *,
        seed: int,
    ) -> Tensor:
        if target_spec.modality != "optical":
            return super().sample_visual_residual(pyramid, target_spec, base, detail, seed=seed)
        desired_visual = base.new_full(base.shape, 0.73)
        return (torch.logit(desired_visual) - torch.logit(base)) * base * (1.0 - base)


def _run_sar_bias_calibration(
    tmp_path: Path, monkeypatch: Any, scene_biases: list[float]
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = tmp_path / "input.pt"
    torch.save({"model": {}, "ema": {"state": {}}}, checkpoint)
    model = _SarBiasCalibrationModel(scene_biases)
    item = {
        "s2": torch.full((10, 4, 4), 0.5),
        "sar": torch.zeros(2, 4, 4),
        "valid": torch.ones(1, 4, 4),
        "gsd": 10.0,
    }

    def identity_prior(
        ignored_model: object,
        physical: Tensor,
        ignored_item: dict[str, object],
        ignored_target: SensorSpec,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del ignored_model, ignored_item, ignored_target
        return physical, torch.zeros(1), torch.zeros(1)

    monkeypatch.setattr(calibration, "load_checkpoint", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        calibration, "ManifestCropDataset", lambda *args, **kwargs: [item] * len(scene_biases)
    )
    monkeypatch.setattr(
        calibration, "manifest_metadata", lambda *args, **kwargs: torch.zeros(1, 8)
    )
    monkeypatch.setattr(calibration, "apply_manifest_temporal_prior", identity_prior)
    monkeypatch.setattr(calibration.torch.cuda, "is_available", lambda: False)

    output = tmp_path / "output.pt"
    result = calibration.calibrate_checkpoint(
        str(checkpoint), "ignored.jsonl", str(output), candidates=3
    )
    return result, torch.load(output, weights_only=False)


def test_calibrate_checkpoint_composes_temporal_prior_for_both_targets(
    tmp_path: Path, monkeypatch: Any
) -> None:
    checkpoint = tmp_path / "input.pt"
    torch.save({"model": {}, "ema": {"state": {}}}, checkpoint)
    model = _PriorAwareCalibrationModel()
    item = {
        "s2": torch.zeros(10, 4, 4),
        "sar": torch.zeros(2, 4, 4),
        "valid": torch.ones(1, 4, 4),
        "gsd": 10.0,
    }
    calls: list[str] = []

    def apply_prior(
        ignored_model: object,
        physical: Tensor,
        ignored_item: dict[str, object],
        target_spec: SensorSpec,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del ignored_model, ignored_item
        calls.append(target_spec.modality)
        offset = 0.1 if target_spec.modality == "optical" else 0.2
        return physical + offset, torch.ones(1), torch.zeros(1)

    monkeypatch.setattr(calibration, "load_checkpoint", lambda *args, **kwargs: model)
    monkeypatch.setattr(calibration, "ManifestCropDataset", lambda *args, **kwargs: [item])
    monkeypatch.setattr(calibration, "manifest_metadata", lambda *args, **kwargs: torch.zeros(1, 8))
    monkeypatch.setattr(calibration, "apply_manifest_temporal_prior", apply_prior)
    monkeypatch.setattr(calibration.torch.cuda, "is_available", lambda: False)

    output = tmp_path / "output.pt"
    result = calibration.calibrate_checkpoint(
        str(checkpoint), "ignored.jsonl", str(output), candidates=2
    )

    assert calls == ["optical", "sar"]
    assert model.amplitude_scales == [("optical", 1.0), ("sar", 1.0)]
    assert model.visual_calls == [("sar", "optical"), ("optical", "sar")]
    assert model.residual_calls == [("optical", 42), ("sar", 42)]
    torch.testing.assert_close(model.detail_bases[0], torch.full_like(model.detail_bases[0], 0.3))
    torch.testing.assert_close(model.detail_bases[1], torch.full_like(model.detail_bases[1], -19.8))
    saved = torch.load(output, weights_only=False)
    torch.testing.assert_close(saved["model"]["calibrated_optical_scale"], torch.tensor(1.0))
    torch.testing.assert_close(saved["model"]["calibrated_sar_scale"], torch.tensor(0.0))
    torch.testing.assert_close(saved["ema"]["state"]["calibrated_optical_scale"], torch.tensor(1.0))
    torch.testing.assert_close(saved["ema"]["state"]["calibrated_sar_scale"], torch.tensor(0.0))
    assert "sar_alpha_scale" not in saved["model"]
    assert result["optical_alpha"] == 1.0
    assert result["sar_alpha"] == 0.0
    assert result["sar_bias_gate_satisfied"] is False


def test_calibrate_checkpoint_sar_bias_aggregates_signed_scene_means(
    tmp_path: Path, monkeypatch: Any
) -> None:
    result, saved = _run_sar_bias_calibration(tmp_path, monkeypatch, [1.0, -1.0])

    assert result["sar_alpha"] == 1.0
    assert result["calibrated_sar_bias_db"] == 0.0
    assert result["calibrated_sar_scene_abs_bias_db"] == 1.0
    assert result["sar_bias_gate_satisfied"] is True
    torch.testing.assert_close(saved["model"]["calibrated_sar_scale"], torch.tensor(1.0))


def test_calibrate_checkpoint_reports_failed_sar_bias_gate(
    tmp_path: Path, monkeypatch: Any
) -> None:
    result, saved = _run_sar_bias_calibration(tmp_path, monkeypatch, [1.0, 1.0])

    assert result["sar_alpha"] == 0.0
    assert result["calibrated_sar_bias_db"] == 1.0
    assert result["calibrated_sar_scene_abs_bias_db"] == 1.0
    assert result["sar_bias_gate_satisfied"] is False
    torch.testing.assert_close(saved["model"]["calibrated_sar_scale"], torch.tensor(0.0))


def test_calibrate_checkpoint_averages_optical_scene_rmses_for_gate(
    tmp_path: Path, monkeypatch: Any
) -> None:
    checkpoint = tmp_path / "input.pt"
    torch.save({"model": {}, "ema": {"state": {}}}, checkpoint)
    model = _HeterogeneousOpticalRmseCalibrationModel()
    item = {
        "s2": torch.full((10, 4, 4), 0.5),
        "sar": torch.zeros(2, 4, 4),
        "valid": torch.ones(1, 4, 4),
        "gsd": 10.0,
    }

    def identity_prior(
        ignored_model: object,
        physical: Tensor,
        ignored_item: dict[str, object],
        ignored_target: SensorSpec,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del ignored_model, ignored_item, ignored_target
        return physical, torch.zeros(1), torch.zeros(1)

    monkeypatch.setattr(calibration, "load_checkpoint", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        calibration, "ManifestCropDataset", lambda *args, **kwargs: [item, item]
    )
    monkeypatch.setattr(
        calibration, "manifest_metadata", lambda *args, **kwargs: torch.zeros(1, 8)
    )
    monkeypatch.setattr(calibration, "apply_manifest_temporal_prior", identity_prior)
    monkeypatch.setattr(calibration.torch.cuda, "is_available", lambda: False)

    result = calibration.calibrate_checkpoint(
        str(checkpoint), "ignored.jsonl", str(tmp_path / "output.pt"), candidates=2
    )

    # Physical errors are 0.0 and 0.4, so evaluator-style aggregation is 0.2, not sqrt(0.08).
    assert result["physical_rgb_rmse"] == pytest.approx(0.2)
    assert result["calibrated_visual_rgb_rmse"] == pytest.approx(0.2)
    # Alpha 1 produces 0.23 error in each scene. It violates 1.05 * 0.2,
    # while sqrt(mean MSE) would incorrectly accept it.
    assert result["optical_alpha"] == 0.0


def test_calibrate_checkpoint_phase_optical_only_routes_packet_and_legacy_flows(
    tmp_path: Path, monkeypatch: Any
) -> None:
    model = SentinelV3(
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
            flow_steps=1,
            id_bridge_enabled=True,
            id_bridge_state="haar_packet",
            id_bridge_state_channels=48,
            id_bridge_anchor_origin=True,
            id_bridge_phase_identifiability=True,
            id_bridge_optical_only=True,
            phase_transport_enabled=True,
            phase_transport_hidden=16,
        )
    ).eval()
    assert model.legacy_residual_dit is not None
    checkpoint = tmp_path / "input.pt"
    torch.save({"model": {}, "ema": {"state": {}}}, checkpoint)
    item = {
        "s2": torch.rand(10, 32, 32),
        "sar": torch.randn(2, 32, 32),
        "valid": torch.ones(1, 32, 32),
        "gsd": 10.0,
    }
    flow_channels: list[tuple[str, int]] = []
    original_flow_velocity = model.flow_velocity

    def physical(
        values: Tensor,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        valid: Tensor,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        del kwargs
        pyramid = model.encode(values, source_spec, valid)
        physical = values.new_zeros(
            (values.shape[0], len(target_spec.channels), *values.shape[-2:])
        )
        return physical, torch.zeros_like(physical), pyramid

    def observe_flow_velocity(
        latent: Tensor,
        time: Tensor,
        pyramid: tuple[Tensor, ...],
        target_spec: SensorSpec,
        visual_channels: int | None = None,
        **kwargs: object,
    ) -> Tensor:
        flow_channels.append((target_spec.modality, latent.shape[1]))
        return original_flow_velocity(
            latent,
            time,
            pyramid,
            target_spec,
            visual_channels,
            **kwargs,
        )

    def identity_prior(
        ignored_model: object,
        physical: Tensor,
        ignored_item: dict[str, object],
        ignored_target: SensorSpec,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del ignored_model, ignored_item, ignored_target
        return physical, torch.zeros(1), torch.zeros(1)

    monkeypatch.setattr(calibration, "load_checkpoint", lambda *args, **kwargs: model)
    monkeypatch.setattr(calibration, "ManifestCropDataset", lambda *args, **kwargs: [item])
    monkeypatch.setattr(calibration, "manifest_metadata", lambda *args, **kwargs: torch.zeros(1, 8))
    monkeypatch.setattr(calibration, "apply_manifest_temporal_prior", identity_prior)
    monkeypatch.setattr(calibration.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(model, "physical", physical)
    monkeypatch.setattr(model, "flow_velocity", observe_flow_velocity)

    calibration.calibrate_checkpoint(
        str(checkpoint), "ignored.jsonl", str(tmp_path / "output.pt"), candidates=2
    )

    assert ("optical", 48) in flow_channels
    assert ("sar", 16) in flow_channels
