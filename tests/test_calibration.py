from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from sentinel_v3 import calibration
from sentinel_v3.sensors import SensorSpec


class _PriorAwareCalibrationModel:
    def __init__(self) -> None:
        self.detail_bases: list[Tensor] = []

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
        value = 0.2 if target_spec.modality == "optical" else -20.0
        physical = source.new_full((source.shape[0], channels, *source.shape[-2:]), value)
        return physical, torch.zeros_like(physical), object()

    def deterministic_detail(self, *args: object, **kwargs: object) -> Tensor:
        del args
        base = kwargs["base"]
        assert isinstance(base, Tensor)
        self.detail_bases.append(base.detach().clone())
        return torch.zeros_like(base)

    def sample_residual(self, *args: object, **kwargs: object) -> Tensor:
        del kwargs
        shape = args[2]
        assert isinstance(shape, tuple)
        return torch.zeros(shape)

    @staticmethod
    def amplitude_scale_name(modality: str) -> str:
        assert modality == "optical"
        return "optical_alpha_scale"


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

    calibration.calibrate_checkpoint(
        str(checkpoint), "ignored.jsonl", str(tmp_path / "output.pt"), candidates=2
    )

    assert calls == ["optical", "sar"]
    torch.testing.assert_close(model.detail_bases[0], torch.full_like(model.detail_bases[0], 0.3))
    torch.testing.assert_close(model.detail_bases[1], torch.full_like(model.detail_bases[1], -19.8))
