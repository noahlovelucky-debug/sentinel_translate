from __future__ import annotations

import datetime as dt

import pytest
import torch

from sentinel_v3 import Observation, TargetRequest, translate
from sentinel_v3.model import SentinelV3
from sentinel_v3.sensors import SENTINEL1, SENTINEL2, ChannelSpec, SensorSpec


@pytest.mark.parametrize(
    ("source", "target", "source_channels", "target_channels"),
    [(SENTINEL1, SENTINEL2, 2, 10), (SENTINEL2, SENTINEL1, 10, 2)],
)
def test_bidirectional_physical_shapes(
    tiny_model: SentinelV3,
    source: SensorSpec,
    target: SensorSpec,
    source_channels: int,
    target_channels: int,
) -> None:
    values = torch.randn(2, source_channels, 32, 32)
    if source.modality == "optical":
        values = values.sigmoid()
    valid = torch.ones(2, 1, 32, 32)
    mean, log_variance, pyramid = tiny_model.physical(values, source, target, valid)
    assert mean.shape == (2, target_channels, 32, 32)
    assert log_variance.shape == mean.shape
    assert [level.shape[-2:] for level in pyramid] == [
        (32, 32), (16, 16), (8, 8), (4, 4)
    ]
    mean.mean().backward()
    assert any(parameter.grad is not None for parameter in tiny_model.encoder.parameters())


def test_arbitrary_channel_descriptions(tiny_model: SentinelV3) -> None:
    custom = SensorSpec(
        "custom-optical",
        "optical",
        (
            ChannelSpec("greenish", "reflectance", 15, 15, wavelength_nm=550),
            ChannelSpec("nirish", "reflectance", 30, 15, wavelength_nm=900),
            ChannelSpec("swirish", "reflectance", 30, 15, wavelength_nm=1600),
        ),
        "reflectance",
    )
    values = torch.rand(1, 3, 32, 32)
    mean, uncertainty, _ = tiny_model.physical(values, custom, SENTINEL1, torch.ones(1, 1, 32, 32))
    assert mean.shape == uncertainty.shape == (1, 2, 32, 32)


def test_public_api_is_seeded_and_physical_does_not_sample(tiny_model: SentinelV3) -> None:
    observation = Observation(
        torch.randn(2, 32, 32), SENTINEL1, dt.date(2020, 1, 2), orbit="ascending"
    )
    target = TargetRequest(SENTINEL2)
    physical = translate(tiny_model.eval(), [observation], target, "physical", num_samples=3, seed=7)
    assert physical.physical.shape == (1, 10, 32, 32)
    assert physical.samples == []
    first = translate(tiny_model, [observation], target, "visual", num_samples=2, seed=7)
    second = translate(tiny_model, [observation], target, "visual", num_samples=2, seed=7)
    assert len(first.samples) == 2
    torch.testing.assert_close(first.samples[0], second.samples[0])
    assert not torch.equal(first.samples[0], first.samples[1])
    torch.testing.assert_close(first.physical, second.physical)
