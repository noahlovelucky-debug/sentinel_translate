from __future__ import annotations

import pytest
import torch

from sentinel_v3.paired_temporal_api import PairedAnchorBatch, translate_paired
from sentinel_v3.paired_temporal_v2 import (
    PairedTemporalConfig,
    SparsePairedAnchorTransport,
)
from sentinel_v3.sensors import SENTINEL1, SENTINEL2


def _model() -> SparsePairedAnchorTransport:
    return SparsePairedAnchorTransport(
        PairedTemporalConfig(
            width=32,
            latent_channels=8,
            attention_heads=4,
            flow_steps=2,
        )
    )


def _batch(
    *,
    frames: int,
    source_channels: int = 2,
    target_channels: int = 10,
    query_observed: bool = True,
) -> dict[str, torch.Tensor]:
    batch = 2
    size = 32
    if frames == 1:
        days = torch.tensor([0.0 if query_observed else -5.0])
    else:
        days = torch.linspace(-30.0, 0.0 if query_observed else -5.0, frames)
    return {
        "observations": torch.randn(batch, frames, source_channels, size, size).tanh(),
        "observation_valid": torch.ones(batch, frames, 1, size, size),
        "observation_days": days[None].expand(batch, -1).clone(),
        "observation_present": torch.ones(batch, frames, dtype=torch.bool),
        "source_anchor": torch.randn(batch, source_channels, size, size).tanh(),
        "source_anchor_valid": torch.ones(batch, 1, size, size),
        "target_anchor": torch.randn(batch, target_channels, size, size).tanh(),
        "target_anchor_valid": torch.ones(batch, 1, size, size),
        "anchor_days": torch.full((batch,), -40.0),
        "target": torch.randn(batch, target_channels, size, size).tanh(),
        "target_valid": torch.ones(batch, 1, size, size),
    }


@pytest.mark.parametrize(("frames", "query_observed"), [(1, True), (1, False), (5, True), (5, False)])
def test_one_to_many_observations_share_one_model(frames: int, query_observed: bool) -> None:
    model = _model()
    batch = _batch(frames=frames, query_observed=query_observed)
    output = model(
        batch["observations"],
        batch["observation_valid"],
        batch["observation_days"],
        batch["observation_present"],
        batch["source_anchor"],
        batch["source_anchor_valid"],
        batch["target_anchor"],
        batch["target_anchor_valid"],
        batch["anchor_days"],
        source_sensor=SENTINEL1,
        target_sensor=SENTINEL2,
    )
    assert output.physical.shape == batch["target_anchor"].shape
    assert output.attention.shape == (2, frames, 1, 8, 8)
    assert output.observation_support.shape == (2, 1, 8, 8)
    torch.testing.assert_close(output.physical, batch["target_anchor"])
    torch.testing.assert_close(output.visual_base, output.physical)
    assert torch.all(output.task_is_translation == float(query_observed))
    assert torch.isfinite(output.log_variance).all()


def test_padding_values_do_not_change_a_single_observation_result() -> None:
    torch.manual_seed(3)
    model = _model().eval()
    single = _batch(frames=1)
    padded = {key: value.clone() for key, value in single.items()}
    padded["observations"] = torch.cat(
        (single["observations"], torch.randn(2, 3, 2, 32, 32)), dim=1
    )
    padded["observation_valid"] = torch.cat(
        (single["observation_valid"], torch.ones(2, 3, 1, 32, 32)), dim=1
    )
    padded["observation_days"] = torch.cat(
        (single["observation_days"], torch.tensor([[99.0, 88.0, 77.0]]).expand(2, -1)),
        dim=1,
    )
    padded["observation_present"] = torch.tensor(
        [[True, False, False, False], [True, False, False, False]]
    )

    def run(values: dict[str, torch.Tensor]):  # type: ignore[no-untyped-def]
        return model(
            values["observations"],
            values["observation_valid"],
            values["observation_days"],
            values["observation_present"],
            values["source_anchor"],
            values["source_anchor_valid"],
            values["target_anchor"],
            values["target_anchor_valid"],
            values["anchor_days"],
            source_sensor=SENTINEL1,
            target_sensor=SENTINEL2,
        )

    with torch.no_grad():
        expected = run(single)
        actual = run(padded)
    torch.testing.assert_close(actual.physical, expected.physical)
    torch.testing.assert_close(actual.log_variance, expected.log_variance)
    torch.testing.assert_close(actual.observation_support, expected.observation_support)
    assert torch.all(actual.attention[:, 1:] == 0)


def test_forecast_has_higher_analytic_uncertainty_than_query_translation() -> None:
    model = _model().eval()
    translation = _batch(frames=1, query_observed=True)
    forecast = {key: value.clone() for key, value in translation.items()}
    forecast["observation_days"].fill_(-20.0)

    def run(values: dict[str, torch.Tensor]) -> torch.Tensor:
        return model(
            values["observations"],
            values["observation_valid"],
            values["observation_days"],
            values["observation_present"],
            values["source_anchor"],
            values["source_anchor_valid"],
            values["target_anchor"],
            values["target_anchor_valid"],
            values["anchor_days"],
            source_sensor=SENTINEL1,
            target_sensor=SENTINEL2,
        ).log_variance

    with torch.no_grad():
        assert float(run(forecast).mean()) > float(run(translation).mean())


def test_one_day_sensor_offset_is_still_translation() -> None:
    model = _model().eval()
    values = _batch(frames=1, query_observed=True)
    values["observation_days"].fill_(-1.0)
    with torch.no_grad():
        output = model(
            values["observations"],
            values["observation_valid"],
            values["observation_days"],
            values["observation_present"],
            values["source_anchor"],
            values["source_anchor_valid"],
            values["target_anchor"],
            values["target_anchor_valid"],
            values["anchor_days"],
            source_sensor=SENTINEL1,
            target_sensor=SENTINEL2,
        )
    assert torch.all(output.task_is_translation == 1)


def test_registered_pair_keeps_distinct_source_and_target_anchor_times() -> None:
    model = _model().eval()
    values = _batch(frames=1, query_observed=True)
    source_days = torch.full((2,), -41.0)
    target_days = torch.full((2,), -40.0)
    with torch.no_grad():
        output = model(
            values["observations"],
            values["observation_valid"],
            values["observation_days"],
            values["observation_present"],
            values["source_anchor"],
            values["source_anchor_valid"],
            values["target_anchor"],
            values["target_anchor_valid"],
            values["anchor_days"],
            source_sensor=SENTINEL1,
            target_sensor=SENTINEL2,
            source_anchor_days=source_days,
            target_anchor_days=target_days,
        )
    assert output.physical.shape == values["target_anchor"].shape
    temporal = model.fusion._time_features(
        values["observation_days"],
        source_days,
        target_days,
        values["observation_present"],
    )
    shifted_target_temporal = model.fusion._time_features(
        values["observation_days"],
        source_days,
        target_days - 1.0,
        values["observation_present"],
    )
    assert not torch.equal(temporal, shifted_target_temporal)
    values["anchor_days"].zero_()
    with pytest.raises(ValueError, match="source_anchor_days"):
        model(
            values["observations"],
            values["observation_valid"],
            values["observation_days"],
            values["observation_present"],
            values["source_anchor"],
            values["source_anchor_valid"],
            values["target_anchor"],
            values["target_anchor_valid"],
            values["anchor_days"],
            source_sensor=SENTINEL1,
            target_sensor=SENTINEL2,
            source_anchor_days=values["anchor_days"],
            target_anchor_days=target_days,
        )


def test_optical_to_sar_uses_the_same_dynamic_channel_core() -> None:
    model = _model()
    batch = _batch(frames=3, source_channels=10, target_channels=2)
    output = model(
        batch["observations"],
        batch["observation_valid"],
        batch["observation_days"],
        batch["observation_present"],
        batch["source_anchor"],
        batch["source_anchor_valid"],
        batch["target_anchor"],
        batch["target_anchor_valid"],
        batch["anchor_days"],
        source_sensor=SENTINEL2,
        target_sensor=SENTINEL1,
    )
    assert output.physical.shape == (2, 2, 32, 32)


def test_flow_backward_and_fixed_seed_visual_preserve_physical() -> None:
    model = _model()
    batch = _batch(frames=3)
    output = model(
        batch["observations"],
        batch["observation_valid"],
        batch["observation_days"],
        batch["observation_present"],
        batch["source_anchor"],
        batch["source_anchor_valid"],
        batch["target_anchor"],
        batch["target_anchor_valid"],
        batch["anchor_days"],
        source_sensor=SENTINEL1,
        target_sensor=SENTINEL2,
    )
    losses = model.visual_flow_loss(
        output, batch["target"], batch["target_valid"], SENTINEL2
    )
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total)
    assert model.bridge.body[-1].weight.grad is not None
    with torch.no_grad():
        first = model.sample_visual(output, batch["target_valid"], SENTINEL2, seed=11)
        second = model.sample_visual(output, batch["target_valid"], SENTINEL2, seed=11)
    assert first.visual is not None
    torch.testing.assert_close(first.physical, output.physical)
    torch.testing.assert_close(first.visual, second.visual)
    torch.testing.assert_close(first.visual, output.physical)


def test_visual_flow_loss_uses_an_optional_replayable_generator() -> None:
    model = _model().eval()
    batch = _batch(frames=2)
    with torch.no_grad():
        output = model(
            batch["observations"],
            batch["observation_valid"],
            batch["observation_days"],
            batch["observation_present"],
            batch["source_anchor"],
            batch["source_anchor_valid"],
            batch["target_anchor"],
            batch["target_anchor_valid"],
            batch["anchor_days"],
            source_sensor=SENTINEL1,
            target_sensor=SENTINEL2,
        )
        first = model.visual_flow_loss(
            output,
            batch["target"],
            batch["target_valid"],
            SENTINEL2,
            generator=torch.Generator().manual_seed(123),
        )
        second = model.visual_flow_loss(
            output,
            batch["target"],
            batch["target_valid"],
            SENTINEL2,
            generator=torch.Generator().manual_seed(123),
        )
    for key in first:
        torch.testing.assert_close(first[key], second[key])


@pytest.mark.parametrize(
    ("days", "present", "match"),
    [
        ([1.0], [True], "future"),
        ([0.0], [False], "at least one"),
        ([-181.0], [True], "horizon"),
    ],
)
def test_invalid_availability_or_time_is_rejected(
    days: list[float], present: list[bool], match: str
) -> None:
    model = _model()
    batch = _batch(frames=1)
    batch["observation_days"] = torch.tensor([days, days])
    batch["observation_present"] = torch.tensor([present, present])
    with pytest.raises(ValueError, match=match):
        model(
            batch["observations"],
            batch["observation_valid"],
            batch["observation_days"],
            batch["observation_present"],
            batch["source_anchor"],
            batch["source_anchor_valid"],
            batch["target_anchor"],
            batch["target_anchor_valid"],
            batch["anchor_days"],
            source_sensor=SENTINEL1,
            target_sensor=SENTINEL2,
        )


def test_zero_block_mean_preserves_local_radiometric_base() -> None:
    residual = torch.randn(2, 3, 16, 16)
    centered = SparsePairedAnchorTransport._zero_block_mean(residual, 4)
    block_means = torch.nn.functional.avg_pool2d(centered, 4, stride=4)
    torch.testing.assert_close(block_means, torch.zeros_like(block_means), atol=1e-6, rtol=0)


def test_public_paired_api_preserves_training_state_and_reports_task() -> None:
    model = _model().train()
    values = _batch(frames=1, query_observed=True)
    batch = PairedAnchorBatch(
        observations=values["observations"],
        observation_valid=values["observation_valid"],
        observation_days=values["observation_days"],
        observation_present=values["observation_present"],
        source_anchor=values["source_anchor"],
        source_anchor_valid=values["source_anchor_valid"],
        target_anchor=values["target_anchor"],
        target_anchor_valid=values["target_anchor_valid"],
        anchor_days=values["anchor_days"],
        source_sensor=SENTINEL1,
        target_sensor=SENTINEL2,
    )
    result = translate_paired(model, batch, mode="visual", seed=9, steps=1)
    assert model.training
    assert result.output.shape == values["target_anchor"].shape
    assert torch.all(result.task_is_translation == 1)
    torch.testing.assert_close(result.output, result.physical)
