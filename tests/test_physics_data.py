from __future__ import annotations

import math
from typing import ClassVar

import torch

from sentinel_v3.data import (
    StatefulShardSampler,
    estimate_registration_shift,
    high_frequency_eligible,
    time_weights,
)
from sentinel_v3.physics import (
    db_to_intensity,
    db_to_normalized_sar,
    gsd_condition,
    intensity_to_db,
    normalized_sar_to_db,
    physical_resample,
)


def test_sar_unit_roundtrip() -> None:
    values = torch.tensor([[[[-30.0]], [[-15.0]]]])
    torch.testing.assert_close(intensity_to_db(db_to_intensity(values)), values)
    torch.testing.assert_close(normalized_sar_to_db(db_to_normalized_sar(values)), values)


def test_sar_downsampling_averages_linear_intensity() -> None:
    values = torch.tensor([[[[0.0, 10.0], [0.0, 10.0]]]])
    reduced = physical_resample(
        values, modality="sar", source_gsd_m=10, target_gsd_m=20, restore_grid=False
    )
    expected = 10 * math.log10((1 + 10 + 1 + 10) / 4)
    torch.testing.assert_close(reduced, torch.tensor([[[[expected]]]]), atol=1e-4, rtol=1e-4)


def test_masked_resampling_keeps_constant_valid_regions_uncontaminated() -> None:
    valid = torch.ones(1, 1, 4, 4)
    valid[..., 0, 0] = 0.0
    optical = torch.full((1, 3, 4, 4), 0.6)
    optical[..., 0, 0] = 0.0
    sar = torch.full((1, 2, 4, 4), -20.0)
    sar[..., 0, 0] = 0.0

    reduced_optical = physical_resample(
        optical,
        modality="optical",
        source_gsd_m=10,
        target_gsd_m=20,
        restore_grid=False,
        valid=valid,
    )
    reduced_sar = physical_resample(
        sar,
        modality="sar",
        source_gsd_m=10,
        target_gsd_m=20,
        restore_grid=False,
        valid=valid,
    )
    torch.testing.assert_close(reduced_optical, torch.full_like(reduced_optical, 0.6))
    torch.testing.assert_close(reduced_sar, torch.full_like(reduced_sar, -20.0), atol=1e-4, rtol=1e-4)


def test_masked_resample_native_gsd_is_exact_passthrough() -> None:
    values = torch.randn(1, 2, 4, 4)
    valid = torch.zeros(1, 1, 4, 4)
    output = physical_resample(
        values,
        modality="sar",
        source_gsd_m=10,
        target_gsd_m=10,
        valid=valid,
    )
    torch.testing.assert_close(output, values)


def test_gsd_condition_values() -> None:
    torch.testing.assert_close(gsd_condition(40, 10, 20), torch.tensor([2.0, 2.0, 1.0]))


def test_time_weights() -> None:
    physical, visual = time_weights(torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(physical, torch.tensor([1.0, 1.0, 0.75, 0.5]))
    torch.testing.assert_close(visual, torch.tensor([1.0, 0.25, 0.0, 0.0]))


def test_high_frequency_audit_is_strict() -> None:
    common = {
        "delta_days": 1,
        "year": 2018,
        "split": "train",
        "valid_fraction": 0.9,
        "cloud_shadow_fraction": 0.1,
    }
    assert high_frequency_eligible(registration_shift_px=0.5, **common)
    assert not high_frequency_eligible(registration_shift_px=0.5001, **common)
    assert not high_frequency_eligible(registration_shift_px=0.0, **{**common, "delta_days": 2})


def test_registration_audit_requires_local_cross_modal_shift_evidence() -> None:
    torch.manual_seed(23)
    field = torch.nn.functional.avg_pool2d(
        torch.randn(1, 1, 48, 48), 7, stride=1, padding=3
    )[0, 0]
    shifted = torch.zeros_like(field)
    shifted[:, 1:] = field[:, :-1]
    optical = field.unsqueeze(0).repeat(10, 1, 1)
    # A monotonic radiometric transform preserves structural gradients.
    radar_shifted = (shifted * 3.0 + 2.0).unsqueeze(0).repeat(2, 1, 1)
    valid = torch.ones(1, 48, 48)

    measured = estimate_registration_shift(optical, radar_shifted, valid=valid)
    assert float(measured) > 0.5

    aligned = estimate_registration_shift(optical, optical[:2] * 2.0 + 3.0, valid=valid)
    assert float(aligned) == 0.0

    unrelated_field = torch.nn.functional.avg_pool2d(
        torch.randn(1, 1, 48, 48), 7, stride=1, padding=3
    )[0, 0]
    unrelated = estimate_registration_shift(
        optical, unrelated_field.unsqueeze(0).repeat(2, 1, 1), valid=valid
    )
    assert float(unrelated) == 0.0


class _Dataset:
    shards: ClassVar[list[dict[str, int]]] = [
        {"count": 5},
        {"count": 5},
        {"count": 5},
        {"count": 5},
    ]
    ends: ClassVar[list[int]] = [5, 10, 15, 20]

    def __len__(self) -> int:
        return 20


def test_sampler_resume_is_exact() -> None:
    sampler = StatefulShardSampler(_Dataset(), replicas=2, rank=0, seed=4)  # type: ignore[arg-type]
    iterator = iter(sampler)
    prefix = [next(iterator) for _ in range(3)]
    state = sampler.state_dict()
    suffix = list(iterator)
    resumed = StatefulShardSampler(_Dataset(), replicas=2, rank=0, seed=4)  # type: ignore[arg-type]
    resumed.load_state_dict(state)
    assert prefix
    assert list(resumed) == suffix
