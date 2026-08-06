from __future__ import annotations

import math
from typing import ClassVar

import torch

from sentinel_v3.data import StatefulShardSampler, time_weights
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


def test_gsd_condition_values() -> None:
    torch.testing.assert_close(gsd_condition(40, 10, 20), torch.tensor([2.0, 2.0, 1.0]))


def test_time_weights() -> None:
    physical, visual = time_weights(torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(physical, torch.tensor([1.0, 0.75, 0.4, 0.2]))
    torch.testing.assert_close(visual, torch.tensor([1.0, 0.5, 0.0, 0.0]))


class _Dataset:
    shards: ClassVar[list[dict[str, int]]] = [
        {"count": 5}, {"count": 5}, {"count": 5}, {"count": 5}
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
