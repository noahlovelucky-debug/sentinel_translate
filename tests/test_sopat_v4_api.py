from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import pytest
import torch

from sentinel_v3.sensors import SENTINEL1, SENTINEL2
from sentinel_v4.api import AnchorPair, Observation, TargetRequest, translate
from sentinel_v4.model import SOPAT, SOPATConfig

_NAIVE_FUTURE = datetime(2024, 2, 10, 13, tzinfo=UTC).replace(tzinfo=None)


def _tiny_model() -> SOPAT:
    torch.manual_seed(23)
    return SOPAT(
        SOPATConfig(
            width=8,
            hidden=32,
            encoder_depth=1,
            heads=4,
            adapter_rank=8,
            transport_heads=4,
            anchor_window_size=2,
        )
    ).eval()


def _scene(*, latest_hours_before_target: float = 12.0) -> tuple[AnchorPair, list[Observation], TargetRequest]:
    torch.manual_seed(29)
    target_time = datetime(2024, 2, 10, 12, tzinfo=UTC)
    source_anchor = torch.empty(2, 16, 16).uniform_(-0.5, 0.5)
    target_anchor = torch.empty(10, 16, 16).uniform_(-0.5, 0.5)
    pair = AnchorPair(
        source_anchor=source_anchor,
        target_anchor=target_anchor.unsqueeze(0),
        source_sensor=SENTINEL1,
        target_sensor=SENTINEL2,
        source_acquired=target_time - timedelta(days=5),
        target_acquired=target_time - timedelta(days=4),
    )
    observations = [
        Observation(
            values=(source_anchor + 0.08).clamp(-1.0, 1.0),
            sensor=SENTINEL1,
            acquired=target_time - timedelta(hours=latest_hours_before_target),
        ),
        Observation(
            values=(source_anchor - 0.04).clamp(-1.0, 1.0),
            sensor="sentinel-1",
            acquired=target_time - timedelta(days=3),
        ),
    ]
    request = TargetRequest(sensor="sentinel-2", acquired=target_time)
    return pair, observations, request


def test_translate_single_scene_preserves_batch_and_never_accepts_target_label() -> None:
    model = _tiny_model()
    pair, observations, request = _scene()

    result = translate(model, pair, observations, request)

    assert result.physical.shape == (1, 10, 16, 16)
    assert result.log_variance.shape == (1, 1, 16, 16)
    assert result.target is request
    assert result.task_is_translation.tolist() == [True]
    assert torch.equal(result.physical, pair.target_anchor)
    signature = inspect.signature(translate)
    assert set(signature.parameters) == {
        "model",
        "anchor_pair",
        "observations",
        "target",
        "translation_tolerance_days",
    }
    assert "target_label" not in signature.parameters


def test_translate_accepts_homogeneous_batch_and_target_broadcast() -> None:
    model = _tiny_model()
    first_pair, first_observations, request = _scene()
    second_pair, second_observations, _ = _scene()
    second_pair = AnchorPair(
        source_anchor=second_pair.source_anchor + 0.01,
        target_anchor=second_pair.target_anchor - 0.01,
        source_sensor=second_pair.source_sensor,
        target_sensor=second_pair.target_sensor,
        source_acquired=second_pair.source_acquired,
        target_acquired=second_pair.target_acquired,
    )

    result = translate(
        model,
        (first_pair, second_pair),
        (first_observations, second_observations[:1]),
        request,
    )

    assert result.physical.shape == (2, 10, 16, 16)
    assert result.task_is_translation.tolist() == [True, True]
    assert isinstance(result.target, tuple)
    assert len(result.target) == 2


def test_translate_marks_forecast_when_all_sources_precede_tolerance() -> None:
    model = _tiny_model()
    pair, observations, request = _scene(latest_hours_before_target=72.0)

    result = translate(model, pair, observations, request, translation_tolerance_days=1.0)

    assert result.task_is_translation.tolist() == [False]


def test_translate_accepts_date_anchors_with_aware_target_timestamp() -> None:
    model = _tiny_model()
    pair, observations, request = _scene()
    pair = AnchorPair(
        source_anchor=pair.source_anchor,
        target_anchor=pair.target_anchor,
        source_sensor=pair.source_sensor,
        target_sensor=pair.target_sensor,
        source_acquired="2024-02-05",
        target_acquired="2024-02-06",
    )

    result = translate(model, pair, observations, request)

    assert result.physical.shape == (1, 10, 16, 16)


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (
            lambda pair, observations, request: (
                pair,
                [
                    Observation(
                        values=observations[0].values,
                        sensor=SENTINEL1,
                        acquired=_NAIVE_FUTURE,
                    )
                ],
                request,
            ),
            "timezone-aware",
        ),
        (
            lambda pair, observations, request: (
                pair,
                [
                    Observation(
                        values=observations[0].values,
                        sensor=SENTINEL1,
                        acquired=datetime(2024, 2, 10, 13, tzinfo=UTC),
                    )
                ],
                request,
            ),
            "never be later",
        ),
        (
            lambda pair, observations, request: (
                pair,
                observations,
                TargetRequest(sensor=SENTINEL2, acquired=request.acquired, gsd_m=20.0),
            ),
            "canonical 10 m",
        ),
    ),
)
def test_translate_rejects_naive_future_and_noncanonical_inputs(mutator, message: str) -> None:
    model = _tiny_model()
    pair, observations, request = mutator(*_scene())

    with pytest.raises(ValueError, match=message):
        translate(model, pair, observations, request)


def test_translate_rejects_mixed_target_sensor_and_grid() -> None:
    model = _tiny_model()
    pair, observations, request = _scene()
    wrong_sensor = TargetRequest(sensor=SENTINEL1, acquired=request.acquired)
    wrong_grid = TargetRequest(
        sensor=SENTINEL2,
        acquired=request.acquired,
        canonical_grid_id="another-canonical-grid",
    )

    with pytest.raises(ValueError, match="target sensor"):
        translate(model, pair, observations, wrong_sensor)
    with pytest.raises(ValueError, match="grid"):
        translate(model, pair, observations, wrong_grid)
