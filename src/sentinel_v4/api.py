"""Public deterministic inference API for SOPAT V4.

The API exposes only historical paired anchors and source observations.  Query
target pixels are intentionally not part of this boundary: they belong only to
training losses and offline evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time

import torch
from torch import Tensor

from sentinel_v3.sensors import SensorSpec, get_sensor

from .model import SOPAT, Pyramid, SOPATOutput

Timestamp = datetime | date | str
Sensor = str | SensorSpec
_DEFAULT_GRID_ID = "canonical-10m"
_CANONICAL_GSD_M = 10.0


@dataclass(frozen=True)
class Observation:
    """One source-modality observation on the canonical 10 m grid."""

    values: Tensor
    sensor: Sensor
    acquired: Timestamp
    valid_mask: Tensor | None = None
    gsd_m: float = _CANONICAL_GSD_M
    canonical_grid_id: str = _DEFAULT_GRID_ID

    @property
    def spec(self) -> SensorSpec:
        return _sensor_spec(self.sensor)

    @property
    def grid_id(self) -> str:
        return self.canonical_grid_id


@dataclass(frozen=True)
class AnchorPair:
    """One registered historical S1/S2 pair on a shared canonical grid."""

    source_anchor: Tensor
    target_anchor: Tensor
    source_sensor: Sensor
    target_sensor: Sensor
    source_acquired: Timestamp
    target_acquired: Timestamp
    source_valid_mask: Tensor | None = None
    target_valid_mask: Tensor | None = None
    source_gsd_m: float = _CANONICAL_GSD_M
    target_gsd_m: float = _CANONICAL_GSD_M
    canonical_grid_id: str = _DEFAULT_GRID_ID

    @property
    def source_spec(self) -> SensorSpec:
        return _sensor_spec(self.source_sensor)

    @property
    def target_spec(self) -> SensorSpec:
        return _sensor_spec(self.target_sensor)

    @property
    def source_values(self) -> Tensor:
        """Compatibility alias for code that names anchor tensors as values."""

        return self.source_anchor

    @property
    def target_values(self) -> Tensor:
        """Compatibility alias for code that names anchor tensors as values."""

        return self.target_anchor

    @property
    def grid_id(self) -> str:
        return self.canonical_grid_id


@dataclass(frozen=True)
class TargetRequest:
    """Requested target sensor, canonical grid, and causal query time."""

    sensor: Sensor
    acquired: Timestamp
    gsd_m: float = _CANONICAL_GSD_M
    canonical_grid_id: str = _DEFAULT_GRID_ID

    @property
    def spec(self) -> SensorSpec:
        return _sensor_spec(self.sensor)

    @property
    def grid_id(self) -> str:
        return self.canonical_grid_id


@dataclass
class SOPATResult:
    """Physical SOPAT output and input request metadata, always batched."""

    physical: Tensor
    log_variance: Tensor
    transported_change: Pyramid
    common_anchor: Tensor
    source_private: Tensor
    target_private: Tensor
    attention: tuple[Tensor, Tensor, Tensor, Tensor]
    observation_support: tuple[Tensor, Tensor, Tensor, Tensor]
    task_is_translation: Tensor
    pre_projection_violation: Tensor
    source_anchor_reconstruction: Tensor
    target_anchor_reconstruction: Tensor
    source_anchor_cross: Tensor
    target_anchor_cross: Tensor
    common_source: Tensor
    common_target: Tensor
    private_source: Tensor
    private_target: Tensor
    target: TargetRequest | tuple[TargetRequest, ...]
    raw_delta: Tensor | None = None

    @classmethod
    def from_output(
        cls, output: SOPATOutput, target: TargetRequest | tuple[TargetRequest, ...]
    ) -> SOPATResult:
        return cls(
            physical=output.physical,
            log_variance=output.log_variance,
            transported_change=output.transported_change,
            common_anchor=output.common_anchor,
            source_private=output.source_private,
            target_private=output.target_private,
            attention=output.attention,
            observation_support=output.observation_support,
            task_is_translation=output.task_is_translation,
            pre_projection_violation=output.pre_projection_violation,
            source_anchor_reconstruction=output.source_anchor_reconstruction,
            target_anchor_reconstruction=output.target_anchor_reconstruction,
            source_anchor_cross=output.source_anchor_cross,
            target_anchor_cross=output.target_anchor_cross,
            common_source=output.common_source,
            common_target=output.common_target,
            private_source=output.private_source,
            private_target=output.private_target,
            target=target,
            raw_delta=output.raw_delta,
        )


@dataclass(frozen=True)
class _ResolvedTime:
    value: datetime
    precision: str


def _sensor_spec(sensor: Sensor) -> SensorSpec:
    if isinstance(sensor, SensorSpec):
        return sensor
    if isinstance(sensor, str):
        return get_sensor(sensor)
    raise TypeError("sensor must be a registered sensor name or SensorSpec")


def _resolved_time(value: Timestamp, name: str) -> _ResolvedTime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} datetime must be timezone-aware")
        return _ResolvedTime(value.astimezone(UTC), "datetime")
    if isinstance(value, date):
        return _ResolvedTime(datetime.combine(value, time.min, tzinfo=UTC), "date")
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a date, timezone-aware datetime, or ISO string")
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"{name} cannot be empty")
    if "T" not in parsed and " " not in parsed:
        try:
            return _ResolvedTime(
                datetime.combine(date.fromisoformat(parsed), time.min, tzinfo=UTC),
                "date",
            )
        except ValueError as error:
            raise ValueError(f"{name} is not an ISO date") from error
    try:
        instant = datetime.fromisoformat(parsed)
    except ValueError as error:
        raise ValueError(f"{name} is not an ISO datetime") from error
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError(f"{name} datetime must be timezone-aware")
    return _ResolvedTime(instant.astimezone(UTC), "datetime")


def _canonical_gsd(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or not math.isclose(
        float(value), _CANONICAL_GSD_M, abs_tol=1e-9
    ):
        raise ValueError(f"{name} must be the canonical 10 m grid; arbitrary GSD is unsupported")


def _canonical_grid(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty canonical grid id")
    return value


def _single_image(values: Tensor, name: str) -> Tensor:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape CxHxW or 1xCxHxW")
    if not values.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be finite")
    if bool((values.abs() > 1.0 + 1e-6).any()):
        raise ValueError(f"{name} must be normalized to [-1, 1]")
    return values


def _single_valid(valid: Tensor | None, image: Tensor, name: str) -> Tensor:
    if valid is None:
        return torch.ones((1, *image.shape[-2:]), device=image.device, dtype=image.dtype)
    if not isinstance(valid, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if valid.ndim == 4 and valid.shape[:2] == (1, 1):
        valid = valid[0]
    if valid.ndim == 2:
        valid = valid.unsqueeze(0)
    if valid.shape != (1, *image.shape[-2:]):
        raise ValueError(f"{name} must have shape HxW, 1xHxW, or 1x1xHxW")
    valid = valid.to(device=image.device, dtype=image.dtype)
    if not bool(torch.isfinite(valid).all()):
        raise ValueError(f"{name} must be finite")
    if bool(((valid < 0.0) | (valid > 1.0)).any()):
        raise ValueError(f"{name} must be in [0, 1]")
    return valid


def _as_anchor_pairs(anchor_pair: AnchorPair | Sequence[AnchorPair]) -> tuple[AnchorPair, ...]:
    if isinstance(anchor_pair, AnchorPair):
        return (anchor_pair,)
    if not isinstance(anchor_pair, Sequence) or isinstance(anchor_pair, (str, bytes)):
        raise TypeError("anchor_pair must be an AnchorPair or a non-empty sequence of AnchorPair")
    pairs = tuple(anchor_pair)
    if not pairs or not all(isinstance(pair, AnchorPair) for pair in pairs):
        raise ValueError("at least one AnchorPair is required")
    return pairs


def _as_observation_sets(
    observations: Sequence[Observation] | Sequence[Sequence[Observation]], batch: int
) -> tuple[tuple[Observation, ...], ...]:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise TypeError("observations must be a sequence")
    if batch == 1 and all(isinstance(item, Observation) for item in observations):
        sets = (tuple(observations),)
    else:
        if len(observations) != batch:
            raise ValueError("batched observations must have one sequence per AnchorPair")
        sets = tuple(tuple(item) if isinstance(item, Sequence) else () for item in observations)
    if any(not value for value in sets):
        raise ValueError("each AnchorPair requires one or more source observations")
    if any(not all(isinstance(item, Observation) for item in value) for value in sets):
        raise TypeError("each observation set must contain Observation values")
    return sets


def _as_targets(
    target: TargetRequest | Sequence[TargetRequest], batch: int
) -> tuple[TargetRequest, ...]:
    if isinstance(target, TargetRequest):
        return (target,) * batch
    if not isinstance(target, Sequence) or isinstance(target, (str, bytes)):
        raise TypeError("target must be a TargetRequest or a sequence of TargetRequest")
    targets = tuple(target)
    if len(targets) != batch or not all(isinstance(value, TargetRequest) for value in targets):
        raise ValueError("batched target requests must match the AnchorPair count")
    return targets


def _validate_pair_sensors(pair: AnchorPair, target: TargetRequest) -> tuple[SensorSpec, SensorSpec]:
    source_sensor = pair.source_spec
    target_sensor = pair.target_spec
    if {source_sensor.name, target_sensor.name} != {"sentinel-1", "sentinel-2"}:
        raise ValueError("SOPAT V4 supports Sentinel-1/Sentinel-2 anchor pairs only")
    if target.spec.name != target_sensor.name:
        raise ValueError("TargetRequest sensor must equal the AnchorPair target sensor")
    return source_sensor, target_sensor


def _validate_scene(
    pair: AnchorPair,
    observations: tuple[Observation, ...],
    target: TargetRequest,
    tolerance_days: float,
) -> tuple[
    SensorSpec,
    SensorSpec,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    tuple[float, ...],
    float,
    float,
]:
    source_sensor, target_sensor = _validate_pair_sensors(pair, target)
    for name, value in (
        ("AnchorPair source_gsd_m", pair.source_gsd_m),
        ("AnchorPair target_gsd_m", pair.target_gsd_m),
        ("TargetRequest gsd_m", target.gsd_m),
    ):
        _canonical_gsd(value, name)
    grid = _canonical_grid(pair.canonical_grid_id, "AnchorPair canonical_grid_id")
    if _canonical_grid(target.canonical_grid_id, "TargetRequest canonical_grid_id") != grid:
        raise ValueError("TargetRequest grid must match the registered AnchorPair grid")

    source_anchor = _single_image(pair.source_anchor, "source_anchor")
    target_anchor = _single_image(pair.target_anchor, "target_anchor")
    if source_anchor.shape[0] != len(source_sensor.channels):
        raise ValueError("source_anchor channels do not match source_sensor")
    if target_anchor.shape[0] != len(target_sensor.channels):
        raise ValueError("target_anchor channels do not match target_sensor")
    if source_anchor.shape[-2:] != target_anchor.shape[-2:]:
        raise ValueError("registered source and target anchors must share a canonical grid")
    source_valid = _single_valid(pair.source_valid_mask, source_anchor, "source_anchor_valid_mask")
    target_valid = _single_valid(pair.target_valid_mask, target_anchor, "target_anchor_valid_mask")

    source_time = _resolved_time(pair.source_acquired, "source_anchor acquired")
    target_anchor_time = _resolved_time(pair.target_acquired, "target_anchor acquired")
    request_time = _resolved_time(target.acquired, "TargetRequest acquired")
    observation_times: list[_ResolvedTime] = []
    observation_values: list[Tensor] = []
    observation_valid: list[Tensor] = []
    for index, observation in enumerate(observations):
        if observation.spec.name != source_sensor.name:
            raise ValueError("source observations must be homogeneous with the AnchorPair source sensor")
        _canonical_gsd(observation.gsd_m, f"Observation {index} gsd_m")
        if _canonical_grid(observation.canonical_grid_id, f"Observation {index} grid") != grid:
            raise ValueError("source observations must share the registered canonical grid")
        values = _single_image(observation.values, f"Observation {index} values")
        if values.shape != source_anchor.shape:
            raise ValueError("source observations must match source anchor channels and spatial grid")
        observation_values.append(values)
        observation_valid.append(_single_valid(observation.valid_mask, values, f"Observation {index} valid_mask"))
        observed_time = _resolved_time(observation.acquired, f"Observation {index} acquired")
        observation_times.append(observed_time)
    if not source_time.value < request_time.value or not target_anchor_time.value < request_time.value:
        raise ValueError("both registered anchors must strictly precede the requested target time")
    if any(resolved.value > request_time.value for resolved in observation_times):
        raise ValueError("source observations must never be later than the requested target time")

    observation_days = tuple(
        (resolved.value - request_time.value).total_seconds() / 86400.0
        for resolved in observation_times
    )
    latest_gap_days = -max(observation_days)
    if latest_gap_days < 0.0:
        raise AssertionError("validated observations unexpectedly occur after target time")
    if latest_gap_days > tolerance_days and any(-day <= tolerance_days for day in observation_days):
        raise ValueError("forecast observations must all precede the translation tolerance")
    source_anchor_days = (source_time.value - request_time.value).total_seconds() / 86400.0
    target_anchor_days = (target_anchor_time.value - request_time.value).total_seconds() / 86400.0
    return (
        source_sensor,
        target_sensor,
        source_anchor,
        source_valid,
        target_anchor,
        target_valid,
        tuple(observation_values),
        tuple(observation_valid),
        observation_days,
        source_anchor_days,
        target_anchor_days,
    )


@torch.inference_mode()
def translate(
    model: SOPAT,
    anchor_pair: AnchorPair | Sequence[AnchorPair],
    observations: Sequence[Observation] | Sequence[Sequence[Observation]],
    target: TargetRequest | Sequence[TargetRequest],
    *,
    translation_tolerance_days: float | None = None,
) -> SOPATResult:
    """Run deterministic physical transport from historical, causal inputs.

    A single request keeps a batch dimension of one.  Batched requests must be
    sensor- and grid-homogeneous because S1/S2 have distinct channel counts.
    """

    if not isinstance(model, SOPAT):
        raise TypeError("translate requires a SOPAT model")
    tolerance = (
        float(model.config.translation_tolerance_days)
        if translation_tolerance_days is None
        else float(translation_tolerance_days)
    )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("translation_tolerance_days must be finite and non-negative")
    pairs = _as_anchor_pairs(anchor_pair)
    observation_sets = _as_observation_sets(observations, len(pairs))
    targets = _as_targets(target, len(pairs))
    scenes = tuple(
        _validate_scene(pair, values, request, tolerance)
        for pair, values, request in zip(pairs, observation_sets, targets, strict=True)
    )
    source_sensor = scenes[0][0]
    target_sensor = scenes[0][1]
    grid = pairs[0].canonical_grid_id
    if any(scene[0].name != source_sensor.name for scene in scenes):
        raise ValueError("batched source observations must use one homogeneous source sensor")
    if any(scene[1].name != target_sensor.name for scene in scenes):
        raise ValueError("batched target requests must use one homogeneous target sensor")
    if any(pair.canonical_grid_id != grid or request.canonical_grid_id != grid for pair, request in zip(pairs, targets, strict=True)):
        raise ValueError("batched target requests must use one homogeneous canonical grid")

    spatial_shape = scenes[0][2].shape[-2:]
    device = scenes[0][2].device
    dtype = scenes[0][2].dtype
    if any(scene[2].shape[-2:] != spatial_shape for scene in scenes):
        raise ValueError("batched anchors must share spatial dimensions")
    if any(
        scene[2].device != device
        or scene[2].dtype != dtype
        or scene[4].device != device
        or scene[4].dtype != dtype
        for scene in scenes
    ):
        raise ValueError("batched anchors must share device and dtype")
    if any(values.device != device or values.dtype != dtype for scene in scenes for values in scene[6]):
        raise ValueError("batched observations must share anchor device and dtype")

    maximum_frames = max(len(scene[6]) for scene in scenes)
    source_channels = len(source_sensor.channels)
    height, width = spatial_shape
    observation_tensor = torch.zeros(
        (len(scenes), maximum_frames, source_channels, height, width), device=device, dtype=dtype
    )
    observation_valid = torch.zeros(
        (len(scenes), maximum_frames, 1, height, width), device=device, dtype=dtype
    )
    observation_days = torch.zeros((len(scenes), maximum_frames), device=device, dtype=dtype)
    observation_present = torch.zeros((len(scenes), maximum_frames), device=device, dtype=torch.bool)
    for index, scene in enumerate(scenes):
        values, valid, days = scene[6], scene[7], scene[8]
        frames = len(values)
        observation_tensor[index, :frames] = torch.stack(values)
        observation_valid[index, :frames] = torch.stack(valid)
        observation_days[index, :frames] = torch.tensor(days, device=device, dtype=dtype)
        observation_present[index, :frames] = True

    was_training = model.training
    model.eval()
    try:
        output = model(
            observations=observation_tensor,
            observation_valid=observation_valid,
            observation_days=observation_days,
            observation_present=observation_present,
            source_anchor=torch.stack([scene[2] for scene in scenes]),
            source_anchor_valid=torch.stack([scene[3] for scene in scenes]),
            target_anchor=torch.stack([scene[4] for scene in scenes]),
            target_anchor_valid=torch.stack([scene[5] for scene in scenes]),
            source_anchor_days=torch.tensor(
                [scene[9] for scene in scenes], device=device, dtype=dtype
            ),
            target_anchor_days=torch.tensor(
                [scene[10] for scene in scenes], device=device, dtype=dtype
            ),
            source_sensor=source_sensor,
            target_sensor=target_sensor,
        )
    finally:
        model.train(was_training)
    latest_days = torch.where(
        observation_present,
        observation_days,
        torch.full_like(observation_days, -float("inf")),
    ).max(dim=1).values
    output.task_is_translation = latest_days.abs() <= tolerance
    result_target: TargetRequest | tuple[TargetRequest, ...]
    result_target = targets[0] if isinstance(anchor_pair, AnchorPair) else targets
    return SOPATResult.from_output(output, result_target)
