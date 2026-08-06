from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

from .sensors import ChannelSpec, SensorSpec, get_sensor

if TYPE_CHECKING:
    from .model import SentinelV3


@dataclass
class Observation:
    values: Tensor
    sensor: str | SensorSpec
    acquired: date | str
    channels: tuple[ChannelSpec, ...] | None = None
    valid_mask: Tensor | None = None
    gsd_m: float = 10.0
    orbit: Literal["ascending", "descending", "unknown"] = "unknown"

    @property
    def spec(self) -> SensorSpec:
        return get_sensor(self.sensor) if isinstance(self.sensor, str) else self.sensor

    @property
    def channel_specs(self) -> tuple[ChannelSpec, ...]:
        return self.channels or self.spec.channels


@dataclass(frozen=True)
class TargetRequest:
    sensor: str | SensorSpec
    gsd_m: float = 10.0
    acquired: date | str | None = None

    @property
    def spec(self) -> SensorSpec:
        return get_sensor(self.sensor) if isinstance(self.sensor, str) else self.sensor


@dataclass
class TranslationResult:
    physical: Tensor
    uncertainty: Tensor
    samples: list[Tensor] = field(default_factory=list)
    target: TargetRequest | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def translate(
    model: SentinelV3,
    observations: list[Observation],
    target: TargetRequest,
    mode: Literal["physical", "visual"],
    num_samples: int = 1,
    seed: int = 42,
) -> TranslationResult:
    if not observations:
        raise ValueError("at least one observation is required")
    if mode not in {"physical", "visual"}:
        raise ValueError("mode must be physical or visual")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    with torch.inference_mode():
        return model.translate(observations, target, mode=mode, num_samples=num_samples, seed=seed)
