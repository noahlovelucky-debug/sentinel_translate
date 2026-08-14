"""Training primitives for bidirectional SOPAT V4.

SOPAT trains one shared model on two homogeneous direction batches per global
optimizer step.  SAR-to-Optical and Optical-to-SAR have incompatible channel
counts, so this module deliberately never concatenates them into a fabricated
mixed tensor batch.  The two forwards instead share the same model, optimizer,
EMA, global step, checkpoint, and DDP reduction.

The public model contract is intentionally small.  ``forward_direction`` sends
only causal inputs to a model.  Query labels are extracted separately by the
objective functions and are never included in model keyword arguments.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

Direction = Literal["sar_to_optical", "optical_to_sar"]
Stage = Literal["factorizer", "physical"]
TrainableScope = Literal["full", "confidence_only"]
SourceCounterfactualMode = Literal["legacy_local_shuffle_v1", "global_cross_tile_v1"]

DIRECTIONS: tuple[Direction, Direction] = ("sar_to_optical", "optical_to_sar")
SOPAT_V4_FORMAT = 1
SOPAT_V4_FAMILY = "sopat_v4"
DEFAULT_V3_INITIALIZATION_PREFIXES: tuple[str, ...] = (
    "encoder.",
    "adapter.",
    "sensor_adapter.",
    "source_adapter.",
    "target_adapter.",
)


@runtime_checkable
class SOPATForwardProtocol(Protocol):
    """Minimal forward signature implemented by ``sentinel_v4.model.SOPAT``."""

    def forward(
        self,
        *,
        observations: Tensor,
        observation_valid: Tensor,
        observation_days: Tensor,
        observation_present: Tensor,
        source_anchor: Tensor,
        source_anchor_valid: Tensor,
        target_anchor: Tensor,
        target_anchor_valid: Tensor,
        source_anchor_days: Tensor,
        target_anchor_days: Tensor,
        source_sensor: object,
        target_sensor: object,
    ) -> object: ...


@dataclass(frozen=True)
class SOPATTrainConfig:
    """Stage and objective settings shared by both translation directions."""

    stage: Stage = "factorizer"
    trainable_scope: TrainableScope = "full"
    direction_weights: Mapping[str, float] = field(
        default_factory=lambda: {"sar_to_optical": 1.0, "optical_to_sar": 1.0}
    )
    learning_rate: float = 2e-4
    encoder_learning_rate: float | None = None
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    factorizer_anchor_cross_weight: float = 1.0
    factorizer_common_alignment_weight: float = 0.5
    factorizer_private_decorrelation_weight: float = 0.1
    physical_charbonnier_weight: float = 1.0
    physical_gradient_weight: float = 0.20
    physical_optical_spectral_weight: float = 0.10
    physical_optical_ndvi_weight: float = 0.10
    physical_sar_statistics_weight: float = 0.10
    physical_anchor_delta_weight: float = 0.20
    physical_null_change_weight: float = 0.10
    physical_null_change_probability: float = 0.25
    physical_nll_weight: float = 0.05
    physical_permutation_weight: float = 0.05
    physical_permutation_probability: float = 0.25
    physical_anchor_regret_weight: float = 0.25
    physical_anchor_regret_margin: float = 0.0
    candidate_weight: float = 0.5
    utility_weight: float = 0.1
    utility_temperature: float = 0.02
    source_shuffle_weight: float = 0.25
    source_shuffle_probability: float = 0.5
    source_shuffle_margin: float = 0.005
    source_counterfactual_mode: SourceCounterfactualMode = "legacy_local_shuffle_v1"
    counterfactual_candidate_ranking_weight: float = 0.25
    counterfactual_candidate_ranking_margin: float = 0.005
    counterfactual_source_effect_floor_weight: float = 0.05
    counterfactual_source_effect_floor: float = 0.005
    counterfactual_confidence_weight: float = 0.10
    counterfactual_confidence_binary_weight: float = 1.0
    counterfactual_confidence_margin: float = 0.10
    structural_pool_kernel: int = 5
    autocast_bfloat16: bool = True

    def __post_init__(self) -> None:
        if self.stage not in {"factorizer", "physical"}:
            raise ValueError("SOPAT stage must be factorizer or physical")
        if self.trainable_scope not in {"full", "confidence_only"}:
            raise ValueError("SOPAT trainable_scope must be full or confidence_only")
        if self.source_counterfactual_mode not in {
            "legacy_local_shuffle_v1",
            "global_cross_tile_v1",
        }:
            raise ValueError(
                "source_counterfactual_mode must be legacy_local_shuffle_v1 or global_cross_tile_v1"
            )
        if self.stage != "physical" and self.trainable_scope != "full":
            raise ValueError("confidence_only scope is valid only for the physical stage")
        if self.learning_rate <= 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        encoder_learning_rate = (
            self.learning_rate
            if self.encoder_learning_rate is None
            else float(self.encoder_learning_rate)
        )
        if encoder_learning_rate <= 0.0 or not np.isfinite(encoder_learning_rate):
            raise ValueError("encoder_learning_rate must be finite and positive")
        object.__setattr__(self, "encoder_learning_rate", encoder_learning_rate)
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        weights = {str(name): float(value) for name, value in self.direction_weights.items()}
        if set(weights) != set(DIRECTIONS):
            raise ValueError("direction_weights must contain exactly both SOPAT directions")
        if any(not np.isfinite(value) or value <= 0.0 for value in weights.values()):
            raise ValueError("direction_weights must be finite and positive")
        object.__setattr__(self, "direction_weights", weights)
        nonnegative = (
            "factorizer_anchor_cross_weight",
            "factorizer_common_alignment_weight",
            "factorizer_private_decorrelation_weight",
            "physical_charbonnier_weight",
            "physical_gradient_weight",
            "physical_optical_spectral_weight",
            "physical_optical_ndvi_weight",
            "physical_sar_statistics_weight",
            "physical_anchor_delta_weight",
            "physical_null_change_weight",
            "physical_nll_weight",
            "physical_permutation_weight",
            "physical_anchor_regret_weight",
            "physical_anchor_regret_margin",
            "candidate_weight",
            "utility_weight",
            "source_shuffle_weight",
            "source_shuffle_margin",
            "counterfactual_candidate_ranking_weight",
            "counterfactual_candidate_ranking_margin",
            "counterfactual_source_effect_floor_weight",
            "counterfactual_source_effect_floor",
            "counterfactual_confidence_weight",
            "counterfactual_confidence_binary_weight",
            "counterfactual_confidence_margin",
        )
        if any(float(getattr(self, name)) < 0.0 for name in nonnegative):
            raise ValueError("SOPAT loss weights and margins must be non-negative")
        for name in (
            "counterfactual_candidate_ranking_weight",
            "counterfactual_candidate_ranking_margin",
            "counterfactual_source_effect_floor_weight",
            "counterfactual_source_effect_floor",
            "counterfactual_confidence_weight",
            "counterfactual_confidence_binary_weight",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "physical_null_change_probability",
            "physical_permutation_probability",
            "source_shuffle_probability",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.utility_temperature <= 0.0 or not np.isfinite(self.utility_temperature):
            raise ValueError("utility_temperature must be finite and positive")
        if self.structural_pool_kernel <= 0 or self.structural_pool_kernel % 2 == 0:
            raise ValueError("structural_pool_kernel must be a positive odd integer")
        if self.stage == "factorizer" and (
            self.factorizer_anchor_cross_weight
            + self.factorizer_common_alignment_weight
            + self.factorizer_private_decorrelation_weight
            <= 0.0
        ):
            raise ValueError("factorizer stage needs at least one non-zero objective weight")
        if self.stage == "physical" and self.physical_charbonnier_weight <= 0.0:
            raise ValueError("physical stage requires physical_charbonnier_weight > 0")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SOPATTrainConfig:
        """Construct from a YAML mapping while rejecting unknown model knobs."""

        allowed = {entry.name for entry in fields(cls)}
        unknown = sorted(set(values).difference(allowed))
        if unknown:
            raise ValueError(f"unknown SOPAT training setting(s): {', '.join(unknown)}")
        return cls(**dict(values))  # type: ignore[arg-type]


@dataclass(frozen=True)
class CoupledStepResult:
    """Detached diagnostics from exactly one two-direction optimizer step."""

    total_loss: float
    gradient_norm: float
    direction_losses: Mapping[str, float]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class FactorizerValidationResult:
    """Anchor-only validation summary for the factorizer stage.

    A factorizer has no physical query prediction to compare against the
    target label.  Its validation signal must therefore remain its declared
    paired-anchor objective instead of a renderer-dependent physical gate.
    """

    weighted_loss: float
    direction_losses: Mapping[str, float]
    metrics: Mapping[str, float]
    batches: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "weighted_loss": self.weighted_loss,
            "direction_losses": dict(self.direction_losses),
            "metrics": dict(self.metrics),
            "batches": dict(self.batches),
        }


class CyclingDirectionIterator:
    """Repeat an exhausted direction loader with deterministic sampler epochs."""

    def __init__(self, loader: Iterable[Mapping[str, object]], *, seed: int = 0) -> None:
        self.loader = loader
        self.seed = int(seed)
        self.cycles = 0
        self._iterator: Iterator[Mapping[str, object]] | None = None

    def _set_epoch(self) -> None:
        sampler = getattr(self.loader, "sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(self.seed + self.cycles)

    def _restart(self) -> None:
        self._set_epoch()
        self._iterator = iter(self.loader)

    def __next__(self) -> Mapping[str, object]:
        if self._iterator is None:
            self._restart()
        assert self._iterator is not None
        try:
            return next(self._iterator)
        except StopIteration:
            self.cycles += 1
            self._restart()
            assert self._iterator is not None
            try:
                return next(self._iterator)
            except StopIteration as error:
                raise RuntimeError("SOPAT direction loader is empty") from error

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "cycles": self.cycles}


class CoupledDirectionIterator:
    """Yield one homogeneous microbatch for each direction every global step."""

    def __init__(
        self,
        loaders: Mapping[str, Iterable[Mapping[str, object]]],
        *,
        seed: int = 0,
    ) -> None:
        missing = set(DIRECTIONS).difference(loaders)
        unexpected = set(loaders).difference(DIRECTIONS)
        if missing or unexpected:
            raise ValueError(f"SOPAT loaders require both directions; missing={missing}, unexpected={unexpected}")
        self._iterators = {
            direction: CyclingDirectionIterator(loaders[direction], seed=seed + index * 1_000_003)
            for index, direction in enumerate(DIRECTIONS)
        }
        self.global_batches = 0

    def __iter__(self) -> CoupledDirectionIterator:
        return self

    def __next__(self) -> dict[Direction, Mapping[str, object]]:
        batch = {direction: next(self._iterators[direction]) for direction in DIRECTIONS}
        self.global_batches += 1
        return batch

    def state_dict(self) -> dict[str, object]:
        return {
            "global_batches": self.global_batches,
            "directions": {direction: iterator.state_dict() for direction, iterator in self._iterators.items()},
        }


_FORWARD_ALIASES: Mapping[str, tuple[str, ...]] = {
    "observations": ("observations", "observation_values"),
    "observation_valid": ("observation_valid",),
    "observation_days": ("observation_days",),
    "observation_present": ("observation_present",),
    "source_anchor": ("source_anchor", "source_anchor_values"),
    "source_anchor_valid": ("source_anchor_valid",),
    "target_anchor": ("target_anchor", "target_anchor_values"),
    "target_anchor_valid": ("target_anchor_valid",),
    "source_anchor_days": ("source_anchor_days", "anchor_days"),
    "target_anchor_days": ("target_anchor_days", "anchor_days"),
}
_LABEL_ALIASES: Mapping[str, tuple[str, ...]] = {
    "target": ("target", "target_values"),
    "target_valid": ("target_valid",),
}


def _validate_direction(direction: str) -> Direction:
    if direction not in DIRECTIONS:
        raise ValueError(f"unsupported SOPAT direction: {direction}")
    return direction  # type: ignore[return-value]


def _tensor_from_aliases(
    batch: Mapping[str, object],
    aliases: Sequence[str],
    name: str,
    device: torch.device | None,
) -> Tensor:
    value = next((batch[candidate] for candidate in aliases if candidate in batch), None)
    if not isinstance(value, Tensor):
        raise TypeError(f"SOPAT batch is missing tensor {name}")
    if device is not None:
        return value.to(device=device, non_blocking=True)
    return value


def forward_input_tensors(
    batch: Mapping[str, object], *, device: torch.device | None = None
) -> dict[str, Tensor]:
    """Return only tensors permitted to cross the causal model boundary."""

    tensors = {
        name: _tensor_from_aliases(batch, aliases, name, device)
        for name, aliases in _FORWARD_ALIASES.items()
    }
    tensors["observation_present"] = tensors["observation_present"].bool()
    return tensors


def supervision_tensors(
    batch: Mapping[str, object], *, device: torch.device | None = None
) -> dict[str, Tensor]:
    """Return query labels for losses/metrics, never for a model forward."""

    values = {
        name: _tensor_from_aliases(batch, aliases, name, device)
        for name, aliases in _LABEL_ALIASES.items()
    }
    if values["target"].ndim != 4 or values["target_valid"].ndim != 4:
        raise ValueError("SOPAT target and target_valid must have shape BxCxHxW and Bx1xHxW")
    if values["target_valid"].shape != (
        values["target"].shape[0],
        1,
        *values["target"].shape[-2:],
    ):
        raise ValueError("SOPAT target_valid does not match target")
    return values


def direction_sensors(direction: str) -> tuple[object, object]:
    """Use the stable Sentinel registry until V4 receives external descriptors."""

    _validate_direction(direction)
    from sentinel_v3.sensors import SENTINEL1, SENTINEL2

    return (SENTINEL1, SENTINEL2) if direction == "sar_to_optical" else (SENTINEL2, SENTINEL1)


def _batch_sensor(batch: Mapping[str, object], name: str, fallback: object) -> object:
    value = batch.get(name)
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        if not value:
            return fallback
        if any(candidate != value[0] for candidate in value[1:]):
            raise ValueError(f"SOPAT batch has mixed {name} values")
        return value[0]
    return value


def forward_direction(
    model: SOPATForwardProtocol | nn.Module,
    batch: Mapping[str, object],
    direction: str,
    *,
    device: torch.device | None = None,
) -> object:
    """Run a causal SOPAT direction forward without passing query labels.

    This function is the sole model-call route used by V4 training and
    evaluation.  Keeping the whitelist here makes target-label leakage an
    inspectable invariant rather than a convention in individual callers.
    """

    direction = _validate_direction(direction)
    tensors = forward_input_tensors(batch, device=device)
    source_sensor, target_sensor = direction_sensors(direction)
    return model(
        observations=tensors["observations"],
        observation_valid=tensors["observation_valid"],
        observation_days=tensors["observation_days"],
        observation_present=tensors["observation_present"],
        source_anchor=tensors["source_anchor"],
        source_anchor_valid=tensors["source_anchor_valid"],
        target_anchor=tensors["target_anchor"],
        target_anchor_valid=tensors["target_anchor_valid"],
        source_anchor_days=tensors["source_anchor_days"],
        target_anchor_days=tensors["target_anchor_days"],
        source_sensor=_batch_sensor(batch, "source_sensor", source_sensor),
        target_sensor=_batch_sensor(batch, "target_sensor", target_sensor),
    )


def _nested_output_value(output: object, name: str) -> object | None:
    if isinstance(output, Mapping) and name in output:
        return output[name]
    if hasattr(output, name):
        return getattr(output, name)
    nested = None
    if isinstance(output, Mapping):
        nested = output.get("factorizer")
    elif hasattr(output, "factorizer"):
        nested = output.factorizer  # type: ignore[attr-defined]
    if isinstance(nested, Mapping):
        return nested.get(name)
    if nested is not None and hasattr(nested, name):
        return getattr(nested, name)
    return None


def output_tensor(
    output: object,
    *names: str,
    required: bool = True,
) -> Tensor | None:
    """Read a tensor from the public SOPAT output or its factorizer payload."""

    for name in names:
        value = _nested_output_value(output, name)
        if isinstance(value, Tensor):
            return value
        if value is not None:
            raise TypeError(f"SOPAT output field {name} must be a tensor")
    if required:
        raise AttributeError(f"SOPAT output is missing required tensor field one of {names}")
    return None


def _resize_valid(valid: Tensor, values: Tensor) -> Tensor:
    if valid.shape[0] != values.shape[0] or valid.shape[1] != 1:
        raise ValueError("SOPAT valid mask must have shape Bx1xHxW")
    if valid.shape[-2:] == values.shape[-2:]:
        return valid.to(values)
    return F.interpolate(valid.float(), size=values.shape[-2:], mode="area").to(values)


def masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    mask = _resize_valid(valid, values).expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_single_channel_mean(values: Tensor, valid: Tensor) -> Tensor:
    """Average a Bx1xHxW diagnostic only where its binary support is valid."""

    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("SOPAT single-channel diagnostic must have shape Bx1xHxW")
    mask = _resize_valid(valid, values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def masked_charbonnier(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("SOPAT prediction and target must have equal shape")
    return masked_mean(torch.sqrt((prediction - target).square() + 1e-6), valid)


def _gradient_loss(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    if prediction.shape[-2] < 2 or prediction.shape[-1] < 2:
        return prediction.new_zeros(())
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    valid_x = valid[..., :, 1:] * valid[..., :, :-1]
    valid_y = valid[..., 1:, :] * valid[..., :-1, :]
    return 0.5 * (
        masked_mean((pred_x - target_x).abs(), valid_x)
        + masked_mean((pred_y - target_y).abs(), valid_y)
    )


def _effective_valid(inputs: Mapping[str, Tensor], labels: Mapping[str, Tensor]) -> Tensor:
    target_valid = labels["target_valid"]
    anchor_valid = inputs["target_anchor_valid"].to(target_valid)
    if anchor_valid.shape != target_valid.shape:
        raise ValueError("target_anchor_valid must match target_valid")
    return target_valid * anchor_valid


def _optical_spectral_loss(prediction: Tensor, target: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    normalized_prediction = F.normalize(prediction, dim=1, eps=1e-6)
    normalized_target = F.normalize(target, dim=1, eps=1e-6)
    cosine = (normalized_prediction * normalized_target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    spectral = masked_mean(1.0 - cosine, valid)
    if prediction.shape[1] <= 6:
        return spectral, prediction.new_zeros(())
    # Paired temporal V4 keeps the canonical S2 ordering: red=B04 index 2,
    # NIR=B08 index 6.  This is intentionally a target-only supervision term.
    pred_reflectance = (prediction + 1.0) * 0.5
    target_reflectance = (target + 1.0) * 0.5
    predicted_ndvi = (pred_reflectance[:, 6:7] - pred_reflectance[:, 2:3]) / (
        pred_reflectance[:, 6:7] + pred_reflectance[:, 2:3] + 1e-4
    )
    target_ndvi = (target_reflectance[:, 6:7] - target_reflectance[:, 2:3]) / (
        target_reflectance[:, 6:7] + target_reflectance[:, 2:3] + 1e-4
    )
    return spectral, masked_mean((predicted_ndvi - target_ndvi).abs(), valid)


def _sar_statistics_loss(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    mask = _resize_valid(valid, prediction).expand_as(prediction)
    count = mask.flatten(2).sum(dim=2).clamp_min(1.0)
    # Variance can be exactly zero on a valid uniform SAR patch.  Computing a
    # square-root derivative at zero then creates inf gradients even when the
    # forward loss is finite, so retain this statistic in FP32 with a small
    # positive floor.
    prediction32 = prediction.float()
    target32 = target.float()
    mask32 = mask.float()
    count = mask32.flatten(2).sum(dim=2).clamp_min(1.0)
    predicted_mean = (prediction32 * mask32).flatten(2).sum(dim=2) / count
    target_mean = (target32 * mask32).flatten(2).sum(dim=2) / count
    predicted_var = (
        ((prediction32 - predicted_mean[..., None, None]).square() * mask32)
        .flatten(2)
        .sum(dim=2)
        / count
    )
    target_var = (
        ((target32 - target_mean[..., None, None]).square() * mask32)
        .flatten(2)
        .sum(dim=2)
        / count
    )
    standard_deviation_epsilon = 1e-6
    return (predicted_mean - target_mean).abs().mean() + (
        (predicted_var.clamp_min(0.0) + standard_deviation_epsilon).sqrt()
        - (target_var.clamp_min(0.0) + standard_deviation_epsilon).sqrt()
    ).abs().mean()


def _common_private_decorrelation(common: Tensor, private: Tensor, valid: Tensor) -> Tensor:
    common_map = common.mean(dim=1, keepdim=True)
    private_map = private.mean(dim=1, keepdim=True)
    mask = _resize_valid(valid, common_map)
    if private_map.shape[-2:] != common_map.shape[-2:]:
        private_map = F.interpolate(private_map, size=common_map.shape[-2:], mode="bilinear", align_corners=False)
    denominator = mask.flatten(1).sum(dim=1, keepdim=True).clamp_min(1.0)
    common_mean = (common_map * mask).flatten(1).sum(dim=1, keepdim=True) / denominator
    private_mean = (private_map * mask).flatten(1).sum(dim=1, keepdim=True) / denominator
    centered_common = (common_map - common_mean[..., None, None]) * mask
    centered_private = (private_map - private_mean[..., None, None]) * mask
    covariance = (centered_common * centered_private).flatten(1).sum(dim=1)
    common_energy = centered_common.square().flatten(1).sum(dim=1).clamp_min(1e-6)
    private_energy = centered_private.square().flatten(1).sum(dim=1).clamp_min(1e-6)
    return (covariance.square() / (common_energy * private_energy)).mean()


def factorizer_objective(
    output: object,
    inputs: Mapping[str, Tensor],
    config: SOPATTrainConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Anchor-only factorization objective for shared/common/private states."""

    source_cross = output_tensor(
        output,
        "source_anchor_reconstruction",
        "source_anchor_cross_reconstruction",
        "reconstructed_source_anchor",
        "source_anchor_cross",
    )
    target_cross = output_tensor(
        output,
        "target_anchor_reconstruction",
        "target_anchor_cross_reconstruction",
        "reconstructed_target_anchor",
        "target_anchor_cross",
    )
    common_source = output_tensor(output, "common_source", "source_common")
    common_target = output_tensor(output, "common_target", "target_common")
    private_source = output_tensor(output, "private_source", "source_private")
    private_target = output_tensor(output, "private_target", "target_private")
    assert source_cross is not None
    assert target_cross is not None
    assert common_source is not None
    assert common_target is not None
    assert private_source is not None
    assert private_target is not None
    source_anchor = inputs["source_anchor"]
    target_anchor = inputs["target_anchor"]
    source_valid = inputs["source_anchor_valid"]
    target_valid = inputs["target_anchor_valid"]
    anchor_cross = 0.5 * (
        masked_charbonnier(source_cross, source_anchor, source_valid)
        + masked_charbonnier(target_cross, target_anchor, target_valid)
    )
    if common_source.shape != common_target.shape:
        raise ValueError("SOPAT common source and target states must share a shape")
    common_valid = _resize_valid(source_valid * target_valid, common_source)
    common_alignment = masked_mean((common_source - common_target).abs(), common_valid)
    private_decorrelation = 0.5 * (
        _common_private_decorrelation(common_source, private_source, common_valid)
        + _common_private_decorrelation(common_target, private_target, common_valid)
    )
    total = (
        config.factorizer_anchor_cross_weight * anchor_cross
        + config.factorizer_common_alignment_weight * common_alignment
        + config.factorizer_private_decorrelation_weight * private_decorrelation
    )
    return total, {
        "factorizer_anchor_cross": anchor_cross.detach(),
        "factorizer_common_alignment": common_alignment.detach(),
        "factorizer_private_decorrelation": private_decorrelation.detach(),
    }


def null_change_batch(batch: Mapping[str, object]) -> dict[str, object]:
    """Replace every source observation by its historical source anchor."""

    inputs = forward_input_tensors(batch)
    observations = inputs["observations"]
    source_anchor = inputs["source_anchor"]
    if observations.shape[0] != source_anchor.shape[0] or observations.shape[2:] != source_anchor.shape[1:]:
        raise ValueError("source anchor is incompatible with SOPAT observations")
    replacement = source_anchor[:, None].expand_as(observations).clone()
    replacement_valid = inputs["source_anchor_valid"][:, None].expand_as(inputs["observation_valid"]).clone()
    result = dict(batch)
    result["observations"] = replacement
    result["observation_valid"] = replacement_valid
    return result


def permutation_batch(
    batch: Mapping[str, object], *, generator: torch.Generator | None = None
) -> dict[str, object]:
    """Permute temporal slots jointly, preserving each observation's timestamp/mask."""

    inputs = forward_input_tensors(batch)
    frames = inputs["observations"].shape[1]
    if frames < 2:
        return dict(batch)
    order = torch.randperm(
        frames,
        device=inputs["observations"].device,
        generator=_generator_for_device(generator, inputs["observations"].device),
    )
    result = dict(batch)
    result["observations"] = inputs["observations"].index_select(1, order)
    result["observation_valid"] = inputs["observation_valid"].index_select(1, order)
    result["observation_days"] = inputs["observation_days"].index_select(1, order)
    result["observation_present"] = inputs["observation_present"].index_select(1, order)
    return result


def latest_only_batch(batch: Mapping[str, object]) -> dict[str, object]:
    """Keep the newest real source observation for a trained latest-only variant."""

    inputs = forward_input_tensors(batch)
    values = inputs["observations"]
    valid = inputs["observation_valid"]
    days = inputs["observation_days"]
    present = inputs["observation_present"].bool()
    if not bool(present.any(dim=1).all()):
        raise ValueError("each SOPAT sample needs at least one real observation")
    sentinel = torch.full_like(days, -float("inf"))
    selected = torch.where(present, days, sentinel).argmax(dim=1)
    batch_size = values.shape[0]
    gather_values = selected[:, None, None, None, None].expand(-1, 1, *values.shape[2:])
    gather_valid = selected[:, None, None, None, None].expand(-1, 1, *valid.shape[2:])
    gather_days = selected[:, None]
    result = dict(batch)
    result["observations"] = values.gather(1, gather_values)
    result["observation_valid"] = valid.gather(1, gather_valid)
    result["observation_days"] = days.gather(1, gather_days)
    result["observation_present"] = torch.ones(
        (batch_size, 1), dtype=torch.bool, device=values.device
    )
    return result


def source_shuffle_batch(
    batch: Mapping[str, object], *, generator: torch.Generator | None = None
) -> dict[str, object]:
    """Counterfactually exchange source histories with a local derangement.

    A random permutation can retain a scene's own source history, making the
    counterfactual silently invalid.  A non-zero cyclic shift is a derangement
    by construction.  The singleton case is intentionally left unchanged here;
    ``physical_objective`` either uses a cross-rank donor or records a zero
    counterfactual term when no donor exists.
    """

    inputs = forward_input_tensors(batch)
    batch_size = inputs["observations"].shape[0]
    if batch_size < 2:
        return dict(batch)
    device = inputs["observations"].device
    offset = torch.randint(
        1,
        batch_size,
        (),
        device=device,
        generator=_generator_for_device(generator, device),
    )
    order = (torch.arange(batch_size, device=device) + offset) % batch_size
    if bool((order == torch.arange(batch_size, device=device)).any()):
        raise RuntimeError("SOPAT source-shuffle derangement construction failed")
    result = dict(batch)
    for name in ("observations", "observation_valid", "observation_days", "observation_present"):
        result[name] = inputs[name].index_select(0, order)
    return result


def _distributed_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _rank_synchronized_probability(
    probability: float, device: torch.device, generator: torch.Generator | None
) -> bool:
    """Sample one Bernoulli decision per global step, never per DDP rank."""

    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    if _distributed_world_size() == 1:
        return _sample_probability(probability, device, generator)
    flag = torch.zeros((), device=device, dtype=torch.int64)
    if dist.get_rank() == 0:
        flag.fill_(int(_sample_probability(probability, device, generator)))
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def _padded_cross_rank_source_histories(inputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Gather one singleton history per rank and select a non-self donor.

    The gathered tensors are detached source inputs.  Their subsequent wrong
    forward remains fully differentiable with respect to model parameters.
    Every rank enters the same collectives, and variable observation counts are
    padded to the global maximum before gathering.
    """

    if _distributed_world_size() < 2:
        raise RuntimeError("cross-rank source shuffle requires initialized world_size >= 2")
    observations = inputs["observations"]
    if observations.shape[0] != 1:
        raise ValueError("cross-rank source shuffle is only needed for local batch_size == 1")
    device = observations.device
    local_frames = torch.tensor([observations.shape[1]], device=device, dtype=torch.int64)
    maximum_frames = local_frames.clone()
    dist.all_reduce(maximum_frames, op=dist.ReduceOp.MAX)
    frames = int(maximum_frames.item())

    def pad_frames(values: Tensor, *, fill: float = 0.0) -> Tensor:
        if values.shape[1] == frames:
            return values.detach()
        padded_shape = (values.shape[0], frames, *values.shape[2:])
        padded = torch.full(padded_shape, fill, dtype=values.dtype, device=values.device)
        padded[:, : values.shape[1]] = values.detach()
        return padded

    padded = {
        "observations": pad_frames(inputs["observations"]),
        "observation_valid": pad_frames(inputs["observation_valid"]),
        "observation_days": pad_frames(inputs["observation_days"]),
        "observation_present": pad_frames(inputs["observation_present"].to(torch.uint8)).bool(),
    }
    gathered: dict[str, list[Tensor]] = {}
    for name, values in padded.items():
        destination = [torch.empty_like(values) for _ in range(_distributed_world_size())]
        # bool NCCL collective support is backend-dependent; use uint8 for the
        # actual gather and restore the public boolean mask afterwards.
        source = values.to(torch.uint8) if name == "observation_present" else values
        if name == "observation_present":
            destination = [torch.empty_like(source) for _ in range(_distributed_world_size())]
        dist.all_gather(destination, source)
        gathered[name] = destination
    donor_rank = (dist.get_rank() + 1) % _distributed_world_size()
    return {
        "observations": gathered["observations"][donor_rank],
        "observation_valid": gathered["observation_valid"][donor_rank],
        "observation_days": gathered["observation_days"][donor_rank],
        "observation_present": gathered["observation_present"][donor_rank].bool(),
    }


def _source_counterfactual_batch(
    batch: Mapping[str, object],
    *,
    mode: SourceCounterfactualMode = "legacy_local_shuffle_v1",
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, object] | None:
    """Build a non-self source-history counterfactual for the configured contract.

    ``global_cross_tile_v1`` accepts only the deterministic donor payload
    pre-collated by the data pipeline.  It intentionally performs no DDP
    collective: each recipient already has its split-global cross-tile donor.
    ``legacy_local_shuffle_v1`` preserves the historical local/cross-rank
    implementation for old tests and checkpoints.
    """

    if mode == "global_cross_tile_v1":
        return _global_cross_tile_counterfactual_batch(batch, device=device)
    if mode != "legacy_local_shuffle_v1":
        raise ValueError(f"unsupported SOPAT source counterfactual mode: {mode}")

    inputs = forward_input_tensors(batch, device=device)
    batch_size = inputs["observations"].shape[0]
    world_size = _distributed_world_size()
    if batch_size >= 2:
        result = source_shuffle_batch(batch, generator=generator)
        _assert_counterfactual_recipient_preserved(batch, result)
        return result
    if world_size == 1:
        return None

    local_size = torch.tensor([batch_size], device=inputs["observations"].device, dtype=torch.int64)
    sizes = [torch.empty_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    gathered_sizes = [int(item.item()) for item in sizes]
    if any(size != 1 for size in gathered_sizes):
        raise RuntimeError(
            "SOPAT cross-rank source shuffle requires every rank to use local batch_size == 1; "
            f"got {gathered_sizes}"
        )
    result = dict(batch)
    result.update(_padded_cross_rank_source_histories(inputs))
    _assert_counterfactual_recipient_preserved(batch, result)
    return result


_GLOBAL_COUNTERFACTUAL_METADATA: tuple[str, ...] = (
    "sopat_example_id",
    "sopat_tile",
    "sopat_grid_id",
    "sopat_cf_donor_sample_id",
    "sopat_cf_donor_tile",
    "sopat_cf_donor_grid_id",
    "sopat_cf_tier",
    "sopat_cf_plan_hash",
)
_GLOBAL_COUNTERFACTUAL_TIERS = frozenset(
    {"same_task_exact_n", "same_task_n_bin", "same_task", "same_orbit"}
)


def _global_counterfactual_metadata(
    batch: Mapping[str, object], *, batch_size: int
) -> dict[str, tuple[str, ...]]:
    """Read the public split-global donor identity contract without coercion."""

    result: dict[str, tuple[str, ...]] = {}
    for name in _GLOBAL_COUNTERFACTUAL_METADATA:
        value = batch.get(name)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(
                f"SOPAT global source counterfactual requires batch-aligned {name} metadata"
            )
        if len(value) != batch_size:
            raise ValueError(
                f"SOPAT global source counterfactual {name} metadata must have length {batch_size}"
            )
        values = tuple(value)
        if any(not isinstance(item, str) or not item for item in values):
            raise ValueError(
                f"SOPAT global source counterfactual {name} metadata must contain non-empty strings"
            )
        result[name] = values  # type: ignore[assignment]
    invalid_tiers = sorted(set(result["sopat_cf_tier"]).difference(_GLOBAL_COUNTERFACTUAL_TIERS))
    if invalid_tiers:
        raise ValueError(
            "SOPAT global source counterfactual has unsupported donor tier "
            f"{invalid_tiers[0]!r}"
        )
    return result


_COUNTERFACTUAL_HISTORY_KEYS = frozenset(
    {"observations", "observation_valid", "observation_days", "observation_present"}
)


def _assert_counterfactual_recipient_preserved(
    original: Mapping[str, object], result: Mapping[str, object]
) -> None:
    """Guard the causal recipient state against accidental donor replacement."""

    for name, value in original.items():
        if name not in _COUNTERFACTUAL_HISTORY_KEYS and result.get(name) is not value:
            raise AssertionError(
                f"SOPAT source counterfactual must preserve recipient field {name}"
            )


def _assert_global_counterfactual_recipient_history(
    original: Mapping[str, object], result: Mapping[str, object]
) -> None:
    """Ensure a global donor never changes the recipient temporal contract."""

    _assert_counterfactual_recipient_preserved(original, result)
    for name in ("observation_days", "observation_present"):
        if result.get(name) is not original.get(name):
            raise AssertionError(
                f"SOPAT global source counterfactual must retain recipient {name}"
            )


def _global_cross_tile_counterfactual_batch(
    batch: Mapping[str, object], *, device: torch.device | None = None
) -> dict[str, object]:
    """Substitute one validated pre-collated global donor history per recipient."""

    inputs = forward_input_tensors(batch, device=device)
    observations = inputs["observations"]
    observation_valid = inputs["observation_valid"]
    observation_days = inputs["observation_days"]
    observation_present = inputs["observation_present"]
    batch_size = observations.shape[0]
    if observations.ndim != 5:
        raise ValueError("SOPAT recipient observations must have shape BxTxCxHxW")
    if observation_valid.shape != (
        batch_size,
        observations.shape[1],
        1,
        *observations.shape[-2:],
    ):
        raise ValueError("SOPAT recipient observation_valid must have shape BxTx1xHxW")
    if observation_days.shape != observations.shape[:2]:
        raise ValueError("SOPAT recipient observation_days must have shape BxT")
    if observation_present.shape != observations.shape[:2]:
        raise ValueError("SOPAT recipient observation_present must have shape BxT")
    values = batch.get("counterfactual_observation_values")
    valid = batch.get("counterfactual_observation_valid")
    if not isinstance(values, Tensor):
        raise TypeError(
            "SOPAT global source counterfactual requires counterfactual_observation_values"
        )
    if not isinstance(valid, Tensor):
        raise TypeError(
            "SOPAT global source counterfactual requires counterfactual_observation_valid"
        )
    if values.shape != observations.shape:
        raise ValueError(
            "SOPAT counterfactual_observation_values must match recipient observations shape"
        )
    if valid.shape != observation_valid.shape:
        raise ValueError(
            "SOPAT counterfactual_observation_valid must match recipient observation_valid shape"
        )
    values = values.to(device=observations.device, dtype=observations.dtype, non_blocking=True)
    valid = valid.to(
        device=observation_valid.device,
        dtype=observation_valid.dtype,
        non_blocking=True,
    )
    metadata = _global_counterfactual_metadata(batch, batch_size=batch_size)
    for recipient, donor, label in (
        ("sopat_example_id", "sopat_cf_donor_sample_id", "sample"),
        ("sopat_tile", "sopat_cf_donor_tile", "tile"),
    ):
        if any(left == right for left, right in zip(metadata[recipient], metadata[donor], strict=True)):
            raise ValueError(
                f"SOPAT global source counterfactual donor must differ from recipient {label}"
            )
    result = dict(batch)
    # Preserve recipient chronology/presence and every anchor/label field.
    result["observations"] = values
    result["observation_valid"] = valid
    _assert_global_counterfactual_recipient_history(batch, result)
    return result


def _local_masked_mean(values: Tensor, valid: Tensor, *, kernel_size: int) -> tuple[Tensor, Tensor]:
    """Return a local mean and local support without invalid-value leakage."""

    mask = _resize_valid(valid, values).to(values)
    safe_values = torch.where(mask.bool(), values, torch.zeros_like(values))
    if kernel_size <= 1:
        support = mask.expand_as(values)
        return safe_values * support / support.clamp_min(1e-6), support
    padding = kernel_size // 2
    numerator = F.avg_pool2d(
        (safe_values.float() * mask.float()),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    denominator = F.avg_pool2d(
        mask.float(),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    )
    return (numerator / denominator.clamp_min(1e-6)).to(values), denominator.to(values)


def _structural_error(
    prediction: Tensor, target: Tensor, valid: Tensor, *, kernel_size: int
) -> Tensor:
    local_error, support = _local_masked_mean((prediction - target).square(), valid, kernel_size=kernel_size)
    local_valid = (support > 0.0).to(valid)
    mask = local_valid.expand_as(local_error)
    numerator = (local_error * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (numerator / denominator + 1e-8).sqrt()


def _low_pass_difference(
    matched: Tensor,
    counterfactual: Tensor,
    valid: Tensor,
    *,
    kernel_size: int,
) -> tuple[Tensor, Tensor]:
    """Return a mask-aware low-pass source effect and its spatial support."""

    if matched.shape != counterfactual.shape:
        raise ValueError("SOPAT counterfactual candidate must match matched candidate shape")
    matched_low_pass, matched_support = _local_masked_mean(matched, valid, kernel_size=kernel_size)
    counterfactual_low_pass, counterfactual_support = _local_masked_mean(
        counterfactual, valid, kernel_size=kernel_size
    )
    support = (matched_support > 0.0).to(matched) * (counterfactual_support > 0.0).to(matched)
    # ``kernel_size == 1`` preserves the candidate channel count in
    # ``_local_masked_mean``.  Downstream support is spatial rather than
    # channel-specific, so normalize it to the public Bx1xHxW mask shape.
    return (matched_low_pass - counterfactual_low_pass).abs(), support[:, :1]


def _utility_oracle(
    candidate: Tensor,
    anchor: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    temperature: float,
    kernel_size: int,
) -> Tensor:
    """Detached per-pixel probability that the candidate beats anchor locally."""

    candidate_error, support = _local_masked_mean(
        (candidate - target).abs(), valid, kernel_size=kernel_size
    )
    anchor_error, _ = _local_masked_mean((anchor - target).abs(), valid, kernel_size=kernel_size)
    candidate_error = candidate_error.mean(dim=1, keepdim=True)
    anchor_error = anchor_error.mean(dim=1, keepdim=True)
    support = (support > 0.0).to(candidate_error)
    support = support[:, :1]
    neutral = torch.full_like(candidate_error, 0.5)
    oracle = torch.sigmoid((anchor_error - candidate_error) / temperature)
    return torch.where(support > 0.0, oracle, neutral).detach()


def physical_objective(
    model: SOPATForwardProtocol | nn.Module,
    output: object,
    batch: Mapping[str, object],
    inputs: Mapping[str, Tensor],
    labels: Mapping[str, Tensor],
    direction: str,
    config: SOPATTrainConfig,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Physical loss with target-only labels and sampled causal counterfactuals."""

    direction = _validate_direction(direction)
    physical = output_tensor(output, "physical")
    assert physical is not None
    target = labels["target"]
    valid = _effective_valid(inputs, labels)
    if physical.shape != target.shape:
        raise ValueError("SOPAT physical output must match the target label shape")
    charbonnier = masked_charbonnier(physical, target, valid)
    gradient = _gradient_loss(physical, target, valid)
    if direction == "sar_to_optical":
        spectral, ndvi = _optical_spectral_loss(physical, target, valid)
        sar_statistics = physical.new_zeros(())
    else:
        spectral = physical.new_zeros(())
        ndvi = physical.new_zeros(())
        sar_statistics = _sar_statistics_loss(physical, target, valid)
    target_delta = target - inputs["target_anchor"]
    predicted_delta = physical - inputs["target_anchor"]
    anchor_delta = masked_charbonnier(predicted_delta, target_delta, valid)
    candidate_physical = output_tensor(output, "candidate_physical", required=False)
    if candidate_physical is None:
        # Compatibility for focused unit models and pre-gated experimental
        # checkpoints.  Native V4 renderers always expose this field.
        candidate_physical = physical
    if candidate_physical.shape != target.shape:
        raise ValueError("SOPAT candidate_physical must match the target label shape")
    candidate = masked_charbonnier(candidate_physical, target, valid)
    confidence_shape = (physical.shape[0], 1, *physical.shape[-2:])
    matched_transport_confidence = output_tensor(
        output, "transport_confidence", required=False
    )
    transport_confidence = matched_transport_confidence
    if transport_confidence is None:
        transport_confidence = torch.ones(confidence_shape, device=physical.device, dtype=physical.dtype)
    if transport_confidence.shape != confidence_shape:
        raise ValueError("SOPAT transport_confidence must have shape Bx1xHxW")
    transport_confidence_logits = output_tensor(
        output, "transport_confidence_logits", required=False
    )
    if transport_confidence_logits is not None and transport_confidence_logits.shape != confidence_shape:
        raise ValueError("SOPAT transport_confidence_logits must have shape Bx1xHxW")
    transport_evidence = output_tensor(output, "transport_evidence", required=False)
    if transport_evidence is None:
        # Older V4 checkpoints and focused toy models expose only a
        # probability.  Preserve their historical utility supervision.
        transport_evidence = torch.ones_like(transport_confidence)
    if transport_evidence.shape != confidence_shape:
        raise ValueError("SOPAT transport_evidence must have shape Bx1xHxW")
    utility_target = _utility_oracle(
        candidate_physical,
        inputs["target_anchor"],
        target,
        valid,
        temperature=config.utility_temperature,
        kernel_size=config.structural_pool_kernel,
    )
    # Source-null and source-invalid pixels must remain an exact anchor-copy
    # route.  They are intentionally excluded from gate supervision rather
    # than assigned an arbitrary 0.5 "neutral" label.  Native SOPAT provides
    # raw logits so BCE remains numerically safe under CUDA BF16 autocast.
    utility_valid = valid * (transport_evidence > 0.0).to(valid)
    with torch.autocast(device_type=physical.device.type, enabled=False):
        if transport_confidence_logits is not None:
            utility_map = F.binary_cross_entropy_with_logits(
                transport_confidence_logits.float(), utility_target.float(), reduction="none"
            )
        else:
            # Compatibility fallback for old checkpoints and compact test
            # models.  Probability BCE is explicitly outside autocast because
            # PyTorch rejects it in mixed-precision CUDA execution.
            utility_map = F.binary_cross_entropy(
                transport_confidence.float().clamp(1e-5, 1.0 - 1e-5),
                utility_target.float(),
                reduction="none",
            )
        utility = _masked_single_channel_mean(utility_map, utility_valid)
    squared_error = (physical - target).square()
    log_variance = output_tensor(output, "log_variance", "physical_log_variance", required=False)
    if log_variance is None:
        log_variance = torch.zeros_like(physical)
    if log_variance.shape != physical.shape:
        if log_variance.shape == (physical.shape[0], 1, *physical.shape[-2:]):
            log_variance = log_variance.expand_as(physical)
        else:
            raise ValueError("SOPAT log_variance must have BxCxHxW or Bx1xHxW shape")
    nll = 0.5 * masked_mean(squared_error * (-log_variance).exp() + log_variance, valid)
    per_example_error = _per_example_stable_rmse(physical, target, valid)
    anchor_error = _per_example_rmse(inputs["target_anchor"], target, valid)
    anchor_regret = torch.relu(per_example_error - anchor_error + config.physical_anchor_regret_margin).mean()
    null_change = physical.new_zeros(())
    source_evidence = physical.new_zeros(())
    if _sample_probability(config.physical_null_change_probability, physical.device, generator):
        null_output = forward_direction(model, null_change_batch(batch), direction, device=physical.device)
        null_physical = output_tensor(null_output, "physical")
        assert null_physical is not None
        if null_physical.shape != physical.shape:
            raise ValueError("SOPAT null-change physical output must match the target label shape")
        # This is an identity constraint, not a detached comparison to the
        # query label.  With every source observation replaced by the historic
        # source anchor, the model must recover the paired target anchor.
        null_change = _per_example_stable_rmse(
            null_physical,
            inputs["target_anchor"],
            inputs["target_anchor_valid"],
        ).mean()
        # Retain real-versus-null source evidence as a diagnostic only.  It
        # must never replace the differentiable null identity objective.
        null_error = _per_example_rmse(null_physical.detach(), target, valid)
        source_evidence = torch.relu(
            per_example_error - null_error + config.physical_anchor_regret_margin
        ).mean()
    permutation = physical.new_zeros(())
    if _sample_probability(config.physical_permutation_probability, physical.device, generator):
        permuted_output = forward_direction(
            model, permutation_batch(batch, generator=generator), direction, device=physical.device
        )
        permuted_physical = output_tensor(permuted_output, "physical")
        assert permuted_physical is not None
        permutation = masked_mean((physical - permuted_physical).abs(), valid)
    source_shuffle = physical.new_zeros(())
    counterfactual_candidate_ranking = physical.new_zeros(())
    counterfactual_source_effect_floor = physical.new_zeros(())
    counterfactual_candidate_effect = physical.new_zeros(())
    counterfactual_confidence = physical.new_zeros(())
    counterfactual_confidence_binary = physical.new_zeros(())
    counterfactual_confidence_margin = physical.new_zeros(())
    # The wrong forward must be present in every DDP rank's graph.  Global
    # mode avoids donor-construction collectives; this one-step Bernoulli
    # synchronization only decides whether all ranks execute the same graph.
    source_shuffle_enabled = _rank_synchronized_probability(
        config.source_shuffle_probability, physical.device, generator
    )
    if source_shuffle_enabled:
        counterfactual_batch = _source_counterfactual_batch(
            batch,
            mode=config.source_counterfactual_mode,
            device=physical.device,
            generator=generator,
        )
        if counterfactual_batch is not None:
            wrong_output = forward_direction(
                model, counterfactual_batch, direction, device=physical.device
            )
            wrong_physical = output_tensor(wrong_output, "physical")
            assert wrong_physical is not None
            if wrong_physical.shape != physical.shape:
                raise ValueError("SOPAT source-shuffle physical output must match target labels")
            wrong_candidate_physical = output_tensor(
                wrong_output, "candidate_physical", required=False
            )
            if wrong_candidate_physical is None:
                wrong_candidate_physical = wrong_physical
            if wrong_candidate_physical.shape != candidate_physical.shape:
                raise ValueError(
                    "SOPAT source-shuffle candidate_physical must match matched candidate shape"
                )
            correct_error = _structural_error(
                physical,
                target,
                valid,
                kernel_size=config.structural_pool_kernel,
            )
            wrong_error = _structural_error(
                wrong_physical,
                target,
                valid,
                kernel_size=config.structural_pool_kernel,
            )
            source_shuffle = torch.relu(
                correct_error - wrong_error + config.source_shuffle_margin
            ).mean()
            # A shuffled history is a deliberately wrong causal source.  Its
            # confidence must be lower than the matched route by a margin.
            # The binary source-authenticity term below additionally prevents
            # a collapsed pair of closed gates from satisfying that ranking.
            wrong_confidence_shape = (wrong_physical.shape[0], 1, *wrong_physical.shape[-2:])
            wrong_transport_confidence_logits = output_tensor(
                wrong_output, "transport_confidence_logits", required=False
            )
            wrong_transport_confidence = output_tensor(
                wrong_output, "transport_confidence", required=False
            )
            wrong_transport_evidence = output_tensor(
                wrong_output, "transport_evidence", required=False
            )
            if wrong_transport_confidence_logits is not None and (
                wrong_transport_confidence_logits.shape != wrong_confidence_shape
            ):
                raise ValueError(
                    "SOPAT source-shuffle transport_confidence_logits must have shape Bx1xHxW"
                )
            if wrong_transport_confidence is not None and (
                wrong_transport_confidence.shape != wrong_confidence_shape
            ):
                raise ValueError(
                    "SOPAT source-shuffle transport_confidence must have shape Bx1xHxW"
                )
            if wrong_transport_evidence is not None and (
                wrong_transport_evidence.shape != wrong_confidence_shape
            ):
                raise ValueError(
                    "SOPAT source-shuffle transport_evidence must have shape Bx1xHxW"
                )
            if wrong_transport_evidence is None:
                # Candidate ranking/effect supervision remains available for
                # pre-evidence checkpoints over target-valid recipient support.
                wrong_transport_evidence = torch.ones(
                    wrong_confidence_shape,
                    device=wrong_physical.device,
                    dtype=wrong_physical.dtype,
                )
            recipient_source_anchor_valid = inputs["source_anchor_valid"]
            if recipient_source_anchor_valid.shape != wrong_confidence_shape:
                raise ValueError(
                    "SOPAT source_anchor_valid must match source-shuffle confidence shape"
                )
            counterfactual_valid = (
                valid
                * recipient_source_anchor_valid.to(valid)
                * (transport_evidence > 0.0).to(valid)
                * (wrong_transport_evidence > 0.0).to(valid)
            )
            counterfactual_example_support = (
                counterfactual_valid.flatten(1).sum(dim=1) > 0.0
            ).to(physical)
            correct_candidate_error = _structural_error(
                candidate_physical,
                target,
                counterfactual_valid,
                kernel_size=config.structural_pool_kernel,
            )
            wrong_candidate_error = _structural_error(
                wrong_candidate_physical,
                target,
                counterfactual_valid,
                kernel_size=config.structural_pool_kernel,
            )
            counterfactual_candidate_ranking_values = torch.relu(
                correct_candidate_error
                - wrong_candidate_error
                + float(config.counterfactual_candidate_ranking_margin)
            )
            counterfactual_candidate_ranking = (
                counterfactual_candidate_ranking_values * counterfactual_example_support
            ).sum() / counterfactual_example_support.sum().clamp_min(1.0)
            low_pass_difference, low_pass_support = _low_pass_difference(
                candidate_physical,
                wrong_candidate_physical,
                counterfactual_valid,
                kernel_size=config.structural_pool_kernel,
            )
            low_pass_mask = _resize_valid(low_pass_support, low_pass_difference).expand_as(
                low_pass_difference
            )
            low_pass_support_count = low_pass_mask.sum()
            counterfactual_candidate_effect = (
                low_pass_difference * low_pass_mask
            ).sum() / low_pass_support_count.clamp_min(1.0)
            counterfactual_source_effect_floor = torch.where(
                low_pass_support_count > 0.0,
                torch.relu(
                    low_pass_difference.new_tensor(config.counterfactual_source_effect_floor)
                    - counterfactual_candidate_effect
                ),
                low_pass_difference.new_zeros(()),
            )
            # ``transport_confidence`` above has a ones fallback for the
            # historical utility objective.  Counterfactual calibration needs
            # a real matched gate, otherwise legacy outputs must remain a
            # strict no-op.
            matched_has_confidence = (
                transport_confidence_logits is not None
                or matched_transport_confidence is not None
            )
            wrong_has_confidence = (
                wrong_transport_confidence_logits is not None
                or wrong_transport_confidence is not None
            )
            if matched_has_confidence and wrong_has_confidence:
                # Source authenticity has exactly two labels: this unshuffled
                # forward received the recipient's matched input history, and
                # this counterfactual forward received a non-self history.
                # No query target values participate in this calibration.
                # Keep BCE and the probability margin in FP32 regardless of
                # the enclosing BF16 autocast context. Native SOPAT exposes
                # logits; probability branches retain focused-test and legacy
                # checkpoint compatibility.
                with torch.autocast(device_type=physical.device.type, enabled=False):
                    matched_confidence_target = (
                        utility_target.float()
                        if config.source_counterfactual_mode == "global_cross_tile_v1"
                        else torch.ones_like(transport_confidence.float())
                    )
                    if transport_confidence_logits is not None:
                        matched_logits = transport_confidence_logits.float()
                        matched_probability = torch.sigmoid(matched_logits)
                        matched_binary_map = F.binary_cross_entropy_with_logits(
                            matched_logits,
                            matched_confidence_target,
                            reduction="none",
                        )
                    else:
                        matched_probability = transport_confidence.float().clamp(0.0, 1.0)
                        matched_binary_map = F.binary_cross_entropy(
                            matched_probability.clamp(1e-5, 1.0 - 1e-5),
                            matched_confidence_target,
                            reduction="none",
                        )
                    if wrong_transport_confidence_logits is not None:
                        wrong_logits = wrong_transport_confidence_logits.float()
                        wrong_probability = torch.sigmoid(wrong_logits)
                        wrong_binary_map = F.binary_cross_entropy_with_logits(
                            wrong_logits,
                            torch.zeros_like(wrong_logits),
                            reduction="none",
                        )
                    else:
                        assert wrong_transport_confidence is not None
                        wrong_probability = wrong_transport_confidence.float().clamp(
                            0.0, 1.0
                        )
                        wrong_binary_map = F.binary_cross_entropy(
                            wrong_probability.clamp(1e-5, 1.0 - 1e-5),
                            torch.zeros_like(wrong_probability),
                            reduction="none",
                        )
                    counterfactual_confidence_binary = _masked_single_channel_mean(
                        0.5 * (matched_binary_map + wrong_binary_map),
                        counterfactual_valid,
                    )
                    counterfactual_confidence_margin = _masked_single_channel_mean(
                        torch.relu(
                            wrong_probability
                            - matched_probability
                            + float(config.counterfactual_confidence_margin)
                        ) * matched_confidence_target,
                        counterfactual_valid,
                    )
                    counterfactual_confidence = (
                        float(config.counterfactual_confidence_binary_weight)
                        * counterfactual_confidence_binary
                        + counterfactual_confidence_margin
                    )
    total = (
        config.physical_charbonnier_weight * charbonnier
        + config.physical_gradient_weight * gradient
        + config.physical_optical_spectral_weight * spectral
        + config.physical_optical_ndvi_weight * ndvi
        + config.physical_sar_statistics_weight * sar_statistics
        + config.physical_anchor_delta_weight * anchor_delta
        + config.physical_null_change_weight * null_change
        + config.physical_nll_weight * nll
        + config.physical_permutation_weight * permutation
        + config.physical_anchor_regret_weight * anchor_regret
        + config.candidate_weight * candidate
        + config.utility_weight * utility
        + config.source_shuffle_weight * source_shuffle
        + config.counterfactual_candidate_ranking_weight * counterfactual_candidate_ranking
        + config.counterfactual_source_effect_floor_weight * counterfactual_source_effect_floor
        + config.counterfactual_confidence_weight * counterfactual_confidence
    )
    return total, {
        "physical_charbonnier": charbonnier.detach(),
        "physical_gradient": gradient.detach(),
        "physical_spectral": spectral.detach(),
        "physical_ndvi": ndvi.detach(),
        "physical_sar_statistics": sar_statistics.detach(),
        "physical_anchor_delta": anchor_delta.detach(),
        "physical_candidate": candidate.detach(),
        "physical_utility": utility.detach(),
        "physical_null_change": null_change.detach(),
        "physical_source_evidence": source_evidence.detach(),
        "physical_nll": nll.detach(),
        "physical_permutation": permutation.detach(),
        "physical_source_shuffle": source_shuffle.detach(),
        "physical_counterfactual_candidate_ranking": counterfactual_candidate_ranking.detach(),
        "physical_counterfactual_source_effect_floor": counterfactual_source_effect_floor.detach(),
        "physical_counterfactual_candidate_effect": counterfactual_candidate_effect.detach(),
        "physical_counterfactual_confidence": counterfactual_confidence.detach(),
        "physical_counterfactual_confidence_binary": counterfactual_confidence_binary.detach(),
        "physical_counterfactual_confidence_margin": counterfactual_confidence_margin.detach(),
        "physical_anchor_regret": anchor_regret.detach(),
        "physical_rmse": per_example_error.mean().detach(),
        "anchor_rmse": anchor_error.mean().detach(),
    }


def sopat_direction_objective(
    model: SOPATForwardProtocol | nn.Module,
    batch: Mapping[str, object],
    direction: str,
    config: SOPATTrainConfig,
    *,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Forward one direction and apply its selected stage objective."""

    direction = _validate_direction(direction)
    inputs = forward_input_tensors(batch, device=device)
    if config.stage == "factorizer":
        output = factorize_sopat_anchors(model, batch, direction, device=device)
        return factorizer_objective(output, inputs, config)
    output = forward_direction(model, batch, direction, device=device)
    labels = supervision_tensors(batch, device=device)
    return physical_objective(model, output, batch, inputs, labels, direction, config, generator=generator)


def factorize_sopat_anchors(
    model: SOPATForwardProtocol | nn.Module,
    batch: Mapping[str, object],
    direction: str,
    *,
    device: torch.device | None = None,
) -> object:
    """Run the anchor-only core route used by the factorizer stage.

    The call intentionally contains no observations, time sequence, or query
    target labels.  Core models without the public shortcut retain the causal
    full-forward compatibility path, but current SOPAT V4 implements it and
    avoids encoding an otherwise unused history stack.
    """

    direction = _validate_direction(direction)
    shortcut = getattr(model, "factorize_anchors", None)
    if not callable(shortcut):
        return forward_direction(model, batch, direction, device=device)
    inputs = forward_input_tensors(batch, device=device)
    source_sensor, target_sensor = direction_sensors(direction)
    return shortcut(
        source_anchor=inputs["source_anchor"],
        source_anchor_valid=inputs["source_anchor_valid"],
        target_anchor=inputs["target_anchor"],
        target_anchor_valid=inputs["target_anchor_valid"],
        source_sensor=_batch_sensor(batch, "source_sensor", source_sensor),
        target_sensor=_batch_sensor(batch, "target_sensor", target_sensor),
    )


class SOPATTrainingModule(nn.Module):
    """DDP-visible wrapper for one coupled, bidirectional model forward.

    DDP must observe both direction-specific renderers in the *same* forward.
    Splitting the directions over a ``no_sync`` microbatch and a synchronized
    microbatch leaves one renderer absent from the latter autograd graph, which
    can silently desynchronize its parameters across ranks.  This wrapper
    therefore computes the two homogeneous direction objectives sequentially
    inside one DDP-visible graph; the caller performs exactly one backward.
    """

    def __init__(self, model: SOPATForwardProtocol | nn.Module, config: SOPATTrainConfig) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("SOPAT training model must be an nn.Module")
        self.model = model
        self.config = config

    def forward(
        self,
        batches: Mapping[str, Mapping[str, object]],
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
        missing = set(DIRECTIONS).difference(batches)
        unexpected = set(batches).difference(DIRECTIONS)
        if missing or unexpected:
            raise ValueError(
                "SOPAT coupled forward requires exactly both directions; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        direction_losses: dict[str, Tensor] = {}
        metrics: dict[str, Tensor] = {}
        total: Tensor | None = None
        for direction in DIRECTIONS:
            loss, direction_metrics = sopat_direction_objective(
                self.model,
                batches[direction],
                direction,
                self.config,
                device=_module_device(self.model),
                generator=generator,
            )
            weighted = loss * float(self.config.direction_weights[direction])
            if not bool(torch.isfinite(weighted)):
                raise FloatingPointError(f"non-finite SOPAT {direction} loss")
            total = weighted if total is None else total + weighted
            direction_losses[direction] = loss.detach()
            for name, value in direction_metrics.items():
                metrics[f"{direction}/{name}"] = value.detach()
        assert total is not None
        return total, {"direction_losses": direction_losses, "metrics": metrics}


def configure_sopat_stage(
    model: nn.Module, stage: Stage, *, trainable_scope: TrainableScope = "full"
) -> None:
    """Delegate stage freezing to the core model when it exposes the contract."""

    if stage not in {"factorizer", "physical"}:
        raise ValueError("SOPAT stage must be factorizer or physical")
    if trainable_scope not in {"full", "confidence_only"}:
        raise ValueError("SOPAT trainable_scope must be full or confidence_only")
    if stage != "physical" and trainable_scope != "full":
        raise ValueError("confidence_only scope is valid only for the physical stage")
    configured = False
    for name in ("set_training_stage", "set_stage"):
        setter = getattr(model, name, None)
        if callable(setter):
            setter(stage)
            configured = True
            break
    if not configured and stage == "physical":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif not configured:
        # Explicit fallback for compact cores that expose stable factorizer names.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        enabled = 0
        for name in ("encoder", "factorizer", "anchor_reconstructors"):
            component = getattr(model, name, None)
            if not isinstance(component, nn.Module):
                continue
            for parameter in component.parameters():
                parameter.requires_grad_(True)
                enabled += 1
        if enabled == 0:
            raise RuntimeError(
                "SOPAT core must expose set_training_stage/set_stage or public "
                "factorizer components for factorizer training"
            )
    if trainable_scope == "full":
        return
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    renderers = getattr(model, "renderers", None)
    if not isinstance(renderers, nn.ModuleDict):
        raise TypeError("confidence_only scope requires model.renderers ModuleDict")
    enabled = 0
    for renderer in renderers.values():
        confidence = getattr(renderer, "confidence", None)
        if not isinstance(confidence, nn.Module):
            raise TypeError("confidence_only scope requires each renderer.confidence module")
        for parameter in confidence.parameters():
            parameter.requires_grad_(True)
            enabled += parameter.numel()
    if enabled == 0:
        raise RuntimeError("confidence_only scope selected no trainable parameters")


@dataclass
class ModelEMA:
    """Small CPU/GPU-safe EMA owned by one bidirectional SOPAT checkpoint."""

    decay: float
    state: dict[str, Tensor]

    @classmethod
    def create(cls, model: nn.Module, decay: float) -> ModelEMA:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        return cls(decay=float(decay), state=state)

    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        if set(current) != set(self.state):
            raise RuntimeError("SOPAT EMA state no longer matches model state")
        with torch.no_grad():
            for name, value in current.items():
                average = self.state[name]
                if average.shape != value.shape:
                    raise RuntimeError(f"SOPAT EMA tensor shape changed: {name}")
                if torch.is_floating_point(average):
                    average.lerp_(value.detach().to(average), 1.0 - self.decay)
                else:
                    average.copy_(value.detach().to(average))

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "state": {name: value.clone() for name, value in self.state.items()}}

    def load_state_dict(self, values: Mapping[str, object]) -> None:
        decay = values.get("decay")
        state = values.get("state")
        if not isinstance(decay, (int, float)) or not isinstance(state, Mapping):
            raise TypeError("invalid SOPAT EMA checkpoint state")
        loaded = {str(name): value for name, value in state.items() if isinstance(value, Tensor)}
        if set(loaded) != set(self.state):
            raise RuntimeError("SOPAT EMA checkpoint keys differ from model")
        for name, value in loaded.items():
            if value.shape != self.state[name].shape:
                raise RuntimeError(f"SOPAT EMA checkpoint shape differs for {name}")
        self.decay = float(decay)
        self.state = {name: value.detach().clone() for name, value in loaded.items()}

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        original = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.state, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(original, strict=True)


def train_coupled_step(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Mapping[str, Mapping[str, object]],
    config: SOPATTrainConfig,
    *,
    ema: ModelEMA | None = None,
    generator: torch.Generator | None = None,
) -> CoupledStepResult:
    """Run one weighted two-direction DDP step with a single optimizer update."""

    missing = set(DIRECTIONS).difference(batches)
    if missing:
        raise ValueError(f"coupled SOPAT step is missing directions: {sorted(missing)}")
    optimizer.zero_grad(set_to_none=True)
    parameter_device = _module_device(module)
    with torch.autocast(
        device_type=parameter_device.type,
        dtype=torch.bfloat16,
        enabled=config.autocast_bfloat16 and parameter_device.type == "cuda",
    ):
        output = module(batches, generator)
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("SOPAT training module must return (loss, diagnostics)")
    total, diagnostics = output
    if not isinstance(total, Tensor) or not isinstance(diagnostics, Mapping):
        raise TypeError("SOPAT training module returned invalid coupled loss or diagnostics")
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("non-finite coupled SOPAT loss")
    total.backward()
    direction_values = diagnostics.get("direction_losses")
    metric_values = diagnostics.get("metrics")
    if not isinstance(direction_values, Mapping) or not isinstance(metric_values, Mapping):
        raise TypeError("SOPAT coupled diagnostics are incomplete")
    direction_losses = {
        direction: float(value.detach())
        for direction, value in direction_values.items()
        if direction in DIRECTIONS and isinstance(value, Tensor)
    }
    if set(direction_losses) != set(DIRECTIONS):
        raise TypeError("SOPAT coupled diagnostics omit a direction loss")
    metrics = {
        str(name): float(value.detach())
        for name, value in metric_values.items()
        if isinstance(value, Tensor)
    }
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("SOPAT stage has no trainable parameters")
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("non-finite SOPAT gradient")
    optimizer.step()
    if ema is not None:
        ema.update(_unwrap_sopat_model(module))
    return CoupledStepResult(
        total_loss=float(total.detach()),
        gradient_norm=float(gradient_norm),
        direction_losses=direction_losses,
        metrics=metrics,
    )


def evaluate_factorizer_loaders(
    model: SOPATForwardProtocol | nn.Module,
    loaders: Mapping[str, Iterable[Mapping[str, object]]],
    config: SOPATTrainConfig,
    *,
    device: torch.device | None = None,
    limit_batches: int | None = None,
) -> FactorizerValidationResult:
    """Evaluate the paired-anchor factorizer without rendering a query image.

    The helper deliberately calls the same anchor shortcut and objective as
    factorizer training.  Neither observations nor target labels are passed
    through the model boundary, which keeps this stage inexpensive and avoids
    selecting a factorizer based on an untrained physical renderer.
    """

    if config.stage != "factorizer":
        raise ValueError("factorizer validation requires SOPAT stage=factorizer")
    missing = set(DIRECTIONS).difference(loaders)
    unexpected = set(loaders).difference(DIRECTIONS)
    if missing or unexpected:
        raise ValueError(
            "factorizer validation requires exactly both directions; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if limit_batches is not None and limit_batches <= 0:
        raise ValueError("factorizer validation limit_batches must be positive")
    was_training = isinstance(model, nn.Module) and model.training
    if isinstance(model, nn.Module):
        model.eval()
    direction_losses: dict[str, float] = {}
    metrics: dict[str, float] = {}
    batches: dict[str, int] = {}
    try:
        with torch.inference_mode():
            for direction in DIRECTIONS:
                total = 0.0
                count = 0
                sums: dict[str, float] = {}
                for index, batch in enumerate(loaders[direction]):
                    if limit_batches is not None and index >= limit_batches:
                        break
                    if not isinstance(batch, Mapping):
                        raise TypeError("factorizer validation loader must yield mapping batches")
                    inputs = forward_input_tensors(batch, device=device)
                    output = factorize_sopat_anchors(model, batch, direction, device=device)
                    loss, values = factorizer_objective(output, inputs, config)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"non-finite factorizer validation loss: {direction}")
                    total += float(loss.detach())
                    count += 1
                    for name, value in values.items():
                        sums[name] = sums.get(name, 0.0) + float(value.detach())
                if count <= 0:
                    raise RuntimeError(f"factorizer validation loader is empty: {direction}")
                direction_losses[direction] = total / count
                batches[direction] = count
                for name, value in sums.items():
                    metrics[f"{direction}/{name}"] = value / count
    finally:
        if isinstance(model, nn.Module) and was_training:
            model.train()
    weighted_loss = sum(
        float(config.direction_weights[direction]) * direction_losses[direction]
        for direction in DIRECTIONS
    )
    if not math.isfinite(weighted_loss):
        raise FloatingPointError("non-finite coupled factorizer validation loss")
    return FactorizerValidationResult(
        weighted_loss=weighted_loss,
        direction_losses=direction_losses,
        metrics=metrics,
        batches=batches,
    )


def capture_rng_state() -> dict[str, object]:
    """Capture all local RNGs so a checkpoint records reproducibility state."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(values: Mapping[str, object]) -> None:
    required = {"python", "numpy", "torch", "cuda"}
    if set(values) != required:
        raise ValueError("SOPAT RNG checkpoint state is incomplete")
    random.setstate(values["python"])  # type: ignore[arg-type]
    np.random.set_state(values["numpy"])  # type: ignore[arg-type]
    torch_state = values["torch"]
    if not isinstance(torch_state, Tensor):
        raise TypeError("SOPAT checkpoint torch RNG state is invalid")
    torch.set_rng_state(torch_state)
    cuda_state = values["cuda"]
    if torch.cuda.is_available():
        if not isinstance(cuda_state, list) or not all(isinstance(value, Tensor) for value in cuda_state):
            raise TypeError("SOPAT checkpoint CUDA RNG state is invalid")
        torch.cuda.set_rng_state_all(cuda_state)


def gather_rng_states(local_state: Mapping[str, object]) -> Mapping[str, object]:
    """Collect one RNG state per DDP rank for a rank-zero checkpoint payload."""

    if not dist.is_available() or not dist.is_initialized():
        return {"0": dict(local_state)}
    values: list[object] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(values, dict(local_state))
    return {str(index): value for index, value in enumerate(values)}


def save_sopat_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: ModelEMA,
    model_config: Mapping[str, object] | object,
    train_config: SOPATTrainConfig | Mapping[str, object],
    protocol_hashes: Mapping[str, Mapping[str, str]],
    global_step: int,
    best_metrics: Mapping[str, object],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    rng_state: Mapping[str, object] | None = None,
    data_state: Mapping[str, object] | None = None,
) -> Path:
    """Atomically save the only SOPAT V4 checkpoint format.

    One checkpoint always contains both directions.  There is deliberately no
    direction-specific optimizer state that can be resumed into a shared V4
    model by accident.
    """

    if global_step < 0:
        raise ValueError("SOPAT global_step cannot be negative")
    resolved_model = _unwrap_sopat_model(model)
    normalized_model_config = _config_mapping(model_config)
    normalized_train_config = _config_mapping(train_config)
    normalized_protocol = _normalize_protocol_hashes(protocol_hashes)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "sopat_v4_format": SOPAT_V4_FORMAT,
        "family": SOPAT_V4_FAMILY,
        "directions": list(DIRECTIONS),
        "architecture": str(normalized_model_config.get("architecture", "sopat_v4")),
        "model_config": normalized_model_config,
        "train_config": normalized_train_config,
        "protocol_hashes": normalized_protocol,
        "model": resolved_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema": ema.state_dict(),
        "rng": dict(rng_state if rng_state is not None else capture_rng_state()),
        "global_step": int(global_step),
        "best_metrics": _json_safe_mapping(best_metrics),
        "data_state": _json_safe_mapping(data_state or {}),
    }
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_sopat_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    ema: ModelEMA | None,
    model_config: Mapping[str, object] | object,
    train_config: SOPATTrainConfig | Mapping[str, object],
    protocol_hashes: Mapping[str, Mapping[str, str]],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    restore_rng: bool = False,
) -> dict[str, object]:
    """Strictly resume a compatible, bidirectional SOPAT V4 checkpoint."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("SOPAT checkpoint payload must be a mapping")
    if int(payload.get("sopat_v4_format", -1)) != SOPAT_V4_FORMAT:
        raise RuntimeError("incompatible SOPAT V4 checkpoint format")
    if payload.get("family") != SOPAT_V4_FAMILY:
        raise RuntimeError("checkpoint does not belong to SOPAT V4")
    if tuple(payload.get("directions", ())) != DIRECTIONS:
        raise RuntimeError("SOPAT checkpoint does not contain exactly both directions")
    expected_model_config = _config_mapping(model_config)
    expected_train_config = _config_mapping(train_config)
    expected_protocol = _normalize_protocol_hashes(protocol_hashes)
    if payload.get("model_config") != expected_model_config:
        raise RuntimeError("SOPAT checkpoint model configuration differs from this run")
    if payload.get("train_config") != expected_train_config:
        raise RuntimeError("SOPAT checkpoint training configuration differs from this run")
    if payload.get("protocol_hashes") != expected_protocol:
        raise RuntimeError("SOPAT checkpoint direction/cache protocol hashes differ from this run")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise TypeError("SOPAT checkpoint is missing model state")
    _unwrap_sopat_model(model).load_state_dict(state, strict=True)
    if optimizer is not None:
        optimizer_state = payload.get("optimizer")
        if not isinstance(optimizer_state, Mapping):
            raise RuntimeError("SOPAT resume checkpoint is missing optimizer state")
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler_state = payload.get("scheduler")
        if not isinstance(scheduler_state, Mapping):
            raise RuntimeError("SOPAT resume checkpoint is missing scheduler state")
        scheduler.load_state_dict(scheduler_state)
    if ema is not None:
        ema_state = payload.get("ema")
        if not isinstance(ema_state, Mapping):
            raise RuntimeError("SOPAT resume checkpoint is missing EMA state")
        ema.load_state_dict(ema_state)
    if restore_rng:
        rng = payload.get("rng")
        if not isinstance(rng, Mapping):
            raise RuntimeError("SOPAT resume checkpoint is missing RNG state")
        local = rng.get(str(dist.get_rank()) if dist.is_available() and dist.is_initialized() else "0")
        if isinstance(local, Mapping):
            restore_rng_state(local)
        else:
            # Single-rank historical saves may store the local state directly.
            restore_rng_state(rng)
    return payload


def initialize_from_sopat_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    model_config: Mapping[str, object] | object,
    protocol_hashes: Mapping[str, Mapping[str, str]],
    use_ema: bool = False,
) -> dict[str, object]:
    """Initialize model weights from a compatible V4 stage without optimizer state.

    This is the deliberate bridge from ``factorizer`` to ``physical``.  Unlike
    ``load_sopat_checkpoint`` it does not require an identical train config,
    and it never reads optimizer, scheduler, EMA, RNG, or best metrics into
    the active process.
    """

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("SOPAT initialization checkpoint payload must be a mapping")
    if int(payload.get("sopat_v4_format", -1)) != SOPAT_V4_FORMAT:
        raise RuntimeError("SOPAT initialization checkpoint has incompatible format")
    if payload.get("family") != SOPAT_V4_FAMILY or tuple(payload.get("directions", ())) != DIRECTIONS:
        raise RuntimeError("SOPAT initialization checkpoint is not a bidirectional V4 checkpoint")
    expected_model_config = _config_mapping(model_config)
    stored_model_config = payload.get("model_config")
    compatible_factorizer_pre_contrast = (
        _checkpoint_stage(payload) == "factorizer"
        and isinstance(stored_model_config, Mapping)
        and expected_model_config.get("transport_parameterization") == "contrastive_null_v1"
        and "transport_parameterization" not in stored_model_config
        and {
            name: value
            for name, value in expected_model_config.items()
            if name != "transport_parameterization"
        }
        == _json_safe_mapping(stored_model_config)
    )
    if stored_model_config != expected_model_config and not compatible_factorizer_pre_contrast:
        raise RuntimeError("SOPAT initialization checkpoint model configuration differs")
    if payload.get("protocol_hashes") != _normalize_protocol_hashes(protocol_hashes):
        raise RuntimeError("SOPAT initialization checkpoint protocol hashes differ")
    state: object
    weight_source = "model"
    if use_ema:
        ema_payload = payload.get("ema")
        if not isinstance(ema_payload, Mapping):
            raise RuntimeError("SOPAT EMA initialization checkpoint has no EMA state")
        state = ema_payload.get("state")
        weight_source = "ema"
    else:
        state = payload.get("model")
    if not isinstance(state, Mapping):
        raise TypeError(f"SOPAT initialization checkpoint has no {weight_source} state")
    incompatible = _unwrap_sopat_model(model).load_state_dict(state, strict=False)
    # V4.1 adds only the conservative evidence gates.  Missing these tensors
    # is a valid initialization path: their constructor bias keeps the model
    # close to anchor-copy until the new utility objective calibrates them.
    allowed_missing = {
        "renderers.optical.confidence.weight",
        "renderers.optical.confidence.bias",
        "renderers.sar.confidence.weight",
        "renderers.sar.confidence.bias",
    }
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing.difference(allowed_missing) or unexpected:
        raise RuntimeError(
            "SOPAT initialization checkpoint has incompatible model tensors: "
            f"missing={sorted(missing.difference(allowed_missing))}, "
            f"unexpected={sorted(unexpected)}"
        )
    return {
        "source": str(path),
        "source_global_step": int(payload.get("global_step", 0)),
        "source_train_stage": _checkpoint_stage(payload),
        "initialized_weight_source": weight_source,
        "initialized_missing_keys": sorted(missing),
    }


def initialize_from_v3_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    allowed_prefixes: Sequence[str] = DEFAULT_V3_INITIALIZATION_PREFIXES,
    require_match: bool = True,
    use_ema: bool = True,
) -> dict[str, object]:
    """Load only whitelisted exact-name/shape tensors from an older V3 checkpoint.

    This intentionally never reads optimizer, scheduler, EMA, direction, or
    release state from V3.  SOPAT physical validity must be established anew.
    """

    if not allowed_prefixes or any(not prefix for prefix in allowed_prefixes):
        raise ValueError("SOPAT V3 initialization needs non-empty allowed prefixes")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping):
        raise TypeError("V3 initialization checkpoint must be a mapping")
    source = payload.get("model", payload.get("model_state"))
    if not isinstance(source, Mapping):
        raise TypeError("V3 initialization checkpoint has no model state mapping")
    resolved_source = dict(source)
    ema_overlaid: list[str] = []
    if use_ema:
        ema = payload.get("ema")
        ema_state = ema.get("state") if isinstance(ema, Mapping) else None
        if isinstance(ema_state, Mapping):
            for name, value in ema_state.items():
                if isinstance(name, str) and isinstance(value, Tensor):
                    resolved_source[name] = value
                    ema_overlaid.append(name)
    resolved = _unwrap_sopat_model(model)
    target = resolved.state_dict()
    loaded: list[str] = []
    skipped_shape: list[str] = []
    for name, target_value in target.items():
        if not any(name.startswith(prefix) for prefix in allowed_prefixes):
            continue
        source_value = resolved_source.get(name)
        if not isinstance(source_value, Tensor):
            continue
        if source_value.shape != target_value.shape:
            skipped_shape.append(name)
            continue
        target[name] = source_value.detach().to(dtype=target_value.dtype).clone()
        loaded.append(name)
    if require_match and not loaded:
        raise RuntimeError("V3 initialization found no allowed exact-name, shape-compatible tensors")
    resolved.load_state_dict(target, strict=True)
    return {
        "source": str(path),
        "allowed_prefixes": list(allowed_prefixes),
        "use_ema": bool(use_ema),
        "ema_overlaid": sorted(ema_overlaid),
        "loaded": sorted(loaded),
        "skipped_shape": sorted(skipped_shape),
    }


def _checkpoint_stage(payload: Mapping[str, object]) -> str | None:
    config = payload.get("train_config")
    if not isinstance(config, Mapping):
        return None
    stage = config.get("stage")
    return str(stage) if isinstance(stage, str) else None


def _sample_probability(
    probability: float, device: torch.device, generator: torch.Generator | None
) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return bool(
        torch.rand(
            (),
            device=device,
            generator=_generator_for_device(generator, device),
        )
        < probability
    )


def _generator_for_device(
    generator: torch.Generator | None, device: torch.device
) -> torch.Generator | None:
    """Return a generator only when its backend matches the sampled tensor.

    PyTorch rejects a CPU generator for CUDA ``rand``/``randperm`` calls.  A
    mismatched caller generator is intentionally ignored: the process-global
    generator remains deterministic after the CLI seeds each DDP rank.
    """

    if generator is None:
        return None
    generator_device = torch.device(generator.device)
    if generator_device.type != device.type:
        return None
    if generator_device.type == "cuda" and generator_device.index not in {None, device.index}:
        return None
    return generator


def _per_example_rmse(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    mask = _resize_valid(valid, prediction).expand_as(prediction)
    numerator = ((prediction - target).square() * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (numerator / denominator).sqrt()


def _per_example_stable_rmse(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Differentiable RMSE with a finite derivative at an exact identity."""

    mask = _resize_valid(valid, prediction).expand_as(prediction)
    numerator = ((prediction - target).square() * mask).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (numerator / denominator + 1e-8).sqrt()


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def _unwrap_sopat_model(model: nn.Module) -> nn.Module:
    module = model
    while hasattr(module, "module") and isinstance(module.module, nn.Module):
        module = module.module
    while isinstance(module, SOPATTrainingModule):
        module = module.model
        while hasattr(module, "module") and isinstance(module.module, nn.Module):
            module = module.module
    return module


def _config_mapping(value: Mapping[str, object] | object) -> dict[str, object]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError("SOPAT configuration must be a mapping or dataclass instance")
    return _json_safe_mapping(value)


def _normalize_protocol_hashes(
    values: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, str]]:
    if set(values) != set(DIRECTIONS):
        raise ValueError("SOPAT protocol hashes must contain both directions")
    normalized: dict[str, dict[str, str]] = {}
    for direction in DIRECTIONS:
        artifacts = values[direction]
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise TypeError(f"SOPAT protocol hashes for {direction} must be a non-empty mapping")
        normalized[direction] = {}
        for name, digest in sorted(artifacts.items()):
            if not isinstance(name, str) or not isinstance(digest, str):
                raise TypeError("SOPAT protocol hash names and values must be strings")
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
                raise ValueError(f"SOPAT protocol hash {direction}/{name} must be SHA-256 hex")
            normalized[direction][name] = digest.lower()
    return normalized


def _json_safe_mapping(values: Mapping[str, object]) -> dict[str, object]:
    """Canonicalize nested config/report mappings without serializing tensors."""

    result: dict[str, object] = {}
    for name, value in sorted(values.items(), key=lambda item: str(item[0])):
        key = str(name)
        if is_dataclass(value) and not isinstance(value, type):
            value = asdict(value)
        if isinstance(value, Mapping):
            result[key] = _json_safe_mapping(value)
        elif isinstance(value, tuple | list):
            result[key] = [_json_safe_value(item) for item in value]
        else:
            result[key] = _json_safe_value(value)
    return result


def _json_safe_value(value: object) -> object:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise TypeError("SOPAT checkpoint metadata cannot contain non-scalar tensors")
        return float(value.detach().cpu())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe_mapping(asdict(value))
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Config values need deterministic JSON-compatible metadata.  Do not use
    # ``repr`` for opaque runtime state because resume equality must be strict.
    raise TypeError(f"SOPAT checkpoint metadata has unsupported type {type(value).__name__}")


def canonical_json_sha256(values: Mapping[str, object]) -> str:
    """Hash a canonical JSON-compatible mapping for protocol construction."""

    import hashlib

    encoded = json.dumps(_json_safe_mapping(values), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
