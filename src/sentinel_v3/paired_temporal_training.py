"""Stage training utilities for sparse paired-anchor cross-sensor transport."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .paired_temporal_v2 import PairedTemporalOutput, SparsePairedAnchorTransport
from .sensors import SENTINEL1, SENTINEL2, SensorSpec

PAIRED_TEMPORAL_CHECKPOINT_FORMAT = 1
PAIRED_TEMPORAL_CHECKPOINT_FAMILY = "sparse_paired_anchor_transport"
PAIRED_TEMPORAL_STAGES = ("physical", "detail", "flow", "balance")


@dataclass(frozen=True)
class PairedTemporalTrainConfig:
    direction: str
    stage: str = "physical"
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    observation_dropout: float = 0.35
    query_observation_dropout: float = 0.50
    one_frame_probability: float = 1.0 / 3.0
    two_to_three_frame_probability: float = 1.0 / 3.0
    four_plus_frame_probability: float = 1.0 / 3.0
    translation_max_delta_days: int = 1
    physical_gradient_weight: float = 0.20
    uncertainty_weight: float = 0.05
    detail_gradient_weight: float = 0.25
    detail_frequency_weight: float = 0.50
    flow_velocity_weight: float = 1.0
    flow_endpoint_weight: float = 0.25
    codec_weight: float = 0.25
    flow_pixel_weight: float = 0.25
    balance_visual_weight: float = 0.20
    balance_frequency_weight: float = 0.20
    balance_rmse_guard_weight: float = 2.0
    visual_rmse_budget: float = 1.05
    minimum_physical_anchor_improvement_percent: float = 5.0
    minimum_source_evidence_improvement_percent: float = 1.0
    pre_projection_violation_maximum: float = 0.001
    minimum_scene_improvement_fraction: float = 0.70
    maximum_scene_rmse_regression_fraction: float = 0.10
    visual_seed: int = 71

    def __post_init__(self) -> None:
        if self.direction not in {"sar_to_optical", "optical_to_sar"}:
            raise ValueError("direction must be sar_to_optical or optical_to_sar")
        if self.stage not in PAIRED_TEMPORAL_STAGES:
            raise ValueError("stage must be physical, detail, flow, or balance")
        if self.learning_rate <= 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("learning rate and gradient clip must be positive")
        if not 0.0 <= self.observation_dropout < 1.0:
            raise ValueError("observation_dropout must be in [0, 1)")
        if not 0.0 <= self.query_observation_dropout <= 1.0:
            raise ValueError("query_observation_dropout must be in [0, 1]")
        probabilities = (
            self.one_frame_probability,
            self.two_to_three_frame_probability,
            self.four_plus_frame_probability,
        )
        if any(probability < 0.0 for probability in probabilities) or sum(probabilities) <= 0.0:
            raise ValueError("count-stratified observation probabilities must be non-negative")
        if not 0 <= self.translation_max_delta_days <= 7:
            raise ValueError("translation_max_delta_days must be in [0, 7]")
        if self.visual_rmse_budget < 1.0:
            raise ValueError("visual_rmse_budget cannot be below physical RMSE")
        for name in (
            "minimum_physical_anchor_improvement_percent",
            "minimum_source_evidence_improvement_percent",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 100.0:
                raise ValueError(f"{name} must be in [0, 100]")
        if self.pre_projection_violation_maximum < 0.0:
            raise ValueError("pre_projection_violation_maximum cannot be negative")
        if not 0.0 <= self.minimum_scene_improvement_fraction <= 1.0:
            raise ValueError("minimum_scene_improvement_fraction must be in [0, 1]")
        if not 0.0 <= self.maximum_scene_rmse_regression_fraction <= 1.0:
            raise ValueError("maximum_scene_rmse_regression_fraction must be in [0, 1]")


class PairedTemporalTrainingModule(nn.Module):
    """Keep model and stage loss inside one DDP-visible forward graph."""

    def __init__(
        self,
        model: SparsePairedAnchorTransport,
        config: PairedTemporalTrainConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def forward(
        self, tensors: Mapping[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor], PairedTemporalOutput]:
        output = forward_paired_temporal(self.model, tensors, self.config.direction)
        loss, metrics = paired_temporal_objective(self.model, output, tensors, self.config)
        return loss, metrics, output


def direction_sensors(direction: str) -> tuple[SensorSpec, SensorSpec]:
    if direction == "sar_to_optical":
        return SENTINEL1, SENTINEL2
    if direction == "optical_to_sar":
        return SENTINEL2, SENTINEL1
    raise ValueError(f"unsupported direction: {direction}")


def paired_tensor_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, Tensor]:
    aliases = {
        "observations": ("observations", "observation_values"),
        "observation_valid": ("observation_valid",),
        "observation_days": ("observation_days",),
        "observation_present": ("observation_present",),
        "source_anchor": ("source_anchor", "source_anchor_values"),
        "source_anchor_valid": ("source_anchor_valid",),
        "target_anchor": ("target_anchor", "target_anchor_values"),
        "target_anchor_valid": ("target_anchor_valid",),
        # Keep the original one-time anchor contract as a compatibility
        # fallback while the paired data layer exposes each sensor's date.
        "anchor_days": ("anchor_days", "target_anchor_days", "source_anchor_days"),
        "source_anchor_days": ("source_anchor_days", "anchor_days"),
        "target_anchor_days": ("target_anchor_days", "anchor_days"),
        "target": ("target", "target_values"),
        "target_valid": ("target_valid",),
    }
    tensors: dict[str, Tensor] = {}
    for canonical, candidates in aliases.items():
        value = next((batch[name] for name in candidates if name in batch), None)
        if not isinstance(value, Tensor):
            raise TypeError(f"paired temporal batch is missing tensor {canonical}")
        tensors[canonical] = value.to(device=device, non_blocking=True)
    tensors["observation_present"] = tensors["observation_present"].bool()
    batch_size = tensors["target"].shape[0]
    target_valid = tensors["target_valid"]
    high_frequency_valid = batch.get("high_frequency_valid")
    if high_frequency_valid is None:
        tensors["high_frequency_valid"] = torch.ones_like(target_valid)
    elif isinstance(high_frequency_valid, Tensor):
        if high_frequency_valid.shape != target_valid.shape:
            raise ValueError("high_frequency_valid must have shape Bx1xHxW")
        tensors["high_frequency_valid"] = high_frequency_valid.to(
            device=device,
            dtype=target_valid.dtype,
            non_blocking=True,
        )
    else:
        raise TypeError("high_frequency_valid must be a tensor when supplied")

    eligible = batch.get("high_frequency_eligible")
    weight = batch.get("high_frequency_weight")
    if weight is None:
        weight = eligible
    if weight is None:
        resolved_weight = torch.ones(batch_size, device=device, dtype=target_valid.dtype)
    elif isinstance(weight, Tensor):
        if weight.shape == (batch_size, 1):
            weight = weight[:, 0]
        if weight.shape != (batch_size,):
            raise ValueError("high_frequency_weight must have shape B or Bx1")
        resolved_weight = weight.to(
            device=device,
            dtype=target_valid.dtype,
            non_blocking=True,
        )
        if not bool(torch.isfinite(resolved_weight).all()) or bool(
            (resolved_weight < 0).any()
        ) or bool((resolved_weight > 1).any()):
            raise ValueError("high_frequency_weight must be finite in [0, 1]")
    else:
        raise TypeError("high_frequency_weight must be a tensor when supplied")
    tensors["high_frequency_weight"] = resolved_weight

    if eligible is None:
        tensors["high_frequency_eligible"] = resolved_weight > 0.0
    elif isinstance(eligible, Tensor):
        if eligible.shape == (batch_size, 1):
            eligible = eligible[:, 0]
        if eligible.shape != (batch_size,):
            raise ValueError("high_frequency_eligible must have shape B or Bx1")
        tensors["high_frequency_eligible"] = eligible.to(
            device=device,
            dtype=torch.bool,
            non_blocking=True,
        )
    else:
        raise TypeError("high_frequency_eligible must be a tensor when supplied")
    return tensors


def apply_observation_dropout(
    tensors: Mapping[str, Tensor],
    *,
    frame_probability: float,
    query_probability: float,
    translation_max_delta_days: int = 1,
    one_frame_probability: float | None = None,
    two_to_three_frame_probability: float | None = None,
    four_plus_frame_probability: float | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Create a sparse observation subset while retaining at least one real frame.

    Padded slots are never resurrected.  When all three count-stratified
    probabilities are supplied, a sample first selects one of the 1, 2--3,
    and 4+ frame regimes that it can realize.  Query-time frames are then
    dropped independently, including the allowed one-day sensor offset.

    Omitting all count probabilities retains the original independent frame
    dropout behavior for callers that rely on it.
    """

    present = tensors["observation_present"].bool()
    days = tensors["observation_days"]
    if present.shape != days.shape:
        raise ValueError("observation_present and observation_days must have equal shape")
    if not 0.0 <= frame_probability < 1.0 or not 0.0 <= query_probability <= 1.0:
        raise ValueError("invalid observation dropout probabilities")
    if not 0 <= translation_max_delta_days <= 7:
        raise ValueError("translation_max_delta_days must be in [0, 7]")
    stratified = (
        one_frame_probability,
        two_to_three_frame_probability,
        four_plus_frame_probability,
    )
    if any(probability is None for probability in stratified) and any(
        probability is not None for probability in stratified
    ):
        raise ValueError("supply all count-stratified observation probabilities or none")
    if all(probability is not None for probability in stratified):
        retained = count_stratified_observation_subsampling(
            present,
            one_frame_probability=float(one_frame_probability),
            two_to_three_frame_probability=float(two_to_three_frame_probability),
            four_plus_frame_probability=float(four_plus_frame_probability),
            generator=generator,
        )
    else:
        random = torch.rand(present.shape, device=present.device, generator=generator)
        retained = present & (random >= frame_probability)

    query_time = present & (days.abs() <= float(translation_max_delta_days))
    query_random = torch.rand(present.shape, device=present.device, generator=generator)
    retained = retained & ~(query_time & (query_random < query_probability))
    for batch_index in range(present.shape[0]):
        if bool(retained[batch_index].any()):
            continue
        # Honor query dropout whenever a causal non-query frame exists.  The
        # sole query-time input is retained only when it is the only option.
        candidates = torch.nonzero(
            present[batch_index] & ~query_time[batch_index], as_tuple=False
        ).flatten()
        if candidates.numel() == 0:
            candidates = torch.nonzero(present[batch_index], as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError("each sample requires at least one present observation")
        # Prefer the newest frame after dropout; it contains the most relevant
        # causal evidence and makes the one-frame regime deterministic.
        newest = candidates[days[batch_index, candidates].argmax()]
        retained[batch_index, newest] = True
    result = dict(tensors)
    result["observation_present"] = retained
    result["observation_valid"] = tensors["observation_valid"] * retained[
        :, :, None, None, None
    ].to(tensors["observation_valid"])
    result["observations"] = tensors["observations"] * retained[
        :, :, None, None, None
    ].to(tensors["observations"])
    return result


def count_stratified_observation_subsampling(
    present: Tensor,
    *,
    one_frame_probability: float,
    two_to_three_frame_probability: float,
    four_plus_frame_probability: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return a reproducible subset that covers sparse frame-count regimes.

    The three probabilities are weights over regimes available to each sample;
    unavailable regimes are omitted and the remaining weights are normalized.
    This makes one-frame samples valid even when their nominal one-frame
    weight is zero, while preserving no-padding-resurrection guarantees.
    """

    if present.ndim != 2:
        raise ValueError("observation_present must have shape BxT")
    probabilities = (
        float(one_frame_probability),
        float(two_to_three_frame_probability),
        float(four_plus_frame_probability),
    )
    if any(probability < 0.0 for probability in probabilities) or sum(probabilities) <= 0.0:
        raise ValueError("count-stratified observation probabilities must be non-negative")
    result = torch.zeros_like(present, dtype=torch.bool)
    for batch_index in range(present.shape[0]):
        candidates = torch.nonzero(present[batch_index].bool(), as_tuple=False).flatten()
        count = int(candidates.numel())
        if count == 0:
            raise ValueError("each sample requires at least one present observation")
        available: list[tuple[int, int, float]] = [(1, 1, probabilities[0])]
        if count >= 2:
            available.append((2, min(3, count), probabilities[1]))
        if count >= 4:
            available.append((4, count, probabilities[2]))
        weights = torch.tensor(
            [entry[2] for entry in available], device=present.device, dtype=torch.float32
        )
        if not bool((weights > 0).any()):
            # The only valid fallback is a real count regime, never padding.
            weights.fill_(1.0)
        choice = int(torch.multinomial(weights, 1, generator=generator).item())
        minimum, maximum, _ = available[choice]
        if minimum == maximum:
            selected_count = minimum
        else:
            selected_count = int(
                torch.randint(
                    minimum,
                    maximum + 1,
                    (1,),
                    device=present.device,
                    generator=generator,
                ).item()
            )
        scores = torch.rand(count, device=present.device, generator=generator)
        selected = candidates[scores.topk(selected_count).indices]
        result[batch_index, selected] = True
    return result


def set_paired_temporal_stage(model: SparsePairedAnchorTransport, stage: str) -> None:
    modules = {
        "physical": (model.adapter, model.fusion, model.carrier, model.physical_head),
        "detail": (model.detail_head,),
        "flow": (model.residual_codec, model.bridge, model.bridge_condition),
        "balance": (
            model.residual_codec,
            model.bridge,
            model.bridge_condition,
        ),
    }
    if stage not in modules:
        raise ValueError(f"unsupported paired temporal stage: {stage}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in modules[stage]:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if stage == "flow":
        _set_release(model.detail_scale, 1.0)
    if stage == "balance":
        _set_release(model.detail_scale, 1.0)
        if float(torch.tanh(model.visual_scale.detach())) <= 0.0:
            _set_release(model.visual_scale, 0.0625)
        model.detail_scale.requires_grad_(True)
        model.visual_scale.requires_grad_(True)


def forward_paired_temporal(
    model: SparsePairedAnchorTransport,
    tensors: Mapping[str, Tensor],
    direction: str,
) -> PairedTemporalOutput:
    source_sensor, target_sensor = direction_sensors(direction)
    anchor_days = tensors["anchor_days"]
    return model(
        tensors["observations"],
        tensors["observation_valid"],
        tensors["observation_days"],
        tensors["observation_present"],
        tensors["source_anchor"],
        tensors["source_anchor_valid"],
        tensors["target_anchor"],
        tensors["target_anchor_valid"],
        anchor_days,
        source_sensor=source_sensor,
        target_sensor=target_sensor,
        source_anchor_days=tensors.get("source_anchor_days", anchor_days),
        target_anchor_days=tensors.get("target_anchor_days", anchor_days),
    )


def paired_temporal_objective(
    model: SparsePairedAnchorTransport,
    output: PairedTemporalOutput,
    tensors: Mapping[str, Tensor],
    config: PairedTemporalTrainConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    target = tensors["target"]
    valid = effective_valid(tensors)
    if config.stage == "physical":
        return _physical_objective(output, target, valid, config)
    if config.stage in {"detail", "balance"}:
        detail_loss, detail_metrics = _detail_objective(
            output,
            target,
            _high_frequency_valid(
                tensors,
                base_valid=valid,
                translation_max_delta_days=config.translation_max_delta_days,
            ),
            config,
        )
        if config.stage == "detail":
            return detail_loss, detail_metrics
    else:
        detail_loss = target.new_zeros(())
        detail_metrics = {}
    _, target_sensor = direction_sensors(config.direction)
    flow = model.visual_flow_loss(output, target, valid, target_sensor)
    flow_loss = (
        config.flow_velocity_weight * flow["flow_velocity"]
        + config.flow_endpoint_weight * flow["flow_endpoint"]
        + config.codec_weight * flow["codec_reconstruction"]
        + config.flow_pixel_weight * flow["flow_endpoint_pixel"]
    )
    metrics = {**detail_metrics, **{name: value.detach() for name, value in flow.items()}}
    if config.stage == "flow":
        return flow_loss, metrics
    sampled = model.sample_visual(
        output,
        valid,
        target_sensor,
        seed=config.visual_seed,
    )
    if sampled.visual is None:
        raise RuntimeError("visual balance stage did not return a sampled image")
    visual_pixel = _masked_mean(
        torch.sqrt((sampled.visual - target).square() + 1e-6), valid
    )
    visual_frequency = _frequency_error(sampled.visual, target, valid)
    physical_rmse = _masked_mean((output.physical.detach() - target).square(), valid).sqrt()
    visual_rmse = _masked_mean((sampled.visual - target).square(), valid).sqrt()
    rmse_excess = torch.relu(visual_rmse - config.visual_rmse_budget * physical_rmse)
    balance = (
        config.balance_visual_weight * visual_pixel
        + config.balance_frequency_weight * visual_frequency
        + config.balance_rmse_guard_weight * rmse_excess
    )
    metrics.update(
        {
            "balance_visual_pixel": visual_pixel.detach(),
            "balance_visual_frequency": visual_frequency.detach(),
            "balance_visual_rmse": visual_rmse.detach(),
            "balance_physical_rmse": physical_rmse.detach(),
            "balance_rmse_excess": rmse_excess.detach(),
        }
    )
    return detail_loss + flow_loss + balance, metrics


def regime_labels(
    tensors: Mapping[str, Tensor], *, translation_max_delta_days: int = 1
) -> tuple[Tensor, Tensor]:
    """Return frame-count bins (1, 2-3, >=4) and translation flags."""

    present = tensors["observation_present"].bool()
    count = present.sum(dim=1)
    bins = torch.where(count == 1, 0, torch.where(count <= 3, 1, 2))
    latest = torch.where(
        present,
        tensors["observation_days"],
        torch.full_like(tensors["observation_days"], -1e6),
    ).max(dim=1).values
    return bins, latest.abs() <= float(translation_max_delta_days)


def validation_regime_variants(
    tensors: Mapping[str, Tensor],
    *,
    translation_max_delta_days: int = 1,
    maximum_observations: int = 8,
) -> list[tuple[str, dict[str, Tensor]]]:
    """Expand each real sequence into deterministic deployment-count subsets.

    Translation variants retain the latest query-time source observation.  A
    forecast variant excludes every query-time source observation.  Selected
    frames are always a subset of the current availability mask, so collator
    padding and previously removed frames can never re-enter validation.
    """

    if not 0 <= translation_max_delta_days <= 7:
        raise ValueError("translation_max_delta_days must be in [0, 7]")
    if maximum_observations <= 0:
        raise ValueError("maximum_observations must be positive")
    present = tensors["observation_present"].bool()
    days = tensors["observation_days"]
    if present.shape != days.shape:
        raise ValueError("observation_present and observation_days must have equal shape")
    batch_size = present.shape[0]
    variants: list[tuple[str, dict[str, Tensor]]] = []
    for sample_index in range(batch_size):
        available = torch.nonzero(present[sample_index], as_tuple=False).flatten().tolist()
        query = [
            index
            for index in available
            if abs(float(days[sample_index, index])) <= float(translation_max_delta_days)
        ]
        historical = [index for index in available if index not in set(query)]
        for mode, candidates in (("translation", available if query else []), ("forecast", historical)):
            if not candidates:
                continue
            ordered = sorted(
                candidates,
                key=lambda index: (float(days[sample_index, index]), index),
                reverse=True,
            )
            counts: list[tuple[str, int]] = [("one", 1)]
            if len(ordered) >= 2:
                counts.append(("two_to_three", min(3, len(ordered))))
            if len(ordered) >= 4:
                counts.append(("four_plus", min(maximum_observations, len(ordered))))
            for count_name, count in counts:
                selected = ordered[:count]
                if mode == "translation" and query:
                    latest_query = max(
                        query,
                        key=lambda index: (float(days[sample_index, index]), index),
                    )
                    if latest_query not in selected:
                        selected[-1] = latest_query
                variants.append(
                    (
                        f"{mode}/{count_name}",
                        _observation_subset(tensors, sample_index, selected),
                    )
                )
    return variants


def _observation_subset(
    tensors: Mapping[str, Tensor], sample_index: int, selected: Sequence[int]
) -> dict[str, Tensor]:
    batch_size = tensors["observation_present"].shape[0]
    subset = {
        name: value[sample_index : sample_index + 1]
        if value.ndim > 0 and value.shape[0] == batch_size
        else value
        for name, value in tensors.items()
    }
    present = torch.zeros_like(subset["observation_present"], dtype=torch.bool)
    present[:, list(selected)] = True
    subset["observation_present"] = present
    retained = present[:, :, None, None, None].to(subset["observations"])
    subset["observations"] = subset["observations"] * retained
    subset["observation_valid"] = subset["observation_valid"] * retained.to(
        subset["observation_valid"]
    )
    return subset


def validation_release_search(
    model: SparsePairedAnchorTransport,
    batches: Iterable[Mapping[str, object]],
    config: PairedTemporalTrainConfig,
    *,
    device: torch.device | None = None,
    detail_candidates: Sequence[float] = (0.0, 1.0),
    texture_candidates: Sequence[float] = (0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, float | bool]:
    """Select one fixed-seed visual release on validation, never best-of-K."""

    resolved_device = device or next(model.parameters()).device
    _, target_sensor = direction_sensors(config.direction)
    selected: tuple[float, float, float, float] | None = None
    # A DataLoader is re-iterable. Materialize only a one-shot iterator so a
    # candidate search never accidentally evaluates the first release alone.
    repeatable_batches: Iterable[Mapping[str, object]] = batches
    if iter(batches) is batches:
        repeatable_batches = tuple(batches)
    was_training = model.training
    original_detail = model.detail_scale.detach().clone()
    original_texture = model.visual_scale.detach().clone()
    model.eval()
    try:
        with torch.no_grad():
            for detail in detail_candidates:
                for texture in texture_candidates:
                    if detail == 0.0 and texture != 0.0:
                        continue
                    _set_release(model.detail_scale, float(detail))
                    _set_release(model.visual_scale, float(texture))
                    physical_errors: list[Tensor] = []
                    visual_errors: list[Tensor] = []
                    frequency_errors: list[Tensor] = []
                    for index, batch in enumerate(repeatable_batches):
                        tensors = paired_tensor_batch(batch, resolved_device)
                        valid = effective_valid(tensors)
                        if not bool(valid.any()):
                            continue
                        output = forward_paired_temporal(model, tensors, config.direction)
                        sampled = model.sample_visual(
                            output,
                            valid,
                            target_sensor,
                            seed=config.visual_seed + index,
                        )
                        assert sampled.visual is not None
                        physical_errors.append(
                            _masked_mean((output.physical - tensors["target"]).square(), valid)
                        )
                        visual_errors.append(
                            _masked_mean((sampled.visual - tensors["target"]).square(), valid)
                        )
                        frequency_errors.append(
                            _frequency_error(sampled.visual, tensors["target"], valid)
                        )
                    if not physical_errors:
                        raise ValueError("release search requires usable validation batches")
                    physical_rmse = torch.stack(physical_errors).mean().sqrt().item()
                    visual_rmse = torch.stack(visual_errors).mean().sqrt().item()
                    frequency = torch.stack(frequency_errors).mean().item()
                    if visual_rmse > config.visual_rmse_budget * physical_rmse:
                        continue
                    candidate = (float(detail), float(texture), frequency, visual_rmse)
                    if selected is None or candidate[2:] < selected[2:]:
                        selected = candidate
            if selected is None:
                selected = (0.0, 0.0, float("inf"), float("inf"))
            _set_release(model.detail_scale, selected[0])
            _set_release(model.visual_scale, selected[1])
    except Exception:
        with torch.no_grad():
            model.detail_scale.copy_(original_detail)
            model.visual_scale.copy_(original_texture)
        raise
    finally:
        model.train(was_training)
    return {
        "detail_release": selected[0],
        "texture_release": selected[1],
        "frequency_error": selected[2],
        "visual_rmse": selected[3],
        "budget_satisfied": selected[2] != float("inf"),
    }


@torch.no_grad()
def evaluate_paired_temporal_batches(
    model: SparsePairedAnchorTransport,
    batches: Iterable[Mapping[str, object]],
    config: PairedTemporalTrainConfig,
    *,
    device: torch.device | None = None,
    limit_batches: int | None = None,
) -> dict[str, dict[str, float]]:
    """Report physical/visual metrics for every task and frame-count regime."""

    resolved_device = device or next(model.parameters()).device
    _, target_sensor = direction_sensors(config.direction)
    rows: dict[str, list[dict[str, float]]] = {}
    was_training = model.training
    model.eval()
    try:
        for batch_index, batch in enumerate(batches):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            tensors = paired_tensor_batch(batch, resolved_device)
            for variant_index, (key, variant) in enumerate(
                validation_regime_variants(
                    tensors,
                    translation_max_delta_days=config.translation_max_delta_days,
                )
            ):
                valid = effective_valid(variant)
                if not bool(valid.any()):
                    continue
                output = forward_paired_temporal(model, variant, config.direction)
                seed = config.visual_seed + batch_index * 1009 + variant_index
                sampled = model.sample_visual(output, valid, target_sensor, seed=seed)
                if sampled.visual is None:
                    raise RuntimeError("paired temporal evaluation did not return visual output")
                physical_rmse = _per_sample_rmse(output.physical, variant["target"], valid)
                visual_rmse = _per_sample_rmse(sampled.visual, variant["target"], valid)
                anchor_rmse = _per_sample_rmse(variant["target_anchor"], variant["target"], valid)
                if config.stage == "physical":
                    null_variant = dict(variant)
                    null_variant["observations"] = (
                        variant["source_anchor"][:, None].expand_as(variant["observations"])
                        * variant["observation_valid"].to(variant["observations"])
                    )
                    null_output = forward_paired_temporal(model, null_variant, config.direction)
                    null_source_rmse = _per_sample_rmse(
                        null_output.physical,
                        variant["target"],
                        valid,
                    )
                else:
                    null_source_rmse = physical_rmse
                target_detail = _highpass(variant["target"] - output.physical)
                detail_valid = _high_frequency_valid(
                    variant,
                    base_valid=valid,
                    translation_max_delta_days=config.translation_max_delta_days,
                )
                detail_mae = _per_sample_masked_mean(
                    (output.deterministic_detail - target_detail).abs(), detail_valid
                )
                detail_zero_mae = _per_sample_masked_mean(target_detail.abs(), detail_valid)
                physical_frequency = _per_sample_frequency_error(
                    output.physical, variant["target"], valid
                )
                visual_frequency = _per_sample_frequency_error(
                    sampled.visual, variant["target"], valid
                )
                if config.stage in {"flow", "balance"}:
                    generator = torch.Generator(device=variant["target"].device).manual_seed(
                        config.visual_seed + 100_003 + batch_index * 1009 + variant_index
                    )
                    flow_metrics = model.visual_flow_loss(
                        output,
                        variant["target"],
                        valid,
                        target_sensor,
                        generator=generator,
                    )
                    flow_objective = float(sum(flow_metrics.values()))
                else:
                    flow_objective = 0.0
                violation = sampled.pre_projection_violation.flatten(1).mean(dim=1)
                valid_fraction = valid.flatten(1).mean(dim=1)
                for sample_index in range(variant["target"].shape[0]):
                    if float(valid_fraction[sample_index]) <= 0.0:
                        continue
                    rows.setdefault(key, []).append(
                        {
                            "anchor_rmse": float(anchor_rmse[sample_index]),
                            "physical_rmse": float(physical_rmse[sample_index]),
                            "visual_rmse": float(visual_rmse[sample_index]),
                            "null_source_rmse": float(null_source_rmse[sample_index]),
                            "detail_mae": float(detail_mae[sample_index]),
                            "detail_zero_mae": float(detail_zero_mae[sample_index]),
                            "physical_frequency": float(physical_frequency[sample_index]),
                            "visual_frequency": float(visual_frequency[sample_index]),
                            "flow_objective": flow_objective,
                            "pre_projection_violation": float(violation[sample_index]),
                            "effective_valid_fraction": float(valid_fraction[sample_index]),
                        }
                    )
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("paired temporal evaluation requires at least one batch")
    result: dict[str, dict[str, float]] = {}
    for key, values in sorted(rows.items()):
        count = len(values)
        anchor = sum(value["anchor_rmse"] for value in values) / count
        physical = sum(value["physical_rmse"] for value in values) / count
        visual = sum(value["visual_rmse"] for value in values) / count
        null_source = sum(value["null_source_rmse"] for value in values) / count
        detail = sum(value["detail_mae"] for value in values) / count
        detail_zero = sum(value["detail_zero_mae"] for value in values) / count
        physical_frequency = sum(value["physical_frequency"] for value in values) / count
        visual_frequency = sum(value["visual_frequency"] for value in values) / count
        visual_frequency_improvement_fraction = sum(
            value["visual_frequency"] < value["physical_frequency"] for value in values
        ) / count
        visual_rmse_regression_fraction = sum(
            value["visual_rmse"] > config.visual_rmse_budget * value["physical_rmse"]
            for value in values
        ) / count
        result[key] = {
            "samples": float(count),
            "anchor_rmse": anchor,
            "physical_rmse": physical,
            "visual_rmse": visual,
            "null_source_rmse": null_source,
            "source_evidence_improvement_percent": 100.0
            * (null_source - physical)
            / max(null_source, 1e-8),
            "detail_mae": detail,
            "detail_zero_mae": detail_zero,
            "detail_improvement_percent": 100.0 * (detail_zero - detail) / max(detail_zero, 1e-8),
            "physical_frequency": physical_frequency,
            "visual_frequency": visual_frequency,
            "visual_frequency_improvement_percent": 100.0
            * (physical_frequency - visual_frequency)
            / max(physical_frequency, 1e-8),
            "visual_frequency_improvement_fraction": visual_frequency_improvement_fraction,
            "visual_rmse_regression_fraction": visual_rmse_regression_fraction,
            "flow_objective": sum(value["flow_objective"] for value in values) / count,
            "physical_anchor_improvement_percent": 100.0
            * (anchor - physical)
            / max(anchor, 1e-8),
            "visual_over_physical": visual / max(physical, 1e-8),
            "pre_projection_violation": sum(
                value["pre_projection_violation"] for value in values
            )
            / count,
            "effective_valid_fraction": sum(
                value["effective_valid_fraction"] for value in values
            )
            / count,
        }
    return result


def save_paired_temporal_checkpoint(
    path: str | Path,
    *,
    model: SparsePairedAnchorTransport,
    config: PairedTemporalTrainConfig,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": PAIRED_TEMPORAL_CHECKPOINT_FORMAT,
        "family": PAIRED_TEMPORAL_CHECKPOINT_FAMILY,
        "architecture": model.config.architecture,
        "model_config": asdict(model.config),
        "train_config": asdict(config),
        "stage": config.stage,
        "direction": config.direction,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "metrics": dict(metrics or {}),
        "protocol": dict(protocol or {}),
    }
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_paired_temporal_checkpoint(
    path: str | Path,
    model: SparsePairedAnchorTransport,
    *,
    direction: str,
    allowed_stages: Sequence[str] = PAIRED_TEMPORAL_STAGES,
    expected_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("paired temporal checkpoint must be a mapping")
    if int(payload.get("format_version", -1)) != PAIRED_TEMPORAL_CHECKPOINT_FORMAT:
        raise RuntimeError("incompatible paired temporal checkpoint format")
    if payload.get("family") != PAIRED_TEMPORAL_CHECKPOINT_FAMILY:
        raise RuntimeError("checkpoint belongs to a different model family")
    if payload.get("architecture") != model.config.architecture:
        raise RuntimeError("checkpoint architecture differs from model")
    if payload.get("model_config") != asdict(model.config):
        raise RuntimeError("checkpoint model configuration differs from model")
    if payload.get("direction") != direction:
        raise RuntimeError("checkpoint direction differs from requested task")
    if payload.get("stage") not in allowed_stages:
        raise RuntimeError("checkpoint stage is not allowed")
    if expected_protocol_sha256 is not None:
        protocol = payload.get("protocol")
        if not isinstance(protocol, Mapping) or protocol.get("sha256") != expected_protocol_sha256:
            raise RuntimeError("checkpoint validation protocol hash differs from this run")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint is missing model weights")
    model.load_state_dict(state)
    return payload


def _physical_objective(
    output: PairedTemporalOutput,
    target: Tensor,
    valid: Tensor,
    config: PairedTemporalTrainConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    error = output.physical - target
    pixel = _masked_mean(torch.sqrt(error.square() + 1e-6), valid)
    gradient = _gradient_error(output.physical, target, valid)
    log_variance = output.log_variance.clamp(-8.0, 4.0)
    uncertainty = _masked_mean(error.abs() * (-log_variance).exp() + log_variance, valid)
    total = (
        pixel
        + config.physical_gradient_weight * gradient
        + config.uncertainty_weight * uncertainty
    )
    return total, {
        "physical_pixel": pixel.detach(),
        "physical_gradient": gradient.detach(),
        "physical_uncertainty": uncertainty.detach(),
    }


def _detail_objective(
    output: PairedTemporalOutput,
    target: Tensor,
    valid: Tensor,
    config: PairedTemporalTrainConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    target_detail = _highpass(target - output.physical.detach()) * valid
    detail = output.deterministic_detail
    pixel = _masked_mean(torch.sqrt((detail - target_detail).square() + 1e-6), valid)
    gradient = _gradient_error(detail, target_detail, valid)
    frequency = _frequency_error(detail, target_detail, valid)
    total = (
        pixel
        + config.detail_gradient_weight * gradient
        + config.detail_frequency_weight * frequency
    )
    return total, {
        "detail_pixel": pixel.detach(),
        "detail_gradient": gradient.detach(),
        "detail_frequency": frequency.detach(),
    }


def effective_valid(tensors: Mapping[str, Tensor]) -> Tensor:
    """Pixels usable by both anchors, the target, and retained observations."""

    target_valid = tensors["target_valid"]
    source_anchor_valid = tensors["source_anchor_valid"]
    target_anchor_valid = tensors["target_anchor_valid"]
    observation_valid = tensors["observation_valid"]
    observation_present = tensors["observation_present"]
    if target_valid.ndim != 4 or target_valid.shape[1] != 1:
        raise ValueError("target_valid must have shape Bx1xHxW")
    batch, _, height, width = target_valid.shape
    for name, values in (
        ("source_anchor_valid", source_anchor_valid),
        ("target_anchor_valid", target_anchor_valid),
    ):
        if values.shape != target_valid.shape:
            raise ValueError(f"{name} must match target_valid")
    if observation_valid.ndim != 5 or observation_valid.shape[0] != batch:
        raise ValueError("observation_valid must have shape BxTx1xHxW")
    if observation_valid.shape[2:] != (1, height, width):
        raise ValueError("observation_valid must share the target grid")
    if observation_present.shape != observation_valid.shape[:2]:
        raise ValueError("observation_present must have shape BxT")
    observation_support = (
        observation_valid
        * observation_present.to(observation_valid)[:, :, None, None, None]
    ).amax(dim=1)
    return (
        target_valid.to(observation_support)
        * source_anchor_valid.to(observation_support)
        * target_anchor_valid.to(observation_support)
        * observation_support
    )


def _high_frequency_valid(
    tensors: Mapping[str, Tensor],
    *,
    base_valid: Tensor | None = None,
    translation_max_delta_days: int = 1,
) -> Tensor:
    """Return the audited per-pixel detail supervision mask.

    Old tensor-only callers have no audit fields and retain the historical
    all-valid behavior.  The paired raster dataset supplies both a local mask
    and a fractional sample weight, which are deliberately not collapsed to a
    boolean so one-day translation pairs retain their lower supervision rate.
    """

    valid = effective_valid(tensors) if base_valid is None else base_valid
    local = tensors.get("high_frequency_valid")
    if local is None:
        local = torch.ones_like(valid)
    if local.shape != valid.shape:
        raise ValueError("high_frequency_valid must match target_valid")
    weight = tensors.get("high_frequency_weight")
    if weight is None:
        weight = torch.ones(valid.shape[0], device=valid.device, dtype=valid.dtype)
    if weight.shape == (valid.shape[0], 1):
        weight = weight[:, 0]
    if weight.shape != (valid.shape[0],):
        raise ValueError("high_frequency_weight must have shape B or Bx1")
    if not 0 <= translation_max_delta_days <= 7:
        raise ValueError("translation_max_delta_days must be in [0, 7]")
    query_observation_retained = (
        tensors["observation_present"].bool()
        & (tensors["observation_days"].abs() <= float(translation_max_delta_days))
    ).any(dim=1)
    return (
        valid
        * local.to(valid)
        * weight.to(valid)[:, None, None, None]
        * query_observation_retained.to(valid)[:, None, None, None]
    )


def _highpass(values: Tensor) -> Tensor:
    return values - F.avg_pool2d(values, 5, stride=1, padding=2)


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    if valid.shape[:2] != (values.shape[0], 1):
        raise ValueError("valid must have shape Bx1xHxW")
    if valid.shape[-2:] != values.shape[-2:]:
        valid = F.interpolate(valid.float(), values.shape[-2:], mode="area").to(values)
    expanded = valid.expand_as(values).to(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def _per_sample_rmse(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    expanded = valid.expand_as(prediction).to(prediction)
    numerator = ((prediction - target).square() * expanded).flatten(1).sum(dim=1)
    denominator = expanded.flatten(1).sum(dim=1).clamp_min(1.0)
    return (numerator / denominator).sqrt()


def _per_sample_masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    expanded = valid.expand_as(values).to(values)
    numerator = (values * expanded).flatten(1).sum(dim=1)
    denominator = expanded.flatten(1).sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def _per_sample_frequency_error(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    fine = _per_sample_masked_mean(
        (_highpass(prediction) - _highpass(target)).abs(), valid
    )
    half_prediction = F.avg_pool2d(prediction, 2, stride=2)
    half_target = F.avg_pool2d(target, 2, stride=2)
    half_valid = F.avg_pool2d(valid.float(), 2, stride=2).to(valid)
    medium = _per_sample_masked_mean(
        (_highpass(half_prediction) - _highpass(half_target)).abs(), half_valid
    )
    return 0.5 * (fine + medium)


def _gradient_error(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    valid_x = valid[..., :, 1:] * valid[..., :, :-1]
    valid_y = valid[..., 1:, :] * valid[..., :-1, :]
    return 0.5 * (
        _masked_mean((pred_x - target_x).abs(), valid_x)
        + _masked_mean((pred_y - target_y).abs(), valid_y)
    )


def _frequency_error(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    fine = _masked_mean((_highpass(prediction) - _highpass(target)).abs(), valid)
    half_prediction = F.avg_pool2d(prediction, 2, stride=2)
    half_target = F.avg_pool2d(target, 2, stride=2)
    half_valid = F.avg_pool2d(valid.float(), 2, stride=2).to(valid)
    medium = _masked_mean(
        (_highpass(half_prediction) - _highpass(half_target)).abs(), half_valid
    )
    return 0.5 * (fine + medium)


def _set_release(parameter: torch.nn.Parameter, release: float) -> None:
    if not 0.0 <= release <= 1.0:
        raise ValueError("release must be in [0, 1]")
    with torch.no_grad():
        parameter.copy_(
            torch.atanh(torch.tensor(min(release, 0.999), device=parameter.device))
        )
