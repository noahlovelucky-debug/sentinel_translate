"""Small, reproducible feasibility training loop for causal temporal transport.

This is deliberately a pilot runner, not a second copy of V3.2's production
trainer.  It establishes that a strict-causal multi-temporal signal improves
on the real-anchor baseline before an expensive distributed training campaign.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .sensors import SENTINEL1, SENTINEL2, SensorSpec
from .temporal_v1 import CausalAnchorDeltaTransport, TemporalTranslationOutput

TEMPORAL_CHECKPOINT_FORMAT = 1
TEMPORAL_CHECKPOINT_FAMILY = "causal_anchor_delta_transport"
_TEMPORAL_STAGES = frozenset(("physical", "detail", "flow", "balance"))


@dataclass(frozen=True)
class TemporalPilotConfig:
    direction: str
    stage: str = "physical"
    max_steps: int = 400
    batch_size: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    gradient_weight: float = 0.20
    uncertainty_weight: float = 0.05
    detail_gradient_weight: float = 0.25
    detail_highpass_weight: float = 0.50
    flow_velocity_weight: float = 1.0
    flow_endpoint_weight: float = 0.25
    flow_codec_weight: float = 0.25
    flow_endpoint_pixel_weight: float = 0.25
    balance_visual_pixel_weight: float = 0.20
    balance_visual_highpass_weight: float = 0.20
    balance_rmse_guard_weight: float = 2.0
    visual_rmse_budget: float = 1.05
    visual_seed: int = 71
    num_workers: int = 0
    seed: int = 42
    amp: bool = True
    log_every: int = 50

    def __post_init__(self) -> None:
        if self.direction not in {"sar_to_optical", "optical_to_sar"}:
            raise ValueError("direction must be sar_to_optical or optical_to_sar")
        if self.stage not in _TEMPORAL_STAGES:
            raise ValueError("stage must be physical, detail, flow, or balance")
        if self.max_steps <= 0 or self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError("steps, batch size, and worker count must be valid")
        if self.learning_rate <= 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("learning rate and gradient clip must be positive")
        if any(
            value < 0.0
            for value in (
                self.gradient_weight,
                self.uncertainty_weight,
                self.detail_gradient_weight,
                self.detail_highpass_weight,
                self.flow_velocity_weight,
                self.flow_endpoint_weight,
                self.flow_codec_weight,
                self.flow_endpoint_pixel_weight,
                self.balance_visual_pixel_weight,
                self.balance_visual_highpass_weight,
                self.balance_rmse_guard_weight,
            )
        ):
            raise ValueError("loss weights must be non-negative")
        if self.visual_rmse_budget < 1.0:
            raise ValueError("visual_rmse_budget must not be below the physical RMSE")


def direction_sensors(direction: str) -> tuple[SensorSpec, SensorSpec]:
    if direction == "sar_to_optical":
        return SENTINEL1, SENTINEL2
    if direction == "optical_to_sar":
        return SENTINEL2, SENTINEL1
    raise ValueError(f"unsupported temporal direction: {direction}")


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    if valid.shape[1] != 1 or valid.shape[0] != values.shape[0]:
        raise ValueError("valid must have shape Bx1xHxW")
    expanded = valid.expand_as(values).to(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def _gradient(values: Tensor) -> tuple[Tensor, Tensor]:
    return values[..., :, 1:] - values[..., :, :-1], values[..., 1:, :] - values[..., :-1, :]


def physical_objective(
    output: TemporalTranslationOutput,
    target_values: Tensor,
    target_valid: Tensor,
    *,
    gradient_weight: float,
    uncertainty_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Robust target-only loss.  Anchor and source masks never mask a target."""

    prediction = output.physical
    if target_values.shape != prediction.shape:
        raise ValueError("target values must match the physical prediction")
    if target_valid.shape != (prediction.shape[0], 1, *prediction.shape[-2:]):
        raise ValueError("target_valid must have shape Bx1xHxW")
    error = prediction - target_values
    charbonnier = _masked_mean(torch.sqrt(error.square() + 1e-6), target_valid)
    pred_dx, pred_dy = _gradient(prediction)
    target_dx, target_dy = _gradient(target_values)
    valid_dx = target_valid[..., :, 1:] * target_valid[..., :, :-1]
    valid_dy = target_valid[..., 1:, :] * target_valid[..., :-1, :]
    gradient = 0.5 * (
        _masked_mean((pred_dx - target_dx).abs(), valid_dx)
        + _masked_mean((pred_dy - target_dy).abs(), valid_dy)
    )
    log_variance = output.log_variance.clamp(-8.0, 4.0)
    uncertainty = _masked_mean(error.abs() * (-log_variance).exp() + log_variance, target_valid)
    total = charbonnier + gradient_weight * gradient + uncertainty_weight * uncertainty
    return total, {
        "physical_charbonnier": charbonnier.detach(),
        "physical_gradient": gradient.detach(),
        "physical_uncertainty": uncertainty.detach(),
    }


def _highpass(values: Tensor) -> Tensor:
    return values - torch.nn.functional.avg_pool2d(values, 5, stride=1, padding=2)


def _multiscale_highpass_error(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Two-band Laplacian loss without changing the full-resolution contract.

    The one-pixel band protects edges while the downsampled band prevents the
    detail head from learning a checkerboard substitute for medium-scale
    structures.  This is intentionally a loss-space decomposition: physical
    output remains in the original radiometric representation.
    """

    fine = _masked_mean((_highpass(prediction) - _highpass(target)).abs(), valid)
    prediction_half = torch.nn.functional.avg_pool2d(prediction, 2, stride=2)
    target_half = torch.nn.functional.avg_pool2d(target, 2, stride=2)
    valid_half = torch.nn.functional.avg_pool2d(valid.float(), 2, stride=2).to(valid)
    medium = _masked_mean(
        (_highpass(prediction_half) - _highpass(target_half)).abs(), valid_half
    )
    return 0.5 * (fine + medium)


def deterministic_detail_objective(
    output: TemporalTranslationOutput,
    target_values: Tensor,
    target_valid: Tensor,
    *,
    gradient_weight: float,
    highpass_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Train detail against the physical residual without changing physical weights."""

    if output.deterministic_detail is None or output.visual_base is None:
        raise ValueError("detail stage requires deterministic detail and visual base")
    # A deterministic branch is only allowed to explain repeatable high
    # frequency.  Slow radiometric change remains the physical path's job.
    target_detail = _highpass(target_values - output.physical.detach()) * target_valid
    prediction = output.deterministic_detail
    pixel = _masked_mean(torch.sqrt((prediction - target_detail).square() + 1e-6), target_valid)
    pred_dx, pred_dy = _gradient(prediction)
    target_dx, target_dy = _gradient(target_detail)
    valid_dx = target_valid[..., :, 1:] * target_valid[..., :, :-1]
    valid_dy = target_valid[..., 1:, :] * target_valid[..., :-1, :]
    gradient = 0.5 * (
        _masked_mean((pred_dx - target_dx).abs(), valid_dx)
        + _masked_mean((pred_dy - target_dy).abs(), valid_dy)
    )
    highpass = _multiscale_highpass_error(prediction, target_detail, target_valid)
    total = pixel + gradient_weight * gradient + highpass_weight * highpass
    return total, {
        "detail_pixel": pixel.detach(),
        "detail_gradient": gradient.detach(),
        "detail_highpass": highpass.detach(),
    }


def visual_balance_objective(
    model: CausalAnchorDeltaTransport,
    output: TemporalTranslationOutput,
    tensors: dict[str, Tensor],
    target_sensor: SensorSpec,
    config: TemporalPilotConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Calibrate the released bridge sample without permitting physical drift.

    A fixed seed makes this training loss reproducible.  The hard release
    decision still belongs to validation-time scale calibration; this loss only
    teaches the learned amplitude parameter that an attractive stochastic
    residual is not acceptable when it crosses the physical RMSE budget.
    """

    sampled = model.sample_visual(
        output,
        tensors["anchor_valid"],
        target_sensor,
        seed=config.visual_seed,
    )
    if sampled.visual is None:
        raise RuntimeError("temporal visual sampler did not return a visual output")
    target_values = tensors["target_values"]
    target_valid = tensors["target_valid"]
    pixel = _masked_mean(
        torch.sqrt((sampled.visual - target_values).square() + 1e-6), target_valid
    )
    highpass = _multiscale_highpass_error(sampled.visual, target_values, target_valid)
    physical_rmse = _rmse(output.physical.detach(), target_values, target_valid)
    visual_rmse = _rmse(sampled.visual, target_values, target_valid)
    rmse_excess = torch.relu(visual_rmse - config.visual_rmse_budget * physical_rmse)
    loss = (
        config.balance_visual_pixel_weight * pixel
        + config.balance_visual_highpass_weight * highpass
        + config.balance_rmse_guard_weight * rmse_excess
    )
    return loss, {
        "balance_visual_pixel": pixel.detach(),
        "balance_visual_highpass": highpass.detach(),
        "balance_visual_rmse": visual_rmse.detach(),
        "balance_physical_rmse": physical_rmse.detach(),
        "balance_rmse_excess": rmse_excess.detach(),
    }


def set_temporal_stage(model: CausalAnchorDeltaTransport, stage: str) -> None:
    """Freeze prior stages so visual training cannot silently move physical output."""

    modules = {
        "physical": (model.adapter, model.fusion, model.physical_head, model.observable_carrier),
        "detail": (model.detail_head,),
        "flow": (model.residual_codec, model.bridge, model.bridge_condition),
        "balance": (model.detail_head, model.residual_codec, model.bridge, model.bridge_condition),
    }
    if stage not in _TEMPORAL_STAGES:
        raise ValueError(f"unsupported temporal stage: {stage}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in modules[stage]:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if stage == "flow":
        # The flow target must use the same base as sampling.  A completed
        # detail stage is therefore fully present while residual transport is
        # trained; calibration later either releases that base or closes the
        # visual branch altogether.
        _release_to_parameter(1.0, model.detail_scale)
    if stage == "balance":
        model.detail_scale.requires_grad_(True)
        model.visual_scale.requires_grad_(True)


def temporal_stage_objective(
    model: CausalAnchorDeltaTransport,
    output: TemporalTranslationOutput,
    tensors: dict[str, Tensor],
    config: TemporalPilotConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    if config.stage == "physical":
        return physical_objective(
            output,
            tensors["target_values"],
            tensors["target_valid"],
            gradient_weight=config.gradient_weight,
            uncertainty_weight=config.uncertainty_weight,
        )
    if config.stage in {"detail", "balance"}:
        detail_loss, detail_metrics = deterministic_detail_objective(
            output,
            tensors["target_values"],
            tensors["target_valid"],
            gradient_weight=config.detail_gradient_weight,
            highpass_weight=config.detail_highpass_weight,
        )
        if config.stage == "detail":
            return detail_loss, detail_metrics
    else:
        detail_loss = torch.zeros((), device=tensors["target_values"].device)
        detail_metrics = {}
    _, target_sensor = direction_sensors(config.direction)
    flow_metrics = model.visual_flow_loss(
        output,
        tensors["target_values"],
        tensors["target_valid"],
        target_sensor,
    )
    flow_loss = (
        config.flow_velocity_weight * flow_metrics["flow_velocity"]
        + config.flow_endpoint_weight * flow_metrics["flow_endpoint"]
        + config.flow_codec_weight * flow_metrics["codec_reconstruction"]
        + config.flow_endpoint_pixel_weight * flow_metrics["flow_endpoint_pixel"]
    )
    metrics = {**detail_metrics, **{key: value.detach() for key, value in flow_metrics.items()}}
    if config.stage != "balance":
        return flow_loss, metrics
    visual_loss, visual_metrics = visual_balance_objective(
        model, output, tensors, target_sensor, config
    )
    return detail_loss + flow_loss + visual_loss, {**metrics, **visual_metrics}


def _tensor_batch(batch: dict[str, object], device: torch.device) -> dict[str, Tensor]:
    keys = (
        "source_values",
        "source_valid",
        "anchor_values",
        "anchor_valid",
        "target_values",
        "target_valid",
        "source_days",
        "anchor_days",
    )
    tensors: dict[str, Tensor] = {}
    for key in keys:
        value = batch.get(key)
        if not isinstance(value, Tensor):
            raise TypeError(f"temporal batch is missing tensor {key}")
        tensors[key] = value.to(device=device, non_blocking=True)
    return tensors


def _forward(
    model: CausalAnchorDeltaTransport,
    tensors: dict[str, Tensor],
    direction: str,
    *,
    zero_source: bool = False,
) -> TemporalTranslationOutput:
    source_sensor, target_sensor = direction_sensors(direction)
    source_values = tensors["source_values"]
    if zero_source:
        source_values = torch.zeros_like(source_values)
    return model(
        source_values,
        tensors["source_valid"],
        tensors["anchor_values"],
        tensors["anchor_valid"],
        tensors["source_days"],
        tensors["anchor_days"],
        source_sensor=source_sensor,
        target_sensor=target_sensor,
    )


def _rmse(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    return _masked_mean((prediction - target).square(), valid).sqrt()


@torch.no_grad()
def evaluate_pilot(
    model: CausalAnchorDeltaTransport,
    dataset: Dataset[dict[str, object]],
    config: TemporalPilotConfig,
    device: torch.device,
    *,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate model, anchor-copy baseline, and causal-source ablation."""

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    was_training = model.training
    model.eval()
    baseline_values: list[float] = []
    physical_values: list[float] = []
    zero_source_values: list[float] = []
    valid_fractions: list[float] = []
    sample_count = 0
    for index, batch in enumerate(loader):
        if limit_batches is not None and index >= limit_batches:
            break
        tensors = _tensor_batch(batch, device)
        sample_count += int(tensors["target_values"].shape[0])
        output = _forward(model, tensors, config.direction)
        zero_source = _forward(model, tensors, config.direction, zero_source=True)
        baseline_values.append(
            float(_rmse(tensors["anchor_values"], tensors["target_values"], tensors["target_valid"]))
        )
        physical_values.append(
            float(_rmse(output.physical, tensors["target_values"], tensors["target_valid"]))
        )
        zero_source_values.append(
            float(
                _rmse(zero_source.physical, tensors["target_values"], tensors["target_valid"])
            )
        )
        valid_fractions.append(float(tensors["target_valid"].float().mean()))
    if was_training:
        model.train()
    if not physical_values:
        raise RuntimeError("temporal pilot evaluation received no batches")
    baseline = float(np.mean(baseline_values))
    physical = float(np.mean(physical_values))
    zero_source = float(np.mean(zero_source_values))
    return {
        "samples": float(sample_count),
        "anchor_rmse": baseline,
        "physical_rmse": physical,
        "zero_source_rmse": zero_source,
        "anchor_improvement_percent": 100.0 * (baseline - physical) / max(baseline, 1e-8),
        "source_ablation_penalty_percent": 100.0 * (zero_source - physical) / max(
            physical, 1e-8
        ),
        "target_valid_fraction": float(np.mean(valid_fractions)),
    }


@torch.no_grad()
def evaluate_temporal(
    model: CausalAnchorDeltaTransport,
    dataset: Dataset[dict[str, object]],
    config: TemporalPilotConfig,
    device: torch.device,
    *,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate the physical/visual Pareto point using one fixed-seed sample.

    This is intentionally not a best-of-K visual evaluation.  Each batch uses
    a deterministic seed derived from its position, making a later checkpoint
    comparison reproducible and preventing a stochastic sample search from
    being mistaken for model quality.
    """

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    was_training = model.training
    model.eval()
    anchor_values: list[float] = []
    physical_values: list[float] = []
    visual_values: list[float] = []
    zero_source_values: list[float] = []
    detail_values: list[float] = []
    highpass_values: list[float] = []
    violation_values: list[float] = []
    samples = 0
    _, target_sensor = direction_sensors(config.direction)
    for index, batch in enumerate(loader):
        if limit_batches is not None and index >= limit_batches:
            break
        tensors = _tensor_batch(batch, device)
        output = _forward(model, tensors, config.direction)
        sampled = model.sample_visual(
            output,
            tensors["anchor_valid"],
            target_sensor,
            seed=config.visual_seed + index,
        )
        if sampled.visual is None:
            raise RuntimeError("temporal sampler did not return a visual output")
        zero_source = _forward(model, tensors, config.direction, zero_source=True)
        target = tensors["target_values"]
        valid = tensors["target_valid"]
        target_detail = _highpass(target - output.physical)
        detail = output.deterministic_detail
        if detail is None:
            raise RuntimeError("temporal model did not return deterministic detail")
        samples += int(target.shape[0])
        anchor_values.append(float(_rmse(tensors["anchor_values"], target, valid)))
        physical_values.append(float(_rmse(output.physical, target, valid)))
        visual_values.append(float(_rmse(sampled.visual, target, valid)))
        zero_source_values.append(float(_rmse(zero_source.physical, target, valid)))
        detail_values.append(float(_masked_mean((detail - target_detail).abs(), valid)))
        highpass_values.append(float(_multiscale_highpass_error(sampled.visual, target, valid)))
        if sampled.pre_projection_violation is not None:
            violation_values.append(float(sampled.pre_projection_violation.mean()))
    if was_training:
        model.train()
    if not physical_values:
        raise RuntimeError("temporal evaluation received no batches")
    physical = float(np.mean(physical_values))
    visual = float(np.mean(visual_values))
    anchor = float(np.mean(anchor_values))
    return {
        "samples": float(samples),
        "anchor_rmse": anchor,
        "physical_rmse": physical,
        "visual_rmse": visual,
        "visual_over_physical": visual / max(physical, 1e-8),
        "zero_source_rmse": float(np.mean(zero_source_values)),
        "anchor_improvement_percent": 100.0 * (anchor - physical) / max(anchor, 1e-8),
        "source_ablation_penalty_percent": 100.0
        * (float(np.mean(zero_source_values)) - physical)
        / max(physical, 1e-8),
        "deterministic_detail_mae": float(np.mean(detail_values)),
        "visual_multiscale_highpass_mae": float(np.mean(highpass_values)),
        "pre_projection_violation": float(np.mean(violation_values)) if violation_values else 0.0,
    }


def _release_to_parameter(release: float, parameter: torch.nn.Parameter) -> None:
    if not 0.0 <= release <= 1.0:
        raise ValueError("release must be in [0, 1]")
    bounded = min(release, 0.999)
    parameter.copy_(torch.atanh(torch.tensor(bounded, device=parameter.device)))


@torch.no_grad()
def calibrate_temporal_visual_release(
    model: CausalAnchorDeltaTransport,
    dataset: Dataset[dict[str, object]],
    config: TemporalPilotConfig,
    device: torch.device,
    *,
    detail_candidates: Sequence[float] = (0.0, 1.0),
    texture_candidates: Sequence[float] = (0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0),
    limit_batches: int | None = None,
) -> dict[str, float | bool]:
    """Choose the strongest fixed-seed release that remains inside RMSE budget.

    High-pass error is the primary selection score because physical RMSE is a
    hard feasibility constraint.  Ties favor less stochastic amplitude, which
    avoids releasing texture merely because it is neutral on a small holdout.
    """

    if not detail_candidates or not texture_candidates:
        raise ValueError("visual release calibration needs non-empty candidate sets")
    original_detail = model.detail_scale.detach().clone()
    original_texture = model.visual_scale.detach().clone()
    selected: tuple[float, float, dict[str, float]] | None = None
    try:
        for detail_release in detail_candidates:
            for texture_release in texture_candidates:
                # Flow is trained against the full detail base.  Releasing a
                # texture without it would change the residual reference, so
                # the safe fallback is deterministic physical only.
                if detail_release == 0.0 and texture_release != 0.0:
                    continue
                _release_to_parameter(float(detail_release), model.detail_scale)
                _release_to_parameter(float(texture_release), model.visual_scale)
                metrics = evaluate_temporal(
                    model, dataset, config, device, limit_batches=limit_batches
                )
                if metrics["visual_over_physical"] > config.visual_rmse_budget:
                    continue
                candidate = (float(detail_release), float(texture_release), metrics)
                if selected is None:
                    selected = candidate
                    continue
                old_detail, old_texture, old_metrics = selected
                old_score = (
                    old_metrics["visual_multiscale_highpass_mae"],
                    old_detail + old_texture,
                )
                new_score = (
                    metrics["visual_multiscale_highpass_mae"],
                    float(detail_release) + float(texture_release),
                )
                if new_score < old_score:
                    selected = candidate
    except Exception:
        model.detail_scale.copy_(original_detail)
        model.visual_scale.copy_(original_texture)
        raise
    if selected is None:
        model.detail_scale.zero_()
        model.visual_scale.zero_()
        metrics = evaluate_temporal(model, dataset, config, device, limit_batches=limit_batches)
        return {
            "detail_release": 0.0,
            "texture_release": 0.0,
            "budget_satisfied": False,
            **metrics,
        }
    detail_release, texture_release, metrics = selected
    _release_to_parameter(detail_release, model.detail_scale)
    _release_to_parameter(texture_release, model.visual_scale)
    return {
        "detail_release": detail_release,
        "texture_release": texture_release,
        "budget_satisfied": True,
        **metrics,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_temporal_checkpoint(
    path: str | Path,
    *,
    model: CausalAnchorDeltaTransport,
    optimizer: torch.optim.Optimizer | None,
    config: TemporalPilotConfig,
    step: int,
    report: Mapping[str, Any],
    protocol: Mapping[str, Any] | None = None,
) -> Path:
    """Write a self-describing temporal checkpoint atomically.

    Temporal V1 checkpoints deliberately cannot resume the legacy V3.2
    optimizer or scheduler.  Later stages may initialize their model weights
    from this file, but always create a new optimizer for their own parameter
    subset so frozen physical parameters cannot be moved accidentally.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": TEMPORAL_CHECKPOINT_FORMAT,
        "family": TEMPORAL_CHECKPOINT_FAMILY,
        "architecture": model.config.architecture,
        "model_config": asdict(model.config),
        "stage": config.stage,
        "direction": config.direction,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "train_config": asdict(config),
        "protocol": dict(protocol or {}),
        "report": dict(report),
    }
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_temporal_checkpoint(
    path: str | Path,
    model: CausalAnchorDeltaTransport,
    *,
    direction: str,
    allowed_stages: Sequence[str] = _TEMPORAL_STAGES,
) -> dict[str, Any]:
    """Load compatible model weights while rejecting cross-task checkpoint use."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("temporal checkpoint payload must be a mapping")
    if int(checkpoint.get("format_version", -1)) != TEMPORAL_CHECKPOINT_FORMAT:
        raise RuntimeError("incompatible temporal checkpoint format")
    if checkpoint.get("family") != TEMPORAL_CHECKPOINT_FAMILY:
        raise RuntimeError("checkpoint does not belong to causal anchor-delta transport")
    if checkpoint.get("architecture") != model.config.architecture:
        raise RuntimeError("checkpoint architecture differs from the instantiated temporal model")
    if checkpoint.get("direction") != direction:
        raise RuntimeError("checkpoint direction differs from the requested temporal task")
    stage = checkpoint.get("stage")
    if stage not in allowed_stages:
        raise RuntimeError("checkpoint stage is not allowed for this temporal run")
    stored_config = checkpoint.get("model_config")
    if stored_config != asdict(model.config):
        raise RuntimeError("checkpoint model configuration differs from the instantiated model")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise TypeError("temporal checkpoint is missing model weights")
    model.load_state_dict(state)
    return checkpoint


def train_temporal_pilot(
    model: CausalAnchorDeltaTransport,
    train_dataset: Dataset[dict[str, object]],
    validation_dataset: Dataset[dict[str, object]],
    config: TemporalPilotConfig,
    *,
    output_dir: str | Path,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run a small physical-first pilot and write a self-contained report/checkpoint."""

    _seed_everything(config.seed)
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(resolved_device)
    set_temporal_stage(model, config.stage)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"temporal {config.stage} stage has no trainable parameters")
    optimizer = AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    worker_options: dict[str, object] = {"num_workers": config.num_workers}
    if config.num_workers:
        worker_options["persistent_workers"] = True
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=len(train_dataset) >= config.batch_size,
        **worker_options,
    )
    if len(loader) == 0:
        raise RuntimeError("temporal pilot train dataset is smaller than batch size")
    initial = evaluate_temporal(model, train_dataset, config, resolved_device)
    iterator: Iterable[dict[str, object]] = iter(loader)
    history: list[dict[str, float]] = []
    model.train()
    autocast = torch.autocast(
        device_type=resolved_device.type,
        dtype=torch.bfloat16,
        enabled=config.amp and resolved_device.type == "cuda",
    )
    for step in range(1, config.max_steps + 1):
        try:
            batch = next(iterator)  # type: ignore[arg-type]
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)  # type: ignore[arg-type]
        tensors = _tensor_batch(batch, resolved_device)
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            output = _forward(model, tensors, config.direction)
            loss, metrics = temporal_stage_objective(model, output, tensors, config)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite temporal pilot loss at step {step}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite temporal pilot gradient at step {step}")
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.max_steps:
            row = {"step": float(step), "loss": float(loss.detach()), "gradient_norm": gradient_norm}
            row.update({key: float(value) for key, value in metrics.items()})
            history.append(row)
    train_final = evaluate_temporal(model, train_dataset, config, resolved_device)
    validation_final = evaluate_temporal(model, validation_dataset, config, resolved_device)
    release = calibrate_temporal_visual_release(
        model, validation_dataset, config, resolved_device
    )
    validation_calibrated = evaluate_temporal(model, validation_dataset, config, resolved_device)
    report: dict[str, Any] = {
        "format_version": TEMPORAL_CHECKPOINT_FORMAT,
        "experiment": "causal_anchor_delta_temporal",
        "architecture": model.config.architecture,
        "direction": config.direction,
        "config": asdict(config),
        "initial_train": initial,
        "final_train": train_final,
        "final_validation": validation_final,
        "visual_release": release,
        "final_validation_calibrated": validation_calibrated,
        "history": history,
        "feasibility": {
            "finite_training": True,
            "train_overfit_improves_anchor": (
                train_final["physical_rmse"] < train_final["anchor_rmse"]
            ),
            "validation_improves_anchor": (
                validation_calibrated["physical_rmse"] < validation_calibrated["anchor_rmse"]
            ),
            "validation_uses_causal_source": (
                validation_calibrated["zero_source_rmse"]
                > validation_calibrated["physical_rmse"]
            ),
            "recommended_for_scale_up": (
                validation_calibrated["anchor_improvement_percent"] >= 5.0
                and validation_calibrated["source_ablation_penalty_percent"] >= 1.0
                and validation_calibrated["visual_over_physical"] <= config.visual_rmse_budget
            ),
        },
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    save_temporal_checkpoint(
        destination / f"temporal_{config.stage}_last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=config.max_steps,
        report=report,
    )
    # Keep the original pilot name as a compatibility alias for early reports.
    if config.stage == "physical":
        save_temporal_checkpoint(
            destination / "temporal_pilot_last.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            step=config.max_steps,
            report=report,
        )
    _atomic_json(destination / "temporal_pilot_report.json", report)
    return report
