from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from .data import StatefulShardSampler, V2ShardDataset, time_weights
from .losses import (
    codec_reconstruction_loss,
    deterministic_detail_loss,
    high_frequency_loss,
    highpass,
    latent_alignment,
    masked_mean,
    physical_loss,
    robust_rms,
)
from .model import ModelConfig, Pyramid, SentinelV3
from .sensors import SENTINEL1, SENTINEL2, SensorSpec


def _weighted_zero(module: nn.Module, device: torch.device) -> Tensor:
    terms = [
        parameter.sum() * 0.0 for parameter in module.parameters() if parameter.requires_grad
    ]
    return sum(terms, torch.zeros((), device=device))


class JointObjective(nn.Module):
    """Stage-aware V3.2 objective with explicit deterministic/stochastic separation."""

    def __init__(
        self,
        model: SentinelV3,
        task_probabilities: list[float] | None = None,
        physical_alignment_samples: int = 4,
        physical_alignment_weight: float = 0.02,
        optical_dists_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "task_probabilities",
            torch.tensor(task_probabilities or [0.5, 0.5], dtype=torch.float32),
        )
        self.physical_alignment_samples = physical_alignment_samples
        self.physical_alignment_weight = physical_alignment_weight
        self.optical_dists_weight = optical_dists_weight
        self.last_direction_losses: list[Tensor] = []

    @staticmethod
    def _assignments(batch_size: int, device: torch.device) -> Tensor:
        tasks = torch.arange(batch_size, device=device) % 2
        return tasks[torch.randperm(batch_size, device=device)]

    def _physical_direction(
        self,
        batch: dict[str, object],
        indices: Tensor,
        source_key: str,
        target_key: str,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        weights: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor], Pyramid]:
        source = batch[source_key][indices]  # type: ignore[index]
        target = batch[target_key][indices]  # type: ignore[index]
        valid = batch["valid"][indices]  # type: ignore[index]
        prediction, log_variance, pyramid = self.model.physical(
            source,
            source_spec,
            target_spec,
            valid,
            input_gsd=batch["input_gsd"][indices],  # type: ignore[index]
            target_gsd=batch["target_gsd"][indices],  # type: ignore[index]
            metadata=batch["metadata"][indices],  # type: ignore[index]
        )
        loss, metrics = physical_loss(
            prediction, log_variance, target, valid, target_spec.modality, weights[indices]
        )
        return loss, metrics, pyramid

    @staticmethod
    def _visual_target(target: Tensor, target_spec: SensorSpec) -> Tensor:
        return target[:, [2, 1, 0]] if target_spec.modality == "optical" else target

    @staticmethod
    def _visual_physical(physical: Tensor, target_spec: SensorSpec) -> Tensor:
        return physical[:, [2, 1, 0]] if target_spec.modality == "optical" else physical

    @staticmethod
    def _deterministic_target(
        target: Tensor,
        base: Tensor,
        valid: Tensor,
        target_spec: SensorSpec,
    ) -> Tensor:
        residual = (target - base.detach()) * valid
        if target_spec.modality == "sar":
            padded = F.pad(residual, (1, 1, 1, 1), mode="reflect")
            neighborhoods = padded.unfold(2, 3, 1).unfold(3, 3, 1)
            residual = neighborhoods.flatten(-2).median(dim=-1).values
        return highpass(residual) * valid

    def _physical_context(
        self,
        batch: dict[str, object],
        indices: Tensor,
        source_key: str,
        target_key: str,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        *,
        joint: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Pyramid]:
        context = nullcontext() if joint else torch.no_grad()
        with context:
            physical, _, pyramid = self.model.physical(
                batch[source_key][indices],  # type: ignore[index]
                source_spec,
                target_spec,
                batch["valid"][indices],  # type: ignore[index]
                input_gsd=batch["input_gsd"][indices],  # type: ignore[index]
                target_gsd=batch["target_gsd"][indices],  # type: ignore[index]
                metadata=batch["metadata"][indices],  # type: ignore[index]
            )
            if self.model.temporal_prior is not None:
                corrected: list[Tensor] = []
                batch_indices = indices.detach().cpu().tolist()
                for local_index, batch_index in enumerate(batch_indices):
                    pair_id = str(batch["pair_id"][batch_index])  # type: ignore[index]
                    _, location_id, s1_date, orbit, s2_date = pair_id.split(":")
                    window_values = batch["window"][batch_index].tolist()  # type: ignore[index]
                    pixel_window = tuple(int(value) for value in window_values)
                    transform_values = batch["augmentation"][batch_index].tolist()  # type: ignore[index]
                    spatial_transform = (
                        bool(transform_values[0]),
                        bool(transform_values[1]),
                        int(transform_values[2]),
                    )
                    acquired = s2_date if target_spec.modality == "optical" else s1_date
                    corrected.append(
                        self.model.apply_temporal_prior(
                            physical[local_index : local_index + 1],
                            target_spec,
                            acquired=acquired,
                            location_id=location_id,
                            pixel_window=pixel_window,  # type: ignore[arg-type]
                            orbit=orbit,
                            exclude_pair_id=pair_id,
                            spatial_transform=spatial_transform,
                        )[0]
                    )
                physical = torch.cat(corrected)
        target = self._visual_target(batch[target_key][indices], target_spec)  # type: ignore[index]
        base = self._visual_physical(physical, target_spec)
        valid = batch["valid"][indices]  # type: ignore[index]
        return target, base, valid, pyramid

    def _detail_direction(
        self,
        batch: dict[str, object],
        indices: Tensor,
        source_key: str,
        target_key: str,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        weights: Tensor,
        *,
        joint: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor], tuple[Tensor, Tensor, Tensor, Pyramid]]:
        target, base, valid, pyramid = self._physical_context(
            batch,
            indices,
            source_key,
            target_key,
            source_spec,
            target_spec,
            joint=joint,
        )
        prediction = self.model.deterministic_detail(
            pyramid, source_spec, target_spec, tuple(target.shape[-2:])
        )
        target_detail = self._deterministic_target(target, base, valid, target_spec)
        loss, metrics = deterministic_detail_loss(
            prediction,
            target_detail,
            valid,
            weights[indices],
            scale=0.08 if target_spec.modality == "optical" else 4.0,
        )
        zero_mae = masked_mean(target_detail.abs(), valid * weights[indices, None, None, None])
        predicted_mae = masked_mean(
            (prediction - target_detail).abs(), valid * weights[indices, None, None, None]
        )
        metrics["detail_mae_improvement"] = (
            (zero_mae - predicted_mae) / zero_mae.clamp_min(1e-8)
        ).detach()
        return loss, metrics, (target, base, prediction, pyramid)

    def _texture_target(
        self,
        target: Tensor,
        base: Tensor,
        detail: Tensor,
        valid: Tensor,
    ) -> Tensor:
        return (highpass((target - base.detach()) * valid) - detail.detach()) * valid

    @staticmethod
    def _optical_dists(
        prediction: Tensor,
        target: Tensor,
        valid: Tensor,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        from .evaluation import perceptual_evaluators

        sample_count = min(2, prediction.shape[0])
        prediction = prediction[:sample_count]
        target = target[:sample_count]
        valid = valid[:sample_count]
        if sample_weight is not None:
            sample_weight = sample_weight[:sample_count]
        _, evaluator = perceptual_evaluators(prediction.device)
        size = (min(64, prediction.shape[-2]), min(64, prediction.shape[-1]))

        def image(values: Tensor) -> Tensor:
            normalized = (0.5 + 0.5 * values / 0.08).clamp(0.0, 1.0) * valid
            return F.interpolate(normalized.float(), size=size, mode="area")

        values = evaluator(image(prediction), image(target)).flatten()
        if sample_weight is None:
            return values.mean()
        weights = sample_weight.to(values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def _codec_direction(
        self,
        batch: dict[str, object],
        indices: Tensor,
        source_key: str,
        target_key: str,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        weights: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        del source_key, source_spec
        target = self._visual_target(batch[target_key][indices], target_spec)  # type: ignore[index]
        valid = batch["valid"][indices]  # type: ignore[index]
        texture = highpass(target * valid) * valid
        raw_latent = self.model.codec.encode(texture, target_spec.modality, standardized=False)
        # Direction filtering differs by rank; collectives inside this branch would deadlock.
        self.model.codec.update_statistics(
            raw_latent, target_spec.modality, synchronize=False
        )
        latent = self.model.codec.normalize(raw_latent, target_spec.modality)
        decoded = self.model.codec.decode(latent, target_spec.modality)
        weight = weights[indices]
        loss, metrics = codec_reconstruction_loss(
            decoded, texture, valid * weight[:, None, None, None], target_spec.modality
        )
        if target_spec.modality == "optical" and self.optical_dists_weight > 0:
            dists = self._optical_dists(decoded, texture, valid, weight)
            loss = loss + self.optical_dists_weight * dists
            metrics["codec_dists"] = dists.detach()
        return loss, metrics

    def _flow_direction(
        self,
        batch: dict[str, object],
        indices: Tensor,
        source_key: str,
        target_key: str,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        weights: Tensor,
        *,
        joint: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        target, base, valid, pyramid = self._physical_context(
            batch,
            indices,
            source_key,
            target_key,
            source_spec,
            target_spec,
            joint=joint,
        )
        detail_context = nullcontext() if joint else torch.no_grad()
        with detail_context:
            detail = self.model.deterministic_detail(
                pyramid, source_spec, target_spec, tuple(target.shape[-2:])
            )
        texture = self._texture_target(target, base, detail, valid)
        with torch.no_grad():
            endpoint_latent = self.model.codec.encode(texture, target_spec.modality)
        noise = torch.randn_like(endpoint_latent)
        time_values = torch.rand(texture.shape[0], device=texture.device, dtype=texture.dtype)
        interpolation = (1 - time_values[:, None, None, None]) * noise + time_values[
            :, None, None, None
        ] * endpoint_latent
        velocity = self.model.flow_velocity(
            interpolation, time_values, pyramid, target_spec, texture.shape[1]
        )
        target_velocity = endpoint_latent - noise
        residual_weights = weights[indices]
        latent_mask = F.interpolate(valid, size=velocity.shape[-2:], mode="area")
        latent_weight = residual_weights[:, None, None, None] * latent_mask
        velocity_loss = masked_mean((velocity - target_velocity).square(), latent_weight)
        predicted_latent = interpolation + (1 - time_values[:, None, None, None]) * velocity
        endpoint = highpass(self.model.codec.decode(predicted_latent, target_spec.modality))
        endpoint_loss, endpoint_metrics = high_frequency_loss(
            endpoint,
            texture,
            valid,
            target_spec.modality,
            sample_weight=residual_weights,
        )
        endpoint_dists = None
        if target_spec.modality == "optical":
            endpoint_dists = self._optical_dists(endpoint, texture, valid, residual_weights)
        target_amplitude = robust_rms(texture, valid)
        amplitude_limit = (
            self.model.config.optical_residual_limit
            if target_spec.modality == "optical"
            else self.model.config.sar_residual_limit_db
        )
        target_amplitude = target_amplitude.clamp_max(amplitude_limit)
        predicted_amplitude = self.model.residual_amplitude(
            pyramid, target_spec, texture.shape[1], tuple(texture.shape[-2:])
        )
        amplitude_mask = F.avg_pool2d(valid, 4, stride=4)
        amplitude_loss = masked_mean(
            ((predicted_amplitude - target_amplitude) / amplitude_limit).abs(),
            residual_weights[:, None, None, None] * amplitude_mask,
        )
        total = velocity_loss + 0.25 * endpoint_loss + 0.2 * amplitude_loss
        if endpoint_dists is not None:
            total = total + 0.05 * endpoint_dists
        metrics = {
            "velocity": velocity_loss.detach(),
            "amplitude": amplitude_loss.detach(),
            **{f"endpoint_{name}": value for name, value in endpoint_metrics.items()},
        }
        if endpoint_dists is not None:
            metrics["endpoint_dists"] = endpoint_dists.detach()
        return total, metrics

    def forward(self, batch: dict[str, object], stage: str) -> tuple[Tensor, dict[str, Tensor]]:
        stage = "flow" if stage == "visual" else stage
        device = batch["s2"].device  # type: ignore[union-attr]
        batch_size = batch["s2"].shape[0]  # type: ignore[union-attr]
        tasks = self._assignments(batch_size, device)
        physical_weights, high_frequency_weights = time_weights(batch["delta_days"])  # type: ignore[arg-type]
        if "hf_eligible" in batch:
            high_frequency_weights = high_frequency_weights * batch["hf_eligible"].to(  # type: ignore[union-attr]
                high_frequency_weights.dtype
            )
        specifications = (
            ("sar_view", "s2_target", SENTINEL1, SENTINEL2),
            ("s2_view", "sar_target", SENTINEL2, SENTINEL1),
        )
        total = torch.zeros((), device=device)
        metrics: dict[str, Tensor] = {}
        self.last_direction_losses = []
        active = 0
        for task_index, (source_key, target_key, source_spec, target_spec) in enumerate(
            specifications
        ):
            indices = torch.nonzero(tasks == task_index, as_tuple=False).flatten()
            if stage in {"detail", "codec", "flow"}:
                indices = indices[high_frequency_weights[indices] > 0]
            if indices.numel() == 0:
                continue
            active += 1
            prefix = "sar2opt" if task_index == 0 else "opt2sar"
            if stage == "physical":
                loss, direction_metrics, _ = self._physical_direction(
                    batch,
                    indices,
                    source_key,
                    target_key,
                    source_spec,
                    target_spec,
                    physical_weights,
                )
                self.last_direction_losses.append(loss)
            elif stage == "detail":
                loss, direction_metrics, _ = self._detail_direction(
                    batch,
                    indices,
                    source_key,
                    target_key,
                    source_spec,
                    target_spec,
                    high_frequency_weights,
                )
            elif stage == "codec":
                loss, direction_metrics = self._codec_direction(
                    batch,
                    indices,
                    source_key,
                    target_key,
                    source_spec,
                    target_spec,
                    high_frequency_weights,
                )
            elif stage == "flow":
                loss, direction_metrics = self._flow_direction(
                    batch,
                    indices,
                    source_key,
                    target_key,
                    source_spec,
                    target_spec,
                    high_frequency_weights,
                )
            elif stage in {"balance", "overfit"}:
                physical_component, physical_metrics, _ = self._physical_direction(
                    batch,
                    indices,
                    source_key,
                    target_key,
                    source_spec,
                    target_spec,
                    physical_weights,
                )
                detail_component, detail_metrics, _ = self._detail_direction(
                    batch,
                    indices,
                    source_key,
                    target_key,
                    source_spec,
                    target_spec,
                    high_frequency_weights,
                    joint=True,
                )
                flow_component, flow_metrics = self._flow_direction(
                    batch,
                    indices,
                    source_key,
                    target_key,
                    source_spec,
                    target_spec,
                    high_frequency_weights,
                    joint=True,
                )
                loss = physical_component + 0.5 * detail_component + 0.5 * flow_component
                direction_metrics = {
                    **physical_metrics,
                    **{f"detail_{name}": value for name, value in detail_metrics.items()},
                    **{f"flow_{name}": value for name, value in flow_metrics.items()},
                }
            else:
                raise ValueError(f"unsupported training stage: {stage}")
            total = total + loss
            metrics.update(
                {f"{prefix}/{name}": value for name, value in direction_metrics.items()}
            )
        total = total / max(active, 1)

        if stage == "physical" and batch_size >= 2:
            count = min(self.physical_alignment_samples, batch_size)
            indices = torch.arange(count, device=device)
            valid = batch["valid"][indices]  # type: ignore[index]
            sar_scene = self.model.encode(
                batch["sar_view"][indices],
                SENTINEL1,
                valid,  # type: ignore[index]
                input_gsd=batch["input_gsd"][indices],  # type: ignore[index]
                target_gsd=batch["target_gsd"][indices],  # type: ignore[index]
                metadata=batch["metadata"][indices],  # type: ignore[index]
            )[-1]
            optical_scene = self.model.encode(
                batch["s2_view"][indices],
                SENTINEL2,
                valid,  # type: ignore[index]
                input_gsd=batch["input_gsd"][indices],  # type: ignore[index]
                target_gsd=batch["target_gsd"][indices],  # type: ignore[index]
                metadata=batch["metadata"][indices],  # type: ignore[index]
            )[-1]
            alignment, alignment_metrics = latent_alignment(sar_scene, optical_scene, valid)
            total = total + self.physical_alignment_weight * alignment
            metrics.update(
                {f"latent/{name}": value for name, value in alignment_metrics.items()}
            )

        if stage in {"detail", "codec", "flow"} and not bool(high_frequency_weights.any()):
            branch = {
                "detail": self.model.detail_head,
                "codec": self.model.codec,
                "flow": self.model.residual_dit,
            }[stage]
            total = total + _weighted_zero(branch, device)
        metrics["loss"] = total.detach()
        return total, metrics


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.tracked = tuple(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        self.state = {
            name: value.detach().float().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    def update(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        active = [
            name
            for name in self.tracked
            if name in parameters and parameters[name].grad is not None
        ]
        if active:
            torch._foreach_lerp_(
                [self.state[name] for name in active],
                [parameters[name].detach().float() for name in active],
                1.0 - self.decay,
            )
        for name, value in model.named_buffers():
            if name in self.state and value.is_floating_point():
                self.state[name].copy_(value.detach().float())

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "state": self.state}

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        model_state = model.state_dict()
        names = [name for name in self.state if name in model_state]
        backup = [model_state[name].detach().clone() for name in names]
        with torch.no_grad():
            for name in names:
                model_state[name].copy_(self.state[name].to(model_state[name]))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in zip(names, backup, strict=True):
                    model_state[name].copy_(value)

    def load_state_dict(
        self, payload: dict[str, object], device: torch.device, model: nn.Module
    ) -> None:
        self.decay = float(payload["decay"])
        state = payload["state"]  # type: ignore[assignment]
        model_state = model.state_dict()
        self.state = {
            name: value.to(device)
            for name, value in state.items()
            if name in model_state and value.shape == model_state[name].shape
        }


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _set_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]


def _scheduler(optimizer: AdamW, warmup: int, maximum: int) -> LambdaLR:
    def factor(step: int) -> float:
        if step < warmup:
            return max(1, step) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, maximum - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


def _stage_learning_rates(
    train_config: dict[str, Any], stage: str
) -> tuple[float, float, float]:
    base = float(train_config.get("learning_rate", 1e-4))
    encoder = float(train_config.get("encoder_learning_rate", 2e-5))
    if stage == "flow" or stage == "visual":
        return 0.0, 0.0, base
    if stage == "balance":
        balance = train_config.get("balance_learning_rates", {})
        return (
            float(balance.get("encoder", 2e-6)),
            float(balance.get("physical_detail", 1e-5)),
            float(balance.get("dit", 1e-5)),
        )
    return encoder, base, base


def _atomic_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _replace_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(target, link.parent))
    temporary.replace(link)


def _load_compatible_state(model: nn.Module, state: dict[str, Tensor]) -> tuple[int, int]:
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in state.items()
        if name in current
        and current[name].shape == value.shape
        and not (
            ".adapters." in name
            and name.endswith(".scale")
            and bool(torch.count_nonzero(value) == 0)
        )
    }
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(current) - len(compatible)


def _match_optimizer_layout(optimizer: AdamW) -> None:
    for parameter, state in optimizer.state.items():
        if parameter.ndim != 4 or not parameter.is_contiguous(
            memory_format=torch.channels_last
        ):
            continue
        for name, value in state.items():
            if isinstance(value, Tensor) and value.ndim == 4:
                state[name] = value.contiguous(memory_format=torch.channels_last)


def _set_trainable(model: SentinelV3, stage: str) -> None:
    stage = "flow" if stage == "visual" else stage
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules: tuple[nn.Module, ...]
    if stage == "physical":
        modules = (model.encoder, model.decoder)
    elif stage == "detail":
        modules = (model.detail_head,)
    elif stage == "codec":
        modules = (model.codec,)
    elif stage == "flow":
        modules = (model.residual_dit,)
    elif stage in {"balance", "overfit"}:
        modules = (model.encoder, model.decoder, model.detail_head, model.residual_dit)
    else:
        raise ValueError(f"unsupported training stage: {stage}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def _optimizer(
    model: SentinelV3, config: dict[str, Any], stage: str, device: torch.device
) -> AdamW:
    encoder_lr, main_lr, residual_lr = _stage_learning_rates(config, stage)
    adapter_lr = float(config.get("adapter_learning_rate", main_lr))
    if stage == "balance":
        adapter_lr = main_lr
    direction_modules = (
        model.encoder.adapters,
        model.encoder.modality_adapter,
        model.decoder.radiometry,
        model.decoder.radiometric_gate,
        model.decoder.radiometric_kernel,
        model.decoder.radiometric_condition,
        model.decoder.radiometric_descriptor,
        model.decoder.radiometric_bias,
        model.decoder.full_resolution_fusion,
        model.decoder.optical_direction_kernel,
        model.decoder.optical_amplitude_head,
        model.decoder.sar_spatial_kernel,
        model.decoder.sar_mean_condition,
        model.decoder.sar_mean_descriptor,
        model.decoder.sar_mean_head,
    )
    direction_parameters = [
        parameter for module in direction_modules for parameter in module.parameters()
    ]
    direction_ids = {id(parameter) for parameter in direction_parameters}
    shared_encoder_parameters = [
        parameter
        for parameter in model.encoder.parameters()
        if id(parameter) not in direction_ids
    ]
    groups = [
        {"name": "encoder", "params": shared_encoder_parameters, "lr": encoder_lr},
        {"name": "direction_adapters", "params": direction_parameters, "lr": adapter_lr},
        {
            "name": "physical_detail",
            "params": [
                parameter
                for parameter in model.decoder.parameters()
                if id(parameter) not in direction_ids
            ]
            + list(model.detail_head.parameters()),
            "lr": main_lr,
        },
        {"name": "codec", "params": list(model.codec.parameters()), "lr": main_lr},
        {"name": "dit", "params": list(model.residual_dit.parameters()), "lr": residual_lr},
    ]
    groups = [
        {
            **group,
            "params": [parameter for parameter in group["params"] if parameter.requires_grad],
        }
        for group in groups
    ]
    groups = [group for group in groups if group["params"]]
    return AdamW(
        groups,
        weight_decay=float(config["weight_decay"]),
        fused=device.type == "cuda",
    )


def _shared_physical_parameters(model: SentinelV3) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad and "adapters" not in name and "modality_adapter" not in name
    ]


def _pcgrad_corrections(
    losses: list[Tensor], parameters: list[nn.Parameter]
) -> list[Tensor | None]:
    if len(losses) < 2:
        return [None] * len(parameters)
    gradients = [
        torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        for loss in losses
    ]
    projected: list[list[Tensor | None]] = [list(values) for values in gradients]
    for left in range(len(projected)):
        for right in range(len(projected)):
            if left == right:
                continue
            pairs = [
                (left_gradient, right_gradient)
                for left_gradient, right_gradient in zip(
                    projected[left], gradients[right], strict=True
                )
                if left_gradient is not None and right_gradient is not None
            ]
            if not pairs:
                continue
            dot = sum(
                (left_gradient * right_gradient).sum()
                for left_gradient, right_gradient in pairs
            )
            norm = sum(
                (right_gradient.square().sum() for _, right_gradient in pairs),
                dot.new_zeros(()),
            )
            if float(dot.detach()) < 0:
                coefficient = dot / norm.clamp_min(1e-12)
                for index, right_gradient in enumerate(gradients[right]):
                    if projected[left][index] is not None and right_gradient is not None:
                        projected[left][index] = (
                            projected[left][index] - coefficient * right_gradient
                        )
    corrections: list[Tensor | None] = []
    for parameter_index in range(len(parameters)):
        raw = [
            values[parameter_index]
            for values in gradients
            if values[parameter_index] is not None
        ]
        adjusted = [
            values[parameter_index]
            for values in projected
            if values[parameter_index] is not None
        ]
        corrections.append(
            None if not raw else torch.stack(adjusted).mean(0) - torch.stack(raw).mean(0)
        )
    return corrections


def _apply_pcgrad_corrections(
    parameters: list[nn.Parameter], corrections: list[Tensor | None], distributed: bool
) -> None:
    for parameter, correction in zip(parameters, corrections, strict=True):
        if correction is None:
            continue
        if distributed:
            dist.all_reduce(correction)
            correction /= dist.get_world_size()
        if parameter.grad is None:
            parameter.grad = correction
        else:
            parameter.grad.add_(correction)


def _checkpoint_payload(
    *,
    model: SentinelV3,
    ema: EMA,
    optimizer: AdamW,
    scheduler: LambdaLR,
    stage: str,
    step: int,
    rank_states: list[object],
    config: dict[str, Any],
    validation_protocol_hash: str,
    best_metrics: dict[str, float],
    quality_gates: dict[str, bool],
    optimizer_states: dict[str, object] | None = None,
    scheduler_states: dict[str, object] | None = None,
) -> dict[str, object]:
    optimizer_states = dict(optimizer_states or {})
    scheduler_states = dict(scheduler_states or {})
    optimizer_states[stage] = optimizer.state_dict()
    scheduler_states[stage] = scheduler.state_dict()
    return {
        "format_version": 4,
        "stage": stage,
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer_states": optimizer_states,
        "scheduler_states": scheduler_states,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rank_states": rank_states,
        "config": config,
        "codec_version": model.codec.version,
        "validation_protocol_hash": validation_protocol_hash,
        "best_metrics": best_metrics,
        "quality_gates": quality_gates,
        "temporal_prior": config.get("temporal_prior"),
    }


def _validate_training_state(
    model: SentinelV3,
    config: dict[str, Any],
    stage: str,
    step: int,
    *,
    full: bool,
) -> dict[str, Any]:
    validation = config.get("validation", {})
    if not bool(validation.get("enabled", False)):
        return {}
    from .evaluation import (
        evaluate_high_frequency_components,
        evaluate_model,
        evaluate_physical_model,
    )

    limit = None if full else int(validation.get("quick_samples", 32))
    if stage in {"detail", "codec"}:
        return evaluate_high_frequency_components(
            model,
            config["paths"]["manifest"],
            str(validation.get("split", "validation_temporal")),
            stage,
            limit=limit,
        )
    if stage == "physical":
        return evaluate_physical_model(
            model,
            config["paths"]["manifest"],
            str(validation.get("split", "validation_temporal")),
            limit=limit,
        )
    return evaluate_model(
        model,
        config["paths"]["manifest"],
        str(validation.get("split", "validation_temporal")),
        seed=int(config["train"]["seed"]),
        limit=limit,
        panels=32,
        panel_root=Path(config["paths"]["reports"]) / stage / f"step_{step:07d}_panels",
    )


def _score(report: dict[str, Any], kind: str) -> float:
    if kind == "physical":
        ratios = (
            float(report.get("sar2opt_rmse", float("inf"))) / 0.03909,
            float(report.get("sar2opt_sam_deg", float("inf"))) / 5.716,
            float(report.get("opt2sar_rmse_db", float("inf"))) / 5.0,
            float(report.get("opt2sar_physical_bias_db", float("inf"))) / 0.5,
        )
        return -max(ratios)
    if kind == "detail":
        return min(
            float(report.get("optical_detail_mae_improvement", -float("inf"))),
            float(report.get("sar_detail_mae_improvement", -float("inf"))),
        )
    if kind == "codec":
        return -max(
            float(report.get("optical_codec_mae", float("inf"))) / 0.02,
            float(report.get("sar_codec_mae", float("inf"))) / 1.0,
        )
    if kind == "visual":
        return float(report.get("lpips_improvement", -float("inf"))) + float(
            report.get("dists_improvement", -float("inf"))
        )
    return _score(report, "physical") + _score(report, "visual")


def train(
    config: dict[str, Any],
    *,
    resume: str | None = None,
    init_model: str | None = None,
    init: str | None = None,
    limit: int | None = None,
) -> None:
    # init is retained only as a Python compatibility alias; the CLI exposes --init-model.
    init_model = init_model or init
    if resume and init_model:
        raise ValueError("resume and init_model are mutually exclusive")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_config = config["train"]
    stage = "flow" if train_config["stage"] == "visual" else str(train_config["stage"])
    channels_last = bool(train_config.get("channels_last", False)) and device.type == "cuda"
    seed = int(train_config["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset: Any = V2ShardDataset(
        config["paths"]["train_shards"],
        augment=True,
        random_gsd=True,
        native_gsd_probability=(
            1.0
            if stage in {"detail", "codec", "flow", "balance"}
            else float(train_config.get("native_gsd_probability", 0.8))
        ),
        audit_high_frequency=stage in {"detail", "codec", "flow", "balance"}
        and bool(train_config.get("registration_audit", True)),
    )
    if limit is not None:
        if stage in {"detail", "codec", "flow", "balance"}:
            eligible_indices: list[int] = []
            for index in range(len(dataset)):
                if bool(dataset[index]["hf_eligible"]):
                    eligible_indices.append(index)
                    if len(eligible_indices) == limit:
                        break
            if len(eligible_indices) < limit:
                raise RuntimeError(
                    f"requested {limit} high-frequency patches but found "
                    f"only {len(eligible_indices)} eligible patches"
                )
            dataset = Subset(dataset, eligible_indices)
        else:
            dataset = Subset(dataset, range(min(limit, len(dataset))))
    sampler = (
        None
        if isinstance(dataset, Subset)
        else StatefulShardSampler(dataset, replicas=world_size, rank=rank, seed=seed)
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        sampler=sampler,
        shuffle=isinstance(dataset, Subset),
        num_workers=int(train_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    model = SentinelV3(ModelConfig(**config["model"])).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    _set_trainable(model, stage)
    if stage == "physical" and not bool(
        train_config.get("train_full_resolution_fusion", False)
    ):
        for parameter in model.decoder.full_resolution_fusion.parameters():
            parameter.requires_grad_(False)
    optimizer = _optimizer(model, train_config, stage, device)
    scheduler = _scheduler(
        optimizer, int(train_config["warmup_steps"]), int(train_config["max_steps"])
    )
    objective = JointObjective(
        model,
        list(train_config.get("task_probabilities", [0.5, 0.5])),
        int(train_config.get("physical_alignment_samples", 4)),
        float(train_config.get("physical_alignment_weight", 0.02)),
        float(train_config.get("optical_dists_weight", 0.1)),
    ).to(device)
    ema = EMA(model, float(train_config["ema_decay"]))
    step = 0
    best_metrics: dict[str, float] = {
        "physical_score": -float("inf"),
        "visual_score": -float("inf"),
        "joint_score": -float("inf"),
        "early_stop_score": -float("inf"),
        "quick_early_stop_score": -float("inf"),
        "full_early_stop_score": -float("inf"),
        "physical_candidate_score": -float("inf"),
    }
    quality_gates: dict[str, bool] = {}
    optimizer_states: dict[str, object] = {}
    scheduler_states: dict[str, object] = {}
    protocol_hash = str(config.get("validation", {}).get("protocol_hash", "unresolved"))

    if init_model:
        initial = torch.load(init_model, map_location="cpu", weights_only=False)
        initial_state = dict(initial["model"])
        temporal_config = initial.get("temporal_prior") or initial.get("config", {}).get(
            "temporal_prior"
        )
        if temporal_config:
            config["temporal_prior"] = temporal_config
            model.configure_temporal_prior(temporal_config)
        if bool(train_config.get("init_use_ema", False)) and "ema" in initial:
            initial_state.update(initial["ema"]["state"])
        loaded, initialized = _load_compatible_state(model, initial_state)
        if (
            stage in {"detail", "codec", "flow", "balance"}
            and bool(train_config.get("require_physical_gate", True))
            and not bool(initial.get("quality_gates", {}).get("physical", False))
        ):
            raise RuntimeError(
                "high-frequency training requires a frozen physical checkpoint that passed validation"
            )
        if (
            stage in {"flow", "balance"}
            and bool(train_config.get("require_codec_gate", True))
            and not bool(initial.get("quality_gates", {}).get("codec", False))
        ):
            raise RuntimeError(
                "flow training requires a frozen codec checkpoint that passed reconstruction gates"
            )
        if (
            stage in {"flow", "balance"}
            and bool(train_config.get("require_detail_gate", True))
            and not bool(initial.get("quality_gates", {}).get("detail", False))
        ):
            raise RuntimeError(
                "flow training requires a deterministic-detail checkpoint that passed its gate"
            )
        if int(initial.get("format_version", 0)) == 4:
            optimizer_states.update(initial.get("optimizer_states", {}))
            scheduler_states.update(initial.get("scheduler_states", {}))
            best_metrics.update(initial.get("best_metrics", {}))
            quality_gates.update(initial.get("quality_gates", {}))
            protocol_hash = str(initial.get("validation_protocol_hash", protocol_hash))
        if rank == 0:
            print(
                json.dumps(
                    {
                        "init_model": init_model,
                        "compatible_tensors": loaded,
                        "new_tensors": initialized,
                    }
                ),
                flush=True,
            )
        ema = EMA(model, float(train_config["ema_decay"]))

    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if int(checkpoint.get("format_version", 0)) != 4:
            raise RuntimeError(
                "only V3.2 format-v4 checkpoints may be resumed; use --init-model for older weights"
            )
        if checkpoint["stage"] != stage:
            raise RuntimeError("a checkpoint may only resume the same V3.2 training stage")
        model.load_state_dict(checkpoint["model"])
        temporal_config = checkpoint.get("temporal_prior") or checkpoint.get("config", {}).get(
            "temporal_prior"
        )
        if temporal_config:
            config["temporal_prior"] = temporal_config
            model.configure_temporal_prior(temporal_config)
        optimizer.load_state_dict(checkpoint["optimizer_states"][stage])
        scheduler.load_state_dict(checkpoint["scheduler_states"][stage])
        ema.load_state_dict(checkpoint["ema"], device, model)
        step = int(checkpoint["step"])
        best_metrics.update(checkpoint.get("best_metrics", {}))
        quality_gates.update(checkpoint.get("quality_gates", {}))
        optimizer_states.update(checkpoint.get("optimizer_states", {}))
        scheduler_states.update(checkpoint.get("scheduler_states", {}))
        protocol_hash = str(checkpoint.get("validation_protocol_hash", protocol_hash))
        states = checkpoint["rank_states"]
        state = states[rank] if rank < len(states) else states[0]
        _set_rng_state(state["rng"])
        if sampler is not None and state.get("sampler") is not None:
            sampler.load_state_dict(state["sampler"])
        if channels_last:
            _match_optimizer_layout(optimizer)

    if (
        stage in {"detail", "codec", "flow", "balance"}
        and bool(train_config.get("require_physical_gate", True))
        and not (resume or init_model)
    ):
        raise RuntimeError(
            "high-frequency stages require --init-model with a passing physical v4 checkpoint"
        )
    if (
        stage in {"flow", "balance"}
        and bool(train_config.get("require_codec_gate", True))
        and not (resume or init_model)
    ):
        raise RuntimeError(
            "flow stages require --init-model with a passing codec v4 checkpoint"
        )

    wrapped: nn.Module = objective
    if distributed:
        wrapped = DistributedDataParallel(
            objective,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=bool(
                train_config.get("find_unused_parameters", True)
            ),
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
    amp_name = str(train_config["amp"])
    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[amp_name]
    accumulation = int(train_config["gradient_accumulation"])
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    output = Path(config["paths"]["output"])
    started = time.time()
    starting_step = step
    stale_validations = 0
    while step < int(train_config["max_steps"]):
        aggregate: dict[str, float] = {}
        pcgrad_corrections: tuple[list[nn.Parameter], list[Tensor | None]] | None = None
        for micro_step in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
                for key, value in batch.items()
            }
            sync_context = (
                wrapped.no_sync()  # type: ignore[attr-defined]
                if distributed and micro_step + 1 < accumulation
                else nullcontext()
            )
            amp_context = (
                torch.autocast(device.type, dtype=amp_dtype)
                if device.type == "cuda" and amp_dtype != torch.float32
                else nullcontext()
            )
            with sync_context, amp_context:
                loss, metrics = wrapped(batch, stage)  # type: ignore[operator]
                loss = loss / accumulation
            if not bool(torch.isfinite(loss.detach())):
                finite_metrics = {
                    name: float(value.detach())
                    for name, value in metrics.items()
                    if bool(torch.isfinite(value.detach()))
                }
                raise FloatingPointError(
                    f"non-finite {stage} loss at step {step}, micro-step {micro_step}; "
                    f"finite metrics={finite_metrics}"
                )
            if (
                stage == "physical"
                and bool(train_config.get("pcgrad", True))
                and micro_step + 1 == accumulation
            ):
                shared = _shared_physical_parameters(model)
                corrections = _pcgrad_corrections(
                    [value / accumulation for value in objective.last_direction_losses], shared
                )
                pcgrad_corrections = (shared, corrections)
            loss.backward()
            for name, value in metrics.items():
                aggregate[name] = aggregate.get(name, 0.0) + float(value) / accumulation
        if pcgrad_corrections is not None:
            _apply_pcgrad_corrections(*pcgrad_corrections, distributed)
        for group in optimizer.param_groups:
            nn.utils.clip_grad_norm_(group["params"], float(train_config["gradient_clip"]))
        optimizer.step()
        scheduler.step()
        ema.update(model)
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if rank == 0 and step % int(train_config["log_every"]) == 0:
            elapsed = time.time() - started
            processed = (step - starting_step) * int(train_config["batch_size"]) * world_size
            print(
                json.dumps(
                    {
                        "step": step,
                        "seconds": round(elapsed, 1),
                        "samples_per_second": round(
                            processed * accumulation / max(elapsed, 1e-6), 1
                        ),
                        **aggregate,
                    }
                ),
                flush=True,
            )

        validate_every = int(train_config["validate_every"])
        save_every = int(train_config["save_every"])
        final_step = step == int(train_config["max_steps"])
        should_validate = step % validate_every == 0 or final_step
        should_save = (
            step % save_every == 0
            or should_validate
            or (final_step and bool(train_config.get("save_final", True)))
        )
        report: dict[str, Any] = {}
        improved_kinds: set[str] = set()
        if should_validate:
            full_every = int(train_config.get("full_validate_every", 5000))
            full_steps = {
                int(value) for value in config.get("validation", {}).get("full_steps", [])
            }
            is_full_validation = step % full_every == 0 or step in full_steps
            if rank == 0:
                with ema.apply_to(model):
                    report = _validate_training_state(
                        model, config, stage, step, full=is_full_validation
                    )
            if distributed:
                report_objects = [report]
                dist.broadcast_object_list(report_objects, src=0)
                report = report_objects[0]
            if report:
                improved = False
                early_kind = stage if stage in {"physical", "detail", "codec"} else "visual"
                early_score = _score(report, early_kind)
                scope = "full" if is_full_validation else "quick"
                early_key = f"{scope}_early_stop_score"
                if early_score > best_metrics[early_key]:
                    best_metrics[early_key] = early_score
                    best_metrics["early_stop_score"] = max(
                        best_metrics["quick_early_stop_score"],
                        best_metrics["full_early_stop_score"],
                    )
                    improved = True
                if is_full_validation:
                    quality_gates.update(report.get("quality_gates", {}))
                    if (
                        stage == "physical"
                        and early_score > best_metrics["physical_candidate_score"]
                    ):
                        best_metrics["physical_candidate_score"] = early_score
                        improved_kinds.add("physical_candidate")
                    for kind in ("physical", "visual", "joint"):
                        value = _score(report, kind)
                        key = f"{kind}_score"
                        gate_passed = bool(report.get("quality_gates", {}).get(kind, False))
                        if gate_passed and value > best_metrics[key]:
                            best_metrics[key] = value
                            improved_kinds.add(kind)
                if not is_full_validation:
                    stale_validations = 0 if improved else stale_validations + 1
            if rank == 0:
                validation_report = report or {
                    "stage": stage,
                    "step": step,
                    "quality_gates": quality_gates,
                    "validation_skipped": stage == "codec",
                }
                validation_report["stage"] = stage
                validation_report["step"] = step
                validation_report["training_metrics"] = aggregate
                report_path = Path(config["paths"]["reports"]) / stage / f"step_{step:07d}.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(validation_report, indent=2) + "\n", encoding="utf-8"
                )
        local_state = {
            "rng": _rng_state(),
            "sampler": sampler.state_dict() if sampler is not None else None,
        }
        if should_save:
            if distributed:
                rank_states: list[object] = [None] * world_size
                dist.all_gather_object(rank_states, local_state)
            else:
                rank_states = [local_state]
            if rank == 0:
                payload = _checkpoint_payload(
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    stage=stage,
                    step=step,
                    rank_states=rank_states,
                    config=config,
                    validation_protocol_hash=protocol_hash,
                    best_metrics=best_metrics,
                    quality_gates=quality_gates,
                    optimizer_states=optimizer_states,
                    scheduler_states=scheduler_states,
                )
                stage_path = output / stage / f"step_{step:07d}.pt"
                _atomic_save(payload, stage_path)
                _replace_symlink(stage_path, output / "latest.pt")
                if report:
                    for kind in improved_kinds:
                        if bool(report.get("quality_gates", {}).get(kind, False)):
                            _replace_symlink(stage_path, output / f"best_{kind}.pt")
                        elif kind == "physical_candidate":
                            _replace_symlink(stage_path, output / "best_physical_candidate.pt")
        if should_validate and stale_validations >= int(
            train_config.get("early_stop_patience", 5)
        ):
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "early_stop": True,
                            "step": step,
                            "stale_validations": stale_validations,
                        }
                    ),
                    flush=True,
                )
            break
    if distributed:
        dist.destroy_process_group()
