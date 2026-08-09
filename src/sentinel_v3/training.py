from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from datetime import timedelta
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

from .data import StatefulIndexSampler, StatefulShardSampler, V2ShardDataset, time_weights
from .losses import (
    anchor_gain_target,
    charbonnier,
    codec_reconstruction_loss,
    cross_modal_identifiability_target,
    detail_reliability_target,
    deterministic_detail_loss,
    deterministic_detail_target,
    frequency_bands,
    high_frequency_loss,
    highpass,
    latent_alignment,
    low_frequency_loss,
    masked_mean,
    phase_alignment_loss,
    phase_identifiability_target,
    phase_transport_gain_target,
    physical_loss,
    robust_rms,
    texture_reliability_gate,
)
from .model import ModelConfig, Pyramid, SentinelV3
from .sensors import SENTINEL1, SENTINEL2, SensorSpec


def _weighted_zero(module: nn.Module, device: torch.device) -> Tensor:
    terms = [
        parameter.sum() * 0.0 for parameter in module.parameters() if parameter.requires_grad
    ]
    return sum(terms, torch.zeros((), device=device))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_high_frequency_eligibility(
    eligibility_path: str | Path, train_index: str | Path
) -> list[int]:
    """Load an explicit registration audit bound to exactly one shard index."""

    sidecar_path = Path(eligibility_path).resolve()
    index_path = Path(train_index).resolve()
    values = json.loads(sidecar_path.read_text(encoding="utf-8"))
    format_version = int(values.get("format_version", 1))
    source_value = values.get("source_index")
    if source_value is None:
        raise RuntimeError("eligibility sidecar must declare source_index")
    source_path = Path(str(source_value))
    if not source_path.is_absolute():
        source_path = (sidecar_path.parent / source_path).resolve()
    else:
        source_path = source_path.resolve()
    if source_path != index_path:
        raise RuntimeError("eligibility sidecar belongs to a different training index")
    if values.get("registration_audited") is False or (
        format_version >= 2 and values.get("registration_audited") is not True
    ):
        raise RuntimeError("high-frequency eligibility sidecar has not completed registration audit")
    if format_version >= 2 and values.get("source_index_sha256") != _file_sha256(index_path):
        raise RuntimeError("eligibility sidecar source_index_sha256 does not match")
    eligible = values.get("eligible_indices")
    if not isinstance(eligible, list):
        raise TypeError("eligibility sidecar eligible_indices must be a list")
    return [int(value) for value in eligible]


def _data_loader_worker_options(train_config: dict[str, Any]) -> dict[str, object]:
    workers = int(train_config["num_workers"])
    options: dict[str, object] = {"num_workers": workers}
    if workers > 0:
        options["persistent_workers"] = bool(train_config["persistent_workers"])
        options["prefetch_factor"] = int(train_config["prefetch_factor"])
    return options


@torch.no_grad()
def _stable_clip_grad_norm_(parameters: list[nn.Parameter], max_norm: float) -> tuple[float, float]:
    """Clip a parameter group without overflowing the FP32 norm reduction."""

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0, 0.0
    maxima = torch.stack([gradient.detach().abs().amax().float() for gradient in gradients])
    if not bool(torch.isfinite(maxima).all()):
        raise FloatingPointError("non-finite gradient values before clipping")
    maximum = float(maxima.max())
    if maximum == 0.0:
        return 0.0, 0.0
    scale = 1.0 / maximum
    scaled_square = sum(
        (gradient.detach().float() * scale).square().sum()
        for gradient in gradients
    )
    norm = maximum * math.sqrt(float(scaled_square))
    coefficient = min(1.0, max_norm / max(norm, 1e-12))
    for gradient in gradients:
        gradient.mul_(coefficient)
    return norm, maximum


def texture_benefit_target(
    physical: Tensor,
    candidate: Tensor,
    target: Tensor,
    valid: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return soft block correctness, signed benefit, and strict-valid 4x4 support."""

    def local_risk(prediction: Tensor) -> Tensor:
        pixel = (prediction - target).abs().mean(dim=1, keepdim=True) / 0.05
        structure = (highpass(prediction) - highpass(target)).abs().mean(
            dim=1, keepdim=True
        ) / 0.02
        return F.avg_pool2d(pixel + 0.25 * structure, 4, stride=4)

    physical_risk = local_risk(physical)
    candidate_risk = local_risk(candidate)
    benefit = physical_risk - candidate_risk
    margin = 0.05 * physical_risk
    scale = (0.05 * physical_risk + 0.01).clamp_min(0.01)
    correctness = torch.sigmoid((benefit - margin) / scale)
    block_valid = (F.avg_pool2d(valid, 4, stride=4) >= 0.999).to(valid.dtype)
    return correctness, benefit, block_valid


class JointObjective(nn.Module):
    """Stage-aware V3.2 objective with explicit deterministic/stochastic separation."""

    def __init__(
        self,
        model: SentinelV3,
        task_probabilities: list[float] | None = None,
        physical_alignment_samples: int = 4,
        physical_alignment_weight: float = 0.02,
        physical_alignment_every: int = 1,
        optical_dists_weight: float = 0.1,
        flow_perceptual_every: int = 8,
        codec_train_modality: str | None = None,
        codec_perceptual_every: int = 8,
        flow_visual_pixel_weight: float = 0.1,
        flow_visual_hf_weight: float = 0.1,
        flow_visual_perceptual_weight: float = 0.025,
        flow_rollout_every: int | None = None,
        flow_rollout_steps: int = 2,
        flow_rollout_samples: int = 2,
        flow_rollout_pixel_weight: float = 0.1,
        flow_rollout_hf_weight: float = 0.1,
        id_bridge_antithetic_weight: float = 0.0,
        risk_flow_steps: int = 4,
        bridge_flow_steps: int = 4,
        phase_transport_hf_weight: float = 0.05,
        phase_transport_utility_weight: float = 0.10,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "task_probabilities",
            torch.tensor(task_probabilities or [0.5, 0.5], dtype=torch.float32),
        )
        self.physical_alignment_samples = physical_alignment_samples
        self.physical_alignment_weight = physical_alignment_weight
        if physical_alignment_every < 1:
            raise ValueError("physical_alignment_every must be positive")
        self.physical_alignment_every = physical_alignment_every
        self.optical_dists_weight = optical_dists_weight
        if codec_train_modality not in {None, "optical", "sar"}:
            raise ValueError("codec_train_modality must be optical, sar, or null")
        self.codec_train_modality = codec_train_modality
        self.codec_perceptual_every = max(1, codec_perceptual_every)
        self.flow_visual_pixel_weight = flow_visual_pixel_weight
        self.flow_visual_hf_weight = flow_visual_hf_weight
        self.flow_visual_perceptual_weight = flow_visual_perceptual_weight
        # flow_perceptual_every remains an initialization alias for v5 configs.
        self.flow_rollout_every = max(
            1, flow_perceptual_every if flow_rollout_every is None else flow_rollout_every
        )
        self.flow_rollout_steps = max(1, flow_rollout_steps)
        self.flow_rollout_samples = max(1, flow_rollout_samples)
        self.flow_rollout_pixel_weight = flow_rollout_pixel_weight
        self.flow_rollout_hf_weight = flow_rollout_hf_weight
        if not math.isfinite(id_bridge_antithetic_weight) or id_bridge_antithetic_weight < 0.0:
            raise ValueError("id_bridge_antithetic_weight must be finite and non-negative")
        self.id_bridge_antithetic_weight = id_bridge_antithetic_weight
        if not math.isfinite(phase_transport_hf_weight) or phase_transport_hf_weight < 0.0:
            raise ValueError("phase_transport_hf_weight must be finite and non-negative")
        self.phase_transport_hf_weight = phase_transport_hf_weight
        if not math.isfinite(phase_transport_utility_weight) or phase_transport_utility_weight < 0.0:
            raise ValueError("phase_transport_utility_weight must be finite and non-negative")
        self.phase_transport_utility_weight = phase_transport_utility_weight
        self.risk_flow_steps = max(1, risk_flow_steps)
        self.bridge_flow_steps = max(1, bridge_flow_steps)
        self.last_direction_losses: list[Tensor] = []
        self.progress = 0.0
        self.current_step = 0

    def set_progress(self, step: int, max_steps: int) -> None:
        self.current_step = step
        self.progress = min(1.0, max(0.0, step / max(max_steps, 1)))

    def _frequency_curriculum(self, reference: Tensor) -> Tensor:
        if self.progress < 0.15:
            values = (0.0, 0.25, 1.0)
        elif self.progress < 0.45:
            values = (0.25, 1.0, 1.0)
        else:
            values = (1.0, 1.0, 1.0)
        return reference.new_tensor(values)

    @staticmethod
    def _assignments(batch_size: int, device: torch.device) -> Tensor:
        tasks = torch.arange(batch_size, device=device) % 2
        return tasks[torch.randperm(batch_size, device=device)]

    def _id_bridge_assignments(
        self,
        batch_size: int,
        device: torch.device,
        *,
        rank: int | None = None,
    ) -> Tensor:
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if self.model.config.id_bridge_optical_only:
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        return (torch.arange(batch_size, device=device) + self.current_step + rank) % 2

    @staticmethod
    def _id_utility_assignments(batch_size: int, device: torch.device) -> Tensor:
        """Train the Optical anchor utility on SAR-to-Optical examples only."""

        return torch.zeros(batch_size, device=device, dtype=torch.long)

    @staticmethod
    def _phase_transport_assignments(batch_size: int, device: torch.device) -> Tensor:
        """Train observable Optical transport from SAR inputs only."""

        return torch.zeros(batch_size, device=device, dtype=torch.long)

    @staticmethod
    def _id_bridge_start(
        mu: Tensor,
        log_sigma: Tensor,
        reliability_logits: Tensor,
        noise_scale: float,
        epsilon: Tensor,
        *,
        q_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if q_state is None:
            q_pred = torch.sigmoid(reliability_logits).mean(dim=1, keepdim=True)
        else:
            if q_state.shape not in {
                mu.shape,
                (mu.shape[0], 1, *mu.shape[-2:]),
            }:
                raise ValueError("id bridge reliability state must be B1HW or match the latent")
            q_pred = q_state
        sigma = noise_scale * torch.sigmoid(log_sigma) * (1.0 - q_pred)
        # Distribution parameters are trained by their own calibrated objectives only.
        return mu + sigma.detach() * epsilon, q_pred, sigma

    @staticmethod
    def _id_bridge_anchor_values(
        mu: Tensor,
        correction: Tensor,
        endpoint: Tensor,
        q_oracle: Tensor,
    ) -> Tensor:
        if mu.shape != correction.shape or mu.shape != endpoint.shape:
            raise ValueError("id bridge anchor tensors must share a latent shape")
        if q_oracle.shape not in {
            mu.shape,
            (mu.shape[0], 1, *mu.shape[-2:]),
        }:
            raise ValueError("id bridge oracle must be B1HW or match the latent grid")
        values = q_oracle * F.smooth_l1_loss(mu, endpoint, reduction="none")
        return values + (1.0 - q_oracle) * 0.05 * correction.abs()

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
        return deterministic_detail_target(target, base, valid, target_spec.modality)

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
                prior_key = f"{target_spec.modality}_temporal_prior"
                coverage_key = f"{target_spec.modality}_temporal_coverage"
                if prior_key in batch:
                    physical = self.model.temporal_prior.compose(
                        physical,
                        batch[prior_key][indices].to(physical.dtype),  # type: ignore[index]
                        batch[coverage_key][indices],  # type: ignore[index]
                        target_spec.modality,
                    )[0]
                else:
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
        prediction, predicted_bands, confidence = (
            self.model.deterministic_detail_with_confidence(
                pyramid, source_spec, target_spec, tuple(target.shape[-2:]), base=base
            )
        )
        target_detail = self._deterministic_target(target, base, valid, target_spec)
        target_bands = frequency_bands(target_detail, levels=3)
        source = batch[source_key][indices]  # type: ignore[index]
        reliability = detail_reliability_target(source, target_bands, valid)
        correctness_targets = []
        for predicted_band, target_band in zip(predicted_bands, target_bands, strict=True):
            zero_error = target_band.detach().abs().mean(dim=1, keepdim=True)
            predicted_error = (
                (predicted_band.detach() - target_band).abs().mean(dim=1, keepdim=True)
            )
            error_scale = F.avg_pool2d(zero_error, 4, stride=4).clamp_min(1e-4)
            benefit = F.avg_pool2d(zero_error - predicted_error, 4, stride=4)
            correctness_targets.append(torch.sigmoid(4.0 * benefit / error_scale))
        correctness = torch.cat(correctness_targets, dim=1)
        curriculum = self._frequency_curriculum(target)
        loss = target.new_zeros(())
        metrics: dict[str, Tensor] = {}
        for level, (predicted_band, target_band) in enumerate(
            zip(predicted_bands, target_bands, strict=True)
        ):
            support = F.interpolate(
                reliability[:, level : level + 1],
                size=valid.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            band_loss, band_metrics = deterministic_detail_loss(
                predicted_band,
                target_band,
                valid * support,
                weights[indices],
                scale=0.08 if target_spec.modality == "optical" else 4.0,
            )
            loss = loss + curriculum[level] * band_loss
            metrics.update(
                {f"band{level}_{name}": value for name, value in band_metrics.items()}
            )
        confidence_weight = (
            F.avg_pool2d(valid, 4, stride=4)
            * weights[indices, None, None, None]
            * (0.25 + reliability)
        )
        probability = confidence.float().clamp(1e-5, 1.0 - 1e-5)
        confidence_target = correctness.float()
        confidence_cross_entropy = -(
            confidence_target * probability.log()
            + (1.0 - confidence_target) * (1.0 - probability).log()
        )
        confidence_loss = masked_mean(confidence_cross_entropy, confidence_weight.float())
        composed_loss, composed_metrics = deterministic_detail_loss(
            prediction,
            target_detail,
            valid,
            weights[indices],
            scale=0.08 if target_spec.modality == "optical" else 4.0,
        )
        loss = (
            loss / curriculum.sum().clamp_min(1.0) + 0.2 * composed_loss + 0.1 * confidence_loss
        )
        metrics.update(composed_metrics)
        metrics["detail_confidence"] = confidence.mean().detach()
        metrics["detail_reliability"] = reliability.mean().detach()
        metrics["detail_correctness_target"] = correctness.mean().detach()
        metrics["detail_confidence_loss"] = confidence_loss.detach()
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

    @staticmethod
    def _optical_visual_perceptual(
        prediction: Tensor,
        target: Tensor,
        valid: Tensor,
        sample_weight: Tensor,
    ) -> tuple[Tensor, Tensor]:
        from .evaluation import perceptual_evaluators

        sample_count = min(2, prediction.shape[0])
        prediction = prediction[:sample_count]
        target = target[:sample_count]
        valid = valid[:sample_count]
        weights = sample_weight[:sample_count].float()
        size = (min(64, prediction.shape[-2]), min(64, prediction.shape[-1]))

        def image(values: Tensor) -> Tensor:
            return F.interpolate(
                (values.clamp(0.0, 1.0) * valid).float(), size=size, mode="area"
            )

        predicted_image = image(prediction)
        target_image = image(target)
        lpips_evaluator, dists_evaluator = perceptual_evaluators(prediction.device)
        # The surrounding training step may use BF16 autocast. Perceptual backbones
        # have substantially larger cross-rank gradient variance in BF16, so keep
        # this sparse auxiliary loss in FP32.
        with torch.autocast(prediction.device.type, enabled=False):
            lpips_values = lpips_evaluator(
                predicted_image * 2.0 - 1.0, target_image * 2.0 - 1.0
            ).flatten()
            dists_values = dists_evaluator(predicted_image, target_image).flatten()
        denominator = weights.sum().clamp_min(1e-8)
        return (
            (lpips_values * weights).sum() / denominator,
            (dists_values * weights).sum() / denominator,
        )

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
        self.model.codec.update_statistics(raw_latent, target_spec.modality, synchronize=False)
        latent = self.model.codec.normalize(raw_latent, target_spec.modality)
        decoded = self.model.codec.decode(latent, target_spec.modality)
        weight = weights[indices]
        loss, metrics = codec_reconstruction_loss(
            decoded, texture, valid * weight[:, None, None, None], target_spec.modality
        )
        if (
            target_spec.modality == "optical"
            and self.optical_dists_weight > 0
            and self.current_step % self.codec_perceptual_every == 0
        ):
            dists = self._optical_dists(decoded, texture, valid, weight)
            loss = loss + self.optical_dists_weight * self.codec_perceptual_every * dists
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
                pyramid, source_spec, target_spec, tuple(target.shape[-2:]), base=base
            )
        texture = self._texture_target(target, base, detail, valid)
        teacher_visual_enabled = (
            self.flow_visual_pixel_weight > 0.0 or self.flow_visual_hf_weight > 0.0
        )
        rollout_enabled = (
            self.flow_rollout_pixel_weight > 0.0
            or self.flow_rollout_hf_weight > 0.0
            or self.flow_visual_perceptual_weight > 0.0
        )
        if target_spec.modality == "optical" and (teacher_visual_enabled or rollout_enabled):
            source = batch[source_key][indices]  # type: ignore[index]
            threshold = 0.35 + 0.05 * torch.rand((), device=texture.device)
            texture_reliability, texture_gate = texture_reliability_gate(
                source, texture, valid, threshold=threshold
            )
        else:
            texture_reliability = F.avg_pool2d(valid, 4, stride=4)
            texture_gate = torch.ones_like(texture_reliability)
        with torch.no_grad():
            endpoint_latent = self.model.codec.encode(texture, target_spec.modality)
        noise = self.model.flow_noise_scale(target_spec) * torch.randn_like(endpoint_latent)
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
        latent_gate = F.interpolate(texture_gate, size=velocity.shape[-2:], mode="nearest")
        latent_mask = latent_mask * latent_gate
        target_amplitude = robust_rms(texture, valid) * texture_gate
        richness = F.interpolate(
            target_amplitude.mean(dim=1, keepdim=True), size=velocity.shape[-2:], mode="area"
        )
        richness = richness / richness.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        latent_weight = (
            residual_weights[:, None, None, None]
            * latent_mask
            * (0.25 + richness.clamp_max(2.0))
        )
        velocity_loss = masked_mean(
            F.smooth_l1_loss(velocity, target_velocity, reduction="none"), latent_weight
        )
        predicted_latent = interpolation + (1 - time_values[:, None, None, None]) * velocity
        endpoint = highpass(self.model.codec.decode(predicted_latent, target_spec.modality))
        endpoint_gate = F.interpolate(texture_gate, size=valid.shape[-2:], mode="nearest")
        # Near-target interpolants leak endpoint information. Preserve velocity coverage at
        # every t, but make endpoint supervision concentrate on source-like low-t states.
        endpoint_weights = residual_weights * (1.0 - time_values).clamp_min(0.05)
        endpoint_loss, endpoint_metrics = high_frequency_loss(
            endpoint,
            texture,
            valid * endpoint_gate,
            target_spec.modality,
            sample_weight=endpoint_weights,
        )
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
        metrics = {
            "velocity": velocity_loss.detach(),
            "amplitude": amplitude_loss.detach(),
            "texture_reliability": texture_reliability.mean().detach(),
            "texture_coverage": texture_gate.mean().detach(),
            "endpoint_source_weight": (1.0 - time_values).mean().detach(),
            **{f"endpoint_{name}": value for name, value in endpoint_metrics.items()},
        }
        if target_spec.modality == "optical":
            shaped_endpoint = self.model.shape_residual_texture(
                endpoint, pyramid, target_spec, amplitude=predicted_amplitude
            )
            composed_endpoint = self.model.compose_visual(
                base, detail, shaped_endpoint, "optical"
            )
            assert isinstance(composed_endpoint, Tensor)
            if teacher_visual_enabled:
                visual_weight = valid * endpoint_weights[:, None, None, None]
                visual_pixel = masked_mean(
                    charbonnier((composed_endpoint - target) / 0.05), visual_weight
                )
                visual_hf_loss, visual_hf_metrics = high_frequency_loss(
                    highpass(composed_endpoint),
                    highpass(target),
                    valid,
                    "optical",
                    sample_weight=endpoint_weights,
                )
                total = (
                    total
                    + self.flow_visual_pixel_weight * visual_pixel
                    + self.flow_visual_hf_weight * visual_hf_loss
                )
                metrics["composed_pixel"] = visual_pixel.detach()
                metrics.update(
                    {f"composed_{name}": value for name, value in visual_hf_metrics.items()}
                )
            if rollout_enabled and self.current_step % self.flow_rollout_every == 0:
                count = min(self.flow_rollout_samples, noise.shape[0])
                rollout_pyramid = tuple(level[:count] for level in pyramid)
                rollout_latent = self.model.integrate_flow(
                    noise[:count],
                    rollout_pyramid,
                    target_spec,
                    texture.shape[1],
                    steps=self.flow_rollout_steps,
                )
                rollout_residual = self.model.codec.decode(rollout_latent, target_spec.modality)
                rollout_texture = self.model.shape_residual_texture(
                    rollout_residual,
                    rollout_pyramid,
                    target_spec,
                    amplitude=predicted_amplitude[:count],
                )
                rollout_visual = self.model.compose_visual(
                    base[:count], detail[:count], rollout_texture, "optical"
                )
                assert isinstance(rollout_visual, Tensor)
                rollout_valid = valid[:count]
                rollout_weights = residual_weights[:count]
                rollout_pixel = masked_mean(
                    charbonnier((rollout_visual - target[:count]) / 0.05),
                    rollout_valid * rollout_weights[:, None, None, None],
                )
                rollout_hf, rollout_hf_metrics = high_frequency_loss(
                    highpass(rollout_visual),
                    highpass(target[:count]),
                    rollout_valid,
                    "optical",
                    sample_weight=rollout_weights,
                )
                rollout_lpips, rollout_dists = self._optical_visual_perceptual(
                    rollout_visual, target[:count], rollout_valid, rollout_weights
                )
                schedule = float(self.flow_rollout_every)
                total = total + schedule * (
                    self.flow_rollout_pixel_weight * rollout_pixel
                    + self.flow_rollout_hf_weight * rollout_hf
                    + self.flow_visual_perceptual_weight * (rollout_lpips + rollout_dists)
                )
                metrics["rollout_pixel"] = rollout_pixel.detach()
                metrics.update(
                    {f"rollout_{name}": value for name, value in rollout_hf_metrics.items()}
                )
                metrics["rollout_lpips"] = rollout_lpips.detach()
                metrics["rollout_dists"] = rollout_dists.detach()
        return total, metrics

    def _risk_direction(
        self,
        batch: dict[str, object],
        indices: Tensor,
        source_key: str,
        target_key: str,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        weights: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if target_spec.modality != "optical":
            raise ValueError("texture risk training is defined for Optical release only")
        target, base, valid, pyramid = self._physical_context(
            batch,
            indices,
            source_key,
            target_key,
            source_spec,
            target_spec,
            joint=False,
        )
        with torch.no_grad():
            latent_size = (target.shape[-2] // 4, target.shape[-1] // 4)
            noise = self.model.flow_noise_scale(target_spec) * torch.randn(
                target.shape[0],
                self.model.config.codec_latent_channels,
                *latent_size,
                device=target.device,
                dtype=target.dtype,
            )
            latent = self.model.integrate_flow(
                noise,
                pyramid,
                target_spec,
                target.shape[1],
                steps=self.risk_flow_steps,
            )
            raw = self.model.codec.decode(latent, target_spec.modality)
            amplitude = self.model.residual_amplitude(
                pyramid, target_spec, target.shape[1], tuple(target.shape[-2:])
            )
            texture = self.model.shape_residual_texture(
                raw,
                pyramid,
                target_spec,
                amplitude=amplitude,
                apply_release_gate=False,
            )
            candidate = self.model.compose_visual(
                base, torch.zeros_like(base), texture, "optical"
            )
            assert isinstance(candidate, Tensor)
            correctness, benefit, block_valid = texture_benefit_target(
                base, candidate, target, valid
            )
        descriptors = self.model.descriptors(
            target_spec.channels[: target.shape[1]], target.device
        )
        logits = self.model.residual_dit.predict_texture_risk_logits(
            pyramid, descriptors, texture
        )
        probability = torch.sigmoid(logits.float())
        hard_positive = (correctness > 0.5).to(probability.dtype)
        positive_fraction = masked_mean(hard_positive, block_valid).detach()
        positive_weight = ((1.0 - positive_fraction) / positive_fraction.clamp_min(0.02)).clamp(
            1.0, 12.0
        )
        class_weight = torch.where(hard_positive.bool(), positive_weight, 1.0)
        evidence = (2.0 * (correctness - 0.5).abs()).clamp_min(0.10)
        sample_weight = (
            weights[indices, None, None, None] * block_valid * class_weight * evidence
        )
        cross_entropy = F.binary_cross_entropy_with_logits(
            logits.float(), correctness.float(), reduction="none"
        )
        classification = masked_mean(cross_entropy, sample_weight)
        calibration = masked_mean(
            (probability - correctness.float()).square(),
            weights[indices, None, None, None] * block_valid,
        )
        total = classification + 0.25 * calibration
        released = probability >= self.model.optical_texture_risk_threshold
        released_benefit = masked_mean(
            benefit,
            block_valid * released.to(block_valid.dtype),
        )
        return total, {
            "classification": classification.detach(),
            "calibration": calibration.detach(),
            "positive_fraction": positive_fraction,
            "target_probability": masked_mean(correctness, block_valid).detach(),
            "predicted_probability": masked_mean(probability, block_valid).detach(),
            "release_fraction": masked_mean(
                released.to(block_valid.dtype), block_valid
            ).detach(),
            "released_benefit": released_benefit.detach(),
            "candidate_benefit": masked_mean(benefit, block_valid).detach(),
        }

    def _bridge_direction(
        self,
        batch: dict[str, object],
        indices: Tensor,
        source_key: str,
        target_key: str,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        weights: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if target_spec.modality != "optical":
            raise ValueError("residual bridge training is defined for Optical only")
        target, base, valid, pyramid = self._physical_context(
            batch,
            indices,
            source_key,
            target_key,
            source_spec,
            target_spec,
            joint=False,
        )
        with torch.no_grad():
            detail = self.model.deterministic_detail(
                pyramid,
                source_spec,
                target_spec,
                tuple(target.shape[-2:]),
                base=base,
            )
            latent_size = (target.shape[-2] // 4, target.shape[-1] // 4)
            latent_gate = self.model.optical_bridge_gate(detail, latent_size)
            full_gate = F.interpolate(
                latent_gate, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
            texture = self._texture_target(target, base, detail, valid) * full_gate
            endpoint_latent = self.model.codec.encode(texture, "optical")
            anchor_latent = self.model.codec.encode(highpass(detail), "optical")
        noise = self.model.flow_noise_scale(target_spec) * torch.randn_like(endpoint_latent)
        noise = noise * latent_gate
        time_values = torch.rand(texture.shape[0], device=texture.device, dtype=texture.dtype)
        interpolation = (1.0 - time_values[:, None, None, None]) * noise + time_values[
            :, None, None, None
        ] * endpoint_latent
        velocity = self.model.flow_velocity(
            interpolation,
            time_values,
            pyramid,
            target_spec,
            texture.shape[1],
            bridge_anchor=anchor_latent,
            use_optical_bridge=True,
        )
        target_velocity = endpoint_latent - noise
        residual_weights = weights[indices]
        latent_valid = F.interpolate(valid, size=latent_size, mode="area") * latent_gate
        velocity_loss = masked_mean(
            F.smooth_l1_loss(velocity, target_velocity, reduction="none"),
            residual_weights[:, None, None, None] * latent_valid,
        )
        predicted_latent = interpolation + (
            1.0 - time_values[:, None, None, None]
        ) * velocity
        endpoint = highpass(self.model.codec.decode(predicted_latent, "optical")) * full_gate
        endpoint_weights = residual_weights * (1.0 - time_values).clamp_min(0.05)
        endpoint_loss, endpoint_metrics = high_frequency_loss(
            endpoint,
            texture,
            valid * full_gate,
            "optical",
            sample_weight=endpoint_weights,
        )
        target_amplitude = robust_rms(texture, valid).clamp_max(
            self.model.config.optical_residual_limit
        )
        predicted_amplitude = self.model.residual_amplitude(
            pyramid, target_spec, texture.shape[1], tuple(texture.shape[-2:])
        )
        amplitude_loss = masked_mean(
            (
                (predicted_amplitude - target_amplitude)
                / self.model.config.optical_residual_limit
            ).abs(),
            residual_weights[:, None, None, None]
            * F.avg_pool2d(valid, 4, stride=4)
            * latent_gate,
        )
        shaped = self.model.shape_residual_texture(
            endpoint,
            pyramid,
            target_spec,
            amplitude=predicted_amplitude,
            apply_release_gate=False,
        ) * full_gate
        composed = self.model.compose_visual(base, detail, shaped, "optical")
        assert isinstance(composed, Tensor)
        visual_weight = valid * endpoint_weights[:, None, None, None]
        visual_pixel = masked_mean(
            charbonnier((composed - target) / 0.05), visual_weight
        )
        visual_hf, visual_hf_metrics = high_frequency_loss(
            highpass(composed),
            highpass(target),
            valid,
            "optical",
            sample_weight=endpoint_weights,
        )
        total = (
            velocity_loss
            + 0.25 * endpoint_loss
            + 0.20 * amplitude_loss
            + 0.10 * visual_pixel
            + 0.10 * visual_hf
        )
        metrics = {
            "velocity": velocity_loss.detach(),
            "amplitude": amplitude_loss.detach(),
            "visual_pixel": visual_pixel.detach(),
            "bridge_coverage": latent_gate.mean().detach(),
            "endpoint_source_weight": (1.0 - time_values).mean().detach(),
            **{f"endpoint_{name}": value for name, value in endpoint_metrics.items()},
            **{f"visual_{name}": value for name, value in visual_hf_metrics.items()},
        }
        if (
            self.flow_visual_perceptual_weight > 0.0
            and self.current_step % self.flow_rollout_every == 0
        ):
            lpips, dists = self._optical_visual_perceptual(
                composed, target, valid, endpoint_weights
            )
            total = total + self.flow_rollout_every * self.flow_visual_perceptual_weight * (
                lpips + dists
            )
            metrics["visual_lpips"] = lpips.detach()
            metrics["visual_dists"] = dists.detach()
        return total, metrics

    def _id_bridge_direction(
        self,
        batch: dict[str, object],
        indices: Tensor,
        source_key: str,
        target_key: str,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        weights: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        target, base, valid, pyramid = self._physical_context(
            batch,
            indices,
            source_key,
            target_key,
            source_spec,
            target_spec,
            joint=False,
        )
        delta = (target - base.detach()) * valid
        mu, correction, anchor_detail, log_sigma, reliability_logits = (
            self.model.predict_id_bridge_origin_components(pyramid, base, target_spec)
        )
        utility_optical = (
            self.model.config.id_bridge_anchor_utility
            and self.model.id_bridge_uses_observable_anchor
            and target_spec.modality == "optical"
        )
        phase_optical = (
            self.model.id_bridge_uses_phase_identifiability
            and target_spec.modality == "optical"
        )
        if phase_optical and self.model.config.id_bridge_anchor_utility:
            raise ValueError("phase id bridge requires an ungated protected Optical anchor")
        if self.model.id_bridge_uses_observable_anchor and target_spec.modality == "optical":
            # Keep the observable prior in pixels; the Haar state transports only its
            # orthogonal innovation and therefore cannot project away anchor detail.
            full_residual = (
                highpass(target - base.detach()) * valid
                if phase_optical
                else highpass(delta) * valid
            )
            innovation_anchor = (
                anchor_detail.detach() if utility_optical or phase_optical else anchor_detail
            )
            innovation_target = self.model.project_id_bridge_residual(
                (full_residual - innovation_anchor) * valid, target_spec
            ) * valid
        else:
            # This bridge owns all high-frequency residuals, including SAR speckle and tails.
            full_residual = self.model.project_id_bridge_residual(delta, target_spec) * valid
            innovation_target = full_residual
        with torch.no_grad():
            endpoint_latent = self.model.encode_id_bridge_residual(innovation_target, target_spec)
        if phase_optical:
            oracle = phase_identifiability_target(
                batch[source_key][indices],  # type: ignore[index]
                frequency_bands(innovation_target, levels=3),
                valid,
            ).detach()
        elif utility_optical:
            oracle = anchor_gain_target(
                self.model.id_bridge_anchor_components(pyramid, base, target_spec),
                full_residual,
                valid,
            ).detach()
        else:
            oracle = cross_modal_identifiability_target(
                pyramid[0].detach(), frequency_bands(full_residual, levels=3), valid
            ).detach()
        if phase_optical:
            q_oracle = self.model.id_bridge_band_fields_to_state(oracle, target_spec)
            q_pred_state = self.model.id_bridge_q_state(reliability_logits, target_spec).detach()
            transport_field = self.model.id_bridge_transport_field(
                reliability_logits, log_sigma, detach=True
            )
            with torch.no_grad():
                anchor_state = self.model.id_bridge_anchor_state(anchor_detail.detach(), target_spec)
        else:
            q_oracle = oracle.mean(dim=1, keepdim=True)
            q_pred_state = None
            transport_field = None
            anchor_state = None
        epsilon = torch.randn_like(mu)
        z0, q_pred, sigma = self._id_bridge_start(
            mu,
            log_sigma,
            reliability_logits,
            self.model.flow_noise_scale(target_spec),
            epsilon,
            q_state=q_pred_state,
        )
        time_values = torch.rand(
            innovation_target.shape[0],
            device=innovation_target.device,
            dtype=innovation_target.dtype,
        )
        time = time_values[:, None, None, None]
        zt = (1.0 - time) * z0 + time * endpoint_latent
        velocity = self.model.flow_velocity(
            zt,
            time_values,
            pyramid,
            target_spec,
            innovation_target.shape[1],
            origin_latent=mu,
            transport_field=transport_field,
            id_bridge_anchor_state=anchor_state,
            use_optical_bridge=False,
        )
        residual_weights = weights[indices]
        time_weight = (1.0 - time_values).clamp_min(0.05)
        latent_valid = F.interpolate(valid, size=mu.shape[-2:], mode="area")
        latent_weight = (
            residual_weights[:, None, None, None]
            * time_weight[:, None, None, None]
            * latent_valid
        )
        velocity_loss = masked_mean(charbonnier(velocity - (endpoint_latent - z0)), latent_weight)
        anchor_values = self._id_bridge_anchor_values(
            mu, correction, endpoint_latent, q_oracle
        )
        anchor_loss = masked_mean(anchor_values, latent_weight)
        reliability_loss = masked_mean(
            F.binary_cross_entropy_with_logits(
                reliability_logits.float(), oracle.float(), reduction="none"
            ),
            latent_weight.float(),
        )
        innovation = (endpoint_latent - mu.detach()).abs()
        target_fraction = (
            innovation / max(self.model.flow_noise_scale(target_spec), 1e-3)
        ).clamp(0.0, 1.0)
        sigma_weight = latent_weight * (1.0 - q_oracle)
        sigma_loss = masked_mean(
            F.smooth_l1_loss(torch.sigmoid(log_sigma), target_fraction, reduction="none"),
            sigma_weight,
        )
        predicted_latent = zt + (1.0 - time) * velocity
        endpoint = self.model.decode_id_bridge_residual(predicted_latent, target_spec)
        endpoint_weights = residual_weights * time_weight
        endpoint_loss, endpoint_metrics = high_frequency_loss(
            endpoint,
            innovation_target,
            valid,
            target_spec.modality,
            sample_weight=endpoint_weights,
        )
        total = (
            velocity_loss
            + 0.25 * endpoint_loss
            + 0.20 * anchor_loss
            + 0.10 * reliability_loss
            + 0.25 * sigma_loss
        )
        metrics = {
            "velocity": velocity_loss.detach(),
            "endpoint_source_weight": time_weight.mean().detach(),
            "anchor": anchor_loss.detach(),
            "reliability": reliability_loss.detach(),
            "sigma_calibration": sigma_loss.detach(),
            "q": q_pred.mean().detach(),
            "sigma": sigma.mean().detach(),
            **{f"endpoint_{name}": value for name, value in endpoint_metrics.items()},
        }
        if phase_optical:
            q_bands = torch.sigmoid(reliability_logits)
            release_bands = self.model.id_bridge_innovation_release_bands(
                reliability_logits, target_spec
            ) * (1.0 - q_bands)
            diagnostic_q = q_bands.detach().float()
            diagnostic_oracle = oracle.detach().float()
            valid_blocks = latent_valid.detach() > 0.999

            def q_statistics(predicted: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
                finite_blocks = valid_blocks & torch.isfinite(predicted) & torch.isfinite(target)
                if not bool(finite_blocks.any()):
                    zero = predicted.new_zeros(())
                    return zero, zero
                predicted_values = predicted[finite_blocks]
                target_values = target[finite_blocks]
                mae = (predicted_values - target_values).abs().mean()
                predicted_centered = predicted_values - predicted_values.mean()
                target_centered = target_values - target_values.mean()
                denominator = torch.sqrt(
                    predicted_centered.square().mean() * target_centered.square().mean()
                )
                if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-12:
                    return mae, predicted.new_zeros(())
                correlation = (predicted_centered * target_centered).mean() / denominator
                return mae, torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)

            q_statistics_per_band = tuple(
                q_statistics(
                    diagnostic_q[:, band_index : band_index + 1],
                    diagnostic_oracle[:, band_index : band_index + 1],
                )
                for band_index in range(3)
            )
            q_mae = torch.stack(tuple(values[0] for values in q_statistics_per_band)).mean()
            q_corr_bands = torch.stack(tuple(values[1] for values in q_statistics_per_band))
            q_corr = q_corr_bands.mean()
            metrics.update(
                {
                    "q_fine": q_bands[:, 0].mean().detach(),
                    "q_mid": q_bands[:, 1].mean().detach(),
                    "q_coarse": q_bands[:, 2].mean().detach(),
                    "oracle_q_fine": diagnostic_oracle[:, 0].mean(),
                    "oracle_q_mid": diagnostic_oracle[:, 1].mean(),
                    "oracle_q_coarse": diagnostic_oracle[:, 2].mean(),
                    "q_mae": q_mae,
                    "q_corr": q_corr,
                    "q_corr_fine": q_corr_bands[0],
                    "q_corr_mid": q_corr_bands[1],
                    "q_corr_coarse": q_corr_bands[2],
                    "release_fine": release_bands[:, 0].mean().detach(),
                    "release_mid": release_bands[:, 1].mean().detach(),
                    "release_coarse": release_bands[:, 2].mean().detach(),
                }
            )
        elif utility_optical:
            anchor_gains = self.model.id_bridge_anchor_gains(reliability_logits)
            metrics.update(
                {
                    "anchor_gain": anchor_gains.mean().detach(),
                    "anchor_gain_fine": anchor_gains[:, 0].mean().detach(),
                    "anchor_gain_mid": anchor_gains[:, 1].mean().detach(),
                    "anchor_gain_coarse": anchor_gains[:, 2].mean().detach(),
                }
            )
        if self.model.id_bridge_uses_observable_anchor and target_spec.modality == "optical":
            origin_residual = anchor_detail + self.model.decode_id_bridge_residual(mu, target_spec)
            origin_hf_loss, origin_hf_metrics = high_frequency_loss(
                origin_residual,
                full_residual,
                valid,
                "optical",
                sample_weight=residual_weights,
            )
            total = total + 0.25 * origin_hf_loss
            metrics.update({f"origin_{name}": value for name, value in origin_hf_metrics.items()})
        if self.current_step % self.flow_rollout_every == 0:
            rollout_latent = self.model.integrate_flow(
                z0,
                pyramid,
                target_spec,
                innovation_target.shape[1],
                steps=self.flow_rollout_steps,
                origin_latent=mu,
                transport_field=transport_field,
                id_bridge_anchor_state=anchor_state,
                use_optical_bridge=False,
            )
            if phase_optical and self.id_bridge_antithetic_weight > 0.0:
                z0_minus = mu - sigma.detach() * epsilon
                rollout_minus = self.model.integrate_flow(
                    z0_minus,
                    pyramid,
                    target_spec,
                    innovation_target.shape[1],
                    steps=self.flow_rollout_steps,
                    origin_latent=mu,
                    transport_field=transport_field,
                    id_bridge_anchor_state=anchor_state,
                    use_optical_bridge=False,
                )
                rollout_weight = residual_weights[:, None, None, None] * latent_valid
                antithetic_center = masked_mean(
                    charbonnier(0.5 * (rollout_latent + rollout_minus) - mu), rollout_weight
                )
                total = total + self.id_bridge_antithetic_weight * antithetic_center
                metrics["antithetic_center"] = antithetic_center.detach()
            rollout_latent = self.model.gate_id_bridge_innovation(
                rollout_latent,
                mu,
                reliability_logits,
                target_spec,
                q_state=q_pred_state,
            )
            rollout_innovation = self.model.decode_id_bridge_residual(rollout_latent, target_spec)
            rollout_visual = self.model.compose_visual(
                base, anchor_detail, rollout_innovation, target_spec.modality
            )
            assert isinstance(rollout_visual, Tensor)
            expanded_valid = valid.expand_as(rollout_visual)
            denominator = expanded_valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
            rollout_rmse = torch.sqrt(
                ((rollout_visual - target).square() * expanded_valid).sum(dim=(1, 2, 3))
                / denominator
            )
            base_rmse = torch.sqrt(
                ((base - target).square() * expanded_valid).sum(dim=(1, 2, 3)) / denominator
            )
            distortion = torch.relu(rollout_rmse / (base_rmse + 1e-6) - 1.05)
            distortion_loss = (distortion * residual_weights).sum() / residual_weights.sum().clamp_min(
                1e-8
            )
            total = total + 0.25 * distortion_loss
            metrics["rollout_distortion_hinge"] = distortion_loss.detach()
            if target_spec.modality == "optical":
                if self.flow_visual_perceptual_weight > 0.0:
                    perceptual_weights = residual_weights
                    if phase_optical:
                        perceptual_weights = perceptual_weights * oracle.mean(dim=(1, 2, 3))
                    rollout_lpips, rollout_dists = self._optical_visual_perceptual(
                        rollout_visual, target, valid, perceptual_weights
                    )
                    total = total + self.flow_rollout_every * self.flow_visual_perceptual_weight * (
                        rollout_lpips + rollout_dists
                    )
                    metrics["rollout_lpips"] = rollout_lpips.detach()
                    metrics["rollout_dists"] = rollout_dists.detach()
            else:
                rollout_hf, rollout_hf_metrics = high_frequency_loss(
                    highpass(rollout_visual),
                    highpass(target),
                    valid,
                    "sar",
                    sample_weight=residual_weights,
                )
                total = total + self.flow_rollout_hf_weight * rollout_hf
                metrics.update(
                    {f"rollout_{name}": value for name, value in rollout_hf_metrics.items()}
                )
        return total, metrics

    def _id_utility_direction(
        self,
        batch: dict[str, object],
        indices: Tensor,
        source_key: str,
        target_key: str,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        weights: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if source_spec.modality != "sar" or target_spec.modality != "optical":
            raise ValueError("id_utility only supports SAR-to-Optical training")
        if not self.model.config.id_bridge_anchor_utility:
            raise ValueError("id_utility requires id_bridge_anchor_utility")
        target, base, valid, pyramid = self._physical_context(
            batch,
            indices,
            source_key,
            target_key,
            source_spec,
            target_spec,
            joint=False,
        )
        full_residual = highpass(target - base.detach()) * valid
        _, _, anchor_detail, _, reliability_logits = (
            self.model.predict_id_bridge_origin_components(pyramid, base, target_spec)
        )
        oracle = anchor_gain_target(
            self.model.id_bridge_anchor_components(pyramid, base, target_spec),
            full_residual,
            valid,
        ).detach()
        residual_weights = weights[indices]
        latent_valid = F.interpolate(valid, size=reliability_logits.shape[-2:], mode="area")
        latent_weight = residual_weights[:, None, None, None] * latent_valid
        reliability_loss = masked_mean(
            F.binary_cross_entropy_with_logits(
                reliability_logits.float(), oracle.float(), reduction="none"
            ),
            latent_weight.float(),
        )
        origin_hf_loss, origin_hf_metrics = high_frequency_loss(
            anchor_detail,
            full_residual,
            valid,
            "optical",
            sample_weight=residual_weights,
        )
        composed = self.model.compose_visual(
            base, anchor_detail, torch.zeros_like(anchor_detail), "optical"
        )
        assert isinstance(composed, Tensor)
        expanded_valid = valid.expand_as(composed)
        denominator = expanded_valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
        visual_rmse = torch.sqrt(
            ((composed - target).square() * expanded_valid).sum(dim=(1, 2, 3))
            / denominator
        )
        base_rmse = torch.sqrt(
            ((base - target).square() * expanded_valid).sum(dim=(1, 2, 3)) / denominator
        )
        distortion = torch.relu(visual_rmse / (base_rmse + 1e-6) - 1.05)
        distortion_loss = (distortion * residual_weights).sum() / residual_weights.sum().clamp_min(
            1e-8
        )
        total = reliability_loss + 0.25 * origin_hf_loss + 0.25 * distortion_loss
        q = torch.sigmoid(reliability_logits).mean(dim=1, keepdim=True)
        anchor_gains = self.model.id_bridge_anchor_gains(reliability_logits)
        metrics = {
            "reliability": reliability_loss.detach(),
            "q": q.mean().detach(),
            "oracle_q": oracle.mean().detach(),
            "anchor_gain": anchor_gains.mean().detach(),
            "anchor_gain_fine": anchor_gains[:, 0].mean().detach(),
            "anchor_gain_mid": anchor_gains[:, 1].mean().detach(),
            "anchor_gain_coarse": anchor_gains[:, 2].mean().detach(),
            "distortion_hinge": distortion_loss.detach(),
            **{f"origin_{name}": value for name, value in origin_hf_metrics.items()},
        }
        if (
            self.flow_visual_perceptual_weight > 0.0
            and self.current_step % self.flow_rollout_every == 0
        ):
            lpips, dists = self._optical_visual_perceptual(
                composed, target, valid, residual_weights
            )
            total = total + self.flow_rollout_every * self.flow_visual_perceptual_weight * (
                lpips + dists
            )
            metrics["lpips"] = lpips.detach()
            metrics["dists"] = dists.detach()
        return total, metrics

    def _phase_transport_direction(
        self,
        batch: dict[str, object],
        indices: Tensor,
        source_key: str,
        target_key: str,
        source_spec: SensorSpec,
        target_spec: SensorSpec,
        weights: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Train only the observable Optical pixel-detail transport head."""

        if source_spec.modality != "sar" or target_spec.modality != "optical":
            raise ValueError("phase_transport only supports SAR-to-Optical training")
        if not self.model.config.phase_transport_enabled:
            raise ValueError("phase_transport requires phase_transport_enabled")
        target, base, valid, pyramid = self._physical_context(
            batch,
            indices,
            source_key,
            target_key,
            source_spec,
            target_spec,
            joint=False,
        )
        full_residual = highpass(target - base.detach()) * valid
        protected_anchor = self.model.id_bridge_anchor_detail(pyramid, base, target_spec).detach()
        residual_after_anchor = full_residual - protected_anchor
        phase_delta, diagnostics = self.model.phase_transport_delta(pyramid, base, target_spec)
        detail = protected_anchor + phase_delta
        residual_weights = weights[indices]
        hf_loss, hf_metrics = high_frequency_loss(
            detail,
            full_residual,
            valid,
            "optical",
            sample_weight=residual_weights,
        )
        visual = self.model.compose_visual(
            base, detail, torch.zeros_like(detail), "optical"
        )
        assert isinstance(visual, Tensor)
        expanded_valid = valid.expand_as(visual)
        denominator = expanded_valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
        visual_rmse = torch.sqrt(
            ((visual - target).square() * expanded_valid).sum(dim=(1, 2, 3)) / denominator
        )
        base_rmse = torch.sqrt(
            ((base - target).square() * expanded_valid).sum(dim=(1, 2, 3)) / denominator
        )
        distortion = torch.relu(visual_rmse / (base_rmse + 1e-6) - 1.05)
        rmse_hinge = (distortion * residual_weights).sum() / residual_weights.sum().clamp_min(
            1e-8
        )
        gains = diagnostics["gain"]
        gate = diagnostics["gate"]
        coherence = diagnostics["coherence"]
        null_calibrated = self.model.config.phase_transport_null_calibrated
        latent_valid = F.interpolate(valid, size=gains.shape[-2:], mode="area")
        latent_weight = residual_weights[:, None, None, None] * latent_valid
        physical_bands = torch.stack(frequency_bands(base.detach(), levels=3), dim=1)
        oracle_gate = phase_transport_gain_target(
            physical_bands,
            residual_after_anchor,
            valid,
            self.model.config.phase_transport_gain_caps,
        ).detach()
        physical_energy = F.avg_pool2d(
            physical_bands.float().square().sum(dim=2), 4, stride=4
        )
        cap_values = physical_energy.new_tensor(self.model.config.phase_transport_gain_caps).view(
            1, 3, 1, 1
        )
        oracle_supported = (
            (physical_energy > 1e-7)
            & (latent_valid.expand_as(physical_energy) >= 0.999)
            & (cap_values > 0.0)
        )
        oracle_weight = residual_weights[:, None, None, None] * oracle_supported.float()
        oracle_active_fraction = (
            ((oracle_gate > 0.0).to(oracle_weight) * oracle_weight).sum()
            / oracle_weight.sum().clamp_min(1e-8)
        )
        latent_block_weight = residual_weights[:, None, None, None] * latent_valid.expand_as(
            physical_energy
        )
        oracle_supported_fraction = (
            (oracle_supported.to(latent_block_weight) * latent_block_weight).sum()
            / latent_block_weight.sum().clamp_min(1e-8)
        )
        if null_calibrated:
            gain_support = diagnostics["gain_support"]
            effective_gate = diagnostics["effective_gate"]
            strict_valid = (latent_valid >= 0.999).to(gain_support)
            support_active = (gain_support.detach() > 0.0).to(gain_support)
            utility_weight = (
                residual_weights[:, None, None, None] * strict_valid * support_active
            )
            gain_utility = masked_mean(
                F.smooth_l1_loss(
                    effective_gate.float(), oracle_gate.float(), beta=0.1, reduction="none"
                ),
                utility_weight.float(),
            )
            support_weight = residual_weights[:, None, None, None] * strict_valid
            support_active_fraction = (
                (support_active * support_weight).sum()
                / support_weight.expand_as(support_active).sum().clamp_min(1e-8)
            )
        else:
            offsets = diagnostics["offset_px"]
            utility_weight = latent_weight * coherence.detach()
            gain_utility = masked_mean(
                F.smooth_l1_loss(
                    gate.float(), oracle_gate.float(), beta=0.1, reduction="none"
                ),
                utility_weight.float(),
            )
        gain_l1 = masked_mean(gains.abs(), latent_weight)
        phase_alignment = phase_alignment_loss(
            diagnostics["source_phase"],
            frequency_bands(target, levels=3),
            valid,
            residual_weights,
        )
        low_frequency_leakage = low_frequency_loss(phase_delta, valid, residual_weights)
        if null_calibrated:
            total = (
                self.phase_transport_hf_weight * hf_loss
                + 0.25 * rmse_hinge
                + 0.005 * gain_l1
                + 0.05 * phase_alignment
                + self.phase_transport_utility_weight * gain_utility
            )
        else:
            offset_magnitude = masked_mean(offsets.abs(), latent_weight)
            offset_tv = masked_mean(offsets.diff(dim=-2).abs(), latent_weight[..., 1:, :])
            offset_tv = offset_tv + masked_mean(
                offsets.diff(dim=-1).abs(), latent_weight[..., :, 1:]
            )
            offset_regularizer = offset_magnitude + offset_tv
            total = (
                self.phase_transport_hf_weight * hf_loss
                + 0.25 * rmse_hinge
                + 0.01 * offset_regularizer
                + 0.005 * gain_l1
                + 0.05 * phase_alignment
                + self.phase_transport_utility_weight * gain_utility
            )
        metrics = {
            **hf_metrics,
            "rmse_hinge": rmse_hinge.detach(),
            "gain_l1": gain_l1.detach(),
            "gain_utility": gain_utility.detach(),
            "phase_alignment": phase_alignment.detach(),
            "gain_signed_mean": gains.mean().detach(),
            "gain_fine": gains[:, 0].mean().detach(),
            "gain_mid": gains[:, 1].mean().detach(),
            "gain_coarse": gains[:, 2].mean().detach(),
            "gate_fine": gate[:, 0].mean().detach(),
            "gate_mid": gate[:, 1].mean().detach(),
            "gate_coarse": gate[:, 2].mean().detach(),
            "oracle_gate_fine": oracle_gate[:, 0].mean().detach(),
            "oracle_gate_mid": oracle_gate[:, 1].mean().detach(),
            "oracle_gate_coarse": oracle_gate[:, 2].mean().detach(),
            "oracle_active_fraction": oracle_active_fraction.detach(),
            "oracle_supported_fraction": oracle_supported_fraction.detach(),
            "coherence_fine": coherence[:, 0].mean().detach(),
            "coherence_mid": coherence[:, 1].mean().detach(),
            "coherence_coarse": coherence[:, 2].mean().detach(),
            "low_frequency_leakage": low_frequency_leakage.detach(),
        }
        if null_calibrated:
            null_level = diagnostics["null_level"]
            metrics.update(
                {
                    "null_level_fine": null_level[:, 0].mean().detach(),
                    "null_level_mid": null_level[:, 1].mean().detach(),
                    "null_level_coarse": null_level[:, 2].mean().detach(),
                    "gain_support_fine": gain_support[:, 0].mean().detach(),
                    "gain_support_mid": gain_support[:, 1].mean().detach(),
                    "gain_support_coarse": gain_support[:, 2].mean().detach(),
                    "effective_gate_fine": effective_gate[:, 0].mean().detach(),
                    "effective_gate_mid": effective_gate[:, 1].mean().detach(),
                    "effective_gate_coarse": effective_gate[:, 2].mean().detach(),
                    "support_active_fraction": support_active_fraction.detach(),
                }
            )
        else:
            metrics.update(
                {
                    "offset_regularizer": offset_regularizer.detach(),
                    "offset_px_abs_mean": offsets.abs().mean().detach(),
                }
            )
        if self.current_step % self.flow_rollout_every == 0:
            lpips, dists = self._optical_visual_perceptual(
                visual, target, valid, residual_weights
            )
            total = total + self.flow_rollout_every * self.flow_visual_perceptual_weight * (
                lpips + dists
            )
            metrics["lpips"] = lpips.detach()
            metrics["dists"] = dists.detach()
        return total, metrics

    def forward(self, batch: dict[str, object], stage: str) -> tuple[Tensor, dict[str, Tensor]]:
        stage = "flow" if stage == "visual" else stage
        device = batch["s2"].device  # type: ignore[union-attr]
        batch_size = batch["s2"].shape[0]  # type: ignore[union-attr]
        if stage in {"risk", "bridge"}:
            tasks = torch.zeros(batch_size, device=device, dtype=torch.long)
        elif stage == "id_utility":
            tasks = self._id_utility_assignments(batch_size, device)
        elif stage == "phase_transport":
            tasks = self._phase_transport_assignments(batch_size, device)
        elif stage == "id_bridge":
            tasks = self._id_bridge_assignments(batch_size, device)
        else:
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
            if (
                stage == "codec"
                and self.codec_train_modality is not None
                and target_spec.modality != self.codec_train_modality
            ):
                continue
            indices = torch.nonzero(tasks == task_index, as_tuple=False).flatten()
            if stage in {
                "detail",
                "codec",
                "flow",
                "risk",
                "bridge",
                "id_bridge",
                "id_utility",
                "phase_transport",
            }:
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
            elif stage == "risk":
                loss, direction_metrics = self._risk_direction(
                    batch,
                    indices,
                    source_key,
                    target_key,
                    source_spec,
                    target_spec,
                    high_frequency_weights,
                )
            elif stage == "bridge":
                loss, direction_metrics = self._bridge_direction(
                    batch,
                    indices,
                    source_key,
                    target_key,
                    source_spec,
                    target_spec,
                    high_frequency_weights,
                )
            elif stage == "id_bridge":
                loss, direction_metrics = self._id_bridge_direction(
                    batch,
                    indices,
                    source_key,
                    target_key,
                    source_spec,
                    target_spec,
                    high_frequency_weights,
                )
            elif stage == "id_utility":
                loss, direction_metrics = self._id_utility_direction(
                    batch,
                    indices,
                    source_key,
                    target_key,
                    source_spec,
                    target_spec,
                    high_frequency_weights,
                )
            elif stage == "phase_transport":
                loss, direction_metrics = self._phase_transport_direction(
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

        if (
            stage == "physical"
            and batch_size >= 2
            and self.current_step % self.physical_alignment_every == 0
        ):
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
            total = total + self.physical_alignment_weight * self.physical_alignment_every * alignment
            metrics.update(
                {f"latent/{name}": value for name, value in alignment_metrics.items()}
            )

        if stage in {
            "detail",
            "codec",
            "flow",
            "risk",
            "bridge",
            "id_bridge",
            "id_utility",
            "phase_transport",
        } and not bool(high_frequency_weights.any()):
            if stage == "id_bridge":
                total = total + _weighted_zero(self.model.id_bridge_origin, device)
                total = total + _weighted_zero(self.model.residual_dit, device)
            elif stage == "id_utility":
                total = total + _weighted_zero(self.model.id_bridge_origin, device)
            elif stage == "phase_transport":
                total = total + _weighted_zero(self.model.phase_transport_head, device)
            else:
                branch = {
                    "detail": self.model.detail_head,
                    "codec": self.model.codec,
                    "flow": self.model.residual_dit,
                    "risk": self.model.residual_dit,
                    "bridge": self.model.residual_dit,
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
    if stage in {
        "flow",
        "visual",
        "risk",
        "bridge",
        "id_bridge",
        "id_utility",
        "phase_transport",
    }:
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
    candidates = dict(state)
    if isinstance(model, SentinelV3) and model.config.phase_transport_null_calibrated:
        for suffix in ("weight", "bias"):
            name = f"phase_transport_head.output.2.{suffix}"
            value = candidates.get(name)
            target = current.get(name)
            if (
                value is not None
                and target is not None
                and value.ndim == target.ndim
                and value.shape[0] == 9
                and target.shape[0] == 3
                and value.shape[1:] == target.shape[1:]
            ):
                candidates[name] = value[:3].clone()
    if isinstance(model, SentinelV3) and model.legacy_residual_dit is not None:
        for name, value in state.items():
            if not name.startswith("residual_dit."):
                continue
            legacy_name = f"legacy_residual_dit.{name.removeprefix('residual_dit.')}"
            if legacy_name in current and legacy_name not in candidates:
                candidates[legacy_name] = value
    compatible = {
        name: value
        for name, value in candidates.items()
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


def _stage_requires_codec_gate(model: SentinelV3, stage: str) -> bool:
    if stage in {"flow", "risk", "bridge", "balance"}:
        return True
    return stage == "id_bridge" and model.config.id_bridge_state == "codec"


def _stage_requires_physical_gate(stage: str) -> bool:
    return stage in {
        "detail",
        "codec",
        "flow",
        "risk",
        "bridge",
        "id_bridge",
        "id_utility",
        "phase_transport",
        "balance",
    }


def _validate_protocol_binding(
    checkpoint: dict[str, object],
    protocol_hash: str,
    *,
    stage: str,
    resume: bool,
) -> bool:
    """Bind resumed and high-frequency v4 work to the active validation protocol."""

    checkpoint_hash = checkpoint.get("validation_protocol_hash")
    matches = isinstance(checkpoint_hash, str) and checkpoint_hash == protocol_hash
    requires_match = resume or (
        _stage_requires_physical_gate(stage) and int(checkpoint.get("format_version", 0)) == 4
    )
    if requires_match and not matches:
        action = "resume" if resume else "high-frequency initialization"
        raise RuntimeError(
            f"{action} requires validation_protocol_hash matching the active config"
        )
    return matches


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
    elif stage == "risk":
        modules = (
            model.residual_dit.texture_risk_candidate,
            model.residual_dit.texture_risk_head,
        )
    elif stage == "bridge":
        modules = (
            model.residual_dit.optical_bridge_anchor,
            model.residual_dit.optical_bridge_adapters,
            model.residual_dit.optical_bridge_output,
            model.residual_dit.optical_bridge_amplitude_head,
        )
    elif stage == "id_bridge":
        modules = (
            model.id_bridge_origin,
            model.residual_dit.input,
            model.residual_dit.scene_projections,
            model.residual_dit.frequency_adapter,
            model.residual_dit.time,
            model.residual_dit.target,
            model.residual_dit.blocks,
            model.residual_dit.output,
            model.residual_dit.origin_projection,
            model.residual_dit.id_bridge_field_projection,
            model.residual_dit.id_bridge_anchor_projection,
        )
    elif stage == "id_utility":
        modules = (model.id_bridge_origin,)
    elif stage == "phase_transport":
        modules = (model.phase_transport_head,)
    elif stage in {"balance", "overfit"}:
        modules = (model.encoder, model.decoder, model.detail_head, model.residual_dit)
    else:
        raise ValueError(f"unsupported training stage: {stage}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if stage == "bridge":
        model.residual_dit.optical_bridge_base_gate.requires_grad_(True)
    if stage == "id_bridge":
        model.residual_dit.condition_gates.requires_grad_(True)


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
        {
            "name": "dit",
            "params": list(model.residual_dit.parameters())
            + list(model.id_bridge_origin.parameters())
            + list(model.phase_transport_head.parameters()),
            "lr": residual_lr,
        },
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


def _accumulate_pcgrad_corrections(
    accumulated: list[Tensor | None] | None,
    corrections: list[Tensor | None],
) -> list[Tensor | None]:
    """Sum microstep PCGrad deltas after each loss has been accumulation-scaled."""

    if accumulated is None:
        return [None if correction is None else correction.clone() for correction in corrections]
    if len(accumulated) != len(corrections):
        raise ValueError("PCGrad correction lengths must match")
    combined: list[Tensor | None] = []
    for previous, current in zip(accumulated, corrections, strict=True):
        if previous is None:
            combined.append(None if current is None else current.clone())
        elif current is None:
            combined.append(previous)
        else:
            combined.append(previous + current)
    return combined


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
        "residual_state": {
            **model.residual_state_metadata(),
            "antithetic_weight": float(config.get("train", {}).get("id_bridge_antithetic_weight", 0.0)),
        },
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
        timeout_minutes = float(config.get("distributed", {}).get("timeout_minutes", 120.0))
        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(minutes=timeout_minutes),
        )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    train_config = config["train"]
    torch.autograd.set_detect_anomaly(bool(train_config.get("detect_anomaly", False)))
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
            if stage
            in {
                "detail",
                "codec",
                "flow",
                "risk",
                "bridge",
                "id_bridge",
                "id_utility",
                "phase_transport",
                "balance",
            }
            else float(train_config.get("native_gsd_probability", 0.8))
        ),
        audit_high_frequency=stage
        in {
            "detail",
            "codec",
            "flow",
            "risk",
            "bridge",
            "id_bridge",
            "id_utility",
            "phase_transport",
            "balance",
        }
        and bool(train_config.get("registration_audit", True))
        and not bool(config["paths"].get("hf_eligibility")),
        temporal_prior_index=(
            config["paths"].get("temporal_prior_shards")
            if stage
            in {
                "detail",
                "flow",
                "risk",
                "bridge",
                "id_bridge",
                "id_utility",
                "phase_transport",
                "balance",
            }
            else None
        ),
    )
    if limit is not None:
        if stage in {
            "detail",
            "codec",
            "flow",
            "risk",
            "bridge",
            "id_bridge",
            "id_utility",
            "phase_transport",
            "balance",
        }:
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
    high_frequency_stage = stage in {
        "detail",
        "codec",
        "flow",
        "risk",
        "bridge",
        "id_bridge",
        "id_utility",
        "phase_transport",
        "balance",
    }
    eligibility_path = config["paths"].get("hf_eligibility")
    if isinstance(dataset, Subset):
        sampler = None
    elif high_frequency_stage and eligibility_path:
        eligibility_file = Path(str(eligibility_path))
        if not eligibility_file.is_file():
            raise RuntimeError(
                f"high-frequency eligibility sidecar is missing: {eligibility_file}"
            )
        sampler = StatefulIndexSampler(
            _load_high_frequency_eligibility(eligibility_file, config["paths"]["train_shards"]),
            replicas=world_size,
            rank=rank,
            seed=seed,
        )
    else:
        sampler = StatefulShardSampler(
            dataset,
            replicas=world_size,
            rank=rank,
            seed=seed,
            high_frequency_only=high_frequency_stage,
        )
    loader = DataLoader(
        dataset,
        batch_size=int(train_config["batch_size"]),
        sampler=sampler,
        shuffle=isinstance(dataset, Subset),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **_data_loader_worker_options(train_config),
    )
    model = SentinelV3(ModelConfig(**config["model"])).to(device)
    model.encoder.set_activation_checkpointing(
        bool(train_config.get("activation_checkpointing", False))
    )
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    _set_trainable(model, stage)
    if stage == "detail" and bool(train_config.get("detail_confidence_only", False)):
        model.detail_head.requires_grad_(False)
        model.detail_head.confidence_heads.requires_grad_(True)
    if stage == "codec" and train_config.get("codec_train_modality"):
        modality = str(train_config["codec_train_modality"])
        model.codec.requires_grad_(False)
        model.codec.input_heads[modality].requires_grad_(True)
        model.codec.output_heads[modality].requires_grad_(True)
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
        task_probabilities=list(train_config.get("task_probabilities", [0.5, 0.5])),
        physical_alignment_samples=int(train_config.get("physical_alignment_samples", 4)),
        physical_alignment_weight=float(train_config.get("physical_alignment_weight", 0.02)),
        physical_alignment_every=int(train_config.get("physical_alignment_every", 1)),
        optical_dists_weight=float(train_config.get("optical_dists_weight", 0.1)),
        flow_perceptual_every=int(train_config.get("flow_perceptual_every", 8)),
        codec_train_modality=train_config.get("codec_train_modality"),
        codec_perceptual_every=int(train_config.get("codec_perceptual_every", 8)),
        flow_visual_pixel_weight=float(train_config.get("flow_visual_pixel_weight", 0.1)),
        flow_visual_hf_weight=float(train_config.get("flow_visual_hf_weight", 0.1)),
        flow_visual_perceptual_weight=float(
            train_config.get("flow_visual_perceptual_weight", 0.025)
        ),
        flow_rollout_every=int(train_config.get("flow_rollout_every", 4)),
        flow_rollout_steps=int(train_config.get("flow_rollout_steps", 2)),
        flow_rollout_samples=int(train_config.get("flow_rollout_samples", 2)),
        flow_rollout_pixel_weight=float(train_config.get("flow_rollout_pixel_weight", 0.1)),
        flow_rollout_hf_weight=float(train_config.get("flow_rollout_hf_weight", 0.1)),
        id_bridge_antithetic_weight=float(
            train_config.get("id_bridge_antithetic_weight", 0.0)
        ),
        phase_transport_hf_weight=float(train_config.get("phase_transport_hf_weight", 0.05)),
        phase_transport_utility_weight=float(
            train_config.get("phase_transport_utility_weight", 0.10)
        ),
        risk_flow_steps=int(train_config.get("risk_flow_steps", 4)),
        bridge_flow_steps=int(train_config.get("bridge_flow_steps", 4)),
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
        "codec_score": -float("inf"),
        "detail_score": -float("inf"),
    }
    quality_gates: dict[str, bool] = {}
    optimizer_states: dict[str, object] = {}
    scheduler_states: dict[str, object] = {}
    protocol_hash = str(config.get("validation", {}).get("protocol_hash", "unresolved"))

    if init_model:
        initial = torch.load(init_model, map_location="cpu", weights_only=False)
        initial_protocol_matches = _validate_protocol_binding(
            initial, protocol_hash, stage=stage, resume=False
        )
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
        optical_detail_override = train_config.get(
            "optical_detail_confidence_threshold_override"
        )
        if optical_detail_override is not None:
            model.set_detail_confidence_threshold("optical", float(optical_detail_override))
        if (
            _stage_requires_physical_gate(stage)
            and bool(train_config.get("require_physical_gate", True))
            and not bool(initial.get("quality_gates", {}).get("physical", False))
        ):
            raise RuntimeError(
                "high-frequency training requires a frozen physical checkpoint that passed validation"
            )
        if (
            _stage_requires_codec_gate(model, stage)
            and bool(train_config.get("require_codec_gate", True))
            and not bool(initial.get("quality_gates", {}).get("codec", False))
        ):
            raise RuntimeError(
                "flow training requires a frozen codec checkpoint that passed reconstruction gates"
            )
        if (
            stage in {"flow", "risk", "bridge", "balance"}
            and bool(train_config.get("require_detail_gate", True))
            and not bool(initial.get("quality_gates", {}).get("detail", False))
        ):
            raise RuntimeError(
                "flow training requires a deterministic-detail checkpoint that passed its gate"
            )
        if int(initial.get("format_version", 0)) == 4 and (
            stage != "physical" or initial_protocol_matches
        ):
            optimizer_states.update(initial.get("optimizer_states", {}))
            scheduler_states.update(initial.get("scheduler_states", {}))
            best_metrics.update(initial.get("best_metrics", {}))
            quality_gates.update(initial.get("quality_gates", {}))
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
        _validate_protocol_binding(checkpoint, protocol_hash, stage=stage, resume=True)
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
        states = checkpoint["rank_states"]
        state = states[rank] if rank < len(states) else states[0]
        _set_rng_state(state["rng"])
        if sampler is not None and state.get("sampler") is not None:
            sampler.load_state_dict(state["sampler"])
        if channels_last:
            _match_optimizer_layout(optimizer)

    if (
        stage
        in {
            "detail",
            "codec",
            "flow",
            "risk",
            "bridge",
            "id_bridge",
            "id_utility",
            "phase_transport",
            "balance",
        }
        and bool(train_config.get("require_physical_gate", True))
        and not (resume or init_model)
    ):
        raise RuntimeError(
            "high-frequency stages require --init-model with a passing physical v4 checkpoint"
        )
    if (
        _stage_requires_codec_gate(model, stage)
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
            find_unused_parameters=bool(train_config.get("find_unused_parameters", True)),
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
        objective.set_progress(step, int(train_config["max_steps"]))
        aggregate: dict[str, float] = {}
        pcgrad_parameters = (
            _shared_physical_parameters(model)
            if stage == "physical" and bool(train_config.get("pcgrad", True))
            else None
        )
        pcgrad_corrections: list[Tensor | None] | None = None
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
                pcgrad_parameters is not None
            ):
                corrections = _pcgrad_corrections(
                    [value / accumulation for value in objective.last_direction_losses],
                    pcgrad_parameters,
                )
                pcgrad_corrections = _accumulate_pcgrad_corrections(
                    pcgrad_corrections, corrections
                )
            loss.backward()
            for name, value in metrics.items():
                aggregate[name] = aggregate.get(name, 0.0) + float(value) / accumulation
        if pcgrad_parameters is not None and pcgrad_corrections is not None:
            _apply_pcgrad_corrections(pcgrad_parameters, pcgrad_corrections, distributed)
        nonfinite_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad.detach()).all())
        ]
        if nonfinite_gradients:
            raise FloatingPointError(
                "non-finite gradients before clipping: "
                + ", ".join(nonfinite_gradients[:16])
            )
        gradient_norm = 0.0
        gradient_max = 0.0
        for group in optimizer.param_groups:
            group_norm, group_max = _stable_clip_grad_norm_(
                group["params"], float(train_config["gradient_clip"])
            )
            gradient_norm = max(gradient_norm, group_norm)
            gradient_max = max(gradient_max, group_max)
        aggregate["gradient_norm"] = gradient_norm
        aggregate["gradient_max"] = gradient_max
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
                    was_training = model.training
                    model.eval()
                    try:
                        report = _validate_training_state(
                            model, config, stage, step, full=is_full_validation
                        )
                    finally:
                        model.train(was_training)
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
                    if stage in {"codec", "detail"}:
                        component_score = _score(report, stage)
                        component_key = f"{stage}_score"
                        if (
                            bool(report.get("quality_gates", {}).get(stage, False))
                            and component_score > best_metrics[component_key]
                        ):
                            best_metrics[component_key] = component_score
                            improved_kinds.add(stage)
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
