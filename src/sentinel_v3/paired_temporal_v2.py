"""Sparse paired-anchor transport for causal cross-sensor image translation.

The model consumes one registered source/target anchor pair and a variable
number of source-modality observations.  An observation may be acquired at the
query time (image translation) or all observations may precede it (forecasting).
Both cases use the same set encoder and explicit availability mask.

Only changes relative to the registered source anchor are transported onto the
real target anchor.  This separates sensor calibration from temporal change and
makes one-frame, few-frame, and many-frame inputs instances of one contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .sensors import ChannelSpec, SensorSpec
from .temporal_v1 import (
    ConditionalBridgeVelocity,
    DescriptorSensorAdapter,
    DynamicChannelProjection,
    ResidualLatentCodec,
)


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return (values + self.layers(values)) / math.sqrt(2.0)


def _descriptors(
    supplied: Tensor | None,
    sensor: SensorSpec,
    *,
    batch: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    values = (
        torch.tensor(
            [channel.descriptor() for channel in sensor.channels],
            device=device,
            dtype=dtype,
        )
        if supplied is None
        else supplied.to(device=device, dtype=dtype)
    )
    if values.ndim == 2:
        if values.shape != (channels, DescriptorSensorAdapter.descriptor_dim):
            raise ValueError("one eight-value descriptor is required per channel")
        return values.unsqueeze(0).expand(batch, -1, -1)
    if values.shape != (batch, channels, DescriptorSensorAdapter.descriptor_dim):
        raise ValueError("descriptors must have shape Cx8 or BxCx8")
    return values


class PairedObservationFusion(nn.Module):
    """Fuse a variable-size causal set around a registered cross-modal pair."""

    def __init__(
        self,
        width: int,
        heads: int,
        maximum_horizon_days: int,
        translation_max_delta_days: int,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("fusion width must be divisible by attention heads")
        self.maximum_horizon_days = maximum_horizon_days
        self.translation_max_delta_days = translation_max_delta_days
        self.time_embedding = nn.Sequential(
            nn.Linear(6, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.pair = nn.Sequential(
            nn.Conv2d(width * 3, width, 3, padding=1),
            _ResidualBlock(width),
        )
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 3, width, 3, padding=1),
            _ResidualBlock(width),
            _ResidualBlock(width),
        )

    def _time_features(
        self,
        observation_days: Tensor,
        source_anchor_days: Tensor,
        target_anchor_days: Tensor,
        observation_present: Tensor,
    ) -> Tensor:
        scaled = observation_days / float(self.maximum_horizon_days)
        since_anchor = (observation_days - source_anchor_days[:, None]) / float(
            self.maximum_horizon_days
        )
        target_anchor_age = target_anchor_days[:, None] / float(self.maximum_horizon_days)
        phase = 2.0 * math.pi * scaled
        query_present = (
            observation_days.abs() <= float(self.translation_max_delta_days)
        ).to(observation_days)
        return torch.stack(
            (
                scaled,
                since_anchor,
                target_anchor_age.expand_as(observation_days),
                torch.sin(phase),
                torch.cos(phase),
                query_present,
            ),
            dim=-1,
        ) * observation_present[..., None].to(observation_days)

    def forward(
        self,
        observation_features: Tensor,
        observation_valid: Tensor,
        source_anchor_features: Tensor,
        target_anchor_features: Tensor,
        observation_days: Tensor,
        source_anchor_days: Tensor,
        target_anchor_days: Tensor,
        observation_present: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if observation_features.ndim != 5:
            raise ValueError("observation features must have shape BxTxCxHxW")
        batch, frames, channels, height, width = observation_features.shape
        if source_anchor_features.shape != (batch, channels, height, width):
            raise ValueError("source anchor features do not match observations")
        if target_anchor_features.shape != source_anchor_features.shape:
            raise ValueError("target anchor features do not share the latent grid")
        if observation_valid.shape != (batch, frames, 1, height * 4, width * 4):
            raise ValueError("observation valid mask must match the four-times latent grid")
        if observation_days.shape != (batch, frames):
            raise ValueError("observation_days must have shape BxT")
        if observation_present.shape != (batch, frames):
            raise ValueError("observation_present must have shape BxT")
        if source_anchor_days.shape not in {(batch,), (batch, 1)}:
            raise ValueError("source_anchor_days must have shape B or Bx1")
        if target_anchor_days.shape not in {(batch,), (batch, 1)}:
            raise ValueError("target_anchor_days must have shape B or Bx1")

        present = observation_present.bool()
        if bool((present.sum(dim=1) < 1).any()):
            raise ValueError("each sample requires at least one observation")
        active_days = observation_days[present]
        if bool((active_days > 0).any()):
            raise ValueError("future source observations are not causal")
        if bool((active_days < -float(self.maximum_horizon_days)).any()):
            raise ValueError("source observation exceeds the temporal horizon")
        flat_source_anchor_days = source_anchor_days.reshape(batch)
        if bool((flat_source_anchor_days >= 0).any()):
            raise ValueError("the registered source anchor must precede the query")
        if bool((flat_source_anchor_days < -float(self.maximum_horizon_days)).any()):
            raise ValueError("registered source anchor exceeds the temporal horizon")
        flat_target_anchor_days = target_anchor_days.reshape(batch)
        if bool((flat_target_anchor_days >= 0).any()):
            raise ValueError("the registered target anchor must precede the query")
        if bool((flat_target_anchor_days < -float(self.maximum_horizon_days)).any()):
            raise ValueError("registered target anchor exceeds the temporal horizon")

        pair = self.pair(
            torch.cat(
                (
                    source_anchor_features,
                    target_anchor_features,
                    target_anchor_features - source_anchor_features,
                ),
                dim=1,
            )
        )
        change = observation_features - source_anchor_features[:, None]
        temporal = self.time_embedding(
            self._time_features(
                observation_days,
                flat_source_anchor_days,
                target_anchor_days.reshape(batch),
                observation_present,
            )
        )
        tokens = change + temporal[:, :, :, None, None]
        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, frames, channels
        )
        query = pair.permute(0, 2, 3, 1).reshape(batch * height * width, 1, channels)
        coverage = F.avg_pool2d(
            observation_valid.reshape(batch * frames, 1, height * 4, width * 4).float(),
            4,
            stride=4,
        ).reshape(batch, frames, height, width)
        usable = (coverage > 0.0) & present[:, :, None, None]
        padding = (~usable).permute(0, 2, 3, 1).reshape(batch * height * width, frames)
        locally_empty = padding.all(dim=1)
        if bool(locally_empty.any()):
            tokens = tokens.clone()
            padding = padding.clone()
            tokens[locally_empty, 0] = 0.0
            padding[locally_empty, 0] = False
        observed, weights = self.attention(
            query,
            tokens,
            tokens,
            key_padding_mask=padding,
            need_weights=True,
            average_attn_weights=True,
        )
        observed = observed.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        attention = weights.reshape(batch, height, width, frames).permute(0, 3, 1, 2)
        attention = attention[:, :, None]
        support = usable.float().sum(dim=1, keepdim=True) / present.float().sum(
            dim=1, keepdim=True
        )[:, :, None, None].clamp_min(1.0)
        fused = self.fuse(torch.cat((target_anchor_features, pair, observed), dim=1))
        return fused, attention, support


class PairedDeltaCarrier(nn.Module):
    """Transport only observed source changes onto target channel descriptors."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.source_embedding = nn.Sequential(
            nn.Linear(DescriptorSensorAdapter.descriptor_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.refine = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), _ResidualBlock(width))
        self.target = DynamicChannelProjection(width, zero_init=True)

    def forward(
        self,
        observations: Tensor,
        observation_valid: Tensor,
        observation_present: Tensor,
        source_anchor: Tensor,
        source_anchor_valid: Tensor,
        source_descriptors: Tensor,
        target_descriptors: Tensor,
        attention: Tensor,
    ) -> Tensor:
        batch, frames, channels, height, width = observations.shape
        expanded_attention = F.interpolate(
            attention.reshape(batch * frames, 1, *attention.shape[-2:]).float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, 1, height, width).to(observations)
        valid = (
            observation_valid.to(observations)
            * source_anchor_valid[:, None].to(observations)
            * observation_present[:, :, None, None, None].to(observations)
        )
        weights = expanded_attention * valid
        change = observations - source_anchor[:, None]
        aggregate = (change * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
        features = torch.einsum(
            "bchw,bcf->bfhw", aggregate, self.source_embedding(source_descriptors)
        ) / math.sqrt(max(1, channels))
        return self.target(self.refine(features), target_descriptors, (height, width))


class PairedPhysicalHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            _ResidualBlock(width),
            _ResidualBlock(width),
        )
        self.delta = DynamicChannelProjection(width, zero_init=True)
        self.log_variance = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.log_variance.weight)
        nn.init.zeros_(self.log_variance.bias)

    def forward(
        self, fused: Tensor, descriptors: Tensor, output_size: tuple[int, int]
    ) -> tuple[Tensor, Tensor]:
        features = self.body(fused)
        delta = self.delta(features, descriptors, output_size)
        log_variance = F.interpolate(
            self.log_variance(features), output_size, mode="bilinear", align_corners=False
        ).clamp(-8.0, 4.0)
        return delta, log_variance


class PairedDetailHead(nn.Module):
    """Predict source-supported high frequency and its release reliability."""

    def __init__(self, width: int, limit: float) -> None:
        super().__init__()
        self.limit = limit
        self.features = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            _ResidualBlock(width),
            _ResidualBlock(width),
        )
        self.detail = DynamicChannelProjection(width, zero_init=True)
        self.confidence = nn.Conv2d(width, 1, 1)
        self.carrier_gain = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.confidence.weight)
        nn.init.constant_(self.confidence.bias, -2.0)

    @staticmethod
    def highpass(values: Tensor) -> Tensor:
        return values - F.avg_pool2d(values, 5, stride=1, padding=2)

    def forward(
        self,
        fused: Tensor,
        descriptors: Tensor,
        carrier: Tensor,
        support: Tensor,
        output_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        features = self.features(fused)
        learned = self.detail(features, descriptors, output_size)
        confidence = torch.sigmoid(
            F.interpolate(self.confidence(features), output_size, mode="bilinear", align_corners=False)
        )
        confidence = confidence * F.interpolate(
            support, output_size, mode="bilinear", align_corners=False
        )
        proposal = torch.tanh(learned) + torch.tanh(self.carrier_gain) * self.highpass(carrier)
        return self.limit * torch.tanh(proposal) * confidence, confidence


@dataclass
class PairedTemporalConfig:
    width: int = 96
    latent_channels: int = 32
    attention_heads: int = 4
    maximum_horizon_days: int = 180
    translation_max_delta_days: int = 1
    flow_steps: int = 4
    deterministic_detail_limit: float = 0.15
    visual_residual_limit: float = 0.20
    texture_block_size: int = 4
    architecture: str = "sparse_paired_anchor_transport_v2"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.width % self.attention_heads:
            raise ValueError("width must be positive and divisible by attention_heads")
        if self.latent_channels <= 0 or self.flow_steps <= 0:
            raise ValueError("latent_channels and flow_steps must be positive")
        if not 1 <= self.maximum_horizon_days <= 3650:
            raise ValueError("maximum_horizon_days must be in [1, 3650]")
        if not 0 <= self.translation_max_delta_days <= 7:
            raise ValueError("translation_max_delta_days must be in [0, 7]")
        if self.texture_block_size <= 0 or self.texture_block_size % 2:
            raise ValueError("texture_block_size must be a positive even integer")
        for name in ("deterministic_detail_limit", "visual_residual_limit"):
            if not 0.0 < float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass
class PairedTemporalOutput:
    physical: Tensor
    log_variance: Tensor
    attention: Tensor
    observation_support: Tensor
    task_is_translation: Tensor
    observable_delta: Tensor
    deterministic_detail: Tensor
    detail_confidence: Tensor
    visual_base: Tensor
    bridge_condition: Tensor
    pre_projection_violation: Tensor
    visual: Tensor | None = None
    stochastic_residual: Tensor | None = None
    residual_amplitude: Tensor | None = None


class SparsePairedAnchorTransport(nn.Module):
    """One registered pair plus one-to-many observations, in either direction."""

    def __init__(self, config: PairedTemporalConfig | None = None) -> None:
        super().__init__()
        self.config = config or PairedTemporalConfig()
        self.adapter = DescriptorSensorAdapter(self.config.width)
        self.fusion = PairedObservationFusion(
            self.config.width,
            self.config.attention_heads,
            self.config.maximum_horizon_days,
            self.config.translation_max_delta_days,
        )
        self.carrier = PairedDeltaCarrier(self.config.width)
        self.physical_head = PairedPhysicalHead(self.config.width)
        self.detail_head = PairedDetailHead(
            self.config.width, self.config.deterministic_detail_limit
        )
        self.residual_codec = ResidualLatentCodec(
            self.config.width, self.config.latent_channels
        )
        self.bridge = ConditionalBridgeVelocity(self.config.latent_channels, self.config.width)
        self.bridge_condition = nn.Sequential(
            nn.Conv2d(self.config.width, self.config.width, 1),
            nn.SiLU(),
            nn.Conv2d(self.config.width, self.config.width, 1),
        )
        nn.init.zeros_(self.bridge_condition[-1].weight)
        nn.init.zeros_(self.bridge_condition[-1].bias)
        self.detail_scale = nn.Parameter(torch.zeros(()))
        self.visual_scale = nn.Parameter(torch.zeros(()))

    @staticmethod
    def bounded_anchor_update(anchor: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
        if anchor.shape != delta.shape:
            raise ValueError("anchor and delta must have the same shape")
        raw = anchor + delta
        signed = torch.tanh(delta)
        positive_room = (1.0 - anchor).clamp_min(0.0)
        negative_room = (1.0 + anchor).clamp_min(0.0)
        bounded = anchor + torch.where(signed >= 0, positive_room * signed, negative_room * signed)
        violation = (raw.abs() > 1.0).to(raw).mean(dim=(1, 2, 3), keepdim=True)
        return bounded, violation

    def _validate_inputs(
        self,
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
    ) -> tuple[int, int, int, int]:
        if observations.ndim != 5:
            raise ValueError("observations must have shape BxTxCxHxW")
        batch, frames, _, height, width = observations.shape
        if frames < 1:
            raise ValueError("at least one observation slot is required")
        if height % 4 or width % 4:
            raise ValueError("spatial dimensions must be divisible by four")
        if height % self.config.texture_block_size or width % self.config.texture_block_size:
            raise ValueError("spatial dimensions must match texture_block_size")
        if observation_valid.shape != (batch, frames, 1, height, width):
            raise ValueError("observation_valid must have shape BxTx1xHxW")
        if observation_days.shape != (batch, frames):
            raise ValueError("observation_days must have shape BxT")
        if observation_present.shape != (batch, frames):
            raise ValueError("observation_present must have shape BxT")
        if source_anchor.ndim != 4 or source_anchor.shape[0] != batch:
            raise ValueError("source_anchor must have shape BxCxHxW")
        if source_anchor.shape[-2:] != (height, width):
            raise ValueError("source anchor and observations must share a grid")
        if source_anchor_valid.shape != (batch, 1, height, width):
            raise ValueError("source_anchor_valid must have shape Bx1xHxW")
        if target_anchor.ndim != 4 or target_anchor.shape[0] != batch:
            raise ValueError("target_anchor must have shape BxCxHxW")
        if target_anchor.shape[-2:] != (height, width):
            raise ValueError("registered anchors must share a grid")
        if target_anchor_valid.shape != (batch, 1, height, width):
            raise ValueError("target_anchor_valid must have shape Bx1xHxW")
        for name, values in (
            ("source_anchor_days", source_anchor_days),
            ("target_anchor_days", target_anchor_days),
        ):
            if values.shape not in {(batch,), (batch, 1)}:
                raise ValueError(f"{name} must have shape B or Bx1")
            flat_days = values.reshape(batch)
            if bool((flat_days >= 0).any()):
                raise ValueError(f"the registered {name} must precede the query")
            if bool((flat_days < -float(self.config.maximum_horizon_days)).any()):
                raise ValueError(f"registered {name} exceeds the temporal horizon")
        return batch, frames, height, width

    def _context(
        self,
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
        source_sensor: SensorSpec,
        target_sensor: SensorSpec,
        source_descriptors: Tensor | None,
        target_descriptors: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, frames, height, width = self._validate_inputs(
            observations,
            observation_valid,
            observation_days,
            observation_present,
            source_anchor,
            source_anchor_valid,
            target_anchor,
            target_anchor_valid,
            source_anchor_days,
            target_anchor_days,
        )
        source_description = _descriptors(
            source_descriptors,
            source_sensor,
            batch=batch,
            channels=observations.shape[2],
            device=observations.device,
            dtype=observations.dtype,
        )
        target_description = _descriptors(
            target_descriptors,
            target_sensor,
            batch=batch,
            channels=target_anchor.shape[1],
            device=target_anchor.device,
            dtype=target_anchor.dtype,
        )
        flattened_observations = observations.reshape(
            batch * frames, observations.shape[2], height, width
        )
        flattened_valid = observation_valid.reshape(batch * frames, 1, height, width)
        flattened_descriptions = source_description[:, None].expand(
            -1, frames, -1, -1
        ).reshape(batch * frames, observations.shape[2], -1)
        observation_features = self.adapter(
            flattened_observations, flattened_descriptions, flattened_valid
        ).reshape(batch, frames, self.config.width, height // 4, width // 4)
        source_anchor_features = self.adapter(
            source_anchor, source_description, source_anchor_valid
        )
        target_anchor_features = self.adapter(
            target_anchor, target_description, target_anchor_valid
        )
        fused, attention, support = self.fusion(
            observation_features,
            observation_valid,
            source_anchor_features,
            target_anchor_features,
            observation_days.to(observations),
            source_anchor_days.to(observations),
            target_anchor_days.to(observations),
            observation_present,
        )
        return fused, attention, support, source_description, target_description, target_anchor_features

    def physical(
        self,
        observations: Tensor,
        observation_valid: Tensor,
        observation_days: Tensor,
        observation_present: Tensor,
        source_anchor: Tensor,
        source_anchor_valid: Tensor,
        target_anchor: Tensor,
        target_anchor_valid: Tensor,
        anchor_days: Tensor | None = None,
        *,
        source_sensor: SensorSpec,
        target_sensor: SensorSpec,
        source_descriptors: Tensor | None = None,
        target_descriptors: Tensor | None = None,
        source_anchor_days: Tensor | None = None,
        target_anchor_days: Tensor | None = None,
    ) -> PairedTemporalOutput:
        if anchor_days is None and (
            source_anchor_days is None or target_anchor_days is None
        ):
            raise ValueError(
                "anchor_days or both source_anchor_days and target_anchor_days are required"
            )
        resolved_source_anchor_days = (
            anchor_days if source_anchor_days is None else source_anchor_days
        )
        resolved_target_anchor_days = (
            anchor_days if target_anchor_days is None else target_anchor_days
        )
        assert resolved_source_anchor_days is not None
        assert resolved_target_anchor_days is not None
        fused, attention, support, source_description, target_description, _ = self._context(
            observations,
            observation_valid,
            observation_days,
            observation_present,
            source_anchor,
            source_anchor_valid,
            target_anchor,
            target_anchor_valid,
            resolved_source_anchor_days,
            resolved_target_anchor_days,
            source_sensor,
            target_sensor,
            source_descriptors,
            target_descriptors,
        )
        carrier = self.carrier(
            observations,
            observation_valid,
            observation_present,
            source_anchor,
            source_anchor_valid,
            source_description,
            target_description,
            attention,
        )
        learned_delta, learned_log_variance = self.physical_head(
            fused, target_description, target_anchor.shape[-2:]
        )
        physical, physical_violation = self.bounded_anchor_update(
            target_anchor, carrier + learned_delta
        )
        latest_day = torch.where(
            observation_present.bool(),
            observation_days.to(physical),
            torch.full_like(observation_days.to(physical), -1e6),
        ).max(dim=1).values
        task_is_translation = (
            latest_day.abs() <= float(self.config.translation_max_delta_days)
        ).to(physical)[:, None, None, None]
        count = observation_present.to(physical).sum(dim=1).clamp_min(1.0)
        scarcity = count.reciprocal()[:, None, None, None]
        forecast_gap = (-latest_day / float(self.config.maximum_horizon_days)).clamp(0.0, 1.0)
        analytic_uncertainty = 0.25 * scarcity + 0.75 * forecast_gap[:, None, None, None]
        log_variance = (learned_log_variance + analytic_uncertainty).clamp(-8.0, 4.0)
        detail, detail_confidence = self.detail_head(
            fused,
            target_description,
            carrier,
            support,
            target_anchor.shape[-2:],
        )
        visual_base, detail_violation = self.bounded_anchor_update(
            physical, detail * torch.tanh(self.detail_scale).clamp_min(0.0)
        )
        condition = self.adapter(physical, target_description, target_anchor_valid)
        condition = condition + self.bridge_condition(fused)
        return PairedTemporalOutput(
            physical=physical,
            log_variance=log_variance.expand_as(physical),
            attention=attention,
            observation_support=support,
            task_is_translation=task_is_translation,
            observable_delta=carrier,
            deterministic_detail=detail,
            detail_confidence=detail_confidence,
            visual_base=visual_base,
            bridge_condition=condition,
            pre_projection_violation=torch.maximum(physical_violation, detail_violation),
        )

    def visual_flow_loss(
        self,
        output: PairedTemporalOutput,
        target: Tensor,
        target_valid: Tensor,
        target_sensor: SensorSpec,
        *,
        target_descriptors: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        if target.shape != output.physical.shape:
            raise ValueError("target must match physical output")
        if target_valid.shape != (target.shape[0], 1, *target.shape[-2:]):
            raise ValueError("target_valid must have shape Bx1xHxW")
        descriptors = _descriptors(
            target_descriptors,
            target_sensor,
            batch=target.shape[0],
            channels=target.shape[1],
            device=target.device,
            dtype=target.dtype,
        )
        residual = self._zero_block_mean(
            (target - output.visual_base.detach()) * target_valid,
            self.config.texture_block_size,
        )
        target_latent = self.residual_codec.encode(residual, descriptors, target_valid)
        origin, log_scale = self.bridge.origin_distribution(output.bridge_condition)
        if generator is None:
            # Keep the training-time RNG path unchanged unless validation
            # explicitly requests a replayable local generator.
            start_noise = torch.randn_like(origin)
            time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        else:
            start_noise = torch.randn(
                origin.shape,
                device=origin.device,
                dtype=origin.dtype,
                generator=generator,
            )
            time = torch.rand(
                target.shape[0],
                device=target.device,
                dtype=target.dtype,
                generator=generator,
            )
        start = origin + log_scale.exp() * start_noise
        interpolated = start.lerp(target_latent, time[:, None, None, None])
        velocity_target = target_latent - start
        velocity = self.bridge(interpolated, time, output.bridge_condition)
        endpoint = interpolated + (1.0 - time[:, None, None, None]) * velocity
        reconstruction = self.residual_codec.decode(
            target_latent, descriptors, target.shape[-2:]
        )
        endpoint_residual = self.residual_codec.decode(
            endpoint, descriptors, target.shape[-2:]
        )
        latent_valid = F.avg_pool2d(target_valid.float(), 4, stride=4).to(target)
        return {
            "flow_velocity": self._masked_mean((velocity - velocity_target).abs(), latent_valid),
            "flow_endpoint": self._masked_mean((endpoint - target_latent).abs(), latent_valid),
            "codec_reconstruction": self._masked_mean(
                (reconstruction - residual).abs(), target_valid
            ),
            "flow_endpoint_pixel": self._masked_mean(
                (endpoint_residual - residual).abs(), target_valid
            ),
        }

    def sample_visual(
        self,
        output: PairedTemporalOutput,
        target_valid: Tensor,
        target_sensor: SensorSpec,
        *,
        seed: int,
        target_descriptors: Tensor | None = None,
        steps: int | None = None,
    ) -> PairedTemporalOutput:
        base = output.visual_base
        descriptors = _descriptors(
            target_descriptors,
            target_sensor,
            batch=base.shape[0],
            channels=base.shape[1],
            device=base.device,
            dtype=base.dtype,
        )
        origin, log_scale = self.bridge.origin_distribution(output.bridge_condition)
        generator = torch.Generator(device=base.device).manual_seed(seed)
        latent = origin + log_scale.exp() * torch.randn(
            origin.shape, device=origin.device, dtype=origin.dtype, generator=generator
        )
        total_steps = self.config.flow_steps if steps is None else int(steps)
        if total_steps <= 0:
            raise ValueError("steps must be positive")
        dt = 1.0 / total_steps
        for index in range(total_steps):
            time = torch.full((base.shape[0],), index * dt, device=base.device, dtype=base.dtype)
            latent = latent + dt * self.bridge(latent, time, output.bridge_condition)
        residual = self.residual_codec.decode(latent, descriptors, base.shape[-2:])
        residual = self._zero_block_mean(residual, self.config.texture_block_size)
        release = torch.tanh(self.visual_scale).clamp_min(0.0)
        amplitude = self.config.visual_residual_limit * release * output.detail_confidence
        residual = residual * amplitude * target_valid
        visual, violation = self.bounded_anchor_update(base, residual)
        return PairedTemporalOutput(
            physical=output.physical,
            log_variance=output.log_variance,
            attention=output.attention,
            observation_support=output.observation_support,
            task_is_translation=output.task_is_translation,
            observable_delta=output.observable_delta,
            deterministic_detail=output.deterministic_detail,
            detail_confidence=output.detail_confidence,
            visual_base=base,
            bridge_condition=output.bridge_condition,
            pre_projection_violation=torch.maximum(output.pre_projection_violation, violation),
            visual=visual,
            stochastic_residual=residual,
            residual_amplitude=amplitude,
        )

    def forward(
        self,
        observations: Tensor,
        observation_valid: Tensor,
        observation_days: Tensor,
        observation_present: Tensor,
        source_anchor: Tensor,
        source_anchor_valid: Tensor,
        target_anchor: Tensor,
        target_anchor_valid: Tensor,
        anchor_days: Tensor | None = None,
        *,
        source_sensor: SensorSpec,
        target_sensor: SensorSpec,
        source_descriptors: Tensor | None = None,
        target_descriptors: Tensor | None = None,
        source_anchor_days: Tensor | None = None,
        target_anchor_days: Tensor | None = None,
    ) -> PairedTemporalOutput:
        return self.physical(
            observations,
            observation_valid,
            observation_days,
            observation_present,
            source_anchor,
            source_anchor_valid,
            target_anchor,
            target_anchor_valid,
            anchor_days,
            source_sensor=source_sensor,
            target_sensor=target_sensor,
            source_descriptors=source_descriptors,
            target_descriptors=target_descriptors,
            source_anchor_days=source_anchor_days,
            target_anchor_days=target_anchor_days,
        )

    @staticmethod
    def _zero_block_mean(values: Tensor, block_size: int) -> Tensor:
        if values.ndim != 4:
            raise ValueError("residual must have shape BxCxHxW")
        if values.shape[-2] % block_size or values.shape[-1] % block_size:
            raise ValueError("residual dimensions must be divisible by texture block size")
        means = F.avg_pool2d(values, block_size, stride=block_size)
        return values - F.interpolate(means, values.shape[-2:], mode="nearest")

    @staticmethod
    def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
        if valid.ndim != 4 or valid.shape[0] != values.shape[0] or valid.shape[1] != 1:
            raise ValueError("valid must have shape Bx1xHxW")
        if valid.shape[-2:] != values.shape[-2:]:
            valid = F.interpolate(valid.float(), values.shape[-2:], mode="area").to(values)
        expanded = valid.expand_as(values).to(values)
        return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def sensor_descriptors(channels: tuple[ChannelSpec, ...], device: torch.device) -> Tensor:
    """Build custom-sensor descriptors for few-shot adapter experiments."""

    return torch.tensor([channel.descriptor() for channel in channels], device=device)
