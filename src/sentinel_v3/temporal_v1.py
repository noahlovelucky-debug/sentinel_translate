"""Causal anchor-delta transport for multi-temporal cross-modal translation.

This module is intentionally separate from the V3.2 single-observation model.
The contract is causal: all source frames are acquired no later than the query,
and the only target-modality input is one earlier, real anchor.  The physical
path predicts an anchor-relative deterministic change.  The optional visual
path transports only the residual left after that physical prediction.

The channel adapters consume sensor descriptors rather than hard-coding a
Sentinel channel count.  This makes a paired, few-shot sensor adapter a viable
future extension without changing the temporal core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .sensors import ChannelSpec, SensorSpec


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _channel_descriptors(
    sensor: SensorSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return torch.tensor(
        [channel.descriptor() for channel in sensor.channels], device=device, dtype=dtype
    )


def _as_batch_descriptors(
    descriptors: Tensor | None,
    sensor: SensorSpec,
    *,
    batch: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if descriptors is None:
        descriptors = _channel_descriptors(sensor, device, dtype)
    else:
        descriptors = descriptors.to(device=device, dtype=dtype)
    if descriptors.ndim == 2:
        if descriptors.shape[0] != channels:
            raise ValueError("one channel descriptor is required for each input channel")
        return descriptors.unsqueeze(0).expand(batch, -1, -1)
    if descriptors.ndim == 3 and descriptors.shape[:2] == (batch, channels):
        return descriptors
    raise ValueError("descriptors must have shape CxD or BxCxD")


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


class DescriptorSensorAdapter(nn.Module):
    """Encode an arbitrary set of sensor channels on a shared 4x latent grid."""

    descriptor_dim = 8

    def __init__(self, width: int) -> None:
        super().__init__()
        self.channel_embedding = nn.Sequential(
            nn.Linear(self.descriptor_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.channel_bias = nn.Sequential(
            nn.Linear(self.descriptor_dim, width),
            nn.Tanh(),
        )
        self.valid_embedding = nn.Conv2d(1, width, 3, padding=1)
        self.stem = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            _ResidualBlock(width),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            _ResidualBlock(width),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            _ResidualBlock(width),
        )

    def forward(self, values: Tensor, descriptors: Tensor, valid: Tensor) -> Tensor:
        if values.ndim != 4 or valid.ndim != 4:
            raise ValueError("values and valid must be BCHW tensors")
        if valid.shape != (values.shape[0], 1, *values.shape[-2:]):
            raise ValueError("valid must have one channel and match values")
        if descriptors.shape[:2] != values.shape[:2] or descriptors.shape[-1] != self.descriptor_dim:
            raise ValueError("descriptors must have shape BxCx8 matching values")
        embedding = self.channel_embedding(descriptors)
        projected = torch.einsum("bchw,bcf->bfhw", values * valid, embedding)
        projected = projected / math.sqrt(max(1, values.shape[1]))
        bias = self.channel_bias(descriptors).mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.stem(projected + bias + self.valid_embedding(valid))


class DynamicChannelProjection(nn.Module):
    """Decode common features into an arbitrary target channel description."""

    descriptor_dim = 8

    def __init__(self, width: int, *, zero_init: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            _ResidualBlock(width),
            nn.Conv2d(width, width, 3, padding=1),
        )
        self.channel_embedding = nn.Sequential(
            nn.Linear(self.descriptor_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.channel_bias = nn.Linear(self.descriptor_dim, 1)
        if zero_init:
            last = self.features[-1]
            assert isinstance(last, nn.Conv2d)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            nn.init.zeros_(self.channel_bias.weight)
            nn.init.zeros_(self.channel_bias.bias)

    def forward(self, features: Tensor, descriptors: Tensor, output_size: tuple[int, int]) -> Tensor:
        if descriptors.ndim != 3 or descriptors.shape[0] != features.shape[0]:
            raise ValueError("descriptors must have shape BxCx8")
        if descriptors.shape[-1] != self.descriptor_dim:
            raise ValueError("target descriptor dimension must be eight")
        decoded = F.interpolate(features, size=output_size, mode="bilinear", align_corners=False)
        decoded = self.features(decoded)
        channel_embedding = self.channel_embedding(descriptors)
        output = torch.einsum("bfhw,bcf->bchw", decoded, channel_embedding)
        output = output / math.sqrt(max(1, decoded.shape[1]))
        return output + self.channel_bias(descriptors).squeeze(-1).unsqueeze(-1).unsqueeze(-1)


class ObservableSourceCarrier(nn.Module):
    """Project time-attended source measurements into target-channel deltas.

    The carrier is deliberately shallow.  It gives physical transport a
    direct route from an observed source change to a target change, while the
    deeper delta head remains available for cross-modal nonlinearities.  Its
    final projection is zero-initialized, preserving the anchor baseline at
    initialization and allowing a few-shot sensor adapter to tune only this
    component later.
    """

    descriptor_dim = 8

    def __init__(self, width: int) -> None:
        super().__init__()
        self.source_embedding = nn.Sequential(
            nn.Linear(self.descriptor_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            _ResidualBlock(width),
        )
        self.target_projection = DynamicChannelProjection(width, zero_init=True)

    def forward(
        self,
        source_values: Tensor,
        source_valid: Tensor,
        source_descriptors: Tensor,
        target_descriptors: Tensor,
        attention: Tensor,
        output_size: tuple[int, int],
    ) -> Tensor:
        if source_values.ndim != 5:
            raise ValueError("source_values must have shape BxTxCxHxW")
        batch, frames, channels, height, width = source_values.shape
        if source_valid.shape != (batch, frames, 1, height, width):
            raise ValueError("source_valid must have shape BxTx1xHxW")
        if source_descriptors.shape != (batch, channels, self.descriptor_dim):
            raise ValueError("source descriptors must have shape BxCx8")
        if attention.shape[:3] != (batch, frames, 1):
            raise ValueError("attention must have shape BxTx1xhxw")
        if output_size != (height, width):
            raise ValueError("carrier output size must match source values")
        expanded_attention = F.interpolate(
            attention.reshape(batch * frames, 1, *attention.shape[-2:]).float(),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, 1, height, width).to(source_values)
        weights = expanded_attention * source_valid.to(source_values)
        aggregate = (source_values * weights).sum(dim=1)
        aggregate = aggregate / weights.sum(dim=1).clamp_min(1e-6)
        features = torch.einsum(
            "bchw,bcf->bfhw", aggregate, self.source_embedding(source_descriptors)
        ) / math.sqrt(max(1, channels))
        return self.target_projection(self.refine(features), target_descriptors, output_size)


class CausalTemporalFusion(nn.Module):
    """Spatial source-time attention with an explicit no-future-frame invariant."""

    def __init__(self, width: int, heads: int, maximum_horizon_days: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("temporal fusion width must be divisible by heads")
        self.maximum_horizon_days = maximum_horizon_days
        self.time_embedding = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 3, width, 3, padding=1),
            _ResidualBlock(width),
        )

    def _time_features(self, days: Tensor) -> Tensor:
        scaled = days / float(self.maximum_horizon_days)
        phase = 2.0 * math.pi * scaled
        return torch.stack((scaled, torch.sin(phase), torch.cos(phase)), dim=-1)

    def forward(
        self,
        source: Tensor,
        anchor: Tensor,
        source_days: Tensor,
        anchor_days: Tensor,
        source_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if source.ndim != 5:
            raise ValueError("source features must have shape BxTxCxHxW")
        batch, frames, channels, height, width = source.shape
        if anchor.shape != (batch, channels, height, width):
            raise ValueError("anchor features must match source spatial shape")
        if source_days.shape != (batch, frames):
            raise ValueError("source_days must have shape BxT")
        if anchor_days.shape not in {(batch,), (batch, 1)}:
            raise ValueError("anchor_days must have shape B or Bx1")
        if source_valid.shape != (batch, frames, 1, height * 4, width * 4):
            raise ValueError("source_valid must match the 4x source grid")
        if bool((source_days > 0).any()):
            raise ValueError("causal temporal fusion received a future source frame")
        if bool((anchor_days.reshape(batch) >= 0).any()):
            raise ValueError("the target-modality anchor must precede the query")
        if bool((source_days < -float(self.maximum_horizon_days)).any()):
            raise ValueError("source frame exceeds the configured temporal horizon")
        if bool((anchor_days.reshape(batch) < -float(self.maximum_horizon_days)).any()):
            raise ValueError("anchor exceeds the configured temporal horizon")

        source_temporal = self.time_embedding(self._time_features(source_days))
        anchor_temporal = self.time_embedding(
            self._time_features(anchor_days.reshape(batch, 1))
        ).reshape(batch, channels, 1, 1)
        source_tokens = (source + source_temporal[:, :, :, None, None]).permute(0, 3, 4, 1, 2)
        source_tokens = source_tokens.reshape(batch * height * width, frames, channels)
        query_tokens = (anchor + anchor_temporal).permute(0, 2, 3, 1)
        query_tokens = query_tokens.reshape(batch * height * width, 1, channels)
        coverage = F.avg_pool2d(
            source_valid.reshape(batch * frames, 1, height * 4, width * 4).float(), 4, stride=4
        ).reshape(batch, frames, height, width)
        key_padding = (coverage <= 0.0).permute(0, 2, 3, 1).reshape(batch * height * width, frames)
        # MultiheadAttention emits NaNs for an all-masked row.  Such a pixel has
        # no source evidence, so leave a zero-valued first token available.
        no_source = key_padding.all(dim=1)
        if bool(no_source.any()):
            source_tokens = source_tokens.clone()
            key_padding = key_padding.clone()
            source_tokens[no_source, 0] = 0.0
            key_padding[no_source, 0] = False
        fused_tokens, weights = self.attention(
            query_tokens,
            source_tokens,
            source_tokens,
            key_padding_mask=key_padding,
            need_weights=True,
            average_attn_weights=True,
        )
        fused = fused_tokens.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        attention = weights.reshape(batch, height, width, frames).permute(0, 3, 1, 2).unsqueeze(2)
        return self.fuse(torch.cat((anchor, fused, fused - anchor), dim=1)), attention


class PhysicalDeltaHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), _ResidualBlock(width), _ResidualBlock(width)
        )
        self.delta = DynamicChannelProjection(width, zero_init=True)
        self.log_variance = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, 1, 1),
        )
        nn.init.zeros_(self.log_variance[-1].weight)
        nn.init.zeros_(self.log_variance[-1].bias)

    def forward(
        self, fused: Tensor, target_descriptors: Tensor, output_size: tuple[int, int]
    ) -> tuple[Tensor, Tensor]:
        features = self.body(fused)
        delta = self.delta(features, target_descriptors, output_size)
        log_variance = F.interpolate(
            self.log_variance(features), size=output_size, mode="bilinear", align_corners=False
        ).clamp(-8.0, 4.0)
        return delta, log_variance


class DeterministicDetailHead(nn.Module):
    """Predict only source-supported, repeatable target detail.

    The direct carrier term transfers high-frequency structure observed in the
    source sequence.  A separate learned term handles cross-modal edge shape.
    Both begin at zero, so the physical output is unchanged until this stage is
    explicitly trained.
    """

    def __init__(self, width: int, detail_limit: float) -> None:
        super().__init__()
        self.detail_limit = detail_limit
        self.features = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            _ResidualBlock(width),
            _ResidualBlock(width),
        )
        self.detail = DynamicChannelProjection(width, zero_init=True)
        self.confidence = nn.Conv2d(width, 1, 1)
        self.carrier_gain = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.confidence.weight)
        nn.init.constant_(self.confidence.bias, -1.5)

    @staticmethod
    def highpass(values: Tensor) -> Tensor:
        return values - F.avg_pool2d(values, 5, stride=1, padding=2)

    def forward(
        self,
        fused: Tensor,
        descriptors: Tensor,
        output_size: tuple[int, int],
        observable_delta: Tensor,
    ) -> tuple[Tensor, Tensor]:
        features = self.features(fused)
        learned = self.detail(features, descriptors, output_size)
        confidence = torch.sigmoid(
            F.interpolate(self.confidence(features), size=output_size, mode="bilinear", align_corners=False)
        )
        carrier = self.highpass(observable_delta)
        proposal = torch.tanh(learned) + torch.tanh(self.carrier_gain) * carrier
        detail = self.detail_limit * torch.tanh(proposal) * confidence
        return detail, confidence


class TextureReliabilityHead(nn.Module):
    """Estimate where a stochastic texture residual may safely be released.

    The result is deliberately a one-channel spatial gate rather than another
    image synthesis head.  It is trained only after physical/detail/flow have
    established their separate responsibilities, and starts near zero so a
    fresh checkpoint cannot inject broad texture into flat or uncertain areas.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            _ResidualBlock(width),
        )
        self.logit = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.logit.weight)
        nn.init.constant_(self.logit.bias, -4.0)

    def forward(self, fused: Tensor, output_size: tuple[int, int]) -> Tensor:
        logits = F.interpolate(
            self.logit(self.features(fused)),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(logits)


class ResidualLatentCodec(nn.Module):
    """A small descriptor-conditioned residual codec used only by visual transport."""

    def __init__(self, width: int, latent_channels: int) -> None:
        super().__init__()
        self.encoder = DescriptorSensorAdapter(width)
        self.to_latent = nn.Conv2d(width, latent_channels, 1)
        self.from_latent = nn.Conv2d(latent_channels, width, 1)
        self.decoder = DynamicChannelProjection(width)

    def encode(self, values: Tensor, descriptors: Tensor, valid: Tensor) -> Tensor:
        return self.to_latent(self.encoder(values, descriptors, valid))

    def decode(self, latent: Tensor, descriptors: Tensor, output_size: tuple[int, int]) -> Tensor:
        return self.decoder(self.from_latent(latent), descriptors, output_size)


class ConditionalBridgeVelocity(nn.Module):
    """Continuous residual bridge velocity, conditioned on causal physical context."""

    def __init__(self, latent_channels: int, width: int) -> None:
        super().__init__()
        self.time_embedding = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.origin = nn.Conv2d(width, latent_channels * 2, 1)
        self.body = nn.Sequential(
            nn.Conv2d(latent_channels + width, width, 3, padding=1),
            _ResidualBlock(width),
            _ResidualBlock(width),
            nn.Conv2d(width, latent_channels, 3, padding=1),
        )
        nn.init.zeros_(self.origin.weight)
        nn.init.zeros_(self.origin.bias)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def origin_distribution(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        origin, raw_log_scale = self.origin(condition).chunk(2, dim=1)
        return origin, raw_log_scale.clamp(-4.0, 0.0)

    def forward(self, latent: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        if time.ndim != 1 or time.shape[0] != latent.shape[0]:
            raise ValueError("bridge time must have shape B")
        phase = 2.0 * math.pi * time
        embedding = self.time_embedding(torch.stack((time, torch.sin(phase), torch.cos(phase)), dim=-1))
        conditioned = condition + embedding[:, :, None, None]
        return self.body(torch.cat((latent, conditioned), dim=1))


@dataclass
class TemporalModelConfig:
    width: int = 96
    latent_channels: int = 32
    temporal_heads: int = 4
    maximum_horizon_days: int = 180
    flow_steps: int = 4
    deterministic_detail_limit: float = 0.15
    visual_residual_limit: float = 0.20
    texture_block_size: int = 4
    architecture: str = "causal_anchor_delta_transport_v1"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.width % self.temporal_heads:
            raise ValueError("width must be positive and divisible by temporal_heads")
        if self.latent_channels <= 0 or self.maximum_horizon_days <= 0 or self.flow_steps <= 0:
            raise ValueError("latent_channels, horizon, and flow_steps must be positive")
        if self.texture_block_size <= 0 or self.texture_block_size % 2:
            raise ValueError("texture_block_size must be a positive even integer")
        for name in ("deterministic_detail_limit", "visual_residual_limit"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass
class TemporalTranslationOutput:
    physical: Tensor
    log_variance: Tensor
    attention: Tensor
    observable_delta: Tensor | None = None
    deterministic_detail: Tensor | None = None
    detail_confidence: Tensor | None = None
    texture_reliability: Tensor | None = None
    deterministic_visual_base: Tensor | None = None
    visual_base: Tensor | None = None
    bridge_condition: Tensor | None = None
    pre_projection_violation: Tensor | None = None
    visual: Tensor | None = None
    stochastic_residual: Tensor | None = None
    residual_amplitude: Tensor | None = None


class CausalAnchorDeltaTransport(nn.Module):
    """Multi-temporal cross-modal model with a radiometrically anchored physical path.

    Inputs use the established V3 normalized representation.  The physical
    projection is an anchor-preserving signed update: it cannot leave [-1, 1]
    and does not need a final clamp to hide an amplitude error.
    """

    def __init__(self, config: TemporalModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalModelConfig()
        self.adapter = DescriptorSensorAdapter(self.config.width)
        self.fusion = CausalTemporalFusion(
            self.config.width,
            self.config.temporal_heads,
            self.config.maximum_horizon_days,
        )
        self.physical_head = PhysicalDeltaHead(self.config.width)
        self.observable_carrier = ObservableSourceCarrier(self.config.width)
        self.detail_head = DeterministicDetailHead(
            self.config.width, self.config.deterministic_detail_limit
        )
        self.texture_reliability = TextureReliabilityHead(self.config.width)
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
        # Both visual releases start closed.  Detail can be trained against its
        # oracle residual before it is allowed to alter deployment output.
        self.detail_scale = nn.Parameter(torch.zeros(()))
        self.visual_scale = nn.Parameter(torch.zeros(()))

    @staticmethod
    def bounded_anchor_update(anchor: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
        """Apply a signed bounded correction and report its unprojected violation."""

        if anchor.shape != delta.shape:
            raise ValueError("anchor and delta must have the same shape")
        raw = anchor + delta
        signed = torch.tanh(delta)
        positive_room = (1.0 - anchor).clamp_min(0.0)
        negative_room = (1.0 + anchor).clamp_min(0.0)
        bounded = anchor + torch.where(signed >= 0, positive_room * signed, negative_room * signed)
        violation = (raw.abs() > 1.0).to(raw.dtype).mean(dim=(1, 2, 3), keepdim=True)
        return bounded, violation

    def _validate_inputs(
        self,
        source_values: Tensor,
        source_valid: Tensor,
        anchor_values: Tensor,
        anchor_valid: Tensor,
        source_days: Tensor,
        anchor_days: Tensor,
    ) -> tuple[int, int, int, int, int]:
        if source_values.ndim != 5:
            raise ValueError("source_values must have shape BxTxCxHxW")
        batch, frames, _, height, width = source_values.shape
        if height % 4 or width % 4:
            raise ValueError("source and anchor spatial dimensions must be divisible by four")
        if height % self.config.texture_block_size or width % self.config.texture_block_size:
            raise ValueError("source and anchor dimensions must match texture_block_size")
        if source_valid.shape != (batch, frames, 1, height, width):
            raise ValueError("source_valid must have shape BxTx1xHxW")
        if anchor_values.ndim != 4 or anchor_values.shape[0] != batch:
            raise ValueError("anchor_values must have shape BxCxHxW")
        if anchor_values.shape[-2:] != (height, width):
            raise ValueError("source and anchor grids must match")
        if anchor_valid.shape != (batch, 1, height, width):
            raise ValueError("anchor_valid must have shape Bx1xHxW")
        if source_days.shape != (batch, frames):
            raise ValueError("source_days must have shape BxT")
        if anchor_days.shape not in {(batch,), (batch, 1)}:
            raise ValueError("anchor_days must have shape B or Bx1")
        return batch, frames, height, width, anchor_values.shape[1]

    def _context(
        self,
        source_values: Tensor,
        source_valid: Tensor,
        anchor_values: Tensor,
        anchor_valid: Tensor,
        source_days: Tensor,
        anchor_days: Tensor,
        source_sensor: SensorSpec,
        target_sensor: SensorSpec,
        source_descriptors: Tensor | None,
        target_descriptors: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, frames, height, width, target_channels = self._validate_inputs(
            source_values,
            source_valid,
            anchor_values,
            anchor_valid,
            source_days,
            anchor_days,
        )
        source_descriptor_batch = _as_batch_descriptors(
            source_descriptors,
            source_sensor,
            batch=batch,
            channels=source_values.shape[2],
            device=source_values.device,
            dtype=source_values.dtype,
        )
        target_descriptor_batch = _as_batch_descriptors(
            target_descriptors,
            target_sensor,
            batch=batch,
            channels=target_channels,
            device=anchor_values.device,
            dtype=anchor_values.dtype,
        )
        source_valid = source_valid.to(device=source_values.device, dtype=source_values.dtype)
        anchor_valid = anchor_valid.to(device=anchor_values.device, dtype=anchor_values.dtype)
        source_days = source_days.to(device=source_values.device, dtype=source_values.dtype)
        anchor_days = anchor_days.to(device=anchor_values.device, dtype=anchor_values.dtype)
        flattened_source = source_values.reshape(batch * frames, source_values.shape[2], height, width)
        flattened_valid = source_valid.reshape(batch * frames, 1, height, width)
        flattened_descriptors = source_descriptor_batch.unsqueeze(1).expand(
            -1, frames, -1, -1
        ).reshape(batch * frames, source_values.shape[2], -1)
        source_features = self.adapter(
            flattened_source, flattened_descriptors, flattened_valid
        ).reshape(batch, frames, self.config.width, height // 4, width // 4)
        anchor_features = self.adapter(anchor_values, target_descriptor_batch, anchor_valid)
        fused, attention = self.fusion(
            source_features, anchor_features, source_days, anchor_days, source_valid
        )
        return fused, attention, target_descriptor_batch, anchor_features

    def physical(
        self,
        source_values: Tensor,
        source_valid: Tensor,
        anchor_values: Tensor,
        anchor_valid: Tensor,
        source_days: Tensor,
        anchor_days: Tensor,
        *,
        source_sensor: SensorSpec,
        target_sensor: SensorSpec,
        source_descriptors: Tensor | None = None,
        target_descriptors: Tensor | None = None,
    ) -> TemporalTranslationOutput:
        fused, attention, target_descriptor_batch, _ = self._context(
            source_values,
            source_valid,
            anchor_values,
            anchor_valid,
            source_days,
            anchor_days,
            source_sensor,
            target_sensor,
            source_descriptors,
            target_descriptors,
        )
        residual_delta, log_variance = self.physical_head(
            fused, target_descriptor_batch, anchor_values.shape[-2:]
        )
        source_descriptor_batch = _as_batch_descriptors(
            source_descriptors,
            source_sensor,
            batch=source_values.shape[0],
            channels=source_values.shape[2],
            device=source_values.device,
            dtype=source_values.dtype,
        )
        observable_delta = self.observable_carrier(
            source_values,
            source_valid,
            source_descriptor_batch,
            target_descriptor_batch,
            attention,
            anchor_values.shape[-2:],
        )
        delta = residual_delta + observable_delta
        physical, physical_violation = self.bounded_anchor_update(anchor_values, delta)
        deterministic_detail, detail_confidence = self.detail_head(
            fused, target_descriptor_batch, anchor_values.shape[-2:], observable_delta
        )
        texture_reliability = self.texture_reliability(fused, anchor_values.shape[-2:])
        deterministic_visual_base, full_detail_violation = self.bounded_anchor_update(
            physical, deterministic_detail
        )
        released_detail = deterministic_detail * torch.tanh(self.detail_scale)
        visual_base, detail_violation = self.bounded_anchor_update(physical, released_detail)
        bridge_condition = self.adapter(physical, target_descriptor_batch, anchor_valid)
        bridge_condition = bridge_condition + self.bridge_condition(fused)
        return TemporalTranslationOutput(
            physical=physical,
            log_variance=log_variance.expand_as(physical),
            attention=attention,
            observable_delta=observable_delta,
            deterministic_detail=deterministic_detail,
            detail_confidence=detail_confidence,
            texture_reliability=texture_reliability,
            deterministic_visual_base=deterministic_visual_base,
            visual_base=visual_base,
            bridge_condition=bridge_condition,
            pre_projection_violation=torch.maximum(
                physical_violation, torch.maximum(full_detail_violation, detail_violation)
            ),
        )

    def visual_flow_loss(
        self,
        output: TemporalTranslationOutput,
        target_values: Tensor,
        target_valid: Tensor,
        target_sensor: SensorSpec,
        *,
        target_descriptors: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Train a bridge from a conditioned origin to the unexplained residual."""

        if target_values.shape != output.physical.shape or target_valid.shape != (
            target_values.shape[0],
            1,
            *target_values.shape[-2:],
        ):
            raise ValueError("target values and valid mask must match physical output")
        descriptors = _as_batch_descriptors(
            target_descriptors,
            target_sensor,
            batch=target_values.shape[0],
            channels=target_values.shape[1],
            device=target_values.device,
            dtype=target_values.dtype,
        )
        # This must be the same release-scaled base used by `sample_visual`.
        # `set_temporal_stage(..., "flow")` opens a completed detail stage;
        # calibration later either keeps that full base or closes visual output.
        base = output.visual_base if output.visual_base is not None else output.physical
        residual = self._zero_block_mean(
            (target_values - base.detach()) * target_valid,
            self.config.texture_block_size,
        )
        target_latent = self.residual_codec.encode(residual, descriptors, target_valid)
        condition = output.bridge_condition
        if condition is None:
            condition = self.adapter(base.detach(), descriptors, target_valid)
        origin, log_scale = self.bridge.origin_distribution(condition)
        noise = torch.randn_like(origin)
        start = origin + log_scale.exp() * noise
        time = torch.rand(target_values.shape[0], device=target_values.device, dtype=target_values.dtype)
        interpolated = start.lerp(target_latent, time[:, None, None, None])
        velocity_target = target_latent - start
        velocity = self.bridge(interpolated, time, condition)
        endpoint = interpolated + (1.0 - time[:, None, None, None]) * velocity
        reconstruction = self.residual_codec.decode(
            target_latent, descriptors, target_values.shape[-2:]
        )
        valid = target_valid.expand_as(target_values)
        latent_valid = F.avg_pool2d(target_valid.float(), 4, stride=4).to(target_values)
        velocity_loss = _masked_latent_mean((velocity - velocity_target).abs(), latent_valid)
        endpoint_loss = _masked_latent_mean((endpoint - target_latent).abs(), latent_valid)
        codec_loss = ((reconstruction - residual).abs() * valid).sum() / valid.sum().clamp_min(1.0)
        endpoint_residual = self.residual_codec.decode(
            endpoint, descriptors, target_values.shape[-2:]
        )
        endpoint_pixel = _masked_latent_mean(
            (endpoint_residual - residual).abs(), target_valid
        )
        return {
            "flow_velocity": velocity_loss,
            "flow_endpoint": endpoint_loss,
            "codec_reconstruction": codec_loss,
            "flow_endpoint_pixel": endpoint_pixel,
        }

    def sample_visual(
        self,
        output: TemporalTranslationOutput,
        anchor_valid: Tensor,
        target_sensor: SensorSpec,
        *,
        seed: int,
        target_descriptors: Tensor | None = None,
        steps: int | None = None,
    ) -> TemporalTranslationOutput:
        """Sample only the target residual; `physical` remains deterministic."""

        physical = output.physical
        base = output.visual_base if output.visual_base is not None else physical
        descriptors = _as_batch_descriptors(
            target_descriptors,
            target_sensor,
            batch=base.shape[0],
            channels=base.shape[1],
            device=base.device,
            dtype=base.dtype,
        )
        condition = output.bridge_condition
        if condition is None:
            condition = self.adapter(base, descriptors, anchor_valid)
        origin, log_scale = self.bridge.origin_distribution(condition)
        generator = torch.Generator(device=physical.device)
        generator.manual_seed(seed)
        latent = origin + log_scale.exp() * torch.randn(
            origin.shape, device=origin.device, dtype=origin.dtype, generator=generator
        )
        total_steps = self.config.flow_steps if steps is None else steps
        if total_steps <= 0:
            raise ValueError("steps must be positive")
        dt = 1.0 / total_steps
        for index in range(total_steps):
            time = torch.full(
                (base.shape[0],), index * dt, device=base.device, dtype=base.dtype
            )
            latent = latent + dt * self.bridge(latent, time, condition)
        residual = self.residual_codec.decode(latent, descriptors, base.shape[-2:])
        residual = self._zero_block_mean(residual, self.config.texture_block_size)
        reliability = output.texture_reliability
        if reliability is None:
            reliability = torch.ones_like(base[:, :1])
        if reliability.shape != (base.shape[0], 1, *base.shape[-2:]):
            raise ValueError("texture reliability must have shape Bx1xHxW")
        # ReLU preserves exact zero release at initialization while allowing a
        # positive scalar calibration after the conditional bridge is trained.
        global_release = torch.tanh(self.visual_scale).clamp_min(0.0)
        amplitude = self.config.visual_residual_limit * global_release * reliability
        residual = residual * amplitude
        visual, visual_violation = self.bounded_anchor_update(base, residual)
        amplitude = residual.square().mean(dim=1, keepdim=True).sqrt()
        return TemporalTranslationOutput(
            physical=physical,
            log_variance=output.log_variance,
            attention=output.attention,
            observable_delta=output.observable_delta,
            deterministic_detail=output.deterministic_detail,
            detail_confidence=output.detail_confidence,
            texture_reliability=reliability,
            deterministic_visual_base=output.deterministic_visual_base,
            visual_base=base,
            bridge_condition=condition,
            pre_projection_violation=torch.maximum(
                output.pre_projection_violation
                if output.pre_projection_violation is not None
                else torch.zeros_like(visual_violation),
                visual_violation,
            ),
            visual=visual,
            stochastic_residual=residual,
            residual_amplitude=amplitude,
        )

    def forward(
        self,
        source_values: Tensor,
        source_valid: Tensor,
        anchor_values: Tensor,
        anchor_valid: Tensor,
        source_days: Tensor,
        anchor_days: Tensor,
        *,
        source_sensor: SensorSpec,
        target_sensor: SensorSpec,
        source_descriptors: Tensor | None = None,
        target_descriptors: Tensor | None = None,
    ) -> TemporalTranslationOutput:
        return self.physical(
            source_values,
            source_valid,
            anchor_values,
            anchor_valid,
            source_days,
            anchor_days,
            source_sensor=source_sensor,
            target_sensor=target_sensor,
            source_descriptors=source_descriptors,
            target_descriptors=target_descriptors,
        )

    @staticmethod
    def _zero_block_mean(values: Tensor, block_size: int) -> Tensor:
        if values.ndim != 4:
            raise ValueError("texture residual must have shape BxCxHxW")
        height, width = values.shape[-2:]
        if height % block_size or width % block_size:
            raise ValueError("texture residual dimensions must be divisible by texture_block_size")
        means = F.avg_pool2d(values, block_size, stride=block_size)
        return values - F.interpolate(means, size=(height, width), mode="nearest")


def channel_descriptors(channels: tuple[ChannelSpec, ...], device: torch.device) -> Tensor:
    """Public helper for a custom sensor adapter without registering it globally."""

    return torch.tensor([channel.descriptor() for channel in channels], device=device)


def _masked_latent_mean(values: Tensor, valid: Tensor) -> Tensor:
    if valid.ndim != 4 or valid.shape[0] != values.shape[0] or valid.shape[1] != 1:
        raise ValueError("valid mask must have shape Bx1xHxW")
    if valid.shape[-2:] != values.shape[-2:]:
        valid = F.interpolate(valid.float(), size=values.shape[-2:], mode="area").to(values)
    expanded = valid.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)
