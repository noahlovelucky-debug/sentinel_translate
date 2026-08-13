"""Deterministic sparse-observation paired-anchor transport (SOPAT V4).

This module is intentionally independent from the V3 inference surface.  It
uses a single model for both Sentinel-1 to Sentinel-2 and Sentinel-2 to
Sentinel-1 transport.  The only image prediction is a bounded, deterministic
update of a registered target anchor; there is no residual-flow or visual path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentinel_v3.model import SceneEncoder
from sentinel_v3.sensors import SensorSpec

Pyramid = tuple[Tensor, Tensor, Tensor, Tensor]


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


def _sensor_descriptors(
    sensor: SensorSpec,
    *,
    batch: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return the V3 descriptor format while rejecting non-S1/S2 channel layouts."""

    if len(sensor.channels) != channels:
        raise ValueError("sensor channel count does not match its image tensor")
    return torch.tensor(
        [channel.descriptor() for channel in sensor.channels], device=device, dtype=dtype
    ).unsqueeze(0).expand(batch, -1, -1)


def _as_batch_days(values: Tensor, batch: int, name: str) -> Tensor:
    if values.shape not in {(batch,), (batch, 1)}:
        raise ValueError(f"{name} must have shape B or Bx1")
    return values.reshape(batch)


class _SymmetricAnchorFactorizer(nn.Module):
    """Factor paired H/8 anchor tokens into common and private state.

    Cross attention is deliberately performed in both directions with shared
    projections.  At H/8 this is a local/window operation for normal patches;
    for unusually large grids tokens are partitioned into fixed windows.
    """

    def __init__(self, hidden: int, heads: int, window_size: int) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.window_size = window_size
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.0)
        self.common = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.source_cross = nn.Conv2d(hidden, hidden, 1)
        self.target_cross = nn.Conv2d(hidden, hidden, 1)

    @staticmethod
    def _window_tokens(values: Tensor, window_size: int) -> tuple[Tensor, tuple[int, int, int, int]]:
        batch, channels, height, width = values.shape
        pad_height = (-height) % window_size
        pad_width = (-width) % window_size
        padded = F.pad(values, (0, pad_width, 0, pad_height))
        padded_height, padded_width = padded.shape[-2:]
        windows = padded.reshape(
            batch,
            channels,
            padded_height // window_size,
            window_size,
            padded_width // window_size,
            window_size,
        )
        windows = windows.permute(0, 2, 4, 3, 5, 1).reshape(
            batch * (padded_height // window_size) * (padded_width // window_size),
            window_size * window_size,
            channels,
        )
        return windows, (height, width, padded_height, padded_width)

    @staticmethod
    def _unwindow_tokens(
        tokens: Tensor, geometry: tuple[int, int, int, int], batch: int, channels: int, window_size: int
    ) -> Tensor:
        height, width, padded_height, padded_width = geometry
        values = tokens.reshape(
            batch,
            padded_height // window_size,
            padded_width // window_size,
            window_size,
            window_size,
            channels,
        )
        values = values.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, padded_height, padded_width)
        return values[..., :height, :width]

    def forward(
        self, source: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if source.shape != target.shape or source.ndim != 4:
            raise ValueError("paired H/8 anchor features must have the same BCHW shape")
        batch, channels, _, _ = source.shape
        source_tokens, geometry = self._window_tokens(source, self.window_size)
        target_tokens, _ = self._window_tokens(target, self.window_size)
        source_context, _ = self.cross(source_tokens, target_tokens, target_tokens, need_weights=False)
        target_context, _ = self.cross(target_tokens, source_tokens, source_tokens, need_weights=False)
        source_cross = self._unwindow_tokens(
            source_context, geometry, batch, channels, self.window_size
        )
        target_cross = self._unwindow_tokens(
            target_context, geometry, batch, channels, self.window_size
        )
        source_cross = source + self.source_cross(source_cross)
        target_cross = target + self.target_cross(target_cross)
        common_source = self.common(torch.cat((source, source_cross), dim=1).permute(0, 2, 3, 1))
        common_target = self.common(torch.cat((target, target_cross), dim=1).permute(0, 2, 3, 1))
        common_source = common_source.permute(0, 3, 1, 2)
        common_target = common_target.permute(0, 3, 1, 2)
        private_source = source - common_source
        private_target = target - common_target
        return (
            common_source,
            common_target,
            private_source,
            private_target,
            source_cross,
            target_cross,
        )


class _ScaleSetAttention(nn.Module):
    """Point-wise masked, unordered set attention for one pyramid scale.

    This block intentionally returns its *unguarded* fused branch feature.
    SOPAT evaluates it once for a real history and once for its matched null
    history, then subtracts the two complete branches.  Applying a
    change-magnitude gate here would make the branches asymmetric and would
    let target-query/fuse biases masquerade as source evidence.
    """

    def __init__(self, channels: int, heads: int, max_horizon_days: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("feature channels must be divisible by transport_heads")
        self.max_horizon_days = max_horizon_days
        self.time = nn.Sequential(nn.Linear(4, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True, dropout=0.0)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            _ResidualBlock(channels),
        )

    def _time_features(
        self,
        observation_days: Tensor,
        source_anchor_days: Tensor,
        target_anchor_days: Tensor,
    ) -> Tensor:
        relative = (observation_days - source_anchor_days[:, None]) / float(self.max_horizon_days)
        query_age = observation_days / float(self.max_horizon_days)
        target_age = target_anchor_days[:, None] / float(self.max_horizon_days)
        phase = 2.0 * math.pi * relative
        return torch.stack(
            (query_age, target_age.expand_as(query_age), torch.sin(phase), torch.cos(phase)),
            dim=-1,
        )

    def forward(
        self,
        changes: Tensor,
        query: Tensor,
        observation_valid: Tensor,
        observation_days: Tensor,
        observation_present: Tensor,
        source_anchor_days: Tensor,
        target_anchor_days: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if changes.ndim != 5:
            raise ValueError("changes must have shape BxTxCxHxW")
        batch, frames, channels, height, width = changes.shape
        if query.shape != (batch, channels, height, width):
            raise ValueError("set-attention query must match change features")
        if observation_valid.shape[:3] != (batch, frames, 1):
            raise ValueError("observation_valid must have shape BxTx1xHxW")
        if observation_days.shape != (batch, frames):
            raise ValueError("observation_days must have shape BxT")
        if observation_present.shape != (batch, frames):
            raise ValueError("observation_present must have shape BxT")

        safe_valid = torch.where(
            observation_present[:, :, None, None, None],
            observation_valid,
            torch.zeros_like(observation_valid),
        )
        coverage = F.interpolate(
            safe_valid.reshape(batch * frames, 1, *safe_valid.shape[-2:]).float(),
            size=(height, width),
            mode="area",
        ).reshape(batch, frames, height, width)
        usable = (coverage > 0.0) & observation_present.bool()[:, :, None, None]
        # Relative time conditions attention keys only.  Values remain literal
        # E(observation)-E(source_anchor), so no temporal prior can manufacture
        # a change when every observation equals its source anchor.
        active_days = torch.where(
            observation_present,
            observation_days,
            torch.zeros_like(observation_days),
        )
        values = torch.where(
            observation_present[:, :, None, None, None], changes, torch.zeros_like(changes)
        )
        keys = values + self.time(
            self._time_features(active_days, source_anchor_days, target_anchor_days)
        )[:, :, :, None, None] * observation_present[:, :, None, None, None].to(changes)
        values = values.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, frames, channels
        )
        keys = keys.permute(0, 3, 4, 1, 2).reshape(batch * height * width, frames, channels)
        query_tokens = query.permute(0, 2, 3, 1).reshape(batch * height * width, 1, channels)
        padding = (~usable).permute(0, 2, 3, 1).reshape(batch * height * width, frames)
        empty = padding.all(dim=1)
        if bool(empty.any()):
            values = values.clone()
            keys = keys.clone()
            padding = padding.clone()
            values[empty, 0] = 0.0
            keys[empty, 0] = 0.0
            padding[empty, 0] = False
        attended, weights = self.attention(
            query_tokens,
            keys,
            values,
            key_padding_mask=padding,
            need_weights=True,
            average_attn_weights=True,
        )
        attended = attended.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        attention = weights.reshape(batch, height, width, frames).permute(0, 3, 1, 2).unsqueeze(2)
        attention = attention * usable[:, :, None].to(attention)
        support = usable.to(changes).sum(dim=1, keepdim=True) / observation_present.to(changes).sum(
            dim=1, keepdim=True
        )[:, :, None, None].clamp_min(1.0)
        # Do not multiply by a per-branch change amplitude here.  The caller
        # subtracts a real and null invocation with identical masks/times,
        # which cancels query, attention, and convolution biases before the
        # hard source-validity guard is applied.
        fused = self.fuse(torch.cat((query, attended), dim=1))
        return fused, attention, support


class _SharedTransportTrunk(nn.Module):
    """Fuse all transported scales into a shared full-resolution representation."""

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        channels = (width, width * 2, width * 4, hidden)
        self.project = nn.ModuleList(nn.Conv2d(value, width, 1) for value in channels)
        self.body = nn.Sequential(
            nn.Conv2d(width * 4, width * 2, 3, padding=1),
            _ResidualBlock(width * 2),
            nn.Conv2d(width * 2, width, 3, padding=1),
            _ResidualBlock(width),
        )

    def forward(self, values: Pyramid, output_size: tuple[int, int]) -> Tensor:
        projected = [
            F.interpolate(layer(feature), size=output_size, mode="bilinear", align_corners=False)
            for layer, feature in zip(self.project, values, strict=True)
        ]
        return self.body(torch.cat(projected, dim=1))


class _TargetRenderer(nn.Module):
    """Render pre-activation heads for one source-conditioned transport arm.

    SOPAT invokes this renderer on matched real and null transport features.
    The public candidate is ``tanh(real_logits - null_logits)``; consequently
    every renderer bias cancels before it can update the target anchor.  The
    module names and parameter shapes deliberately stay stable so an anchor
    factorizer checkpoint can initialize the contrastive physical stage.
    """

    confidence_prior_logit = -2.0

    def __init__(self, width: int, channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            _ResidualBlock(width),
        )
        self.delta = nn.Conv2d(width, channels, 1)
        self.confidence = nn.Conv2d(width, 1, 1)
        self.variance = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.confidence.weight)
        nn.init.constant_(self.confidence.bias, -2.0)
        nn.init.zeros_(self.variance.weight)
        nn.init.zeros_(self.variance.bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return pre-tanh candidate, confidence, and variance logits."""

        features = self.features(features)
        return self.delta(features), self.confidence(features), self.variance(features)


@dataclass
class SOPATConfig:
    width: int = 64
    hidden: int = 768
    encoder_depth: int = 12
    heads: int = 12
    adapter_rank: int = 64
    transport_heads: int = 4
    max_horizon_days: int = 180
    translation_tolerance_days: int = 1
    anchor_window_size: int = 8
    architecture: str = "sopat_v4"
    transport_parameterization: str = "contrastive_null_v1"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.hidden <= 0 or self.encoder_depth <= 0:
            raise ValueError("width, hidden, and encoder_depth must be positive")
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if any(channels % self.transport_heads for channels in (self.width, self.width * 2, self.width * 4, self.hidden)):
            raise ValueError("all FPN channels must be divisible by transport_heads")
        if self.adapter_rank <= 0 or self.max_horizon_days <= 0:
            raise ValueError("adapter_rank and max_horizon_days must be positive")
        if self.translation_tolerance_days < 0:
            raise ValueError("translation_tolerance_days must be non-negative")
        if self.anchor_window_size <= 0:
            raise ValueError("anchor_window_size must be positive")
        if self.transport_parameterization != "contrastive_null_v1":
            raise ValueError(
                "SOPAT requires transport_parameterization='contrastive_null_v1'"
            )


@dataclass
class SOPATOutput:
    physical: Tensor
    candidate_physical: Tensor
    candidate_logits: Tensor
    transport_confidence: Tensor
    transport_confidence_logits: Tensor
    transport_evidence: Tensor
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
    raw_delta: Tensor | None = None


@dataclass
class SOPATFactorizerOutput:
    """Anchor-only output used by the inexpensive factorizer training stage."""

    source_anchor_reconstruction: Tensor
    target_anchor_reconstruction: Tensor
    source_anchor_cross: Tensor
    target_anchor_cross: Tensor
    common_anchor: Tensor
    common_source: Tensor
    common_target: Tensor
    source_private: Tensor
    target_private: Tensor
    private_source: Tensor
    private_target: Tensor


class SOPAT(nn.Module):
    """One deterministic paired-anchor transport model for both S1/S2 directions."""

    supported_sensor_names = frozenset({"sentinel-1", "sentinel-2"})
    canonical_gsd_m = 10.0

    def __init__(self, config: SOPATConfig | None = None) -> None:
        super().__init__()
        self.config = config or SOPATConfig()
        self.encoder = SceneEncoder(
            width=self.config.width,
            hidden=self.config.hidden,
            depth=self.config.encoder_depth,
            heads=self.config.heads,
            adapter_rank=self.config.adapter_rank,
        )
        self.factorizer = _SymmetricAnchorFactorizer(
            self.config.hidden, self.config.heads, self.config.anchor_window_size
        )
        channels = (self.config.width, self.config.width * 2, self.config.width * 4, self.config.hidden)
        self.set_attention = nn.ModuleList(
            _ScaleSetAttention(value, self.config.transport_heads, self.config.max_horizon_days)
            for value in channels
        )
        self.transport = _SharedTransportTrunk(self.config.width, self.config.hidden)
        self.renderers = nn.ModuleDict(
            {
                "optical": _TargetRenderer(self.config.width, channels=10),
                "sar": _TargetRenderer(self.config.width, channels=2),
            }
        )
        self.anchor_reconstructors = nn.ModuleDict(
            {
                "optical": nn.Conv2d(self.config.hidden, 10, 1),
                "sar": nn.Conv2d(self.config.hidden, 2, 1),
            }
        )

    @staticmethod
    def bounded_anchor_update(anchor: Tensor, fraction: Tensor) -> tuple[Tensor, Tensor]:
        """Apply an anchor-room fraction without a post-hoc projection.

        ``fraction`` is clipped defensively for direct callers, while the
        renderer emits it through ``tanh``.  The resulting proposal is in the
        normalized radiometric range whenever the registered anchor is in that
        range, so a later ``clamp`` cannot hide an amplitude failure.
        """

        if anchor.shape != fraction.shape:
            raise ValueError("anchor and fraction must have the same shape")
        signed = fraction.clamp(-1.0, 1.0)
        positive_room = (1.0 - anchor).clamp_min(0.0)
        negative_room = (1.0 + anchor).clamp_min(0.0)
        bounded = anchor + torch.where(signed >= 0, positive_room * signed, negative_room * signed)
        # The proposal itself is parameterized inside the available room.  For
        # valid normalized anchors this is exactly zero, while malformed
        # out-of-range anchor input remains observable to diagnostics.
        violation = (bounded.abs() > 1.0).to(bounded).mean(dim=(1, 2, 3))
        return bounded, violation

    @staticmethod
    def _condition(
        batch: int, device: torch.device, dtype: torch.dtype, modality: str
    ) -> Tensor:
        # The shared V3 encoder consumes an 11-value conditioning vector.  Keep
        # direction information explicit and leave all unsupported metadata out.
        condition = torch.zeros(batch, 11, device=device, dtype=dtype)
        condition[:, 0] = 1.0 if modality == "optical" else -1.0
        return condition

    @classmethod
    def _validate_sensor_pair(cls, source_sensor: SensorSpec, target_sensor: SensorSpec) -> None:
        if source_sensor.name not in cls.supported_sensor_names or target_sensor.name not in cls.supported_sensor_names:
            raise ValueError("SOPAT V4 supports Sentinel-1 and Sentinel-2 only")
        if source_sensor.modality == target_sensor.modality:
            raise ValueError("SOPAT V4 requires an S1-to-S2 or S2-to-S1 sensor pair")

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
        source_sensor: SensorSpec,
        target_sensor: SensorSpec,
    ) -> tuple[int, int, int, int]:
        self._validate_sensor_pair(source_sensor, target_sensor)
        if observations.ndim != 5:
            raise ValueError("observations must have shape BxTxCxHxW")
        batch, frames, source_channels, height, width = observations.shape
        if frames < 1:
            raise ValueError("at least one observation slot is required")
        if height < 8 or width < 8 or height % 8 or width % 8:
            raise ValueError("spatial dimensions must be divisible by eight")
        if source_channels != len(source_sensor.channels):
            raise ValueError("observations do not match source_sensor channels")
        if observation_valid.shape != (batch, frames, 1, height, width):
            raise ValueError("observation_valid must have shape BxTx1xHxW")
        if observation_days.shape != (batch, frames):
            raise ValueError("observation_days must have shape BxT")
        if observation_present.shape != (batch, frames):
            raise ValueError("observation_present must have shape BxT")
        if source_anchor.shape != (batch, source_channels, height, width):
            raise ValueError("source_anchor must have shape BxCsxHxW")
        if source_anchor_valid.shape != (batch, 1, height, width):
            raise ValueError("source_anchor_valid must have shape Bx1xHxW")
        if target_anchor.ndim != 4 or target_anchor.shape[0] != batch or target_anchor.shape[-2:] != (height, width):
            raise ValueError("target_anchor must have shape BxCtxHxW on the canonical grid")
        if target_anchor.shape[1] != len(target_sensor.channels):
            raise ValueError("target_anchor does not match target_sensor channels")
        if target_anchor_valid.shape != (batch, 1, height, width):
            raise ValueError("target_anchor_valid must have shape Bx1xHxW")
        if not bool(observation_present.bool().any(dim=1).all()):
            raise ValueError("each sample requires at least one present observation")
        source_days = _as_batch_days(source_anchor_days, batch, "source_anchor_days")
        target_days = _as_batch_days(target_anchor_days, batch, "target_anchor_days")
        active_days = observation_days[observation_present.bool()]
        if bool((active_days > 0).any()):
            raise ValueError("future observations are not causal")
        if bool((active_days < -float(self.config.max_horizon_days)).any()):
            raise ValueError("observation exceeds max_horizon_days")
        if bool((source_days >= 0).any()) or bool((target_days >= 0).any()):
            raise ValueError("registered anchors must precede the target time")
        if bool((source_days < -float(self.config.max_horizon_days)).any()) or bool(
            (target_days < -float(self.config.max_horizon_days)).any()
        ):
            raise ValueError("registered anchor exceeds max_horizon_days")
        return batch, frames, height, width

    def _encode(
        self, values: Tensor, valid: Tensor, sensor: SensorSpec
    ) -> Pyramid:
        batch, channels, _, _ = values.shape
        descriptors = _sensor_descriptors(
            sensor, batch=batch, channels=channels, device=values.device, dtype=values.dtype
        )
        return self.encoder(
            values,
            descriptors[0],
            valid.to(device=values.device, dtype=values.dtype),
            self._condition(batch, values.device, values.dtype, sensor.modality),
            sensor.modality,
        )

    @staticmethod
    def _add_common_to_target(target: Pyramid, common: Tensor) -> Pyramid:
        adjusted = list(target)
        adjusted[-1] = adjusted[-1] + common
        return tuple(adjusted)  # type: ignore[return-value]

    @staticmethod
    def _anchor_reconstruction(
        common: Tensor, private: Tensor, anchor: Tensor, renderer: nn.Conv2d) -> Tensor:
        """Render a cross-factorized anchor image for the factorizer objective."""

        reconstruction = renderer(common + private)
        if reconstruction.shape[1] != anchor.shape[1]:
            raise ValueError("anchor reconstruction renderer has incompatible output channels")
        reconstruction = F.interpolate(
            reconstruction, size=anchor.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.tanh(reconstruction)

    def set_training_stage(self, stage: str) -> None:
        """Select the trainable V4 components for anchor or physical training."""

        if stage not in {"factorizer", "physical"}:
            raise ValueError("stage must be factorizer or physical")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == "factorizer":
            components: tuple[nn.Module, ...] = (
                self.encoder,
                self.factorizer,
                self.anchor_reconstructors,
            )
        else:
            components = (
                self.encoder,
                self.factorizer,
                self.set_attention,
                self.transport,
                self.renderers,
            )
        for component in components:
            for parameter in component.parameters():
                parameter.requires_grad_(True)
        self.training_stage = stage

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Delegate training-memory checkpointing to the shared scene encoder."""

        if not isinstance(enabled, bool):
            raise TypeError("activation_checkpointing must be a bool")
        self.encoder.set_activation_checkpointing(enabled)

    def factorize_anchors(
        self,
        *,
        source_anchor: Tensor,
        source_anchor_valid: Tensor,
        target_anchor: Tensor,
        target_anchor_valid: Tensor,
        source_sensor: SensorSpec,
        target_sensor: SensorSpec,
    ) -> SOPATFactorizerOutput:
        """Factor one registered pair without encoding the observation set."""

        self._validate_sensor_pair(source_sensor, target_sensor)
        if source_anchor.ndim != 4 or target_anchor.ndim != 4:
            raise ValueError("anchors must have shape BxCxHxW")
        batch, source_channels, height, width = source_anchor.shape
        if height < 8 or width < 8 or height % 8 or width % 8:
            raise ValueError("anchor spatial dimensions must be divisible by eight")
        if target_anchor.shape[0] != batch or target_anchor.shape[-2:] != (height, width):
            raise ValueError("registered anchors must share batch and spatial dimensions")
        if source_channels != len(source_sensor.channels):
            raise ValueError("source_anchor does not match source_sensor channels")
        if target_anchor.shape[1] != len(target_sensor.channels):
            raise ValueError("target_anchor does not match target_sensor channels")
        if source_anchor_valid.shape != (batch, 1, height, width):
            raise ValueError("source_anchor_valid must have shape Bx1xHxW")
        if target_anchor_valid.shape != (batch, 1, height, width):
            raise ValueError("target_anchor_valid must have shape Bx1xHxW")

        source_features = self._encode(source_anchor, source_anchor_valid, source_sensor)
        target_features = self._encode(target_anchor, target_anchor_valid, target_sensor)
        (
            common_source,
            common_target,
            source_private,
            target_private,
            _source_cross_latent,
            _target_cross_latent,
        ) = self.factorizer(source_features[-1], target_features[-1])
        source_anchor_cross = self._anchor_reconstruction(
            common_target,
            source_private,
            source_anchor,
            self.anchor_reconstructors[source_sensor.modality],
        )
        target_anchor_cross = self._anchor_reconstruction(
            common_source,
            target_private,
            target_anchor,
            self.anchor_reconstructors[target_sensor.modality],
        )
        return SOPATFactorizerOutput(
            source_anchor_reconstruction=source_anchor_cross,
            target_anchor_reconstruction=target_anchor_cross,
            source_anchor_cross=source_anchor_cross,
            target_anchor_cross=target_anchor_cross,
            common_anchor=0.5 * (common_source + common_target),
            common_source=common_source,
            common_target=common_target,
            source_private=source_private,
            target_private=target_private,
            private_source=source_private,
            private_target=target_private,
        )

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
        source_sensor: SensorSpec,
        target_sensor: SensorSpec,
    ) -> SOPATOutput:
        """Transport source-anchor-relative changes without accepting a target label."""

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
            source_sensor,
            target_sensor,
        )
        source_anchor_days = _as_batch_days(source_anchor_days, batch, "source_anchor_days").to(observations)
        target_anchor_days = _as_batch_days(target_anchor_days, batch, "target_anchor_days").to(observations)
        observation_days = observation_days.to(observations)
        observation_present = observation_present.bool()

        source_anchor_features = self._encode(source_anchor, source_anchor_valid, source_sensor)
        target_anchor_features = self._encode(target_anchor, target_anchor_valid, target_sensor)
        # Padded observation slots are ignored before entering the encoder.  In
        # particular, a slot may contain arbitrary values/days/valid masks.
        # The real and matched-null histories must use *exactly* the same
        # source-valid support.  The null history replaces every present
        # observation by the registered source anchor; it retains its original
        # timestamps and slots so all learned time/query/fuse biases cancel
        # only through a true source-conditioned contrast.
        present_image = observation_present[:, :, None, None, None]
        safe_observations = torch.where(present_image, observations, torch.zeros_like(observations))
        shared_observation_valid = torch.where(
            present_image,
            observation_valid * source_anchor_valid[:, None].to(observation_valid),
            torch.zeros_like(observation_valid),
        )
        null_observations = torch.where(
            present_image,
            source_anchor[:, None].expand_as(observations),
            torch.zeros_like(observations),
        )
        # This is a structural guard, never a learned transport value.  It is
        # deliberately derived in pixel space before encoding, then combined
        # with the shared valid support below.  Thus a source-invalid or
        # source-identical history can never manufacture an update through a
        # temporal/query/fuse bias.
        raw_change_evidence = (
            (safe_observations - source_anchor[:, None]).abs().mean(dim=2, keepdim=True)
            * shared_observation_valid.to(observations)
        ).sum(dim=1)
        shared_valid = shared_observation_valid.sum(dim=1) > 0.0
        raw_change_gate = (raw_change_evidence > 0.0) & shared_valid
        # Encode matched real/null source histories in one shared call.  This
        # keeps their encoder arithmetic identical when observation == anchor
        # while avoiding an anchor-copy bypass around set attention.
        paired_observations = torch.cat((safe_observations, null_observations), dim=0)
        paired_valid = torch.cat((shared_observation_valid, shared_observation_valid), dim=0)
        flattened_observations = paired_observations.reshape(
            2 * batch * frames, observations.shape[2], height, width
        )
        flattened_valid = paired_valid.reshape(2 * batch * frames, 1, height, width)
        paired_observation_features = self._encode(
            flattened_observations, flattened_valid, source_sensor
        )
        real_observation_features: list[Tensor] = []
        null_observation_features: list[Tensor] = []
        for level in paired_observation_features:
            paired_level = level.reshape(
                2, batch, frames, level.shape[1], level.shape[2], level.shape[3]
            )
            real_observation_features.append(paired_level[0])
            null_observation_features.append(paired_level[1])

        (
            common_source,
            common_target,
            source_private,
            target_private,
            _source_anchor_cross_latent,
            _target_anchor_cross_latent,
        ) = self.factorizer(source_anchor_features[-1], target_anchor_features[-1])
        common_anchor = 0.5 * (common_source + common_target)
        source_anchor_cross = self._anchor_reconstruction(
            common_target,
            source_private,
            source_anchor,
            self.anchor_reconstructors[source_sensor.modality],
        )
        target_anchor_cross = self._anchor_reconstruction(
            common_source,
            target_private,
            target_anchor,
            self.anchor_reconstructors[target_sensor.modality],
        )
        # The public cross fields are image-space reconstructions for the
        # factorizer objective.  Keep H/8 cross-attended features internal so
        # callers cannot accidentally compare latent channels with imagery.
        source_anchor_reconstruction = source_anchor_cross
        target_anchor_reconstruction = target_anchor_cross

        target_queries = self._add_common_to_target(target_anchor_features, common_anchor)
        real_transport_features: list[Tensor] = []
        null_transport_features: list[Tensor] = []
        transported: list[Tensor] = []
        attention: list[Tensor] = []
        support: list[Tensor] = []
        for real_feature, null_feature, anchor_feature, query, set_attention in zip(
            real_observation_features,
            null_observation_features,
            source_anchor_features,
            target_queries,
            self.set_attention,
            strict=True,
        ):
            # The two branches receive the same anchor, query, validity,
            # timestamps and slot mask.  Differencing their complete fused
            # outputs cancels all non-source biases without weakening the
            # source-dependent set-attention route.
            real_change = real_feature - anchor_feature[:, None]
            null_change = null_feature - anchor_feature[:, None]
            real_fused, scale_attention, scale_support = set_attention(
                real_change,
                query,
                shared_observation_valid,
                observation_days,
                observation_present,
                source_anchor_days,
                target_anchor_days,
            )
            null_fused, _null_attention, _null_support = set_attention(
                null_change,
                query,
                shared_observation_valid,
                observation_days,
                observation_present,
                source_anchor_days,
                target_anchor_days,
            )
            real_transport_features.append(real_fused)
            null_transport_features.append(null_fused)
            transported.append(real_fused - null_fused)
            attention.append(scale_attention)
            support.append(scale_support)

        transported_change: Pyramid = tuple(transported)  # type: ignore[assignment]
        # Do not feed ``real_fused - null_fused`` through one trunk: its
        # convolution bias would reintroduce a null update.  Complete both
        # branches through the shared trunk and renderer, then difference the
        # pre-activation heads so every internal bias cancels.
        real_transport = self.transport(tuple(real_transport_features), (height, width))
        null_transport = self.transport(tuple(null_transport_features), (height, width))
        renderer = self.renderers[target_sensor.modality]
        real_candidate_logits, real_confidence_logits, real_variance_logits = renderer(real_transport)
        null_candidate_logits, null_confidence_logits, null_variance_logits = renderer(null_transport)
        candidate_logits = real_candidate_logits - null_candidate_logits
        confidence_logits = (
            torch.full_like(real_confidence_logits, renderer.confidence_prior_logit)
            + real_confidence_logits
            - null_confidence_logits
        )
        log_variance = (real_variance_logits - null_variance_logits).clamp(-8.0, 4.0)
        # Structural source evidence preserves exact identity for null source
        # change.  The learned confidence is separate: it decides whether a
        # non-null, transported candidate is useful for this target location.
        source_evidence = raw_change_gate.to(candidate_logits)
        candidate_fraction = torch.tanh(candidate_logits) * source_evidence
        transport_confidence = torch.sigmoid(confidence_logits) * source_evidence
        candidate_physical, _candidate_violation = self.bounded_anchor_update(
            target_anchor, candidate_fraction
        )
        physical_fraction = candidate_fraction * transport_confidence
        physical, pre_projection_violation = self.bounded_anchor_update(target_anchor, physical_fraction)
        raw_delta = physical - target_anchor
        latest_days = torch.where(
            observation_present,
            observation_days,
            torch.full_like(observation_days, -float("inf")),
        ).max(dim=1).values
        task_is_translation = latest_days.abs() <= float(self.config.translation_tolerance_days)
        return SOPATOutput(
            physical=physical,
            candidate_physical=candidate_physical,
            candidate_logits=candidate_logits,
            transport_confidence=transport_confidence,
            # Keep the learned logit independent of structural evidence.  A
            # caller can inspect calibration separately from the hard causal
            # guard, and training can exclude no-evidence pixels rather than
            # teaching their arbitrary logits toward a neutral probability.
            transport_confidence_logits=confidence_logits,
            transport_evidence=source_evidence,
            log_variance=log_variance,
            transported_change=transported_change,
            common_anchor=common_anchor,
            source_private=source_private,
            target_private=target_private,
            attention=tuple(attention),  # type: ignore[arg-type]
            observation_support=tuple(support),  # type: ignore[arg-type]
            task_is_translation=task_is_translation,
            pre_projection_violation=pre_projection_violation,
            source_anchor_reconstruction=source_anchor_reconstruction,
            target_anchor_reconstruction=target_anchor_reconstruction,
            source_anchor_cross=source_anchor_cross,
            target_anchor_cross=target_anchor_cross,
            common_source=common_source,
            common_target=common_target,
            private_source=source_private,
            private_target=target_private,
            raw_delta=raw_delta,
        )
