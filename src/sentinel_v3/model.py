from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from .api import Observation, TargetRequest, TranslationResult
from .losses import (
    frequency_bands,
    haar_dwt2,
    haar_packet_dwt2,
    haar_packet_idwt2,
    highpass,
)
from .physics import gsd_condition
from .sensors import SENTINEL1, SENTINEL2, ChannelSpec, SensorSpec
from .temporal_prior import TemporalPriorConfig, TemporalPriorStore

Pyramid = tuple[Tensor, Tensor, Tensor, Tensor]


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return (values + self.block(values)) / math.sqrt(2.0)


class ChannelProjector(nn.Module):
    descriptor_dim = 8

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(self.descriptor_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.quadratic_embedding = nn.Sequential(
            nn.Linear(self.descriptor_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.detail_embedding = nn.Sequential(
            nn.Linear(self.descriptor_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(self.descriptor_dim + 2, max(4, width // 2)),
            nn.SiLU(),
            nn.Linear(max(4, width // 2), 1),
        )
        self.bias = nn.Sequential(nn.Linear(self.descriptor_dim, width), nn.Tanh())
        self.spatial = nn.Conv2d(width + 1, width, 3, padding=1)
        self.refine = ResidualBlock(width)
        self.refine_gate = nn.Parameter(torch.zeros(()))
        for layer in (
            self.quadratic_embedding[-1],
            self.detail_embedding[-1],
            self.channel_gate[-1],
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, values: Tensor, descriptors: Tensor, valid: Tensor) -> Tensor:
        if values.ndim != 4 or descriptors.ndim not in (2, 3):
            raise ValueError("values must be BCHW and descriptors must be CD or BCD")
        if descriptors.ndim == 2:
            descriptors = descriptors.unsqueeze(0).expand(values.shape[0], -1, -1)
        if descriptors.shape[:2] != values.shape[:2]:
            raise ValueError("one channel descriptor is required for each input channel")
        statistics = torch.stack((values.mean(dim=(-2, -1)), values.std(dim=(-2, -1))), dim=-1)
        gates = (
            torch.softmax(
                self.channel_gate(torch.cat((descriptors, statistics), dim=-1)), dim=1
            )
            * values.shape[1]
        )
        gated = values * gates.unsqueeze(-1)
        local_mean = F.avg_pool2d(values, 5, stride=1, padding=2)
        projected = torch.einsum("bchw,bcf->bfhw", gated, self.embedding(descriptors))
        projected += torch.einsum(
            "bchw,bcf->bfhw",
            gated.square() * gated.sign(),
            self.quadratic_embedding(descriptors),
        )
        projected += torch.einsum(
            "bchw,bcf->bfhw",
            (values - local_mean) * gates.unsqueeze(-1),
            self.detail_embedding(descriptors),
        )
        projected += self.bias(descriptors).mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        projected /= math.sqrt(max(1, values.shape[1]))
        projected = self.spatial(torch.cat((projected, valid), dim=1))
        return projected + torch.tanh(self.refine_gate) * self.refine(projected)


class ConditionMoE(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, experts: int = 4) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.SiLU(),
                    nn.Linear(output_dim, output_dim),
                )
                for _ in range(experts)
            ]
        )
        self.gate = nn.Linear(input_dim, experts)

    def forward(self, values: Tensor) -> Tensor:
        mixture = torch.softmax(self.gate(values), dim=-1)
        outputs = torch.stack([expert(values) for expert in self.experts], dim=1)
        return torch.einsum("be,bed->bd", mixture, outputs)


class LowRankResidualAdapter(nn.Module):
    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        rank = min(rank, hidden)
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        # The zero-initialized up projection preserves the imported checkpoint.
        # A non-zero gate is required so the up projection receives a gradient.
        self.scale = nn.Parameter(torch.tensor(math.atanh(0.1)))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, values: Tensor) -> Tensor:
        return values + torch.tanh(self.scale) * self.up(F.silu(self.down(values)))


class SceneEncoder(nn.Module):
    adapter_layers = (3, 6, 9, 12)

    def __init__(
        self,
        width: int = 64,
        hidden: int = 768,
        depth: int = 12,
        heads: int = 12,
        adapter_rank: int = 64,
    ) -> None:
        super().__init__()
        self.projector = ChannelProjector(width)
        self.level1 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), ResidualBlock(width * 2)
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1),
            ResidualBlock(width * 4),
        )
        self.level3 = nn.Sequential(
            nn.Conv2d(width * 4, hidden, 3, stride=2, padding=1), ResidualBlock(hidden)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.condition = ConditionMoE(11, hidden)
        self.adapters = nn.ModuleDict()
        for layer_number in self.adapter_layers:
            if layer_number <= depth:
                self.adapters[str(layer_number)] = nn.ModuleDict(
                    {
                        "optical": LowRankResidualAdapter(hidden, adapter_rank),
                        "sar": LowRankResidualAdapter(hidden, adapter_rank),
                    }
                )
        self.modality_adapter = nn.ModuleDict(
            {"optical": nn.Conv2d(hidden, hidden, 1), "sar": nn.Conv2d(hidden, hidden, 1)}
        )
        for adapter in self.modality_adapter.values():
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    @staticmethod
    def _position(height: int, width: int, hidden: int, device: torch.device) -> Tensor:
        if hidden % 4:
            raise ValueError("hidden dimension must be divisible by four")
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device),
            torch.linspace(-1, 1, width, device=device),
            indexing="ij",
        )
        frequencies = torch.exp(torch.linspace(0, math.log(1000.0), hidden // 4, device=device))
        position = torch.cat(
            (
                torch.sin(x[..., None] * frequencies),
                torch.cos(x[..., None] * frequencies),
                torch.sin(y[..., None] * frequencies),
                torch.cos(y[..., None] * frequencies),
            ),
            dim=-1,
        )
        return position.reshape(1, height * width, hidden)

    def forward(
        self,
        values: Tensor,
        descriptors: Tensor,
        valid: Tensor,
        condition: Tensor,
        modality: str,
    ) -> Pyramid:
        full = self.projector(values, descriptors, valid)
        half = self.level1(full)
        quarter = self.level2(half)
        eighth = self.level3(quarter)
        batch, channels, height, width = eighth.shape
        tokens = eighth.flatten(2).transpose(1, 2)
        tokens = tokens + self._position(height, width, channels, values.device).to(
            tokens.dtype
        )
        tokens = tokens + self.condition(condition).unsqueeze(1)
        for layer_number, layer in enumerate(self.transformer.layers, start=1):
            tokens = layer(tokens)
            if str(layer_number) in self.adapters:
                tokens = self.adapters[str(layer_number)][modality](tokens)
        if self.transformer.norm is not None:
            tokens = self.transformer.norm(tokens)
        shared = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        shared = shared + self.modality_adapter[modality](eighth)
        return full, half, quarter, shared


class DynamicPhysicalDecoder(nn.Module):
    def __init__(self, width: int = 64, hidden: int = 768) -> None:
        super().__init__()
        self.up2 = nn.Conv2d(hidden, width * 4, 3, padding=1)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(width * 8, width * 4, 3, padding=1), ResidualBlock(width * 4)
        )
        self.up1 = nn.Conv2d(width * 4, width * 2, 3, padding=1)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(width * 4, width * 2, 3, padding=1), ResidualBlock(width * 2)
        )
        self.full = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1), ResidualBlock(width)
        )
        self.full_resolution_fusion = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Conv2d(width * 2, width, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(width, width, 3, padding=1),
                )
                for modality in ("optical", "sar")
            }
        )
        self.kernel = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width + 1))
        self.log_variance_kernel = nn.Sequential(
            nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width + 1)
        )
        self.gsd_modulation = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width * 2)
        )
        self.radiometry = nn.ModuleDict(
            {
                "optical": nn.Sequential(nn.Conv2d(width, width, 1), nn.SiLU()),
                "sar": nn.Sequential(nn.Conv2d(width, width, 1), nn.SiLU()),
            }
        )
        self.radiometric_gate = nn.ParameterDict(
            {"optical": nn.Parameter(torch.zeros(())), "sar": nn.Parameter(torch.zeros(()))}
        )
        self.radiometric_kernel = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width + 1)
                )
                for modality in ("optical", "sar")
            }
        )
        self.radiometric_condition = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(11, width), nn.SiLU(), nn.Linear(width, width)
                )
                for modality in ("optical", "sar")
            }
        )
        self.radiometric_descriptor = nn.ModuleDict(
            {modality: nn.Linear(8, width) for modality in ("optical", "sar")}
        )
        self.radiometric_bias = nn.ModuleDict(
            {modality: nn.Linear(width, 1) for modality in ("optical", "sar")}
        )
        self.optical_direction_kernel = nn.Sequential(
            nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width + 1)
        )
        self.optical_amplitude_head = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.SiLU(), nn.Conv2d(width, 1, 1)
        )
        self.sar_spatial_kernel = nn.Sequential(
            nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width + 1)
        )
        self.sar_mean_condition = nn.Sequential(
            nn.Linear(11, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.sar_mean_descriptor = nn.Linear(8, width)
        self.sar_mean_head = nn.Linear(width, 1)
        for kernel in self.radiometric_kernel.values():
            nn.init.zeros_(kernel[-1].weight)
            nn.init.zeros_(kernel[-1].bias)
        for head in self.radiometric_bias.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for fusion in self.full_resolution_fusion.values():
            nn.init.zeros_(fusion[-1].weight)
            nn.init.zeros_(fusion[-1].bias)
        for head in (
            self.optical_direction_kernel[-1],
            self.optical_amplitude_head[-1],
            self.sar_spatial_kernel[-1],
            self.sar_mean_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.gsd_modulation[-1].weight)
        nn.init.zeros_(self.gsd_modulation[-1].bias)

    @staticmethod
    def _dynamic(features: Tensor, parameters: Tensor) -> Tensor:
        weights, bias = parameters[..., :-1], parameters[..., -1]
        return torch.einsum("bfhw,of->bohw", features, weights) + bias.view(1, -1, 1, 1)

    @staticmethod
    def _optical_factorization(
        base_logits: Tensor, direction_delta: Tensor, amplitude_delta: Tensor
    ) -> Tensor:
        direction_delta = direction_delta - direction_delta.mean(dim=1, keepdim=True)
        corrected_logits = base_logits.float()
        corrected_logits = corrected_logits + 0.5 * torch.tanh(direction_delta.float())
        corrected_logits = corrected_logits + 0.5 * torch.tanh(amplitude_delta.float())
        return torch.sigmoid(corrected_logits).to(base_logits.dtype)

    @staticmethod
    def _sar_factorization(
        base: Tensor,
        spatial_delta_db: Tensor,
        scene_delta_db: Tensor,
        valid: Tensor,
    ) -> Tensor:
        denominator = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        spatial_mean = (spatial_delta_db * valid).sum(dim=(-2, -1), keepdim=True)
        spatial_delta_db = spatial_delta_db - spatial_mean / denominator
        correction_db = spatial_delta_db + scene_delta_db[:, :, None, None]
        corrected = base.float() + correction_db.float()
        return (-20.0 + 25.0 * torch.tanh(corrected / 25.0)).to(base.dtype)

    def forward(
        self,
        pyramid: Pyramid,
        target_descriptors: Tensor,
        modality: str,
        output_size: tuple[int, int],
        scale_condition: Tensor | None = None,
        scene_condition: Tensor | None = None,
        valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        full, half, quarter, shared = pyramid
        decoded = F.interpolate(
            self.up2(shared), size=quarter.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.fuse2(torch.cat((decoded, quarter), dim=1))
        decoded = F.interpolate(
            self.up1(decoded), size=half.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.fuse1(torch.cat((decoded, half), dim=1))
        decoded = F.interpolate(decoded, size=output_size, mode="bilinear", align_corners=False)
        features = self.full(decoded)
        features = features + self.full_resolution_fusion[modality](
            torch.cat((features, full), dim=1)
        )
        if scale_condition is not None:
            shift, scale = self.gsd_modulation(scale_condition.float()).chunk(2, dim=-1)
            features = features * (1 + 0.1 * torch.tanh(scale)[:, :, None, None])
            features = features + 0.1 * shift[:, :, None, None]
        correction = self.radiometry[modality](features)
        features = features + torch.tanh(self.radiometric_gate[modality]) * correction
        base = self._dynamic(features, self.kernel(target_descriptors))
        radiometric_delta = self._dynamic(
            correction, self.radiometric_kernel[modality](target_descriptors)
        )
        if scene_condition is not None:
            conditioned = self.radiometric_condition[modality](scene_condition.float())[:, None]
            described = self.radiometric_descriptor[modality](target_descriptors)[None]
            global_delta = self.radiometric_bias[modality](
                F.silu(conditioned + described)
            ).squeeze(-1)
            radiometric_delta = radiometric_delta + global_delta[:, :, None, None]
        correction_limit = 2.0 if modality == "optical" else 5.0
        base = base + correction_limit * torch.tanh(radiometric_delta)
        log_variance = self._dynamic(
            features, self.log_variance_kernel(target_descriptors)
        ).clamp(-8.0, 3.0)
        if modality == "optical":
            direction_delta = self._dynamic(
                features, self.optical_direction_kernel(target_descriptors)
            )
            amplitude_delta = self.optical_amplitude_head(features)
            mean = self._optical_factorization(base, direction_delta, amplitude_delta)
        else:
            spatial_delta_db = 4.0 * torch.tanh(
                self._dynamic(features, self.sar_spatial_kernel(target_descriptors)) / 4.0
            )
            if scene_condition is None:
                scene_delta_db = base.new_zeros(base.shape[:2])
            else:
                conditioned = self.sar_mean_condition(scene_condition.float())[:, None]
                described = self.sar_mean_descriptor(target_descriptors)[None]
                scene_delta_db = 5.0 * torch.tanh(
                    self.sar_mean_head(F.silu(conditioned + described)).squeeze(-1) / 5.0
                )
            if valid is None:
                valid = base.new_ones(base.shape[0], 1, *base.shape[-2:])
            mean = self._sar_factorization(base, spatial_delta_db, scene_delta_db, valid)
        return mean, log_variance


class MultiscaleDetailHead(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.input_heads = nn.ModuleDict(
            {
                "optical": nn.Conv2d(width, width, 3, padding=1),
                "sar": nn.Conv2d(width, width, 3, padding=1),
            }
        )
        self.base_heads = nn.ModuleDict(
            {
                "optical": nn.Conv2d(3, width, 3, padding=1),
                "sar": nn.Conv2d(2, width, 3, padding=1),
            }
        )
        self.projections = nn.ModuleList(
            [
                nn.Conv2d(width * 2, width, 1),
                nn.Conv2d(width * 4, width, 1),
                nn.Conv2d(hidden, width, 1),
            ]
        )
        self.trunk = nn.Sequential(
            nn.Conv2d(width * 4, width * 2, 3, padding=1),
            ResidualBlock(width * 2),
            nn.Conv2d(width * 2, width, 3, padding=1),
            ResidualBlock(width),
        )
        self.output_heads = nn.ModuleDict(
            {
                "optical": nn.Conv2d(width, 3, 3, padding=1),
                "sar": nn.Conv2d(width, 2, 3, padding=1),
            }
        )
        self.confidence_heads = nn.ModuleDict(
            {
                "optical": nn.Conv2d(width, 3, 3, padding=1),
                "sar": nn.Conv2d(width, 3, 3, padding=1),
            }
        )
        for head in self.output_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head in self.base_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head in self.confidence_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -0.5)

    def forward_with_confidence(
        self,
        pyramid: Pyramid,
        source_modality: str,
        target_modality: str,
        output_size: tuple[int, int],
        base: Tensor | None = None,
        confidence_threshold: Tensor | None = None,
        *,
        hard_gate: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, ...], Tensor]:
        full, half, quarter, eighth = pyramid
        levels = [self.input_heads[source_modality](full)]
        if base is not None:
            normalized_base = base if target_modality == "optical" else (base + 20.0) / 20.0
            base_features = self.base_heads[target_modality](normalized_base)
            if base_features.shape[-2:] != full.shape[-2:]:
                base_features = F.interpolate(
                    base_features, size=full.shape[-2:], mode="bilinear", align_corners=False
                )
            levels[0] = levels[0] + base_features
        for level, projection in zip((half, quarter, eighth), self.projections, strict=True):
            projected = projection(level)
            levels.append(
                F.interpolate(
                    projected, size=full.shape[-2:], mode="bilinear", align_corners=False
                )
            )
        features = self.trunk(torch.cat(levels, dim=1))
        raw = self.output_heads[target_modality](features)
        raw = F.interpolate(raw, size=output_size, mode="bilinear", align_corners=False)
        limit = 0.08 if target_modality == "optical" else 4.0
        bands = frequency_bands(limit * torch.tanh(raw / limit), levels=3)
        confidence = torch.sigmoid(self.confidence_heads[target_modality](features))
        confidence = F.interpolate(
            confidence, size=(output_size[0] // 4, output_size[1] // 4), mode="area"
        )
        if self.training and not hard_gate:
            gate = confidence
        else:
            threshold = (
                confidence.new_tensor(0.55)
                if confidence_threshold is None
                else confidence_threshold
            )
            gate = (confidence >= threshold.to(confidence)).to(confidence.dtype)
        full_confidence = F.interpolate(gate, size=output_size, mode="nearest")
        detail = sum(
            band * full_confidence[:, index : index + 1] for index, band in enumerate(bands)
        )
        return highpass(detail), bands, confidence

    def forward(
        self,
        pyramid: Pyramid,
        source_modality: str,
        target_modality: str,
        output_size: tuple[int, int],
        base: Tensor | None = None,
    ) -> Tensor:
        return self.forward_with_confidence(
            pyramid, source_modality, target_modality, output_size, base
        )[0]


class SpatialFrequencyAdapter(nn.Module):
    """Zero-gated spatial/frequency correction for the multiscale DiT condition."""

    def __init__(self, hidden: int, bottleneck: int = 32) -> None:
        super().__init__()
        self.down = nn.Conv2d(hidden, bottleneck, 1)
        self.spatial = nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck)
        self.up = nn.Conv2d(bottleneck, hidden, 1)
        self.low_gain = nn.Parameter(torch.zeros(bottleneck))
        self.high_gain = nn.Parameter(torch.zeros(bottleneck))
        self.output_gate = nn.Parameter(torch.zeros(()))

    def forward(self, values: Tensor) -> Tensor:
        reduced = F.silu(self.down(values))
        spectrum = torch.fft.rfft2(reduced.float(), norm="ortho")
        height, width = reduced.shape[-2], reduced.shape[-1]
        fy = torch.fft.fftfreq(height, device=values.device).abs()[:, None]
        fx = torch.fft.rfftfreq(width, device=values.device).abs()[None, :]
        radius = torch.sqrt(fy.square() + fx.square()).clamp(0.0, 0.5) * 2.0
        gain = (
            self.low_gain[:, None, None] * (1.0 - radius)
            + self.high_gain[:, None, None] * radius
        )
        frequency = torch.fft.irfft2(
            spectrum * (1.0 + torch.tanh(gain)[None]), s=(height, width), norm="ortho"
        ).to(values.dtype)
        correction = self.up(self.spatial(reduced) + frequency - reduced)
        return values + torch.tanh(self.output_gate) * correction


class ResidualCodec(nn.Module):
    """Modality-specific I/O heads with a shared 4x, 16-channel residual trunk."""

    version = "local-residual-codec-v1"

    def __init__(self, width: int = 64, latent_channels: int = 16) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.input_heads = nn.ModuleDict(
            {
                "optical": nn.Conv2d(3, width, 3, padding=1),
                "sar": nn.Conv2d(2, width, 3, padding=1),
            }
        )
        self.encoder = nn.Sequential(
            ResidualBlock(width),
            nn.Conv2d(width, width * 2, 4, stride=2, padding=1),
            ResidualBlock(width * 2),
            nn.Conv2d(width * 2, latent_channels, 4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, width * 2, 4, stride=2, padding=1),
            ResidualBlock(width * 2),
            nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1),
            ResidualBlock(width),
        )
        self.output_heads = nn.ModuleDict(
            {
                "optical": nn.Conv2d(width, 3, 3, padding=1),
                "sar": nn.Conv2d(width, 2, 3, padding=1),
            }
        )
        for modality in ("optical", "sar"):
            self.register_buffer(f"{modality}_mean", torch.zeros(latent_channels))
            self.register_buffer(f"{modality}_std", torch.ones(latent_channels))

    def statistics(self, modality: str) -> tuple[Tensor, Tensor]:
        return getattr(self, f"{modality}_mean"), getattr(self, f"{modality}_std")

    def normalize(self, latent: Tensor, modality: str) -> Tensor:
        mean, std = self.statistics(modality)
        return (latent - mean[None, :, None, None]) / std[None, :, None, None]

    def denormalize(self, latent: Tensor, modality: str) -> Tensor:
        mean, std = self.statistics(modality)
        return latent * std[None, :, None, None] + mean[None, :, None, None]

    @torch.no_grad()
    def set_statistics(self, modality: str, mean: Tensor, std: Tensor) -> None:
        if mean.shape != (self.latent_channels,) or std.shape != (self.latent_channels,):
            raise ValueError("codec statistics must have one value per latent channel")
        getattr(self, f"{modality}_mean").copy_(mean)
        getattr(self, f"{modality}_std").copy_(std.clamp_min(1e-4))

    @torch.no_grad()
    def update_statistics(
        self,
        latent: Tensor,
        modality: str,
        momentum: float = 0.01,
        *,
        synchronize: bool = True,
    ) -> None:
        working = latent.float()
        total = working.sum(dim=(0, 2, 3))
        total_square = working.square().sum(dim=(0, 2, 3))
        count = working.new_tensor(working.shape[0] * working.shape[2] * working.shape[3])
        if synchronize and dist.is_available() and dist.is_initialized():
            dist.all_reduce(total)
            dist.all_reduce(total_square)
            dist.all_reduce(count)
        mean = total / count.clamp_min(1)
        variance = total_square / count.clamp_min(1) - mean.square()
        std = variance.clamp_min(1e-8).sqrt()
        running_mean, running_std = self.statistics(modality)
        running_mean.lerp_(mean, momentum)
        running_std.lerp_(std, momentum)

    def encode(self, values: Tensor, modality: str, *, standardized: bool = True) -> Tensor:
        latent = self.encoder(self.input_heads[modality](values))
        if standardized:
            latent = self.normalize(latent, modality)
        return latent

    def decode(self, latent: Tensor, modality: str, *, standardized: bool = True) -> Tensor:
        if standardized:
            latent = self.denormalize(latent, modality)
        return self.output_heads[modality](self.decoder(latent))


def _time_embedding(time: Tensor, dimension: int) -> Tensor:
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, dimension // 2, device=time.device)
        / max(1, dimension // 2 - 1)
    )
    values = time[:, None] * frequencies[None] * 1000.0
    return torch.cat((values.sin(), values.cos()), dim=-1)


class FlowBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden * 4))

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        shift1, scale1, shift2, scale2 = self.modulation(condition).chunk(4, dim=-1)
        normalized = self.norm1(values) * (1 + scale1[:, None]) + shift1[:, None]
        values = (
            values + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        )
        normalized = self.norm2(values) * (1 + scale2[:, None]) + shift2[:, None]
        return values + self.mlp(normalized)


class IdentifiabilityBridgeOrigin(nn.Module):
    """Predict a residual-flow origin from scene evidence and physical detail."""

    def __init__(
        self,
        pyramid_channels: Sequence[int],
        latent_channels: int = 16,
        hidden: int = 512,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.pyramid_projections = nn.ModuleList(
            [nn.Conv2d(channels, hidden, 1) for channels in pyramid_channels]
        )
        # Two Haar levels contribute three high-pass bands per source channel at each level.
        self.physical_heads = nn.ModuleDict(
            {
                "optical": nn.Conv2d(3 * 3 * 2, hidden, 1),
                "sar": nn.Conv2d(2 * 3 * 2, hidden, 1),
            }
        )
        self.trunk = nn.Sequential(ResidualBlock(hidden), ResidualBlock(hidden))
        output_channels = 2 * latent_channels + 3
        self.output_heads = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.GroupNorm(_groups(hidden), hidden),
                    nn.SiLU(),
                    nn.Conv2d(hidden, output_channels, 1),
                )
                for modality in ("optical", "sar")
            }
        )
        for head in self.output_heads.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _target_modality(target: str | SensorSpec) -> str:
        modality = target if isinstance(target, str) else target.modality
        if modality not in {"optical", "sar"}:
            raise ValueError("id bridge target modality must be optical or sar")
        return modality

    @staticmethod
    def _physical_highpass(physical: Tensor, latent_size: tuple[int, int]) -> Tensor:
        first = haar_dwt2(physical)
        second = haar_dwt2(first[:, :, 0])
        batch = physical.shape[0]
        first_high = first[:, :, 1:].reshape(batch, -1, *first.shape[-2:])
        second_high = second[:, :, 1:].reshape(batch, -1, *second.shape[-2:])
        first_high = F.interpolate(
            first_high, size=latent_size, mode="bilinear", align_corners=False
        )
        second_high = F.interpolate(
            second_high, size=latent_size, mode="bilinear", align_corners=False
        )
        return torch.cat((first_high, second_high), dim=1)

    def forward(
        self,
        pyramid: Pyramid,
        physical: Tensor,
        target: str | SensorSpec,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if physical.ndim != 4:
            raise ValueError("id bridge physical base must be BCHW")
        modality = self._target_modality(target)
        expected_channels = 3 if modality == "optical" else 2
        if physical.shape[1] != expected_channels:
            raise ValueError(f"{modality} id bridge expects {expected_channels} physical channels")
        if len(pyramid) != len(self.pyramid_projections):
            raise ValueError("id bridge requires the complete encoder pyramid")
        latent_size = (physical.shape[-2] // 4, physical.shape[-1] // 4)
        physical_features = self.physical_heads[modality](
            self._physical_highpass(physical, latent_size)
        )
        scene = sum(
            projection(
                F.interpolate(level, size=latent_size, mode="bilinear", align_corners=False)
            )
            for level, projection in zip(pyramid, self.pyramid_projections, strict=True)
        )
        output = self.output_heads[modality](self.trunk(scene + physical_features))
        mu, log_sigma, reliability_logits = output.split(
            (self.latent_channels, self.latent_channels, 3), dim=1
        )
        return mu, log_sigma, reliability_logits


class ResidualDiT(nn.Module):
    def __init__(
        self,
        pyramid_channels: Sequence[int],
        latent_channels: int = 16,
        hidden: int = 512,
        depth: int = 8,
        heads: int = 8,
        zero_output: bool = False,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.input = nn.Linear(latent_channels, hidden)
        self.scene_projections = nn.ModuleList(
            [nn.Conv2d(channels, hidden, 1) for channels in pyramid_channels]
        )
        self.legacy_scene = nn.Conv2d(pyramid_channels[-1], hidden, 1)
        self.condition_gates = nn.Parameter(torch.zeros(depth, 4))
        self.frequency_adapter = SpatialFrequencyAdapter(hidden)
        self.time = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.target = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([FlowBlock(hidden, heads) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, latent_channels))
        self.amplitude_head = nn.Conv2d(hidden, 3, 1)
        self.texture_risk_candidate = nn.Sequential(
            nn.Conv2d(4, hidden // 4, 1),
            nn.SiLU(),
            nn.Conv2d(hidden // 4, hidden, 1),
        )
        self.texture_risk_head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 4, 1, 1),
        )
        self.origin_projection = nn.Conv2d(latent_channels, hidden, 1)
        self.optical_bridge_anchor = nn.Conv2d(latent_channels, hidden, 1)
        self.optical_bridge_adapters = nn.ModuleList(
            [LowRankResidualAdapter(hidden, min(64, hidden)) for _ in range(depth)]
        )
        self.optical_bridge_output = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, latent_channels)
        )
        self.optical_bridge_base_gate = nn.Parameter(torch.tensor(-4.0))
        self.optical_bridge_amplitude_head = nn.Conv2d(hidden, 3, 1)
        nn.init.zeros_(self.amplitude_head.weight)
        nn.init.constant_(self.amplitude_head.bias, math.log(0.1 / 0.9))
        nn.init.zeros_(self.texture_risk_head[-1].weight)
        nn.init.constant_(self.texture_risk_head[-1].bias, -2.2)
        nn.init.zeros_(self.origin_projection.weight)
        nn.init.zeros_(self.origin_projection.bias)
        if zero_output:
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)
        nn.init.zeros_(self.optical_bridge_anchor.weight)
        nn.init.zeros_(self.optical_bridge_anchor.bias)
        nn.init.zeros_(self.optical_bridge_output[-1].weight)
        nn.init.zeros_(self.optical_bridge_output[-1].bias)
        nn.init.zeros_(self.optical_bridge_amplitude_head.weight)
        nn.init.constant_(self.optical_bridge_amplitude_head.bias, math.log(0.02 / 0.98))

    def multiscale_condition(
        self, pyramid: Pyramid | Tensor, latent_size: tuple[int, int]
    ) -> list[Tensor]:
        if isinstance(pyramid, Tensor):
            projected = self.legacy_scene(pyramid)
            projected = F.interpolate(
                projected, size=latent_size, mode="bilinear", align_corners=False
            )
            return [
                torch.zeros_like(projected),
                torch.zeros_like(projected),
                torch.zeros_like(projected),
                projected,
            ]
        conditions = []
        for level, projection in zip(pyramid, self.scene_projections, strict=True):
            projected = projection(level)
            conditions.append(
                F.interpolate(projected, size=latent_size, mode="bilinear", align_corners=False)
            )
        return conditions

    def _fused_condition(
        self, conditions: list[Tensor], block_index: int | None = None
    ) -> Tensor:
        if block_index is None:
            weights = torch.ones(4, device=conditions[0].device, dtype=conditions[0].dtype)
        else:
            weights = torch.tanh(self.condition_gates[block_index]).to(conditions[0].dtype)
        fused = sum(
            weight * condition for weight, condition in zip(weights, conditions, strict=True)
        )
        return self.frequency_adapter(fused)

    def predict_amplitude(
        self,
        pyramid: Pyramid | Tensor,
        target_descriptors: Tensor,
        channels: int,
        output_size: tuple[int, int],
        *,
        use_optical_bridge: bool = False,
    ) -> Tensor:
        latent_size = (output_size[0] // 4, output_size[1] // 4)
        conditions = self.multiscale_condition(pyramid, latent_size)
        scene = self._fused_condition(conditions)
        target = self.target(target_descriptors.mean(dim=0)).view(1, -1, 1, 1)
        head = self.optical_bridge_amplitude_head if use_optical_bridge else self.amplitude_head
        return torch.sigmoid(head(F.silu(scene + target))[:, :channels])

    @staticmethod
    def _texture_candidate_features(texture: Tensor) -> Tensor:
        luminance = texture.mean(dim=1, keepdim=True)
        chroma = (texture - luminance).abs().mean(dim=1, keepdim=True)
        dx = F.pad(luminance[..., :, 1:] - luminance[..., :, :-1], (0, 1))
        dy = F.pad(luminance[..., 1:, :] - luminance[..., :-1, :], (0, 0, 0, 1))
        gradient = torch.sqrt(dx.square() + dy.square() + 1e-8)
        return torch.cat(
            (
                F.avg_pool2d(luminance.abs(), 4, stride=4),
                F.avg_pool2d(luminance.square(), 4, stride=4).sqrt(),
                F.avg_pool2d(chroma, 4, stride=4),
                F.avg_pool2d(gradient, 4, stride=4),
            ),
            dim=1,
        )

    def predict_texture_risk_logits(
        self,
        pyramid: Pyramid | Tensor,
        target_descriptors: Tensor,
        texture: Tensor,
    ) -> Tensor:
        latent_size = (texture.shape[-2] // 4, texture.shape[-1] // 4)
        conditions = self.multiscale_condition(pyramid, latent_size)
        scene = self._fused_condition(conditions)
        target = self.target(target_descriptors.mean(dim=0)).view(1, -1, 1, 1)
        candidate = self.texture_risk_candidate(self._texture_candidate_features(texture))
        return self.texture_risk_head(F.silu(scene + target + candidate))

    def forward(
        self,
        latent: Tensor,
        time: Tensor,
        pyramid: Pyramid | Tensor,
        target_descriptors: Tensor,
        *,
        origin_latent: Tensor | None = None,
        bridge_anchor: Tensor | None = None,
        use_optical_bridge: bool = False,
    ) -> Tensor:
        batch, channels, height, width = latent.shape
        if channels != self.latent_channels:
            raise ValueError(f"expected {self.latent_channels} latent channels, got {channels}")
        values = self.input(latent.flatten(2).transpose(1, 2))
        if origin_latent is not None:
            if origin_latent.shape != latent.shape:
                raise ValueError("origin latent must match flow latent")
            origin = self.origin_projection(origin_latent).flatten(2).transpose(1, 2)
            values = values + origin
        if use_optical_bridge:
            if bridge_anchor is None or bridge_anchor.shape != latent.shape:
                raise ValueError("Optical bridge requires an anchor latent matching flow latent")
            anchor = self.optical_bridge_anchor(bridge_anchor).flatten(2).transpose(1, 2)
            values = values + anchor
        conditions = self.multiscale_condition(pyramid, (height, width))
        condition = self.time(_time_embedding(time, values.shape[-1]))
        condition = condition + self.target(target_descriptors.mean(dim=0)).unsqueeze(0).expand(
            batch, -1
        )
        for index, block in enumerate(self.blocks):
            scene = self._fused_condition(conditions, index)
            values = values + scene.flatten(2).transpose(1, 2)
            values = block(values, condition)
            if use_optical_bridge:
                values = self.optical_bridge_adapters[index](values)
        output = self.output(values)
        if use_optical_bridge:
            output = (
                torch.sigmoid(self.optical_bridge_base_gate) * output
                + self.optical_bridge_output(values)
            )
        return output.transpose(1, 2).reshape(batch, channels, height, width)


@dataclass
class ModelConfig:
    width: int = 64
    hidden: int = 768
    encoder_depth: int = 12
    heads: int = 12
    adapter_rank: int = 64
    dit_hidden: int = 512
    dit_depth: int = 8
    dit_heads: int = 8
    codec_width: int = 64
    codec_latent_channels: int = 16
    flow_steps: int = 16
    flow_noise_scale: float = 0.35
    optical_flow_noise_scale: float | None = None
    sar_flow_noise_scale: float | None = None
    optical_residual_limit: float = 0.15
    sar_residual_limit_db: float = 6.0
    optical_texture_risk_threshold: float = 0.0
    optical_bridge_enabled: bool = False
    optical_bridge_density_threshold: float = 1.0
    id_bridge_enabled: bool = False
    id_bridge_state: str = "codec"
    id_bridge_state_channels: int = 48
    id_bridge_optical_state_scale: float = 0.03
    id_bridge_sar_state_scale: float = 4.0
    id_bridge_anchor_origin: bool = False
    id_bridge_optical_innovation_scale: float = 1.0
    id_bridge_sar_innovation_scale: float = 1.0
    architecture: str = "v3.2"

    def __post_init__(self) -> None:
        if self.id_bridge_state not in {"codec", "haar_packet"}:
            raise ValueError("id_bridge_state must be codec or haar_packet")
        if self.id_bridge_state == "haar_packet" and self.id_bridge_state_channels != 48:
            raise ValueError("haar_packet id_bridge_state_channels must be 48")
        for name in ("id_bridge_optical_state_scale", "id_bridge_sar_state_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("id_bridge_optical_innovation_scale", "id_bridge_sar_innovation_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


class SentinelV3(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.encoder = SceneEncoder(
            cfg.width, cfg.hidden, cfg.encoder_depth, cfg.heads, cfg.adapter_rank
        )
        self.decoder = DynamicPhysicalDecoder(cfg.width, cfg.hidden)
        self.detail_head = MultiscaleDetailHead(cfg.width, cfg.hidden)
        self.codec = ResidualCodec(cfg.codec_width, cfg.codec_latent_channels)
        self.id_bridge_latent_channels = (
            cfg.id_bridge_state_channels
            if cfg.id_bridge_enabled and cfg.id_bridge_state == "haar_packet"
            else cfg.codec_latent_channels
        )
        self.residual_dit = ResidualDiT(
            (cfg.width, cfg.width * 2, cfg.width * 4, cfg.hidden),
            self.id_bridge_latent_channels,
            cfg.dit_hidden,
            cfg.dit_depth,
            cfg.dit_heads,
            zero_output=cfg.id_bridge_enabled and cfg.id_bridge_state == "haar_packet",
        )
        self.id_bridge_origin = IdentifiabilityBridgeOrigin(
            (cfg.width, cfg.width * 2, cfg.width * 4, cfg.hidden),
            self.id_bridge_latent_channels,
            cfg.dit_hidden,
        )
        self.register_buffer("optical_alpha_scale", torch.ones(()))
        self.register_buffer("optical_bridge_alpha_scale", torch.ones(()))
        self.register_buffer("sar_alpha_scale", torch.ones(()))
        self.register_buffer("optical_texture_amplitude_floor", torch.zeros(()))
        self.register_buffer("optical_bridge_texture_amplitude_floor", torch.zeros(()))
        self.register_buffer("optical_anchor_band_scales", torch.zeros(3))
        self.register_buffer("optical_anchor_density_gain", torch.zeros(()))
        self.register_buffer("optical_anchor_density_threshold", torch.ones(()))
        self.register_buffer("optical_anchor_source_gain", torch.zeros(()))
        self.register_buffer("optical_anchor_source_threshold", torch.ones(()))
        self.register_buffer(
            "optical_texture_risk_threshold",
            torch.tensor(cfg.optical_texture_risk_threshold),
        )
        self.register_buffer("optical_detail_confidence_threshold", torch.tensor(0.55))
        self.register_buffer("sar_detail_confidence_threshold", torch.tensor(0.55))
        self.temporal_prior: TemporalPriorStore | None = None

    def configure_temporal_prior(self, config: dict[str, object] | None) -> None:
        self.temporal_prior = (
            TemporalPriorStore(TemporalPriorConfig.from_dict(config)) if config else None
        )

    def apply_temporal_prior(
        self,
        physical: Tensor,
        target: SensorSpec,
        *,
        acquired: date | str,
        location_id: str | None,
        pixel_window: tuple[int, int, int, int] | None,
        orbit: str = "unknown",
        exclude_pair_id: str | None = None,
        spatial_transform: tuple[bool, bool, int] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        zero = physical.new_zeros(())
        if self.temporal_prior is None or location_id is None or pixel_window is None:
            return physical, zero, zero
        queried = self.temporal_prior.query(
            location_id=location_id,
            acquired=acquired,
            modality=target.modality,
            pixel_window=pixel_window,
            orbit=orbit,
            device=physical.device,
            dtype=physical.dtype,
            exclude_pair_id=exclude_pair_id,
        )
        if queried is None:
            return physical, zero, zero
        prior, coverage = queried
        if spatial_transform is not None:
            flip_x, flip_y, rotations = spatial_transform
            if flip_x:
                prior, coverage = (torch.flip(values, (-1,)) for values in (prior, coverage))
            if flip_y:
                prior, coverage = (torch.flip(values, (-2,)) for values in (prior, coverage))
            if rotations:
                prior, coverage = (
                    torch.rot90(values, rotations, (-2, -1)) for values in (prior, coverage)
                )
        composed, violation = self.temporal_prior.compose(
            physical, prior, coverage, target.modality
        )
        return composed, coverage.float().mean(), violation

    @staticmethod
    def descriptors(channels: Iterable[ChannelSpec], device: torch.device) -> Tensor:
        return torch.tensor(
            [channel.descriptor() for channel in channels], device=device, dtype=torch.float32
        )

    @staticmethod
    def condition(
        batch: int,
        device: torch.device,
        input_gsd: float | Tensor,
        target_gsd: float | Tensor,
        metadata: Tensor | None = None,
    ) -> Tensor:
        if isinstance(input_gsd, Tensor) or isinstance(target_gsd, Tensor):
            input_values = torch.as_tensor(
                input_gsd, device=device, dtype=torch.float32
            ).reshape(-1)
            target_values = torch.as_tensor(
                target_gsd, device=device, dtype=torch.float32
            ).reshape(-1)
            input_values = (
                input_values.expand(batch) if input_values.numel() == 1 else input_values
            )
            target_values = (
                target_values.expand(batch) if target_values.numel() == 1 else target_values
            )
            scale = torch.stack(
                (
                    torch.log2(input_values / 10.0),
                    torch.log2(input_values / 10.0),
                    torch.log2(target_values / 10.0),
                ),
                dim=-1,
            )
        else:
            scale = gsd_condition(input_gsd, 10.0, target_gsd).to(device).expand(batch, -1)
        if metadata is None:
            metadata = torch.zeros(batch, 8, device=device)
        return torch.cat((metadata.to(device), scale), dim=-1)

    def encode(
        self,
        values: Tensor,
        sensor: SensorSpec,
        valid: Tensor,
        *,
        channels: tuple[ChannelSpec, ...] | None = None,
        input_gsd: float | Tensor = 10.0,
        target_gsd: float | Tensor = 10.0,
        metadata: Tensor | None = None,
    ) -> Pyramid:
        descriptors = self.descriptors(channels or sensor.channels, values.device)
        condition = self.condition(
            values.shape[0], values.device, input_gsd, target_gsd, metadata
        )
        return self.encoder(values, descriptors, valid, condition, sensor.modality)

    def physical(
        self,
        values: Tensor,
        source: SensorSpec,
        target: SensorSpec,
        valid: Tensor,
        *,
        source_channels: tuple[ChannelSpec, ...] | None = None,
        input_gsd: float | Tensor = 10.0,
        target_gsd: float | Tensor = 10.0,
        metadata: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Pyramid]:
        pyramid = self.encode(
            values,
            source,
            valid,
            channels=source_channels,
            input_gsd=input_gsd,
            target_gsd=target_gsd,
            metadata=metadata,
        )
        descriptors = self.descriptors(target.channels, values.device)
        scene_condition = self.condition(
            values.shape[0], values.device, input_gsd, target_gsd, metadata
        )
        mean, log_variance = self.decoder(
            pyramid,
            descriptors,
            target.modality,
            values.shape[-2:],
            scene_condition[:, -3:],
            scene_condition,
            valid,
        )
        return mean * valid, log_variance, pyramid

    def deterministic_detail(
        self,
        pyramid: Pyramid,
        source: SensorSpec,
        target: SensorSpec,
        output_size: tuple[int, int],
        base: Tensor | None = None,
    ) -> Tensor:
        detail = self.detail_head.forward_with_confidence(
            pyramid,
            source.modality,
            target.modality,
            output_size,
            base,
            getattr(self, f"{target.modality}_detail_confidence_threshold"),
            hard_gate=True,
        )[0]
        if target.modality == "optical" and base is not None:
            anchor_bands = frequency_bands(base, levels=3)
            scales = self.optical_anchor_band_scales.to(base)
            fine_scale: Tensor = scales[0]
            if float(self.optical_anchor_density_gain) > 0.0:
                density = F.avg_pool2d(
                    anchor_bands[0].abs().mean(dim=1, keepdim=True), 4, stride=4
                )
                normalized = density / density.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
                gate = (normalized >= self.optical_anchor_density_threshold.to(normalized)).to(
                    base.dtype
                )
                gate = F.avg_pool2d(gate, 3, stride=1, padding=1)
                gate = F.interpolate(
                    gate, size=base.shape[-2:], mode="bilinear", align_corners=False
                )
                fine_scale = fine_scale + self.optical_anchor_density_gain.to(base) * gate
            if float(self.optical_anchor_source_gain) > 0.0:
                source_density = F.avg_pool2d(
                    highpass(pyramid[0]).abs().mean(dim=1, keepdim=True), 4, stride=4
                )
                normalized_source = source_density / source_density.mean(
                    dim=(-2, -1), keepdim=True
                ).clamp_min(1e-6)
                source_gate = (
                    normalized_source
                    >= self.optical_anchor_source_threshold.to(normalized_source)
                ).to(base.dtype)
                source_gate = F.avg_pool2d(source_gate, 3, stride=1, padding=1)
                source_gate = F.interpolate(
                    source_gate,
                    size=base.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                fine_scale = fine_scale + self.optical_anchor_source_gain.to(base) * source_gate
            anchor_detail = fine_scale * anchor_bands[0]
            anchor_detail = anchor_detail + sum(
                scale * band for scale, band in zip(scales[1:], anchor_bands[1:], strict=True)
            )
            detail = detail + anchor_detail
        return highpass(detail)

    def visual_detail(
        self,
        pyramid: Pyramid,
        source: SensorSpec,
        target: SensorSpec,
        output_size: tuple[int, int],
        base: Tensor,
    ) -> Tensor:
        if self.config.id_bridge_enabled:
            if self.id_bridge_uses_observable_anchor:
                return self.id_bridge_anchor_detail(pyramid, base, target)
            return torch.zeros_like(base)
        return self.deterministic_detail(pyramid, source, target, output_size, base=base)

    def deterministic_detail_with_confidence(
        self,
        pyramid: Pyramid,
        source: SensorSpec,
        target: SensorSpec,
        output_size: tuple[int, int],
        base: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...], Tensor]:
        return self.detail_head.forward_with_confidence(
            pyramid,
            source.modality,
            target.modality,
            output_size,
            base,
            getattr(self, f"{target.modality}_detail_confidence_threshold"),
        )

    @torch.no_grad()
    def set_detail_confidence_threshold(self, modality: str, value: float) -> None:
        if modality not in {"optical", "sar"} or not 0.0 <= value <= 1.01:
            raise ValueError("detail confidence threshold must be in [0, 1.01]")
        getattr(self, f"{modality}_detail_confidence_threshold").fill_(value)

    @torch.no_grad()
    def set_optical_anchor_band_scales(self, values: Sequence[float]) -> None:
        if len(values) != 3 or any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "three finite non-negative Optical anchor band scales are required"
            )
        self.optical_anchor_band_scales.copy_(
            torch.tensor(values, device=self.optical_anchor_band_scales.device)
        )

    @torch.no_grad()
    def set_optical_anchor_density(self, gain: float, threshold: float) -> None:
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("Optical anchor density gain must be finite and non-negative")
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("Optical anchor density threshold must be finite and positive")
        self.optical_anchor_density_gain.fill_(gain)
        self.optical_anchor_density_threshold.fill_(threshold)

    @torch.no_grad()
    def set_optical_anchor_source_density(self, gain: float, threshold: float) -> None:
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("Optical source anchor gain must be finite and non-negative")
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("Optical source anchor threshold must be finite and positive")
        self.optical_anchor_source_gain.fill_(gain)
        self.optical_anchor_source_threshold.fill_(threshold)

    def flow_velocity(
        self,
        latent: Tensor,
        time: Tensor,
        pyramid: Pyramid | Tensor,
        target: SensorSpec,
        visual_channels: int | None = None,
        *,
        origin_latent: Tensor | None = None,
        bridge_anchor: Tensor | None = None,
        use_optical_bridge: bool | None = None,
    ) -> Tensor:
        channels = visual_channels or (3 if target.modality == "optical" else 2)
        descriptors = self.descriptors(target.channels[:channels], latent.device)
        use_bridge = (
            target.modality == "optical" and self.config.optical_bridge_enabled
            if use_optical_bridge is None
            else use_optical_bridge
        )
        return self.residual_dit(
            latent,
            time,
            pyramid,
            descriptors,
            origin_latent=origin_latent,
            bridge_anchor=bridge_anchor,
            use_optical_bridge=use_bridge,
        )

    def predict_id_bridge_origin(
        self,
        pyramid: Pyramid,
        physical: Tensor,
        target: SensorSpec,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mu, _, _, log_sigma, reliability_logits = self.predict_id_bridge_origin_components(
            pyramid, physical, target
        )
        return mu, log_sigma, reliability_logits

    def predict_id_bridge_origin_components(
        self,
        pyramid: Pyramid,
        physical: Tensor,
        target: SensorSpec,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return innovation origin, raw correction, pixel anchor, and distribution parameters."""

        correction, log_sigma, reliability_logits = self.id_bridge_origin(
            pyramid, physical, target
        )
        anchor_detail = self.id_bridge_anchor_detail(pyramid, physical, target)
        if anchor_detail.shape != physical.shape:
            raise RuntimeError("id bridge anchor detail must match the physical visual shape")
        if self.id_bridge_uses_observable_anchor:
            correction_gate = torch.sigmoid(reliability_logits).mean(dim=1, keepdim=True).detach()
            mu = correction_gate * correction
        else:
            mu = correction
        return mu, correction, anchor_detail, log_sigma, reliability_logits

    @property
    def id_bridge_uses_haar_packet(self) -> bool:
        return self.config.id_bridge_enabled and self.config.id_bridge_state == "haar_packet"

    @property
    def id_bridge_uses_observable_anchor(self) -> bool:
        return self.id_bridge_uses_haar_packet and self.config.id_bridge_anchor_origin

    @staticmethod
    def _id_bridge_visual_channels(target: SensorSpec) -> int:
        if target.modality == "optical":
            return 3
        if target.modality == "sar":
            return 2
        raise ValueError("id bridge target modality must be optical or sar")

    def id_bridge_anchor_detail(
        self,
        pyramid: Pyramid,
        physical: Tensor,
        target: SensorSpec,
    ) -> Tensor:
        if not (self.id_bridge_uses_observable_anchor and target.modality == "optical"):
            return torch.zeros_like(physical)
        with torch.no_grad():
            anchor_detail = self.deterministic_detail(
                pyramid,
                SENTINEL1,
                SENTINEL2,
                tuple(physical.shape[-2:]),
                base=physical,
            )
        return anchor_detail.detach()

    def _project_haar_packet_state(self, state: Tensor, target: SensorSpec) -> Tensor:
        if state.ndim != 4:
            raise ValueError("id bridge Haar state must be BCHW")
        visual_channels = self._id_bridge_visual_channels(target)
        native_channels = visual_channels * 16
        if state.shape[1] not in {native_channels, self.config.id_bridge_state_channels}:
            raise ValueError("id bridge Haar state has an unexpected channel count")
        projected = state.clone()
        projected[:, :native_channels:16] = 0
        if target.modality == "sar" and projected.shape[1] == self.config.id_bridge_state_channels:
            projected[:, native_channels:] = 0
        return projected

    def project_id_bridge_residual(self, residual: Tensor, target: SensorSpec) -> Tensor:
        """Remove residual-state components that are not allowed in an id bridge sample."""

        if not self.id_bridge_uses_haar_packet:
            return highpass(residual)
        visual_channels = self._id_bridge_visual_channels(target)
        if residual.ndim != 4 or residual.shape[1] != visual_channels:
            raise ValueError("id bridge residual has unexpected visual channels")
        coefficients = haar_packet_dwt2(residual)
        return haar_packet_idwt2(self._project_haar_packet_state(coefficients, target))

    def encode_id_bridge_residual(self, residual: Tensor, target: SensorSpec) -> Tensor:
        """Encode an id bridge endpoint without routing Haar states through the codec."""

        if not self.id_bridge_uses_haar_packet:
            return self.codec.encode(residual, target.modality)
        coefficients = self._project_haar_packet_state(haar_packet_dwt2(residual), target)
        if target.modality == "sar":
            coefficients = torch.cat(
                (
                    coefficients,
                    coefficients.new_zeros(
                        coefficients.shape[0],
                        self.config.id_bridge_state_channels - coefficients.shape[1],
                        *coefficients.shape[-2:],
                    ),
                ),
                dim=1,
            )
        scale = float(getattr(self.config, f"id_bridge_{target.modality}_state_scale"))
        return self._project_haar_packet_state(coefficients / scale, target)

    def decode_id_bridge_residual(self, state: Tensor, target: SensorSpec) -> Tensor:
        """Decode an id bridge state, enforcing Haar coarse and SAR padding constraints."""

        if not self.id_bridge_uses_haar_packet:
            return highpass(self.codec.decode(state, target.modality))
        if state.ndim != 4 or state.shape[1] != self.id_bridge_latent_channels:
            raise ValueError("id bridge Haar state has an unexpected channel count")
        # The projection is deliberately the final state operation before inverse Haar.
        projected = self._project_haar_packet_state(state, target)
        visual_channels = self._id_bridge_visual_channels(target)
        scale = float(getattr(self.config, f"id_bridge_{target.modality}_state_scale"))
        return haar_packet_idwt2(projected[:, : visual_channels * 16] * scale)

    def gate_id_bridge_innovation(
        self,
        latent: Tensor,
        mu: Tensor,
        reliability_logits: Tensor,
        target: SensorSpec,
    ) -> Tensor:
        if latent.ndim != 4 or mu.shape != latent.shape:
            raise ValueError("id bridge latent and origin must have matching BCHW shapes")
        if reliability_logits.shape != (latent.shape[0], 3, *latent.shape[-2:]):
            raise ValueError("id bridge reliability logits must be B3HW on the latent grid")
        q = torch.sigmoid(reliability_logits).mean(dim=1, keepdim=True)
        scale = float(getattr(self.config, f"id_bridge_{target.modality}_innovation_scale"))
        # Legacy codec and unanchored Haar checkpoints retain their prior transport behavior.
        if not self.id_bridge_uses_observable_anchor:
            return latent
        return mu + scale * (1.0 - q) * (latent - mu)

    def residual_state_metadata(self) -> dict[str, object]:
        return {
            "kind": self.config.id_bridge_state,
            "channels": (
                self.config.id_bridge_state_channels
                if self.config.id_bridge_state == "haar_packet"
                else self.config.codec_latent_channels
            ),
            "optical_scale": self.config.id_bridge_optical_state_scale,
            "sar_scale": self.config.id_bridge_sar_state_scale,
            "anchor_origin": self.config.id_bridge_anchor_origin,
            "optical_innovation_scale": self.config.id_bridge_optical_innovation_scale,
            "sar_innovation_scale": self.config.id_bridge_sar_innovation_scale,
        }

    def sample_id_bridge_residual(
        self,
        pyramid: Pyramid,
        physical: Tensor,
        target: SensorSpec,
        *,
        seed: int,
        steps: int | None = None,
        return_origin: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Sample a residual from the identifiability-conditioned flow origin."""

        mu, _, anchor_detail, log_sigma, reliability_logits = (
            self.predict_id_bridge_origin_components(pyramid, physical, target)
        )
        q = torch.sigmoid(reliability_logits).mean(dim=1, keepdim=True)
        sigma = self.flow_noise_scale(target) * torch.sigmoid(log_sigma) * (1.0 - q)
        generator = torch.Generator(device=mu.device).manual_seed(seed)
        epsilon = torch.randn(
            mu.shape,
            generator=generator,
            device=mu.device,
            dtype=mu.dtype,
        )
        z0 = mu + sigma * epsilon
        latent = self.integrate_flow(
            z0,
            pyramid,
            target,
            physical.shape[1],
            steps=steps or self.config.flow_steps,
            origin_latent=mu,
            use_optical_bridge=False,
        )
        if self.id_bridge_uses_observable_anchor:
            innovation_gate = float(
                getattr(self.config, f"id_bridge_{target.modality}_innovation_scale")
            ) * (1.0 - q)
            correction_gate = q.detach()
        else:
            innovation_gate = torch.ones_like(q)
            correction_gate = torch.ones_like(q)
        latent = self.gate_id_bridge_innovation(latent, mu, reliability_logits, target)
        residual = self.decode_id_bridge_residual(latent, target)
        if return_origin:
            return residual, {
                "mu": mu,
                "q": q,
                "sigma": sigma,
                "anchor_detail": anchor_detail,
                "correction_gate": correction_gate,
                "innovation_gate": innovation_gate,
            }
        return residual

    def sample_visual_residual(
        self,
        pyramid: Pyramid,
        target: SensorSpec,
        base: Tensor,
        detail: Tensor,
        *,
        seed: int,
        steps: int | None = None,
    ) -> Tensor:
        if self.config.id_bridge_enabled:
            residual = self.sample_id_bridge_residual(
                pyramid, base, target, seed=seed, steps=steps
            )
            assert isinstance(residual, Tensor)
            return residual
        return self.sample_residual(
            pyramid,
            target,
            tuple(base.shape),
            seed=seed,
            steps=steps,
            bridge_anchor=detail,
        )

    def residual_amplitude(
        self,
        pyramid: Pyramid | Tensor,
        target: SensorSpec,
        visual_channels: int,
        output_size: tuple[int, int],
    ) -> Tensor:
        descriptors = self.descriptors(
            target.channels[:visual_channels], next(self.parameters()).device
        )
        normalized = self.residual_dit.predict_amplitude(
            pyramid,
            descriptors,
            visual_channels,
            output_size,
            use_optical_bridge=(
                target.modality == "optical" and self.config.optical_bridge_enabled
            ),
        )
        limit = (
            self.config.optical_residual_limit
            if target.modality == "optical"
            else self.config.sar_residual_limit_db
        )
        scale = getattr(self, self.amplitude_scale_name(target.modality))
        return (normalized * limit * scale).clamp_max(limit)

    def amplitude_scale_name(self, modality: str) -> str:
        if modality == "optical" and self.config.optical_bridge_enabled:
            return "optical_bridge_alpha_scale"
        if modality not in {"optical", "sar"}:
            raise ValueError("amplitude scale is defined for optical or sar")
        return f"{modality}_alpha_scale"

    @torch.no_grad()
    def set_amplitude_scale(self, modality: str, value: float) -> None:
        if modality not in {"optical", "sar"} or not 0.0 <= value <= 1.0:
            raise ValueError("amplitude scale must be in [0, 1] for optical or sar")
        getattr(self, self.amplitude_scale_name(modality)).fill_(value)

    @torch.no_grad()
    def set_optical_texture_amplitude_floor(self, value: float) -> None:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("optical texture amplitude floor must be finite and non-negative")
        getattr(self, self.texture_amplitude_floor_name()).fill_(value)

    def texture_amplitude_floor_name(self) -> str:
        if self.config.optical_bridge_enabled:
            return "optical_bridge_texture_amplitude_floor"
        return "optical_texture_amplitude_floor"

    def texture_release_gate(self, amplitude: Tensor, target: SensorSpec) -> Tensor:
        """Release complete 4x4 texture blocks only when their predicted amplitude is sufficient."""
        if target.modality != "optical":
            return torch.ones_like(amplitude[:, :1])
        block_amplitude = amplitude.mean(dim=1, keepdim=True)
        floor = getattr(self, self.texture_amplitude_floor_name()).to(amplitude)
        return (block_amplitude >= floor).to(amplitude.dtype)

    def texture_release_probability(
        self, pyramid: Pyramid | Tensor, target: SensorSpec, texture: Tensor
    ) -> Tensor:
        if target.modality != "optical":
            return torch.ones(
                texture.shape[0],
                1,
                texture.shape[-2] // 4,
                texture.shape[-1] // 4,
                device=texture.device,
                dtype=texture.dtype,
            )
        descriptors = self.descriptors(target.channels[: texture.shape[1]], texture.device)
        logits = self.residual_dit.predict_texture_risk_logits(pyramid, descriptors, texture)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def set_optical_texture_risk_threshold(self, value: float) -> None:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Optical texture risk threshold must be in [0, 1]")
        self.optical_texture_risk_threshold.fill_(value)

    def shape_residual_texture(
        self,
        residual: Tensor,
        pyramid: Pyramid | Tensor,
        target: SensorSpec,
        *,
        amplitude: Tensor | None = None,
        apply_release_gate: bool = True,
    ) -> Tensor:
        """Apply the same zero-mean, block-RMS and amplitude path used at inference."""
        channels = residual.shape[1]
        output_size = tuple(residual.shape[-2:])
        residual = highpass(residual)
        if target.modality == "sar":
            local_mean = F.avg_pool2d(residual, 4, stride=4)
            residual = residual - F.interpolate(local_mean, size=output_size, mode="nearest")
        block_rms = torch.sqrt(
            F.avg_pool2d(residual.square(), 4, stride=4) + 1e-8
        ).clamp_min(1e-4)
        unit = residual / F.interpolate(block_rms, size=output_size, mode="nearest")
        if amplitude is None:
            amplitude = self.residual_amplitude(pyramid, target, channels, output_size)
        amplitude = amplitude * self.texture_release_gate(amplitude, target)
        residual = highpass(unit) * F.interpolate(amplitude, size=output_size, mode="nearest")
        if target.modality == "optical" and channels == 3:
            luminance = residual.mean(dim=1, keepdim=True)
            chroma = residual - luminance
            chroma_scale = (
                0.03 / chroma.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
            ).clamp_max(1.0)
            chroma = chroma * chroma_scale
            residual = luminance + chroma
        elif target.modality == "sar":
            local_mean = F.avg_pool2d(residual, 4, stride=4)
            residual = residual - F.interpolate(local_mean, size=output_size, mode="nearest")
        if (
            apply_release_gate
            and target.modality == "optical"
            and not self.config.optical_bridge_enabled
            and float(self.optical_texture_risk_threshold) > 0.0
        ):
            probability = self.texture_release_probability(pyramid, target, residual)
            gate = (probability >= self.optical_texture_risk_threshold).to(residual.dtype)
            residual = residual * F.interpolate(gate, size=output_size, mode="nearest")
        return residual

    @staticmethod
    def compose_visual(
        physical: Tensor,
        deterministic_detail: Tensor,
        stochastic_residual: Tensor | str | None = None,
        modality: str | None = None,
        *,
        return_violation: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # V3.1 accepted (physical, residual, modality); keep it as an init/eval bridge.
        if isinstance(stochastic_residual, str):
            modality = stochastic_residual
            stochastic = deterministic_detail
            deterministic = torch.zeros_like(stochastic)
        else:
            deterministic = deterministic_detail
            stochastic = (
                torch.zeros_like(deterministic)
                if stochastic_residual is None
                else stochastic_residual
            )
        if modality is None:
            raise ValueError("modality is required")
        additive = physical + deterministic + stochastic
        if modality == "optical":
            violation = ((additive < 0.0) | (additive > 1.0)).to(additive.dtype).mean()
            base = physical.clamp(1e-4, 1 - 1e-4)
            delta = (deterministic + stochastic) / (base * (1 - base)).clamp_min(1e-3)
            composed = torch.sigmoid(torch.logit(base) + delta)
        else:
            violation = ((additive < -50.0) | (additive > 5.0)).to(additive.dtype).mean()
            composed = additive.clamp(-50.0, 5.0)
        return (composed, violation) if return_violation else composed

    def sample_residual(
        self,
        pyramid: Pyramid | Tensor,
        target: SensorSpec,
        shape: tuple[int, int, int, int],
        *,
        seed: int,
        steps: int | None = None,
        bridge_anchor: Tensor | None = None,
    ) -> Tensor:
        batch, channels, height, width = shape
        generator = torch.Generator(device=next(self.parameters()).device).manual_seed(seed)
        latent = self.flow_noise_scale(target) * torch.randn(
            (batch, self.config.codec_latent_channels, height // 4, width // 4),
            generator=generator,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )
        anchor_latent: Tensor | None = None
        bridge_gate: Tensor | None = None
        use_bridge = target.modality == "optical" and self.config.optical_bridge_enabled
        if use_bridge:
            if bridge_anchor is None:
                return torch.zeros(
                    (batch, channels, height, width),
                    device=latent.device,
                    dtype=latent.dtype,
                )
            anchor_latent = self.codec.encode(highpass(bridge_anchor), "optical")
            bridge_gate = self.optical_bridge_gate(bridge_anchor, latent.shape[-2:])
            latent = latent * bridge_gate
        latent = self.integrate_flow(
            latent,
            pyramid,
            target,
            channels,
            steps=steps or self.config.flow_steps,
            bridge_anchor=anchor_latent,
            use_optical_bridge=use_bridge,
        )
        residual = self.codec.decode(latent, target.modality)
        residual = self.shape_residual_texture(residual, pyramid, target)
        if bridge_gate is not None:
            residual = residual * F.interpolate(
                bridge_gate, size=residual.shape[-2:], mode="bilinear", align_corners=False
            )
        return residual

    def optical_bridge_gate(
        self, anchor_detail: Tensor, latent_size: tuple[int, int]
    ) -> Tensor:
        energy = F.interpolate(
            highpass(anchor_detail).abs().mean(dim=1, keepdim=True),
            size=latent_size,
            mode="area",
        )
        normalized = energy / energy.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        gate = (
            normalized >= self.config.optical_bridge_density_threshold
        ).to(anchor_detail.dtype)
        return F.avg_pool2d(gate, 3, stride=1, padding=1)

    def flow_noise_scale(self, target: SensorSpec) -> float:
        configured = getattr(self.config, f"{target.modality}_flow_noise_scale")
        return self.config.flow_noise_scale if configured is None else float(configured)

    def integrate_flow(
        self,
        latent: Tensor,
        pyramid: Pyramid | Tensor,
        target: SensorSpec,
        channels: int,
        *,
        steps: int,
        origin_latent: Tensor | None = None,
        bridge_anchor: Tensor | None = None,
        use_optical_bridge: bool | None = None,
    ) -> Tensor:
        """Integrate a residual flow with the same differentiable Heun solver used at inference."""
        if steps < 1:
            raise ValueError("flow integration steps must be positive")
        batch = latent.shape[0]
        dt = 1.0 / steps
        for index in range(steps):
            time = torch.full((batch,), index / steps, device=latent.device, dtype=latent.dtype)
            first = self.flow_velocity(
                latent,
                time,
                pyramid,
                target,
                channels,
                origin_latent=origin_latent,
                bridge_anchor=bridge_anchor,
                use_optical_bridge=use_optical_bridge,
            )
            proposal = latent + dt * first
            next_time = torch.full(
                (batch,), (index + 1) / steps, device=latent.device, dtype=latent.dtype
            )
            second = self.flow_velocity(
                proposal,
                next_time,
                pyramid,
                target,
                channels,
                origin_latent=origin_latent,
                bridge_anchor=bridge_anchor,
                use_optical_bridge=use_optical_bridge,
            )
            latent = latent + 0.5 * dt * (first + second)
        return latent

    def forward(self, action: str, **kwargs: object) -> object:
        if action == "physical":
            return self.physical(**kwargs)  # type: ignore[arg-type]
        if action == "encode":
            return self.encode(**kwargs)  # type: ignore[arg-type]
        if action == "flow":
            return self.flow_velocity(**kwargs)  # type: ignore[arg-type]
        if action == "detail":
            return self.deterministic_detail(**kwargs)  # type: ignore[arg-type]
        raise ValueError(f"unsupported forward action: {action}")

    @staticmethod
    def _metadata(observation: Observation, batch: int, device: torch.device) -> Tensor:
        acquired = (
            date.fromisoformat(observation.acquired)
            if isinstance(observation.acquired, str)
            else observation.acquired
        )
        phase = 2 * math.pi * acquired.timetuple().tm_yday / 366.0
        orbit = {"ascending": -1.0, "descending": 1.0, "unknown": 0.0}[observation.orbit]
        vector = torch.tensor(
            (0.0, orbit, math.sin(phase), math.cos(phase), 0, 0, 0, 1), device=device
        )
        return vector.expand(batch, -1)

    def translate(
        self,
        observations: list[Observation],
        target: TargetRequest,
        *,
        mode: str,
        num_samples: int,
        seed: int,
    ) -> TranslationResult:
        if len(observations) != 1:
            raise NotImplementedError(
                "V3.2 currently accepts one registered source observation"
            )
        observation = observations[0]
        values = (
            observation.values.unsqueeze(0)
            if observation.values.ndim == 3
            else observation.values
        )
        valid = observation.valid_mask
        if valid is None:
            valid = torch.ones(
                values.shape[0], 1, *values.shape[-2:], device=values.device, dtype=values.dtype
            )
        elif valid.ndim == 3:
            valid = valid.unsqueeze(0)
        metadata = self._metadata(observation, values.shape[0], values.device)
        physical, log_variance, pyramid = self.physical(
            values,
            observation.spec,
            target.spec,
            valid,
            source_channels=observation.channel_specs,
            input_gsd=observation.gsd_m,
            target_gsd=target.gsd_m,
            metadata=metadata,
        )
        target_date = target.acquired or observation.acquired
        physical, prior_coverage, prior_violation = self.apply_temporal_prior(
            physical,
            target.spec,
            acquired=target_date,
            location_id=observation.location_id,
            pixel_window=observation.pixel_window,
            orbit=observation.orbit,
        )
        result = TranslationResult(
            physical=physical,
            uncertainty=torch.exp(0.5 * log_variance) * valid,
            target=target,
            metadata={
                "seed": seed,
                "mode": mode,
                "source_sensor": observation.spec.name,
                "temporal_prior_coverage": float(prior_coverage),
                "temporal_prior_pre_projection_violation": float(prior_violation),
            },
        )
        if mode == "physical":
            return result
        visual_indices = (
            [target.spec.channel_names.index(name) for name in ("B04", "B03", "B02")]
            if target.spec.modality == "optical"
            else list(range(len(target.spec.channels)))
        )
        base = physical[:, visual_indices]
        detail = (
            self.visual_detail(
                pyramid,
                observation.spec,
                target.spec,
                tuple(base.shape[-2:]),
                base,
            )
            * valid
        )
        result.deterministic_detail = detail
        violations: list[Tensor] = []
        for sample_index in range(num_samples):
            texture = (
                self.sample_visual_residual(
                    pyramid,
                    target.spec,
                    base,
                    detail,
                    seed=seed + sample_index,
                )
                * valid
            )
            sample, violation = self.compose_visual(
                base, detail, texture, target.spec.modality, return_violation=True
            )
            result.samples.append(sample * valid)
            violations.append(violation)
            if sample_index == 0:
                result.stochastic_residual = texture
        if not self.config.id_bridge_enabled:
            result.residual_amplitude = self.residual_amplitude(
                pyramid, target.spec, base.shape[1], tuple(base.shape[-2:])
            )
        result.pre_projection_violation = torch.stack(violations).mean()
        return result
