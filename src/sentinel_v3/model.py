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
from .losses import highpass
from .physics import gsd_condition
from .sensors import ChannelSpec, SensorSpec

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
        for head in self.output_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        pyramid: Pyramid,
        source_modality: str,
        target_modality: str,
        output_size: tuple[int, int],
    ) -> Tensor:
        full, half, quarter, eighth = pyramid
        levels = [self.input_heads[source_modality](full)]
        for level, projection in zip((half, quarter, eighth), self.projections, strict=True):
            projected = projection(level)
            levels.append(
                F.interpolate(
                    projected, size=full.shape[-2:], mode="bilinear", align_corners=False
                )
            )
        features = self.trunk(torch.cat(levels, dim=1))
        detail = self.output_heads[target_modality](features)
        detail = F.interpolate(detail, size=output_size, mode="bilinear", align_corners=False)
        limit = 0.08 if target_modality == "optical" else 4.0
        return highpass(limit * torch.tanh(detail / limit))


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
    def update_statistics(self, latent: Tensor, modality: str, momentum: float = 0.01) -> None:
        working = latent.float()
        total = working.sum(dim=(0, 2, 3))
        total_square = working.square().sum(dim=(0, 2, 3))
        count = working.new_tensor(working.shape[0] * working.shape[2] * working.shape[3])
        if dist.is_available() and dist.is_initialized():
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


class ResidualDiT(nn.Module):
    def __init__(
        self,
        pyramid_channels: Sequence[int],
        latent_channels: int = 16,
        hidden: int = 512,
        depth: int = 8,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.input = nn.Linear(latent_channels, hidden)
        self.scene_projections = nn.ModuleList(
            [nn.Conv2d(channels, hidden, 1) for channels in pyramid_channels]
        )
        self.legacy_scene = nn.Conv2d(pyramid_channels[-1], hidden, 1)
        self.condition_gates = nn.Parameter(torch.zeros(depth, 4))
        self.time = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.target = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([FlowBlock(hidden, heads) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, latent_channels))
        self.amplitude_head = nn.Conv2d(hidden, 3, 1)
        nn.init.zeros_(self.amplitude_head.weight)
        nn.init.constant_(self.amplitude_head.bias, math.log(0.1 / 0.9))

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
        return sum(
            weight * condition for weight, condition in zip(weights, conditions, strict=True)
        )

    def predict_amplitude(
        self,
        pyramid: Pyramid | Tensor,
        target_descriptors: Tensor,
        channels: int,
        output_size: tuple[int, int],
    ) -> Tensor:
        latent_size = (output_size[0] // 4, output_size[1] // 4)
        conditions = self.multiscale_condition(pyramid, latent_size)
        scene = self._fused_condition(conditions)
        target = self.target(target_descriptors.mean(dim=0)).view(1, -1, 1, 1)
        return torch.sigmoid(self.amplitude_head(F.silu(scene + target))[:, :channels])

    def forward(
        self,
        latent: Tensor,
        time: Tensor,
        pyramid: Pyramid | Tensor,
        target_descriptors: Tensor,
    ) -> Tensor:
        batch, channels, height, width = latent.shape
        if channels != self.latent_channels:
            raise ValueError(f"expected {self.latent_channels} latent channels, got {channels}")
        values = self.input(latent.flatten(2).transpose(1, 2))
        conditions = self.multiscale_condition(pyramid, (height, width))
        condition = self.time(_time_embedding(time, values.shape[-1]))
        condition = condition + self.target(target_descriptors.mean(dim=0)).unsqueeze(0).expand(
            batch, -1
        )
        for index, block in enumerate(self.blocks):
            scene = self._fused_condition(conditions, index)
            values = values + scene.flatten(2).transpose(1, 2)
            values = block(values, condition)
        return self.output(values).transpose(1, 2).reshape(batch, channels, height, width)


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
    optical_residual_limit: float = 0.15
    sar_residual_limit_db: float = 6.0
    architecture: str = "v3.2"


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
        self.residual_dit = ResidualDiT(
            (cfg.width, cfg.width * 2, cfg.width * 4, cfg.hidden),
            cfg.codec_latent_channels,
            cfg.dit_hidden,
            cfg.dit_depth,
            cfg.dit_heads,
        )
        self.register_buffer("optical_alpha_scale", torch.ones(()))
        self.register_buffer("sar_alpha_scale", torch.ones(()))

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
    ) -> Tensor:
        return self.detail_head(pyramid, source.modality, target.modality, output_size)

    def flow_velocity(
        self,
        latent: Tensor,
        time: Tensor,
        pyramid: Pyramid | Tensor,
        target: SensorSpec,
        visual_channels: int | None = None,
    ) -> Tensor:
        channels = visual_channels or (3 if target.modality == "optical" else 2)
        descriptors = self.descriptors(target.channels[:channels], latent.device)
        return self.residual_dit(latent, time, pyramid, descriptors)

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
            pyramid, descriptors, visual_channels, output_size
        )
        limit = (
            self.config.optical_residual_limit
            if target.modality == "optical"
            else self.config.sar_residual_limit_db
        )
        scale = getattr(self, f"{target.modality}_alpha_scale")
        return (normalized * limit * scale).clamp_max(limit)

    @torch.no_grad()
    def set_amplitude_scale(self, modality: str, value: float) -> None:
        if modality not in {"optical", "sar"} or not 0.0 <= value <= 1.0:
            raise ValueError("amplitude scale must be in [0, 1] for optical or sar")
        getattr(self, f"{modality}_alpha_scale").fill_(value)

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
    ) -> Tensor:
        batch, channels, height, width = shape
        generator = torch.Generator(device=next(self.parameters()).device).manual_seed(seed)
        latent = torch.randn(
            (batch, self.config.codec_latent_channels, height // 4, width // 4),
            generator=generator,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )
        steps = steps or self.config.flow_steps
        dt = 1.0 / steps
        for index in range(steps):
            time = torch.full((batch,), index / steps, device=latent.device, dtype=latent.dtype)
            first = self.flow_velocity(latent, time, pyramid, target, channels)
            proposal = latent + dt * first
            next_time = torch.full(
                (batch,), (index + 1) / steps, device=latent.device, dtype=latent.dtype
            )
            second = self.flow_velocity(proposal, next_time, pyramid, target, channels)
            latent = latent + 0.5 * dt * (first + second)
        residual = highpass(self.codec.decode(latent, target.modality))
        block_rms = F.avg_pool2d(residual.square(), 4, stride=4).sqrt().clamp_min(1e-4)
        unit = residual / block_rms.repeat_interleave(4, -2).repeat_interleave(4, -1)
        amplitude = self.residual_amplitude(pyramid, target, channels, (height, width))
        return highpass(unit) * amplitude.repeat_interleave(4, -2).repeat_interleave(4, -1)

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
        result = TranslationResult(
            physical=physical,
            uncertainty=torch.exp(0.5 * log_variance) * valid,
            target=target,
            metadata={"seed": seed, "mode": mode, "source_sensor": observation.spec.name},
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
            self.deterministic_detail(
                pyramid, observation.spec, target.spec, tuple(base.shape[-2:])
            )
            * valid
        )
        result.deterministic_detail = detail
        violations: list[Tensor] = []
        for sample_index in range(num_samples):
            texture = (
                self.sample_residual(
                    pyramid, target.spec, tuple(base.shape), seed=seed + sample_index
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
        result.residual_amplitude = self.residual_amplitude(
            pyramid, target.spec, base.shape[1], tuple(base.shape[-2:])
        )
        result.pre_projection_violation = torch.stack(violations).mean()
        return result
