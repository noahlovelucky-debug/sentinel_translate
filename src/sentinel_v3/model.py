from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .api import Observation, TargetRequest, TranslationResult
from .losses import highpass
from .physics import gsd_condition
from .sensors import ChannelSpec, SensorSpec


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
        self.width = width
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
            nn.Linear(self.descriptor_dim + 2, width // 2),
            nn.SiLU(),
            nn.Linear(width // 2, 1),
        )
        self.bias = nn.Sequential(nn.Linear(self.descriptor_dim, width), nn.Tanh())
        self.spatial = nn.Conv2d(width + 1, width, 3, padding=1)
        self.refine = ResidualBlock(width)
        self.refine_gate = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.quadratic_embedding[-1].weight)
        nn.init.zeros_(self.quadratic_embedding[-1].bias)
        nn.init.zeros_(self.detail_embedding[-1].weight)
        nn.init.zeros_(self.detail_embedding[-1].bias)
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)

    def forward(self, values: Tensor, descriptors: Tensor, valid: Tensor) -> Tensor:
        if values.ndim != 4 or descriptors.ndim not in (2, 3):
            raise ValueError("values must be BCHW and descriptors must be CD or BCD")
        if descriptors.ndim == 2:
            descriptors = descriptors.unsqueeze(0).expand(values.shape[0], -1, -1)
        if descriptors.shape[:2] != values.shape[:2]:
            raise ValueError("one channel descriptor is required for each input channel")
        statistics = torch.stack(
            (values.mean(dim=(-2, -1)), values.std(dim=(-2, -1))), dim=-1
        )
        gates = torch.softmax(
            self.channel_gate(torch.cat((descriptors, statistics), dim=-1)), dim=1
        )
        gates = gates * values.shape[1]
        gated = values * gates.unsqueeze(-1)
        weights = self.embedding(descriptors)
        quadratic_weights = self.quadratic_embedding(descriptors)
        detail_weights = self.detail_embedding(descriptors)
        biases = self.bias(descriptors)
        local_mean = F.avg_pool2d(values, 5, stride=1, padding=2)
        detail = values - local_mean
        projected = torch.einsum("bchw,bcf->bfhw", gated, weights)
        projected += torch.einsum(
            "bchw,bcf->bfhw", gated.square() * gated.sign(), quadratic_weights
        )
        projected += torch.einsum("bchw,bcf->bfhw", detail * gates.unsqueeze(-1), detail_weights)
        projected += biases.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        projected /= math.sqrt(max(1, values.shape[1]))
        projected = self.spatial(torch.cat((projected, valid), dim=1))
        return projected + torch.tanh(self.refine_gate) * self.refine(projected)


class ConditionMoE(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, experts: int = 4) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(input_dim, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim)) for _ in range(experts)]
        )
        self.gate = nn.Linear(input_dim, experts)

    def forward(self, values: Tensor) -> Tensor:
        mixture = torch.softmax(self.gate(values), dim=-1)
        outputs = torch.stack([expert(values) for expert in self.experts], dim=1)
        return torch.einsum("be,bed->bd", mixture, outputs)


class SceneEncoder(nn.Module):
    def __init__(self, width: int = 64, hidden: int = 768, depth: int = 12, heads: int = 12) -> None:
        super().__init__()
        self.projector = ChannelProjector(width)
        self.level1 = nn.Sequential(nn.Conv2d(width, width * 2, 3, stride=2, padding=1), ResidualBlock(width * 2))
        self.level2 = nn.Sequential(nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1), ResidualBlock(width * 4))
        self.level3 = nn.Sequential(nn.Conv2d(width * 4, hidden, 3, stride=2, padding=1), ResidualBlock(hidden))
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
            [torch.sin(x[..., None] * frequencies), torch.cos(x[..., None] * frequencies),
             torch.sin(y[..., None] * frequencies), torch.cos(y[..., None] * frequencies)], dim=-1
        )
        return position.reshape(1, height * width, hidden)

    def forward(
        self,
        values: Tensor,
        descriptors: Tensor,
        valid: Tensor,
        condition: Tensor,
        modality: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        full = self.projector(values, descriptors, valid)
        half = self.level1(full)
        quarter = self.level2(half)
        eighth = self.level3(quarter)
        batch, channels, height, width = eighth.shape
        tokens = eighth.flatten(2).transpose(1, 2)
        tokens = tokens + self._position(height, width, channels, values.device).to(tokens.dtype)
        tokens = tokens + self.condition(condition).unsqueeze(1)
        shared = self.transformer(tokens).transpose(1, 2).reshape(batch, channels, height, width)
        shared = shared + self.modality_adapter[modality](eighth)
        return full, half, quarter, shared


class DynamicPhysicalDecoder(nn.Module):
    def __init__(self, width: int = 64, hidden: int = 768) -> None:
        super().__init__()
        self.up2 = nn.Conv2d(hidden, width * 4, 3, padding=1)
        self.fuse2 = nn.Sequential(nn.Conv2d(width * 8, width * 4, 3, padding=1), ResidualBlock(width * 4))
        self.up1 = nn.Conv2d(width * 4, width * 2, 3, padding=1)
        self.fuse1 = nn.Sequential(nn.Conv2d(width * 4, width * 2, 3, padding=1), ResidualBlock(width * 2))
        self.full = nn.Sequential(nn.Conv2d(width * 2, width, 3, padding=1), ResidualBlock(width))
        self.detail = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1),
            ResidualBlock(width),
            ResidualBlock(width),
        )
        self.kernel = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width + 1))
        self.detail_kernel = nn.Sequential(
            nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width + 1)
        )
        self.log_variance_kernel = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width + 1))
        self.gsd_modulation = nn.Sequential(
            nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width * 2)
        )
        nn.init.zeros_(self.detail_kernel[-1].weight)
        nn.init.zeros_(self.detail_kernel[-1].bias)
        nn.init.zeros_(self.gsd_modulation[-1].weight)
        nn.init.zeros_(self.gsd_modulation[-1].bias)
        self.radiometry = nn.ModuleDict(
            {
                "optical": nn.Conv2d(width, width, 1),
                "sar": nn.Conv2d(width, width, 1),
            }
        )

    @staticmethod
    def _dynamic(features: Tensor, parameters: Tensor) -> Tensor:
        weights, bias = parameters[..., :-1], parameters[..., -1]
        return torch.einsum("bfhw,of->bohw", features, weights) + bias.view(1, -1, 1, 1)

    def forward(
        self,
        pyramid: tuple[Tensor, Tensor, Tensor, Tensor],
        target_descriptors: Tensor,
        modality: str,
        output_size: tuple[int, int],
        scale_condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        full_source, half, quarter, shared = pyramid
        decoded = F.interpolate(self.up2(shared), size=quarter.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.fuse2(torch.cat((decoded, quarter), dim=1))
        decoded = F.interpolate(self.up1(decoded), size=half.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.fuse1(torch.cat((decoded, half), dim=1))
        decoded = F.interpolate(decoded, size=output_size, mode="bilinear", align_corners=False)
        features = self.full(decoded)
        if scale_condition is not None:
            shift, scale = self.gsd_modulation(scale_condition.float()).chunk(2, dim=-1)
            features = features * (1 + 0.1 * torch.tanh(scale)[:, :, None, None])
            features = features + 0.1 * shift[:, :, None, None]
        features = self.radiometry[modality](features)
        base = self._dynamic(features, self.kernel(target_descriptors))
        detail_features = self.detail(torch.cat((features, full_source), dim=1))
        detail = self._dynamic(detail_features, self.detail_kernel(target_descriptors))
        detail = detail - F.avg_pool2d(detail, 9, stride=1, padding=4)
        log_variance = self._dynamic(features, self.log_variance_kernel(target_descriptors)).clamp(-8.0, 3.0)
        if modality == "optical":
            mean = (torch.sigmoid(base) + 0.08 * torch.tanh(detail)).clamp(0.0, 1.0)
        else:
            mean = -20.0 + 25.0 * torch.tanh(base / 25.0)
            mean = mean + 4.0 * torch.tanh(detail / 4.0)
        return mean, log_variance


def _time_embedding(time: Tensor, dimension: int) -> Tensor:
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(0, dimension // 2, device=time.device) / max(1, dimension // 2 - 1)
    )
    values = time[:, None] * frequencies[None] * 1000.0
    return torch.cat((values.sin(), values.cos()), dim=-1)


class FlowBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden * 4))

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        shift1, scale1, shift2, scale2 = self.modulation(condition).chunk(4, dim=-1)
        normalized = self.norm1(values) * (1 + scale1[:, None]) + shift1[:, None]
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        values = values + attended
        normalized = self.norm2(values) * (1 + scale2[:, None]) + shift2[:, None]
        return values + self.mlp(normalized)


class ResidualDiT(nn.Module):
    def __init__(self, max_channels: int = 3, hidden: int = 768, depth: int = 12, heads: int = 12) -> None:
        super().__init__()
        self.max_channels = max_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(max_channels, hidden // 4, 4, stride=4), ResidualBlock(hidden // 4)
        )
        self.input = nn.Linear(hidden // 4, hidden)
        self.scene = nn.Conv2d(hidden, hidden, 1)
        self.time = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.target = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([FlowBlock(hidden, heads) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 4))
        self.amplitude_head = nn.Conv2d(hidden, max_channels, 1)
        self.decoder = nn.Sequential(
            ResidualBlock(hidden // 4), nn.ConvTranspose2d(hidden // 4, max_channels, 4, stride=4)
        )
        nn.init.zeros_(self.amplitude_head.weight)
        nn.init.constant_(self.amplitude_head.bias, math.log(0.1 / 0.9))

    def encode_residual(self, residual: Tensor) -> Tensor:
        if residual.shape[1] > self.max_channels:
            raise ValueError("visual residual has too many channels")
        padded = F.pad(residual, (0, 0, 0, 0, 0, self.max_channels - residual.shape[1]))
        return self.encoder(padded)

    def decode_residual(self, latent: Tensor, channels: int) -> Tensor:
        return self.decoder(latent)[:, :channels]

    def predict_amplitude(
        self,
        scene: Tensor,
        target_descriptors: Tensor,
        channels: int,
        output_size: tuple[int, int],
    ) -> Tensor:
        condition = self.target(target_descriptors.mean(dim=0)).view(1, -1, 1, 1)
        logits = self.amplitude_head(F.silu(scene + condition))[:, :channels]
        block_size = (output_size[0] // 4, output_size[1] // 4)
        return torch.sigmoid(
            F.interpolate(logits, size=block_size, mode="bilinear", align_corners=False)
        )

    def forward(self, latent: Tensor, time: Tensor, scene: Tensor, target_descriptors: Tensor) -> Tensor:
        batch, channels, height, width = latent.shape
        values = self.input(latent.flatten(2).transpose(1, 2))
        scene_tokens = F.interpolate(self.scene(scene), size=(height, width), mode="bilinear", align_corners=False)
        values = values + scene_tokens.flatten(2).transpose(1, 2)
        condition = self.time(_time_embedding(time, values.shape[-1]))
        condition = condition + self.target(target_descriptors.mean(dim=0)).unsqueeze(0).expand(batch, -1)
        for block in self.blocks:
            values = block(values, condition)
        return self.output(values).transpose(1, 2).reshape(batch, channels, height, width)


@dataclass
class ModelConfig:
    width: int = 64
    hidden: int = 768
    encoder_depth: int = 12
    dit_depth: int = 12
    heads: int = 12
    flow_steps: int = 16
    optical_residual_limit: float = 0.15
    sar_residual_limit_db: float = 6.0
    architecture: str = "v3.1.1"


class SentinelV3(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.encoder = SceneEncoder(cfg.width, cfg.hidden, cfg.encoder_depth, cfg.heads)
        self.decoder = DynamicPhysicalDecoder(cfg.width, cfg.hidden)
        self.residual_dit = ResidualDiT(3, cfg.hidden, cfg.dit_depth, cfg.heads)

    @staticmethod
    def descriptors(channels: Iterable[ChannelSpec], device: torch.device) -> Tensor:
        return torch.tensor([channel.descriptor() for channel in channels], device=device, dtype=torch.float32)

    @staticmethod
    def condition(
        batch: int,
        device: torch.device,
        input_gsd: float | Tensor,
        target_gsd: float | Tensor,
        metadata: Tensor | None = None,
    ) -> Tensor:
        if isinstance(input_gsd, Tensor) or isinstance(target_gsd, Tensor):
            input_values = torch.as_tensor(input_gsd, device=device, dtype=torch.float32).reshape(-1)
            target_values = torch.as_tensor(target_gsd, device=device, dtype=torch.float32).reshape(-1)
            input_values = input_values.expand(batch) if input_values.numel() == 1 else input_values
            target_values = target_values.expand(batch) if target_values.numel() == 1 else target_values
            scale = torch.stack(
                (torch.log2(input_values / 10.0), torch.log2(input_values / 10.0), torch.log2(target_values / 10.0)),
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        channel_specs = channels or sensor.channels
        descriptors = self.descriptors(channel_specs, values.device)
        condition = self.condition(values.shape[0], values.device, input_gsd, target_gsd, metadata)
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
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        pyramid = self.encode(
            values, source, valid, channels=source_channels, input_gsd=input_gsd,
            target_gsd=target_gsd, metadata=metadata
        )
        descriptors = self.descriptors(target.channels, values.device)
        scale_condition = self.condition(
            values.shape[0], values.device, input_gsd, target_gsd, metadata
        )[:, -3:]
        mean, log_variance = self.decoder(
            pyramid,
            descriptors,
            target.modality,
            values.shape[-2:],
            scale_condition,
        )
        return mean * valid, log_variance, pyramid

    def forward(self, action: str, **kwargs: object) -> object:
        if action == "physical":
            return self.physical(**kwargs)  # type: ignore[arg-type]
        if action == "encode":
            return self.encode(**kwargs)  # type: ignore[arg-type]
        if action == "flow":
            return self.flow_velocity(**kwargs)  # type: ignore[arg-type]
        raise ValueError(f"unsupported forward action: {action}")

    def flow_velocity(
        self, latent: Tensor, time: Tensor, scene: Tensor, target: SensorSpec, visual_channels: int
    ) -> Tensor:
        descriptors = self.descriptors(target.channels[:visual_channels], latent.device)
        return self.residual_dit(latent, time, scene, descriptors)

    def residual_amplitude(
        self,
        scene: Tensor,
        target: SensorSpec,
        visual_channels: int,
        output_size: tuple[int, int],
    ) -> Tensor:
        descriptors = self.descriptors(target.channels[:visual_channels], scene.device)
        normalized = self.residual_dit.predict_amplitude(
            scene, descriptors, visual_channels, output_size
        )
        limit = (
            self.config.optical_residual_limit
            if target.modality == "optical"
            else self.config.sar_residual_limit_db
        )
        return normalized * limit

    @staticmethod
    def compose_visual(physical: Tensor, residual: Tensor, modality: str) -> Tensor:
        composed = physical + residual
        return composed.clamp(0.0, 1.0) if modality == "optical" else composed

    def sample_residual(
        self,
        scene: Tensor,
        target: SensorSpec,
        shape: tuple[int, int, int, int],
        *,
        seed: int,
        steps: int | None = None,
    ) -> Tensor:
        batch, channels, height, width = shape
        generator = torch.Generator(device=scene.device).manual_seed(seed)
        latent_shape = (batch, self.config.hidden // 4, height // 4, width // 4)
        latent = torch.randn(latent_shape, generator=generator, device=scene.device, dtype=scene.dtype)
        steps = steps or self.config.flow_steps
        dt = 1.0 / steps
        for index in range(steps):
            time = torch.full((batch,), index / steps, device=scene.device, dtype=scene.dtype)
            first = self.flow_velocity(latent, time, scene, target, channels)
            proposal = latent + dt * first
            next_time = torch.full((batch,), (index + 1) / steps, device=scene.device, dtype=scene.dtype)
            second = self.flow_velocity(proposal, next_time, scene, target, channels)
            latent = latent + 0.5 * dt * (first + second)
        residual = highpass(self.residual_dit.decode_residual(latent, channels))
        block_rms = F.avg_pool2d(residual.square(), 4, stride=4).sqrt().clamp_min(1e-4)
        unit = residual / block_rms.repeat_interleave(4, -2).repeat_interleave(4, -1)
        unit = highpass(torch.tanh(unit))
        unit_rms = F.avg_pool2d(unit.square(), 4, stride=4).sqrt().clamp_min(1e-4)
        unit = unit / unit_rms.repeat_interleave(4, -2).repeat_interleave(4, -1)
        amplitude = self.residual_amplitude(scene, target, channels, (height, width))
        return unit * amplitude.repeat_interleave(4, -2).repeat_interleave(4, -1)

    @staticmethod
    def _metadata(observation: Observation, batch: int, device: torch.device) -> Tensor:
        acquired = date.fromisoformat(observation.acquired) if isinstance(observation.acquired, str) else observation.acquired
        phase = 2 * math.pi * acquired.timetuple().tm_yday / 366.0
        orbit = {"ascending": -1.0, "descending": 1.0, "unknown": 0.0}[observation.orbit]
        vector = torch.tensor((0.0, orbit, math.sin(phase), math.cos(phase), 0, 0, 0, 1), device=device)
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
            raise NotImplementedError("V3 Sentinel validation currently accepts one source observation")
        observation = observations[0]
        values = observation.values
        if values.ndim == 3:
            values = values.unsqueeze(0)
        valid = observation.valid_mask
        if valid is None:
            valid = torch.ones(values.shape[0], 1, *values.shape[-2:], device=values.device, dtype=values.dtype)
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
        if mode == "visual":
            if target.spec.modality == "optical":
                visual_indices = [target.spec.channel_names.index(name) for name in ("B04", "B03", "B02")]
            else:
                visual_indices = list(range(len(target.spec.channels)))
            base = physical[:, visual_indices]
            for sample_index in range(num_samples):
                residual = self.sample_residual(
                    pyramid[-1], target.spec, tuple(base.shape), seed=seed + sample_index
                )
                result.samples.append(
                    self.compose_visual(base, residual, target.spec.modality) * valid
                )
        return result
