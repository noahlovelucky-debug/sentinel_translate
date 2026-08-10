"""Deterministic downstream SCL probe training and scene-level statistics.

The cache contract is intentionally small so cache construction can stay separate
from the probe.  A cache is a ``torch.save``-ed mapping with these fields:

* ``sample_id``, ``scene_id``, ``tile``, and ``split``: length-B string sequences;
* ``sar``: float tensor shaped ``[B, 2, H, W]``;
* ``real_optical`` and ``synthetic_optical``: float tensors shaped ``[B, 10, H, W]``;
* ``label``: integer tensor shaped ``[B, H, W]`` with values ``-1``, ``0``, or ``1``;
* ``sar_valid``: boolean or 0/1 tensor shaped ``[B, 1, H, W]``.

The only accepted optional shape is ``[B, H, W]`` for ``sar_valid``; it is
canonicalized to ``[B, 1, H, W]`` on load.  The probe fits normalizers from
the training split's real SAR and real optical tensors only.  Synthetic optical
values always use the real-optical normalizer.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

PROBE_GROUPS: tuple[str, ...] = (
    "sar_only",
    "optical_only",
    "sar_real_optical",
    "sar_synthetic_optical",
    "synthetic_optical_only",
    "sar_mixed_optical",
)
PROBE_INPUT_CHANNELS = 12
SAR_CHANNELS = 2
OPTICAL_CHANNELS = 10
STATISTICAL_COMPARISONS: dict[str, tuple[str, str]] = {
    "C-A": ("sar_synthetic_optical", "sar_only"),
    "B-A": ("sar_real_optical", "sar_only"),
    "C-B": ("sar_synthetic_optical", "sar_real_optical"),
    "mixed-C": ("sar_mixed_optical", "sar_synthetic_optical"),
}


def cache_contract() -> dict[str, str]:
    """Return the public, minimal cache schema used by this module."""

    return {
        "sample_id": "length-B sequence[str]",
        "scene_id": "length-B sequence[str]",
        "tile": "length-B sequence[str]",
        "split": "length-B sequence[str]",
        "sar": "float32 [B, 2, H, W]",
        "real_optical": "float32 [B, 10, H, W]",
        "synthetic_optical": "float32 [B, 10, H, W]",
        "label": "int64 [B, H, W] with values -1, 0, 1",
        "sar_valid": "bool or 0/1 [B, 1, H, W]",
    }


def _as_string_tuple(value: object, name: str, batch_size: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a length-B sequence of strings")
    if len(value) != batch_size:
        raise ValueError(f"{name} has length {len(value)}, expected {batch_size}")
    return tuple(str(item) for item in value)


def _require_tensor(mapping: Mapping[str, object], name: str) -> Tensor:
    value = mapping.get(name)
    if not isinstance(value, Tensor):
        raise TypeError(f"cache field {name!r} must be a torch.Tensor")
    return value.detach()


@dataclass(frozen=True)
class ProbeCache:
    """Validated in-memory representation of one or more cached probe batches."""

    sample_id: tuple[str, ...]
    scene_id: tuple[str, ...]
    tile: tuple[str, ...]
    split: tuple[str, ...]
    sar: Tensor
    real_optical: Tensor
    synthetic_optical: Tensor
    label: Tensor
    sar_valid: Tensor

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> ProbeCache:
        """Validate one cache mapping against :func:`cache_contract`."""

        required = set(cache_contract())
        missing = sorted(required.difference(mapping))
        if missing:
            raise KeyError(f"cache is missing required fields: {missing}")

        sar = _require_tensor(mapping, "sar").float().contiguous()
        real_optical = _require_tensor(mapping, "real_optical").float().contiguous()
        synthetic_optical = _require_tensor(mapping, "synthetic_optical").float().contiguous()
        label = _require_tensor(mapping, "label").long().contiguous()
        sar_valid = _require_tensor(mapping, "sar_valid")
        if sar.ndim != 4 or sar.shape[1] != SAR_CHANNELS:
            raise ValueError("sar must have shape [B, 2, H, W]")
        batch_size, _, height, width = sar.shape
        expected_optical_shape = (batch_size, OPTICAL_CHANNELS, height, width)
        if tuple(real_optical.shape) != expected_optical_shape:
            raise ValueError("real_optical must have shape [B, 10, H, W] matching sar")
        if tuple(synthetic_optical.shape) != expected_optical_shape:
            raise ValueError("synthetic_optical must have shape [B, 10, H, W] matching sar")
        if tuple(label.shape) != (batch_size, height, width):
            raise ValueError("label must have shape [B, H, W] matching sar")
        if sar_valid.ndim == 3:
            sar_valid = sar_valid.unsqueeze(1)
        if tuple(sar_valid.shape) != (batch_size, 1, height, width):
            raise ValueError("sar_valid must have shape [B, 1, H, W] matching sar")
        if not torch.isfinite(sar).all():
            raise ValueError("sar contains non-finite values")
        if not torch.isfinite(real_optical).all():
            raise ValueError("real_optical contains non-finite values")
        if not torch.isfinite(synthetic_optical).all():
            raise ValueError("synthetic_optical contains non-finite values")
        if not bool(((sar_valid == 0) | (sar_valid == 1)).all()):
            raise ValueError("sar_valid must contain only boolean or 0/1 values")
        if not bool(((label == -1) | (label == 0) | (label == 1)).all()):
            raise ValueError("label must contain only -1, 0, or 1")
        return cls(
            sample_id=_as_string_tuple(mapping["sample_id"], "sample_id", batch_size),
            scene_id=_as_string_tuple(mapping["scene_id"], "scene_id", batch_size),
            tile=_as_string_tuple(mapping["tile"], "tile", batch_size),
            split=_as_string_tuple(mapping["split"], "split", batch_size),
            sar=sar,
            real_optical=real_optical,
            synthetic_optical=synthetic_optical,
            label=label,
            sar_valid=sar_valid.bool().contiguous(),
        )

    def __len__(self) -> int:
        return self.sar.shape[0]

    def select(self, indices: Sequence[int] | Tensor) -> ProbeCache:
        """Return a cache subset without changing sample metadata alignment."""

        if isinstance(indices, Tensor):
            index_values = [int(index) for index in indices.detach().cpu().tolist()]
        else:
            index_values = [int(index) for index in indices]
        if not index_values:
            raise ValueError("cache selection must contain at least one sample")
        if min(index_values) < 0 or max(index_values) >= len(self):
            raise IndexError("cache selection index is out of range")
        index = torch.tensor(index_values, dtype=torch.long, device=self.sar.device)
        return ProbeCache(
            sample_id=tuple(self.sample_id[value] for value in index_values),
            scene_id=tuple(self.scene_id[value] for value in index_values),
            tile=tuple(self.tile[value] for value in index_values),
            split=tuple(self.split[value] for value in index_values),
            sar=self.sar.index_select(0, index),
            real_optical=self.real_optical.index_select(0, index),
            synthetic_optical=self.synthetic_optical.index_select(0, index),
            label=self.label.index_select(0, index),
            sar_valid=self.sar_valid.index_select(0, index),
        )

    def indices_for_split(self, split: str) -> list[int]:
        return [index for index, value in enumerate(self.split) if value == split]

    def for_split(self, split: str) -> ProbeCache:
        indices = self.indices_for_split(split)
        if not indices:
            raise ValueError(f"cache has no samples for split {split!r}")
        return self.select(indices)

    @classmethod
    def concat(cls, caches: Sequence[ProbeCache]) -> ProbeCache:
        if not caches:
            raise ValueError("at least one cache is required")
        reference_shape = tuple(caches[0].sar.shape[1:])
        if any(tuple(cache.sar.shape[1:]) != reference_shape for cache in caches):
            raise ValueError("all caches must have matching [2, H, W] SAR shapes")
        return cls(
            sample_id=tuple(item for cache in caches for item in cache.sample_id),
            scene_id=tuple(item for cache in caches for item in cache.scene_id),
            tile=tuple(item for cache in caches for item in cache.tile),
            split=tuple(item for cache in caches for item in cache.split),
            sar=torch.cat([cache.sar for cache in caches], dim=0),
            real_optical=torch.cat([cache.real_optical for cache in caches], dim=0),
            synthetic_optical=torch.cat([cache.synthetic_optical for cache in caches], dim=0),
            label=torch.cat([cache.label for cache in caches], dim=0),
            sar_valid=torch.cat([cache.sar_valid for cache in caches], dim=0),
        )


def _cache_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    raw_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    resolved: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.pt")))
        else:
            resolved.append(path)
    if not resolved:
        raise FileNotFoundError("no .pt cache files were found")
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cache files do not exist: {missing}")
    return resolved


def load_probe_cache(paths: str | Path | Iterable[str | Path]) -> ProbeCache:
    """Load and concatenate one or more validated cache files."""

    caches: list[ProbeCache] = []
    for path in _cache_paths(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError(f"cache {path} must contain a mapping")
        caches.append(ProbeCache.from_mapping(payload))
    return ProbeCache.concat(caches)


@dataclass(frozen=True)
class ProbeStats:
    """Train-only modality normalizers; optical moments are always real-optical moments."""

    sar_mean: Tensor
    sar_std: Tensor
    optical_mean: Tensor
    optical_std: Tensor
    class_counts: Tensor
    class_weights: Tensor
    sar_pixels: int
    optical_pixels: int
    label_pixels: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sar_mean": self.sar_mean.flatten().tolist(),
            "sar_std": self.sar_std.flatten().tolist(),
            "optical_mean": self.optical_mean.flatten().tolist(),
            "optical_std": self.optical_std.flatten().tolist(),
            "class_counts": self.class_counts.flatten().tolist(),
            "class_weights": self.class_weights.flatten().tolist(),
            "sar_pixels": self.sar_pixels,
            "optical_pixels": self.optical_pixels,
            "label_pixels": self.label_pixels,
        }


def fit_probe_stats(
    cache: ProbeCache,
    *,
    train_split: str = "train",
    epsilon: float = 1e-6,
) -> ProbeStats:
    """Fit SAR and real-optical population moments using training samples only."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    train = cache.for_split(train_split)
    valid = train.sar_valid.float()
    valid_count = valid.sum(dim=(0, 2, 3)).clamp_min(0)
    if int(valid_count.item()) == 0:
        raise ValueError("training SAR has no valid pixels")
    sar_count = valid_count.expand(SAR_CHANNELS)
    sar_mean = (train.sar * valid).sum(dim=(0, 2, 3)) / sar_count
    sar_variance = (
        (train.sar - sar_mean.view(1, SAR_CHANNELS, 1, 1)).square() * valid
    ).sum(dim=(0, 2, 3)) / sar_count
    optical_mean = train.real_optical.mean(dim=(0, 2, 3))
    optical_variance = (
        train.real_optical - optical_mean.view(1, OPTICAL_CHANNELS, 1, 1)
    ).square().mean(dim=(0, 2, 3))
    class_counts = torch.stack(((train.label == 0).sum(), (train.label == 1).sum())).long()
    if int(class_counts.min().item()) == 0:
        raise ValueError("training labels must contain both SCL proxy classes")
    class_weights = class_counts.float().rsqrt()
    class_weights = class_weights / class_weights.mean()
    return ProbeStats(
        sar_mean=sar_mean.view(SAR_CHANNELS, 1, 1),
        sar_std=sar_variance.sqrt().clamp_min(epsilon).view(SAR_CHANNELS, 1, 1),
        optical_mean=optical_mean.view(OPTICAL_CHANNELS, 1, 1),
        optical_std=optical_variance.sqrt().clamp_min(epsilon).view(OPTICAL_CHANNELS, 1, 1),
        class_counts=class_counts,
        class_weights=class_weights,
        sar_pixels=int(valid_count.item()),
        optical_pixels=int(train.real_optical.shape[0] * train.real_optical.shape[2] * train.real_optical.shape[3]),
        label_pixels=int(class_counts.sum().item()),
    )


def _stat_on(values: Tensor, stats: Tensor) -> Tensor:
    return stats.to(device=values.device, dtype=values.dtype)


def normalize_sar(sar: Tensor, sar_valid: Tensor, stats: ProbeStats) -> Tensor:
    """Normalize SAR and force every invalid SAR location to an exact zero."""

    if sar.ndim != 4 or sar.shape[1] != SAR_CHANNELS:
        raise ValueError("sar must have shape [B, 2, H, W]")
    if tuple(sar_valid.shape) != (sar.shape[0], 1, sar.shape[2], sar.shape[3]):
        raise ValueError("sar_valid must have shape [B, 1, H, W] matching sar")
    normalized = (sar - _stat_on(sar, stats.sar_mean)) / _stat_on(sar, stats.sar_std)
    return torch.where(sar_valid.bool(), normalized, torch.zeros_like(normalized))


def normalize_optical(optical: Tensor, stats: ProbeStats) -> Tensor:
    """Normalize either real or synthetic optical values with real-optical moments."""

    if optical.ndim != 4 or optical.shape[1] != OPTICAL_CHANNELS:
        raise ValueError("optical must have shape [B, 10, H, W]")
    return (optical - _stat_on(optical, stats.optical_mean)) / _stat_on(optical, stats.optical_std)


def _pad_probe_channels(values: Tensor) -> Tensor:
    if values.ndim != 4 or values.shape[1] > PROBE_INPUT_CHANNELS:
        raise ValueError("probe input must be [B, C<=12, H, W]")
    padded = values.new_zeros(
        (values.shape[0], PROBE_INPUT_CHANNELS, values.shape[2], values.shape[3])
    )
    padded[:, : values.shape[1]] = values
    return padded


def _stable_mixed_real_mask(sample_ids: Sequence[str], device: torch.device) -> Tensor:
    values = []
    for sample_id in sample_ids:
        digest = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=8).digest()
        values.append((int.from_bytes(digest, "little") & 1) == 0)
    return torch.tensor(values, dtype=torch.bool, device=device)


def sample_mixed_real_mask(
    batch_size: int,
    *,
    generator: torch.Generator,
    probability: float = 0.5,
    device: torch.device | None = None,
) -> Tensor:
    """Sample the real/synthetic choice used only by the mixed-optical group."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    sampled = torch.rand(batch_size, generator=generator) < probability
    return sampled if device is None else sampled.to(device=device)


@dataclass(frozen=True)
class RoutedProbeInputs:
    group: str
    sar: Tensor
    optical: Tensor
    label: Tensor
    mixed_real_mask: Tensor | None = None


def route_probe_inputs(
    cache: ProbeCache,
    stats: ProbeStats,
    group: str,
    *,
    mixed_real_mask: Tensor | None = None,
    device: torch.device | str | None = None,
) -> RoutedProbeInputs:
    """Route one experimental group into fixed-width two-stream probe inputs.

    Every route returns exactly 12 SAR-stream and 12 optical-stream channels.
    Missing streams and padding channels are constructed with ``zeros_like`` and
    are therefore strictly zero rather than normalized placeholders.
    """

    if group not in PROBE_GROUPS:
        raise ValueError(f"unknown probe group {group!r}; expected one of {PROBE_GROUPS}")
    selected_device = torch.device(device) if device is not None else cache.sar.device
    sar_values = cache.sar.to(selected_device)
    sar_valid = cache.sar_valid.to(selected_device)
    real_values = cache.real_optical.to(selected_device)
    synthetic_values = cache.synthetic_optical.to(selected_device)
    sar_stream = _pad_probe_channels(normalize_sar(sar_values, sar_valid, stats))
    real_stream = _pad_probe_channels(normalize_optical(real_values, stats))
    synthetic_stream = _pad_probe_channels(normalize_optical(synthetic_values, stats))
    zero_sar = torch.zeros_like(sar_stream)
    zero_optical = torch.zeros_like(real_stream)

    if group == "sar_only":
        routed_sar, routed_optical, routed_mask = sar_stream, zero_optical, None
    elif group == "optical_only":
        routed_sar, routed_optical, routed_mask = zero_sar, real_stream, None
    elif group == "sar_real_optical":
        routed_sar, routed_optical, routed_mask = sar_stream, real_stream, None
    elif group == "sar_synthetic_optical":
        routed_sar, routed_optical, routed_mask = sar_stream, synthetic_stream, None
    elif group == "synthetic_optical_only":
        routed_sar, routed_optical, routed_mask = zero_sar, synthetic_stream, None
    else:
        if mixed_real_mask is None:
            mixed_real_mask = _stable_mixed_real_mask(cache.sample_id, selected_device)
        mask = mixed_real_mask.to(device=selected_device, dtype=torch.bool).reshape(-1)
        if mask.shape[0] != len(cache):
            raise ValueError("mixed_real_mask must have one value per batch sample")
        routed_sar = sar_stream
        routed_optical = torch.where(mask[:, None, None, None], real_stream, synthetic_stream)
        routed_mask = mask
    return RoutedProbeInputs(
        group=group,
        sar=routed_sar,
        optical=routed_optical,
        label=cache.label.to(selected_device),
        mixed_real_mask=routed_mask,
    )


def _group_norm_groups(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = _group_norm_groups(output_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class _StreamEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.level0 = _ConvBlock(PROBE_INPUT_CHANNELS, width)
        self.level1 = _ConvBlock(width, width * 2)
        self.level2 = _ConvBlock(width * 2, width * 4)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        level0 = self.level0(values)
        level1 = self.level1(F.max_pool2d(level0, kernel_size=2, ceil_mode=True))
        level2 = self.level2(F.max_pool2d(level1, kernel_size=2, ceil_mode=True))
        return level0, level1, level2


class TwoStreamLightUNet(nn.Module):
    """A fixed 12-channel-per-stream U-Net used for every probe group."""

    def __init__(self, *, width: int = 16) -> None:
        super().__init__()
        if width < 4:
            raise ValueError("width must be at least 4")
        self.width = width
        self.sar_encoder = _StreamEncoder(width)
        self.optical_encoder = _StreamEncoder(width)
        self.fuse0 = _ConvBlock(width * 2, width)
        self.fuse1 = _ConvBlock(width * 4, width * 2)
        self.fuse2 = _ConvBlock(width * 8, width * 4)
        self.bottleneck = _ConvBlock(width * 4, width * 4)
        self.decode1 = _ConvBlock(width * 6, width * 2)
        self.decode0 = _ConvBlock(width * 3, width)
        self.head = nn.Conv2d(width, 2, kernel_size=1)

    def forward(self, sar: Tensor, optical: Tensor) -> Tensor:
        expected = (PROBE_INPUT_CHANNELS, sar.shape[-2], sar.shape[-1])
        if sar.ndim != 4 or optical.ndim != 4 or tuple(sar.shape[1:]) != expected:
            raise ValueError("sar must have shape [B, 12, H, W]")
        if tuple(optical.shape) != tuple(sar.shape):
            raise ValueError("optical must have the same [B, 12, H, W] shape as sar")
        sar0, sar1, sar2 = self.sar_encoder(sar)
        optical0, optical1, optical2 = self.optical_encoder(optical)
        fused0 = self.fuse0(torch.cat((sar0, optical0), dim=1))
        fused1 = self.fuse1(torch.cat((sar1, optical1), dim=1))
        fused2 = self.fuse2(torch.cat((sar2, optical2), dim=1))
        bottleneck = self.bottleneck(fused2)
        decoded1 = self.decode1(
            torch.cat(
                (F.interpolate(bottleneck, size=fused1.shape[-2:], mode="bilinear", align_corners=False), fused1),
                dim=1,
            )
        )
        decoded0 = self.decode0(
            torch.cat(
                (F.interpolate(decoded1, size=fused0.shape[-2:], mode="bilinear", align_corners=False), fused0),
                dim=1,
            )
        )
        return self.head(decoded0)


def probe_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def fixed_group_parameter_counts(*, width: int = 16) -> dict[str, int]:
    """Construct each group model to make the equal-capacity invariant explicit."""

    return {group: probe_parameter_count(TwoStreamLightUNet(width=width)) for group in PROBE_GROUPS}


def _safe_divide(numerator: float, denominator: float) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def binary_confusion(prediction: Tensor, label: Tensor) -> Tensor:
    """Compute target-row, prediction-column 2x2 confusion while ignoring -1 labels."""

    if prediction.ndim == label.ndim + 1:
        if prediction.shape[1] != 2:
            raise ValueError("logit prediction must have two classes")
        prediction = prediction.argmax(dim=1)
    if prediction.shape != label.shape:
        raise ValueError("prediction and label must have matching [B, H, W] shapes")
    valid = label >= 0
    if not bool(valid.any()):
        return torch.zeros((2, 2), dtype=torch.long, device=label.device)
    targets = label[valid].long()
    values = prediction[valid].long()
    if not bool(((values == 0) | (values == 1)).all()):
        raise ValueError("predictions must contain only classes 0 and 1")
    return torch.bincount(targets * 2 + values, minlength=4).reshape(2, 2)


def metrics_from_confusion(
    confusion: Tensor,
    *,
    total_pixels: int | None = None,
) -> dict[str, float | int | list[list[int]]]:
    """Return binary metrics and label coverage, with NaN for undefined class values."""

    if tuple(confusion.shape) != (2, 2):
        raise ValueError("confusion must have shape [2, 2]")
    matrix = confusion.detach().to(dtype=torch.float64, device="cpu")
    total = float(matrix.sum().item())
    resolved_total_pixels = int(total) if total_pixels is None else int(total_pixels)
    if resolved_total_pixels < int(total):
        raise ValueError("total_pixels cannot be smaller than the valid label count")
    ious: list[float] = []
    f1_scores: list[float] = []
    recalls: list[float] = []
    for class_index in range(2):
        intersection = float(matrix[class_index, class_index].item())
        target_count = float(matrix[class_index, :].sum().item())
        predicted_count = float(matrix[:, class_index].sum().item())
        union = float(target_count + predicted_count - intersection)
        ious.append(_safe_divide(intersection, union))
        f1_scores.append(_safe_divide(2.0 * intersection, target_count + predicted_count))
        recalls.append(_safe_divide(intersection, target_count))
    finite_ious = [value for value in ious if math.isfinite(value)]
    finite_f1_scores = [value for value in f1_scores if math.isfinite(value)]
    finite_recalls = [value for value in recalls if math.isfinite(value)]
    macro_iou = float(np.mean(finite_ious)) if finite_ious else float("nan")
    macro_f1 = float(np.mean(finite_f1_scores)) if finite_f1_scores else float("nan")
    balanced_accuracy = float(np.mean(finite_recalls)) if finite_recalls else float("nan")
    return {
        "confusion": [[int(value) for value in row] for row in confusion.detach().cpu().tolist()],
        "support": int(total),
        "valid_pixels": int(total),
        "total_pixels": resolved_total_pixels,
        "valid_coverage": _safe_divide(total, float(resolved_total_pixels)),
        "accuracy": _safe_divide(float(torch.diagonal(matrix).sum().item()), total),
        "iou_0": ious[0],
        "iou_1": ious[1],
        "macro_iou": macro_iou,
        "f1_0": f1_scores[0],
        "f1_1": f1_scores[1],
        "macro_f1": macro_f1,
        "recall_0": recalls[0],
        "recall_1": recalls[1],
        "balanced_accuracy": balanced_accuracy,
    }


@dataclass(frozen=True)
class ProbeEvaluation:
    """Pooled and scene-level segmentation scores for one group and split."""

    per_scene: dict[str, dict[str, float | int | list[list[int]]]]
    pooled: dict[str, float | int | list[list[int]]]
    scene_equal_macro_iou: float

    def to_dict(self) -> dict[str, object]:
        return {
            "per_scene": self.per_scene,
            "pooled": self.pooled,
            "scene_equal_macro_iou": self.scene_equal_macro_iou,
        }


def evaluate_scene_predictions(
    prediction: Tensor,
    label: Tensor,
    scene_ids: Sequence[str],
) -> ProbeEvaluation:
    """Aggregate confusion and metrics independently for every scene."""

    if prediction.shape[0] != len(scene_ids) or label.shape[0] != len(scene_ids):
        raise ValueError("one scene_id is required for each prediction and label batch item")
    if prediction.ndim == label.ndim + 1:
        predicted_labels = prediction.argmax(dim=1)
    else:
        predicted_labels = prediction
    per_scene_confusion: dict[str, Tensor] = {}
    per_scene_total_pixels: dict[str, int] = {}
    for index, scene_id in enumerate(scene_ids):
        scene_confusion = binary_confusion(predicted_labels[index : index + 1], label[index : index + 1])
        if scene_id not in per_scene_confusion:
            per_scene_confusion[scene_id] = scene_confusion
            per_scene_total_pixels[scene_id] = int(label[index].numel())
        else:
            per_scene_confusion[scene_id] = per_scene_confusion[scene_id] + scene_confusion
            per_scene_total_pixels[scene_id] += int(label[index].numel())
    per_scene = {
        scene_id: metrics_from_confusion(
            per_scene_confusion[scene_id], total_pixels=per_scene_total_pixels[scene_id]
        )
        for scene_id in sorted(per_scene_confusion)
    }
    pooled_confusion = sum(
        per_scene_confusion.values(), torch.zeros((2, 2), dtype=torch.long, device=label.device)
    )
    scene_scores = [float(metrics["macro_iou"]) for metrics in per_scene.values()]
    finite_scores = [value for value in scene_scores if math.isfinite(value)]
    return ProbeEvaluation(
        per_scene=per_scene,
        pooled=metrics_from_confusion(pooled_confusion, total_pixels=int(label.numel())),
        scene_equal_macro_iou=(float(np.mean(finite_scores)) if finite_scores else float("nan")),
    )


def _concat_evaluation_batches(
    predictions: list[Tensor], labels: list[Tensor], scene_ids: list[str]
) -> ProbeEvaluation:
    if not predictions:
        raise ValueError("evaluation requires at least one batch")
    return evaluate_scene_predictions(torch.cat(predictions), torch.cat(labels), scene_ids)


def augment_routed_inputs(
    routed: RoutedProbeInputs,
    *,
    generator: torch.Generator,
) -> RoutedProbeInputs:
    """Apply one synchronized, deterministic spatial augmentation to both streams and labels."""

    horizontal = bool(torch.randint(0, 2, (1,), generator=generator).item())
    vertical = bool(torch.randint(0, 2, (1,), generator=generator).item())
    turns = int(torch.randint(0, 4, (1,), generator=generator).item())

    def transform(values: Tensor) -> Tensor:
        if horizontal:
            values = values.flip(-1)
        if vertical:
            values = values.flip(-2)
        return torch.rot90(values, turns, dims=(-2, -1))

    return RoutedProbeInputs(
        group=routed.group,
        sar=transform(routed.sar),
        optical=transform(routed.optical),
        label=transform(routed.label),
        mixed_real_mask=routed.mixed_real_mask,
    )


@dataclass(frozen=True)
class ProbeTrainConfig:
    """The single training protocol shared by every modality group and seed."""

    epochs: int = 12
    steps_per_epoch: int = 100
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    width: int = 16
    train_split: str = "train"
    dev_split: str = "dev"
    test_split: str = "test"
    mixed_real_probability: float = 0.5
    augment: bool = True

    def validate(self) -> None:
        if self.epochs < 1 or self.steps_per_epoch < 1:
            raise ValueError("epochs and steps_per_epoch must be positive")
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if not 0.0 <= self.mixed_real_probability <= 1.0:
            raise ValueError("mixed_real_probability must be in [0, 1]")
        if self.width < 4:
            raise ValueError("width must be at least 4")

    def protocol_dict(self) -> dict[str, object]:
        return {
            "augment": self.augment,
            "optimizer": "AdamW",
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "width": self.width,
            "mixed_real_probability": self.mixed_real_probability,
            "mixed_evaluation_route": "sar_synthetic_optical",
        }


@dataclass(frozen=True)
class ProbeSeedResult:
    group: str
    evaluation_input_group: str
    seed: int
    parameter_count: int
    selected_epoch: int
    selected_dev_scene_equal_macro_iou: float
    dev_history: tuple[float, ...]
    evaluations: dict[str, ProbeEvaluation]

    def to_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "evaluation_input_group": self.evaluation_input_group,
            "seed": self.seed,
            "parameter_count": self.parameter_count,
            "selected_epoch": self.selected_epoch,
            "selected_dev_scene_equal_macro_iou": self.selected_dev_scene_equal_macro_iou,
            "dev_history": list(self.dev_history),
            "evaluations": {split: result.to_dict() for split, result in self.evaluations.items()},
        }


def set_probe_determinism(seed: int) -> None:
    """Seed all local RNG sources used by the probe protocol."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model has no parameters") from error


def _evaluation_input_group(group: str) -> str:
    """Keep mixed training 50:50 while evaluating it on synthetic optical only."""

    return "sar_synthetic_optical" if group == "sar_mixed_optical" else group


@torch.inference_mode()
def evaluate_probe_model(
    model: nn.Module,
    cache: ProbeCache,
    stats: ProbeStats,
    group: str,
    *,
    batch_size: int = 16,
) -> ProbeEvaluation:
    """Evaluate a model and preserve a separate confusion matrix for each scene."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = _model_device(model)
    was_training = model.training
    model.eval()
    predictions: list[Tensor] = []
    labels: list[Tensor] = []
    scene_ids: list[str] = []
    evaluation_group = _evaluation_input_group(group)
    for start in range(0, len(cache), batch_size):
        stop = min(start + batch_size, len(cache))
        batch = cache.select(range(start, stop))
        routed = route_probe_inputs(batch, stats, evaluation_group, device=device)
        predictions.append(model(routed.sar, routed.optical).detach().cpu())
        labels.append(routed.label.detach().cpu())
        scene_ids.extend(batch.scene_id)
    if was_training:
        model.train()
    return _concat_evaluation_batches(predictions, labels, scene_ids)


def _loss_for_labels(logits: Tensor, label: Tensor, class_weights: Tensor) -> Tensor:
    if not bool((label >= 0).any()):
        return logits.sum() * 0.0
    weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    if tuple(weights.shape) != (2,):
        raise ValueError("class_weights must have shape [2]")
    return F.cross_entropy(logits, label, weight=weights, ignore_index=-1)


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_probe_seed(
    cache: ProbeCache,
    stats: ProbeStats,
    group: str,
    *,
    seed: int,
    config: ProbeTrainConfig,
    device: torch.device | str = "cpu",
) -> ProbeSeedResult:
    """Train one group/seed and choose its epoch by dev scene-equal macro IoU."""

    if group not in PROBE_GROUPS:
        raise ValueError(f"unknown probe group {group!r}")
    config.validate()
    set_probe_determinism(seed)
    selected_device = torch.device(device)
    train_cache = cache.for_split(config.train_split)
    dev_cache = cache.for_split(config.dev_split)
    model = TwoStreamLightUNet(width=config.width).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 10_003)
    augment_generator = torch.Generator(device="cpu").manual_seed(seed + 20_003)
    best_epoch = 1
    best_score = -float("inf")
    best_state = _clone_state_dict(model)
    dev_history: list[float] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        for _ in range(config.steps_per_epoch):
            indices = torch.randint(
                len(train_cache), (config.batch_size,), generator=batch_generator
            ).tolist()
            batch = train_cache.select(indices)
            mixed_mask = None
            if group == "sar_mixed_optical":
                mixed_mask = sample_mixed_real_mask(
                    len(batch),
                    generator=augment_generator,
                    probability=config.mixed_real_probability,
                )
            routed = route_probe_inputs(
                batch,
                stats,
                group,
                mixed_real_mask=mixed_mask,
                device=selected_device,
            )
            if config.augment:
                routed = augment_routed_inputs(routed, generator=augment_generator)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_labels(
                model(routed.sar, routed.optical), routed.label, stats.class_weights
            )
            loss.backward()
            optimizer.step()
        dev_evaluation = evaluate_probe_model(
            model, dev_cache, stats, group, batch_size=config.eval_batch_size
        )
        score = dev_evaluation.scene_equal_macro_iou
        dev_history.append(score)
        if math.isfinite(score) and score > best_score:
            best_epoch = epoch
            best_score = score
            best_state = _clone_state_dict(model)
    model.load_state_dict(best_state)
    evaluations: dict[str, ProbeEvaluation] = {
        config.dev_split: evaluate_probe_model(
            model, dev_cache, stats, group, batch_size=config.eval_batch_size
        )
    }
    test_indices = cache.indices_for_split(config.test_split)
    if test_indices:
        evaluations[config.test_split] = evaluate_probe_model(
            model,
            cache.select(test_indices),
            stats,
            group,
            batch_size=config.eval_batch_size,
        )
    return ProbeSeedResult(
        group=group,
        evaluation_input_group=_evaluation_input_group(group),
        seed=seed,
        parameter_count=probe_parameter_count(model),
        selected_epoch=best_epoch,
        selected_dev_scene_equal_macro_iou=best_score,
        dev_history=tuple(dev_history),
        evaluations=evaluations,
    )


@dataclass(frozen=True)
class ProbeSuiteResult:
    stats: ProbeStats
    protocol: dict[str, object]
    groups: dict[str, tuple[ProbeSeedResult, ...]]

    def to_dict(self) -> dict[str, object]:
        return {
            "stats": self.stats.to_dict(),
            "protocol": self.protocol,
            "groups": {
                group: [result.to_dict() for result in results]
                for group, results in self.groups.items()
            },
        }


def run_probe_suite(
    cache: ProbeCache,
    *,
    config: ProbeTrainConfig | None = None,
    seeds: Sequence[int] = (13, 17, 29),
    groups: Sequence[str] = PROBE_GROUPS,
    device: torch.device | str = "cpu",
) -> ProbeSuiteResult:
    """Run the same training protocol for all requested groups and three seeds by default."""

    config = ProbeTrainConfig() if config is None else config
    config.validate()
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(groups)) != len(groups) or any(group not in PROBE_GROUPS for group in groups):
        raise ValueError("groups must be a unique subset of PROBE_GROUPS")
    stats = fit_probe_stats(cache, train_split=config.train_split)
    result_groups: dict[str, tuple[ProbeSeedResult, ...]] = {}
    expected_parameters: int | None = None
    for group in groups:
        group_results = tuple(
            train_probe_seed(
                cache,
                stats,
                group,
                seed=int(seed),
                config=config,
                device=device,
            )
            for seed in seeds
        )
        counts = {result.parameter_count for result in group_results}
        if len(counts) != 1:
            raise RuntimeError("parameter count changed between seeds")
        count = counts.pop()
        if expected_parameters is None:
            expected_parameters = count
        elif count != expected_parameters:
            raise RuntimeError("parameter count changed between probe groups")
        result_groups[group] = group_results
    return ProbeSuiteResult(stats=stats, protocol=config.protocol_dict(), groups=result_groups)


def scene_scores_from_suite(
    suite: ProbeSuiteResult,
    *,
    split: str = "test",
) -> dict[str, dict[str, float]]:
    """Average each scene's macro IoU across the fixed seed set for statistics."""

    scores: dict[str, dict[str, float]] = {}
    for group, seed_results in suite.groups.items():
        by_scene: dict[str, list[float]] = {}
        for result in seed_results:
            if split not in result.evaluations:
                raise ValueError(f"group {group!r} has no {split!r} evaluation")
            for scene_id, metrics in result.evaluations[split].per_scene.items():
                score = float(metrics["macro_iou"])
                if math.isfinite(score):
                    by_scene.setdefault(scene_id, []).append(score)
        scores[group] = {
            scene_id: float(np.mean(values)) for scene_id, values in sorted(by_scene.items()) if values
        }
    return scores


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    scene_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "scene_count": self.scene_count,
        }


@dataclass(frozen=True)
class PermutationResult:
    estimate: float
    p_value: float
    scene_count: int
    exact: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "estimate": self.estimate,
            "p_value": self.p_value,
            "scene_count": self.scene_count,
            "exact": self.exact,
        }


def _paired_deltas(
    candidate: Mapping[str, float], baseline: Mapping[str, float]
) -> tuple[tuple[str, ...], np.ndarray]:
    scene_ids = tuple(
        scene_id
        for scene_id in sorted(set(candidate).intersection(baseline))
        if math.isfinite(float(candidate[scene_id])) and math.isfinite(float(baseline[scene_id]))
    )
    if not scene_ids:
        raise ValueError("paired statistics require at least one common finite scene")
    deltas = np.asarray(
        [float(candidate[scene_id]) - float(baseline[scene_id]) for scene_id in scene_ids],
        dtype=np.float64,
    )
    return scene_ids, deltas


def paired_bootstrap_scene_delta(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Percentile bootstrap of a scene-paired mean metric delta."""

    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    scene_ids, deltas = _paired_deltas(candidate, baseline)
    generator = np.random.default_rng(seed)
    sampled_means = np.empty(resamples, dtype=np.float64)
    batch_size = min(1024, resamples)
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = generator.integers(0, len(deltas), size=(stop - start, len(deltas)))
        sampled_means[start:stop] = deltas[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        estimate=float(deltas.mean()),
        ci_lower=float(np.quantile(sampled_means, alpha)),
        ci_upper=float(np.quantile(sampled_means, 1.0 - alpha)),
        scene_count=len(scene_ids),
    )


def paired_permutation_test(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    permutations: int = 10_000,
    seed: int = 0,
) -> PermutationResult:
    """Two-sided paired sign-flip permutation test of the mean scene delta."""

    if permutations < 1:
        raise ValueError("permutations must be positive")
    scene_ids, deltas = _paired_deltas(candidate, baseline)
    observed = float(deltas.mean())
    if len(deltas) <= 16:
        bits = np.arange(1 << len(deltas), dtype=np.uint32)[:, None]
        shifts = np.arange(len(deltas), dtype=np.uint32)[None, :]
        signs = np.where(((bits >> shifts) & 1) == 0, -1.0, 1.0)
        null_statistics = (signs * deltas[None, :]).mean(axis=1)
        p_value = float(np.mean(np.abs(null_statistics) >= abs(observed) - 1e-15))
        exact = True
    else:
        generator = np.random.default_rng(seed)
        exceedances = 0
        remaining = permutations
        while remaining:
            count = min(1024, remaining)
            signs = generator.integers(0, 2, size=(count, len(deltas)), dtype=np.int8) * 2 - 1
            null_statistics = (signs * deltas[None, :]).mean(axis=1)
            exceedances += int(np.count_nonzero(np.abs(null_statistics) >= abs(observed) - 1e-15))
            remaining -= count
        p_value = (exceedances + 1) / (permutations + 1)
        exact = False
    return PermutationResult(
        estimate=observed,
        p_value=p_value,
        scene_count=len(scene_ids),
        exact=exact,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm's step-down family-wise-error correction deterministically."""

    if not p_values:
        return {}
    for name, value in p_values.items():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid p-value for {name!r}: {value}")
    ordered = sorted(p_values.items(), key=lambda item: (float(item[1]), item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, float(value) * (total - rank)))
        adjusted[name] = running
    return adjusted


def oracle_headroom_recovery(group_scene_scores: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Report how much SAR-to-real-optical oracle headroom synthetic optical recovers."""

    required = ("sar_only", "sar_real_optical", "sar_synthetic_optical")
    missing = [group for group in required if group not in group_scene_scores]
    if missing:
        raise KeyError(f"oracle headroom recovery requires groups: {missing}")
    scene_ids = sorted(
        set(group_scene_scores["sar_only"])
        .intersection(group_scene_scores["sar_real_optical"])
        .intersection(group_scene_scores["sar_synthetic_optical"])
    )
    finite_scenes = [
        scene_id
        for scene_id in scene_ids
        if all(
            math.isfinite(float(group_scene_scores[group][scene_id])) for group in required
        )
    ]
    if not finite_scenes:
        raise ValueError("oracle headroom recovery requires common finite scene scores")
    oracle_gain = float(
        np.mean(
            [
                group_scene_scores["sar_real_optical"][scene_id]
                - group_scene_scores["sar_only"][scene_id]
                for scene_id in finite_scenes
            ]
        )
    )
    synthetic_gain = float(
        np.mean(
            [
                group_scene_scores["sar_synthetic_optical"][scene_id]
                - group_scene_scores["sar_only"][scene_id]
                for scene_id in finite_scenes
            ]
        )
    )
    recovery = _safe_divide(synthetic_gain, oracle_gain)
    return {
        "scene_count": len(finite_scenes),
        "oracle_gain": oracle_gain,
        "synthetic_gain": synthetic_gain,
        "recovery": recovery,
    }


def summarize_probe_statistics(
    group_scene_scores: Mapping[str, Mapping[str, float]],
    *,
    bootstrap_resamples: int = 10_000,
    permutation_samples: int = 10_000,
    seed: int = 0,
    synthetic_ci_lower_threshold: float = 0.02,
) -> dict[str, object]:
    """Run paired scene statistics, Holm correction, gate, and recovery summary."""

    missing_groups = sorted(
        {
            group
            for comparison in STATISTICAL_COMPARISONS.values()
            for group in comparison
            if group not in group_scene_scores
        }
    )
    if missing_groups:
        raise KeyError(f"missing groups for statistical summary: {missing_groups}")
    comparisons: dict[str, dict[str, object]] = {}
    raw_p_values: dict[str, float] = {}
    for index, (name, (candidate_group, baseline_group)) in enumerate(
        STATISTICAL_COMPARISONS.items()
    ):
        bootstrap = paired_bootstrap_scene_delta(
            group_scene_scores[candidate_group],
            group_scene_scores[baseline_group],
            resamples=bootstrap_resamples,
            seed=seed + index * 2,
        )
        permutation = paired_permutation_test(
            group_scene_scores[candidate_group],
            group_scene_scores[baseline_group],
            permutations=permutation_samples,
            seed=seed + index * 2 + 1,
        )
        raw_p_values[name] = permutation.p_value
        comparisons[name] = {
            "candidate": candidate_group,
            "baseline": baseline_group,
            "bootstrap": bootstrap.to_dict(),
            "permutation": permutation.to_dict(),
        }
    adjusted = holm_adjust(raw_p_values)
    for name, value in adjusted.items():
        comparisons[name]["holm_p_value"] = value
    real_gain = float(comparisons["B-A"]["bootstrap"]["estimate"])  # type: ignore[index]
    synthetic_ci_lower = float(comparisons["C-A"]["bootstrap"]["ci_lower"])  # type: ignore[index]
    gate = real_gain > 0.0 and synthetic_ci_lower > synthetic_ci_lower_threshold
    return {
        "comparisons": comparisons,
        "holm_method": "Holm step-down over C-A, B-A, C-B, mixed-C",
        "gate": {
            "passed": gate,
            "real_gain_B_minus_A": real_gain,
            "synthetic_ci_lower_C_minus_A": synthetic_ci_lower,
            "synthetic_ci_lower_threshold": synthetic_ci_lower_threshold,
            "rule": "B-A > 0 and lower(C-A 95% paired-bootstrap CI) > 0.02",
        },
        "oracle_headroom_recovery": oracle_headroom_recovery(group_scene_scores),
    }
