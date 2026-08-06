from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def db_to_intensity(values_db: Tensor) -> Tensor:
    return torch.pow(10.0, values_db / 10.0)


def intensity_to_db(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    return 10.0 * torch.log10(values.clamp_min(epsilon))


def normalized_sar_to_db(values: Tensor) -> Tensor:
    """Convert the legacy per-polarization [-1, 1] representation to dB."""
    minimum = values.new_tensor((-35.0, -45.0)).view(1, 2, 1, 1)
    maximum = values.new_tensor((5.0, -5.0)).view(1, 2, 1, 1)
    return (values + 1.0) * 0.5 * (maximum - minimum) + minimum


def db_to_normalized_sar(values: Tensor) -> Tensor:
    minimum = values.new_tensor((-35.0, -45.0)).view(1, 2, 1, 1)
    maximum = values.new_tensor((5.0, -5.0)).view(1, 2, 1, 1)
    return 2.0 * (values - minimum) / (maximum - minimum) - 1.0


def normalized_s2_to_reflectance(values: Tensor) -> Tensor:
    return (values + 1.0) * 0.5


def reflectance_to_normalized_s2(values: Tensor) -> Tensor:
    return values * 2.0 - 1.0


def _blur(values: Tensor, sigma: float) -> Tensor:
    if sigma <= 0:
        return values
    radius = max(1, math.ceil(3.0 * sigma))
    locations = torch.arange(-radius, radius + 1, device=values.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (locations / sigma).square())
    kernel /= kernel.sum()
    channels = values.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    working = values.float()
    working = F.conv2d(working, horizontal, padding=(0, radius), groups=channels)
    working = F.conv2d(working, vertical, padding=(radius, 0), groups=channels)
    return working.to(values.dtype)


def physical_resample(
    values: Tensor,
    *,
    modality: str,
    source_gsd_m: float,
    target_gsd_m: float,
    restore_grid: bool = True,
) -> Tensor:
    if source_gsd_m <= 0 or target_gsd_m <= 0:
        raise ValueError("GSD values must be positive")
    if target_gsd_m < source_gsd_m:
        raise ValueError("synthetic views cannot invent resolution")
    ratio = target_gsd_m / source_gsd_m
    if math.isclose(ratio, 1.0):
        return values
    output_size = (max(1, round(values.shape[-2] / ratio)), max(1, round(values.shape[-1] / ratio)))
    sigma = max(0.0, 0.5 * math.sqrt(ratio * ratio - 1.0))
    physical = db_to_intensity(values) if modality == "sar" else values
    # Area interpolation is the exact box/MTF integration for integer Sentinel views.
    # A Gaussian prefilter is only needed for non-integer scale ratios.
    filtered = physical if float(ratio).is_integer() else _blur(physical, sigma)
    reduced = F.interpolate(filtered.float(), size=output_size, mode="area")
    if restore_grid:
        reduced = F.interpolate(reduced, size=values.shape[-2:], mode="bilinear", align_corners=False)
    if modality == "sar":
        reduced = intensity_to_db(reduced)
    return reduced.to(values.dtype)


def gsd_condition(input_gsd_m: float, working_gsd_m: float, target_gsd_m: float) -> Tensor:
    values = (
        math.log2(input_gsd_m / 10.0),
        math.log2(input_gsd_m / working_gsd_m),
        math.log2(target_gsd_m / 10.0),
    )
    return torch.tensor(values, dtype=torch.float32)
