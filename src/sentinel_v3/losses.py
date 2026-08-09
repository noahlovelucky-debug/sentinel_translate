from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)


def gradients(values: Tensor) -> tuple[Tensor, Tensor]:
    return values[..., 1:, :] - values[..., :-1, :], values[..., :, 1:] - values[..., :, :-1]


def charbonnier(values: Tensor, epsilon: float = 1e-3) -> Tensor:
    return torch.sqrt(values.square() + epsilon * epsilon)


def haar_dwt2(values: Tensor) -> Tensor:
    """Return the orthonormal 2D Haar coefficients in LL, LH, HL, HH order."""

    if values.ndim != 4:
        raise ValueError("Haar DWT expects BCHW values")
    height, width = values.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError("Haar DWT requires even spatial dimensions")
    top_left = values[..., 0::2, 0::2]
    top_right = values[..., 0::2, 1::2]
    bottom_left = values[..., 1::2, 0::2]
    bottom_right = values[..., 1::2, 1::2]
    half = values.new_tensor(0.5)
    return torch.stack(
        (
            (top_left + top_right + bottom_left + bottom_right) * half,
            (top_left - top_right + bottom_left - bottom_right) * half,
            (top_left + top_right - bottom_left - bottom_right) * half,
            (top_left - top_right - bottom_left + bottom_right) * half,
        ),
        dim=2,
    )


def haar_idwt2(coefficients: Tensor) -> Tensor:
    """Invert :func:`haar_dwt2` without interpolation or learned filtering."""

    if coefficients.ndim != 5 or coefficients.shape[2] != 4:
        raise ValueError("Haar IDWT expects BC4HW coefficients")
    ll, lh, hl, hh = coefficients.unbind(dim=2)
    half = coefficients.new_tensor(0.5)
    height, width = ll.shape[-2:]
    values = ll.new_empty(*ll.shape[:-2], height * 2, width * 2)
    values[..., 0::2, 0::2] = (ll + lh + hl + hh) * half
    values[..., 0::2, 1::2] = (ll - lh + hl - hh) * half
    values[..., 1::2, 0::2] = (ll + lh - hl - hh) * half
    values[..., 1::2, 1::2] = (ll - lh - hl + hh) * half
    return values


def haar_packet_dwt2(values: Tensor) -> Tensor:
    """Return a two-level orthonormal Haar packet with flattened 16C channels."""

    if values.ndim != 4:
        raise ValueError("Haar packet DWT expects BCHW values")
    height, width = values.shape[-2:]
    if height % 4 or width % 4:
        raise ValueError("Haar packet DWT requires dimensions divisible by four")
    first = haar_dwt2(values).flatten(1, 2)
    return haar_dwt2(first).flatten(1, 2)


def haar_packet_idwt2(coefficients: Tensor) -> Tensor:
    """Invert :func:`haar_packet_dwt2` exactly from flattened packet channels."""

    if coefficients.ndim != 4:
        raise ValueError("Haar packet IDWT expects BCHW coefficients")
    batch, channels, height, width = coefficients.shape
    if channels % 16:
        raise ValueError("Haar packet IDWT requires a channel count divisible by sixteen")
    source_channels = channels // 16
    first = haar_idwt2(coefficients.reshape(batch, source_channels * 4, 4, height, width))
    return haar_idwt2(
        first.reshape(batch, source_channels, 4, first.shape[-2], first.shape[-1])
    )


def complex_magnitude(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Differentiable complex magnitude with a defined derivative at zero."""

    return torch.sqrt(values.real.square() + values.imag.square() + epsilon * epsilon)


def robust_rms(values: Tensor, mask: Tensor, block_size: int = 4) -> Tensor:
    squared = values.square()
    # Soft clipping keeps a few strong reflectors from setting every texture amplitude.
    scale = torch.sqrt(masked_mean(squared, mask) + 1e-8).detach()
    clipped = squared.clamp_max(9.0 * scale.square().clamp_min(1e-8))
    pooled = F.avg_pool2d(clipped * mask, block_size, stride=block_size)
    counts = F.avg_pool2d(mask, block_size, stride=block_size).clamp_min(1e-6)
    return torch.sqrt(pooled / counts + 1e-8)


def edge_f1_surrogate(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    pred_dy, pred_dx = gradients(prediction)
    target_dy, target_dx = gradients(target)
    pred_edge = F.pad(pred_dy.abs().mean(1, keepdim=True), (0, 0, 0, 1))
    pred_edge += F.pad(pred_dx.abs().mean(1, keepdim=True), (0, 1))
    target_edge = F.pad(target_dy.abs().mean(1, keepdim=True), (0, 0, 0, 1))
    target_edge += F.pad(target_dx.abs().mean(1, keepdim=True), (0, 1))
    scale = target_edge.flatten(1).mean(1).view(-1, 1, 1, 1).clamp_min(1e-4)
    pred_probability = 1.0 - torch.exp(-pred_edge / scale)
    target_probability = 1.0 - torch.exp(-target_edge / scale)
    intersection = masked_mean(pred_probability * target_probability, mask)
    denominator = masked_mean(pred_probability + target_probability, mask).clamp_min(1e-6)
    return 1.0 - 2.0 * intersection / denominator


def deterministic_detail_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    sample_weight: Tensor | None = None,
    scale: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    prediction = prediction / scale
    target = target / scale
    weight = mask
    if sample_weight is not None:
        weight = weight * sample_weight.view(-1, 1, 1, 1)
    reconstruction = masked_mean(charbonnier(prediction - target), weight)
    pred_dy, pred_dx = gradients(prediction)
    target_dy, target_dx = gradients(target)
    gradient = masked_mean(charbonnier(pred_dy - target_dy), weight[..., 1:, :])
    gradient += masked_mean(charbonnier(pred_dx - target_dx), weight[..., :, 1:])
    edge = edge_f1_surrogate(prediction, target, weight)
    local_structure = structural_loss(prediction, target, weight)
    active = (weight.sum() > 0).to(prediction.dtype)
    edge = edge * active
    local_structure = local_structure * active
    # MAE is the hard deterministic-detail gate; auxiliary terms shape edges without
    # rewarding a dense hallucinated residual that is worse than the zero baseline.
    sparsity = masked_mean(prediction.abs(), weight)
    total = (
        reconstruction + 0.1 * gradient + 0.03 * edge + 0.02 * local_structure + 0.02 * sparsity
    )
    return total, {
        "detail_charbonnier": reconstruction.detach(),
        "detail_gradient": gradient.detach(),
        "detail_edge": edge.detach(),
        "detail_ssim": local_structure.detach(),
        "detail_sparsity": sparsity.detach(),
    }


def local_spectrum_loss(
    prediction: Tensor, target: Tensor, mask: Tensor, tile_size: int = 16
) -> Tensor:
    height = min(tile_size, prediction.shape[-2])
    width = min(tile_size, prediction.shape[-1])
    usable_height = prediction.shape[-2] // height * height
    usable_width = prediction.shape[-1] // width * width

    def tiles(values: Tensor) -> Tensor:
        values = values[..., :usable_height, :usable_width]
        return values.unfold(2, height, height).unfold(3, width, width)

    tiled_mask = tiles(mask)
    pred_spectrum = torch.fft.rfft2(
        (tiles(prediction) * tiled_mask).float(), dim=(-2, -1), norm="ortho"
    )
    pred_spectrum = complex_magnitude(pred_spectrum)
    target_spectrum = torch.fft.rfft2(
        (tiles(target) * tiled_mask).float(), dim=(-2, -1), norm="ortho"
    )
    target_spectrum = complex_magnitude(target_spectrum)
    distance = (
        (torch.log1p(pred_spectrum) - torch.log1p(target_spectrum)).abs().mean(dim=(1, 4, 5))
    )
    tile_weight = tiled_mask.mean(dim=(1, 4, 5))
    return (distance * tile_weight).sum() / tile_weight.sum().clamp_min(1e-8)


def codec_reconstruction_loss(
    prediction: Tensor, target: Tensor, mask: Tensor, modality: str
) -> tuple[Tensor, dict[str, Tensor]]:
    scale = 0.08 if modality == "optical" else 4.0
    prediction = prediction / scale
    target = target / scale
    reconstruction = masked_mean(charbonnier(prediction - target), mask)
    pred_dy, pred_dx = gradients(prediction)
    target_dy, target_dx = gradients(target)
    gradient = masked_mean(charbonnier(pred_dy - target_dy), mask[..., 1:, :])
    gradient += masked_mean(charbonnier(pred_dx - target_dx), mask[..., :, 1:])
    spectrum = log_spectral_distance(prediction, target, mask)
    local_spectrum = local_spectrum_loss(prediction, target, mask)
    total = reconstruction + 0.2 * gradient + 0.15 * spectrum + 0.1 * local_spectrum
    metrics = {
        "codec_charbonnier": reconstruction.detach(),
        "codec_gradient": gradient.detach(),
        "codec_spectrum": spectrum.detach(),
        "codec_local_spectrum": local_spectrum.detach(),
    }
    if modality == "optical":
        structure = structural_loss(
            F.avg_pool2d(prediction, 2), F.avg_pool2d(target, 2), F.avg_pool2d(mask, 2)
        )
        structure = structure * (mask.sum() > 0).to(prediction.dtype)
        total = total + 0.1 * structure
        metrics["codec_structure"] = structure.detach()
    else:
        pred_variance = (
            F.avg_pool2d(prediction.square(), 9, 1, 4)
            - F.avg_pool2d(prediction, 9, 1, 4).square()
        )
        target_variance = (
            F.avg_pool2d(target.square(), 9, 1, 4) - F.avg_pool2d(target, 9, 1, 4).square()
        )
        variance = masked_mean(
            (pred_variance.clamp_min(0).sqrt() - target_variance.clamp_min(0).sqrt()).abs(),
            mask,
        )
        total = total + 0.2 * variance
        metrics["codec_local_variance"] = variance.detach()
        metrics["codec_speckle_scale"] = variance.detach()
    return total, metrics


def spectral_angle(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    cosine = F.cosine_similarity(prediction, target, dim=1, eps=1e-6).clamp(-1 + 1e-6, 1 - 1e-6)
    return masked_mean(torch.acos(cosine).unsqueeze(1), mask)


def structural_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mu_x = F.avg_pool2d(prediction, 7, stride=1, padding=3)
    mu_y = F.avg_pool2d(target, 7, stride=1, padding=3)
    var_x = F.avg_pool2d(prediction.square(), 7, stride=1, padding=3) - mu_x.square()
    var_y = F.avg_pool2d(target.square(), 7, stride=1, padding=3) - mu_y.square()
    covariance = F.avg_pool2d(prediction * target, 7, stride=1, padding=3) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + 1e-4) * (2 * covariance + 9e-4)) / (
        (mu_x.square() + mu_y.square() + 1e-4) * (var_x + var_y + 9e-4)
    )
    return 1.0 - masked_mean(ssim, mask)


def physical_loss(
    prediction: Tensor,
    log_variance: Tensor,
    target: Tensor,
    mask: Tensor,
    modality: str,
    sample_weight: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    weight = sample_weight.view(-1, 1, 1, 1) * mask
    error = prediction - target
    unit_scale = 0.05 if modality == "optical" else 5.0
    normalized_error = error / unit_scale
    huber = masked_mean(
        F.huber_loss(
            prediction, target, reduction="none", delta=0.05 if modality == "optical" else 1.0
        ),
        weight,
    )
    mse = masked_mean(error.square(), weight)
    nll = masked_mean(0.5 * (error.square() * torch.exp(-log_variance) + log_variance), weight)
    normalized_huber = masked_mean(
        F.huber_loss(normalized_error, torch.zeros_like(normalized_error), reduction="none"),
        weight,
    )
    normalized_mse = masked_mean(normalized_error.square(), weight)
    normalized_log_variance = log_variance - 2.0 * math.log(unit_scale)
    normalized_nll = masked_mean(
        0.5
        * (
            normalized_error.square() * torch.exp(-normalized_log_variance)
            + normalized_log_variance
        ),
        weight,
    )
    pred_dy, pred_dx = gradients(prediction)
    target_dy, target_dx = gradients(target)
    gradient = masked_mean((pred_dy - target_dy).abs(), weight[..., 1:, :])
    gradient += masked_mean((pred_dx - target_dx).abs(), weight[..., :, 1:])
    structure = structural_loss(prediction / unit_scale, target / unit_scale, weight)
    expanded_mask = mask.expand_as(error)
    sample_denominator = expanded_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    sample_bias = (error * expanded_mask).sum(dim=(1, 2, 3)) / sample_denominator
    active_samples = sample_weight * (sample_denominator > 1.0).to(sample_weight.dtype)
    global_bias = (sample_bias.abs() * active_samples).sum() / active_samples.sum().clamp_min(
        1e-8
    )
    channel_denominator = mask.sum(dim=(2, 3)).clamp_min(1.0)
    channel_bias_values = (error * mask).sum(dim=(2, 3)) / channel_denominator
    channel_bias = masked_mean(channel_bias_values.abs(), active_samples[:, None])
    normalized_gradient = gradient / unit_scale
    total = (
        normalized_huber
        + 0.5 * normalized_mse
        + 0.01 * normalized_nll
        + 0.1 * structure
        + 0.05 * normalized_gradient
    )
    metrics = {
        "huber": huber.detach(),
        "mse": mse.detach(),
        "rmse": torch.sqrt(mse.detach().clamp_min(0.0)),
        "nll": nll.detach(),
        "ssim_loss": structure.detach(),
        "gradient": gradient.detach(),
        "bias": global_bias.detach(),
        "channel_bias": channel_bias.detach(),
        "normalized_huber": normalized_huber.detach(),
        "normalized_mse": normalized_mse.detach(),
    }
    if modality == "optical":
        sam = spectral_angle(prediction, target, weight)
        magnitude_error = (
            torch.linalg.vector_norm(prediction, dim=1, keepdim=True)
            - torch.linalg.vector_norm(target, dim=1, keepdim=True)
        ).abs()
        magnitude = masked_mean(magnitude_error, weight)
        sam_gate_radians = math.radians(5.716)
        magnitude_scale = math.sqrt(prediction.shape[1]) * 0.05
        total = (
            total
            + 0.1 * sam / sam_gate_radians
            + 0.1 * magnitude / magnitude_scale
            + 0.1 * channel_bias / 0.05
        )
        metrics["sam"] = sam.detach()
        metrics["spectral_magnitude"] = magnitude.detach()
    else:
        relation = masked_mean(
            ((prediction[:, :1] - prediction[:, 1:2]) - (target[:, :1] - target[:, 1:2])).abs(),
            weight,
        )
        total = (
            total
            + 0.05 * relation / unit_scale
            + 0.5 * global_bias / 0.5
            + 0.1 * channel_bias / 0.5
        )
        metrics["polarization"] = relation.detach()
    return total, metrics


def latent_alignment(
    sar_scene: Tensor, optical_scene: Tensor, mask: Tensor, temperature: float = 0.07
) -> tuple[Tensor, dict[str, Tensor]]:
    sar_global = F.normalize(sar_scene.flatten(2).mean(-1), dim=-1)
    optical_global = F.normalize(optical_scene.flatten(2).mean(-1), dim=-1)
    logits = sar_global @ optical_global.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    info_nce = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    local_mask = F.interpolate(mask, size=sar_scene.shape[-2:], mode="area")
    dense = 1.0 - F.cosine_similarity(sar_scene, optical_scene, dim=1).unsqueeze(1)
    dense_loss = masked_mean(dense, local_mask)
    return info_nce + 0.25 * dense_loss, {
        "info_nce": info_nce.detach(),
        "dense_cosine": dense_loss.detach(),
    }


def low_frequency_loss(
    residual: Tensor, mask: Tensor, sample_weight: Tensor | None = None
) -> Tensor:
    low = F.avg_pool2d(residual, 4, stride=4)
    low_mask = F.avg_pool2d(mask, 4, stride=4)
    if sample_weight is not None:
        low_mask = low_mask * sample_weight.view(-1, 1, 1, 1)
    return masked_mean(low.square(), low_mask)


def frequency_bands(values: Tensor, levels: int = 3) -> tuple[Tensor, ...]:
    """Return fine-to-coarse Laplacian bands on the original pixel grid."""
    if levels < 1:
        raise ValueError("frequency decomposition needs at least one level")
    smallest = 2**levels
    if values.shape[-2] % smallest or values.shape[-1] % smallest:
        raise ValueError(f"frequency dimensions must be divisible by {smallest}")
    bands: list[Tensor] = []
    current = values
    output_size = values.shape[-2:]
    for _ in range(levels):
        low = F.avg_pool2d(current, 2, stride=2)
        reconstructed = F.interpolate(
            low, size=current.shape[-2:], mode="bilinear", align_corners=False
        )
        band = current - reconstructed
        bands.append(
            F.interpolate(band, size=output_size, mode="bilinear", align_corners=False)
            if band.shape[-2:] != output_size
            else band
        )
        current = low
    return tuple(bands)


def highpass(values: Tensor, block_size: int = 8) -> Tensor:
    levels = int(math.log2(block_size))
    if 2**levels != block_size:
        raise ValueError("highpass block_size must be a power of two")
    return sum(frequency_bands(values, levels), torch.zeros_like(values))


def detail_reliability_target(
    source: Tensor,
    target_bands: tuple[Tensor, ...],
    mask: Tensor,
) -> Tensor:
    """Soft oracle used only to teach whether target edges are supported by the source."""
    source_gray = source.float().mean(dim=1, keepdim=True)
    source_gray = (
        source_gray - source_gray.mean(dim=(-2, -1), keepdim=True)
    ) / source_gray.std(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    source_dy, source_dx = gradients(source_gray)
    source_energy = F.pad(source_dx.abs(), (0, 1, 0, 0)) + F.pad(source_dy.abs(), (0, 0, 0, 1))
    source_energy = source_energy / source_energy.mean(dim=(-2, -1), keepdim=True).clamp_min(
        1e-3
    )
    targets = []
    for level, band in enumerate(target_bands):
        target_energy = band.float().abs().mean(dim=1, keepdim=True)
        target_energy = target_energy / target_energy.mean(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-4)
        scale = 2**level
        support = torch.minimum(source_energy, target_energy).clamp(0.0, 2.0) / 2.0
        if scale > 1:
            support = F.avg_pool2d(support, scale, stride=scale)
            support = F.interpolate(
                support, size=mask.shape[-2:], mode="bilinear", align_corners=False
            )
        support = F.avg_pool2d(support * mask, 4, stride=4)
        targets.append(support.clamp(0.0, 1.0))
    return torch.cat(targets, dim=1)


def cross_modal_identifiability_target(
    source_features: Tensor,
    target_bands: tuple[Tensor, ...],
    mask: Tensor,
) -> Tensor:
    """Measure local cross-modal frequency support for each target band."""

    if source_features.ndim != 4 or mask.ndim != 4:
        raise ValueError("identifiability inputs must be BCHW")
    source_energy = highpass(source_features.float()).abs().mean(dim=1, keepdim=True)
    valid = mask.float()
    targets = []
    for level, band in enumerate(target_bands):
        if band.shape[-2:] != source_energy.shape[-2:]:
            raise ValueError("target bands must match source feature resolution")
        scale = 2**level
        if scale > 1:
            kernel = 2 * scale - 1
            source = F.avg_pool2d(source_energy, kernel, stride=1, padding=scale - 1)
            target = F.avg_pool2d(
                band.float().abs().mean(dim=1, keepdim=True),
                kernel,
                stride=1,
                padding=scale - 1,
            )
        else:
            source = source_energy
            target = band.float().abs().mean(dim=1, keepdim=True)
        source = source * valid
        target = target * valid
        source_target = F.avg_pool2d(source * target, 4, stride=4)
        source_square = F.avg_pool2d(source.square(), 4, stride=4)
        target_square = F.avg_pool2d(target.square(), 4, stride=4)
        cosine = source_target / torch.sqrt(source_square * target_square + 1e-8)
        source_mean = F.avg_pool2d(source, 4, stride=4)
        target_mean = F.avg_pool2d(target, 4, stride=4)
        balance = 2.0 * torch.sqrt(source_mean * target_mean) / (
            source_mean + target_mean + 1e-8
        )
        coverage = F.avg_pool2d(valid, 4, stride=4)
        support = (coverage >= 0.999).to(cosine.dtype)
        targets.append((cosine.clamp_min(0.0) * balance * support).clamp(0.0, 1.0))
    return torch.cat(targets, dim=1)


def phase_identifiability_target(
    source: Tensor,
    target_bands: tuple[Tensor, ...],
    mask: Tensor,
) -> Tensor:
    """Return blockwise cross-modal orientation support for three frequency bands.

    The squared gradient dot product makes the target invariant to contrast-sign
    reversals while still rejecting perpendicular structure.  It is a train-only
    oracle: inference never receives target pixels through this path.
    """

    if source.ndim != 4 or mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("phase identifiability inputs require source BCHW and mask B1HW")
    if len(target_bands) != 3:
        raise ValueError("phase identifiability requires fine, mid, and coarse target bands")
    batch, _, height, width = source.shape
    if (
        mask.shape[0] != batch
        or mask.shape[-2:] != (height, width)
        or height % 4
        or width % 4
    ):
        raise ValueError("phase identifiability inputs must share dimensions divisible by four")
    for band in target_bands:
        if not isinstance(band, Tensor) or band.ndim != 4:
            raise ValueError("phase target bands must be BCHW tensors")
        if band.shape[0] != batch or band.shape[-2:] != (height, width):
            raise ValueError("phase target bands must match the source resolution")

    source_gray = source.float().mean(dim=1, keepdim=True)
    source_gray = (source_gray - source_gray.mean(dim=(-2, -1), keepdim=True)) / (
        source_gray.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
    )
    valid = mask.float()
    targets: list[Tensor] = []
    for level, band in enumerate(target_bands):
        scale = 2**level
        if scale > 1:
            source_level = F.interpolate(
                F.avg_pool2d(source_gray, scale, stride=scale),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        else:
            source_level = source_gray
        target_level = band.float().mean(dim=1, keepdim=True)
        source_dy, source_dx = gradients(source_level)
        target_dy, target_dx = gradients(target_level)
        source_dy = F.pad(source_dy, (0, 0, 0, 1))
        source_dx = F.pad(source_dx, (0, 1, 0, 0))
        target_dy = F.pad(target_dy, (0, 0, 0, 1))
        target_dx = F.pad(target_dx, (0, 1, 0, 0))
        source_energy = source_dx.square() + source_dy.square()
        target_energy = target_dx.square() + target_dy.square()
        valid_count = valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        source_mean_energy = (source_energy * valid).sum(
            dim=(-2, -1), keepdim=True
        ) / valid_count
        target_mean_energy = (target_energy * valid).sum(
            dim=(-2, -1), keepdim=True
        ) / valid_count
        source_dx = source_dx / source_mean_energy.clamp_min(1e-12).sqrt()
        source_dy = source_dy / source_mean_energy.clamp_min(1e-12).sqrt()
        target_dx = target_dx / target_mean_energy.clamp_min(1e-12).sqrt()
        target_dy = target_dy / target_mean_energy.clamp_min(1e-12).sqrt()
        source_normalized = source_dx.square() + source_dy.square()
        target_normalized = target_dx.square() + target_dy.square()
        dot = source_dx * target_dx + source_dy * target_dy
        coherence = dot.square() / (source_normalized * target_normalized + 1e-8)
        balance = 2.0 * torch.sqrt(source_normalized * target_normalized) / (
            source_normalized + target_normalized + 1e-8
        )
        score = F.avg_pool2d(coherence * balance * valid, 4, stride=4)
        coverage = F.avg_pool2d(valid, 4, stride=4)
        targets.append(torch.where(coverage >= 0.999, score, torch.zeros_like(score)))
    return torch.cat(targets, dim=1).clamp(0.0, 1.0)


def phase_alignment_loss(
    source_phase: Tensor,
    target_bands: tuple[Tensor, ...],
    mask: Tensor,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Align source phase maps with target-band orientations during training only."""

    if source_phase.ndim != 4 or source_phase.shape[1] != 3:
        raise ValueError("phase alignment source maps must be B3HW")
    if len(target_bands) != 3:
        raise ValueError("phase alignment requires three target bands")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("phase alignment mask must be B1HW")
    batch, _, height, width = source_phase.shape
    if mask.shape != (batch, 1, height, width):
        raise ValueError("phase alignment mask must match source phase dimensions")
    for band in target_bands:
        if not isinstance(band, Tensor) or band.ndim != 4:
            raise ValueError("phase alignment target bands must be BCHW tensors")
        if band.shape[0] != batch or band.shape[-2:] != (height, width):
            raise ValueError("phase alignment target bands must match source maps")
    if sample_weight is not None and sample_weight.shape not in {(batch,), (batch, 1)}:
        raise ValueError("phase alignment sample weights must be B or B1")

    target_phase = torch.cat(
        tuple(band.float().mean(dim=1, keepdim=True) for band in target_bands), dim=1
    )
    source_dy, source_dx = gradients(source_phase.float())
    target_dy, target_dx = gradients(target_phase)
    source_dy = F.pad(source_dy, (0, 0, 0, 1))
    source_dx = F.pad(source_dx, (0, 1, 0, 0))
    target_dy = F.pad(target_dy, (0, 0, 0, 1))
    target_dx = F.pad(target_dx, (0, 1, 0, 0))
    source_energy = source_dx.square() + source_dy.square()
    target_energy = target_dx.square() + target_dy.square()
    epsilon = 1e-6
    orientation = (source_dx * target_dx + source_dy * target_dy).square() / (
        (source_energy + epsilon) * (target_energy + epsilon)
    )
    weight = target_energy * mask.float()
    if sample_weight is not None:
        weight = weight * sample_weight.to(weight).reshape(batch, 1, 1, 1)
    denominator = weight.sum(dim=(-2, -1))
    coherence = (orientation * weight).sum(dim=(-2, -1)) / denominator.clamp_min(1e-8)
    active = (denominator > 1e-8).to(coherence.dtype)
    values = (1.0 - coherence.clamp(0.0, 1.0)) * active
    return values.sum() / active.sum().clamp_min(1.0)


def phase_transport_gain_target(
    physical_bands: Tensor,
    residual_after_anchor: Tensor,
    valid: Tensor,
    gain_caps: Sequence[float],
    block_size: int = 4,
) -> Tensor:
    """Return strict-valid blockwise sigmoid-gate targets for phase transport."""

    if not isinstance(physical_bands, Tensor) or physical_bands.ndim != 5:
        raise ValueError("phase transport bands must be B3CHW")
    if physical_bands.shape[1] != 3 or physical_bands.shape[2] < 1:
        raise ValueError("phase transport bands must contain three nonempty frequency bands")
    if not isinstance(residual_after_anchor, Tensor) or residual_after_anchor.ndim != 4:
        raise ValueError("phase transport residual must be BCHW")
    if not isinstance(valid, Tensor) or valid.ndim != 4 or valid.shape[1] != 1:
        raise ValueError("phase transport valid mask must be B1HW")
    batch, _, channels, height, width = physical_bands.shape
    if (
        residual_after_anchor.shape != (batch, channels, height, width)
        or valid.shape != (batch, 1, height, width)
    ):
        raise ValueError("phase transport inputs must share B, C, H, and W")
    if (
        physical_bands.device != residual_after_anchor.device
        or physical_bands.device != valid.device
    ):
        raise ValueError("phase transport inputs must share a device")
    if not physical_bands.is_floating_point() or not residual_after_anchor.is_floating_point():
        raise TypeError("phase transport bands and residual must be floating point")
    if not (valid.is_floating_point() or valid.dtype == torch.bool):
        raise TypeError("phase transport valid mask must be floating point or bool")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("phase transport block_size must be a positive integer")
    if height % block_size or width % block_size:
        raise ValueError("phase transport dimensions must be divisible by block_size")
    if not isinstance(gain_caps, (tuple, list)) or len(gain_caps) != 3:
        raise ValueError("phase transport gain_caps must contain three values")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in gain_caps):
        raise TypeError("phase transport gain_caps must contain numeric values")
    caps = tuple(float(value) for value in gain_caps)
    if any(not math.isfinite(value) or value < 0.0 for value in caps):
        raise ValueError("phase transport gain_caps must be finite and non-negative")
    # Keep this training oracle free of host synchronizations on the GPU hot path.
    mask = torch.nan_to_num(valid.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    components = torch.nan_to_num(
        physical_bands.float(), nan=0.0, posinf=0.0, neginf=0.0
    ) * mask.unsqueeze(1)
    residual = torch.nan_to_num(
        residual_after_anchor.float(), nan=0.0, posinf=0.0, neginf=0.0
    ) * mask
    numerator = F.avg_pool2d(
        (components * residual.unsqueeze(1)).sum(dim=2), block_size, stride=block_size
    )
    energy = F.avg_pool2d(components.square().sum(dim=2), block_size, stride=block_size)
    coverage = F.avg_pool2d(mask, block_size, stride=block_size)
    cap_values = numerator.new_tensor(caps).view(1, 3, 1, 1)
    beta = torch.minimum((numerator / energy.clamp_min(1e-8)).clamp_min(0.0), cap_values)
    gate = beta / cap_values.clamp_min(1e-8)
    supported = (energy > 1e-7) & (coverage >= 0.999) & (cap_values > 0.0)
    return torch.where(supported, gate, torch.zeros_like(gate)).clamp(0.0, 1.0)


def anchor_gain_target(
    raw_components: tuple[Tensor, Tensor, Tensor],
    full_residual: Tensor,
    valid: Tensor,
    maximum_gain: float = 3.0,
) -> Tensor:
    """Return soft local utility targets for the three Optical anchor components."""

    if isinstance(maximum_gain, bool) or not isinstance(maximum_gain, (int, float)):
        raise TypeError("maximum_gain must be a finite positive scalar")
    if not math.isfinite(maximum_gain) or maximum_gain <= 0.0:
        raise ValueError("maximum_gain must be finite and positive")
    if not isinstance(raw_components, tuple) or len(raw_components) != 3:
        raise ValueError("anchor gain target requires exactly three components")
    if full_residual.ndim != 4 or valid.ndim != 4 or valid.shape[1] != 1:
        raise ValueError("anchor gain inputs require residual BCHW and valid B1HW tensors")
    if full_residual.shape[0] != valid.shape[0] or full_residual.shape[-2:] != valid.shape[-2:]:
        raise ValueError("anchor gain residual and valid shapes must match")
    height, width = full_residual.shape[-2:]
    if height % 8 or width % 8:
        raise ValueError("anchor gain inputs require dimensions divisible by eight")
    for component in raw_components:
        if not isinstance(component, Tensor) or component.shape != full_residual.shape:
            raise ValueError("anchor gain components must match the full residual shape")
        if component.device != full_residual.device:
            raise ValueError("anchor gain components must share the residual device")

    mask = valid.to(full_residual)
    components = torch.stack(
        tuple(highpass(component) * mask for component in raw_components), dim=1
    )
    residual = full_residual * mask
    numerator = F.avg_pool2d(
        (components * residual.unsqueeze(1)).sum(dim=2), 4, stride=4
    )
    energy = F.avg_pool2d(components.square().sum(dim=2), 4, stride=4)
    beta = (numerator / (energy + 1e-8)).clamp(0.0, maximum_gain)
    coverage = F.avg_pool2d(mask, 4, stride=4)
    supported = (energy > 1e-7) & (coverage >= 0.999)
    default = torch.zeros_like(beta)
    default[:, :1] = 1.0
    beta = torch.where(supported, beta, default)
    beta = F.avg_pool2d(beta, 3, stride=1, padding=1)
    beta = torch.where(supported, beta, default)
    return beta / (1.0 + beta)


def texture_reliability_gate(
    source: Tensor,
    texture: Tensor,
    mask: Tensor,
    *,
    threshold: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """Return source-supported texture reliability and a binary 4x4 release gate."""
    reliability = detail_reliability_target(
        source, frequency_bands(texture, levels=3), mask
    ).mean(dim=1, keepdim=True)
    cutoff = torch.as_tensor(threshold, device=reliability.device, dtype=reliability.dtype)
    return reliability, (reliability >= cutoff).to(reliability.dtype)


def deterministic_detail_target(
    target: Tensor,
    base: Tensor,
    mask: Tensor,
    modality: str,
) -> Tensor:
    residual = (target - base.detach()) * mask
    if modality == "sar":
        padded = F.pad(residual, (1, 1, 1, 1), mode="reflect")
        neighborhoods = padded.unfold(2, 3, 1).unfold(3, 3, 1)
        residual = neighborhoods.flatten(-2).median(dim=-1).values
    return highpass(residual) * mask


def log_spectral_distance(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    sample_weight: Tensor | None = None,
) -> Tensor:
    prediction_spectrum = torch.fft.rfft2((prediction * mask).float(), norm="ortho")
    target_spectrum = torch.fft.rfft2((target * mask).float(), norm="ortho")
    distance = (
        (
            torch.log1p(complex_magnitude(prediction_spectrum))
            - torch.log1p(complex_magnitude(target_spectrum))
        )
        .abs()
        .mean(dim=(1, 2, 3))
    )
    if sample_weight is None:
        return distance.mean()
    weights = sample_weight.to(distance.dtype) * (mask.flatten(1).sum(1) > 0).to(distance.dtype)
    return (distance * weights).sum() / weights.sum().clamp_min(1e-8)


def high_frequency_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    modality: str,
    sample_weight: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    scale = 0.08 if modality == "optical" else 4.0
    prediction = prediction / scale
    target = target / scale
    weight = mask
    if sample_weight is not None:
        weight = weight * sample_weight.view(-1, 1, 1, 1)
    reconstruction = masked_mean((prediction - target).abs(), weight)
    pred_dy, pred_dx = gradients(prediction)
    target_dy, target_dx = gradients(target)
    gradient = masked_mean((pred_dy - target_dy).abs(), weight[..., 1:, :])
    gradient += masked_mean((pred_dx - target_dx).abs(), weight[..., :, 1:])
    spectrum = log_spectral_distance(prediction, target, mask, sample_weight)
    low = low_frequency_loss(prediction, mask, sample_weight)
    total = reconstruction + 0.2 * gradient + 0.1 * spectrum + 0.2 * low
    metrics = {
        "hf_reconstruction": reconstruction.detach(),
        "hf_gradient": gradient.detach(),
        "hf_spectrum": spectrum.detach(),
        "low_frequency": low.detach(),
    }
    if modality == "sar":
        local_prediction_variance = F.avg_pool2d(prediction.square(), 9, 1, 4)
        local_target_variance = F.avg_pool2d(target.square(), 9, 1, 4)
        speckle = masked_mean(
            (
                torch.sqrt(local_prediction_variance + 1e-6)
                - torch.sqrt(local_target_variance + 1e-6)
            ).abs(),
            weight,
        )
        total = total + 0.2 * speckle
        metrics["speckle_scale"] = speckle.detach()
    return total, metrics
