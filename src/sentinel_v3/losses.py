from __future__ import annotations

import math

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
    total = reconstruction + 0.25 * gradient + 0.2 * edge + 0.1 * local_structure
    return total, {
        "detail_charbonnier": reconstruction.detach(),
        "detail_gradient": gradient.detach(),
        "detail_edge": edge.detach(),
        "detail_ssim": local_structure.detach(),
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
    ).abs()
    target_spectrum = torch.fft.rfft2(
        (tiles(target) * tiled_mask).float(), dim=(-2, -1), norm="ortho"
    ).abs()
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


def highpass(values: Tensor, block_size: int = 4) -> Tensor:
    if values.shape[-2] % block_size or values.shape[-1] % block_size:
        raise ValueError("highpass dimensions must be divisible by block_size")
    block_mean = F.avg_pool2d(values, block_size, stride=block_size)
    low = block_mean.repeat_interleave(block_size, -2).repeat_interleave(block_size, -1)
    return values - low


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
        (torch.log1p(prediction_spectrum.abs()) - torch.log1p(target_spectrum.abs()))
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
