from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp_min(1.0)


def gradients(values: Tensor) -> tuple[Tensor, Tensor]:
    return values[..., 1:, :] - values[..., :-1, :], values[..., :, 1:] - values[..., :, :-1]


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
    huber = masked_mean(F.huber_loss(prediction, target, reduction="none", delta=0.05 if modality == "optical" else 1.0), weight)
    mse = masked_mean(error.square(), weight)
    nll = masked_mean(0.5 * (error.square() * torch.exp(-log_variance) + log_variance), weight)
    pred_dy, pred_dx = gradients(prediction)
    target_dy, target_dx = gradients(target)
    gradient = masked_mean((pred_dy - target_dy).abs(), weight[..., 1:, :])
    gradient += masked_mean((pred_dx - target_dx).abs(), weight[..., :, 1:])
    structure = structural_loss(prediction, target, weight)
    mse_weight = 0.5 if modality == "optical" else 0.05
    total = huber + mse_weight * mse + 0.05 * nll + 0.1 * structure + 0.05 * gradient
    metrics = {
        "huber": huber.detach(),
        "mse": mse.detach(),
        "rmse": torch.sqrt(mse.detach().clamp_min(0.0)),
        "nll": nll.detach(),
        "ssim_loss": structure.detach(),
        "gradient": gradient.detach(),
    }
    if modality == "optical":
        sam = spectral_angle(prediction, target, weight)
        total = total + 0.1 * sam
        metrics["sam"] = sam.detach()
    else:
        relation = masked_mean(((prediction[:, :1] - prediction[:, 1:2]) - (target[:, :1] - target[:, 1:2])).abs(), weight)
        total = total + 0.05 * relation
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
    return info_nce + 0.25 * dense_loss, {"info_nce": info_nce.detach(), "dense_cosine": dense_loss.detach()}


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


def log_spectral_distance(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    sample_weight: Tensor | None = None,
) -> Tensor:
    prediction_spectrum = torch.fft.rfft2((prediction * mask).float(), norm="ortho")
    target_spectrum = torch.fft.rfft2((target * mask).float(), norm="ortho")
    distance = (
        torch.log1p(prediction_spectrum.abs())
        - torch.log1p(target_spectrum.abs())
    ).abs().mean(dim=(1, 2, 3))
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
            (torch.sqrt(local_prediction_variance + 1e-6) - torch.sqrt(local_target_variance + 1e-6)).abs(),
            weight,
        )
        total = total + 0.2 * speckle
        metrics["speckle_scale"] = speckle.detach()
    return total, metrics
