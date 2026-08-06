from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .evaluation import ManifestCropDataset, load_checkpoint, manifest_metadata
from .losses import masked_mean
from .model import SentinelV3
from .sensors import SENTINEL1, SENTINEL2


def calibrate_amplitude_scale(
    physical: Tensor,
    deterministic_detail: Tensor,
    stochastic_texture: Tensor,
    target: Tensor,
    mask: Tensor,
    modality: str,
    *,
    maximum_rmse_ratio: float = 1.05,
    candidates: int = 101,
) -> tuple[float, dict[str, float]]:
    """Select the largest alpha satisfying the deterministic RMSE guardrail."""
    if candidates < 2 or maximum_rmse_ratio < 1.0:
        raise ValueError("invalid calibration search")
    physical_rmse = torch.sqrt(masked_mean((physical - target).square(), mask))
    selected = 0.0
    selected_rmse = physical_rmse
    for alpha in torch.linspace(0.0, 1.0, candidates, device=physical.device):
        visual = SentinelV3.compose_visual(
            physical,
            deterministic_detail,
            stochastic_texture * alpha,
            modality,
        )
        assert isinstance(visual, Tensor)
        rmse = torch.sqrt(masked_mean((visual - target).square(), mask))
        if rmse <= maximum_rmse_ratio * physical_rmse:
            selected = float(alpha)
            selected_rmse = rmse
    return selected, {
        "physical_rmse": float(physical_rmse),
        "calibrated_visual_rmse": float(selected_rmse),
        "rmse_ratio": float(selected_rmse / physical_rmse.clamp_min(1e-8)),
    }


def calibrate_checkpoint(
    checkpoint: str,
    manifest: str,
    output: str,
    *,
    seed: int = 42,
    limit: int | None = None,
    candidates: int = 101,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint, device)
    model.set_amplitude_scale("optical", 1.0)
    model.set_amplitude_scale("sar", 1.0)
    dataset = ManifestCropDataset(manifest, "validation_temporal", limit=limit)
    alphas = torch.linspace(0.0, 1.0, candidates, device=device)
    optical_mse = torch.zeros(candidates, device=device)
    sar_bias = torch.zeros(candidates, device=device)
    physical_mse = torch.zeros((), device=device)
    for index, item in enumerate(dataset):
        s2 = item["s2"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        sar = item["sar"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        valid = item["valid"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        metadata = manifest_metadata(item, device)
        gsd = float(item["gsd"])
        with torch.inference_mode():
            optical, _, sar_pyramid = model.physical(
                sar,
                SENTINEL1,
                SENTINEL2,
                valid,
                input_gsd=gsd,
                target_gsd=gsd,
                metadata=metadata,
            )
            radar, _, optical_pyramid = model.physical(
                s2,
                SENTINEL2,
                SENTINEL1,
                valid,
                input_gsd=gsd,
                target_gsd=gsd,
                metadata=metadata,
            )
            optical_base = optical[:, [2, 1, 0]]
            target_rgb = s2[:, [2, 1, 0]]
            optical_detail = (
                model.deterministic_detail(
                    sar_pyramid, SENTINEL1, SENTINEL2, tuple(s2.shape[-2:])
                )
                * valid
            )
            radar_detail = (
                model.deterministic_detail(
                    optical_pyramid, SENTINEL2, SENTINEL1, tuple(sar.shape[-2:])
                )
                * valid
            )
            optical_texture = (
                model.sample_residual(
                    sar_pyramid, SENTINEL2, tuple(optical_base.shape), seed=seed + index
                )
                * valid
            )
            radar_texture = (
                model.sample_residual(
                    optical_pyramid, SENTINEL1, tuple(radar.shape), seed=seed + index
                )
                * valid
            )
            physical_mse += masked_mean((optical_base - target_rgb).square(), valid)
            for alpha_index, alpha in enumerate(alphas):
                optical_visual = SentinelV3.compose_visual(
                    optical_base, optical_detail, optical_texture * alpha, "optical"
                )
                radar_visual = SentinelV3.compose_visual(
                    radar, radar_detail, radar_texture * alpha, "sar"
                )
                assert isinstance(optical_visual, Tensor) and isinstance(radar_visual, Tensor)
                optical_mse[alpha_index] += masked_mean(
                    (optical_visual - target_rgb).square(), valid
                )
                sar_bias[alpha_index] += masked_mean(radar_visual - sar, valid).abs()
    physical_rmse = torch.sqrt(physical_mse / len(dataset))
    visual_rmse = torch.sqrt(optical_mse / len(dataset))
    mean_sar_bias = sar_bias / len(dataset)
    optical_valid = visual_rmse <= 1.05 * physical_rmse
    sar_valid = mean_sar_bias <= 0.5
    optical_index = (
        int(torch.nonzero(optical_valid, as_tuple=False)[-1]) if optical_valid.any() else 0
    )
    sar_index = int(torch.nonzero(sar_valid, as_tuple=False)[-1]) if sar_valid.any() else 0
    optical_alpha = float(alphas[optical_index])
    sar_alpha = float(alphas[sar_index])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model"]["optical_alpha_scale"] = torch.tensor(optical_alpha)
    payload["model"]["sar_alpha_scale"] = torch.tensor(sar_alpha)
    if "ema" in payload:
        payload["ema"]["state"]["optical_alpha_scale"] = torch.tensor(optical_alpha)
        payload["ema"]["state"]["sar_alpha_scale"] = torch.tensor(sar_alpha)
    result = {
        "split": "validation_temporal",
        "samples": len(dataset),
        "seed": seed,
        "optical_alpha": optical_alpha,
        "sar_alpha": sar_alpha,
        "physical_rgb_rmse": float(physical_rmse),
        "calibrated_visual_rgb_rmse": float(visual_rmse[optical_index]),
        "calibrated_sar_bias_db": float(mean_sar_bias[sar_index]),
    }
    payload["calibration"] = result
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return result
