from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .evaluation import (
    ManifestCropDataset,
    apply_manifest_temporal_prior,
    edge_f1,
    load_checkpoint,
    manifest_metadata,
    perceptual_metrics,
    radial_psd_distance,
)
from .losses import deterministic_detail_target, frequency_bands, highpass, masked_mean
from .model import SentinelV3
from .sensors import SENTINEL1, SENTINEL2


def select_texture_release_candidate(
    candidates: list[dict[str, float | bool]],
) -> dict[str, float | bool]:
    """Select texture only when it improves the same-detail zero-alpha baseline."""

    release = [candidate for candidate in candidates if candidate.get("release_beneficial")]
    if release:
        return max(
            release,
            key=lambda candidate: (
                float(candidate["lpips_improvement"])
                + float(candidate["dists_improvement"])
            ),
        )
    anchor_fallback = [
        candidate
        for candidate in candidates
        if bool(candidate.get("detail_enabled"))
        and float(candidate.get("alpha", -1.0)) == 0.0
        and candidate.get("visual_beneficial")
    ]
    if anchor_fallback:
        return max(
            anchor_fallback,
            key=lambda candidate: (
                float(candidate["lpips_improvement"])
                + float(candidate["dists_improvement"])
            ),
        )
    return next(
        candidate
        for candidate in candidates
        if float(candidate.get("alpha", -1.0)) == 0.0
        and not bool(candidate.get("detail_enabled"))
    )


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
                    sar_pyramid,
                    SENTINEL1,
                    SENTINEL2,
                    tuple(s2.shape[-2:]),
                    base=optical_base,
                )
                * valid
            )
            radar_detail = (
                model.deterministic_detail(
                    optical_pyramid,
                    SENTINEL2,
                    SENTINEL1,
                    tuple(sar.shape[-2:]),
                    base=radar,
                )
                * valid
            )
            optical_texture = (
                model.sample_residual(
                    sar_pyramid,
                    SENTINEL2,
                    tuple(optical_base.shape),
                    seed=seed + index,
                    bridge_anchor=optical_detail,
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
    optical_scale_name = model.amplitude_scale_name("optical")
    payload["model"][optical_scale_name] = torch.tensor(optical_alpha)
    payload["model"]["sar_alpha_scale"] = torch.tensor(sar_alpha)
    if "ema" in payload:
        payload["ema"]["state"][optical_scale_name] = torch.tensor(optical_alpha)
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


def calibrate_detail_confidence_thresholds(
    checkpoint: str,
    manifest: str,
    output: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Maximize released detail coverage subject to no validation MAE degradation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint, device)
    dataset = ManifestCropDataset(manifest, "validation_temporal", limit=limit)
    candidates = torch.cat(
        (torch.linspace(0.50, 0.95, 46, device=device), torch.tensor([1.01], device=device))
    )
    errors = {
        "optical": torch.zeros_like(candidates),
        "sar": torch.zeros_like(candidates),
    }
    coverages = {
        "optical": torch.zeros_like(candidates),
        "sar": torch.zeros_like(candidates),
    }
    zero_errors = {"optical": 0.0, "sar": 0.0}
    counts = {"optical": 0, "sar": 0}
    for item in dataset:
        if int(item["delta_days"]) > 1:
            continue
        s2 = item["s2"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        sar = item["sar"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        valid = item["valid"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        metadata = manifest_metadata(item, device)
        gsd = float(item["gsd"])
        directions = (
            (sar, s2[:, [2, 1, 0]], SENTINEL1, SENTINEL2, "optical"),
            (s2, sar, SENTINEL2, SENTINEL1, "sar"),
        )
        with torch.inference_mode():
            for source, target, source_spec, target_spec, modality in directions:
                physical, _, pyramid = model.physical(
                    source,
                    source_spec,
                    target_spec,
                    valid,
                    input_gsd=gsd,
                    target_gsd=gsd,
                    metadata=metadata,
                )
                physical = apply_manifest_temporal_prior(model, physical, item, target_spec)[0]
                base = physical[:, [2, 1, 0]] if modality == "optical" else physical
                target_detail = deterministic_detail_target(target, base, valid, modality)
                _, bands, confidence = model.deterministic_detail_with_confidence(
                    pyramid, source_spec, target_spec, tuple(base.shape[-2:]), base
                )
                zero_errors[modality] += float(masked_mean(target_detail.abs(), valid))
                counts[modality] += 1
                gate = (confidence[None] >= candidates[:, None, None, None, None]).to(
                    confidence.dtype
                )[:, 0]
                gate = F.interpolate(gate, size=base.shape[-2:], mode="nearest")
                detail = highpass(
                    sum(band * gate[:, level : level + 1] for level, band in enumerate(bands))
                )
                expanded_valid = valid.expand(detail.shape[0], -1, -1, -1)
                denominator = expanded_valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
                denominator = denominator * detail.shape[1]
                errors[modality] += ((detail - target_detail).abs() * expanded_valid).sum(
                    dim=(1, 2, 3)
                ) / denominator
                coverages[modality] += gate.mean(dim=(1, 2, 3))
    selected: dict[str, float] = {}
    result: dict[str, Any] = {
        "split": "validation_temporal",
        "samples": counts["optical"],
    }
    for modality in ("optical", "sar"):
        count = max(counts[modality], 1)
        mean_error = errors[modality] / count
        zero_error = zero_errors[modality] / count
        valid = mean_error <= zero_error * (1.0 + 1e-5)
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        index = int(valid_indices[0]) if valid_indices.numel() else len(candidates) - 1
        threshold = float(candidates[index])
        selected[modality] = threshold
        result[f"{modality}_threshold"] = threshold
        result[f"{modality}_coverage"] = float(coverages[modality][index] / count)
        result[f"{modality}_detail_mae_improvement"] = float(
            (zero_error - mean_error[index]) / max(zero_error, 1e-8)
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for modality, threshold in selected.items():
        name = f"{modality}_detail_confidence_threshold"
        payload["model"][name] = torch.tensor(threshold)
        if "ema" in payload:
            payload["ema"]["state"][name] = torch.tensor(threshold)
    detail_gate = all(
        float(result[f"{modality}_detail_mae_improvement"]) >= -1e-5
        for modality in ("optical", "sar")
    )
    payload.setdefault("quality_gates", {})["detail"] = detail_gate
    result["quality_gate"] = detail_gate
    payload["detail_confidence_calibration"] = result
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return result


def calibrate_texture_release(
    checkpoint: str,
    manifest: str,
    output: str,
    *,
    seed: int = 42,
    limit: int | None = 32,
    amplitude_floors: tuple[float, ...] = (
        0.0,
        0.002,
        0.004,
        0.006,
        0.008,
        0.010,
        0.012,
        0.015,
        0.020,
        0.030,
        0.050,
        0.160,
    ),
    alpha_values: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 1.0),
    detail_values: tuple[bool, ...] = (False, True),
) -> dict[str, Any]:
    """Calibrate sparse Optical texture release against paired validation risk.

    Flow inference is cached once per scene. Cheap distortion constraints prune the
    search before mandatory LPIPS/DISTS scoring. When no non-zero release improves
    both perceptual metrics and structure, the checkpoint safely publishes physical.
    """
    if not amplitude_floors or any(value < 0.0 for value in amplitude_floors):
        raise ValueError("amplitude floors must be non-negative")
    if not alpha_values or any(not 0.0 <= value <= 1.0 for value in alpha_values):
        raise ValueError("alpha values must be in [0, 1]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint, device)
    model.set_amplitude_scale("optical", 1.0)
    model.set_optical_texture_amplitude_floor(0.0)
    model.set_optical_texture_risk_threshold(0.0)
    dataset = ManifestCropDataset(manifest, "validation_temporal", limit=limit)
    cached: list[dict[str, Tensor]] = []
    with torch.inference_mode():
        for index, item in enumerate(dataset):
            s2 = item["s2"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            sar = item["sar"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            valid = item["valid"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            metadata = manifest_metadata(item, device)
            gsd = float(item["gsd"])
            physical, _, pyramid = model.physical(
                sar,
                SENTINEL1,
                SENTINEL2,
                valid,
                input_gsd=gsd,
                target_gsd=gsd,
                metadata=metadata,
            )
            physical = apply_manifest_temporal_prior(model, physical, item, SENTINEL2)[0]
            base = physical[:, [2, 1, 0]]
            target = s2[:, [2, 1, 0]]
            detail = (
                model.deterministic_detail(
                    pyramid, SENTINEL1, SENTINEL2, tuple(base.shape[-2:]), base=base
                )
                * valid
            )
            texture = (
                model.sample_residual(
                    pyramid,
                    SENTINEL2,
                    tuple(base.shape),
                    seed=seed + index,
                    bridge_anchor=detail,
                )
                * valid
            )
            amplitude = model.residual_amplitude(
                pyramid, SENTINEL2, base.shape[1], tuple(base.shape[-2:])
            )
            cached.append(
                {
                    "physical": base.cpu(),
                    "target": target.cpu(),
                    "detail": detail.cpu(),
                    "texture": texture.cpu(),
                    "amplitude": amplitude.cpu(),
                    "valid": valid.cpu(),
                }
            )
    if not cached:
        raise RuntimeError("texture release calibration requires validation samples")

    physical_rmse = sum(
        float(
            torch.sqrt(masked_mean((item["physical"] - item["target"]).square(), item["valid"]))
        )
        for item in cached
    ) / len(cached)
    candidates: list[dict[str, float | bool]] = []
    seen: set[tuple[bool, float, float]] = set()
    for detail_enabled in detail_values:
        for floor in amplitude_floors:
            for alpha in alpha_values:
                key = (
                    detail_enabled,
                    0.0 if alpha == 0.0 else float(floor),
                    0.0 if alpha == 0.0 else float(alpha),
                )
                if key in seen:
                    continue
                seen.add(key)
                rmse_sum = 0.0
                violation_sum = 0.0
                bad_scenes = 0
                for item in cached:
                    amplitude = item["amplitude"]
                    gate = (amplitude.mean(dim=1, keepdim=True) >= floor).to(amplitude.dtype)
                    texture = item["texture"] * F.interpolate(
                        gate, size=item["texture"].shape[-2:], mode="nearest"
                    )
                    visual, violation = SentinelV3.compose_visual(
                        item["physical"],
                        item["detail"] if detail_enabled else torch.zeros_like(item["detail"]),
                        texture * alpha,
                        "optical",
                        return_violation=True,
                    )
                    assert isinstance(visual, Tensor)
                    physical_scene_rmse = torch.sqrt(
                        masked_mean((item["physical"] - item["target"]).square(), item["valid"])
                    )
                    visual_scene_rmse = torch.sqrt(
                        masked_mean((visual - item["target"]).square(), item["valid"])
                    )
                    rmse_sum += float(visual_scene_rmse)
                    violation_sum += float(violation)
                    bad_scenes += int(visual_scene_rmse > 1.05 * physical_scene_rmse)
                visual_rmse = rmse_sum / len(cached)
                violation = violation_sum / len(cached)
                bad_fraction = bad_scenes / len(cached)
                candidates.append(
                    {
                        "detail_enabled": detail_enabled,
                        "amplitude_floor": key[1],
                        "alpha": key[2],
                        "visual_rgb_rmse": visual_rmse,
                        "rmse_ratio": visual_rmse / max(physical_rmse, 1e-8),
                        "pre_projection_violation": violation,
                        "bad_scene_fraction": bad_fraction,
                        "distortion_safe": (
                            visual_rmse <= 1.05 * physical_rmse
                            and violation <= 0.001
                            and bad_fraction <= 0.10
                        ),
                    }
                )

    physical_lpips = 0.0
    physical_dists = 0.0
    for item in cached:
        with redirect_stdout(StringIO()):
            lpips, dists = perceptual_metrics(
                item["physical"].to(device), item["target"].to(device)
            )
        physical_lpips += lpips
        physical_dists += dists
    physical_lpips /= len(cached)
    physical_dists /= len(cached)
    safe_candidates = [candidate for candidate in candidates if candidate["distortion_safe"]]
    for candidate in safe_candidates:
        lpips_sum = 0.0
        dists_sum = 0.0
        edge_visual = 0.0
        edge_physical = 0.0
        psd_visual = 0.0
        psd_physical = 0.0
        floor = float(candidate["amplitude_floor"])
        alpha = float(candidate["alpha"])
        detail_enabled = bool(candidate["detail_enabled"])
        for item in cached:
            amplitude = item["amplitude"].to(device)
            gate = (amplitude.mean(dim=1, keepdim=True) >= floor).to(amplitude.dtype)
            texture = item["texture"].to(device) * F.interpolate(
                gate, size=item["texture"].shape[-2:], mode="nearest"
            )
            physical = item["physical"].to(device)
            target = item["target"].to(device)
            detail = (
                item["detail"].to(device)
                if detail_enabled
                else torch.zeros_like(item["detail"], device=device)
            )
            valid = item["valid"].to(device)
            visual = SentinelV3.compose_visual(physical, detail, texture * alpha, "optical")
            assert isinstance(visual, Tensor)
            with redirect_stdout(StringIO()):
                lpips, dists = perceptual_metrics(visual, target)
            lpips_sum += lpips
            dists_sum += dists
            edge_visual += float(edge_f1(visual, target, valid))
            edge_physical += float(edge_f1(physical, target, valid))
            psd_visual += float(radial_psd_distance(visual, target, valid))
            psd_physical += float(radial_psd_distance(physical, target, valid))
        count = len(cached)
        candidate.update(
            {
                "lpips_improvement": (physical_lpips - lpips_sum / count)
                / max(physical_lpips, 1e-8),
                "dists_improvement": (physical_dists - dists_sum / count)
                / max(physical_dists, 1e-8),
                "physical_edge_f1": edge_physical / count,
                "visual_edge_f1": edge_visual / count,
                "physical_psd_distance": psd_physical / count,
                "visual_psd_distance": psd_visual / count,
            }
        )
        candidate["visual_beneficial"] = bool(
            candidate["lpips_improvement"] > 0.0
            and candidate["dists_improvement"] > 0.0
            and candidate["visual_edge_f1"] > candidate["physical_edge_f1"]
            and candidate["visual_psd_distance"] < candidate["physical_psd_distance"]
        )

    detail_baselines = {
        enabled: next(
            candidate
            for candidate in safe_candidates
            if bool(candidate["detail_enabled"]) == enabled
            and float(candidate["alpha"]) == 0.0
        )
        for enabled in (False, True)
        if any(
            bool(candidate["detail_enabled"]) == enabled
            and float(candidate["alpha"]) == 0.0
            for candidate in safe_candidates
        )
    }
    for candidate in safe_candidates:
        baseline = detail_baselines[bool(candidate["detail_enabled"])]
        candidate["release_beneficial"] = bool(
            float(candidate["alpha"]) > 0.0
            and candidate["lpips_improvement"] > baseline["lpips_improvement"]
            and candidate["dists_improvement"] > baseline["dists_improvement"]
            and candidate["visual_edge_f1"] > baseline["visual_edge_f1"]
            and candidate["visual_psd_distance"] < baseline["visual_psd_distance"]
        )

    release_beneficial = [
        candidate for candidate in safe_candidates if candidate.get("release_beneficial")
    ]
    selected = select_texture_release_candidate(safe_candidates)
    selected_floor = float(selected["amplitude_floor"])
    selected_alpha = float(selected["alpha"])
    selected_detail_enabled = bool(selected["detail_enabled"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    optical_scale_name = model.amplitude_scale_name("optical")
    amplitude_floor_name = model.texture_amplitude_floor_name()
    payload["model"][amplitude_floor_name] = torch.tensor(selected_floor)
    payload["model"][optical_scale_name] = torch.tensor(selected_alpha)
    if not selected_detail_enabled:
        payload["model"]["optical_detail_confidence_threshold"] = torch.tensor(1.01)
        payload["model"]["optical_anchor_band_scales"] = torch.zeros(3)
        payload["model"]["optical_anchor_density_gain"] = torch.tensor(0.0)
    if selected_alpha > 0.0:
        payload["model"]["optical_texture_risk_threshold"] = torch.tensor(0.0)
    if "ema" in payload:
        payload["ema"]["state"][amplitude_floor_name] = torch.tensor(selected_floor)
        payload["ema"]["state"][optical_scale_name] = torch.tensor(selected_alpha)
        if not selected_detail_enabled:
            payload["ema"]["state"]["optical_detail_confidence_threshold"] = torch.tensor(1.01)
            payload["ema"]["state"]["optical_anchor_band_scales"] = torch.zeros(3)
            payload["ema"]["state"]["optical_anchor_density_gain"] = torch.tensor(0.0)
        if selected_alpha > 0.0:
            payload["ema"]["state"]["optical_texture_risk_threshold"] = torch.tensor(0.0)
    result: dict[str, Any] = {
        "split": "validation_temporal",
        "samples": len(cached),
        "seed": seed,
        "physical_rgb_rmse": physical_rmse,
        "physical_lpips": physical_lpips,
        "physical_dists": physical_dists,
        "selected_amplitude_floor": selected_floor,
        "selected_alpha": selected_alpha,
        "selected_detail_enabled": selected_detail_enabled,
        "amplitude_is_useful_risk_proxy": bool(release_beneficial),
        "selected": selected,
        "candidates": candidates,
    }
    payload["texture_release_calibration"] = result
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return result


def _calibration_anchor_detail(
    stacked_bands: Tensor,
    source_density: Tensor,
    valid: Tensor,
    candidate: dict[str, Any],
) -> Tensor:
    """Match deployed anchor projection and valid-pixel masking during calibration."""

    raw_anchor = SentinelV3.source_aware_optical_anchor(
        (stacked_bands[:, 0], stacked_bands[:, 1], stacked_bands[:, 2]),
        source_density,
        stacked_bands.new_tensor(candidate["band_scales"]),
        float(candidate["density_gain"]),
        float(candidate["density_threshold"]),
        float(candidate["source_gain"]),
        float(candidate["source_threshold"]),
    )
    return highpass(raw_anchor) * valid


def calibrate_anchor_detail(
    checkpoint: str,
    manifest: str,
    output: str,
    *,
    limit: int | None = 32,
) -> dict[str, Any]:
    """Calibrate physical-anchored Laplacian detail under all visual guardrails."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint, device)
    model.set_detail_confidence_threshold("optical", 1.01)
    model.set_optical_anchor_band_scales((0.0, 0.0, 0.0))
    model.set_optical_anchor_density(0.0, 1.0)
    model.set_optical_anchor_source_density(0.0, 1.0)
    dataset = ManifestCropDataset(manifest, "validation_temporal", limit=limit)
    cached: list[dict[str, Tensor]] = []
    with torch.inference_mode():
        for item in dataset:
            s2 = item["s2"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            sar = item["sar"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            valid = item["valid"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            metadata = manifest_metadata(item, device)
            gsd = float(item["gsd"])
            physical, _, pyramid = model.physical(
                sar,
                SENTINEL1,
                SENTINEL2,
                valid,
                input_gsd=gsd,
                target_gsd=gsd,
                metadata=metadata,
            )
            physical = apply_manifest_temporal_prior(model, physical, item, SENTINEL2)[0]
            base = physical[:, [2, 1, 0]]
            cached.append(
                {
                    "physical": base.cpu(),
                    "target": s2[:, [2, 1, 0]].cpu(),
                    "valid": valid.cpu(),
                    "bands": torch.stack(frequency_bands(base, levels=3), dim=1).cpu(),
                    "source_density": F.avg_pool2d(
                        highpass(pyramid[0]).abs().mean(dim=1, keepdim=True), 4, stride=4
                    ).cpu(),
                }
            )
    if not cached:
        raise RuntimeError("anchor detail calibration requires validation samples")
    scalar_values = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
    scale_candidates: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    scale_candidates.extend((value, value, value) for value in scalar_values)
    for level in range(3):
        level_values = (
            (0.05, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)
            if level == 0
            else (0.05, 0.10, 0.20, 0.30)
        )
        for value in level_values:
            scales = [0.0, 0.0, 0.0]
            scales[level] = value
            scale_candidates.append(tuple(scales))
    scale_candidates.extend(
        (
            (0.05, 0.10, 0.0),
            (0.05, 0.15, 0.0),
            (0.10, 0.20, 0.0),
            (0.0, 0.10, 0.20),
            (0.0, 0.15, 0.30),
            (0.05, 0.10, 0.20),
            (0.10, 0.15, 0.30),
        )
    )
    parameter_candidates = [
        {
            "band_scales": scales,
            "density_gain": 0.0,
            "density_threshold": 1.0,
            "source_gain": 0.0,
            "source_threshold": 1.0,
        }
        for scales in scale_candidates
    ]
    for base_scale in (0.20, 0.25, 0.30):
        for peak_scale in (0.50, 0.60, 0.80):
            for threshold in (1.0, 1.5, 2.0):
                parameter_candidates.append(
                    {
                        "band_scales": (base_scale, 0.0, 0.0),
                        "density_gain": peak_scale - base_scale,
                        "density_threshold": threshold,
                        "source_gain": 0.0,
                        "source_threshold": 1.0,
                    }
                )
    for base_scale in (0.10, 0.20, 0.30):
        for density_gain in (0.0, 0.20, 0.30):
            for source_gain in (0.10, 0.20, 0.30, 0.40):
                for source_threshold in (0.75, 1.0, 1.5, 2.0):
                    parameter_candidates.append(
                        {
                            "band_scales": (base_scale, 0.0, 0.0),
                            "density_gain": density_gain,
                            "density_threshold": 1.0,
                            "source_gain": source_gain,
                            "source_threshold": source_threshold,
                        }
                    )
    for base_scale in (0.10, 0.15, 0.20):
        for density_gain in (0.0, 0.10, 0.20):
            for source_gain in (0.40, 0.45, 0.50, 0.55):
                for source_threshold in (1.0, 1.15, 1.30, 1.50):
                    parameter_candidates.append(
                        {
                            "band_scales": (base_scale, 0.0, 0.0),
                            "density_gain": density_gain,
                            "density_threshold": 1.0,
                            "source_gain": source_gain,
                            "source_threshold": source_threshold,
                        }
                    )
    for base_scale in (0.10, 0.15, 0.20):
        for density_gain in (0.0, 0.10):
            for source_gain in (0.60, 0.65, 0.70):
                for source_threshold in (1.0, 1.15, 1.30, 1.50):
                    parameter_candidates.append(
                        {
                            "band_scales": (base_scale, 0.0, 0.0),
                            "density_gain": density_gain,
                            "density_threshold": 1.0,
                            "source_gain": source_gain,
                            "source_threshold": source_threshold,
                        }
                    )

    def anchor_detail(item: dict[str, Tensor], candidate: dict[str, Any]) -> Tensor:
        return _calibration_anchor_detail(
            item["bands"],
            item["source_density"],
            item["valid"],
            candidate,
        )

    physical_rmse = sum(
        float(
            torch.sqrt(masked_mean((item["physical"] - item["target"]).square(), item["valid"]))
        )
        for item in cached
    ) / len(cached)
    candidates: list[dict[str, Any]] = []
    for parameters in parameter_candidates:
        rmse_sum = 0.0
        violation_sum = 0.0
        bad_scenes = 0
        for item in cached:
            detail = anchor_detail(item, parameters)
            visual, violation = SentinelV3.compose_visual(
                item["physical"],
                detail,
                torch.zeros_like(detail),
                "optical",
                return_violation=True,
            )
            assert isinstance(visual, Tensor)
            physical_scene_rmse = torch.sqrt(
                masked_mean((item["physical"] - item["target"]).square(), item["valid"])
            )
            visual_scene_rmse = torch.sqrt(
                masked_mean((visual - item["target"]).square(), item["valid"])
            )
            rmse_sum += float(visual_scene_rmse)
            violation_sum += float(violation)
            bad_scenes += int(visual_scene_rmse > 1.05 * physical_scene_rmse)
        count = len(cached)
        visual_rmse = rmse_sum / count
        violation = violation_sum / count
        bad_fraction = bad_scenes / count
        candidates.append(
            {
                "band_scales": list(parameters["band_scales"]),
                "density_gain": float(parameters["density_gain"]),
                "density_threshold": float(parameters["density_threshold"]),
                "source_gain": float(parameters["source_gain"]),
                "source_threshold": float(parameters["source_threshold"]),
                "visual_rgb_rmse": visual_rmse,
                "rmse_ratio": visual_rmse / max(physical_rmse, 1e-8),
                "pre_projection_violation": violation,
                "bad_scene_fraction": bad_fraction,
                "distortion_safe": (
                    visual_rmse <= 1.05 * physical_rmse
                    and violation <= 0.001
                    and bad_fraction <= 0.10
                ),
            }
        )
    physical_lpips = 0.0
    physical_dists = 0.0
    for item in cached:
        with redirect_stdout(StringIO()):
            lpips, dists = perceptual_metrics(
                item["physical"].to(device), item["target"].to(device)
            )
        physical_lpips += lpips
        physical_dists += dists
    physical_lpips /= len(cached)
    physical_dists /= len(cached)
    for candidate in (entry for entry in candidates if entry["distortion_safe"]):
        scales = candidate["band_scales"]
        lpips_sum = dists_sum = edge_visual = edge_physical = 0.0
        psd_visual = psd_physical = 0.0
        for item in cached:
            physical = item["physical"].to(device)
            target = item["target"].to(device)
            valid = item["valid"].to(device)
            device_item = {name: value.to(device) for name, value in item.items()}
            detail = anchor_detail(device_item, candidate)
            visual = SentinelV3.compose_visual(
                physical, detail, torch.zeros_like(detail), "optical"
            )
            assert isinstance(visual, Tensor)
            with redirect_stdout(StringIO()):
                lpips, dists = perceptual_metrics(visual, target)
            lpips_sum += lpips
            dists_sum += dists
            edge_visual += float(edge_f1(visual, target, valid))
            edge_physical += float(edge_f1(physical, target, valid))
            psd_visual += float(radial_psd_distance(visual, target, valid))
            psd_physical += float(radial_psd_distance(physical, target, valid))
        count = len(cached)
        candidate.update(
            {
                "lpips_improvement": (physical_lpips - lpips_sum / count)
                / max(physical_lpips, 1e-8),
                "dists_improvement": (physical_dists - dists_sum / count)
                / max(physical_dists, 1e-8),
                "physical_edge_f1": edge_physical / count,
                "visual_edge_f1": edge_visual / count,
                "physical_psd_distance": psd_physical / count,
                "visual_psd_distance": psd_visual / count,
            }
        )
        candidate["beneficial"] = bool(
            (
                any(float(value) > 0.0 for value in scales)
                or float(candidate["density_gain"]) > 0.0
                or float(candidate["source_gain"]) > 0.0
            )
            and candidate["lpips_improvement"] > 0.0
            and candidate["dists_improvement"] > 0.0
            and candidate["visual_edge_f1"] > candidate["physical_edge_f1"]
            and candidate["visual_psd_distance"] < candidate["physical_psd_distance"]
        )
    beneficial = [entry for entry in candidates if entry.get("beneficial")]
    selected = (
        max(
            beneficial,
            key=lambda entry: (
                float(entry["lpips_improvement"]) + float(entry["dists_improvement"])
            ),
        )
        if beneficial
        else candidates[0]
    )
    selected_scales = tuple(float(value) for value in selected["band_scales"])
    selected_density_gain = float(selected["density_gain"])
    selected_density_threshold = float(selected["density_threshold"])
    selected_source_gain = float(selected["source_gain"])
    selected_source_threshold = float(selected["source_threshold"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    scales_tensor = torch.tensor(selected_scales)
    payload["model"]["optical_anchor_band_scales"] = scales_tensor
    payload["model"]["optical_anchor_density_gain"] = torch.tensor(selected_density_gain)
    payload["model"]["optical_anchor_density_threshold"] = torch.tensor(
        selected_density_threshold
    )
    payload["model"]["optical_anchor_source_gain"] = torch.tensor(selected_source_gain)
    payload["model"]["optical_anchor_source_threshold"] = torch.tensor(
        selected_source_threshold
    )
    payload["model"]["optical_detail_confidence_threshold"] = torch.tensor(1.01)
    if "ema" in payload:
        payload["ema"]["state"]["optical_anchor_band_scales"] = scales_tensor
        payload["ema"]["state"]["optical_anchor_density_gain"] = torch.tensor(
            selected_density_gain
        )
        payload["ema"]["state"]["optical_anchor_density_threshold"] = torch.tensor(
            selected_density_threshold
        )
        payload["ema"]["state"]["optical_anchor_source_gain"] = torch.tensor(
            selected_source_gain
        )
        payload["ema"]["state"]["optical_anchor_source_threshold"] = torch.tensor(
            selected_source_threshold
        )
        payload["ema"]["state"]["optical_detail_confidence_threshold"] = torch.tensor(1.01)
    result: dict[str, Any] = {
        "split": "validation_temporal",
        "samples": len(cached),
        "physical_rgb_rmse": physical_rmse,
        "physical_lpips": physical_lpips,
        "physical_dists": physical_dists,
        "selected_band_scales": list(selected_scales),
        "selected_density_gain": selected_density_gain,
        "selected_density_threshold": selected_density_threshold,
        "selected_source_gain": selected_source_gain,
        "selected_source_threshold": selected_source_threshold,
        "anchor_detail_is_beneficial": bool(beneficial),
        "selected": selected,
        "candidates": candidates,
    }
    payload["anchor_detail_calibration"] = result
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return result
