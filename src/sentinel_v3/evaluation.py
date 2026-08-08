from __future__ import annotations

import json
import math
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset

from .losses import (
    detail_reliability_target,
    deterministic_detail_target,
    frequency_bands,
    highpass,
    masked_mean,
    spectral_angle,
)
from .model import ModelConfig, SentinelV3
from .sensors import SENTINEL1, SENTINEL2
from .validation import ValidationProtocol, protocol_records, validation_protocol_hash

S2_KEYS = (
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "swir16",
    "swir22",
)


def aggregate_scene_bias(signed_biases: list[float]) -> tuple[float, float]:
    if not signed_biases:
        raise ValueError("at least one scene bias is required")
    return (
        abs(sum(signed_biases) / len(signed_biases)),
        sum(abs(value) for value in signed_biases) / len(signed_biases),
    )


class ManifestCropDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        crop_size: int = 256,
        limit: int | None = None,
    ) -> None:
        if split == "validation_temporal":
            protocol = ValidationProtocol(crop_size=crop_size)
            self.records = protocol_records(manifest, protocol)
        else:
            self.records = []
            with Path(manifest).open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record["split"] == split or record["refit_split"] == split:
                        self.records.append(record)
            self.records.sort(key=lambda record: record["pair_id"])
        if limit is not None and limit < len(self.records):
            if split == "validation_temporal":
                indices = np.linspace(0, len(self.records) - 1, limit, dtype=np.int64)
                self.records = [self.records[int(index)] for index in indices]
            else:
                self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"no records for split {split}")
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        import rasterio
        from rasterio.windows import Window

        record = self.records[index]
        size = self.crop_size
        row = max(0, (int(record["height"]) - size) // 2)
        col = max(0, (int(record["width"]) - size) // 2)
        window = Window(col, row, size, size)
        s2_raw = []
        for key in S2_KEYS:
            with rasterio.open(record["s2"][key]) as source:
                s2_raw.append(source.read(1, window=window))
        sar_raw = []
        for key in ("vv", "vh"):
            with rasterio.open(record["sar"][key]) as source:
                sar_raw.append(source.read(1, window=window))
        with rasterio.open(record["scl"]) as source:
            scl = source.read(1, window=window)
        s2_values = np.stack(s2_raw).astype(np.float32) / 10000.0
        sar_encoded = np.stack(sar_raw).astype(np.float32)
        sar_db = sar_encoded / 200.0 - 50.0
        valid = (
            np.isin(scl, ValidationProtocol().mask_scl_codes)
            & np.all(np.stack(s2_raw) > 0, axis=0)
            & np.all(sar_encoded > 0, axis=0)
        )
        s2_values[:, ~valid] = 0
        sar_db[:, ~valid] = 0
        return {
            "s2": torch.from_numpy(s2_values.clip(0, 1)),
            "sar": torch.from_numpy(sar_db),
            "valid": torch.from_numpy(valid[None].astype(np.float32)),
            "pair_id": record["pair_id"],
            "delta_days": record["delta_days"],
            "orbit": record["orbit"],
            "s1_date": record["s1_date"],
            "s2_date": record["s2_date"],
            "gsd": record["gsd"],
            "tile": record["tile"],
            "pixel_window": (col, row, size, size),
        }


def manifest_metadata(item: dict[str, object], device: torch.device) -> Tensor:
    """Build the same eight physical conditions used by the training shards."""
    s1_day = date.fromisoformat(str(item["s1_date"]))
    s2_day = date.fromisoformat(str(item["s2_date"]))
    phase_s1 = 2.0 * math.pi * s1_day.timetuple().tm_yday / 366.0
    phase_s2 = 2.0 * math.pi * s2_day.timetuple().tm_yday / 366.0
    orbit = {"ascending": -1.0, "descending": 1.0, "unknown": 0.0}.get(str(item["orbit"]), 0.0)
    return torch.tensor(
        [
            [
                float(item["delta_days"]) / 3.0,
                orbit,
                math.sin(phase_s1),
                math.cos(phase_s1),
                math.sin(phase_s2),
                math.cos(phase_s2),
                math.log(max(float(item["gsd"]), 0.1)) / 4.0,
                1.0,
            ]
        ],
        device=device,
        dtype=torch.float32,
    )


def load_checkpoint(
    path: str | Path, device: torch.device, *, use_ema: bool = True
) -> SentinelV3:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("format_version", 0)) != 4:
        raise RuntimeError("evaluation requires a V3.2 format-v4 checkpoint")
    model = SentinelV3(ModelConfig(**payload["config"]["model"]))
    state = dict(payload["model"])
    if use_ema and "ema" in payload:
        state.update(payload["ema"]["state"])
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = (
        "decoder.radiometric_kernel.",
        "decoder.radiometric_condition.",
        "decoder.radiometric_descriptor.",
        "decoder.radiometric_bias.",
        "decoder.full_resolution_fusion.",
        "decoder.optical_direction_kernel.",
        "decoder.optical_amplitude_head.",
        "decoder.sar_spatial_kernel.",
        "decoder.sar_mean_condition.",
        "decoder.sar_mean_descriptor.",
        "decoder.sar_mean_head.",
        "detail_head.base_heads.",
        "detail_head.confidence_heads.",
        "residual_dit.frequency_adapter.",
        "residual_dit.texture_risk_candidate.",
        "residual_dit.texture_risk_head.",
        "optical_texture_amplitude_floor",
        "optical_anchor_band_scales",
        "optical_texture_risk_threshold",
        "optical_detail_confidence_threshold",
        "sar_detail_confidence_threshold",
    )
    missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(allowed_missing_prefixes)
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"incompatible checkpoint: missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    temporal_config = payload.get("temporal_prior") or payload["config"].get("temporal_prior")
    model.configure_temporal_prior(temporal_config)
    return model.to(device).eval()


def apply_manifest_temporal_prior(
    model: SentinelV3,
    physical: Tensor,
    item: dict[str, object],
    target_sensor: object,
) -> tuple[Tensor, Tensor, Tensor]:
    target = SENTINEL2 if target_sensor == SENTINEL2 else SENTINEL1
    acquired = item["s2_date"] if target.modality == "optical" else item["s1_date"]
    return model.apply_temporal_prior(
        physical,
        target,
        acquired=str(acquired),
        location_id=str(item["tile"]),
        pixel_window=item["pixel_window"],  # type: ignore[arg-type]
        orbit=str(item["orbit"]),
    )


def _edge(values: Tensor) -> Tensor:
    gray = values.mean(dim=1, keepdim=True)
    dx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1))
    dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-8)


def edge_f1(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    predicted = _edge(prediction)
    reference = _edge(target)
    threshold = (
        torch.quantile(reference[mask.bool().expand_as(reference)], 0.8)
        if mask.any()
        else reference.new_tensor(0.1)
    )
    predicted_binary = predicted >= threshold
    target_binary = reference >= threshold
    valid = mask.bool()
    true_positive = (predicted_binary & target_binary & valid).sum()
    precision = true_positive / (predicted_binary & valid).sum().clamp_min(1)
    recall = true_positive / (target_binary & valid).sum().clamp_min(1)
    return 2 * precision * recall / (precision + recall).clamp_min(1e-8)


def radial_psd_distance(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    pred_spectrum = (
        torch.fft.rfft2((prediction * mask).float(), norm="ortho").abs().square().mean((0, 1))
    )
    target_spectrum = (
        torch.fft.rfft2((target * mask).float(), norm="ortho").abs().square().mean((0, 1))
    )
    height, half_width = pred_spectrum.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=prediction.device),
        torch.arange(half_width, device=prediction.device),
        indexing="ij",
    )
    y = torch.minimum(y, height - y)
    radius = torch.sqrt(y.float().square() + x.float().square()).long()
    bins = int(radius.max()) + 1
    counts = torch.zeros(bins, device=prediction.device).scatter_add_(
        0, radius.flatten(), torch.ones_like(radius, dtype=torch.float32).flatten()
    )
    pred_radial = torch.zeros(bins, device=prediction.device).scatter_add_(
        0, radius.flatten(), pred_spectrum.flatten()
    ) / counts.clamp_min(1)
    target_radial = torch.zeros(bins, device=prediction.device).scatter_add_(
        0, radius.flatten(), target_spectrum.flatten()
    ) / counts.clamp_min(1)
    return (torch.log1p(pred_radial) - torch.log1p(target_radial)).abs().mean()


def equivalent_number_of_looks(values_db: Tensor, mask: Tensor) -> Tensor:
    intensity = torch.pow(10.0, values_db / 10.0)
    mean = masked_mean(intensity, mask)
    variance = masked_mean((intensity - mean).square(), mask)
    return mean.square() / variance.clamp_min(1e-8)


def histogram_distance(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    selected_prediction = prediction[mask.bool().expand_as(prediction)]
    selected_target = target[mask.bool().expand_as(target)]
    pred_hist = torch.histc(selected_prediction.float(), bins=80, min=-45, max=5)
    target_hist = torch.histc(selected_target.float(), bins=80, min=-45, max=5)
    pred_hist /= pred_hist.sum().clamp_min(1)
    target_hist /= target_hist.sum().clamp_min(1)
    return (pred_hist - target_hist).abs().mean()


def tail_quantile_error(
    prediction: Tensor, target: Tensor, mask: Tensor, quantile: float
) -> Tensor:
    """Absolute dB error at a SAR distribution tail quantile."""
    selected_prediction = prediction[mask.bool().expand_as(prediction)].float()
    selected_target = target[mask.bool().expand_as(target)].float()
    if selected_prediction.numel() == 0:
        return prediction.new_tensor(0.0)
    return (
        torch.quantile(selected_prediction, quantile)
        - torch.quantile(selected_target, quantile)
    ).abs()


_PERCEPTUAL_CACHE: dict[str, tuple[torch.nn.Module, torch.nn.Module]] = {}


def perceptual_evaluators(device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    key = str(device)
    if key in _PERCEPTUAL_CACHE:
        return _PERCEPTUAL_CACHE[key]
    try:
        checkpoint_root = Path(torch.hub.get_dir()) / "checkpoints"
        required = (
            checkpoint_root / "alexnet-owt-7be5be79.pth",
            checkpoint_root / "vgg16-397923af.pth",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        import lpips
        from DISTS_pytorch import DISTS

        dists_weights_path = files("DISTS_pytorch").joinpath("weights.pt")
        if not dists_weights_path.is_file():
            raise FileNotFoundError(dists_weights_path)
        lpips_evaluator = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
        dists_evaluator = DISTS(load_weights=False)
        weights = torch.load(dists_weights_path, map_location="cpu", weights_only=True)
        dists_evaluator.alpha.data.copy_(weights["alpha"])
        dists_evaluator.beta.data.copy_(weights["beta"])
        dists_evaluator = dists_evaluator.to(device).eval().requires_grad_(False)
    except (ImportError, RuntimeError, OSError, FileNotFoundError) as error:
        raise RuntimeError(
            "LPIPS and DISTS weights are mandatory for V3.2 validation; "
            "run scripts/download_eval_weights.sh and install the eval extra"
        ) from error
    _PERCEPTUAL_CACHE[key] = (lpips_evaluator, dists_evaluator)
    return _PERCEPTUAL_CACHE[key]


def perceptual_metrics(prediction: Tensor, target: Tensor) -> tuple[float, float]:
    lpips_evaluator, dists_evaluator = perceptual_evaluators(prediction.device)
    with torch.inference_mode():
        lpips_value = float(
            lpips_evaluator(prediction.clamp(0, 1) * 2 - 1, target.clamp(0, 1) * 2 - 1).mean()
        )
        dists_value = float(dists_evaluator(prediction.clamp(0, 1), target.clamp(0, 1)).mean())
    return lpips_value, dists_value


def _image(values: Tensor, kind: str, size: int = 192) -> Image.Image:
    array = values.detach().float().cpu()
    if kind == "rgb":
        array = array[:3]
        array = (array.clamp(0, 0.3) / 0.3).pow(0.7)
    elif kind == "map":
        array = array.abs().mean(dim=0, keepdim=True)
        array = (array / torch.quantile(array.flatten(), 0.98).clamp_min(1e-6)).clamp(0, 1)
        array = array.repeat(3, 1, 1)
    else:
        array = ((array[:1] + 35.0) / 40.0).clamp(0, 1).repeat(3, 1, 1)
    return Image.fromarray((array * 255).byte().permute(1, 2, 0).numpy()).resize(
        (size, size), Image.Resampling.BILINEAR
    )


def save_panel(path: Path, titles: list[str], images: list[Image.Image]) -> None:
    width = images[0].width
    canvas = Image.new("RGB", (width * len(images), images[0].height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(zip(titles, images, strict=True)):
        canvas.paste(image, (index * width, 28))
        draw.text((index * width + 4, 7), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def evaluate_model(
    model: SentinelV3,
    manifest: str,
    split: str,
    *,
    seed: int = 42,
    limit: int | None = None,
    panels: int = 32,
    panel_root: Path | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    dataset = ManifestCropDataset(manifest, split, limit=limit)
    # Resolve evaluators before the expensive raster loop; missing weights must fail fast.
    perceptual_evaluators(device)
    sums: dict[str, float] = {}
    scene_counts = {
        "edge_improved": 0,
        "dists_improved": 0,
        "edge_or_dists_improved": 0,
        "rgb_rmse_degraded_over_5pct": 0,
    }
    for index, item in enumerate(dataset):
        s2 = item["s2"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        sar = item["sar"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        valid = item["valid"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        metadata = manifest_metadata(item, device)
        gsd = float(item["gsd"])
        with torch.inference_mode():
            s2_mean, _, sar_pyramid = model.physical(
                sar,
                SENTINEL1,
                SENTINEL2,
                valid,
                input_gsd=gsd,
                target_gsd=gsd,
                metadata=metadata,
            )
            sar_mean, _, optical_pyramid = model.physical(
                s2,
                SENTINEL2,
                SENTINEL1,
                valid,
                input_gsd=gsd,
                target_gsd=gsd,
                metadata=metadata,
            )
            s2_mean, optical_prior_coverage, optical_prior_violation = (
                apply_manifest_temporal_prior(model, s2_mean, item, SENTINEL2)
            )
            sar_mean, sar_prior_coverage, sar_prior_violation = apply_manifest_temporal_prior(
                model, sar_mean, item, SENTINEL1
            )
            optical_detail = (
                model.deterministic_detail(
                    sar_pyramid,
                    SENTINEL1,
                    SENTINEL2,
                    tuple(s2.shape[-2:]),
                    base=s2_mean[:, [2, 1, 0]],
                )
                * valid
            )
            sar_detail = (
                model.deterministic_detail(
                    optical_pyramid,
                    SENTINEL2,
                    SENTINEL1,
                    tuple(sar.shape[-2:]),
                    base=sar_mean,
                )
                * valid
            )
            optical_texture = (
                model.sample_residual(
                    sar_pyramid, SENTINEL2, (1, 3, *s2.shape[-2:]), seed=seed + index
                )
                * valid
            )
            sar_texture = (
                model.sample_residual(
                    optical_pyramid, SENTINEL1, tuple(sar.shape), seed=seed + index
                )
                * valid
            )
            physical_rgb = s2_mean[:, [2, 1, 0]]
            target_rgb = s2[:, [2, 1, 0]]
            optical_visual, violation = model.compose_visual(
                physical_rgb,
                optical_detail,
                optical_texture,
                "optical",
                return_violation=True,
            )
            sar_visual, sar_violation = model.compose_visual(
                sar_mean, sar_detail, sar_texture, "sar", return_violation=True
            )
        physical_lpips, physical_dists = perceptual_metrics(physical_rgb, target_rgb)
        visual_lpips, visual_dists = perceptual_metrics(optical_visual, target_rgb)
        metrics = {
            "sar2opt_rmse": torch.sqrt(masked_mean((s2_mean - s2).square(), valid)),
            "sar2opt_sam_deg": spectral_angle(s2_mean, s2, valid) * (180.0 / math.pi),
            "opt2sar_rmse_db": torch.sqrt(masked_mean((sar_mean - sar).square(), valid)),
            "opt2sar_physical_signed_bias_db": masked_mean(sar_mean - sar, valid),
            "opt2sar_physical_scene_abs_bias_db": masked_mean(sar_mean - sar, valid).abs(),
            "opt2sar_visual_signed_bias_db": masked_mean(sar_visual - sar, valid),
            "opt2sar_visual_scene_abs_bias_db": masked_mean(sar_visual - sar, valid).abs(),
            "physical_rgb_rmse": torch.sqrt(
                masked_mean((physical_rgb - target_rgb).square(), valid)
            ),
            "visual_rgb_rmse": torch.sqrt(
                masked_mean((optical_visual - target_rgb).square(), valid)
            ),
            "physical_edge_f1": edge_f1(physical_rgb, target_rgb, valid),
            "visual_edge_f1": edge_f1(optical_visual, target_rgb, valid),
            "physical_optical_psd_distance": radial_psd_distance(
                physical_rgb, target_rgb, valid
            ),
            "visual_optical_psd_distance": radial_psd_distance(
                optical_visual, target_rgb, valid
            ),
            "pre_projection_violation": violation,
            "sar_pre_projection_violation": sar_violation,
            "sar_mean_psd_distance": radial_psd_distance(sar_mean, sar, valid),
            "sar_visual_psd_distance": radial_psd_distance(sar_visual, sar, valid),
            "sar_mean_enl_error": (
                equivalent_number_of_looks(sar_mean, valid)
                - equivalent_number_of_looks(sar, valid)
            ).abs(),
            "sar_visual_enl_error": (
                equivalent_number_of_looks(sar_visual, valid)
                - equivalent_number_of_looks(sar, valid)
            ).abs(),
            "sar_mean_histogram_distance": histogram_distance(sar_mean, sar, valid),
            "sar_visual_histogram_distance": histogram_distance(sar_visual, sar, valid),
            "sar_mean_p01_error_db": tail_quantile_error(sar_mean, sar, valid, 0.01),
            "sar_visual_p01_error_db": tail_quantile_error(sar_visual, sar, valid, 0.01),
            "sar_mean_p99_error_db": tail_quantile_error(sar_mean, sar, valid, 0.99),
            "sar_visual_p99_error_db": tail_quantile_error(sar_visual, sar, valid, 0.99),
            "physical_lpips": physical_lpips,
            "visual_lpips": visual_lpips,
            "physical_dists": physical_dists,
            "visual_dists": visual_dists,
            "optical_temporal_prior_coverage": optical_prior_coverage,
            "sar_temporal_prior_coverage": sar_prior_coverage,
            "optical_temporal_prior_pre_projection_violation": optical_prior_violation,
            "sar_temporal_prior_pre_projection_violation": sar_prior_violation,
        }
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + float(value)
        edge_improved = bool(metrics["visual_edge_f1"] > metrics["physical_edge_f1"])
        dists_improved = bool(visual_dists < physical_dists)
        scene_counts["edge_improved"] += int(edge_improved)
        scene_counts["dists_improved"] += int(dists_improved)
        scene_counts["edge_or_dists_improved"] += int(edge_improved or dists_improved)
        scene_counts["rgb_rmse_degraded_over_5pct"] += int(
            metrics["visual_rgb_rmse"] > 1.05 * metrics["physical_rgb_rmse"]
        )
        if panel_root is not None and index < panels:
            save_panel(
                panel_root / f"{index:03d}_sar2opt.png",
                ["Input SAR", "Physical", "Detail", "Texture", "Visual", "Reference"],
                [
                    _image(sar[0], "sar"),
                    _image(physical_rgb[0], "rgb"),
                    _image(optical_detail[0], "map"),
                    _image(optical_texture[0], "map"),
                    _image(optical_visual[0], "rgb"),
                    _image(target_rgb[0], "rgb"),
                ],
            )
    report: dict[str, Any] = {
        "split": split,
        "samples": len(dataset),
        "seed": seed,
        "protocol_hash": validation_protocol_hash(manifest)
        if split == "validation_temporal"
        else None,
    }
    report.update({name: value / len(dataset) for name, value in sums.items()})
    report.update(
        {f"scene_{name}_fraction": value / len(dataset) for name, value in scene_counts.items()}
    )
    report["opt2sar_physical_bias_db"] = abs(report.pop("opt2sar_physical_signed_bias_db"))
    report["opt2sar_visual_bias_db"] = abs(report.pop("opt2sar_visual_signed_bias_db"))
    report["lpips_improvement"] = (report["physical_lpips"] - report["visual_lpips"]) / max(
        report["physical_lpips"], 1e-8
    )
    report["dists_improvement"] = (report["physical_dists"] - report["visual_dists"]) / max(
        report["physical_dists"], 1e-8
    )
    physical_gate = (
        report["sar2opt_rmse"] <= 0.03909
        and report["sar2opt_sam_deg"] <= 5.716
        and report["opt2sar_rmse_db"] <= 5.0
        and report["opt2sar_physical_bias_db"] <= 0.5
    )
    optical_visual_gate = (
        report["visual_rgb_rmse"] <= 1.05 * report["physical_rgb_rmse"]
        and report["lpips_improvement"] >= 0.05
        and report["dists_improvement"] >= 0.05
        and report["visual_edge_f1"] > report["physical_edge_f1"]
        and report["visual_optical_psd_distance"] < report["physical_optical_psd_distance"]
        and report["pre_projection_violation"] <= 0.001
        and report["scene_edge_or_dists_improved_fraction"] >= 0.70
        and report["scene_rgb_rmse_degraded_over_5pct_fraction"] <= 0.10
    )
    sar_visual_gate = (
        report["opt2sar_visual_bias_db"] <= 0.5
        and report["sar_visual_psd_distance"] < report["sar_mean_psd_distance"]
        and report["sar_visual_enl_error"] < report["sar_mean_enl_error"]
        and report["sar_visual_histogram_distance"] < report["sar_mean_histogram_distance"]
        and report["sar_visual_p01_error_db"] < report["sar_mean_p01_error_db"]
        and report["sar_visual_p99_error_db"] < report["sar_mean_p99_error_db"]
    )
    report["quality_gates"] = {
        "physical": physical_gate,
        "optical_visual": optical_visual_gate,
        "sar_visual": sar_visual_gate,
        "visual": optical_visual_gate and sar_visual_gate,
        "joint": physical_gate and optical_visual_gate and sar_visual_gate,
    }
    return report


def evaluate_physical_model(
    model: SentinelV3,
    manifest: str,
    split: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    dataset = ManifestCropDataset(manifest, split, limit=limit)
    sums = {
        "sar2opt_squared_error": 0.0,
        "sar2opt_sam_deg": 0.0,
        "opt2sar_squared_error": 0.0,
        "opt2sar_signed_bias_db": 0.0,
        "opt2sar_scene_abs_bias_db": 0.0,
    }
    slices: dict[str, dict[str, dict[str, float]]] = {
        "delta_days": {},
        "orbit": {},
    }

    def update_slice(group: str, name: str, values: dict[str, float]) -> None:
        bucket = slices[group].setdefault(
            name,
            {
                "samples": 0.0,
                "sar2opt_squared_error": 0.0,
                "sar2opt_sam_deg": 0.0,
                "opt2sar_squared_error": 0.0,
                "opt2sar_signed_bias_db": 0.0,
                "opt2sar_scene_abs_bias_db": 0.0,
            },
        )
        bucket["samples"] += 1.0
        for key, value in values.items():
            bucket[key] += value

    for item in dataset:
        s2 = item["s2"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        sar = item["sar"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        valid = item["valid"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        metadata = manifest_metadata(item, device)
        gsd = float(item["gsd"])
        with torch.inference_mode():
            s2_mean = model.physical(
                sar,
                SENTINEL1,
                SENTINEL2,
                valid,
                input_gsd=gsd,
                target_gsd=gsd,
                metadata=metadata,
            )[0]
            sar_mean = model.physical(
                s2,
                SENTINEL2,
                SENTINEL1,
                valid,
                input_gsd=gsd,
                target_gsd=gsd,
                metadata=metadata,
            )[0]
            s2_mean = apply_manifest_temporal_prior(model, s2_mean, item, SENTINEL2)[0]
            sar_mean = apply_manifest_temporal_prior(model, sar_mean, item, SENTINEL1)[0]
        values = {
            "sar2opt_squared_error": float(masked_mean((s2_mean - s2).square(), valid)),
            "sar2opt_sam_deg": float(spectral_angle(s2_mean, s2, valid) * (180.0 / math.pi)),
            "opt2sar_squared_error": float(masked_mean((sar_mean - sar).square(), valid)),
            "opt2sar_signed_bias_db": float(masked_mean(sar_mean - sar, valid)),
            "opt2sar_scene_abs_bias_db": float(masked_mean(sar_mean - sar, valid).abs()),
        }
        for key, value in values.items():
            sums[key] += value
        update_slice("delta_days", str(item["delta_days"]), values)
        update_slice("orbit", str(item["orbit"]), values)

    def finalize(values: dict[str, float]) -> dict[str, float | int]:
        count = int(values["samples"])
        return {
            "samples": count,
            "sar2opt_rmse": math.sqrt(values["sar2opt_squared_error"] / count),
            "sar2opt_sam_deg": values["sar2opt_sam_deg"] / count,
            "opt2sar_rmse_db": math.sqrt(values["opt2sar_squared_error"] / count),
            "opt2sar_physical_bias_db": abs(values["opt2sar_signed_bias_db"] / count),
            "opt2sar_scene_abs_bias_db": values["opt2sar_scene_abs_bias_db"] / count,
        }

    report: dict[str, Any] = {
        "split": split,
        "samples": len(dataset),
        "protocol_hash": validation_protocol_hash(manifest)
        if split == "validation_temporal"
        else None,
        "sar2opt_rmse": math.sqrt(sums["sar2opt_squared_error"] / len(dataset)),
        "sar2opt_sam_deg": sums["sar2opt_sam_deg"] / len(dataset),
        "opt2sar_rmse_db": math.sqrt(sums["opt2sar_squared_error"] / len(dataset)),
        "opt2sar_physical_bias_db": abs(sums["opt2sar_signed_bias_db"] / len(dataset)),
        "opt2sar_scene_abs_bias_db": sums["opt2sar_scene_abs_bias_db"] / len(dataset),
        "slices": {
            group: {name: finalize(values) for name, values in buckets.items()}
            for group, buckets in slices.items()
        },
    }
    physical_gate = (
        report["sar2opt_rmse"] <= 0.03909
        and report["sar2opt_sam_deg"] <= 5.716
        and report["opt2sar_rmse_db"] <= 5.0
        and report["opt2sar_physical_bias_db"] <= 0.5
    )
    report["quality_gates"] = {"physical": physical_gate}
    return report


def evaluate_high_frequency_components(
    model: SentinelV3,
    manifest: str,
    split: str,
    stage: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    if stage not in {"detail", "codec"}:
        raise ValueError("component validation supports detail or codec")
    device = next(model.parameters()).device
    dataset = ManifestCropDataset(manifest, split, limit=limit)
    sums: dict[str, float] = {}
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
            (sar, s2, SENTINEL1, SENTINEL2, "optical"),
            (s2, sar, SENTINEL2, SENTINEL1, "sar"),
        )
        with torch.inference_mode():
            for source, target, source_spec, target_spec, name in directions:
                target_visual = target[:, [2, 1, 0]] if name == "optical" else target
                if stage == "codec":
                    texture = highpass(target_visual * valid) * valid
                    latent = model.codec.encode(texture, name)
                    reconstruction = model.codec.decode(latent, name)
                    codec_mae = masked_mean((reconstruction - texture).abs(), valid)
                    sums[f"{name}_codec_mae"] = sums.get(f"{name}_codec_mae", 0.0) + float(
                        codec_mae
                    )
                    counts[name] += 1
                    continue
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
                base = physical[:, [2, 1, 0]] if name == "optical" else physical
                target_detail = deterministic_detail_target(target_visual, base, valid, name)
                prediction, _, confidence = model.deterministic_detail_with_confidence(
                    pyramid,
                    source_spec,
                    target_spec,
                    tuple(base.shape[-2:]),
                    base=base,
                )
                target_bands = frequency_bands(target_detail, levels=3)
                reliability = detail_reliability_target(source, target_bands, valid)
                selective_mask = valid * F.interpolate(
                    reliability.mean(dim=1, keepdim=True),
                    size=valid.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                zero_mae = masked_mean(target_detail.abs(), valid)
                detail_mae = masked_mean((prediction - target_detail).abs(), valid)
                selective_zero = masked_mean(target_detail.abs(), selective_mask)
                selective_detail = masked_mean(
                    (prediction - target_detail).abs(), selective_mask
                )
                sums[f"{name}_detail_zero_mae"] = sums.get(
                    f"{name}_detail_zero_mae", 0.0
                ) + float(zero_mae)
                sums[f"{name}_detail_mae"] = sums.get(f"{name}_detail_mae", 0.0) + float(
                    detail_mae
                )
                sums[f"{name}_selective_zero_mae"] = sums.get(
                    f"{name}_selective_zero_mae", 0.0
                ) + float(selective_zero)
                sums[f"{name}_selective_detail_mae"] = sums.get(
                    f"{name}_selective_detail_mae", 0.0
                ) + float(selective_detail)
                sums[f"{name}_detail_coverage"] = sums.get(
                    f"{name}_detail_coverage", 0.0
                ) + float(
                    (confidence >= getattr(model, f"{name}_detail_confidence_threshold"))
                    .float()
                    .mean()
                )
                counts[name] += 1
    report: dict[str, Any] = {
        "split": split,
        "samples": sum(counts.values()) // 2,
        "eligible_samples": counts["optical"],
        "protocol_hash": validation_protocol_hash(manifest)
        if split == "validation_temporal"
        else None,
    }
    for name in ("optical", "sar"):
        count = max(counts[name], 1)
        if stage == "codec":
            report[f"{name}_codec_mae"] = sums[f"{name}_codec_mae"] / count
            continue
        zero = sums[f"{name}_detail_zero_mae"] / count
        detail = sums[f"{name}_detail_mae"] / count
        report[f"{name}_detail_zero_mae"] = zero
        report[f"{name}_detail_mae"] = detail
        report[f"{name}_detail_mae_improvement"] = (zero - detail) / max(zero, 1e-8)
        selective_zero = sums[f"{name}_selective_zero_mae"] / count
        selective_detail = sums[f"{name}_selective_detail_mae"] / count
        report[f"{name}_selective_detail_mae_improvement"] = (
            selective_zero - selective_detail
        ) / max(selective_zero, 1e-8)
        report[f"{name}_detail_coverage"] = sums[f"{name}_detail_coverage"] / count
    if stage == "codec":
        report["quality_gates"] = {
            "codec": (report["optical_codec_mae"] <= 0.02 and report["sar_codec_mae"] <= 1.0)
        }
    else:
        report["quality_gates"] = {
            "detail": all(
                report[f"{name}_detail_mae_improvement"] >= -1e-4
                and report[f"{name}_selective_detail_mae_improvement"] >= -1e-4
                for name in ("optical", "sar")
            )
        }
    return report


def evaluate(
    checkpoint: str,
    manifest: str,
    split: str,
    output: str,
    *,
    limit: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint, device)
    panel_root = Path(output).with_suffix("").with_name(Path(output).stem + "_panels")
    report = evaluate_model(
        model,
        manifest,
        split,
        seed=seed,
        limit=limit,
        panels=32,
        panel_root=panel_root,
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
