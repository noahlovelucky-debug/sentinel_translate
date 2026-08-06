from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset

from .losses import masked_mean, spectral_angle
from .model import ModelConfig, SentinelV3
from .sensors import SENTINEL1, SENTINEL2

S2_KEYS = ("blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22")


class ManifestCropDataset(Dataset[dict[str, object]]):
    def __init__(self, manifest: str | Path, split: str, crop_size: int = 256, limit: int | None = None) -> None:
        self.records = []
        with Path(manifest).open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["split"] == split or record["refit_split"] == split:
                    self.records.append(record)
        if limit is not None:
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
        valid = np.isin(scl, (2, 4, 5, 6, 7)) & np.all(np.stack(s2_raw) > 0, axis=0) & np.all(sar_encoded > 0, axis=0)
        s2_values[:, ~valid] = 0
        sar_db[:, ~valid] = 0
        return {
            "s2": torch.from_numpy(s2_values.clip(0, 1)),
            "sar": torch.from_numpy(sar_db),
            "valid": torch.from_numpy(valid[None].astype(np.float32)),
            "pair_id": record["pair_id"],
            "s1_date": record["s1_date"],
            "s2_date": record["s2_date"],
            "orbit": record["orbit"],
            "delta_days": record["delta_days"],
        }


def load_checkpoint(path: str | Path, device: torch.device, *, use_ema: bool = True) -> SentinelV3:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SentinelV3(ModelConfig(**payload["config"]["model"]))
    state = payload["model"]
    if use_ema and "ema" in payload:
        state = dict(state)
        state.update(payload["ema"]["state"])
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {
        "residual_dit.amplitude_head.weight",
        "residual_dit.amplitude_head.bias",
    }
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected or missing:
        raise RuntimeError(
            f"incompatible checkpoint: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return model.to(device).eval()


def _edge(values: Tensor) -> Tensor:
    gray = values.mean(dim=1, keepdim=True)
    dx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1))
    dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-8)


def edge_f1(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    predicted = _edge(prediction)
    reference = _edge(target)
    threshold = torch.quantile(reference[mask.bool().expand_as(reference)], 0.8) if mask.any() else reference.new_tensor(0.1)
    predicted_binary = predicted >= threshold
    target_binary = reference >= threshold
    valid = mask.bool()
    true_positive = (predicted_binary & target_binary & valid).sum()
    precision = true_positive / (predicted_binary & valid).sum().clamp_min(1)
    recall = true_positive / (target_binary & valid).sum().clamp_min(1)
    return 2 * precision * recall / (precision + recall).clamp_min(1e-8)


def radial_psd_distance(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    prediction = prediction * mask
    target = target * mask
    pred_spectrum = torch.fft.rfft2(prediction.float(), norm="ortho").abs().square().mean((0, 1))
    target_spectrum = torch.fft.rfft2(target.float(), norm="ortho").abs().square().mean((0, 1))
    height, half_width = pred_spectrum.shape
    y, x = torch.meshgrid(torch.arange(height, device=prediction.device), torch.arange(half_width, device=prediction.device), indexing="ij")
    y = torch.minimum(y, height - y)
    radius = torch.sqrt(y.float().square() + x.float().square()).long()
    bins = int(radius.max()) + 1
    pred_radial = torch.zeros(bins, device=prediction.device).scatter_add_(0, radius.flatten(), pred_spectrum.flatten())
    target_radial = torch.zeros(bins, device=prediction.device).scatter_add_(0, radius.flatten(), target_spectrum.flatten())
    counts = torch.zeros(bins, device=prediction.device).scatter_add_(0, radius.flatten(), torch.ones_like(radius, dtype=torch.float32).flatten())
    return (torch.log1p(pred_radial / counts) - torch.log1p(target_radial / counts)).abs().mean()


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


def optional_perceptual(prediction: Tensor, target: Tensor) -> tuple[float | None, float | None]:
    """Return LPIPS/DISTS only when their optional, locally installed evaluators exist."""
    scaled_prediction = prediction.clamp(0, 1) * 2 - 1
    scaled_target = target.clamp(0, 1) * 2 - 1
    lpips_value: float | None = None
    dists_value: float | None = None
    try:
        import lpips

        evaluator = lpips.LPIPS(net="alex").to(prediction.device).eval()
        with torch.inference_mode():
            lpips_value = float(evaluator(scaled_prediction, scaled_target).mean())
    except (ImportError, RuntimeError, OSError):
        pass
    try:
        from importlib.resources import files

        from DISTS_pytorch import DISTS

        evaluator = DISTS(load_weights=False)
        weights = torch.load(files("DISTS_pytorch").joinpath("weights.pt"), map_location="cpu")
        evaluator.alpha.data.copy_(weights["alpha"])
        evaluator.beta.data.copy_(weights["beta"])
        evaluator = evaluator.to(prediction.device).eval()
        with torch.inference_mode():
            dists_value = float(evaluator(prediction.clamp(0, 1), target.clamp(0, 1)).mean())
    except (ImportError, RuntimeError, OSError):
        pass
    return lpips_value, dists_value


def retrieval_metrics(sar_latents: list[Tensor], optical_latents: list[Tensor]) -> dict[str, float]:
    sar = F.normalize(torch.cat(sar_latents), dim=-1)
    optical = F.normalize(torch.cat(optical_latents), dim=-1)
    similarities = sar @ optical.T
    ranks = similarities.argsort(dim=1, descending=True)
    labels = torch.arange(ranks.shape[0], device=ranks.device)[:, None]
    return {
        "recall_at_1": float((ranks[:, :1] == labels).any(dim=1).float().mean()),
        "recall_at_5": float((ranks[:, :5] == labels).any(dim=1).float().mean()),
    }


def _image(values: Tensor, kind: str, size: int = 192) -> Image.Image:
    array = values.detach().float().cpu()
    if kind in {"rgb", "rgb_ready"}:
        array = array[[2, 1, 0]] if array.shape[0] >= 3 else array.repeat(3, 1, 1)
        if kind == "rgb_ready":
            array = array[[2, 1, 0]]
        array = (array.clamp(0, 0.3) / 0.3).pow(0.7)
    elif kind == "map":
        array = array.abs().mean(dim=0, keepdim=True)
        maximum = torch.quantile(array.flatten(), 0.98).clamp_min(1e-6)
        array = (array / maximum).clamp(0, 1).repeat(3, 1, 1)
    else:
        array = array[:1]
        array = ((array + 35.0) / 40.0).clamp(0, 1).repeat(3, 1, 1)
    result = (array * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(result).resize((size, size), Image.Resampling.BILINEAR)


def save_panel(path: Path, titles: list[str], images: list[Image.Image]) -> None:
    width = images[0].width
    canvas = Image.new("RGB", (width * len(images), images[0].height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(zip(titles, images, strict=True)):
        canvas.paste(image, (index * width, 28))
        draw.text((index * width + 4, 7), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


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
    dataset = ManifestCropDataset(manifest, split, limit=limit)
    sums: dict[str, float] = {}
    optional_sums: dict[str, list[float]] = {
        "physical_lpips": [], "visual_lpips": [], "physical_dists": [], "visual_dists": []
    }
    sar_latents: list[Tensor] = []
    optical_latents: list[Tensor] = []
    panel_root = Path(output).with_suffix("").with_name(Path(output).stem + "_panels")
    for index, item in enumerate(dataset):
        s2 = item["s2"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        sar = item["sar"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        valid = item["valid"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        with torch.inference_mode():
            s2_mean, s2_log_variance, sar_scene = model.physical(sar, SENTINEL1, SENTINEL2, valid)
            sar_mean, sar_log_variance, optical_scene = model.physical(s2, SENTINEL2, SENTINEL1, valid)
            optical_residuals = [
                model.sample_residual(
                    sar_scene[-1], SENTINEL2, (1, 3, *s2.shape[-2:]), seed=seed + index * 3 + sample
                )
                for sample in range(3)
            ]
            sar_residual = model.sample_residual(optical_scene[-1], SENTINEL1, tuple(sar.shape), seed=seed + index)
            optical_samples = [
                model.compose_visual(
                    s2_mean[:, [2, 1, 0]], residual, SENTINEL2.modality
                )
                for residual in optical_residuals
            ]
            s2_visual = optical_samples[0]
            sar_visual = model.compose_visual(sar_mean, sar_residual, SENTINEL1.modality)
        s2_target_rgb = s2[:, [2, 1, 0]]
        metrics = {
            "sar2opt_rmse": torch.sqrt(masked_mean((s2_mean - s2).square(), valid)),
            "sar2opt_sam": spectral_angle(s2_mean, s2, valid),
            "opt2sar_rmse_db": torch.sqrt(masked_mean((sar_mean - sar).square(), valid)),
            "opt2sar_bias_db": masked_mean(sar_visual - sar, valid).abs(),
            "physical_rgb_rmse": torch.sqrt(masked_mean((s2_mean[:, [2, 1, 0]] - s2_target_rgb).square(), valid)),
            "visual_rgb_rmse": torch.sqrt(masked_mean((s2_visual - s2_target_rgb).square(), valid)),
            "physical_edge_f1": edge_f1(s2_mean[:, [2, 1, 0]], s2_target_rgb, valid),
            "optical_edge_f1": edge_f1(s2_visual, s2_target_rgb, valid),
            "physical_optical_psd_distance": radial_psd_distance(s2_mean[:, [2, 1, 0]], s2_target_rgb, valid),
            "optical_psd_distance": radial_psd_distance(s2_visual, s2_target_rgb, valid),
            "optical_out_of_bounds_fraction": (((s2_visual < 0) | (s2_visual > 1)) * valid.bool()).sum() / valid.expand_as(s2_visual).sum().clamp_min(1),
            "sar_mean_psd_distance": radial_psd_distance(sar_mean, sar, valid),
            "sar_psd_distance": radial_psd_distance(sar_visual, sar, valid),
            "sar_mean_enl_error": (equivalent_number_of_looks(sar_mean, valid) - equivalent_number_of_looks(sar, valid)).abs(),
            "sar_enl_error": (equivalent_number_of_looks(sar_visual, valid) - equivalent_number_of_looks(sar, valid)).abs(),
            "sar_mean_histogram_distance": histogram_distance(sar_mean, sar, valid),
            "sar_histogram_distance": histogram_distance(sar_visual, sar, valid),
        }
        physical_lpips, physical_dists = optional_perceptual(s2_mean[:, [2, 1, 0]], s2_target_rgb)
        visual_lpips, visual_dists = optional_perceptual(s2_visual, s2_target_rgb)
        for name, value in (
            ("physical_lpips", physical_lpips), ("visual_lpips", visual_lpips),
            ("physical_dists", physical_dists), ("visual_dists", visual_dists),
        ):
            if value is not None:
                optional_sums[name].append(value)
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + float(value)
        sar_latents.append(sar_scene[-1].flatten(2).mean(-1).cpu())
        optical_latents.append(optical_scene[-1].flatten(2).mean(-1).cpu())
        if index < 32:
            save_panel(
                panel_root / f"{index:03d}_sar2opt.png",
                ["Input SAR", "Physical RGB", "Sample 1", "Sample 2", "Sample 3", "Reference RGB", "Abs error", "Uncertainty", "Edges"],
                [
                    _image(sar[0], "sar"),
                    _image(s2_mean[0], "rgb"),
                    *[_image(sample[0], "rgb_ready") for sample in optical_samples],
                    _image(s2[0], "rgb"),
                    _image(s2_visual[0] - s2_target_rgb[0], "map"),
                    _image(torch.exp(0.5 * s2_log_variance[0]), "map"),
                    _image(_edge(s2_visual)[0], "map"),
                ],
            )
            save_panel(
                panel_root / f"{index:03d}_opt2sar.png",
                ["Input RGB", "False color", "Mean VV/VH", "Sample VV/VH", "Reference VV/VH", "Abs error", "Uncertainty"],
                [
                    _image(s2[0], "rgb"),
                    _image(s2[0, [7, 4, 1]], "rgb_ready"),
                    _image(sar_mean[0], "sar"),
                    _image(sar_visual[0], "sar"),
                    _image(sar[0], "sar"),
                    _image(sar_visual[0] - sar[0], "map"),
                    _image(torch.exp(0.5 * sar_log_variance[0]), "map"),
                ],
            )
    report: dict[str, Any] = {"split": split, "samples": len(dataset), "seed": seed}
    report.update({name: value / len(dataset) for name, value in sums.items()})
    for name, values in optional_sums.items():
        report[name] = sum(values) / len(values) if values else None
    report["lpips_improvement"] = (
        None if report["physical_lpips"] is None else
        (report["physical_lpips"] - report["visual_lpips"]) / max(report["physical_lpips"], 1e-8)
    )
    report["dists_improvement"] = (
        None if report["physical_dists"] is None else
        (report["physical_dists"] - report["visual_dists"]) / max(report["physical_dists"], 1e-8)
    )
    report["retrieval"] = retrieval_metrics(sar_latents, optical_latents)
    report["quality_gates"] = {
        "lpips_improves_5_percent": report["lpips_improvement"] is not None and report["lpips_improvement"] >= 0.05,
        "dists_improves_5_percent": report["dists_improvement"] is not None and report["dists_improvement"] >= 0.05,
        "edge_improves": report["optical_edge_f1"] > report["physical_edge_f1"],
        "optical_psd_improves": report["optical_psd_distance"] < report["physical_optical_psd_distance"],
        "rgb_rmse_within_5_percent": report["visual_rgb_rmse"] <= 1.05 * report["physical_rgb_rmse"],
        "optical_bounds_within_0_1_percent": report["optical_out_of_bounds_fraction"] <= 0.001,
        "sar_bias_within_0_5_db": report["opt2sar_bias_db"] <= 0.5,
        "sar_psd_improves": report["sar_psd_distance"] < report["sar_mean_psd_distance"],
        "sar_enl_improves": report["sar_enl_error"] < report["sar_mean_enl_error"],
        "sar_histogram_improves": report["sar_histogram_distance"] < report["sar_mean_histogram_distance"],
        "joint_selection_requires_baseline_report": True,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
