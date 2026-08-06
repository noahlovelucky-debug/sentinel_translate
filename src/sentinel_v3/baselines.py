from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Literal

import torch

from .evaluation import ManifestCropDataset, manifest_metadata
from .losses import masked_mean, spectral_angle
from .physics import db_to_normalized_sar, normalized_s2_to_reflectance
from .validation import validation_protocol_hash


def _legacy_build_system() -> Any:
    legacy_root = "/data/sentinel_translate/code"
    if legacy_root not in sys.path:
        sys.path.insert(0, legacy_root)
    from sentinel_translate.models import build_system

    return build_system


def _load_legacy_states(
    model: torch.nn.Module, paths: list[str | Path], *, direction: str
) -> None:
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_direction = payload.get("direction")
        if checkpoint_direction is not None and checkpoint_direction != direction:
            raise ValueError(f"checkpoint direction mismatch: {path}")
        state = payload.get("ema") or payload["model"]
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"unexpected checkpoint tensors in {path}: {incompatible.unexpected_keys}"
            )


def load_legacy_sar2opt(
    kind: Literal["v1_mean", "v2_refiner"],
    checkpoint: str | Path,
    device: torch.device,
    *,
    mean_checkpoint: str | Path | None = None,
) -> torch.nn.Module:
    build_system = _legacy_build_system()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_system("sar2s2", payload["config"]["model"])
    paths = [checkpoint]
    if kind == "v2_refiner":
        if mean_checkpoint is None:
            raise ValueError("V2 Refiner evaluation requires its V1 Mean checkpoint")
        paths = [mean_checkpoint, checkpoint]
    _load_legacy_states(model, paths, direction="sar2s2")
    return model.to(device).eval()


def evaluate_legacy_sar2opt(
    kind: Literal["v1_mean", "v2_refiner"],
    checkpoint: str,
    manifest: str,
    *,
    mean_checkpoint: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_legacy_sar2opt(kind, checkpoint, device, mean_checkpoint=mean_checkpoint)
    dataset = ManifestCropDataset(manifest, "validation_temporal", limit=limit)
    squared_error = 0.0
    sam_degrees = 0.0
    for item in dataset:
        source = db_to_normalized_sar(item["sar"].unsqueeze(0).to(device))  # type: ignore[union-attr]
        target = item["s2"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        valid = item["valid"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        metadata = manifest_metadata(item, device)
        with torch.inference_mode():
            mean, pyramid = model.mean_prediction(source, valid, None, metadata)
            if kind == "v2_refiner":
                normalized = model.refiner(mean, pyramid, valid)[0]
            else:
                normalized = mean
            prediction = normalized_s2_to_reflectance(normalized) * valid
        squared_error += float(masked_mean((prediction - target).square(), valid))
        sam_degrees += float(spectral_angle(prediction, target, valid) * (180 / math.pi))
    return {
        "model": kind,
        "checkpoint": checkpoint,
        "split": "validation_temporal",
        "samples": len(dataset),
        "protocol_hash": validation_protocol_hash(manifest),
        "sar2opt_rmse": math.sqrt(squared_error / len(dataset)),
        "sar2opt_sam_deg": sam_degrees / len(dataset),
    }
