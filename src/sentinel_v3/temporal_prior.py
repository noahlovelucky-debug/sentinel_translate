from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

S2_ASSET_KEYS = (
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


@dataclass(frozen=True)
class TemporalPriorConfig:
    manifest: str
    manifest_sha256: str
    version: str = "train-seasonal-v2"
    train_years: tuple[int, ...] = (2017, 2018)
    neighbors: int = 6
    time_scale_days: float = 30.0
    sar_neighbors: int = 8
    sar_kernel: Literal["uniform", "exponential"] = "uniform"
    optical_amplitude_weight: float = 0.75
    sar_weight: float = 0.80

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TemporalPriorConfig:
        values = dict(values)
        values["train_years"] = tuple(int(year) for year in values["train_years"])
        return cls(**values)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def temporal_prior_config(
    manifest: str | Path,
    *,
    optical_amplitude_weight: float = 0.75,
    sar_weight: float = 0.80,
) -> TemporalPriorConfig:
    path = Path(manifest).resolve()
    return TemporalPriorConfig(
        manifest=str(path),
        manifest_sha256=file_sha256(path),
        optical_amplitude_weight=optical_amplitude_weight,
        sar_weight=sar_weight,
    )


def _day_distance(left: date, right: date) -> int:
    difference = abs(left.timetuple().tm_yday - right.timetuple().tm_yday)
    return min(difference, 365 - difference)


class TemporalPriorStore:
    """Train-only seasonal memory for locations observed during 2017-2018."""

    def __init__(self, config: TemporalPriorConfig) -> None:
        self.config = config
        manifest = Path(config.manifest)
        if file_sha256(manifest) != config.manifest_sha256:
            raise RuntimeError("temporal-prior manifest hash does not match its checkpoint")
        records: dict[str, list[dict[str, Any]]] = {}
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("split") != "train":
                    continue
                if int(record.get("year", 0)) not in config.train_years:
                    continue
                records.setdefault(str(record["tile"]), []).append(record)
        if not records:
            raise RuntimeError("temporal prior contains no eligible train records")
        self._records = records
        self._records_by_id = {
            str(record["pair_id"]): record
            for location_records in records.values()
            for record in location_records
        }

    @property
    def locations(self) -> frozenset[str]:
        return frozenset(self._records)

    def _nearest(
        self,
        location_id: str,
        acquired: date,
        modality: Literal["optical", "sar"],
        orbit: str,
        exclude_pair_id: str | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        candidates = self._records.get(location_id, ())
        dated: list[tuple[int, str, dict[str, Any]]] = []
        for record in candidates:
            if modality == "sar" and orbit != "unknown" and record["orbit"] != orbit:
                continue
            date_key = "s2_date" if modality == "optical" else "s1_date"
            observed = date.fromisoformat(record[date_key])
            dated.append((_day_distance(acquired, observed), str(record["pair_id"]), record))
        dated.sort(key=lambda item: (item[0], item[1]))
        unique: list[tuple[int, str, dict[str, Any]]] = []
        seen: set[str] = set()
        excluded_record = self._records_by_id.get(exclude_pair_id or "")
        excluded_identity = (
            str(
                excluded_record["s2_date"]
                if modality == "optical"
                else excluded_record["s1_date"]
            )
            if excluded_record is not None
            else None
        )
        for item in dated:
            record = item[2]
            identity = str(record["s2_date"] if modality == "optical" else record["s1_date"])
            if identity == excluded_identity or identity in seen:
                continue
            seen.add(identity)
            unique.append(item)
        limit = self.config.neighbors if modality == "optical" else self.config.sar_neighbors
        selected = unique[:limit]
        return [
            (
                (
                    1.0
                    if modality == "sar" and self.config.sar_kernel == "uniform"
                    else math.exp(-distance / self.config.time_scale_days)
                ),
                record,
            )
            for distance, _, record in selected
        ]

    def query(
        self,
        *,
        location_id: str,
        acquired: date | str,
        modality: Literal["optical", "sar"],
        pixel_window: tuple[int, int, int, int],
        orbit: str = "unknown",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        exclude_pair_id: str | None = None,
    ) -> tuple[Tensor, Tensor] | None:
        target_date = date.fromisoformat(acquired) if isinstance(acquired, str) else acquired
        nearest = self._nearest(
            location_id, target_date, modality, orbit, exclude_pair_id=exclude_pair_id
        )
        if not nearest:
            return None
        _, _, width, height = pixel_window
        channels = len(S2_ASSET_KEYS) if modality == "optical" else 2
        weighted = np.zeros((channels, height, width), dtype=np.float64)
        weights = np.zeros((1, height, width), dtype=np.float64)
        for temporal_weight, record in nearest:
            values, valid = self._read_record(
                str(record["pair_id"]), modality, pixel_window
            )
            sample_weight = temporal_weight * valid[None]
            weighted += values * sample_weight
            weights += sample_weight
        covered = weights > 0
        prior = (weighted / np.maximum(weights, 1e-12)).astype(np.float32)
        return (
            torch.as_tensor(prior, device=device, dtype=dtype).unsqueeze(0),
            torch.as_tensor(covered, device=device, dtype=dtype).unsqueeze(0),
        )

    @lru_cache(maxsize=16)  # noqa: B019 - store lifetime owns this bounded raster cache.
    def _read_record(
        self,
        pair_id: str,
        modality: Literal["optical", "sar"],
        pixel_window: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._read_record_uncached(pair_id, modality, pixel_window)

    def _read_record_uncached(
        self,
        pair_id: str,
        modality: Literal["optical", "sar"],
        pixel_window: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        import rasterio
        from rasterio.windows import Window

        record = self._records_by_id[pair_id]
        window = Window(*pixel_window)
        if modality == "optical":
            raw = []
            for key in S2_ASSET_KEYS:
                with rasterio.open(record["s2"][key]) as source:
                    raw.append(source.read(1, window=window))
            with rasterio.open(record["scl"]) as source:
                scl = source.read(1, window=window)
            encoded = np.stack(raw)
            valid = np.isin(scl, (4, 5, 6, 7)) & np.all(encoded > 0, axis=0)
            values = (encoded.astype(np.float32) / 10000.0).clip(0.0, 1.0)
        else:
            raw = []
            for key in ("vv", "vh"):
                with rasterio.open(record["sar"][key]) as source:
                    raw.append(source.read(1, window=window))
            encoded = np.stack(raw)
            valid = np.all(encoded > 0, axis=0)
            values = encoded.astype(np.float32) / 200.0 - 50.0
        return values, valid

    def full_scene_prior(
        self,
        *,
        location_id: str,
        acquired: date | str,
        modality: Literal["optical", "sar"],
        orbit: str,
        exclude_pair_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_date = date.fromisoformat(acquired) if isinstance(acquired, str) else acquired
        nearest = self._nearest(
            location_id,
            target_date,
            modality,
            orbit,
            exclude_pair_id=exclude_pair_id,
        )
        if not nearest:
            raise RuntimeError("no leave-one-out temporal neighbors are available")
        record = nearest[0][1]
        width, height = int(record["width"]), int(record["height"])
        channels = len(S2_ASSET_KEYS) if modality == "optical" else 2
        weighted = np.zeros((channels, height, width), dtype=np.float32)
        weights = np.zeros((1, height, width), dtype=np.float32)
        for temporal_weight, neighbor in nearest:
            values, valid = self._read_record_uncached(
                str(neighbor["pair_id"]), modality, (0, 0, width, height)
            )
            sample_weight = np.float32(temporal_weight) * valid[None]
            weighted += values * sample_weight
            weights += sample_weight
        covered = weights > 0
        return weighted / np.maximum(weights, 1e-12), covered

    def compose(
        self,
        physical: Tensor,
        prior: Tensor,
        coverage: Tensor,
        modality: Literal["optical", "sar"],
    ) -> tuple[Tensor, Tensor]:
        if physical.shape != prior.shape:
            raise ValueError("temporal prior and physical output must have the same shape")
        if modality == "optical":
            direction = prior / prior.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
            model_amplitude = physical.square().sum(dim=1, keepdim=True).sqrt()
            prior_amplitude = prior.square().sum(dim=1, keepdim=True).sqrt()
            candidate = direction * torch.lerp(
                model_amplitude, prior_amplitude, self.config.optical_amplitude_weight
            )
            violation = ((candidate < 0.0) | (candidate > 1.0)).to(candidate.dtype).mean()
            candidate = candidate.clamp(0.0, 1.0)
        else:
            candidate = torch.lerp(physical, prior, self.config.sar_weight)
            violation = ((candidate < -50.0) | (candidate > 5.0)).to(candidate.dtype).mean()
            candidate = candidate.clamp(-50.0, 5.0)
        return torch.where(coverage.bool(), candidate, physical), violation


def configure_checkpoint_temporal_prior(
    checkpoint: str | Path,
    manifest: str | Path,
    output: str | Path,
    *,
    optical_amplitude_weight: float = 0.75,
    sar_weight: float = 0.80,
) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("format_version", 0)) != 4:
        raise RuntimeError("temporal-prior configuration requires a V3.2 format-v4 checkpoint")
    config = temporal_prior_config(
        manifest,
        optical_amplitude_weight=optical_amplitude_weight,
        sar_weight=sar_weight,
    )
    values = config.to_dict()
    payload["temporal_prior"] = values
    payload.setdefault("config", {})["temporal_prior"] = values
    # A new physical report must establish the gate for this composed predictor.
    payload.setdefault("quality_gates", {})["physical"] = False
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return values
