from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .schema import CLEAR_SCL_CODES, S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER


@dataclass(frozen=True)
class ValidationProtocol:
    name: str = "validation_temporal_v32"
    split: str = "validation_temporal"
    expected_samples: int = 463
    crop_size: int = 256
    crop: str = "center"
    mask_scl_codes: tuple[int, ...] = CLEAR_SCL_CODES
    optical_units: str = "surface_reflectance_0_1"
    sar_units: str = "decibel_backscatter"
    optical_channels: tuple[str, ...] = S2_CHANNEL_ORDER
    sar_channels: tuple[str, ...] = SAR_CHANNEL_ORDER


def validation_protocol_sidecar(manifest: str | Path) -> Path:
    return Path(manifest).resolve().parent / "validation_protocol.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_protocol_for_manifest(
    manifest: str | Path, *, crop_size: int | None = None
) -> ValidationProtocol:
    """Load a dataset-owned protocol sidecar, with the historic 463 fallback."""

    manifest_path = Path(manifest).resolve()
    sidecar = validation_protocol_sidecar(manifest_path)
    if not sidecar.is_file():
        protocol = ValidationProtocol()
        return replace(protocol, crop_size=crop_size) if crop_size is not None else protocol
    values = json.loads(sidecar.read_text(encoding="utf-8"))
    expected_hash = values.get("manifest_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise RuntimeError("validation protocol sidecar must declare manifest_sha256")
    if expected_hash != _file_sha256(manifest_path):
        raise RuntimeError("validation protocol sidecar manifest hash does not match")
    optical = tuple(values.get("s2_channel_order", ()))
    sar = tuple(values.get("sar_channel_order", ()))
    if optical != S2_CHANNEL_ORDER:
        raise RuntimeError("validation protocol sidecar must use canonical S2 channel order")
    if sar != SAR_CHANNEL_ORDER:
        raise RuntimeError("validation protocol sidecar must use canonical SAR channel order")
    mask_codes = values.get("mask_scl_codes")
    if not isinstance(mask_codes, (list, tuple)) or tuple(mask_codes) != CLEAR_SCL_CODES:
        raise RuntimeError("validation protocol sidecar must use canonical clear SCL codes")
    units = values.get("units")
    unit_values = units if isinstance(units, dict) else {}
    optical_units = values.get("optical_units", unit_values.get("optical"))
    sar_units = values.get("sar_units", unit_values.get("sar"))
    if optical_units != "surface_reflectance_0_1":
        raise RuntimeError("validation protocol sidecar optical units must be surface_reflectance_0_1")
    if sar_units != "decibel_backscatter":
        raise RuntimeError("validation protocol sidecar SAR units must be decibel_backscatter")
    crop = values.get("crop", {})
    if not isinstance(crop, dict) or crop.get("kind") != "center":
        raise RuntimeError("validation protocol sidecar crop kind must be center")
    if values.get("split") != "validation_temporal":
        raise RuntimeError("validation protocol sidecar split must be validation_temporal")
    expected_samples = values.get("expected_samples")
    if isinstance(expected_samples, bool) or not isinstance(expected_samples, int) or expected_samples <= 0:
        raise RuntimeError("validation protocol sidecar expected_samples must be positive")
    sidecar_crop_size = int(values.get("crop_size", crop.get("size", 256)))
    protocol = ValidationProtocol(
        name=str(values.get("name", "validation_temporal_v32")),
        split="validation_temporal",
        expected_samples=expected_samples,
        crop_size=sidecar_crop_size,
        crop=str(crop.get("kind", values.get("crop_kind", "center"))),
        mask_scl_codes=tuple(int(value) for value in mask_codes),
        optical_units=optical_units,
        sar_units=sar_units,
        optical_channels=optical,
        sar_channels=sar,
    )
    return replace(protocol, crop_size=crop_size) if crop_size is not None else protocol


def protocol_records(
    manifest: str | Path, protocol: ValidationProtocol | None = None
) -> list[dict[str, Any]]:
    protocol = protocol or validation_protocol_for_manifest(manifest)
    records = []
    with Path(manifest).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] == protocol.split or record["refit_split"] == protocol.split:
                records.append(record)
    records.sort(key=lambda record: record["pair_id"])
    if len(records) != protocol.expected_samples:
        raise RuntimeError(
            f"{protocol.name} requires exactly {protocol.expected_samples} pairs, got {len(records)}"
        )
    return records


def validation_protocol_hash(
    manifest: str | Path, protocol: ValidationProtocol | None = None
) -> str:
    protocol = protocol or validation_protocol_for_manifest(manifest)
    records = protocol_records(manifest, protocol)
    payload = {
        "protocol": asdict(protocol),
        "records": [
            {
                "pair_id": record["pair_id"],
                "s1_date": record["s1_date"],
                "s2_date": record["s2_date"],
                "height": record["height"],
                "width": record["width"],
            }
            for record in records
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
