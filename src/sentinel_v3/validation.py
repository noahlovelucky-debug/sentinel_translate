from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationProtocol:
    name: str = "validation_temporal_v32"
    split: str = "validation_temporal"
    expected_samples: int = 463
    crop_size: int = 256
    crop: str = "center"
    mask_scl_codes: tuple[int, ...] = (2, 4, 5, 6, 7)
    optical_units: str = "surface_reflectance_0_1"
    sar_units: str = "decibel_backscatter"
    optical_channels: tuple[str, ...] = (
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
    sar_channels: tuple[str, ...] = ("vv", "vh")


def protocol_records(
    manifest: str | Path, protocol: ValidationProtocol | None = None
) -> list[dict[str, Any]]:
    protocol = protocol or ValidationProtocol()
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
    protocol = protocol or ValidationProtocol()
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
