from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import torch

from sentinel_v3.evaluation import ManifestCropDataset, load_checkpoint
from sentinel_v3.losses import masked_mean, spectral_angle
from sentinel_v3.sensors import SENTINEL1, SENTINEL2


def manifest_metadata(item: dict[str, object], device: torch.device) -> torch.Tensor:
    s1_day = date.fromisoformat(str(item["s1_date"]))
    s2_day = date.fromisoformat(str(item["s2_date"]))
    phase_s1 = 2.0 * math.pi * s1_day.timetuple().tm_yday / 366.0
    phase_s2 = 2.0 * math.pi * s2_day.timetuple().tm_yday / 366.0
    orbit = -1.0 if item["orbit"] == "ascending" else 1.0
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol-hash", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    dataset = ManifestCropDataset(args.manifest, "validation_temporal")
    dataset.records.sort(key=lambda record: record["pair_id"])
    if args.limit is not None and args.limit < len(dataset.records):
        indices = np.linspace(0, len(dataset.records) - 1, args.limit, dtype=np.int64)
        dataset.records = [dataset.records[int(index)] for index in indices]
    optical_mse = 0.0
    optical_sam = 0.0
    sar_mse = 0.0
    sar_bias = 0.0
    for index, item in enumerate(dataset):
        record = dataset.records[index]
        item["gsd"] = record["gsd"]
        s2 = item["s2"].unsqueeze(0).to(device)
        sar = item["sar"].unsqueeze(0).to(device)
        valid = item["valid"].unsqueeze(0).to(device)
        metadata = manifest_metadata(item, device)
        gsd = float(record["gsd"])
        with torch.inference_mode():
            optical = model.physical(
                sar,
                SENTINEL1,
                SENTINEL2,
                valid,
                input_gsd=gsd,
                target_gsd=gsd,
                metadata=metadata,
            )[0]
            radar = model.physical(
                s2,
                SENTINEL2,
                SENTINEL1,
                valid,
                input_gsd=gsd,
                target_gsd=gsd,
                metadata=metadata,
            )[0]
        optical_mse += float(masked_mean((optical - s2).square(), valid))
        optical_sam += float(spectral_angle(optical, s2, valid) * (180 / math.pi))
        sar_mse += float(masked_mean((radar - sar).square(), valid))
        sar_bias += float(masked_mean(radar - sar, valid).abs())
    report = {
        "model": "v3.1_physical",
        "checkpoint": args.checkpoint,
        "split": "validation_temporal",
        "samples": len(dataset),
        "protocol_hash": args.protocol_hash,
        "sar2opt_rmse": math.sqrt(optical_mse / len(dataset)),
        "sar2opt_sam_deg": optical_sam / len(dataset),
        "opt2sar_rmse_db": math.sqrt(sar_mse / len(dataset)),
        "opt2sar_physical_bias_db": sar_bias / len(dataset),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
