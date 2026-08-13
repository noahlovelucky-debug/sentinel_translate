"""Build the immutable full SOPAT V4 role index without reading raster pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from sentinel_v4.data import build_sopat_v4_index, write_sopat_v4_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/sopat_v4_full_index_source.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/noah/datasets/sopat_v4_2017_2024/index.jsonl"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise TypeError("SOPAT index config requires a data mapping")
    data = payload["data"]
    index = build_sopat_v4_index(
        data["manifest"],
        splits=(data["train_split"], data["validation_split"]),
        min_observations=int(data["minimum_observations"]),
        max_observations=int(data["maximum_observations"]),
        horizon_days=int(data["horizon_days"]),
        anchor_max_delta_days=int(data["anchor_pair_max_delta_days"]),
        max_anchors_per_query=int(data["maximum_anchors_per_query"]),
        translation_tolerance_days=int(data["translation_max_delta_days"]),
        orbit=str(data["orbit"]),
    )
    file_sha256 = write_sopat_v4_index(args.output, index)
    counts = {
        f"{direction}/{split}": len(index.select(direction=direction, split=split))
        for direction in ("sar_to_optical", "optical_to_sar")
        for split in (str(data["train_split"]), str(data["validation_split"]))
    }
    print(
        json.dumps(
            {
                "output": str(args.output),
                "file_sha256": file_sha256,
                "content_sha256": index.content_hash,
                "protocol_sha256": index.protocol_hash,
                "counts": counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
