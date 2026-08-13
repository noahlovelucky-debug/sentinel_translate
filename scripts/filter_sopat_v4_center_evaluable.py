"""Filter explicit paired-temporal validation indexes by fixed-center support.

The source indexes are never modified. Training rows are copied unchanged;
validation rows are retained only when the fixed center crop contains at least
one pixel valid in both the query target and historical target anchor. This is
the exact minimum support required by the masked SOPAT evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel_v3.paired_temporal_data import (
    PairedTemporalIndex,
    PairedTemporalRasterDataset,
    load_paired_temporal_index,
    write_paired_temporal_index,
)

DIRECTIONS = ("sar_to_optical", "optical_to_sar")


def filter_indexes(
    manifest: Path,
    source_root: Path,
    output_root: Path,
    *,
    crop_size: int,
    maximum_observations: int,
) -> dict[str, object]:
    """Write immutable center-evaluable copies and return an audit summary."""

    result: dict[str, object] = {
        "protocol": "sopat_v4_fixed_center_evaluable_v1",
        "manifest": str(manifest),
        "crop_size": crop_size,
        "directions": {},
    }
    directional: dict[str, object] = {}
    for direction in DIRECTIONS:
        split_report: dict[str, object] = {}
        for split in ("train", "validation"):
            source = source_root / direction / f"{split}.jsonl"
            index = load_paired_temporal_index(source, direction=direction)  # type: ignore[arg-type]
            kept = index.samples
            dropped: list[str] = []
            if split == "validation":
                dataset = PairedTemporalRasterDataset(
                    manifest,
                    index,
                    crop_size=crop_size,
                    crop_mode="center",
                    max_observations=maximum_observations,
                    registration_audit=False,
                )
                selected = []
                for position, sample in enumerate(dataset.samples):
                    try:
                        dataset[position]
                    except RuntimeError as error:
                        if "no evaluable target/anchor pixels" not in str(error):
                            raise
                        dropped.append(sample.sample_id)
                    else:
                        selected.append(sample)
                kept = tuple(selected)
            filtered = PairedTemporalIndex(config=index.config, samples=tuple(kept))
            destination = output_root / direction / f"{split}.jsonl"
            write_paired_temporal_index(destination, filtered)
            split_report[split] = {
                "source": str(source),
                "output": str(destination),
                "input_samples": len(index.samples),
                "output_samples": len(kept),
                "dropped_sample_ids": dropped,
            }
        directional[direction] = split_report
    result["directions"] = directional
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "center_filter_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--maximum-observations", type=int, default=4)
    args = parser.parse_args()
    report = filter_indexes(
        args.manifest,
        args.source_root,
        args.output_root,
        crop_size=args.crop_size,
        maximum_observations=args.maximum_observations,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
