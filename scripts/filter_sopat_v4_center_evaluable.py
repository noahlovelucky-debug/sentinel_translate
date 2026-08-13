"""Filter paired-temporal validation rows by fixed-center evaluability only.

The operation deliberately leaves training rows untouched.  It is used by the
full V4 index publisher before role migration, so the cache's explicit V3
indexes and the V4 role index retain exactly the same sample rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel_v3.dataset_builder import file_sha256
from sentinel_v3.paired_temporal_data import (
    PairedTemporalIndex,
    PairedTemporalRasterDataset,
    load_paired_temporal_index,
    write_paired_temporal_index,
)

DIRECTIONS = ("sar_to_optical", "optical_to_sar")


def filter_validation_index(
    manifest: Path,
    index: PairedTemporalIndex,
    *,
    crop_size: int,
    maximum_observations: int,
) -> tuple[PairedTemporalIndex, tuple[str, ...]]:
    """Return rows whose fixed-center target/anchor support is nonempty.

    No target-pixel measurement is made for training.  The validation rule is
    exactly the evaluator's minimum support condition, not a quality ranking.
    """

    dataset = PairedTemporalRasterDataset(
        manifest,
        index,
        crop_size=crop_size,
        crop_mode="center",
        max_observations=maximum_observations,
        registration_audit=False,
    )
    kept = []
    dropped: list[str] = []
    for position, sample in enumerate(dataset.samples):
        try:
            dataset[position]
        except RuntimeError as error:
            if "no evaluable target/anchor pixels" not in str(error):
                raise
            dropped.append(sample.sample_id)
        else:
            kept.append(sample)
    return PairedTemporalIndex(config=index.config, samples=tuple(kept)), tuple(dropped)


def _path_for(source_root: Path, direction: str, label: str) -> Path:
    return source_root / direction / f"{label}.jsonl"


def filter_indexes(
    manifest: Path,
    source_root: Path,
    output_root: Path,
    *,
    crop_size: int,
    maximum_observations: int,
    validation_label: str = "validation",
) -> dict[str, object]:
    """Write immutable center-evaluable copies of explicit V3 indexes.

    This compatibility CLI retains the original ``train.jsonl`` /
    ``validation.jsonl`` layout.  Full V4 publication uses
    :func:`filter_validation_index` directly and writes actual split names.
    """

    result: dict[str, object] = {
        "protocol": "sopat_v4_fixed_center_evaluable_v2",
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "crop_size": crop_size,
        "train_pixel_filtered": False,
        "directions": {},
    }
    directional: dict[str, object] = {}
    for direction in DIRECTIONS:
        split_report: dict[str, object] = {}
        train_source = _path_for(source_root, direction, "train")
        train_index = load_paired_temporal_index(train_source, direction=direction)  # type: ignore[arg-type]
        train_destination = _path_for(output_root, direction, "train")
        write_paired_temporal_index(train_destination, train_index)
        split_report["train"] = {
            "source": str(train_source),
            "source_sha256": file_sha256(train_source),
            "output": str(train_destination),
            "output_sha256": file_sha256(train_destination),
            "input_samples": len(train_index),
            "output_samples": len(train_index),
            "pixel_filtered": False,
        }
        validation_source = _path_for(source_root, direction, validation_label)
        validation_index = load_paired_temporal_index(validation_source, direction=direction)  # type: ignore[arg-type]
        filtered, dropped = filter_validation_index(
            manifest,
            validation_index,
            crop_size=crop_size,
            maximum_observations=maximum_observations,
        )
        validation_destination = _path_for(output_root, direction, validation_label)
        write_paired_temporal_index(validation_destination, filtered)
        split_report["validation"] = {
            "source": str(validation_source),
            "source_sha256": file_sha256(validation_source),
            "output": str(validation_destination),
            "output_sha256": file_sha256(validation_destination),
            "input_samples": len(validation_index),
            "output_samples": len(filtered),
            "dropped_sample_ids": list(dropped),
            "pixel_filtered": True,
        }
        directional[direction] = split_report
    result["directions"] = directional
    output_root.mkdir(parents=True, exist_ok=True)
    report = output_root / "center_filter_report.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--maximum-observations", type=int, default=4)
    parser.add_argument("--validation-label", default="validation")
    args = parser.parse_args()
    report = filter_indexes(
        args.manifest,
        args.source_root,
        args.output_root,
        crop_size=args.crop_size,
        maximum_observations=args.maximum_observations,
        validation_label=args.validation_label,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
