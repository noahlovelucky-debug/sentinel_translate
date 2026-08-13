"""Build a strict-causal temporal pilot and test it on real Sentinel TIFFs.

The script intentionally stops before any expensive distributed training.  It
is the go/no-go experiment for Causal Anchor-Delta Transport: a trained model
must outperform copying the real past anchor and must lose accuracy when the
causal source sequence is blanked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel_v3.temporal_data import (
    ALL_DIRECTIONS,
    TemporalRasterDataset,
    build_temporal_index,
    load_pair_records,
    write_temporal_index,
)
from sentinel_v3.temporal_training import (
    TemporalPilotConfig,
    load_temporal_checkpoint,
    train_temporal_pilot,
)
from sentinel_v3.temporal_v1 import CausalAnchorDeltaTransport, TemporalModelConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/datasets/sentinel_translate_v32_2017_2024/manifests/pairs.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports_temporal_v1/feasibility"),
    )
    parser.add_argument("--direction", choices=ALL_DIRECTIONS, required=True)
    parser.add_argument("--stage", choices=("physical", "detail", "flow", "balance"), default="physical")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="required for detail/flow/balance; loads compatible model weights but never optimizer state",
    )
    parser.add_argument("--source-frames", type=int, default=4)
    parser.add_argument("--horizon-days", type=int, default=180)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    records = load_pair_records(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    indexes = {}
    datasets = {}
    for split, limit in (
        ("train", args.train_samples),
        ("validation_temporal", args.validation_samples),
    ):
        index = build_temporal_index(
            records,
            source_frames=args.source_frames,
            horizon_days=args.horizon_days,
            split=split,
            orbit="ascending",
            directions=(args.direction,),
            max_samples=limit,
        )
        if not index.samples:
            raise RuntimeError(f"no strict-causal {args.direction} samples for {split}")
        index.assert_causality(records)
        index_path = args.output / f"{split}_temporal_index.jsonl"
        write_temporal_index(index_path, index)
        indexes[split] = index
        datasets[split] = TemporalRasterDataset(
            records,
            index,
            crop_size=args.crop_size,
            minimum_valid_fraction=0.80,
            seed=args.seed,
            cache_in_memory=args.cache,
            max_cache_items=max(args.batch_size * 4, 8) if args.cache else None,
        )
    if args.stage != "physical" and args.init_checkpoint is None:
        raise SystemExit("--init-checkpoint is required for non-physical temporal stages")
    model = CausalAnchorDeltaTransport(
        TemporalModelConfig(
            width=args.width,
            latent_channels=args.latent_channels,
            maximum_horizon_days=args.horizon_days,
        )
    )
    if args.init_checkpoint is not None:
        load_temporal_checkpoint(args.init_checkpoint, model, direction=args.direction)
    config = TemporalPilotConfig(
        direction=args.direction,
        stage=args.stage,
        max_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    report = train_temporal_pilot(
        model,
        datasets["train"],
        datasets["validation_temporal"],
        config,
        output_dir=args.output,
        device=args.device,
    )
    report["data"] = {
        split: {
            "index_samples": len(index),
            "dataset_samples": len(datasets[split]),
            "index_path": str((args.output / f"{split}_temporal_index.jsonl").resolve()),
        }
        for split, index in indexes.items()
    }
    report["data"]["protocol"] = {
        "source_frames": args.source_frames,
        "horizon_days": args.horizon_days,
        "orbit": "ascending",
        "single_direction": args.direction,
    }
    path = args.output / "temporal_pilot_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["feasibility"]["recommended_for_scale_up"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
