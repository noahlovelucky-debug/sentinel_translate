"""Run equal-capacity downstream SCL probes from validated cached tensors."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from sentinel_v3.downstream_probe import (
    PROBE_GROUPS,
    STATISTICAL_COMPARISONS,
    ProbeTrainConfig,
    cache_contract,
    load_probe_cache,
    run_probe_suite,
    scene_scores_from_suite,
    summarize_probe_statistics,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        nargs="+",
        required=True,
        help="one or more cache .pt files, or directories containing .pt cache files",
    )
    parser.add_argument("--output", required=True, help="destination JSON report")
    parser.add_argument("--device", default="auto", help="torch device or 'auto'")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--summary-split", default="test")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(13, 17, 29),
        metavar="SEED",
        help="one or more seeds used by every selected group",
    )
    parser.add_argument("--groups", nargs="+", choices=PROBE_GROUPS, default=PROBE_GROUPS)
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="write per-seed results only; useful for per-group or per-seed parallel jobs",
    )
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cache = load_probe_cache(args.cache)
    config = ProbeTrainConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        width=args.width,
        train_split=args.train_split,
        dev_split=args.dev_split,
        test_split=args.test_split,
        augment=not args.no_augment,
    )
    suite = run_probe_suite(
        cache,
        config=config,
        seeds=tuple(args.seeds),
        groups=tuple(args.groups),
        device=_device(args.device),
    )
    required_summary_groups = {
        group for pair in STATISTICAL_COMPARISONS.values() for group in pair
    }
    if args.skip_summary:
        scene_scores: dict[str, dict[str, float]] | None = None
        statistics: dict[str, object] | None = None
    else:
        selected_groups = set(args.groups)
        missing_summary_groups = sorted(required_summary_groups.difference(selected_groups))
        if missing_summary_groups:
            raise ValueError(
                "summary requires all comparison groups; pass --skip-summary for a subset: "
                f"{missing_summary_groups}"
            )
        scene_scores = scene_scores_from_suite(suite, split=args.summary_split)
        statistics = summarize_probe_statistics(
            scene_scores,
            bootstrap_resamples=10_000,
            permutation_samples=10_000,
        )
    report = {
        "format_version": 1,
        "cache_contract": cache_contract(),
        "cache_samples": len(cache),
        "seeds": list(args.seeds),
        "groups": list(args.groups),
        "summary_split": args.summary_split,
        "summary_skipped": bool(args.skip_summary),
        "suite": suite.to_dict(),
        "scene_scores": scene_scores,
        "statistics": statistics,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


if __name__ == "__main__":
    main()
