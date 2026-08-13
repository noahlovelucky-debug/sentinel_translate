"""Plan or materialize the full local paired-temporal acquisition chunk cache.

The command is a dry-run by default.  It never starts training.  ``--execute``
is required to decode source TIFFs and publish local ``.npy`` chunks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel_v3.paired_temporal_chunk_cache import (
    DEFAULT_CHUNK_CACHE_BUDGET_BYTES,
    DEFAULT_CHUNK_CACHE_MINIMUM_FREE_BYTES,
    DEFAULT_CHUNK_CACHE_ROOT,
    DEFAULT_CHUNK_CACHE_WORKERS,
    DEFAULT_WINDOWS_PER_ACQUISITION,
    GIB,
    assert_paired_temporal_chunk_cache_budget,
    build_paired_temporal_chunk_cache_plan,
    materialize_paired_temporal_chunk_cache,
    verify_paired_temporal_chunk_cache,
)

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "paired_temporal_v2_full.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--destination", type=Path, default=DEFAULT_CHUNK_CACHE_ROOT)
    parser.add_argument("--budget-gib", type=float, default=DEFAULT_CHUNK_CACHE_BUDGET_BYTES / GIB)
    parser.add_argument(
        "--minimum-free-gib", type=float, default=DEFAULT_CHUNK_CACHE_MINIMUM_FREE_BYTES / GIB
    )
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument(
        "--windows-per-acquisition", type=int, default=DEFAULT_WINDOWS_PER_ACQUISITION
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_CHUNK_CACHE_WORKERS)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="decode source TIFFs and atomically publish the approved local cache",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="rebuild even hash-valid acquisition chunks",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify an already completed cache and exit; cannot be combined with --execute",
    )
    return parser


def _bytes_from_gib(value: float, option: str) -> int:
    if value <= 0.0:
        raise ValueError(f"{option} must be positive")
    return int(value * GIB)


def main() -> None:
    args = _parser().parse_args()
    if args.verify:
        if args.execute:
            raise ValueError("--verify and --execute cannot be used together")
        print(json.dumps(verify_paired_temporal_chunk_cache(args.destination), indent=2, sort_keys=True))
        return
    plan = build_paired_temporal_chunk_cache_plan(
        args.config,
        destination_root=args.destination,
        budget_bytes=_bytes_from_gib(args.budget_gib, "--budget-gib"),
        minimum_free_bytes=_bytes_from_gib(args.minimum_free_gib, "--minimum-free-gib"),
        crop_size=args.crop_size,
        windows_per_acquisition=args.windows_per_acquisition,
    )
    print(json.dumps(plan.report(), indent=2, sort_keys=True))
    if not args.execute:
        return
    assert_paired_temporal_chunk_cache_budget(plan)
    result = materialize_paired_temporal_chunk_cache(
        plan,
        resume=not args.no_resume,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
