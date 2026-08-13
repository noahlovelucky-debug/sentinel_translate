"""Prepare a bounded local raw-TIFF cache for paired V2 feasibility only.

The default command is a dry-run.  It builds the four fixed 64-sample indexes,
stats only the files they reference, and prints the hard budget decision.  Use
``--execute`` only after reviewing that report; this script never launches
training or decodes raster pixels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from sentinel_v3.paired_temporal_local_cache import (
    DEFAULT_BUDGET_BYTES,
    DEFAULT_CACHE_ROOT,
    DEFAULT_COPY_RATE_BYTES_PER_SECOND,
    DEFAULT_MINIMUM_FREE_BYTES,
    assert_paired_temporal_feasibility_cache_budget,
    build_paired_temporal_feasibility_cache_plan,
    materialize_paired_temporal_feasibility_cache,
)

GIB = 1024**3
MIB = 1024**2
_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "paired_temporal_v2_feasibility.yaml"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--destination", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--budget-gib", type=float, default=DEFAULT_BUDGET_BYTES / GIB)
    parser.add_argument(
        "--minimum-free-gib", type=float, default=DEFAULT_MINIMUM_FREE_BYTES / GIB
    )
    parser.add_argument(
        "--rate-limit-mib-per-second",
        "--copy-rate-mib-per-second",
        dest="rate_limit_mib_per_second",
        type=float,
        default=DEFAULT_COPY_RATE_BYTES_PER_SECOND / MIB,
        help="serial source-copy rate limit; use 0 to disable throttling",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="copy approved raw files after printing the dry-run report",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="re-copy existing same-size destination files",
    )
    return parser


def _bytes_from_gib(value: float, option: str) -> int:
    if value <= 0.0:
        raise ValueError(f"{option} must be positive")
    return int(value * GIB)


def _rate_bytes_from_mib(value: float, option: str) -> int:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{option} must be a finite non-negative number")
    if value == 0.0:
        return 0
    result = int(value * MIB)
    if result <= 0:
        raise ValueError(f"{option} is too small to represent at least one byte per second")
    return result


def main() -> None:
    args = _parser().parse_args()
    budget_bytes = _bytes_from_gib(args.budget_gib, "--budget-gib")
    minimum_free_bytes = _bytes_from_gib(args.minimum_free_gib, "--minimum-free-gib")
    rate_limit_bytes_per_second = _rate_bytes_from_mib(
        args.rate_limit_mib_per_second, "--rate-limit-mib-per-second"
    )
    plan = build_paired_temporal_feasibility_cache_plan(
        args.config,
        destination_root=args.destination,
        budget_bytes=budget_bytes,
        minimum_free_bytes=minimum_free_bytes,
    )
    report = plan.report()
    report["copy_rate_limit_bytes_per_second"] = rate_limit_bytes_per_second
    report["copy_rate_limit_mib_per_second"] = args.rate_limit_mib_per_second
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.execute:
        return
    assert_paired_temporal_feasibility_cache_budget(plan)
    result = materialize_paired_temporal_feasibility_cache(
        plan,
        resume=not args.no_resume,
        rate_limit_bytes_per_second=rate_limit_bytes_per_second,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
