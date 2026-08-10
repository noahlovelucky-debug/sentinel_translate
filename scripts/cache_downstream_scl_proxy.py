"""Prepare, populate, and verify the leakage-safe SCL proxy cache.

The script intentionally has no training behavior.  ``prepare`` materializes the
fixed benchmark index, ``cache`` writes one rank's SAR-only physical outputs, and
``finalize`` refuses to publish a cache until every indexed entry verifies.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from sentinel_v3.downstream_data import (
    CachePlan,
    benchmark_samples,
    cache_rank_samples,
    file_sha256,
    finalize_cache,
    materialize_probe_cache,
    prepare_cache,
)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _path_value(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"downstream cache config requires paths.{key}")
    return value


def load_plan(config_path: str | Path) -> CachePlan:
    """Load the small cache-specific config and bind it to its digest."""

    resolved_config = Path(config_path).resolve()
    values = _mapping(yaml.safe_load(resolved_config.read_text(encoding="utf-8")), "config")
    paths = _mapping(values.get("paths", values), "paths")
    checkpoint_value = paths.get("checkpoint", values.get("checkpoint"))
    checkpoint_sha256 = values.get("checkpoint_sha256")
    if isinstance(checkpoint_value, dict):
        checkpoint = checkpoint_value.get("path")
        checkpoint_sha256 = checkpoint_value.get("sha256", checkpoint_sha256)
    else:
        checkpoint = checkpoint_value
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("downstream cache config requires checkpoint path")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError("downstream cache config requires checkpoint SHA-256")
    cache = _mapping(values.get("cache", {}), "cache")
    crop_size = cache.get("crop_size", values.get("crop_size", 256))
    if not isinstance(crop_size, int):
        raise TypeError("downstream cache crop_size must be an integer")
    return CachePlan(
        manifest=Path(_path_value(paths, "manifest")).resolve(),
        train_shards=Path(_path_value(paths, "train_shards")).resolve(),
        checkpoint=Path(checkpoint).resolve(),
        checkpoint_sha256=checkpoint_sha256,
        cache_root=Path(_path_value(paths, "cache_root")).resolve(),
        config_path=resolved_config,
        config_sha256=file_sha256(resolved_config),
        crop_size=crop_size,
    )


def _rank_argument(value: int | None, environment: str, default: int) -> int:
    if value is not None:
        return value
    raw = os.environ.get(environment)
    return int(raw) if raw is not None else default


def _device(value: str) -> torch.device:
    """Map torchrun's local rank to its GPU when the caller leaves ``cuda`` default."""

    local_rank = os.environ.get("LOCAL_RANK")
    if value == "cuda" and local_rank is not None:
        return torch.device(f"cuda:{local_rank}")
    return torch.device(value)


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _prepare(plan: CachePlan) -> None:
    samples = benchmark_samples(plan)
    provenance = prepare_cache(plan, samples)
    _print(
        {
            "action": "prepare",
            "cache_root": str(plan.cache_root),
            "samples": len(samples),
            "provenance_sha256": file_sha256(plan.cache_root / "provenance.json"),
            "protocol": provenance["protocol_version"],
        }
    )


def _cache(plan: CachePlan, args: argparse.Namespace) -> None:
    rank = _rank_argument(args.rank, "RANK", 0)
    world_size = _rank_argument(args.world_size, "WORLD_SIZE", 1)
    result = cache_rank_samples(
        plan,
        rank=rank,
        world_size=world_size,
        device=_device(args.device),
        resume=not args.no_resume,
    )
    _print({"action": "cache", "rank": rank, "world_size": world_size, **result})


def _finalize(plan: CachePlan) -> None:
    manifest = finalize_cache(plan)
    _print(
        {
            "action": "finalize",
            "cache_root": str(plan.cache_root),
            "entries": len(manifest["entries"]),
            "cache_manifest": str(plan.cache_root / "cache_manifest.json"),
        }
    )


def _materialize(plan: CachePlan, args: argparse.Namespace) -> None:
    manifest = materialize_probe_cache(
        plan,
        dev_tiles=tuple(args.dev_tiles),
        chunk_size=args.chunk_size,
    )
    _print(
        {
            "action": "materialize",
            "materialized_root": str(plan.cache_root / "materialized"),
            "chunks": len(manifest["entries"]),
            "manifest": str(plan.cache_root / "materialized" / "manifest.json"),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="downstream cache YAML")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("prepare", help="write deterministic sample index and provenance")
    subcommands.add_parser("finalize", help="verify every entry and write cache manifest")
    materialize = subcommands.add_parser(
        "materialize", help="join finalized synthetic outputs with real probe inputs"
    )
    materialize.add_argument(
        "--dev-tile",
        dest="dev_tiles",
        action="append",
        required=True,
        help="fixed canonical train tile assigned to dev; repeat for each tile",
    )
    materialize.add_argument(
        "--chunk-size", type=int, default=32, help="samples per materialized probe chunk"
    )
    cache = subcommands.add_parser("cache", help="cache this distributed rank")
    cache.add_argument("--rank", type=int, help="rank; defaults to RANK or 0")
    cache.add_argument("--world-size", type=int, help="world size; defaults to WORLD_SIZE or 1")
    cache.add_argument("--device", default="cuda", help="torch device for physical inference")
    cache.add_argument("--no-resume", action="store_true", help="regenerate existing entries")
    all_steps = subcommands.add_parser(
        "all", help="prepare, cache, and finalize one local rank"
    )
    all_steps.add_argument(
        "--device", default="cuda", help="torch device for physical inference"
    )
    all_steps.add_argument(
        "--no-resume", action="store_true", help="regenerate existing entries"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan = load_plan(args.config)
    if args.command == "prepare":
        _prepare(plan)
    elif args.command == "cache":
        _cache(plan, args)
    elif args.command == "finalize":
        _finalize(plan)
    elif args.command == "materialize":
        _materialize(plan, args)
    elif args.command == "all":
        if _rank_argument(None, "WORLD_SIZE", 1) != 1:
            raise RuntimeError("all only supports WORLD_SIZE=1; use prepare/cache/finalize")
        _prepare(plan)
        _cache(
            plan,
            argparse.Namespace(
                rank=0, world_size=1, device=args.device, no_resume=args.no_resume
            ),
        )
        _finalize(plan)
    else:  # pragma: no cover - argparse keeps this exhaustive.
        raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
