#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from sentinel_v3.data import (
    REGISTRATION_AUDIT_METHOD,
    REGISTRATION_AUDIT_VERSION,
    REGISTRATION_MIN_IMPROVEMENT,
    REGISTRATION_MIN_NCC,
    REGISTRATION_SEARCH_RADIUS_PX,
    estimate_registration_shift,
    high_frequency_eligible,
)


def audit_shard(
    task: tuple[int, int, dict[str, object], str, tuple[int, ...]]
) -> tuple[int, list[int]]:
    torch.set_num_threads(1)
    shard_index, start, descriptor, split, hf_years = task
    shard = torch.load(str(descriptor["path"]), map_location="cpu", weights_only=False)
    eligible = []
    for local in range(int(descriptor["count"])):
        metadata = shard["metadata"][local]
        delta_days = round(abs(float(metadata[0])) * 3.0)
        pair_id = str(shard["pair_id"][local])
        try:
            year = int(pair_id.split(":", 1)[0])
        except ValueError:
            year = -1
        joint_valid = shard["joint_valid"][local].float()
        s2_valid = shard.get("s2_valid", shard["joint_valid"])[local].float()
        shift = (
            estimate_registration_shift(
                shard["s2"][local].float(),
                shard["sar"][local].float(),
                valid=joint_valid,
            )
            if delta_days <= 1
            else torch.tensor(float("inf"))
        )
        if high_frequency_eligible(
            delta_days=delta_days,
            year=year,
            split=split,
            registration_shift_px=shift,
            valid_fraction=float(joint_valid.mean()),
            cloud_shadow_fraction=float(1.0 - s2_valid.mean()),
            train_years=hf_years,
        ):
            eligible.append(start + local)
    return shard_index, eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--hf-years",
        help="comma-separated override; otherwise use hf_years/train_years from the index",
    )
    args = parser.parse_args()
    index_path = Path(args.index).resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if args.hf_years:
        hf_years: Iterable[str] = args.hf_years.split(",")
    else:
        hf_years = index.get("hf_years", index.get("train_years", (2017, 2018)))
    normalized_hf_years = tuple(sorted({int(year) for year in hf_years}))
    if not normalized_hf_years:
        raise ValueError("hf years cannot be empty")
    tasks = []
    start = 0
    for shard_index, descriptor in enumerate(index["shards"]):
        tasks.append(
            (
                shard_index,
                start,
                descriptor,
                str(index.get("split", "unknown")),
                normalized_hf_years,
            )
        )
        start += int(descriptor["count"])
    eligible_indices: list[int] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, (_, values) in enumerate(executor.map(audit_shard, tasks), start=1):
            eligible_indices.extend(values)
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    json.dumps(
                        {
                            "audited_shards": completed,
                            "total_shards": len(tasks),
                            "eligible_patches": len(eligible_indices),
                        }
                    ),
                    flush=True,
                )
    payload = {
        "format_version": 2,
        "source_index": str(index_path),
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "train_years": [int(year) for year in index.get("train_years", normalized_hf_years)],
        "hf_years": list(normalized_hf_years),
        "registration_audited": True,
        "registration_audit": {
            "method": REGISTRATION_AUDIT_METHOD,
            "version": REGISTRATION_AUDIT_VERSION,
            "search_radius_px": REGISTRATION_SEARCH_RADIUS_PX,
            "minimum_ncc": REGISTRATION_MIN_NCC,
            "minimum_improvement": REGISTRATION_MIN_IMPROVEMENT,
        },
        "samples": start,
        "eligible_samples": len(eligible_indices),
        "maximum_shift_px": 0.5,
        "minimum_valid_fraction": 0.8,
        "maximum_cloud_shadow_fraction": 0.2,
        "eligible_indices": sorted(eligible_indices),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
