#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from sentinel_v3.data import estimate_registration_shift, high_frequency_eligible


def audit_shard(task: tuple[int, int, dict[str, object], str]) -> tuple[int, list[int]]:
    torch.set_num_threads(1)
    shard_index, start, descriptor, split = task
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
                shard["s2"][local].float(), shard["sar"][local].float()
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
        ):
            eligible.append(start + local)
    return shard_index, eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    index_path = Path(args.index).resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    tasks = []
    start = 0
    for shard_index, descriptor in enumerate(index["shards"]):
        tasks.append((shard_index, start, descriptor, str(index.get("split", "unknown"))))
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
        "format_version": 1,
        "source_index": str(index_path),
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
