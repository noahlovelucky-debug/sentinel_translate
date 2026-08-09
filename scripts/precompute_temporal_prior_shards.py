#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sentinel_v3.schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER
from sentinel_v3.temporal_prior import TemporalPriorStore, temporal_prior_config

_STORE: TemporalPriorStore | None = None


def _initialize(manifest: str, shard_index: str) -> None:
    global _STORE
    torch.set_num_threads(1)
    _STORE = TemporalPriorStore(temporal_prior_config(manifest, shard_index=shard_index))


def _build_one(task: tuple[int, str, str]) -> dict[str, object]:
    index, source_path, destination_path = task
    assert _STORE is not None
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    pair_ids = [str(value) for value in source["pair_id"]]
    if len(set(pair_ids)) != 1:
        raise RuntimeError(f"source shard mixes pair IDs: {source_path}")
    pair_id = pair_ids[0]
    destination = Path(destination_path)
    if destination.is_file():
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        if (
            int(existing.get("format_version", 0)) == 1
            and existing.get("pair_id") == pair_ids
            and torch.equal(existing["window"], source["window"])
        ):
            return {
                "index": index,
                "source": source_path,
                "path": str(destination),
                "count": len(pair_ids),
                "pair_id": pair_id,
            }
    _, location_id, s1_date, orbit, s2_date = pair_id.split(":")
    windows = source["window"].cpu().numpy().astype(np.int64)
    optical, optical_coverage = _STORE.windows_prior(
        location_id=location_id,
        acquired=s2_date,
        modality="optical",
        orbit=orbit,
        windows=windows.tolist(),
        exclude_pair_id=pair_id,
    )
    sar, sar_coverage = _STORE.windows_prior(
        location_id=location_id,
        acquired=s1_date,
        modality="sar",
        orbit=orbit,
        windows=windows.tolist(),
        exclude_pair_id=pair_id,
    )
    payload = {
        "format_version": 1,
        "pair_id": pair_ids,
        "window": source["window"],
        "optical": torch.from_numpy(optical.astype(np.float16)),
        "optical_coverage": torch.from_numpy(optical_coverage).bool(),
        "sar": torch.from_numpy(sar.astype(np.float16)),
        "sar_coverage": torch.from_numpy(sar_coverage).bool(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return {
        "index": index,
        "source": source_path,
        "path": str(destination),
        "count": len(pair_ids),
        "pair_id": pair_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    source_index: dict[str, Any] = json.loads(Path(args.shard_index).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            index,
            str(shard["path"]),
            str(output / f"prior_{index:06d}.pt"),
        )
        for index, shard in enumerate(source_index["shards"])
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    completed: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize,
        initargs=(args.manifest, args.shard_index),
    ) as executor:
        futures = {executor.submit(_build_one, task): task[0] for task in tasks}
        for completed_count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            completed.append(result)
            print(
                json.dumps(
                    {
                        "completed": completed_count,
                        "total": len(tasks),
                        "index": result["index"],
                        "pair_id": result["pair_id"],
                    }
                ),
                flush=True,
            )
    completed.sort(key=lambda item: int(item["index"]))
    prior_config = temporal_prior_config(args.manifest, shard_index=args.shard_index)
    index_payload = {
        "format_version": 2,
        "source_index": str(Path(args.shard_index).resolve()),
        "source_index_sha256": hashlib.sha256(Path(args.shard_index).read_bytes()).hexdigest(),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": prior_config.manifest_sha256,
        "temporal_prior_version": prior_config.version,
        "train_years": list(prior_config.train_years),
        "s2_channel_order": list(S2_CHANNEL_ORDER),
        "sar_channel_order": list(SAR_CHANNEL_ORDER),
        "shards": completed,
    }
    destination = output / "index.json"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
