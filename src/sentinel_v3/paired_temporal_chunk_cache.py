"""Deduplicated, memory-mapped acquisition cache for paired-temporal V2.

The raw paired-temporal manifest contains complete 2560-pixel acquisitions,
while training consumes many aligned 256-pixel windows from the same scene.
This module materializes each required acquisition once into a local
``values.npy`` / ``valid.npy`` pair.  It deliberately keeps the causal sample
index separate from the cache: a cached query target may be present on disk,
but it can only be read through the ``target_values`` route of its sample.

The cache is restricted to the configured train and validation splits.  A
completed cache has a content-hashed final index; a partial cache is never
accepted by :class:`PairedTemporalChunkDataset`.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .data import registration_shift_audit
from .dataset_builder import PairRecord, file_sha256
from .paired_temporal_data import (
    ALL_DIRECTIONS,
    ALL_TASK_MODES,
    Direction,
    Modality,
    PairedTemporalIndex,
    PairedTemporalIndexConfig,
    PairedTemporalSample,
    assert_paired_temporal_causality,
    build_paired_temporal_index,
    centered_crop_window,
    load_pair_records,
    load_paired_temporal_index,
    write_paired_temporal_index,
)
from .schema import CLEAR_SCL_CODES, S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER

GIB = 1024**3
CHUNK_CACHE_FORMAT_VERSION = 1
DEFAULT_CHUNK_CACHE_ROOT = Path("/home/noah/datasets/sentinel_translate_paired_v2_chunks")
DEFAULT_CHUNK_CACHE_BUDGET_BYTES = 180 * GIB
DEFAULT_CHUNK_CACHE_MINIMUM_FREE_BYTES = 80 * GIB
DEFAULT_CHUNK_CACHE_WORKERS = 4
DEFAULT_WINDOW_SIZE = 256
DEFAULT_WINDOWS_PER_ACQUISITION = 64
DEFAULT_MAX_MMAP_ARRAYS = 64
_METADATA_RESERVE_BYTES = 256 * 1024 * 1024
# ``open_memmap`` writes a compact NumPy header plus one JSON sidecar per
# acquisition.  These conservative per-pending reserves avoid charging a
# resumed run for the entire cache-wide metadata allowance.
_PENDING_ARRAY_HEADER_RESERVE_BYTES = 4096
_PENDING_CHUNK_METADATA_RESERVE_BYTES = 4096
_STATIC_METADATA_RESERVE_BYTES = 4 * 1024 * 1024


class ChunkCacheIntegrityError(RuntimeError):
    """Raised when a completed local acquisition cache is missing or corrupt."""


@dataclass(frozen=True)
class ChunkWindow:
    """One aligned raster window expressed as rasterio-style ``col,row,w,h``."""

    col: int
    row: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if min(self.col, self.row) < 0 or min(self.width, self.height) <= 0:
            raise ValueError("chunk window coordinates and dimensions must be positive")

    def to_list(self) -> list[int]:
        return [self.col, self.row, self.width, self.height]


@dataclass(frozen=True)
class ChunkGrid:
    """A spatial grid and the immutable window table shared by its scenes."""

    grid_id: str
    tile: str
    width: int
    height: int
    crs: str
    transform: tuple[float, ...]
    gsd: float
    windows: tuple[ChunkWindow, ...]
    center_window_index: int = 0

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("a chunk grid needs at least one window")
        if not 0 <= self.center_window_index < len(self.windows):
            raise ValueError("center_window_index is outside the window table")
        for window in self.windows:
            if window.col + window.width > self.width or window.row + window.height > self.height:
                raise ValueError("chunk window lies outside its grid")

    @property
    def center_window(self) -> ChunkWindow:
        return self.windows[self.center_window_index]

    def to_dict(self) -> dict[str, object]:
        return {
            "grid_id": self.grid_id,
            "tile": self.tile,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "transform": list(self.transform),
            "gsd": self.gsd,
            "windows": [window.to_list() for window in self.windows],
            "center_window_index": self.center_window_index,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> ChunkGrid:
        windows = tuple(ChunkWindow(*map(int, value)) for value in values["windows"])  # type: ignore[index,arg-type]
        return cls(
            grid_id=str(values["grid_id"]),
            tile=str(values["tile"]),
            width=int(values["width"]),
            height=int(values["height"]),
            crs=str(values["crs"]),
            transform=tuple(float(value) for value in values["transform"]),  # type: ignore[index,arg-type]
            gsd=float(values["gsd"]),
            windows=windows,
            center_window_index=int(values.get("center_window_index", 0)),
        )


@dataclass(frozen=True)
class ChunkAcquisition:
    """One modality-specific, path-deduplicated acquisition to materialize."""

    acquisition_id: str
    modality: Modality
    grid_id: str
    source_paths: tuple[str, ...]
    relative_directory: Path
    channels: int
    window_count: int

    @property
    def values_relative_path(self) -> Path:
        return self.relative_directory / "values.npy"

    @property
    def valid_relative_path(self) -> Path:
        return self.relative_directory / "valid.npy"

    @property
    def metadata_relative_path(self) -> Path:
        return self.relative_directory / "chunk.json"

    def to_dict(self) -> dict[str, object]:
        return {
            "acquisition_id": self.acquisition_id,
            "modality": self.modality,
            "grid_id": self.grid_id,
            "source_paths": list(self.source_paths),
            "relative_directory": self.relative_directory.as_posix(),
            "channels": self.channels,
            "window_count": self.window_count,
        }


@dataclass(frozen=True)
class ChunkRecordRoute:
    """The only acquisition identities the runtime dataset needs per pair."""

    optical_acquisition_id: str
    sar_acquisition_id: str
    grid_id: str
    split: str

    def to_dict(self) -> dict[str, str]:
        return {
            "optical_acquisition_id": self.optical_acquisition_id,
            "sar_acquisition_id": self.sar_acquisition_id,
            "grid_id": self.grid_id,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> ChunkRecordRoute:
        return cls(
            optical_acquisition_id=str(values["optical_acquisition_id"]),
            sar_acquisition_id=str(values["sar_acquisition_id"]),
            grid_id=str(values["grid_id"]),
            split=str(values["split"]),
        )


@dataclass(frozen=True)
class ChunkCacheIndexEntry:
    """One selected causal index stored in the completed cache."""

    direction: Direction
    split: str
    index: PairedTemporalIndex
    relative_path: Path


@dataclass(frozen=True)
class PairedTemporalChunkCachePlan:
    """Immutable dry-run selection and capacity decision for the chunk cache."""

    config_path: Path
    config_sha256: str
    source_manifest: Path
    source_manifest_sha256: str
    destination_root: Path
    source_root: Path
    train_split: str
    validation_split: str
    crop_size: int
    windows_per_acquisition: int
    indexes: tuple[ChunkCacheIndexEntry, ...]
    grids: tuple[ChunkGrid, ...]
    acquisitions: tuple[ChunkAcquisition, ...]
    routes: Mapping[str, ChunkRecordRoute]
    records: Mapping[str, PairRecord]
    estimated_values_bytes: int
    estimated_valid_bytes: int
    metadata_reserve_bytes: int
    budget_bytes: int
    minimum_free_bytes: int
    free_bytes: int

    @property
    def estimated_target_bytes(self) -> int:
        return self.estimated_values_bytes + self.estimated_valid_bytes + self.metadata_reserve_bytes

    @property
    def estimated_free_after_materialization(self) -> int:
        return self.free_bytes - self.estimated_target_bytes

    @property
    def budget_ok(self) -> bool:
        return self.estimated_target_bytes <= self.budget_bytes

    @property
    def free_space_ok(self) -> bool:
        return self.estimated_free_after_materialization >= self.minimum_free_bytes

    @property
    def allowed_to_materialize(self) -> bool:
        return self.budget_ok and self.free_space_ok

    def report(self) -> dict[str, object]:
        by_modality: dict[str, int] = {"optical": 0, "sar": 0}
        for acquisition in self.acquisitions:
            by_modality[acquisition.modality] += 1
        return {
            "format_version": CHUNK_CACHE_FORMAT_VERSION,
            "action": "dry_run",
            "config": str(self.config_path),
            "config_sha256": self.config_sha256,
            "source_manifest": str(self.source_manifest),
            "source_manifest_sha256": self.source_manifest_sha256,
            "destination_root": str(self.destination_root),
            "splits": [self.train_split, self.validation_split],
            "indexes": [
                {
                    "direction": entry.direction,
                    "split": entry.split,
                    "samples": len(entry.index),
                    "path": entry.relative_path.as_posix(),
                }
                for entry in self.indexes
            ],
            "grids": len(self.grids),
            "windows_per_acquisition": self.windows_per_acquisition,
            "crop_size": self.crop_size,
            "acquisitions": {
                "total": len(self.acquisitions),
                "optical": by_modality["optical"],
                "sar": by_modality["sar"],
            },
            "estimated_values_bytes": self.estimated_values_bytes,
            "estimated_valid_bytes": self.estimated_valid_bytes,
            "metadata_reserve_bytes": self.metadata_reserve_bytes,
            "estimated_target_bytes": self.estimated_target_bytes,
            "budget_bytes": self.budget_bytes,
            "minimum_free_bytes": self.minimum_free_bytes,
            "free_bytes": self.free_bytes,
            "estimated_free_after_materialization": self.estimated_free_after_materialization,
            "budget_ok": self.budget_ok,
            "free_space_ok": self.free_space_ok,
            "allowed_to_materialize": self.allowed_to_materialize,
        }


def build_paired_temporal_chunk_cache_plan(
    config_path: str | Path,
    *,
    destination_root: str | Path = DEFAULT_CHUNK_CACHE_ROOT,
    budget_bytes: int = DEFAULT_CHUNK_CACHE_BUDGET_BYTES,
    minimum_free_bytes: int = DEFAULT_CHUNK_CACHE_MINIMUM_FREE_BYTES,
    crop_size: int | None = None,
    windows_per_acquisition: int = DEFAULT_WINDOWS_PER_ACQUISITION,
    free_bytes: int | None = None,
) -> PairedTemporalChunkCachePlan:
    """Select full train/validation acquisitions without opening any TIFF.

    The paired causal indexes are built in both directions using the supplied
    V2 YAML.  Only records referenced by those train/validation indexes enter
    the route table; test, buffer, and unused-spatial records cannot appear in
    a materialization plan.
    """

    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    if minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes cannot be negative")
    if windows_per_acquisition <= 0:
        raise ValueError("windows_per_acquisition must be positive")
    config_file = _absolute_lexical_path(config_path)
    values = _load_yaml_mapping(config_file)
    data = _required_mapping(values, "data")
    manifest = _absolute_lexical_path(_required_string(data, "manifest"), base=config_file.parent)
    destination = _absolute_lexical_path(destination_root)
    configured_crop_size = _required_int(data, "crop_size")
    selected_crop_size = configured_crop_size if crop_size is None else int(crop_size)
    if selected_crop_size <= 0 or selected_crop_size % 4:
        raise ValueError("crop_size must be positive and divisible by four")
    train_split = _required_string(data, "train_split")
    validation_split = _required_string(data, "validation_split")
    if train_split == validation_split:
        raise ValueError("train and validation splits must differ")
    records = load_pair_records(manifest)
    record_map = {record.pair_id: record for record in records}
    if len(record_map) != len(records):
        raise ValueError("paired temporal manifest contains duplicate pair_id values")
    common = _index_config_from_data(data)
    selected_indexes: list[ChunkCacheIndexEntry] = []
    required_pair_ids: set[str] = set()
    for direction in ALL_DIRECTIONS:
        for split in (train_split, validation_split):
            index = build_paired_temporal_index(
                records,
                direction=direction,
                split=split,
                **common,
            )
            if not index:
                raise RuntimeError(f"paired chunk cache selection is empty: {direction}/{split}")
            assert_paired_temporal_causality(index, record_map, asset_root=manifest.parent)
            selected_indexes.append(
                ChunkCacheIndexEntry(
                    direction=direction,
                    split=split,
                    index=index,
                    relative_path=Path("indexes") / direction / f"{split}.jsonl",
                )
            )
            for sample in index:
                required_pair_ids.add(sample.query_pair_id)
                required_pair_ids.add(sample.anchor_pair_id)
                required_pair_ids.update(sample.observation_pair_ids)
    selected_records = {pair_id: record_map[pair_id] for pair_id in sorted(required_pair_ids)}
    allowed_splits = {train_split, validation_split}
    invalid_records = [
        record.pair_id for record in selected_records.values() if record.split not in allowed_splits
    ]
    if invalid_records:
        raise RuntimeError(
            "chunk cache index reached a record outside train/validation: "
            + ", ".join(invalid_records[:3])
        )

    grids_by_key: dict[tuple[object, ...], ChunkGrid] = {}
    routes: dict[str, ChunkRecordRoute] = {}
    pending_assets: dict[tuple[str, tuple[str, ...]], tuple[Modality, str, tuple[str, ...]]] = {}
    source_root = manifest.parent
    for record in selected_records.values():
        grid = _grid_for_record(
            record,
            crop_size=selected_crop_size,
            windows_per_acquisition=windows_per_acquisition,
        )
        grid_key = _grid_key(record)
        prior_grid = grids_by_key.setdefault(grid_key, grid)
        if prior_grid != grid:
            raise RuntimeError(f"inconsistent grid windows for {record.pair_id}")
        optical_paths = _record_source_paths(record, "optical", source_root)
        sar_paths = _record_source_paths(record, "sar", source_root)
        optical_id = _acquisition_id("optical", optical_paths)
        sar_id = _acquisition_id("sar", sar_paths)
        routes[record.pair_id] = ChunkRecordRoute(
            optical_acquisition_id=optical_id,
            sar_acquisition_id=sar_id,
            grid_id=grid.grid_id,
            split=record.split,
        )
        for modality, paths, acquisition_id in (
            ("optical", optical_paths, optical_id),
            ("sar", sar_paths, sar_id),
        ):
            key = (acquisition_id, paths)
            existing = pending_assets.get(key)
            if existing is not None and existing[1] != grid.grid_id:
                raise RuntimeError(
                    "one physical acquisition was assigned to incompatible grids: " + acquisition_id
                )
            pending_assets[key] = (modality, grid.grid_id, paths)

    _assert_plan_causality(selected_indexes, routes)
    grids = tuple(sorted(grids_by_key.values(), key=lambda grid: grid.grid_id))
    grid_by_id = {grid.grid_id: grid for grid in grids}
    acquisitions: list[ChunkAcquisition] = []
    for acquisition_id, paths in sorted(pending_assets):
        modality, grid_id, source_paths = pending_assets[(acquisition_id, paths)]
        grid = grid_by_id[grid_id]
        acquisitions.append(
            ChunkAcquisition(
                acquisition_id=acquisition_id,
                modality=modality,
                grid_id=grid_id,
                source_paths=source_paths,
                relative_directory=Path("acquisitions") / modality / acquisition_id,
                channels=len(S2_CHANNEL_ORDER) if modality == "optical" else len(SAR_CHANNEL_ORDER),
                window_count=len(grid.windows),
            )
        )
    values_bytes = sum(
        acquisition.window_count
        * acquisition.channels
        * selected_crop_size
        * selected_crop_size
        * np.dtype(np.float16).itemsize
        for acquisition in acquisitions
    )
    valid_bytes = sum(
        acquisition.window_count * selected_crop_size * selected_crop_size * np.dtype(np.uint8).itemsize
        for acquisition in acquisitions
    )
    available = free_bytes
    if available is None:
        existing_parent = _existing_parent(destination)
        available = shutil.disk_usage(existing_parent).free
    return PairedTemporalChunkCachePlan(
        config_path=config_file,
        config_sha256=file_sha256(config_file),
        source_manifest=manifest,
        source_manifest_sha256=file_sha256(manifest),
        destination_root=destination,
        source_root=source_root,
        train_split=train_split,
        validation_split=validation_split,
        crop_size=selected_crop_size,
        windows_per_acquisition=windows_per_acquisition,
        indexes=tuple(selected_indexes),
        grids=grids,
        acquisitions=tuple(acquisitions),
        routes=routes,
        records=selected_records,
        estimated_values_bytes=values_bytes,
        estimated_valid_bytes=valid_bytes,
        metadata_reserve_bytes=_METADATA_RESERVE_BYTES,
        budget_bytes=budget_bytes,
        minimum_free_bytes=minimum_free_bytes,
        free_bytes=int(available),
    )


def assert_paired_temporal_chunk_cache_budget(plan: PairedTemporalChunkCachePlan) -> None:
    """Fail before any write when the planned cache violates a hard floor."""

    if not plan.budget_ok:
        raise RuntimeError(
            f"paired chunk cache estimate {plan.estimated_target_bytes} exceeds budget {plan.budget_bytes}"
        )
    if not plan.free_space_ok:
        raise RuntimeError(
            "paired chunk cache would leave less than the required free space: "
            f"{plan.estimated_free_after_materialization} < {plan.minimum_free_bytes}"
        )


def materialize_paired_temporal_chunk_cache(
    plan: PairedTemporalChunkCachePlan,
    *,
    resume: bool = True,
    workers: int = DEFAULT_CHUNK_CACHE_WORKERS,
) -> dict[str, object]:
    """Write normalized local chunks once per acquisition and publish atomically.

    A valid per-acquisition ``chunk.json`` makes interrupted runs resumable.
    The top-level ``cache_index.json`` is written only after every selected
    chunk verifies, so consumers cannot accidentally use a partial cache.
    """

    # The plan budget is immutable.  Do not use its recorded free-space value
    # here: it may be stale and a resumed cache needs only its pending chunks.
    if not plan.budget_ok:
        raise RuntimeError(
            f"paired chunk cache estimate {plan.estimated_target_bytes} exceeds budget {plan.budget_bytes}"
        )
    if workers <= 0:
        raise ValueError("workers must be positive")
    root = plan.destination_root
    root.mkdir(parents=True, exist_ok=True)
    grid_by_id = {grid.grid_id: grid for grid in plan.grids}
    reusable: list[ChunkAcquisition] = []
    pending: list[ChunkAcquisition] = []
    for acquisition in plan.acquisitions:
        if resume and _verify_chunk_directory(root, acquisition, grid_by_id[acquisition.grid_id]):
            reusable.append(acquisition)
        else:
            pending.append(acquisition)
    # Count only bytes that still need a staged local write.  Invalid existing
    # chunks remain until replacement is atomically published, so each pending
    # item is conservatively charged for a complete new values/valid pair.
    remaining_required_bytes = _remaining_materialization_bytes(
        pending,
        grid_by_id,
    )
    current_free = shutil.disk_usage(_existing_parent(root)).free
    if current_free - remaining_required_bytes < plan.minimum_free_bytes:
        raise RuntimeError(
            "paired chunk cache no longer has the required free-space reserve: "
            f"{current_free - remaining_required_bytes} < {plan.minimum_free_bytes}"
        )
    # A top-level index is the completed-cache publication marker.  Remove it
    # only after the read-only reuse inspection, before touching routing/index
    # metadata, so an interrupted rebuild is never consumed as complete.
    (root / "cache_index.json").unlink(missing_ok=True)
    _write_static_cache_metadata(plan)

    written: list[dict[str, object]] = []
    if workers == 1:
        for acquisition in pending:
            written.append(_materialize_acquisition(root, acquisition, grid_by_id[acquisition.grid_id]))
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="paired-chunk") as executor:
            futures = {
                executor.submit(_materialize_acquisition, root, acquisition, grid_by_id[acquisition.grid_id]): acquisition
                for acquisition in pending
            }
            for future in as_completed(futures):
                written.append(future.result())
    written_by_id = {str(entry["acquisition_id"]): entry for entry in written}
    chunk_entries: list[dict[str, object]] = []
    for acquisition in plan.acquisitions:
        grid = grid_by_id[acquisition.grid_id]
        if not _verify_chunk_directory(root, acquisition, grid):
            raise ChunkCacheIntegrityError(
                f"chunk failed verification after materialization: {acquisition.acquisition_id}"
            )
        entry = written_by_id.get(acquisition.acquisition_id)
        if entry is None:
            entry = _load_chunk_metadata(root / acquisition.metadata_relative_path)
        chunk_entries.append(entry)
    index_payload = _completed_cache_index_payload(plan, chunk_entries)
    _atomic_json(root / "cache_index.json", index_payload)
    return {
        "format_version": CHUNK_CACHE_FORMAT_VERSION,
        "destination_root": str(root),
        "copied_acquisitions": len(written),
        "reused_acquisitions": len(reusable),
        "acquisitions": len(plan.acquisitions),
        "workers": workers,
        "cache_index": str(root / "cache_index.json"),
    }


def verify_paired_temporal_chunk_cache(cache_root: str | Path) -> dict[str, object]:
    """Verify a completed cache's hashes, shapes, and final publication index."""

    root = _absolute_lexical_path(cache_root)
    payload = _load_json_mapping(root / "cache_index.json")
    if int(payload.get("format_version", -1)) != CHUNK_CACHE_FORMAT_VERSION:
        raise ChunkCacheIntegrityError("unsupported paired temporal chunk cache format")
    routing_path = root / str(payload.get("routing_path", "routing.json"))
    if not routing_path.is_file() or file_sha256(routing_path) != str(payload.get("routing_sha256")):
        raise ChunkCacheIntegrityError("cached routing table is missing or corrupt")
    for entry in _required_sequence(payload, "indexes"):
        path = root / str(entry["relative_path"])
        if not path.is_file() or file_sha256(path) != str(entry["sha256"]):
            raise ChunkCacheIntegrityError(f"cached sample index is missing or corrupt: {path}")
    grids = {
        grid.grid_id: grid
        for grid in (ChunkGrid.from_dict(value) for value in _required_sequence(payload, "grids"))
    }
    verified = 0
    for value in _required_sequence(payload, "acquisitions"):
        acquisition = _acquisition_from_completed_dict(value)
        grid = grids.get(acquisition.grid_id)
        if grid is None or not _verify_chunk_directory(root, acquisition, grid):
            raise ChunkCacheIntegrityError(f"invalid chunk: {acquisition.acquisition_id}")
        verified += 1
    return {"cache_root": str(root), "verified_acquisitions": verified, "valid": True}


class PairedTemporalChunkDataset(Dataset[dict[str, object]]):
    """Read sample-by-window tensors from a completed local chunk cache only.

    ``window_mode=None`` is intentionally split-aware: training uses all 64
    fixed windows, while validation uses its center window.  This keeps model
    selection repeatable without silently changing the causal sample index.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        direction: Direction | None = None,
        split: str | None = None,
        index: str | Path | PairedTemporalIndex | Sequence[PairedTemporalSample] | None = None,
        window_mode: Literal["all", "center"] | None = None,
        minimum_valid_fraction: float = 0.80,
        max_observations: int | None = None,
        pad_observations_to: int | None = None,
        max_mmap_arrays: int = DEFAULT_MAX_MMAP_ARRAYS,
        registration_audit: bool = True,
        maximum_registration_shift_px: float = 0.5,
    ) -> None:
        if not 0.0 <= minimum_valid_fraction <= 1.0:
            raise ValueError("minimum_valid_fraction must be in [0, 1]")
        if maximum_registration_shift_px < 0.0:
            raise ValueError("maximum_registration_shift_px must be non-negative")
        if max_observations is not None and max_observations <= 0:
            raise ValueError("max_observations must be positive")
        if max_mmap_arrays <= 0:
            raise ValueError("max_mmap_arrays must be positive")
        if (
            max_observations is not None
            and pad_observations_to is not None
            and max_observations != pad_observations_to
        ):
            raise ValueError("max_observations and pad_observations_to disagree")
        self.cache_root = _absolute_lexical_path(cache_root)
        payload = _load_json_mapping(self.cache_root / "cache_index.json")
        if int(payload.get("format_version", -1)) != CHUNK_CACHE_FORMAT_VERSION:
            raise ChunkCacheIntegrityError("cache is incomplete or has an unsupported format")
        self.crop_size = int(payload["crop_size"])
        self.grids = {
            grid.grid_id: grid
            for grid in (ChunkGrid.from_dict(value) for value in _required_sequence(payload, "grids"))
        }
        self.acquisitions = {
            acquisition.acquisition_id: acquisition
            for acquisition in (_acquisition_from_completed_dict(value) for value in _required_sequence(payload, "acquisitions"))
        }
        routing_path = self.cache_root / str(payload["routing_path"])
        if not routing_path.is_file() or file_sha256(routing_path) != str(payload["routing_sha256"]):
            raise ChunkCacheIntegrityError("cached routing table is missing or corrupt")
        routing_payload = _load_json_mapping(routing_path)
        routes_value = _required_mapping(routing_payload, "routes")
        self.routes = {
            str(pair_id): ChunkRecordRoute.from_dict(route)  # type: ignore[arg-type]
            for pair_id, route in routes_value.items()
        }
        self.index = _load_cache_dataset_index(
            self.cache_root,
            payload,
            direction=direction,
            split=split,
            index=index,
        )
        self.samples = self.index.samples
        if not self.samples:
            raise ValueError("paired chunk dataset index is empty")
        self.direction = self.index.config.direction
        actual_split = {sample.split for sample in self.samples}
        if len(actual_split) != 1:
            raise ValueError("paired chunk dataset requires one split per instance")
        self.split = next(iter(actual_split))
        requested_padding = pad_observations_to if pad_observations_to is not None else max_observations
        observed_maximum = max(sample.observation_count for sample in self.samples)
        if requested_padding is not None and requested_padding < observed_maximum:
            raise ValueError("observation padding length is smaller than an indexed sequence")
        self.padded_observations = requested_padding
        selected_mode = window_mode or ("all" if self.split == str(payload["train_split"]) else "center")
        if selected_mode not in {"all", "center"}:
            raise ValueError("window_mode must be 'all' or 'center'")
        self.window_mode = selected_mode
        self.minimum_valid_fraction = float(minimum_valid_fraction)
        self.registration_audit = bool(registration_audit)
        self.maximum_registration_shift_px = float(maximum_registration_shift_px)
        self.max_mmap_arrays = int(max_mmap_arrays)
        self._mmap_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._sample_window_indexes: tuple[tuple[int, ...], ...] = tuple(
            self._window_indexes_for_sample(sample) for sample in self.samples
        )
        if len({len(values) for values in self._sample_window_indexes}) != 1:
            raise RuntimeError("chunk grids have incompatible selected window counts")
        self.windows_per_sample = len(self._sample_window_indexes[0])
        _assert_dataset_routing(self.samples, self.routes, self.grids, self.acquisitions)

    def __getstate__(self) -> dict[str, object]:
        # DataLoader spawn pickles the dataset.  Do not leave inherited file
        # descriptors open in the parent or serialize a memmap into a worker.
        self.close()
        state = dict(self.__dict__)
        state["_mmap_cache"] = OrderedDict()
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError, ValueError):
            # Destructors may run after interpreter/module teardown.
            return

    def close(self) -> None:
        """Close all local memmaps held by this worker's bounded LRU cache."""

        cache = getattr(self, "_mmap_cache", None)
        if cache is None:
            return
        while cache:
            _, array = cache.popitem(last=False)
            _close_memmap(array)

    def __len__(self) -> int:
        return len(self.samples) * self.windows_per_sample

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample_index, offset = divmod(index, self.windows_per_sample)
        sample = self.samples[sample_index]
        window_index = self._sample_window_indexes[sample_index][offset]
        return self._read_sample_window(sample, window_index)

    def _window_indexes_for_sample(self, sample: PairedTemporalSample) -> tuple[int, ...]:
        grid = self.grids[self.routes[sample.query_pair_id].grid_id]
        if self.window_mode == "center":
            return (grid.center_window_index,)
        return tuple(range(len(grid.windows)))

    def _read_sample_window(
        self, sample: PairedTemporalSample, window_index: int
    ) -> dict[str, object]:
        query_route = self.routes[sample.query_pair_id]
        anchor_route = self.routes[sample.anchor_pair_id]
        observation_routes = tuple(self.routes[pair_id] for pair_id in sample.observation_pair_ids)
        source_key = "sar_acquisition_id" if sample.source_modality == "sar" else "optical_acquisition_id"
        target_key = "sar_acquisition_id" if sample.target_modality == "sar" else "optical_acquisition_id"
        source_anchor_values, source_anchor_valid = self._read_acquisition_window(
            getattr(anchor_route, source_key), window_index
        )
        target_anchor_values, target_anchor_valid = self._read_acquisition_window(
            getattr(anchor_route, target_key), window_index
        )
        observation_arrays = [
            self._read_acquisition_window(getattr(route, source_key), window_index)
            for route in observation_routes
        ]
        observation_values = [value for value, _ in observation_arrays]
        observation_valid = [valid for _, valid in observation_arrays]
        target_values, target_valid = self._read_acquisition_window(
            getattr(query_route, target_key), window_index
        )
        base_valid = np.logical_and.reduce((source_anchor_valid, target_anchor_valid, target_valid))
        if self.window_mode == "center" and not bool(
            np.logical_and(target_valid, target_anchor_valid).any()
        ):
            raise ChunkCacheIntegrityError(
                "center paired temporal crop has no evaluable target/anchor pixels: "
                f"{sample.sample_id}"
            )
        query_date = date.fromisoformat(sample.query_date)
        observation_days = np.asarray(
            [(date.fromisoformat(value) - query_date).days for value in sample.observation_dates],
            dtype=np.float32,
        )
        source_anchor_days = np.float32(
            (date.fromisoformat(sample.source_anchor_date) - query_date).days
        )
        target_anchor_days = np.float32(
            (date.fromisoformat(sample.target_anchor_date) - query_date).days
        )
        if np.any(observation_days > 0) or source_anchor_days >= 0 or target_anchor_days >= 0:
            raise RuntimeError(f"causal dates changed while reading {sample.sample_id}")
        high_frequency_valid = np.zeros_like(target_valid, dtype=np.float32)
        registration_shift_px = float("inf")
        registration_zero_ncc = float("nan")
        registration_best_ncc = float("nan")
        registration_evidence_supported = False
        high_frequency_eligible = False
        high_frequency_weight = 0.0
        if self.registration_audit and sample.task_mode == "translation":
            query_source_values = observation_values[-1]
            query_source_valid = observation_valid[-1]
            high_frequency_valid = np.logical_and.reduce(
                (base_valid, query_source_valid.astype(bool))
            ).astype(np.float32)
            optical = target_values if sample.target_modality == "optical" else query_source_values
            sar = target_values if sample.target_modality == "sar" else query_source_values
            registration = registration_shift_audit(
                torch.from_numpy(optical.copy()),
                torch.from_numpy(sar.copy()),
                valid=torch.from_numpy(high_frequency_valid.copy()),
            )
            registration_shift_px = float(registration.shift_px)
            registration_zero_ncc = float(registration.zero_ncc)
            registration_best_ncc = float(registration.best_ncc)
            registration_evidence_supported = registration.evidence_supported
            high_frequency_eligible = (
                registration_evidence_supported
                and registration_shift_px <= self.maximum_registration_shift_px
                and float(high_frequency_valid.mean()) >= self.minimum_valid_fraction
            )
            if high_frequency_eligible:
                high_frequency_weight = 1.0 if int(observation_days[-1]) == 0 else 0.25
        return self._item_from_arrays(
            sample=sample,
            source_anchor_values=source_anchor_values,
            source_anchor_valid=source_anchor_valid,
            target_anchor_values=target_anchor_values,
            target_anchor_valid=target_anchor_valid,
            observation_values=observation_values,
            observation_valid=observation_valid,
            observation_days=observation_days,
            source_anchor_days=source_anchor_days,
            target_anchor_days=target_anchor_days,
            target_values=target_values,
            target_valid=target_valid,
            high_frequency_valid=high_frequency_valid,
            high_frequency_eligible=high_frequency_eligible,
            high_frequency_weight=high_frequency_weight,
            registration_shift_px=registration_shift_px,
            registration_zero_ncc=registration_zero_ncc,
            registration_best_ncc=registration_best_ncc,
            registration_evidence_supported=registration_evidence_supported,
        )

    def _read_acquisition_window(
        self, acquisition_id: str, window_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        acquisition = self.acquisitions.get(acquisition_id)
        if acquisition is None:
            raise ChunkCacheIntegrityError(f"route references unknown acquisition: {acquisition_id}")
        values = self._mmap(acquisition.values_relative_path)
        if not 0 <= window_index < values.shape[0]:
            raise ChunkCacheIntegrityError(f"window index is outside cached acquisition: {acquisition_id}")
        # Copy the value slice before mapping validity.  With a deliberately
        # small cache budget, opening validity may evict and close values.
        values_window = np.asarray(values[window_index], dtype=np.float32).copy()
        valid = self._mmap(acquisition.valid_relative_path)
        if not 0 <= window_index < valid.shape[0]:
            raise ChunkCacheIntegrityError(f"window index is outside cached acquisition: {acquisition_id}")
        valid_window = np.asarray(valid[window_index], dtype=np.float32).copy()
        # Indexing the memmap first is intentional: do not deserialize or copy
        # an entire 64-window acquisition for a single training example.
        return values_window, valid_window

    def _mmap(self, relative_path: Path) -> np.ndarray:
        path = self.cache_root / relative_path
        cached = self._mmap_cache.get(path)
        if cached is not None:
            self._mmap_cache.move_to_end(path)
            return cached
        if not path.is_file():
            raise ChunkCacheIntegrityError(f"missing local chunk file: {path}")
        try:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ChunkCacheIntegrityError(f"cannot memory-map local chunk file: {path}") from error
        if not isinstance(values, np.ndarray):
            raise ChunkCacheIntegrityError(f"chunk is not an ndarray: {path}")
        self._mmap_cache[path] = values
        while len(self._mmap_cache) > self.max_mmap_arrays:
            _, evicted = self._mmap_cache.popitem(last=False)
            _close_memmap(evicted)
        return values

    def _item_from_arrays(
        self,
        *,
        sample: PairedTemporalSample,
        source_anchor_values: np.ndarray,
        source_anchor_valid: np.ndarray,
        target_anchor_values: np.ndarray,
        target_anchor_valid: np.ndarray,
        observation_values: Sequence[np.ndarray],
        observation_valid: Sequence[np.ndarray],
        observation_days: np.ndarray,
        source_anchor_days: np.float32,
        target_anchor_days: np.float32,
        target_values: np.ndarray,
        target_valid: np.ndarray,
        high_frequency_valid: np.ndarray,
        high_frequency_eligible: bool,
        high_frequency_weight: float,
        registration_shift_px: float,
        registration_zero_ncc: float,
        registration_best_ncc: float,
        registration_evidence_supported: bool,
    ) -> dict[str, object]:
        source_channels = len(SAR_CHANNEL_ORDER) if sample.source_modality == "sar" else len(S2_CHANNEL_ORDER)
        sequence_length = self.padded_observations or len(observation_values)
        padded_values = np.zeros(
            (sequence_length, source_channels, self.crop_size, self.crop_size), dtype=np.float32
        )
        padded_valid = np.zeros((sequence_length, 1, self.crop_size, self.crop_size), dtype=np.float32)
        padded_days = np.zeros((sequence_length,), dtype=np.float32)
        present = np.zeros((sequence_length,), dtype=bool)
        count = len(observation_values)
        padded_values[:count] = np.stack(observation_values).astype(np.float32)
        padded_valid[:count] = np.stack(observation_valid).astype(np.float32)
        padded_days[:count] = observation_days
        present[:count] = True
        return {
            "source_anchor_values": torch.from_numpy(source_anchor_values.astype(np.float32)),
            "source_anchor_valid": torch.from_numpy(source_anchor_valid.astype(np.float32)),
            "target_anchor_values": torch.from_numpy(target_anchor_values.astype(np.float32)),
            "target_anchor_valid": torch.from_numpy(target_anchor_valid.astype(np.float32)),
            "observation_values": torch.from_numpy(padded_values),
            "observation_valid": torch.from_numpy(padded_valid),
            "observation_days": torch.from_numpy(padded_days),
            "observation_present": torch.from_numpy(present),
            "source_anchor_days": torch.tensor(source_anchor_days, dtype=torch.float32),
            "target_anchor_days": torch.tensor(target_anchor_days, dtype=torch.float32),
            "anchor_days": torch.tensor(target_anchor_days, dtype=torch.float32),
            "target_values": torch.from_numpy(target_values.astype(np.float32)),
            "target_valid": torch.from_numpy(target_valid.astype(np.float32)),
            "high_frequency_valid": torch.from_numpy(high_frequency_valid.astype(np.float32)),
            "high_frequency_eligible": torch.tensor(high_frequency_eligible),
            "high_frequency_weight": torch.tensor(high_frequency_weight, dtype=torch.float32),
            "registration_shift_px": torch.tensor(registration_shift_px, dtype=torch.float32),
            "registration_zero_ncc": torch.tensor(registration_zero_ncc, dtype=torch.float32),
            "registration_best_ncc": torch.tensor(registration_best_ncc, dtype=torch.float32),
            "registration_evidence_supported": torch.tensor(registration_evidence_supported),
            "sample_id": sample.sample_id,
            "direction": sample.direction,
            "task_mode": sample.task_mode,
        }


PairedTemporalChunkCacheDataset = PairedTemporalChunkDataset


def _close_memmap(array: np.ndarray) -> None:
    """Release the OS mapping behind one ``np.load(..., mmap_mode='r')`` array."""

    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def deterministic_chunk_windows(
    *,
    tile: str,
    width: int,
    height: int,
    crop_size: int = DEFAULT_WINDOW_SIZE,
    windows_per_acquisition: int = DEFAULT_WINDOWS_PER_ACQUISITION,
    grid_fingerprint: str = "",
) -> tuple[ChunkWindow, ...]:
    """Return center-first fixed windows shared by every acquisition on a grid.

    For a 2560x2560 Sentinel scene the center is exactly
    ``(1152, 1152, 256, 256)`` and the other 63 windows are selected from the
    remaining 99 members of the non-overlapping 10x10 lattice.  Their ranking
    derives from SHA-256 rather than Python's randomized ``hash()``.
    """

    if crop_size <= 0 or crop_size % 4:
        raise ValueError("crop_size must be positive and divisible by four")
    if width < crop_size or height < crop_size:
        raise ValueError("raster is smaller than crop_size")
    if windows_per_acquisition <= 0:
        raise ValueError("windows_per_acquisition must be positive")
    center = ChunkWindow(*centered_crop_window(width=width, height=height, crop_size=crop_size))
    lattice = [
        ChunkWindow(col=col, row=row, width=crop_size, height=crop_size)
        for row in range(0, height - crop_size + 1, crop_size)
        for col in range(0, width - crop_size + 1, crop_size)
    ]
    # The 10x10 Sentinel lattice has no exact central 256-pixel cell because
    # the centered crop begins at 1152.  Replace its closest lattice slot with
    # the exact center, leaving 99 mutually non-overlapping candidates.  The
    # centered crop may overlap those candidates by design; only the 63 drawn
    # from this remaining table are required to be mutually non-overlapping.
    replaced = min(
        lattice,
        key=lambda window: (
            abs(window.col - center.col) + abs(window.row - center.row),
            window.row,
            window.col,
        ),
    )
    remaining = [window for window in lattice if window != replaced]
    if len(remaining) < windows_per_acquisition - 1:
        raise ValueError(
            "grid has too few non-overlapping candidate windows for requested cache size"
        )
    token = f"{tile}|{width}|{height}|{crop_size}|{grid_fingerprint}".encode()
    ranked = sorted(
        remaining,
        key=lambda window: hashlib.sha256(
            token + f"|{window.col}|{window.row}".encode("ascii")
        ).digest(),
    )
    return (center, *ranked[: windows_per_acquisition - 1])


def _grid_for_record(
    record: PairRecord, *, crop_size: int, windows_per_acquisition: int
) -> ChunkGrid:
    token = _grid_token(record)
    grid_id = "grid-" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
    return ChunkGrid(
        grid_id=grid_id,
        tile=record.tile,
        width=int(record.width),
        height=int(record.height),
        crs=str(record.crs),
        transform=tuple(float(value) for value in record.transform[:6]),
        gsd=float(record.gsd),
        windows=deterministic_chunk_windows(
            tile=record.tile,
            width=int(record.width),
            height=int(record.height),
            crop_size=crop_size,
            windows_per_acquisition=windows_per_acquisition,
            grid_fingerprint=token,
        ),
    )


def _grid_key(record: PairRecord) -> tuple[object, ...]:
    return (
        record.tile,
        int(record.width),
        int(record.height),
        str(record.crs),
        tuple(float(value) for value in record.transform[:6]),
        float(record.gsd),
    )


def _grid_token(record: PairRecord) -> str:
    return json.dumps(
        {
            "tile": record.tile,
            "width": int(record.width),
            "height": int(record.height),
            "crs": str(record.crs),
            "transform": [float(value) for value in record.transform[:6]],
            "gsd": float(record.gsd),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _record_source_paths(record: PairRecord, modality: Modality, root: Path) -> tuple[str, ...]:
    values: Iterable[str]
    if modality == "optical":
        values = (*[record.s2[channel] for channel in S2_CHANNEL_ORDER], record.scl)
    else:
        values = tuple(record.sar[channel] for channel in SAR_CHANNEL_ORDER)
    return tuple(str(_absolute_lexical_path(value, base=root)) for value in values)


def _acquisition_id(modality: Modality, source_paths: tuple[str, ...]) -> str:
    encoded = json.dumps(
        {"modality": modality, "source_paths": list(source_paths)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{modality}-{hashlib.sha256(encoded).hexdigest()}"


def _assert_plan_causality(
    indexes: Sequence[ChunkCacheIndexEntry], routes: Mapping[str, ChunkRecordRoute]
) -> None:
    for entry in indexes:
        for sample in entry.index:
            query = routes[sample.query_pair_id]
            anchor = routes[sample.anchor_pair_id]
            source_attr = "sar_acquisition_id" if sample.source_modality == "sar" else "optical_acquisition_id"
            target_attr = "sar_acquisition_id" if sample.target_modality == "sar" else "optical_acquisition_id"
            query_target = getattr(query, target_attr)
            inputs = [getattr(anchor, source_attr), getattr(anchor, target_attr)]
            inputs.extend(getattr(routes[pair_id], source_attr) for pair_id in sample.observation_pair_ids)
            if query_target in inputs:
                raise RuntimeError(f"cached routing leaks a query target into inputs: {sample.sample_id}")


def _write_static_cache_metadata(plan: PairedTemporalChunkCachePlan) -> None:
    root = plan.destination_root
    for entry in plan.indexes:
        write_paired_temporal_index(root / entry.relative_path, entry.index)
    routing_payload = {
        "format_version": CHUNK_CACHE_FORMAT_VERSION,
        "routes": {pair_id: route.to_dict() for pair_id, route in sorted(plan.routes.items())},
    }
    _atomic_json(root / "routing.json", routing_payload)
    _atomic_json(root / "plan.json", _plan_payload(plan))
    _atomic_json(
        root / "provenance.json",
        {
            "format_version": CHUNK_CACHE_FORMAT_VERSION,
            "config_path": str(plan.config_path),
            "config_sha256": plan.config_sha256,
            "source_manifest": str(plan.source_manifest),
            "source_manifest_sha256": plan.source_manifest_sha256,
            "normalization": "paired_temporal_v3_normalized",
            "optical_channels": list(S2_CHANNEL_ORDER),
            "sar_channels": list(SAR_CHANNEL_ORDER),
            "crop_size": plan.crop_size,
            "windows_per_acquisition": plan.windows_per_acquisition,
            "splits": [plan.train_split, plan.validation_split],
        },
    )


def _plan_payload(plan: PairedTemporalChunkCachePlan) -> dict[str, object]:
    return {
        "format_version": CHUNK_CACHE_FORMAT_VERSION,
        "config_path": str(plan.config_path),
        "config_sha256": plan.config_sha256,
        "source_manifest": str(plan.source_manifest),
        "source_manifest_sha256": plan.source_manifest_sha256,
        "destination_root": str(plan.destination_root),
        "train_split": plan.train_split,
        "validation_split": plan.validation_split,
        "crop_size": plan.crop_size,
        "windows_per_acquisition": plan.windows_per_acquisition,
        "grids": [grid.to_dict() for grid in plan.grids],
        "acquisitions": [acquisition.to_dict() for acquisition in plan.acquisitions],
        "indexes": [
            {
                "direction": entry.direction,
                "split": entry.split,
                "relative_path": entry.relative_path.as_posix(),
                "samples": len(entry.index),
            }
            for entry in plan.indexes
        ],
        "routing_path": "routing.json",
        "report": plan.report(),
    }


def _materialize_acquisition(
    root: Path, acquisition: ChunkAcquisition, grid: ChunkGrid
) -> dict[str, object]:
    """Read every source TIFF once and atomically publish one acquisition dir."""

    final_dir = root / acquisition.relative_directory
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = final_dir.parent / f".{acquisition.acquisition_id}.{uuid.uuid4().hex}.tmp"
    staging_dir.mkdir(parents=False, exist_ok=False)
    try:
        values_path = staging_dir / "values.npy"
        valid_path = staging_dir / "valid.npy"
        _write_acquisition_arrays(values_path, valid_path, acquisition, grid)
        metadata = _chunk_metadata_for_paths(
            acquisition,
            grid,
            values_path=values_path,
            valid_path=valid_path,
        )
        _atomic_json(staging_dir / "chunk.json", metadata)
        backup_dir: Path | None = None
        if final_dir.exists():
            backup_dir = final_dir.parent / f".{acquisition.acquisition_id}.{uuid.uuid4().hex}.old"
            os.replace(final_dir, backup_dir)
        try:
            os.replace(staging_dir, final_dir)
        except BaseException:
            if backup_dir is not None and backup_dir.exists() and not final_dir.exists():
                os.replace(backup_dir, final_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
        return metadata
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _write_acquisition_arrays(
    values_path: Path,
    valid_path: Path,
    acquisition: ChunkAcquisition,
    grid: ChunkGrid,
) -> None:
    """Open each raw band once and fill all cached windows in one pass."""

    import rasterio
    from rasterio.windows import Window

    value_shape = (len(grid.windows), acquisition.channels, grid.center_window.height, grid.center_window.width)
    valid_shape = (len(grid.windows), 1, grid.center_window.height, grid.center_window.width)
    values_out = np.lib.format.open_memmap(values_path, mode="w+", dtype=np.float16, shape=value_shape)
    valid_out = np.lib.format.open_memmap(valid_path, mode="w+", dtype=np.uint8, shape=valid_shape)
    try:
        sources = [rasterio.open(path) for path in acquisition.source_paths]
        try:
            for path, source in zip(acquisition.source_paths, sources, strict=True):
                _assert_source_grid(source, grid, Path(path))
            for window_index, window in enumerate(grid.windows):
                raster_window = Window(window.col, window.row, window.width, window.height)
                raw = np.stack([source.read(1, window=raster_window) for source in sources])
                expected = (len(sources), window.height, window.width)
                if raw.shape != expected:
                    raise RuntimeError(f"source returned wrong window shape: {raw.shape} != {expected}")
                if acquisition.modality == "optical":
                    optical_raw = raw[: len(S2_CHANNEL_ORDER)]
                    scl = raw[len(S2_CHANNEL_ORDER)]
                    valid = np.isin(scl, CLEAR_SCL_CODES) & np.all(optical_raw > 0, axis=0)
                    normalized = np.clip(optical_raw.astype(np.float32) / 10000.0, 0.0, 1.0) * 2.0 - 1.0
                else:
                    valid = np.all(raw > 0, axis=0)
                    values_db = raw.astype(np.float32) / 200.0 - 50.0
                    minimum = np.asarray((-35.0, -45.0), dtype=np.float32)[:, None, None]
                    maximum = np.asarray((5.0, -5.0), dtype=np.float32)[:, None, None]
                    normalized = np.clip(
                        2.0 * (values_db - minimum) / (maximum - minimum) - 1.0,
                        -1.0,
                        1.0,
                    )
                normalized[:, ~valid] = 0.0
                values_out[window_index] = normalized.astype(np.float16)
                valid_out[window_index, 0] = valid.astype(np.uint8)
        finally:
            for source in sources:
                source.close()
        values_out.flush()
        valid_out.flush()
    finally:
        del values_out
        del valid_out


def _assert_source_grid(source: Any, grid: ChunkGrid, path: Path) -> None:
    if source.crs is None:
        raise RuntimeError(f"raster has no CRS: {path}")
    transform = tuple(float(value) for value in source.transform[:6])
    if (
        int(source.width) != grid.width
        or int(source.height) != grid.height
        or source.crs.to_string() != grid.crs
        or len(transform) != len(grid.transform)
        or any(not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9) for left, right in zip(transform, grid.transform))
    ):
        raise RuntimeError(f"raster grid does not match paired temporal manifest: {path}")


def _chunk_metadata_for_paths(
    acquisition: ChunkAcquisition,
    grid: ChunkGrid,
    *,
    values_path: Path,
    valid_path: Path,
) -> dict[str, object]:
    return {
        "format_version": CHUNK_CACHE_FORMAT_VERSION,
        "acquisition_id": acquisition.acquisition_id,
        "modality": acquisition.modality,
        "grid_id": acquisition.grid_id,
        "relative_directory": acquisition.relative_directory.as_posix(),
        "values": _array_file_descriptor(values_path, np.float16, (len(grid.windows), acquisition.channels, grid.center_window.height, grid.center_window.width)),
        "valid": _array_file_descriptor(valid_path, np.uint8, (len(grid.windows), 1, grid.center_window.height, grid.center_window.width)),
    }


def _estimated_acquisition_bytes(acquisition: ChunkAcquisition, grid: ChunkGrid) -> int:
    """Return the final array bytes for one acquisition, excluding headers."""

    height = grid.center_window.height
    width = grid.center_window.width
    values = acquisition.window_count * acquisition.channels * height * width * np.dtype(np.float16).itemsize
    valid = acquisition.window_count * height * width * np.dtype(np.uint8).itemsize
    return values + valid


def _npy_header_bytes(dtype: np.dtype[Any] | type[np.generic], shape: tuple[int, ...]) -> int:
    """Return the exact ``.npy`` header size written by ``open_memmap``."""

    buffer = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        buffer,
        {
            "descr": np.dtype(dtype).str,
            "fortran_order": False,
            "shape": shape,
        },
    )
    return buffer.tell()


def _estimated_acquisition_materialization_bytes(
    acquisition: ChunkAcquisition, grid: ChunkGrid
) -> int:
    """Estimate a pending acquisition's arrays, headers, and sidecar exactly."""

    height = grid.center_window.height
    width = grid.center_window.width
    values_shape = (acquisition.window_count, acquisition.channels, height, width)
    valid_shape = (acquisition.window_count, 1, height, width)
    return (
        _estimated_acquisition_bytes(acquisition, grid)
        + _npy_header_bytes(np.float16, values_shape)
        + _npy_header_bytes(np.uint8, valid_shape)
        + _PENDING_CHUNK_METADATA_RESERVE_BYTES
    )


def _remaining_materialization_bytes(
    pending: Sequence[ChunkAcquisition], grids: Mapping[str, ChunkGrid]
) -> int:
    """Return all bytes a resumed invocation must still be able to write."""

    return _STATIC_METADATA_RESERVE_BYTES + sum(
        _estimated_acquisition_materialization_bytes(acquisition, grids[acquisition.grid_id])
        for acquisition in pending
    )


def _array_file_descriptor(path: Path, dtype: np.dtype[Any] | type[np.generic], shape: tuple[int, ...]) -> dict[str, object]:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "dtype": np.dtype(dtype).str,
        "shape": list(shape),
    }


def _verify_chunk_directory(root: Path, acquisition: ChunkAcquisition, grid: ChunkGrid) -> bool:
    metadata_path = root / acquisition.metadata_relative_path
    try:
        metadata = _load_chunk_metadata(metadata_path)
        if str(metadata.get("acquisition_id")) != acquisition.acquisition_id:
            return False
        if str(metadata.get("modality")) != acquisition.modality or str(metadata.get("grid_id")) != grid.grid_id:
            return False
        for name, dtype, shape in (
            (
                "values",
                np.float16,
                (len(grid.windows), acquisition.channels, grid.center_window.height, grid.center_window.width),
            ),
            ("valid", np.uint8, (len(grid.windows), 1, grid.center_window.height, grid.center_window.width)),
        ):
            descriptor = _required_mapping(metadata, name)
            path = metadata_path.parent / str(descriptor["path"])
            if not path.is_file() or path.stat().st_size != int(descriptor["size_bytes"]):
                return False
            if file_sha256(path) != str(descriptor["sha256"]):
                return False
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype) or array.shape != shape:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, ChunkCacheIntegrityError):
        return False


def _completed_cache_index_payload(
    plan: PairedTemporalChunkCachePlan, chunk_entries: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    indexes = []
    for entry in plan.indexes:
        path = plan.destination_root / entry.relative_path
        indexes.append(
            {
                "direction": entry.direction,
                "split": entry.split,
                "relative_path": entry.relative_path.as_posix(),
                "samples": len(entry.index),
                "sha256": file_sha256(path),
            }
        )
    return {
        "format_version": CHUNK_CACHE_FORMAT_VERSION,
        "plan_sha256": _payload_sha256(_plan_payload(plan)),
        "config_sha256": plan.config_sha256,
        "source_manifest_sha256": plan.source_manifest_sha256,
        "crop_size": plan.crop_size,
        "windows_per_acquisition": plan.windows_per_acquisition,
        "train_split": plan.train_split,
        "validation_split": plan.validation_split,
        "grids": [grid.to_dict() for grid in plan.grids],
        "routing_path": "routing.json",
        "routing_sha256": file_sha256(plan.destination_root / "routing.json"),
        "indexes": indexes,
        "acquisitions": list(chunk_entries),
    }


def _load_cache_dataset_index(
    root: Path,
    cache_index: Mapping[str, object],
    *,
    direction: Direction | None,
    split: str | None,
    index: str | Path | PairedTemporalIndex | Sequence[PairedTemporalSample] | None,
) -> PairedTemporalIndex:
    if index is not None:
        if isinstance(index, PairedTemporalIndex):
            selected = index
        elif isinstance(index, (str, Path)):
            selected = load_paired_temporal_index(index)
        else:
            samples = tuple(index)
            if not samples:
                raise ValueError("custom paired chunk index cannot be empty")
            directions = {sample.direction for sample in samples}
            if len(directions) != 1:
                raise ValueError("custom paired chunk index needs one direction")
            selected = PairedTemporalIndex(
                config=PairedTemporalIndexConfig(direction=next(iter(directions))),
                samples=samples,
            )
        if direction is not None and selected.config.direction != direction:
            raise ValueError("custom index direction disagrees with direction")
        if split is not None and any(sample.split != split for sample in selected):
            raise ValueError("custom index split disagrees with split")
        return selected
    matching = []
    for value in _required_sequence(cache_index, "indexes"):
        candidate_direction = str(value["direction"])
        candidate_split = str(value["split"])
        if direction is not None and candidate_direction != direction:
            continue
        if split is not None and candidate_split != split:
            continue
        matching.append(value)
    if len(matching) != 1:
        raise ValueError("choose exactly one cached direction/split index")
    selected_value = matching[0]
    path = root / str(selected_value["relative_path"])
    if not path.is_file() or file_sha256(path) != str(selected_value["sha256"]):
        raise ChunkCacheIntegrityError(f"cached sample index is missing or corrupt: {path}")
    return load_paired_temporal_index(path)


def _assert_dataset_routing(
    samples: Sequence[PairedTemporalSample],
    routes: Mapping[str, ChunkRecordRoute],
    grids: Mapping[str, ChunkGrid],
    acquisitions: Mapping[str, ChunkAcquisition],
) -> None:
    for sample in samples:
        required = (sample.query_pair_id, sample.anchor_pair_id, *sample.observation_pair_ids)
        if any(pair_id not in routes for pair_id in required):
            raise ChunkCacheIntegrityError(f"cached routing is missing a pair used by {sample.sample_id}")
        grid_ids = {routes[pair_id].grid_id for pair_id in required}
        if len(grid_ids) != 1 or next(iter(grid_ids)) not in grids:
            raise ChunkCacheIntegrityError(f"sample does not have one cached spatial grid: {sample.sample_id}")
        query = routes[sample.query_pair_id]
        anchor = routes[sample.anchor_pair_id]
        source_attr = "sar_acquisition_id" if sample.source_modality == "sar" else "optical_acquisition_id"
        target_attr = "sar_acquisition_id" if sample.target_modality == "sar" else "optical_acquisition_id"
        used = [getattr(anchor, source_attr), getattr(anchor, target_attr)]
        used.extend(getattr(routes[pair_id], source_attr) for pair_id in sample.observation_pair_ids)
        if getattr(query, target_attr) in used:
            raise ChunkCacheIntegrityError(f"cached target leakage detected: {sample.sample_id}")
        route_ids = [getattr(routes[pair_id], attribute) for pair_id in required for attribute in ("optical_acquisition_id", "sar_acquisition_id")]
        missing = [acquisition_id for acquisition_id in route_ids if acquisition_id not in acquisitions]
        if missing:
            raise ChunkCacheIntegrityError(f"cached acquisition metadata is missing: {missing[0]}")


def _acquisition_from_completed_dict(values: Mapping[str, object]) -> ChunkAcquisition:
    directory = Path(str(values["relative_directory"]))
    values_descriptor = _required_mapping(values, "values")
    shape = tuple(int(value) for value in values_descriptor["shape"])  # type: ignore[index,arg-type]
    if len(shape) != 4:
        raise ChunkCacheIntegrityError("cached values shape must be WCHW")
    return ChunkAcquisition(
        acquisition_id=str(values["acquisition_id"]),
        modality=str(values["modality"]),  # type: ignore[arg-type]
        grid_id=str(values["grid_id"]),
        source_paths=(),
        relative_directory=directory,
        channels=shape[1],
        window_count=shape[0],
    )


def _index_config_from_data(data: Mapping[str, object]) -> dict[str, object]:
    task_modes = tuple(str(value) for value in data.get("task_modes", ALL_TASK_MODES))
    if not task_modes or any(value not in ALL_TASK_MODES for value in task_modes):
        raise ValueError("data.task_modes is invalid")
    return {
        "min_observations": _required_int(data, "minimum_observations"),
        "max_observations": _required_int(data, "maximum_observations"),
        "horizon_days": _required_int(data, "horizon_days"),
        "anchor_max_delta_days": _required_int(data, "anchor_pair_max_delta_days"),
        "max_anchors_per_query": _required_int(data, "maximum_anchors_per_query"),
        "translation_max_delta_days": _required_int(data, "translation_max_delta_days"),
        "orbit": _required_string(data, "orbit"),
        "task_modes": task_modes,
    }


def _load_yaml_mapping(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, Mapping):
        raise TypeError(f"expected YAML mapping: {path}")
    return values


def _required_mapping(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping at {key}")
    return value


def _required_sequence(values: Mapping[str, object], key: str) -> Sequence[Mapping[str, object]]:
    value = values.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ChunkCacheIntegrityError(f"expected sequence at {key}")
    if not all(isinstance(entry, Mapping) for entry in value):
        raise ChunkCacheIntegrityError(f"expected mapping entries at {key}")
    return value  # type: ignore[return-value]


def _required_string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected non-empty string at {key}")
    return value


def _required_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool):
        raise TypeError(f"expected integer at {key}")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected integer at {key}") from error


def _absolute_lexical_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = os.path.expanduser(os.fspath(value))
    if not os.path.isabs(path):
        path = os.path.join(os.fspath(base if base is not None else Path.cwd()), path)
    return Path(os.path.abspath(os.path.normpath(path)))


def _existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise RuntimeError(f"no existing parent for {path}")
        current = parent
    return current


def _atomic_json(path: Path, values: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ChunkCacheIntegrityError(f"cannot read cache metadata: {path}") from error
    if not isinstance(values, Mapping):
        raise ChunkCacheIntegrityError(f"cache metadata is not a mapping: {path}")
    return values


def _load_chunk_metadata(path: Path) -> Mapping[str, object]:
    return _load_json_mapping(path)


def _payload_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
