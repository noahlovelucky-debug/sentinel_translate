"""Read-only SOPAT V4 adapter for completed V3 acquisition chunk caches.

The V3 cache may contain target acquisitions because it is deduplicated by
physical acquisition.  This adapter never turns that fact into an input route:
it validates a V4 index against the published V3 sample indexes and passes the
resulting direction-homogeneous index to ``PairedTemporalChunkDataset``.
There is intentionally no raw TIFF or NFS fallback path in this module.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sentinel_v3.dataset_builder import file_sha256
from sentinel_v3.paired_temporal_chunk_cache import (
    CHUNK_CACHE_FORMAT_VERSION,
    ChunkCacheIntegrityError,
    PairedTemporalChunkDataset,
    verify_paired_temporal_chunk_cache,
)
from sentinel_v3.paired_temporal_data import (
    ALL_DIRECTIONS,
    Direction,
    PairedTemporalIndex,
    load_paired_temporal_index,
)
from sentinel_v3.schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER

from .data import (
    SOPAT_CANONICAL_GSD_METERS,
    SOPAT_CANONICALIZATION_VERSION,
    SOPAT_NORMALIZATION_VERSION,
    SOPATDirectionDataset,
    SOPATExampleV4,
    SOPATIndexV4,
    assert_sopat_v4_causality,
    load_sopat_v4_index,
    paired_temporal_index_from_sopat_v4,
    paired_temporal_protocol_hash,
    sensor_schema_hash,
)


class SOPATChunkCachePreflightError(RuntimeError):
    """Raised when a V3 cache cannot safely serve an SOPAT V4 index."""


@dataclass(frozen=True)
class SOPATChunkCachePreflight:
    """Immutable evidence that one V4 index matches one completed local cache."""

    cache_root: Path
    cache_index_sha256: str
    cache_plan_sha256: str
    index_content_sha256: str
    index_file_sha256: str | None
    source_manifest_sha256: str
    crop_size: int
    windows_per_acquisition: int
    directions: tuple[Direction, ...]
    splits: tuple[str, ...]
    examples: int
    verified_chunks: bool

    def to_dict(self) -> dict[str, object]:
        """Return checkpoint-safe primitive metadata without mutable paths."""

        return {
            "cache_root": str(self.cache_root),
            "cache_index_sha256": self.cache_index_sha256,
            "cache_plan_sha256": self.cache_plan_sha256,
            "index_content_sha256": self.index_content_sha256,
            "index_file_sha256": self.index_file_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "crop_size": self.crop_size,
            "windows_per_acquisition": self.windows_per_acquisition,
            "directions": list(self.directions),
            "splits": list(self.splits),
            "examples": self.examples,
            "verified_chunks": self.verified_chunks,
        }


def preflight_sopat_v4_chunk_cache(
    cache_root: str | Path,
    index: SOPATIndexV4 | str | Path,
    *,
    verify_chunks: bool = False,
) -> SOPATChunkCachePreflight:
    """Validate that a published V3 cache is exact enough for a V4 index.

    ``verify_chunks=False`` checks the cheap, immutable publication metadata
    and V3 index hashes.  Set it for a slower full SHA/shape sweep before a
    long run.  Runtime data loading still only uses local ``.npy`` chunks.
    """

    resolved_index, index_file_sha256 = _coerce_index(index)
    assert_sopat_v4_causality(resolved_index)
    root = _absolute_path(cache_root)
    cache_index_path = root / "cache_index.json"
    payload = _load_json_mapping(cache_index_path)
    if int(payload.get("format_version", -1)) != CHUNK_CACHE_FORMAT_VERSION:
        raise SOPATChunkCachePreflightError("chunk cache is incomplete or has an unsupported format")
    _validate_cache_publication(root, payload)
    _validate_cache_plan(root, payload)
    provenance = _load_json_mapping(root / "provenance.json")
    _validate_cache_provenance(payload, provenance)
    grids = _cache_grids(payload)
    routes = _cache_routes(root, payload)
    acquisitions = _cache_acquisition_ids(payload)
    cached_indexes = _cache_indexes(root, payload)
    _validate_examples_against_cache(
        resolved_index,
        payload=payload,
        grids=grids,
        routes=routes,
        acquisitions=acquisitions,
        cached_indexes=cached_indexes,
    )
    if verify_chunks:
        try:
            verify_paired_temporal_chunk_cache(root)
        except ChunkCacheIntegrityError as error:
            raise SOPATChunkCachePreflightError(str(error)) from error
    directions = tuple(direction for direction in ALL_DIRECTIONS if any(
        example.direction == direction for example in resolved_index.examples
    ))
    splits = tuple(sorted({example.split for example in resolved_index.examples}))
    return SOPATChunkCachePreflight(
        cache_root=root,
        cache_index_sha256=file_sha256(cache_index_path),
        cache_plan_sha256=_required_string(payload, "plan_sha256"),
        index_content_sha256=resolved_index.content_hash,
        index_file_sha256=index_file_sha256,
        source_manifest_sha256=_optional_string(payload, "source_manifest_sha256"),
        crop_size=_required_int(payload, "crop_size"),
        windows_per_acquisition=_required_int(payload, "windows_per_acquisition"),
        directions=directions,
        splits=splits,
        examples=len(resolved_index),
        verified_chunks=bool(verify_chunks),
    )


def sopat_chunk_dataset_from_cache(
    cache_root: str | Path,
    index: SOPATIndexV4 | str | Path,
    *,
    direction: Direction,
    split: str,
    window_mode: Literal["all", "center"] | None = None,
    permutation_seed: int = 0,
    permute_observations: bool = True,
    verify_chunks: bool = False,
    **chunk_dataset_kwargs: object,
) -> SOPATDirectionDataset:
    """Construct a local-only, direction-homogeneous SOPAT chunk dataset.

    The call fails before opening a tensor when its V4 route differs from the
    completed cache.  ``PairedTemporalChunkDataset`` maps only local NumPy
    chunks; this wrapper deliberately exposes no raw-source argument.
    """

    resolved_index, _ = _coerce_index(index)
    preflight_sopat_v4_chunk_cache(
        cache_root,
        resolved_index,
        verify_chunks=verify_chunks,
    )
    legacy_index = paired_temporal_index_from_sopat_v4(
        resolved_index,
        direction=direction,
        split=split,
    )
    backend = PairedTemporalChunkDataset(
        cache_root,
        direction=direction,
        split=split,
        index=legacy_index,
        window_mode=window_mode,
        **chunk_dataset_kwargs,
    )
    return SOPATDirectionDataset(
        resolved_index,
        backend,
        direction=direction,
        split=split,
        permutation_seed=permutation_seed,
        permute_observations=permute_observations,
    )


def _coerce_index(index: SOPATIndexV4 | str | Path) -> tuple[SOPATIndexV4, str | None]:
    if isinstance(index, SOPATIndexV4):
        return index, None
    path = Path(index)
    return load_sopat_v4_index(path), file_sha256(path)


def _validate_cache_publication(root: Path, payload: Mapping[str, object]) -> None:
    routing_path = root / _required_string(payload, "routing_path")
    if not routing_path.is_file() or file_sha256(routing_path) != _required_string(
        payload, "routing_sha256"
    ):
        raise SOPATChunkCachePreflightError("chunk cache routing table is missing or corrupt")
    for entry in _required_mapping_sequence(payload, "indexes"):
        path = root / _required_string(entry, "relative_path")
        if not path.is_file() or file_sha256(path) != _required_string(entry, "sha256"):
            raise SOPATChunkCachePreflightError(f"chunk cache index is missing or corrupt: {path}")
    if not (root / "provenance.json").is_file():
        raise SOPATChunkCachePreflightError("chunk cache provenance.json is missing")


def _validate_cache_plan(root: Path, payload: Mapping[str, object]) -> None:
    """Bind the final publication marker to its immutable V3 dry-run plan."""

    plan_path = root / "plan.json"
    plan = _load_json_mapping(plan_path)
    if int(plan.get("format_version", -1)) != CHUNK_CACHE_FORMAT_VERSION:
        raise SOPATChunkCachePreflightError("chunk cache plan has an unsupported format")
    expected = _required_string(payload, "plan_sha256")
    actual = _canonical_json_sha256(plan)
    if actual != expected:
        raise SOPATChunkCachePreflightError("chunk cache plan digest differs from publication")
    for key in (
        "config_sha256",
        "source_manifest_sha256",
        "crop_size",
        "windows_per_acquisition",
        "train_split",
        "validation_split",
    ):
        if plan.get(key) != payload.get(key):
            raise SOPATChunkCachePreflightError(
                f"chunk cache plan field differs from publication: {key}"
            )


def _validate_cache_provenance(
    payload: Mapping[str, object], provenance: Mapping[str, object]
) -> None:
    if int(provenance.get("format_version", -1)) != CHUNK_CACHE_FORMAT_VERSION:
        raise SOPATChunkCachePreflightError("chunk cache provenance has an unsupported format")
    if _required_string(provenance, "normalization") != SOPAT_NORMALIZATION_VERSION:
        raise SOPATChunkCachePreflightError("chunk cache normalization does not match SOPAT V4")
    if tuple(_required_string_sequence(provenance, "optical_channels")) != tuple(S2_CHANNEL_ORDER):
        raise SOPATChunkCachePreflightError("chunk cache optical channel schema does not match SOPAT V4")
    if tuple(_required_string_sequence(provenance, "sar_channels")) != tuple(SAR_CHANNEL_ORDER):
        raise SOPATChunkCachePreflightError("chunk cache SAR channel schema does not match SOPAT V4")
    if _required_int(provenance, "crop_size") != _required_int(payload, "crop_size"):
        raise SOPATChunkCachePreflightError("chunk cache crop size differs from its provenance")
    if _required_int(provenance, "windows_per_acquisition") != _required_int(
        payload, "windows_per_acquisition"
    ):
        raise SOPATChunkCachePreflightError("chunk cache window count differs from its provenance")
    if _optional_string(provenance, "source_manifest_sha256") != _optional_string(
        payload, "source_manifest_sha256"
    ):
        raise SOPATChunkCachePreflightError("chunk cache manifest digest differs from its provenance")


def _cache_grids(payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    grids: dict[str, Mapping[str, object]] = {}
    for grid in _required_mapping_sequence(payload, "grids"):
        grid_id = _required_string(grid, "grid_id")
        gsd = float(grid.get("gsd", float("nan")))
        if not math.isfinite(gsd) or abs(gsd - SOPAT_CANONICAL_GSD_METERS) > 1e-9:
            raise SOPATChunkCachePreflightError(
                f"chunk cache grid {grid_id} is not canonical 10m"
            )
        if grid_id in grids:
            raise SOPATChunkCachePreflightError(f"chunk cache repeats grid {grid_id}")
        grids[grid_id] = grid
    if not grids:
        raise SOPATChunkCachePreflightError("chunk cache has no canonical grids")
    return grids


def _cache_routes(root: Path, payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    routing_path = root / _required_string(payload, "routing_path")
    routing = _load_json_mapping(routing_path)
    routes_value = routing.get("routes")
    if not isinstance(routes_value, Mapping):
        raise SOPATChunkCachePreflightError("chunk cache routing table has no routes mapping")
    routes: dict[str, Mapping[str, object]] = {}
    for pair_id, route in routes_value.items():
        if not isinstance(pair_id, str) or not pair_id or not isinstance(route, Mapping):
            raise SOPATChunkCachePreflightError("chunk cache routing table has an invalid route")
        routes[pair_id] = route
    return routes


def _cache_acquisition_ids(payload: Mapping[str, object]) -> set[str]:
    ids: set[str] = set()
    for acquisition in _required_mapping_sequence(payload, "acquisitions"):
        acquisition_id = _required_string(acquisition, "acquisition_id")
        if acquisition_id in ids:
            raise SOPATChunkCachePreflightError(
                f"chunk cache repeats acquisition {acquisition_id}"
            )
        ids.add(acquisition_id)
    if not ids:
        raise SOPATChunkCachePreflightError("chunk cache has no completed acquisitions")
    return ids


def _cache_indexes(
    root: Path, payload: Mapping[str, object]
) -> dict[tuple[Direction, str], PairedTemporalIndex]:
    indexes: dict[tuple[Direction, str], PairedTemporalIndex] = {}
    for entry in _required_mapping_sequence(payload, "indexes"):
        direction = _coerce_direction(_required_string(entry, "direction"))
        split = _required_string(entry, "split")
        key = direction, split
        if key in indexes:
            raise SOPATChunkCachePreflightError(
                f"chunk cache repeats index {direction}/{split}"
            )
        try:
            loaded = load_paired_temporal_index(root / _required_string(entry, "relative_path"))
        except (OSError, TypeError, ValueError) as error:
            raise SOPATChunkCachePreflightError(
                f"cannot load chunk cache index {direction}/{split}"
            ) from error
        if loaded.config.direction != direction:
            raise SOPATChunkCachePreflightError(
                f"chunk cache index direction differs from publication entry: {direction}/{split}"
            )
        if loaded.config.split != split:
            raise SOPATChunkCachePreflightError(
                f"chunk cache index split differs from publication entry: {direction}/{split}"
            )
        indexes[key] = loaded
    return indexes


def _validate_examples_against_cache(
    index: SOPATIndexV4,
    *,
    payload: Mapping[str, object],
    grids: Mapping[str, Mapping[str, object]],
    routes: Mapping[str, Mapping[str, object]],
    acquisitions: set[str],
    cached_indexes: Mapping[tuple[Direction, str], PairedTemporalIndex],
) -> None:
    manifest_sha256 = _optional_string(payload, "source_manifest_sha256")
    for example in index.examples:
        key = example.direction, example.split
        cached = cached_indexes.get(key)
        if cached is None:
            raise SOPATChunkCachePreflightError(
                f"cache has no index for SOPAT example {example.direction}/{example.split}"
            )
        if example.provenance.sensor_schema_hash != sensor_schema_hash():
            raise SOPATChunkCachePreflightError("SOPAT index sensor schema does not match active schema")
        if example.provenance.normalization_version != SOPAT_NORMALIZATION_VERSION:
            raise SOPATChunkCachePreflightError("SOPAT index normalization does not match cache")
        if example.provenance.canonicalization_version != SOPAT_CANONICALIZATION_VERSION:
            raise SOPATChunkCachePreflightError("SOPAT index canonicalization does not match cache")
        if example.provenance.source_manifest_sha256 and (
            example.provenance.source_manifest_sha256 != manifest_sha256
        ):
            raise SOPATChunkCachePreflightError("SOPAT index manifest digest differs from cache")
        if example.provenance.source_protocol_hash != paired_temporal_protocol_hash(cached.config):
            raise SOPATChunkCachePreflightError(
                f"SOPAT protocol differs from cached index: {example.direction}/{example.split}"
            )
        if example.canonical_grid.grid_id not in grids:
            raise SOPATChunkCachePreflightError(
                f"SOPAT example grid is absent from chunk cache: {example.sample_id}"
            )
        _validate_example_grid(example, grids[example.canonical_grid.grid_id])
        projected = paired_temporal_index_from_sopat_v4(
            SOPATIndexV4(config=index.config, examples=(example,)),
            direction=example.direction,
            split=example.split,
        ).samples[0]
        cached_by_id = {sample.sample_id: sample for sample in cached.samples}
        if cached_by_id.get(example.sample_id) != projected:
            raise SOPATChunkCachePreflightError(
                f"SOPAT example differs from the cached causal index: {example.sample_id}"
            )
        _validate_example_routes(example, routes, acquisitions)


def _validate_example_routes(
    example: SOPATExampleV4,
    routes: Mapping[str, Mapping[str, object]],
    acquisitions: set[str],
) -> None:
    # Keep this in the cache module so a V4 index always goes through its own
    # explicit target-leakage check before V3's local-only reader is created.
    target = example.target
    anchor = example.anchor_pair
    observation_records = tuple(observation.measurement.record_id for observation in example.observations)
    required = (target.record_id, anchor.registration_id, *observation_records)
    route_values: list[Mapping[str, object]] = []
    for pair_id in required:
        route = routes.get(pair_id)
        if route is None:
            raise SOPATChunkCachePreflightError(
                f"chunk cache route is missing SOPAT pair {pair_id}"
            )
        if _required_string(route, "grid_id") != example.canonical_grid.grid_id:
            raise SOPATChunkCachePreflightError(
                f"chunk cache route has a different grid for {example.sample_id}"
            )
        if _required_string(route, "split") != example.split:
            raise SOPATChunkCachePreflightError(
                f"chunk cache route crosses split isolation for {example.sample_id}"
            )
        route_values.append(route)
    source_key = "sar_acquisition_id" if example.source_modality == "sar" else "optical_acquisition_id"
    target_key = "sar_acquisition_id" if example.target_modality == "sar" else "optical_acquisition_id"
    query_route = route_values[0]
    anchor_route = route_values[1]
    input_ids = {
        _required_string(anchor_route, source_key),
        _required_string(anchor_route, target_key),
        *(_required_string(route, source_key) for route in route_values[2:]),
    }
    target_id = _required_string(query_route, target_key)
    if target_id in input_ids:
        raise SOPATChunkCachePreflightError(
            f"cached target acquisition leaks into SOPAT input route: {example.sample_id}"
        )
    # Routing remains dual-modality even when the V3 plan did not materialize
    # an acquisition for a role this SOPAT example never consumes.
    for route in route_values:
        _required_string(route, "optical_acquisition_id")
        _required_string(route, "sar_acquisition_id")
    missing = sorted((input_ids | {target_id}).difference(acquisitions))
    if missing:
        raise SOPATChunkCachePreflightError(
            f"chunk cache acquisition metadata is missing {missing[0]}"
        )


def _validate_example_grid(
    example: SOPATExampleV4, cached_grid: Mapping[str, object]
) -> None:
    grid = example.canonical_grid
    if (
        _required_string(cached_grid, "tile") != grid.tile
        or _required_int(cached_grid, "width") != grid.width
        or _required_int(cached_grid, "height") != grid.height
        or _required_string(cached_grid, "crs") != grid.crs
    ):
        raise SOPATChunkCachePreflightError(
            f"chunk cache grid geometry differs for {example.sample_id}"
        )
    raw_transform = cached_grid.get("transform")
    if not isinstance(raw_transform, Sequence) or isinstance(raw_transform, (str, bytes)):
        raise SOPATChunkCachePreflightError("chunk cache grid transform must be a sequence")
    try:
        transform = tuple(float(value) for value in raw_transform)
    except (TypeError, ValueError) as error:
        raise SOPATChunkCachePreflightError("chunk cache grid transform is invalid") from error
    if len(transform) != len(grid.transform) or any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
        for left, right in zip(transform, grid.transform, strict=True)
    ):
        raise SOPATChunkCachePreflightError(
            f"chunk cache grid transform differs for {example.sample_id}"
        )


def _absolute_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.path.normpath(os.fspath(value))))


def _load_json_mapping(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SOPATChunkCachePreflightError(f"cannot read chunk cache metadata: {path}") from error
    if not isinstance(values, Mapping):
        raise SOPATChunkCachePreflightError(f"chunk cache metadata is not a mapping: {path}")
    return values


def _required_mapping_sequence(values: Mapping[str, object], key: str) -> Sequence[Mapping[str, object]]:
    value = values.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SOPATChunkCachePreflightError(f"chunk cache {key} must be a sequence")
    if not all(isinstance(entry, Mapping) for entry in value):
        raise SOPATChunkCachePreflightError(f"chunk cache {key} must contain mappings")
    return value  # type: ignore[return-value]


def _required_string_sequence(values: Mapping[str, object], key: str) -> Sequence[str]:
    value = values.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SOPATChunkCachePreflightError(f"chunk cache {key} must be a sequence")
    if not all(isinstance(entry, str) for entry in value):
        raise SOPATChunkCachePreflightError(f"chunk cache {key} must contain strings")
    return value  # type: ignore[return-value]


def _required_string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise SOPATChunkCachePreflightError(f"chunk cache {key} must be a non-empty string")
    return value


def _optional_string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key, "")
    if not isinstance(value, str):
        raise SOPATChunkCachePreflightError(f"chunk cache {key} must be a string")
    return value


def _required_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool):
        raise SOPATChunkCachePreflightError(f"chunk cache {key} must be an integer")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise SOPATChunkCachePreflightError(f"chunk cache {key} must be an integer") from error


def _coerce_direction(value: str) -> Direction:
    if value not in ALL_DIRECTIONS:
        raise SOPATChunkCachePreflightError(f"unsupported chunk cache direction: {value!r}")
    return value  # type: ignore[return-value]


def _canonical_json_sha256(values: Mapping[str, object]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
