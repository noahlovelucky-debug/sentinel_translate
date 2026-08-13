"""Build a bounded local asset cache for paired-temporal feasibility runs.

The feasibility cache is intentionally a copy-only operation.  It selects the
four fixed 64-sample indexes from the paired feasibility YAML, copies the raw
files those indexes reference, and rewrites a small manifest whose paths stay
inside the local cache.  No raster is decoded here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

import yaml

from .dataset_builder import PairRecord
from .paired_temporal_data import (
    ALL_DIRECTIONS,
    Direction,
    PairedTemporalIndex,
    build_paired_temporal_index,
    load_pair_records,
    source_modality,
    target_modality,
    write_pair_records,
    write_paired_temporal_index,
)
from .schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER

GIB = 1024**3
DEFAULT_CACHE_ROOT = Path("/home/noah/datasets/sentinel_translate_paired_v2_feasibility")
DEFAULT_BUDGET_BYTES = 30 * GIB
DEFAULT_MINIMUM_FREE_BYTES = 80 * GIB
# Feasibility copies are unlimited unless an operator explicitly opts into a cap.
DEFAULT_COPY_RATE_BYTES_PER_SECOND = 0
CACHE_FORMAT_VERSION = 1
_METADATA_RESERVE_BYTES = 4 * 1024 * 1024
_COPY_BUFFER_BYTES = 1024 * 1024


class _CopyRateLimiter:
    """Serial token bucket shared by all asset copies in one materialization."""

    def __init__(
        self,
        bytes_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        rate = float(bytes_per_second)
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("rate limit must be a finite non-negative number")
        self._rate = rate
        self._clock = clock
        self._sleep = sleep
        # A one-buffer burst keeps the cap responsive without allowing an
        # entire large TIFF to bypass the configured global rate.
        self._capacity = min(rate, float(_COPY_BUFFER_BYTES))
        self._tokens = self._capacity
        self._updated_at = clock()

    @property
    def bytes_per_second(self) -> float:
        return self._rate

    def consume(self, byte_count: int) -> None:
        """Reserve transfer bandwidth before writing ``byte_count`` bytes."""

        if byte_count < 0:
            raise ValueError("byte_count cannot be negative")
        if byte_count == 0 or self._rate == 0.0:
            return

        now = self._clock()
        elapsed = max(0.0, now - self._updated_at)
        available = min(self._capacity, self._tokens + elapsed * self._rate)
        required = float(byte_count)
        if required <= available:
            self._tokens = available - required
            self._updated_at = now
            return

        delay = (required - available) / self._rate
        self._sleep(delay)
        # Model bytes generated while sleeping as immediately consumed.  The
        # next caller refills from this scheduled timestamp, so this remains a
        # single bucket across files rather than resetting at file boundaries.
        self._tokens = 0.0
        self._updated_at = now + delay


@dataclass(frozen=True)
class LocalCacheAsset:
    """One globally deduplicated source file and its local cache location."""

    source: Path
    relative_destination: Path
    size_bytes: int
    allocated_bytes: int
    references: int

    @property
    def destination_name(self) -> str:
        return self.relative_destination.as_posix()


@dataclass(frozen=True)
class LocalCacheIndex:
    """One fixed feasibility index and its cache-relative JSONL path."""

    direction: Direction
    split: str
    index: PairedTemporalIndex
    relative_destination: Path


@dataclass(frozen=True)
class PairedTemporalLocalCachePlan:
    """Dry-run result which can later be materialized without re-selection."""

    config_path: Path
    config_sha256: str
    source_manifest: Path
    destination_root: Path
    local_manifest_relative: Path
    indexes: tuple[LocalCacheIndex, ...]
    local_records: tuple[PairRecord, ...]
    assets: tuple[LocalCacheAsset, ...]
    logical_source_bytes: int
    stat_source_bytes: int
    allocated_source_bytes: int
    metadata_reserve_bytes: int
    budget_bytes: int
    minimum_free_bytes: int
    free_bytes: int

    @property
    def local_manifest(self) -> Path:
        return self.destination_root / self.local_manifest_relative

    @property
    def estimated_target_bytes(self) -> int:
        return self.stat_source_bytes + self.metadata_reserve_bytes

    @property
    def estimated_free_after_copy(self) -> int:
        return self.free_bytes - self.estimated_target_bytes

    @property
    def budget_ok(self) -> bool:
        return self.estimated_target_bytes <= self.budget_bytes

    @property
    def free_space_ok(self) -> bool:
        return self.estimated_free_after_copy >= self.minimum_free_bytes

    @property
    def allowed_to_materialize(self) -> bool:
        return self.budget_ok and self.free_space_ok

    def report(self) -> dict[str, object]:
        return {
            "format_version": CACHE_FORMAT_VERSION,
            "action": "dry_run",
            "config": str(self.config_path),
            "config_sha256": self.config_sha256,
            "source_manifest": str(self.source_manifest),
            "destination_root": str(self.destination_root),
            "local_manifest": str(self.local_manifest),
            "indexes": [
                {
                    "direction": entry.direction,
                    "split": entry.split,
                    "samples": len(entry.index),
                    "destination": entry.relative_destination.as_posix(),
                }
                for entry in self.indexes
            ],
            "pair_records": len(self.local_records),
            "unique_files": len(self.assets),
            "logical_file_references": sum(asset.references for asset in self.assets),
            "logical_source_bytes": self.logical_source_bytes,
            "stat_source_bytes": self.stat_source_bytes,
            "source_actual_stat_bytes": self.stat_source_bytes,
            "allocated_source_bytes": self.allocated_source_bytes,
            "metadata_reserve_bytes": self.metadata_reserve_bytes,
            "estimated_target_bytes": self.estimated_target_bytes,
            "budget_bytes": self.budget_bytes,
            "minimum_free_bytes": self.minimum_free_bytes,
            "free_bytes": self.free_bytes,
            "estimated_free_after_copy": self.estimated_free_after_copy,
            "budget_ok": self.budget_ok,
            "free_space_ok": self.free_space_ok,
            "allowed_to_materialize": self.allowed_to_materialize,
        }


def build_paired_temporal_feasibility_cache_plan(
    config_path: str | Path,
    *,
    destination_root: str | Path = DEFAULT_CACHE_ROOT,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES,
) -> PairedTemporalLocalCachePlan:
    """Select fixed paired feasibility inputs and calculate a copy-only plan.

    This reads the JSONL manifest and stats the selected source files, but does
    not create directories, copy assets, or decode TIFF values.
    """

    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    if minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes cannot be negative")
    resolved_config = Path(_lexical_path(config_path))
    values = _load_yaml_mapping(resolved_config)
    data = _required_mapping(values, "data")
    source_manifest = Path(_lexical_path(_required_string(data, "manifest")))
    train_limit = _required_int(data, "max_train_samples")
    validation_limit = _required_int(data, "max_validation_samples")
    if train_limit != 64 or validation_limit != 64:
        raise ValueError(
            "paired temporal feasibility cache requires exactly 64 train and validation samples"
        )
    destination = Path(_lexical_path(destination_root))
    records = load_pair_records(source_manifest)
    record_map = {record.pair_id: record for record in records}
    if len(record_map) != len(records):
        raise ValueError(
            "paired temporal feasibility manifest contains duplicate pair_id values"
        )

    common = {
        "min_observations": _required_int(data, "minimum_observations"),
        "max_observations": _required_int(data, "maximum_observations"),
        "horizon_days": _required_int(data, "horizon_days"),
        "anchor_max_delta_days": _required_int(data, "anchor_pair_max_delta_days"),
        "max_anchors_per_query": _required_int(data, "maximum_anchors_per_query"),
        "translation_max_delta_days": _required_int(data, "translation_max_delta_days"),
        "orbit": _required_string(data, "orbit"),
        "task_modes": _task_modes(data.get("task_modes")),
        "asset_root": source_manifest.parent,
    }
    split_limits = (
        ("train", _required_string(data, "train_split"), train_limit),
        ("validation", _required_string(data, "validation_split"), validation_limit),
    )
    indexes: list[LocalCacheIndex] = []
    required_record_ids: set[str] = set()
    asset_references: dict[str, int] = {}
    for direction in ALL_DIRECTIONS:
        for label, split, limit in split_limits:
            index = build_paired_temporal_index(
                records,
                direction=direction,
                split=split,
                max_samples=limit,
                **common,
            )
            if len(index) != limit:
                raise RuntimeError(
                    f"feasibility selection has {len(index)} {direction} {label} samples, expected {limit}"
                )
            indexes.append(
                LocalCacheIndex(
                    direction=direction,
                    split=label,
                    index=index,
                    relative_destination=Path("indexes") / direction / f"{label}.jsonl",
                )
            )
            for sample in index:
                query = record_map[sample.query_pair_id]
                anchor = record_map[sample.anchor_pair_id]
                observations = tuple(
                    record_map[pair_id] for pair_id in sample.observation_pair_ids
                )
                required_record_ids.update(
                    (sample.query_pair_id, sample.anchor_pair_id, *sample.observation_pair_ids)
                )
                _add_record_modality_references(
                    asset_references,
                    query,
                    target_modality(direction),
                    source_manifest.parent,
                )
                _add_record_modality_references(
                    asset_references,
                    anchor,
                    source_modality(direction),
                    source_manifest.parent,
                )
                _add_record_modality_references(
                    asset_references,
                    anchor,
                    target_modality(direction),
                    source_manifest.parent,
                )
                for observation in observations:
                    _add_record_modality_references(
                        asset_references,
                        observation,
                        source_modality(direction),
                        source_manifest.parent,
                    )

    selected_records = tuple(
        sorted((record_map[value] for value in required_record_ids), key=lambda x: x.pair_id)
    )
    assets = _plan_assets(asset_references)
    destination_by_source = {str(asset.source): asset.relative_destination for asset in assets}
    local_records = tuple(
        _localize_record(
            record,
            source_manifest_parent=source_manifest.parent,
            destination_by_source=destination_by_source,
        )
        for record in selected_records
    )
    stat_source_bytes = sum(asset.size_bytes for asset in assets)
    allocated_source_bytes = sum(asset.allocated_bytes for asset in assets)
    logical_source_bytes = sum(asset.size_bytes * asset.references for asset in assets)
    free_bytes = shutil.disk_usage(_existing_parent(destination)).free
    return PairedTemporalLocalCachePlan(
        config_path=resolved_config,
        config_sha256=file_sha256(resolved_config),
        source_manifest=source_manifest,
        destination_root=destination,
        local_manifest_relative=Path("manifests") / "pairs.jsonl",
        indexes=tuple(indexes),
        local_records=local_records,
        assets=assets,
        logical_source_bytes=logical_source_bytes,
        stat_source_bytes=stat_source_bytes,
        allocated_source_bytes=allocated_source_bytes,
        metadata_reserve_bytes=_METADATA_RESERVE_BYTES,
        budget_bytes=budget_bytes,
        minimum_free_bytes=minimum_free_bytes,
        free_bytes=free_bytes,
    )


def assert_paired_temporal_feasibility_cache_budget(plan: PairedTemporalLocalCachePlan) -> None:
    """Refuse materialization before the fixed budget and free-space gates pass."""

    if not plan.budget_ok:
        raise RuntimeError(
            "paired temporal local cache exceeds budget: "
            f"estimated={plan.estimated_target_bytes}, budget={plan.budget_bytes}"
        )
    current_free_bytes = shutil.disk_usage(_existing_parent(plan.destination_root)).free
    current_free_after_copy = current_free_bytes - plan.estimated_target_bytes
    if current_free_after_copy < plan.minimum_free_bytes:
        raise RuntimeError(
            "paired temporal local cache would violate minimum free space: "
            f"after_copy={current_free_after_copy}, minimum={plan.minimum_free_bytes}"
        )


def materialize_paired_temporal_feasibility_cache(
    plan: PairedTemporalLocalCachePlan,
    *,
    resume: bool = True,
    rate_limit_bytes_per_second: float = DEFAULT_COPY_RATE_BYTES_PER_SECOND,
) -> dict[str, object]:
    """Copy a pre-approved plan and atomically publish its local metadata.

    ``rate_limit_bytes_per_second`` is one serial token bucket for every new
    asset in this materialization.  Set it to zero to disable throttling.
    """

    assert_paired_temporal_feasibility_cache_budget(plan)
    rate_limiter = _CopyRateLimiter(rate_limit_bytes_per_second)
    copied = 0
    reused = 0
    for asset in plan.assets:
        destination = plan.destination_root / asset.relative_destination
        if resume and _destination_matches(destination, asset.size_bytes):
            _assert_source_size(asset)
            reused += 1
            continue
        _copy_asset_atomically(asset, destination, rate_limiter=rate_limiter)
        copied += 1

    local_manifest = plan.local_manifest
    write_pair_records(local_manifest, plan.local_records)
    for entry in plan.indexes:
        write_paired_temporal_index(
            plan.destination_root / entry.relative_destination, entry.index
        )
    checksum_manifest = _write_checksum_manifest(plan)
    return {
        "action": "materialize",
        "destination_root": str(plan.destination_root),
        "copied_files": copied,
        "reused_files": reused,
        "rate_limit_bytes_per_second": rate_limiter.bytes_per_second,
        "local_manifest": str(local_manifest),
        "checksum_manifest": str(checksum_manifest),
        "checksum_manifest_sha256": file_sha256(checksum_manifest),
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise TypeError("paired temporal feasibility config must be a mapping")
    return dict(values)


def _required_mapping(values: Mapping[str, object], key: str) -> dict[str, object]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"paired temporal feasibility config requires {key} mapping")
    return dict(value)


def _required_string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"paired temporal feasibility config requires non-empty {key}")
    return value


def _required_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"paired temporal feasibility config requires integer {key}")
    return int(value)


def _task_modes(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("paired temporal feasibility config requires data.task_modes list")
    result = tuple(str(item) for item in value)
    if not result:
        raise ValueError("paired temporal feasibility data.task_modes cannot be empty")
    return result


def _add_record_modality_references(
    references: dict[str, int],
    record: PairRecord,
    modality: str,
    source_manifest_parent: Path,
) -> None:
    for value in _modality_asset_values(record, modality):
        source = str(_source_asset_path(value, source_manifest_parent))
        references[source] = references.get(source, 0) + 1


def _plan_assets(references: Mapping[str, int]) -> tuple[LocalCacheAsset, ...]:
    assets: list[LocalCacheAsset] = []
    for source_text, count in sorted(references.items()):
        source = Path(source_text)
        try:
            stat = source.stat()
        except OSError as error:
            raise RuntimeError(
                f"cannot stat selected paired temporal asset: {source}"
            ) from error
        if not source.is_file():
            raise RuntimeError(
                f"selected paired temporal asset is not a regular file: {source}"
            )
        digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        name = source.name or "asset"
        assets.append(
            LocalCacheAsset(
                source=source,
                relative_destination=Path("assets") / digest / name,
                size_bytes=int(stat.st_size),
                allocated_bytes=int(getattr(stat, "st_blocks", 0)) * 512,
                references=int(count),
            )
        )
    return tuple(assets)


def _localize_record(
    record: PairRecord,
    *,
    source_manifest_parent: Path,
    destination_by_source: Mapping[str, Path],
) -> PairRecord:
    def localized(value: str) -> str:
        source = str(_source_asset_path(value, source_manifest_parent))
        destination = destination_by_source.get(source)
        if destination is not None:
            return (Path("..") / destination).as_posix()
        # A selected record can carry a modality that no fixed index consumes.
        # Do not retain its remote path: an accidental read should fail locally
        # instead of silently widening the feasibility corpus.
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        return (Path("..") / "not_materialized" / digest / Path(value).name).as_posix()

    return replace(
        record,
        s2={channel: localized(record.s2[channel]) for channel in S2_CHANNEL_ORDER},
        scl=localized(record.scl),
        sar={channel: localized(record.sar[channel]) for channel in SAR_CHANNEL_ORDER},
    )


def _record_asset_values(record: PairRecord) -> tuple[str, ...]:
    return (
        *(record.s2[channel] for channel in S2_CHANNEL_ORDER),
        record.scl,
        *(record.sar[channel] for channel in SAR_CHANNEL_ORDER),
    )


def _modality_asset_values(record: PairRecord, modality: str) -> tuple[str, ...]:
    if modality == "optical":
        return (*(record.s2[channel] for channel in S2_CHANNEL_ORDER), record.scl)
    if modality == "sar":
        return tuple(record.sar[channel] for channel in SAR_CHANNEL_ORDER)
    raise ValueError(f"unsupported paired temporal modality: {modality!r}")


def _source_asset_path(value: str, root: Path) -> Path:
    expanded = os.path.expanduser(value)
    if not os.path.isabs(expanded):
        expanded = os.path.join(os.fspath(root), expanded)
    return Path(_lexical_path(expanded))


def _lexical_path(value: str | Path) -> str:
    return os.path.abspath(os.path.normpath(os.path.expanduser(os.fspath(value))))


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise RuntimeError(f"cannot find an existing filesystem parent for {path}")
        candidate = parent
    return candidate


def _destination_matches(path: Path, expected_size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == expected_size
    except OSError:
        return False


def _assert_source_size(asset: LocalCacheAsset) -> None:
    try:
        actual_size = asset.source.stat().st_size
    except OSError as error:
        raise RuntimeError(
            f"cannot stat selected paired temporal asset: {asset.source}"
        ) from error
    if actual_size != asset.size_bytes:
        raise RuntimeError(
            f"selected source asset changed since dry-run: {asset.source} "
            f"expected={asset.size_bytes}, actual={actual_size}"
        )


def _copy_asset_atomically(
    asset: LocalCacheAsset,
    destination: Path,
    *,
    rate_limiter: _CopyRateLimiter,
) -> None:
    _assert_source_size(asset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with asset.source.open("rb") as source, temporary.open("wb") as target:
            _copy_stream(source, target, rate_limiter)
        copied_size = temporary.stat().st_size
        _assert_source_size(asset)
        if copied_size != asset.size_bytes:
            raise RuntimeError(
                f"copied paired temporal asset size mismatch: {asset.source} "
                f"expected={asset.size_bytes}, actual={copied_size}"
            )
        os.replace(temporary, destination)
        if destination.stat().st_size != asset.size_bytes:
            raise RuntimeError(f"destination size mismatch after atomic copy: {destination}")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _copy_stream(source: BinaryIO, target: BinaryIO, rate_limiter: _CopyRateLimiter) -> None:
    while chunk := source.read(_COPY_BUFFER_BYTES):
        rate_limiter.consume(len(chunk))
        target.write(chunk)


def _write_checksum_manifest(plan: PairedTemporalLocalCachePlan) -> Path:
    entries: list[dict[str, object]] = []
    for asset in plan.assets:
        destination = plan.destination_root / asset.relative_destination
        if not _destination_matches(destination, asset.size_bytes):
            raise RuntimeError(
                f"local paired temporal asset failed final size verification: {destination}"
            )
        entries.append(
            {
                "source": str(asset.source),
                "destination": asset.destination_name,
                "size_bytes": asset.size_bytes,
                "sha256": file_sha256(destination),
                "references": asset.references,
            }
        )
    index_entries = []
    for entry in plan.indexes:
        destination = plan.destination_root / entry.relative_destination
        if not destination.is_file():
            raise RuntimeError(f"local paired temporal index was not written: {destination}")
        index_entries.append(
            {
                "direction": entry.direction,
                "split": entry.split,
                "samples": len(entry.index),
                "destination": entry.relative_destination.as_posix(),
                "sha256": file_sha256(destination),
            }
        )
    if not plan.local_manifest.is_file():
        raise RuntimeError(
            f"local paired temporal manifest was not written: {plan.local_manifest}"
        )
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "config": str(plan.config_path),
        "config_sha256": plan.config_sha256,
        "source_manifest": str(plan.source_manifest),
        "local_manifest": plan.local_manifest_relative.as_posix(),
        "local_manifest_sha256": file_sha256(plan.local_manifest),
        "indexes": index_entries,
        "assets": entries,
    }
    destination = plan.destination_root / "cache_manifest.json"
    _atomic_json(destination, payload)
    return destination


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
