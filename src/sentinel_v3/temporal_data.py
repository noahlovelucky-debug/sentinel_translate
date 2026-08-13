"""Causal multi-frame raster samples built from the immutable pair manifest.

The existing manifest contains paired Sentinel-1/Sentinel-2 observations.  A
``TemporalSample`` deliberately stores only references to those observations;
the TIFFs remain the source of truth.  Every sample is checked twice: while
the index is built and again before a raster dataset can read it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_builder import PairRecord
from .schema import CLEAR_SCL_CODES, S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER

Direction = Literal["sar_to_optical", "optical_to_sar"]
Modality = Literal["optical", "sar"]

SAR_TO_OPTICAL: Direction = "sar_to_optical"
OPTICAL_TO_SAR: Direction = "optical_to_sar"
ALL_DIRECTIONS: tuple[Direction, ...] = (SAR_TO_OPTICAL, OPTICAL_TO_SAR)
TEMPORAL_INDEX_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TemporalIndexConfig:
    """Selection constraints for a causal temporal index."""

    source_frames: int = 1
    horizon_days: int = 180
    split: str | None = None
    orbit: str | None = "ascending"
    directions: tuple[Direction, ...] = ALL_DIRECTIONS
    max_samples: int | None = None

    def __post_init__(self) -> None:
        if self.source_frames <= 0:
            raise ValueError("source_frames must be positive")
        if not 1 <= self.horizon_days <= 180:
            raise ValueError("horizon_days must be in [1, 180]")
        if self.split == "":
            raise ValueError("split cannot be empty")
        if self.orbit == "":
            raise ValueError("orbit cannot be empty")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when supplied")
        directions = _normalize_directions(self.directions)
        object.__setattr__(self, "directions", directions)


@dataclass(frozen=True)
class TemporalSample:
    """One fully causal query, anchor, and source-frame selection."""

    sample_id: str
    direction: Direction
    split: str
    tile: str
    year: int
    orbit: str
    query_pair_id: str
    anchor_pair_id: str
    source_pair_ids: tuple[str, ...]
    query_date: str
    anchor_date: str
    source_dates: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.direction not in ALL_DIRECTIONS:
            raise ValueError(f"unsupported temporal direction: {self.direction!r}")
        source_pair_ids = tuple(str(value) for value in self.source_pair_ids)
        source_dates = tuple(str(value) for value in self.source_dates)
        if not source_pair_ids:
            raise ValueError("a temporal sample needs at least one source frame")
        if len(source_pair_ids) != len(source_dates):
            raise ValueError("source_pair_ids and source_dates must have equal length")
        if len(source_pair_ids) != len(set(source_pair_ids)):
            raise ValueError("source_pair_ids must be unique")
        date.fromisoformat(self.query_date)
        date.fromisoformat(self.anchor_date)
        for source_date in source_dates:
            date.fromisoformat(source_date)
        object.__setattr__(self, "source_pair_ids", source_pair_ids)
        object.__setattr__(self, "source_dates", source_dates)

    @property
    def source_frames(self) -> int:
        return len(self.source_pair_ids)

    @property
    def source_modality(self) -> Modality:
        return source_modality(self.direction)

    @property
    def target_modality(self) -> Modality:
        return target_modality(self.direction)

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["format_version"] = TEMPORAL_INDEX_FORMAT_VERSION
        values["source_pair_ids"] = list(self.source_pair_ids)
        values["source_dates"] = list(self.source_dates)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TemporalSample:
        payload = dict(values)
        payload.pop("format_version", None)
        payload["source_pair_ids"] = tuple(str(value) for value in payload["source_pair_ids"])
        payload["source_dates"] = tuple(str(value) for value in payload["source_dates"])
        payload["year"] = int(payload["year"])
        payload["direction"] = str(payload["direction"])
        return cls(**payload)


@dataclass(frozen=True)
class TemporalIndex:
    """An immutable, sequence-like collection of temporal samples."""

    config: TemporalIndexConfig
    samples: tuple[TemporalSample, ...]

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        for sample in samples:
            if sample.direction not in self.config.directions:
                raise ValueError("temporal sample direction is outside the index configuration")
            if sample.source_frames != self.config.source_frames:
                raise ValueError("temporal sample does not contain the configured source frame count")
            if self.config.split is not None and sample.split != self.config.split:
                raise ValueError("temporal sample split is outside the index configuration")
            if self.config.orbit is not None and sample.orbit != self.config.orbit:
                raise ValueError("temporal sample orbit is outside the index configuration")
        object.__setattr__(self, "samples", samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[TemporalSample]:
        return iter(self.samples)

    def __getitem__(self, index: int | slice) -> TemporalSample | TemporalIndex:
        if isinstance(index, slice):
            return TemporalIndex(config=self.config, samples=self.samples[index])
        return self.samples[index]

    def subset(self, *, start: int = 0, limit: int | None = None) -> TemporalIndex:
        """Return a slice without relaxing the original selection constraints."""

        if start < 0:
            raise ValueError("start must be non-negative")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when supplied")
        stop = None if limit is None else start + limit
        return TemporalIndex(config=self.config, samples=self.samples[start:stop])

    def assert_causality(
        self,
        records: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
        *,
        asset_root: str | Path | None = None,
    ) -> None:
        assert_strict_causality(self, records, asset_root=asset_root)


def source_modality(direction: Direction) -> Modality:
    if direction == SAR_TO_OPTICAL:
        return "sar"
    if direction == OPTICAL_TO_SAR:
        return "optical"
    raise ValueError(f"unsupported temporal direction: {direction!r}")


def target_modality(direction: Direction) -> Modality:
    if direction == SAR_TO_OPTICAL:
        return "optical"
    if direction == OPTICAL_TO_SAR:
        return "sar"
    raise ValueError(f"unsupported temporal direction: {direction!r}")


def load_pair_records(
    path: str | Path,
    *,
    start: int = 0,
    limit: int | None = None,
) -> list[PairRecord]:
    """Read manifest ``PairRecord`` entries from a JSONL file in stable order."""

    records: list[PairRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid manifest JSON on line {line_number}: {path}") from error
            if not isinstance(values, dict):
                raise TypeError(f"manifest line {line_number} is not a JSON object: {path}")
            records.append(PairRecord.from_dict(values))
    return _slice_records(sorted(records, key=lambda record: record.pair_id), start=start, limit=limit)


def write_pair_records(path: str | Path, records: Iterable[PairRecord]) -> None:
    """Atomically write ``PairRecord`` entries as canonical JSONL."""

    _write_jsonl(Path(path), (record.to_dict() for record in sorted(records, key=_pair_id)))


def read_pair_records(path: str | Path, **kwargs: object) -> list[PairRecord]:
    """Alias for :func:`load_pair_records` for conventional JSONL readers."""

    return load_pair_records(path, **kwargs)


def write_temporal_index(
    path: str | Path, index: TemporalIndex | Sequence[TemporalSample]
) -> None:
    """Atomically write temporal sample references as JSONL."""

    if isinstance(index, TemporalIndex):
        samples = index.samples
        config = asdict(index.config)
    else:
        samples = tuple(index)
        config = None

    def rows() -> Iterator[dict[str, object]]:
        for sample in sorted(samples, key=_sample_sort_key):
            row = sample.to_dict()
            if config is not None:
                row["index_config"] = config
            yield row

    _write_jsonl(Path(path), rows())


def load_temporal_index(
    path: str | Path,
    *,
    source_frames: int | None = None,
    horizon_days: int | None = None,
    start: int = 0,
    limit: int | None = None,
) -> TemporalIndex:
    """Load JSONL temporal references and infer their fixed source-frame count."""

    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when supplied")
    samples: list[TemporalSample] = []
    serialized_config: dict[str, object] | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid temporal-index JSON on line {line_number}: {path}") from error
            if not isinstance(values, dict):
                raise TypeError(f"temporal-index line {line_number} is not a JSON object: {path}")
            row_config = values.get("index_config")
            if row_config is not None:
                if not isinstance(row_config, dict):
                    raise ValueError(f"temporal-index line {line_number} has invalid index_config")
                normalized_config = _temporal_index_config_dict(row_config)
                if serialized_config is None:
                    serialized_config = normalized_config
                elif serialized_config != normalized_config:
                    raise ValueError("temporal-index rows have inconsistent index_config values")
            values.pop("index_config", None)
            samples.append(TemporalSample.from_dict(values))
    sorted_samples = sorted(samples, key=_sample_sort_key)
    inferred_frames = source_frames
    if inferred_frames is None:
        inferred_frames = (
            int(serialized_config["source_frames"])
            if serialized_config is not None
            else (sorted_samples[0].source_frames if sorted_samples else 1)
        )
    inferred_horizon = horizon_days
    if inferred_horizon is None:
        inferred_horizon = int(serialized_config["horizon_days"]) if serialized_config else 180
    directions = (
        tuple(str(value) for value in serialized_config["directions"])
        if serialized_config is not None
        else tuple(sorted({sample.direction for sample in sorted_samples})) or ALL_DIRECTIONS
    )
    config = TemporalIndexConfig(
        source_frames=inferred_frames,
        horizon_days=inferred_horizon,
        split=None if serialized_config is None else _optional_string(serialized_config.get("split")),
        orbit=None if serialized_config is None else _optional_string(serialized_config.get("orbit")),
        directions=directions,  # type: ignore[arg-type]
        max_samples=None if serialized_config is None else _optional_int(serialized_config.get("max_samples")),
    )
    return TemporalIndex(config=config, samples=tuple(_slice_records(sorted_samples, start=start, limit=limit)))


def read_temporal_index(path: str | Path, **kwargs: object) -> TemporalIndex:
    """Alias for :func:`load_temporal_index` for conventional JSONL readers."""

    return load_temporal_index(path, **kwargs)


def build_temporal_index(
    records: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
    *,
    source_frames: int = 1,
    horizon_days: int = 180,
    split: str | None = None,
    orbit: str | None = "ascending",
    directions: Iterable[Direction] = ALL_DIRECTIONS,
    max_samples: int | None = None,
    asset_root: str | Path | None = None,
) -> TemporalIndex:
    """Build a causally valid index without loading raster pixels.

    Candidate observations are isolated by the exact ``split``, ``tile``,
    ``year``, and ``orbit`` of each query.  The nearest previous target-modality
    asset is used as the anchor.  The source sequence contains the most recent
    distinct source-modality assets in the 180-day causal horizon at or before
    the query date, in chronological order.  A target that lacks any required
    observation is simply omitted.
    """

    config = TemporalIndexConfig(
        source_frames=source_frames,
        horizon_days=horizon_days,
        split=split,
        orbit=orbit,
        directions=tuple(directions),
        max_samples=max_samples,
    )
    resolved_records, inferred_root = _coerce_records(records)
    root = _resolve_asset_root(asset_root, inferred_root)
    record_map = _pair_record_map(resolved_records)
    groups: dict[tuple[str, str, int, str], list[PairRecord]] = {}
    for record in resolved_records:
        if config.split is not None and record.split != config.split:
            continue
        if config.orbit is not None and record.orbit != config.orbit:
            continue
        groups.setdefault(_isolation_key(record), []).append(record)

    samples: list[TemporalSample] = []
    for group_records in groups.values():
        ordered_group = sorted(group_records, key=_pair_id)
        for direction in config.directions:
            built = _build_group_samples(
                ordered_group,
                direction=direction,
                source_frames=config.source_frames,
                horizon_days=config.horizon_days,
                asset_root=root,
            )
            samples.extend(built)

    samples.sort(key=_sample_sort_key)
    if config.max_samples is not None:
        samples = samples[: config.max_samples]
    index = TemporalIndex(config=config, samples=tuple(samples))
    assert_strict_causality(index, record_map, asset_root=root)
    return index


def assert_strict_causality(
    index: TemporalIndex | Sequence[TemporalSample],
    records: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
    *,
    source_frames: int | None = None,
    horizon_days: int | None = None,
    asset_root: str | Path | None = None,
) -> None:
    """Raise ``AssertionError`` when an index violates its causal contract.

    This intentionally checks both dates and physical asset identities.  A
    manually edited index cannot smuggle the query target raster into a source
    input, and records from another split, tile, year, orbit, or grid cannot be
    used as a temporal neighbor.
    """

    if isinstance(index, TemporalIndex):
        samples = index.samples
        required_frames = index.config.source_frames if source_frames is None else source_frames
        maximum_horizon = index.config.horizon_days if horizon_days is None else horizon_days
        configured_split = index.config.split
        configured_orbit = index.config.orbit
    else:
        samples = tuple(index)
        required_frames = source_frames
        maximum_horizon = 180 if horizon_days is None else horizon_days
        configured_split = None
        configured_orbit = None
    if required_frames is not None and required_frames <= 0:
        raise ValueError("source_frames must be positive")
    if not 1 <= maximum_horizon <= 180:
        raise ValueError("horizon_days must be in [1, 180]")

    resolved_records, inferred_root = _coerce_records(records)
    root = _resolve_asset_root(asset_root, inferred_root)
    record_map = _pair_record_map(resolved_records)
    seen_sample_ids: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen_sample_ids:
            _causal_failure(sample, "duplicate sample_id")
        seen_sample_ids.add(sample.sample_id)
        if required_frames is not None and sample.source_frames != required_frames:
            _causal_failure(sample, "wrong source frame count")
        if configured_split is not None and sample.split != configured_split:
            _causal_failure(sample, "sample split is outside the index configuration")
        if configured_orbit is not None and sample.orbit != configured_orbit:
            _causal_failure(sample, "sample orbit is outside the index configuration")
        try:
            query = record_map[sample.query_pair_id]
            anchor = record_map[sample.anchor_pair_id]
            sources = tuple(record_map[pair_id] for pair_id in sample.source_pair_ids)
        except KeyError as error:
            _causal_failure(sample, f"references a missing pair record: {error.args[0]}")

        if _isolation_key(query) != (sample.split, sample.tile, sample.year, sample.orbit):
            _causal_failure(sample, "sample isolation metadata does not match its query")
        for candidate in (anchor, *sources):
            if _isolation_key(candidate) != _isolation_key(query):
                _causal_failure(sample, "source or anchor crosses split/tile/year/orbit isolation")
            if not _grids_match(query, candidate):
                _causal_failure(sample, "source or anchor does not share the query grid")

        target_kind = target_modality(sample.direction)
        source_kind = source_modality(sample.direction)
        query_date = _acquired_date(query, target_kind)
        anchor_date = _acquired_date(anchor, target_kind)
        source_dates = tuple(_acquired_date(record, source_kind) for record in sources)
        if sample.query_date != query_date.isoformat():
            _causal_failure(sample, "stored query date differs from its target asset")
        if sample.anchor_date != anchor_date.isoformat():
            _causal_failure(sample, "stored anchor date differs from its target asset")
        if sample.source_dates != tuple(value.isoformat() for value in source_dates):
            _causal_failure(sample, "stored source dates differ from their source assets")
        if not anchor_date < query_date:
            _causal_failure(sample, "anchor is not strictly before the query")
        if (query_date - anchor_date).days > maximum_horizon:
            _causal_failure(sample, "anchor exceeds the causal horizon")
        if any(source_date > query_date for source_date in source_dates):
            _causal_failure(sample, "source frame is after the query")
        if any((query_date - source_date).days > maximum_horizon for source_date in source_dates):
            _causal_failure(sample, "source frame exceeds the causal horizon")
        if tuple(source_dates) != tuple(sorted(source_dates)):
            _causal_failure(sample, "source frames are not in chronological order")

        target_assets = set(_asset_paths(query, target_kind, root=root, include_mask=True))
        anchor_assets = set(_asset_paths(anchor, target_kind, root=root, include_mask=True))
        if target_assets.intersection(anchor_assets):
            _causal_failure(sample, "query target asset appears in the anchor")
        seen_source_assets: set[tuple[str, ...]] = set()
        for source in sources:
            source_assets = _asset_paths(source, source_kind, root=root, include_mask=False)
            if target_assets.intersection(source_assets):
                _causal_failure(sample, "query target asset appears in source inputs")
            identity = tuple(source_assets)
            if identity in seen_source_assets:
                _causal_failure(sample, "source inputs contain a repeated physical asset")
            seen_source_assets.add(identity)


assert_temporal_causality = assert_strict_causality


class TemporalRasterDataset(Dataset[dict[str, object]]):
    """Read deterministic aligned crops for a :class:`TemporalIndex`.

    Values use the old V3 normalized representation: optical reflectance is
    ``[-1, 1]`` and SAR is per-polarization normalized dB in ``[-1, 1]``.
    Invalid values are exact zeros, while their modality-specific validity mask
    remains available to the caller.
    """

    def __init__(
        self,
        manifest: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
        index: str | Path | TemporalIndex | Sequence[TemporalSample],
        *,
        crop_size: int = 256,
        crop_attempts: int = 24,
        minimum_valid_fraction: float = 0.80,
        seed: int = 0,
        cache_in_memory: bool = False,
        cache: bool | None = None,
        max_cache_items: int | None = None,
        start: int = 0,
        limit: int | None = None,
        asset_root: str | Path | None = None,
    ) -> None:
        if crop_size <= 0 or crop_size % 4:
            raise ValueError("crop_size must be positive and divisible by four")
        if crop_attempts <= 0:
            raise ValueError("crop_attempts must be positive")
        if not 0.0 <= minimum_valid_fraction <= 1.0:
            raise ValueError("minimum_valid_fraction must be in [0, 1]")
        if max_cache_items is not None and max_cache_items <= 0:
            raise ValueError("max_cache_items must be positive when supplied")
        if start < 0:
            raise ValueError("start must be non-negative")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when supplied")
        if cache is not None:
            cache_in_memory = cache

        records, inferred_root = _coerce_records(manifest)
        self.records = _pair_record_map(records)
        self.asset_root = _resolve_asset_root(asset_root, inferred_root)
        self.index = _coerce_index(index)
        assert_strict_causality(self.index, self.records, asset_root=self.asset_root)
        directions = {sample.direction for sample in self.index.samples}
        if len(directions) > 1:
            raise ValueError(
                "TemporalRasterDataset requires a single-direction index; build or filter one task at a time"
            )
        stop = None if limit is None else start + limit
        self.samples = self.index.samples[start:stop]
        self.crop_size = crop_size
        self.crop_attempts = crop_attempts
        self.minimum_valid_fraction = minimum_valid_fraction
        self.seed = int(seed)
        self.cache_in_memory = bool(cache_in_memory)
        self.max_cache_items = max_cache_items
        self._cache: OrderedDict[int, dict[str, object]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self.samples)
        if not 0 <= index < len(self.samples):
            raise IndexError(index)
        if self.cache_in_memory and index in self._cache:
            self._cache.move_to_end(index)
            return _clone_item(self._cache[index])

        item = self._read_item(self.samples[index])
        if self.cache_in_memory:
            self._cache[index] = _clone_item(item)
            if self.max_cache_items is not None and len(self._cache) > self.max_cache_items:
                self._cache.popitem(last=False)
        return item

    def _read_item(self, sample: TemporalSample) -> dict[str, object]:
        query = self.records[sample.query_pair_id]
        anchor = self.records[sample.anchor_pair_id]
        sources = tuple(self.records[pair_id] for pair_id in sample.source_pair_ids)
        if query.width < self.crop_size or query.height < self.crop_size:
            raise ValueError(f"raster_smaller_than_crop: {sample.sample_id}")

        generator = np.random.default_rng(_crop_seed(sample.sample_id, self.seed))
        for _ in range(self.crop_attempts):
            row = int(generator.integers(0, query.height - self.crop_size + 1))
            col = int(generator.integers(0, query.width - self.crop_size + 1))
            window = (col, row, self.crop_size, self.crop_size)
            source_values: list[np.ndarray] = []
            source_valid: list[np.ndarray] = []
            for source in sources:
                values, valid = self._read_modality_window(source, sample.source_modality, window)
                source_values.append(values)
                source_valid.append(valid)
            anchor_values, anchor_valid = self._read_modality_window(
                anchor, sample.target_modality, window
            )
            target_values, target_valid = self._read_modality_window(
                query, sample.target_modality, window
            )
            all_valid = np.logical_and.reduce((target_valid, anchor_valid, *source_valid))
            if float(all_valid.mean()) < self.minimum_valid_fraction:
                continue
            query_date = date.fromisoformat(sample.query_date)
            source_days = np.asarray(
                [(date.fromisoformat(value) - query_date).days for value in sample.source_dates],
                dtype=np.float32,
            )
            anchor_days = np.float32((date.fromisoformat(sample.anchor_date) - query_date).days)
            if np.any(source_days > 0) or anchor_days >= 0:
                raise RuntimeError(f"causal dates changed while reading {sample.sample_id}")
            return {
                "source_values": torch.from_numpy(np.stack(source_values).astype(np.float32)),
                "source_valid": torch.from_numpy(np.stack(source_valid).astype(np.float32)),
                "anchor_values": torch.from_numpy(anchor_values.astype(np.float32)),
                "anchor_valid": torch.from_numpy(anchor_valid.astype(np.float32)),
                "target_values": torch.from_numpy(target_values.astype(np.float32)),
                "target_valid": torch.from_numpy(target_valid.astype(np.float32)),
                "source_days": torch.from_numpy(source_days),
                "anchor_days": torch.tensor(anchor_days, dtype=torch.float32),
                "sample_id": sample.sample_id,
                "direction": sample.direction,
            }
        raise RuntimeError(
            f"insufficient valid temporal crops after {self.crop_attempts} attempts: {sample.sample_id}"
        )

    def _read_modality_window(
        self,
        record: PairRecord,
        modality: Modality,
        window: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        import rasterio
        from rasterio.windows import Window

        raster_window = Window(*window)
        width, height = window[2:]
        if modality == "optical":
            raw_bands = []
            for channel in S2_CHANNEL_ORDER:
                path = self._resolve_asset(record.s2[channel])
                with rasterio.open(path) as source:
                    self._assert_raster_grid(source, record, path)
                    raw_bands.append(source.read(1, window=raster_window))
            scl_path = self._resolve_asset(record.scl)
            with rasterio.open(scl_path) as source:
                self._assert_raster_grid(source, record, scl_path)
                scl = source.read(1, window=raster_window)
            raw = _require_window_shape(np.stack(raw_bands), (len(S2_CHANNEL_ORDER), height, width))
            scl = _require_window_shape(scl, (height, width))
            valid = np.isin(scl, CLEAR_SCL_CODES) & np.all(raw > 0, axis=0)
            values = np.clip(raw.astype(np.float32) / 10000.0, 0.0, 1.0) * 2.0 - 1.0
        else:
            raw_bands = []
            for polarization in SAR_CHANNEL_ORDER:
                path = self._resolve_asset(record.sar[polarization])
                with rasterio.open(path) as source:
                    self._assert_raster_grid(source, record, path)
                    raw_bands.append(source.read(1, window=raster_window))
            raw = _require_window_shape(np.stack(raw_bands), (len(SAR_CHANNEL_ORDER), height, width))
            valid = np.all(raw > 0, axis=0)
            values_db = raw.astype(np.float32) / 200.0 - 50.0
            minimum = np.asarray((-35.0, -45.0), dtype=np.float32)[:, None, None]
            maximum = np.asarray((5.0, -5.0), dtype=np.float32)[:, None, None]
            values = np.clip(2.0 * (values_db - minimum) / (maximum - minimum) - 1.0, -1.0, 1.0)
        values[:, ~valid] = 0.0
        return values.astype(np.float32), valid[None].astype(np.float32)

    def _resolve_asset(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.asset_root / path
        return path.resolve(strict=False)

    @staticmethod
    def _assert_raster_grid(source: Any, record: PairRecord, path: Path) -> None:
        if source.crs is None:
            raise RuntimeError(f"raster has no CRS: {path}")
        actual = (
            int(source.width),
            int(source.height),
            source.crs.to_string(),
            tuple(float(value) for value in source.transform[:6]),
            abs(float(source.transform.a)),
        )
        expected = _grid_signature(record)
        if not _grid_tuples_match(actual, expected):
            raise RuntimeError(f"raster grid does not match manifest record: {path}")


def _build_group_samples(
    records: Sequence[PairRecord],
    *,
    direction: Direction,
    source_frames: int,
    horizon_days: int,
    asset_root: Path,
) -> list[TemporalSample]:
    target_kind = target_modality(direction)
    source_kind = source_modality(direction)
    samples: list[TemporalSample] = []
    queries = sorted(records, key=lambda record: (_acquired_date(record, target_kind), record.pair_id))
    for query in queries:
        query_date = _acquired_date(query, target_kind)
        target_identity = set(_asset_paths(query, target_kind, root=asset_root, include_mask=True))
        anchor_candidates = [
            candidate
            for candidate in records
            if _grids_match(query, candidate)
            and _acquired_date(candidate, target_kind) < query_date
            and (query_date - _acquired_date(candidate, target_kind)).days <= horizon_days
            and not target_identity.intersection(
                _asset_paths(candidate, target_kind, root=asset_root, include_mask=True)
            )
        ]
        anchor = _select_latest_unique(
            anchor_candidates,
            modality=target_kind,
            root=asset_root,
            include_mask=True,
        )
        if anchor is None:
            continue

        source_candidates = [
            candidate
            for candidate in records
            if _grids_match(query, candidate)
            and _acquired_date(candidate, source_kind) <= query_date
            and (query_date - _acquired_date(candidate, source_kind)).days <= horizon_days
            and not target_identity.intersection(
                _asset_paths(candidate, source_kind, root=asset_root, include_mask=False)
            )
        ]
        sources = _select_recent_distinct(
            source_candidates,
            modality=source_kind,
            count=source_frames,
            root=asset_root,
        )
        if len(sources) != source_frames:
            continue
        anchor_date = _acquired_date(anchor, target_kind)
        source_dates = tuple(_acquired_date(record, source_kind) for record in sources)
        sample = TemporalSample(
            sample_id=_temporal_sample_id(direction, query, anchor, sources),
            direction=direction,
            split=query.split,
            tile=query.tile,
            year=query.year,
            orbit=query.orbit,
            query_pair_id=query.pair_id,
            anchor_pair_id=anchor.pair_id,
            source_pair_ids=tuple(record.pair_id for record in sources),
            query_date=query_date.isoformat(),
            anchor_date=anchor_date.isoformat(),
            source_dates=tuple(value.isoformat() for value in source_dates),
        )
        samples.append(sample)
    return samples


def _select_latest_unique(
    candidates: Sequence[PairRecord],
    *,
    modality: Modality,
    root: Path,
    include_mask: bool,
) -> PairRecord | None:
    selected = _select_recent_distinct(
        candidates,
        modality=modality,
        count=1,
        root=root,
        include_mask=include_mask,
    )
    return selected[0] if selected else None


def _select_recent_distinct(
    candidates: Sequence[PairRecord],
    *,
    modality: Modality,
    count: int,
    root: Path,
    include_mask: bool = False,
) -> tuple[PairRecord, ...]:
    """Choose most recent unique physical assets, then restore chronological order."""

    by_asset: dict[tuple[str, ...], PairRecord] = {}
    for candidate in candidates:
        identity = _asset_paths(candidate, modality, root=root, include_mask=include_mask)
        current = by_asset.get(identity)
        if current is None or (_acquired_date(candidate, modality), candidate.pair_id) < (
            _acquired_date(current, modality),
            current.pair_id,
        ):
            by_asset[identity] = candidate
    most_recent = sorted(
        by_asset.values(),
        key=lambda candidate: (-_acquired_date(candidate, modality).toordinal(), candidate.pair_id),
    )[:count]
    return tuple(sorted(most_recent, key=lambda candidate: (_acquired_date(candidate, modality), candidate.pair_id)))


def _coerce_index(index: str | Path | TemporalIndex | Sequence[TemporalSample]) -> TemporalIndex:
    if isinstance(index, TemporalIndex):
        return index
    if isinstance(index, (str, Path)):
        return load_temporal_index(index)
    samples = tuple(index)
    source_frames = samples[0].source_frames if samples else 1
    directions = tuple(sorted({sample.direction for sample in samples})) or ALL_DIRECTIONS
    return TemporalIndex(
        config=TemporalIndexConfig(source_frames=source_frames, orbit=None, directions=directions),
        samples=tuple(sorted(samples, key=_sample_sort_key)),
    )


def _coerce_records(
    records: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
) -> tuple[tuple[PairRecord, ...], Path | None]:
    if isinstance(records, (str, Path)):
        path = Path(records).resolve()
        return tuple(load_pair_records(path)), path.parent
    if isinstance(records, Mapping):
        resolved = tuple(records.values())
    else:
        resolved = tuple(records)
    if not all(isinstance(record, PairRecord) for record in resolved):
        raise TypeError("records must be PairRecord objects or a PairRecord manifest path")
    return resolved, None


def _pair_record_map(records: Iterable[PairRecord]) -> dict[str, PairRecord]:
    by_id: dict[str, PairRecord] = {}
    for record in records:
        if record.pair_id in by_id:
            raise ValueError(f"manifest contains duplicate pair_id: {record.pair_id}")
        by_id[record.pair_id] = record
    return by_id


def _slice_records(
    values: Sequence[Any], *, start: int, limit: int | None
) -> list[Any]:
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when supplied")
    stop = None if limit is None else start + limit
    return list(values[start:stop])


def _temporal_index_config_dict(values: Mapping[str, object]) -> dict[str, object]:
    try:
        config = TemporalIndexConfig(
            source_frames=int(values["source_frames"]),
            horizon_days=int(values["horizon_days"]),
            split=_optional_string(values.get("split")),
            orbit=_optional_string(values.get("orbit")),
            directions=tuple(str(value) for value in values["directions"]),  # type: ignore[arg-type]
            max_samples=_optional_int(values.get("max_samples")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid temporal-index index_config") from error
    return {
        "source_frames": config.source_frames,
        "horizon_days": config.horizon_days,
        "split": config.split,
        "orbit": config.orbit,
        "directions": list(config.directions),
        "max_samples": config.max_samples,
    }


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("expected a string or null")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("expected an integer or null")
    return int(value)


def _normalize_directions(directions: Iterable[Direction]) -> tuple[Direction, ...]:
    normalized = tuple(dict.fromkeys(str(direction) for direction in directions))
    if not normalized:
        raise ValueError("directions must be non-empty")
    invalid = tuple(direction for direction in normalized if direction not in ALL_DIRECTIONS)
    if invalid:
        raise ValueError(f"unsupported temporal directions: {invalid}")
    return tuple(normalized)  # type: ignore[return-value]


def _isolation_key(record: PairRecord) -> tuple[str, str, int, str]:
    return record.split, record.tile, int(record.year), record.orbit


def _acquired_date(record: PairRecord, modality: Modality) -> date:
    return date.fromisoformat(record.s2_date if modality == "optical" else record.s1_date)


def _grid_signature(record: PairRecord) -> tuple[int, int, str, tuple[float, ...], float]:
    return (
        int(record.width),
        int(record.height),
        str(record.crs),
        tuple(float(value) for value in record.transform[:6]),
        float(record.gsd),
    )


def _grids_match(left: PairRecord, right: PairRecord) -> bool:
    return _grid_tuples_match(_grid_signature(left), _grid_signature(right))


def _grid_tuples_match(
    left: tuple[int, int, str, tuple[float, ...], float],
    right: tuple[int, int, str, tuple[float, ...], float],
) -> bool:
    return (
        left[:3] == right[:3]
        and len(left[3]) == len(right[3])
        and all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9) for a, b in zip(left[3], right[3]))
        and math.isclose(left[4], right[4], rel_tol=0.0, abs_tol=1e-9)
    )


def _asset_paths(
    record: PairRecord,
    modality: Modality,
    *,
    root: Path,
    include_mask: bool,
) -> tuple[str, ...]:
    if modality == "optical":
        values = [record.s2[channel] for channel in S2_CHANNEL_ORDER]
        if include_mask:
            values.append(record.scl)
    else:
        values = [record.sar[channel] for channel in SAR_CHANNEL_ORDER]
    return tuple(_asset_identity(value, root) for value in values)


def _asset_identity(value: str, root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve(strict=False))


def _resolve_asset_root(asset_root: str | Path | None, inferred_root: Path | None) -> Path:
    if asset_root is not None:
        return Path(asset_root).expanduser().resolve(strict=False)
    if inferred_root is not None:
        return inferred_root.resolve(strict=False)
    return Path.cwd().resolve()


def _pair_id(record: PairRecord) -> str:
    return record.pair_id


def _sample_sort_key(sample: TemporalSample) -> tuple[str, str, str, str]:
    return sample.direction, sample.query_date, sample.query_pair_id, sample.sample_id


def _temporal_sample_id(
    direction: Direction,
    query: PairRecord,
    anchor: PairRecord,
    sources: Sequence[PairRecord],
) -> str:
    source_ids = ",".join(record.pair_id for record in sources)
    return f"{direction}:{query.pair_id}:anchor={anchor.pair_id}:sources={source_ids}"


def _crop_seed(sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _require_window_shape(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if values.shape != shape:
        raise RuntimeError(f"raster window has shape {values.shape}, expected {shape}")
    return values


def _clone_item(item: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in item.items()
    }


def _causal_failure(sample: TemporalSample, reason: str) -> None:
    raise AssertionError(f"strict causal temporal index violation for {sample.sample_id}: {reason}")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, default=_json_default) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_default(value: object) -> object:
    """Keep index writing portable when a caller supplied NumPy scalar metadata."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
