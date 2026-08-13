"""Sparse causal paired-anchor temporal samples for V3 raster training.

Each sample has one historical, registered source/target anchor pair, an
ordered variable-length sequence of source-modality observations, and a later
target-modality query label.  The index stores references into ``pairs.jsonl``;
the dataset reads only the selected aligned TIFF windows.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import registration_shift_audit
from .dataset_builder import PairRecord
from .schema import CLEAR_SCL_CODES, S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER

Direction = Literal["sar_to_optical", "optical_to_sar"]
Modality = Literal["optical", "sar"]
TaskMode = Literal["translation", "forecast"]
CropMode = Literal["random_valid", "center"]

SAR_TO_OPTICAL: Direction = "sar_to_optical"
OPTICAL_TO_SAR: Direction = "optical_to_sar"
ALL_DIRECTIONS: tuple[Direction, ...] = (SAR_TO_OPTICAL, OPTICAL_TO_SAR)
TRANSLATION: TaskMode = "translation"
FORECAST: TaskMode = "forecast"
ALL_TASK_MODES: tuple[TaskMode, ...] = (TRANSLATION, FORECAST)
PAIRED_TEMPORAL_INDEX_FORMAT_VERSION = 1


def centered_crop_window(
    *, width: int, height: int, crop_size: int
) -> tuple[int, int, int, int]:
    """Return the one canonical center crop in rasterio ``col,row,w,h`` order."""

    if width < crop_size or height < crop_size:
        raise ValueError("raster is smaller than crop_size")
    return (
        (width - crop_size) // 2,
        (height - crop_size) // 2,
        crop_size,
        crop_size,
    )


@dataclass(frozen=True)
class PairedTemporalIndexConfig:
    """Selection constraints for one single-direction paired temporal index."""

    direction: Direction
    min_observations: int = 1
    max_observations: int | None = None
    horizon_days: int = 180
    anchor_max_delta_days: int = 1
    max_anchors_per_query: int = 1
    translation_max_delta_days: int = 1
    split: str | None = None
    orbit: str | None = "ascending"
    task_modes: tuple[TaskMode, ...] = ALL_TASK_MODES
    max_samples: int | None = None

    def __post_init__(self) -> None:
        if self.direction not in ALL_DIRECTIONS:
            raise ValueError(f"unsupported paired temporal direction: {self.direction!r}")
        if self.min_observations <= 0:
            raise ValueError("min_observations must be positive")
        if self.max_observations is not None and self.max_observations < self.min_observations:
            raise ValueError("max_observations must be at least min_observations")
        if not 1 <= self.horizon_days <= 180:
            raise ValueError("horizon_days must be in [1, 180]")
        if not 0 <= self.anchor_max_delta_days <= 1:
            raise ValueError("anchor_max_delta_days must be in [0, 1]")
        if self.max_anchors_per_query <= 0:
            raise ValueError("max_anchors_per_query must be positive")
        if not 0 <= self.translation_max_delta_days <= 1:
            raise ValueError("translation_max_delta_days must be in [0, 1]")
        if self.split == "":
            raise ValueError("split cannot be empty")
        if self.orbit == "":
            raise ValueError("orbit cannot be empty")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when supplied")
        object.__setattr__(self, "task_modes", _normalize_task_modes(self.task_modes))


@dataclass(frozen=True)
class PairedTemporalSample:
    """One historical paired anchor, source observation sequence, and label."""

    sample_id: str
    direction: Direction
    task_mode: TaskMode
    split: str
    tile: str
    year: int
    orbit: str
    query_pair_id: str
    anchor_pair_id: str
    observation_pair_ids: tuple[str, ...]
    query_date: str
    source_anchor_date: str
    target_anchor_date: str
    observation_dates: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.direction not in ALL_DIRECTIONS:
            raise ValueError(f"unsupported paired temporal direction: {self.direction!r}")
        if self.task_mode not in ALL_TASK_MODES:
            raise ValueError(f"unsupported paired temporal task mode: {self.task_mode!r}")
        observation_pair_ids = tuple(str(value) for value in self.observation_pair_ids)
        observation_dates = tuple(str(value) for value in self.observation_dates)
        if not observation_pair_ids:
            raise ValueError("a paired temporal sample needs at least one observation")
        if len(observation_pair_ids) != len(observation_dates):
            raise ValueError("observation_pair_ids and observation_dates must have equal length")
        if len(observation_pair_ids) != len(set(observation_pair_ids)):
            raise ValueError("observation_pair_ids must be unique")
        date.fromisoformat(self.query_date)
        date.fromisoformat(self.source_anchor_date)
        date.fromisoformat(self.target_anchor_date)
        for observation_date in observation_dates:
            date.fromisoformat(observation_date)
        object.__setattr__(self, "observation_pair_ids", observation_pair_ids)
        object.__setattr__(self, "observation_dates", observation_dates)

    @property
    def observation_count(self) -> int:
        return len(self.observation_pair_ids)

    @property
    def source_anchor_pair_id(self) -> str:
        """The source side of the explicit registered anchor pair."""

        return self.anchor_pair_id

    @property
    def target_anchor_pair_id(self) -> str:
        """The target side of the explicit registered anchor pair."""

        return self.anchor_pair_id

    @property
    def source_modality(self) -> Modality:
        return source_modality(self.direction)

    @property
    def target_modality(self) -> Modality:
        return target_modality(self.direction)

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["format_version"] = PAIRED_TEMPORAL_INDEX_FORMAT_VERSION
        values["observation_pair_ids"] = list(self.observation_pair_ids)
        values["observation_dates"] = list(self.observation_dates)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> PairedTemporalSample:
        payload = dict(values)
        payload.pop("format_version", None)
        payload["observation_pair_ids"] = tuple(str(value) for value in payload["observation_pair_ids"])
        payload["observation_dates"] = tuple(str(value) for value in payload["observation_dates"])
        payload["year"] = int(payload["year"])
        payload["direction"] = str(payload["direction"])
        payload["task_mode"] = str(payload["task_mode"])
        return cls(**payload)


@dataclass(frozen=True)
class PairedTemporalIndex:
    """Immutable, sequence-like collection of one-direction paired samples."""

    config: PairedTemporalIndexConfig
    samples: tuple[PairedTemporalSample, ...]

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        seen_sample_ids: set[str] = set()
        for sample in samples:
            if sample.sample_id in seen_sample_ids:
                raise ValueError(f"paired temporal index contains duplicate sample_id: {sample.sample_id}")
            seen_sample_ids.add(sample.sample_id)
            if sample.direction != self.config.direction:
                raise ValueError("paired temporal index must contain exactly one direction")
            if sample.task_mode not in self.config.task_modes:
                raise ValueError("sample task mode is outside the index configuration")
            if sample.observation_count < self.config.min_observations:
                raise ValueError("sample has fewer observations than the index minimum")
            if (
                self.config.max_observations is not None
                and sample.observation_count > self.config.max_observations
            ):
                raise ValueError("sample has more observations than the index maximum")
            if self.config.split is not None and sample.split != self.config.split:
                raise ValueError("sample split is outside the index configuration")
            if self.config.orbit is not None and sample.orbit != self.config.orbit:
                raise ValueError("sample orbit is outside the index configuration")
        object.__setattr__(self, "samples", samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[PairedTemporalSample]:
        return iter(self.samples)

    def __getitem__(self, index: int | slice) -> PairedTemporalSample | PairedTemporalIndex:
        if isinstance(index, slice):
            return PairedTemporalIndex(config=self.config, samples=self.samples[index])
        return self.samples[index]

    def subset(self, *, start: int = 0, limit: int | None = None) -> PairedTemporalIndex:
        """Return a sample slice without relaxing the causal selection constraints."""

        return PairedTemporalIndex(
            config=self.config,
            samples=tuple(_slice_values(self.samples, start=start, limit=limit)),
        )

    def assert_causality(
        self,
        records: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
        *,
        asset_root: str | Path | None = None,
    ) -> None:
        assert_paired_temporal_causality(self, records, asset_root=asset_root)


def source_modality(direction: Direction) -> Modality:
    if direction == SAR_TO_OPTICAL:
        return "sar"
    if direction == OPTICAL_TO_SAR:
        return "optical"
    raise ValueError(f"unsupported paired temporal direction: {direction!r}")


def target_modality(direction: Direction) -> Modality:
    if direction == SAR_TO_OPTICAL:
        return "optical"
    if direction == OPTICAL_TO_SAR:
        return "sar"
    raise ValueError(f"unsupported paired temporal direction: {direction!r}")


def load_pair_records(
    path: str | Path,
    *,
    start: int = 0,
    limit: int | None = None,
) -> list[PairRecord]:
    """Load sorted ``PairRecord`` values from a manifest JSONL file."""

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
    return _slice_values(sorted(records, key=lambda record: record.pair_id), start=start, limit=limit)


def read_pair_records(path: str | Path, **kwargs: object) -> list[PairRecord]:
    """Alias for :func:`load_pair_records`."""

    return load_pair_records(path, **kwargs)


def write_pair_records(path: str | Path, records: Iterable[PairRecord]) -> None:
    """Atomically write canonical pair-record JSONL."""

    _write_jsonl(
        Path(path),
        (record.to_dict() for record in sorted(records, key=lambda record: record.pair_id)),
    )


def build_paired_temporal_index(
    records: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
    *,
    direction: Direction,
    min_observations: int = 1,
    max_observations: int | None = None,
    horizon_days: int = 180,
    anchor_max_delta_days: int = 1,
    max_anchors_per_query: int = 1,
    translation_max_delta_days: int = 1,
    split: str | None = None,
    orbit: str | None = "ascending",
    task_modes: Iterable[TaskMode] = ALL_TASK_MODES,
    max_samples: int | None = None,
    asset_root: str | Path | None = None,
) -> PairedTemporalIndex:
    """Build one-direction causal samples without reading raster pixels.

    The historical anchor is one manifest pair whose source and target
    acquisitions differ by at most ``anchor_max_delta_days``.  Observation
    sequences deliberately retain every eligible distinct source frame when
    ``max_observations`` is ``None``; this leaves training-time observation
    dropout free to choose a subsequence without rebuilding the index.
    """

    config = PairedTemporalIndexConfig(
        direction=direction,
        min_observations=min_observations,
        max_observations=max_observations,
        horizon_days=horizon_days,
        anchor_max_delta_days=anchor_max_delta_days,
        max_anchors_per_query=max_anchors_per_query,
        translation_max_delta_days=translation_max_delta_days,
        split=split,
        orbit=orbit,
        task_modes=tuple(task_modes),
        max_samples=max_samples,
    )
    resolved_records, inferred_root = _coerce_records(records)
    root = _resolve_asset_root(asset_root, inferred_root)
    record_map = _pair_record_map(resolved_records)
    groups: dict[tuple[str, str, str], list[PairRecord]] = {}
    for record in resolved_records:
        if config.split is not None and record.split != config.split:
            continue
        if config.orbit is not None and record.orbit != config.orbit:
            continue
        groups.setdefault(_isolation_key(record), []).append(record)

    # Asset identities are expensive on large network-backed manifests.  The
    # per-group views retain the legacy ordering rules while normalizing each
    # source/target path at most once for this index build.
    identity_cache: dict[str, str] = {}
    group_caches = tuple(
        _GroupSelectionCache(
            group_records,
            config=config,
            root=root,
            identity_cache=identity_cache,
        )
        for group_records in groups.values()
    )
    samples: list[PairedTemporalSample] = []
    if config.max_samples is None:
        for group_cache in group_caches:
            samples.extend(group_cache.build_all_samples())
    else:
        # ``_sample_sort_key`` orders primarily by query date and pair id.
        # Processing that same global query order means that, once a complete
        # query contributes enough samples, later queries cannot enter the
        # requested prefix.  We still build every selected anchor for the
        # boundary query before applying the final sort/slice.
        scheduled_queries: list[
            tuple[date, str, int, _GroupSelectionCache, _IndexedPair]
        ] = []
        for group_cache in group_caches:
            for query in group_cache.queries:
                scheduled_queries.append(
                    (
                        query.target_date,
                        query.record.pair_id,
                        len(scheduled_queries),
                        group_cache,
                        query,
                    )
                )
        scheduled_queries.sort(key=lambda value: value[:3])
        for _, _, _, group_cache, query in scheduled_queries:
            samples.extend(group_cache.build_query_samples(query))
            if len(samples) >= config.max_samples:
                break
    samples.sort(key=_sample_sort_key)
    if config.max_samples is not None:
        samples = samples[: config.max_samples]
    index = PairedTemporalIndex(config=config, samples=tuple(samples))
    assert_paired_temporal_causality(index, record_map, asset_root=root)
    return index


def write_paired_temporal_index(
    path: str | Path,
    index: PairedTemporalIndex | Sequence[PairedTemporalSample],
) -> None:
    """Atomically write paired temporal references as JSONL.

    Config is carried on every data row for straightforward inspection.  An
    empty index gets one metadata row so a read/write round trip still retains
    its direction and constraints.
    """

    if isinstance(index, PairedTemporalIndex):
        samples = index.samples
        config = _config_to_dict(index.config)
    else:
        samples = tuple(index)
        config = None

    def rows() -> Iterator[dict[str, object]]:
        if not samples and config is not None:
            yield {
                "record_type": "paired_temporal_index_metadata",
                "format_version": PAIRED_TEMPORAL_INDEX_FORMAT_VERSION,
                "index_config": config,
            }
        for sample in sorted(samples, key=_sample_sort_key):
            row = sample.to_dict()
            if config is not None:
                row["index_config"] = config
            yield row

    _write_jsonl(Path(path), rows())


def load_paired_temporal_index(
    path: str | Path,
    *,
    direction: Direction | None = None,
    start: int = 0,
    limit: int | None = None,
) -> PairedTemporalIndex:
    """Load paired temporal samples, preserving their serialized constraints."""

    _validate_slice(start=start, limit=limit)
    samples: list[PairedTemporalSample] = []
    serialized_config: dict[str, object] | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid paired temporal-index JSON on line {line_number}: {path}"
                ) from error
            if not isinstance(values, dict):
                raise TypeError(f"paired temporal-index line {line_number} is not a JSON object: {path}")
            row_config = values.get("index_config")
            if row_config is not None:
                if not isinstance(row_config, dict):
                    raise ValueError(f"paired temporal-index line {line_number} has invalid index_config")
                normalized_config = _config_dict_from_values(row_config)
                if serialized_config is None:
                    serialized_config = normalized_config
                elif serialized_config != normalized_config:
                    raise ValueError("paired temporal-index rows have inconsistent index_config values")
            if values.get("record_type") == "paired_temporal_index_metadata":
                continue
            values.pop("index_config", None)
            samples.append(PairedTemporalSample.from_dict(values))

    samples.sort(key=_sample_sort_key)
    if serialized_config is not None:
        config = _config_from_dict(serialized_config)
        if direction is not None and direction != config.direction:
            raise ValueError("requested direction differs from serialized paired temporal index")
    else:
        inferred_direction = direction
        if inferred_direction is None and samples:
            directions = {sample.direction for sample in samples}
            if len(directions) != 1:
                raise ValueError("paired temporal index must contain exactly one direction")
            inferred_direction = samples[0].direction
        if inferred_direction is None:
            raise ValueError("an empty unconfigured paired temporal index needs direction=")
        config = PairedTemporalIndexConfig(direction=inferred_direction)
    return PairedTemporalIndex(
        config=config,
        samples=tuple(_slice_values(samples, start=start, limit=limit)),
    )


def read_paired_temporal_index(path: str | Path, **kwargs: object) -> PairedTemporalIndex:
    """Alias for :func:`load_paired_temporal_index`."""

    return load_paired_temporal_index(path, **kwargs)


def assert_paired_temporal_causality(
    index: PairedTemporalIndex | Sequence[PairedTemporalSample],
    records: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
    *,
    asset_root: str | Path | None = None,
) -> None:
    """Assert temporal dates, physical assets, isolation, and grid integrity.

    The target-modality query raster is label-only: neither side of the paired
    anchor nor any source observation may reuse any of its physical assets.
    """

    if isinstance(index, PairedTemporalIndex):
        config = index.config
        samples = index.samples
    else:
        samples = tuple(index)
        directions = {sample.direction for sample in samples}
        if len(directions) > 1:
            raise AssertionError("paired temporal index contains more than one direction")
        config = PairedTemporalIndexConfig(direction=next(iter(directions), SAR_TO_OPTICAL))
    resolved_records, inferred_root = _coerce_records(records)
    root = _resolve_asset_root(asset_root, inferred_root)
    record_map = _pair_record_map(resolved_records)
    seen_sample_ids: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen_sample_ids:
            _causal_failure(sample, "duplicate sample_id")
        seen_sample_ids.add(sample.sample_id)
        if sample.direction != config.direction:
            _causal_failure(sample, "index contains more than one direction")
        if sample.task_mode not in config.task_modes:
            _causal_failure(sample, "task mode is outside the index configuration")
        if sample.observation_count < config.min_observations:
            _causal_failure(sample, "too few source observations")
        if (
            config.max_observations is not None
            and sample.observation_count > config.max_observations
        ):
            _causal_failure(sample, "too many source observations")
        if config.split is not None and sample.split != config.split:
            _causal_failure(sample, "sample split is outside the index configuration")
        if config.orbit is not None and sample.orbit != config.orbit:
            _causal_failure(sample, "sample orbit is outside the index configuration")
        try:
            query = record_map[sample.query_pair_id]
            anchor = record_map[sample.anchor_pair_id]
            observations = tuple(record_map[pair_id] for pair_id in sample.observation_pair_ids)
        except KeyError as error:
            _causal_failure(sample, f"references a missing pair record: {error.args[0]}")

        if query.split != sample.split or query.tile != sample.tile or query.orbit != sample.orbit:
            _causal_failure(sample, "sample isolation metadata does not match its query")
        for candidate in (anchor, *observations):
            if _isolation_key(candidate) != _isolation_key(query):
                _causal_failure(sample, "input crosses split/tile/orbit isolation")
            if not _grids_match(query, candidate):
                _causal_failure(sample, "input does not share the query grid")

        source_kind = source_modality(sample.direction)
        target_kind = target_modality(sample.direction)
        query_date = _acquired_date(query, target_kind)
        source_anchor_date = _acquired_date(anchor, source_kind)
        target_anchor_date = _acquired_date(anchor, target_kind)
        observation_dates = tuple(_acquired_date(record, source_kind) for record in observations)
        if sample.query_date != query_date.isoformat():
            _causal_failure(sample, "stored query date differs from its target asset")
        if sample.source_anchor_date != source_anchor_date.isoformat():
            _causal_failure(sample, "stored source anchor date differs from its source asset")
        if sample.target_anchor_date != target_anchor_date.isoformat():
            _causal_failure(sample, "stored target anchor date differs from its target asset")
        if sample.observation_dates != tuple(value.isoformat() for value in observation_dates):
            _causal_failure(sample, "stored observation dates differ from their source assets")
        if not target_anchor_date < query_date:
            _causal_failure(sample, "target anchor is not strictly before the query")
        if not source_anchor_date < query_date:
            _causal_failure(sample, "source anchor is not strictly before the query")
        if (query_date - target_anchor_date).days > config.horizon_days:
            _causal_failure(sample, "target anchor exceeds the causal horizon")
        if (query_date - source_anchor_date).days > config.horizon_days:
            _causal_failure(sample, "source anchor exceeds the causal horizon")
        if abs((source_anchor_date - target_anchor_date).days) > config.anchor_max_delta_days:
            _causal_failure(sample, "source and target anchor dates are not a registered pair")
        if abs(int(anchor.delta_days)) > config.anchor_max_delta_days:
            _causal_failure(sample, "anchor manifest delta exceeds the registered-pair allowance")
        if any(observation_date > query_date for observation_date in observation_dates):
            _causal_failure(sample, "source observation is after the query")
        if any(
            (query_date - observation_date).days > config.horizon_days
            for observation_date in observation_dates
        ):
            _causal_failure(sample, "source observation exceeds the causal horizon")
        if tuple(observation_dates) != tuple(sorted(observation_dates)):
            _causal_failure(sample, "source observations are not in chronological order")
        expected_mode = _task_mode_for_observations(
            query_date,
            observation_dates,
            translation_max_delta_days=config.translation_max_delta_days,
        )
        if sample.task_mode != expected_mode:
            _causal_failure(sample, "task mode does not match the final observation date")

        query_target_assets = set(_asset_paths(query, target_kind, root=root, include_mask=True))
        source_anchor_assets = _asset_paths(anchor, source_kind, root=root, include_mask=True)
        target_anchor_assets = _asset_paths(anchor, target_kind, root=root, include_mask=True)
        input_assets = set(source_anchor_assets).union(target_anchor_assets)
        if query_target_assets.intersection(input_assets):
            _causal_failure(sample, "query target asset appears in anchor inputs")
        seen_observation_assets: set[tuple[str, ...]] = set()
        source_anchor_identity = tuple(source_anchor_assets)
        for observation in observations:
            observation_assets = _asset_paths(observation, source_kind, root=root, include_mask=True)
            if query_target_assets.intersection(observation_assets):
                _causal_failure(sample, "query target asset appears in source observations")
            identity = tuple(observation_assets)
            if identity == source_anchor_identity:
                _causal_failure(sample, "source anchor is repeated as an observation")
            if identity in seen_observation_assets:
                _causal_failure(sample, "source observations contain a repeated physical asset")
            seen_observation_assets.add(identity)


assert_strict_paired_temporal_causality = assert_paired_temporal_causality


class PairedTemporalRasterDataset(Dataset[dict[str, object]]):
    """Read deterministic aligned crops from a paired temporal index.

    Values use old V3 units: optical reflectance is mapped to ``[-1, 1]`` and
    SAR dB is normalized per polarization to ``[-1, 1]``.  The index remains
    variable-length; this dataset pads observations to a stable leading
    dimension and marks real frames with ``observation_present``.  Training
    uses ``crop_mode='random_valid'`` to seek dense joint support, while
    validation uses ``crop_mode='center'`` to read one fixed center crop and
    preserve its true validity masks for masked evaluation.
    """

    def __init__(
        self,
        manifest: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
        index: str | Path | PairedTemporalIndex | Sequence[PairedTemporalSample],
        *,
        crop_size: int = 256,
        crop_attempts: int = 24,
        crop_mode: CropMode = "random_valid",
        minimum_valid_fraction: float = 0.80,
        seed: int = 0,
        cache_in_memory: bool = False,
        cache: bool | None = None,
        max_cache_items: int | None = None,
        max_observations: int | None = None,
        pad_observations_to: int | None = None,
        registration_audit: bool = True,
        maximum_registration_shift_px: float = 0.5,
        start: int = 0,
        limit: int | None = None,
        asset_root: str | Path | None = None,
    ) -> None:
        if crop_size <= 0 or crop_size % 4:
            raise ValueError("crop_size must be positive and divisible by four")
        if crop_attempts <= 0:
            raise ValueError("crop_attempts must be positive")
        if crop_mode not in {"random_valid", "center"}:
            raise ValueError("crop_mode must be 'random_valid' or 'center'")
        if not 0.0 <= minimum_valid_fraction <= 1.0:
            raise ValueError("minimum_valid_fraction must be in [0, 1]")
        if max_cache_items is not None and max_cache_items <= 0:
            raise ValueError("max_cache_items must be positive when supplied")
        _validate_slice(start=start, limit=limit)
        if cache is not None:
            cache_in_memory = cache
        if (
            max_observations is not None
            and pad_observations_to is not None
            and max_observations != pad_observations_to
        ):
            raise ValueError("max_observations and pad_observations_to disagree")
        requested_padding = (
            pad_observations_to if pad_observations_to is not None else max_observations
        )
        if requested_padding is not None and requested_padding <= 0:
            raise ValueError("observation padding length must be positive")
        if maximum_registration_shift_px < 0.0:
            raise ValueError("maximum_registration_shift_px cannot be negative")

        records, inferred_root = _coerce_records(manifest)
        self.records = _pair_record_map(records)
        self.asset_root = _resolve_asset_root(asset_root, inferred_root)
        self.index = _coerce_index(index)
        assert_paired_temporal_causality(self.index, self.records, asset_root=self.asset_root)
        stop = None if limit is None else start + limit
        self.samples = self.index.samples[start:stop]
        observed_maximum = max((sample.observation_count for sample in self.samples), default=0)
        if requested_padding is not None and requested_padding < observed_maximum:
            raise ValueError("observation padding length is smaller than an indexed sequence")
        self.padded_observations = requested_padding
        self.crop_size = crop_size
        self.crop_attempts = crop_attempts
        self.crop_mode = crop_mode
        self.minimum_valid_fraction = minimum_valid_fraction
        self.registration_audit = bool(registration_audit)
        self.maximum_registration_shift_px = float(maximum_registration_shift_px)
        self.seed = int(seed)
        self.cache_in_memory = bool(cache_in_memory)
        self.max_cache_items = max_cache_items
        self._cache: OrderedDict[int, dict[str, object]] = OrderedDict()
        # A shared tensor keeps persistent DataLoader workers synchronized with
        # the main-process epoch without requiring a worker restart.
        self._epoch_state = torch.zeros((), dtype=torch.int64).share_memory_()
        self._last_seen_epoch = 0

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic random-crop epoch.

        ``random_valid`` folds this value into its crop seed, so each epoch
        gets a new but replayable crop sequence.  ``center`` deliberately
        ignores it.  The state is shared so persistent DataLoader workers
        observe updates before their next item read.
        """

        self._epoch_state.fill_(int(epoch))
        self._synchronize_epoch()

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self.samples)
        if not 0 <= index < len(self.samples):
            raise IndexError(index)
        self._synchronize_epoch()
        if self.cache_in_memory and index in self._cache:
            self._cache.move_to_end(index)
            return _clone_item(self._cache[index])
        item = self._read_item(self.samples[index])
        if self.cache_in_memory:
            self._cache[index] = _clone_item(item)
            if self.max_cache_items is not None and len(self._cache) > self.max_cache_items:
                self._cache.popitem(last=False)
        return item

    def _synchronize_epoch(self) -> None:
        """Drop process-local random-crop cache entries after an epoch change."""

        epoch = int(self._epoch_state.item())
        if epoch == self._last_seen_epoch:
            return
        self._last_seen_epoch = epoch
        if self.crop_mode == "random_valid":
            self._cache.clear()

    def _read_item(self, sample: PairedTemporalSample) -> dict[str, object]:
        query = self.records[sample.query_pair_id]
        anchor = self.records[sample.anchor_pair_id]
        observations = tuple(self.records[pair_id] for pair_id in sample.observation_pair_ids)
        if query.width < self.crop_size or query.height < self.crop_size:
            raise ValueError(f"raster_smaller_than_crop: {sample.sample_id}")

        if self.crop_mode == "center":
            windows = (
                centered_crop_window(
                    width=query.width,
                    height=query.height,
                    crop_size=self.crop_size,
                ),
            )
        else:
            generator = np.random.default_rng(
                _crop_seed(
                    sample.sample_id,
                    self.seed,
                    epoch=int(self._epoch_state.item()),
                )
            )
            windows = tuple(
                (
                    int(generator.integers(0, query.width - self.crop_size + 1)),
                    int(generator.integers(0, query.height - self.crop_size + 1)),
                    self.crop_size,
                    self.crop_size,
                )
                for _ in range(self.crop_attempts)
            )
        for window in windows:
            source_anchor_values, source_anchor_valid = self._read_modality_window(
                anchor, sample.source_modality, window
            )
            target_anchor_values, target_anchor_valid = self._read_modality_window(
                anchor, sample.target_modality, window
            )
            observation_values: list[np.ndarray] = []
            observation_valid: list[np.ndarray] = []
            for observation in observations:
                values, valid = self._read_modality_window(observation, sample.source_modality, window)
                observation_values.append(values)
                observation_valid.append(valid)
            target_values, target_valid = self._read_modality_window(
                query, sample.target_modality, window
            )
            base_valid = np.logical_and.reduce(
                (source_anchor_valid, target_anchor_valid, target_valid)
            )
            observation_support = np.logical_or.reduce(np.stack(observation_valid), axis=0)
            joint_support = np.logical_and(base_valid, observation_support)
            evaluation_support = np.logical_and(target_valid, target_anchor_valid)
            if self.crop_mode == "center":
                if not bool(evaluation_support.any()):
                    raise RuntimeError(
                        "center paired temporal crop has no evaluable target/anchor pixels: "
                        f"{sample.sample_id}"
                    )
            elif float(joint_support.mean()) < self.minimum_valid_fraction:
                continue
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
            if (
                np.any(observation_days > 0)
                or source_anchor_days >= 0
                or target_anchor_days >= 0
            ):
                raise RuntimeError(f"causal dates changed while reading {sample.sample_id}")
            high_frequency_valid = np.zeros_like(target_valid, dtype=np.float32)
            registration_shift_px = float("inf")
            registration_zero_ncc = float("nan")
            registration_best_ncc = float("nan")
            registration_evidence_supported = False
            high_frequency_eligible = False
            high_frequency_weight = 0.0
            if self.registration_audit and sample.task_mode == TRANSLATION:
                # The final chronological observation is the actual source
                # evidence that made this sample a translation.  Never reopen
                # the source side of the query pair here: in the reverse
                # direction that asset may have been acquired after the target.
                query_source_values = observation_values[-1]
                query_source_valid = observation_valid[-1]
                high_frequency_valid = np.logical_and.reduce(
                    (
                        base_valid,
                        query_source_valid.astype(bool),
                    )
                ).astype(np.float32)
                optical = (
                    target_values if sample.target_modality == "optical" else query_source_values
                )
                sar = target_values if sample.target_modality == "sar" else query_source_values
                registration = registration_shift_audit(
                    torch.from_numpy(optical),
                    torch.from_numpy(sar),
                    valid=torch.from_numpy(high_frequency_valid),
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
                    high_frequency_weight = (
                        1.0 if int(observation_days[-1]) == 0 else 0.25
                    )
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
        raise RuntimeError(
            "insufficient valid paired temporal crops after "
            f"{self.crop_attempts} attempts: {sample.sample_id}"
        )

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
            (sequence_length, source_channels, self.crop_size, self.crop_size),
            dtype=np.float32,
        )
        padded_valid = np.zeros(
            (sequence_length, 1, self.crop_size, self.crop_size), dtype=np.float32
        )
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
            "source_anchor_days": torch.tensor(
                source_anchor_days, dtype=torch.float32
            ),
            "target_anchor_days": torch.tensor(
                target_anchor_days, dtype=torch.float32
            ),
            # Compatibility alias for the original one-anchor-time contract.
            "anchor_days": torch.tensor(target_anchor_days, dtype=torch.float32),
            "target_values": torch.from_numpy(target_values.astype(np.float32)),
            "target_valid": torch.from_numpy(target_valid.astype(np.float32)),
            "high_frequency_valid": torch.from_numpy(
                high_frequency_valid.astype(np.float32)
            ),
            "high_frequency_eligible": torch.tensor(high_frequency_eligible),
            "high_frequency_weight": torch.tensor(
                high_frequency_weight, dtype=torch.float32
            ),
            "registration_shift_px": torch.tensor(
                registration_shift_px, dtype=torch.float32
            ),
            "registration_zero_ncc": torch.tensor(
                registration_zero_ncc, dtype=torch.float32
            ),
            "registration_best_ncc": torch.tensor(
                registration_best_ncc, dtype=torch.float32
            ),
            "registration_evidence_supported": torch.tensor(
                registration_evidence_supported
            ),
            "sample_id": sample.sample_id,
            "direction": sample.direction,
            "task_mode": sample.task_mode,
        }

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
            values = np.clip(
                2.0 * (values_db - minimum) / (maximum - minimum) - 1.0,
                -1.0,
                1.0,
            )
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
        if not _grid_tuples_match(actual, _grid_signature(record)):
            raise RuntimeError(f"raster grid does not match manifest record: {path}")


PairedTemporalDataset = PairedTemporalRasterDataset


def collate_paired_temporal(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Collate sparse paired samples with batch-local observation padding.

    Dataset items may use different fixed padding lengths (or be produced by
    separate workers with different local sequence maxima).  This collator
    preserves all present observations and pads only the batch's observation
    axis.  Padded values, validity masks, and days are zero; the boolean
    ``observation_present`` mask is the authoritative indicator of a real
    frame.
    """

    if not samples:
        raise ValueError("collate_paired_temporal requires at least one sample")
    tensor_keys = (
        "source_anchor_values",
        "source_anchor_valid",
        "target_anchor_values",
        "target_anchor_valid",
        "target_values",
        "target_valid",
        "high_frequency_valid",
        "high_frequency_eligible",
        "high_frequency_weight",
        "registration_shift_px",
        "registration_zero_ncc",
        "registration_best_ncc",
        "registration_evidence_supported",
        "source_anchor_days",
        "target_anchor_days",
        "anchor_days",
    )
    observation_keys = (
        "observation_values",
        "observation_valid",
        "observation_days",
        "observation_present",
    )
    string_keys = ("sample_id", "direction", "task_mode")
    required = set(tensor_keys).union(observation_keys, string_keys)
    missing = [key for key in required if any(key not in sample for sample in samples)]
    if missing:
        raise KeyError(f"paired temporal samples are missing required keys: {sorted(missing)}")

    observation_values = [_require_tensor(sample["observation_values"], "observation_values") for sample in samples]
    observation_valid = [_require_tensor(sample["observation_valid"], "observation_valid") for sample in samples]
    observation_days = [_require_tensor(sample["observation_days"], "observation_days") for sample in samples]
    observation_present = [
        _require_tensor(sample["observation_present"], "observation_present").to(dtype=torch.bool)
        for sample in samples
    ]
    maximum_observations = max(values.shape[0] for values in observation_values)
    padded_values: list[torch.Tensor] = []
    padded_valid: list[torch.Tensor] = []
    padded_days: list[torch.Tensor] = []
    padded_present: list[torch.Tensor] = []
    for values, valid, days, present in zip(
        observation_values, observation_valid, observation_days, observation_present, strict=True
    ):
        _validate_observation_tensors(values, valid, days, present)
        padded_values.append(_pad_observation_tensor(values, maximum_observations))
        padded_valid.append(_pad_observation_tensor(valid, maximum_observations))
        padded_days.append(_pad_observation_tensor(days, maximum_observations))
        padded_present.append(_pad_observation_tensor(present, maximum_observations).to(torch.bool))

    batch: dict[str, object] = {
        "observation_values": torch.stack(padded_values),
        "observation_valid": torch.stack(padded_valid),
        "observation_days": torch.stack(padded_days),
        "observation_present": torch.stack(padded_present),
    }
    for key in tensor_keys:
        batch[key] = torch.stack([_require_tensor(sample[key], key) for sample in samples])
    for key in string_keys:
        batch[key] = [str(sample[key]) for sample in samples]
    return batch


def _build_group_samples(
    records: Sequence[PairRecord],
    *,
    config: PairedTemporalIndexConfig,
    root: Path,
) -> list[PairedTemporalSample]:
    source_kind = source_modality(config.direction)
    target_kind = target_modality(config.direction)
    samples: list[PairedTemporalSample] = []
    queries = sorted(records, key=lambda record: (_acquired_date(record, target_kind), record.pair_id))
    for query in queries:
        query_date = _acquired_date(query, target_kind)
        query_target_assets = set(_asset_paths(query, target_kind, root=root, include_mask=True))
        anchors = [
            candidate
            for candidate in records
            if _eligible_anchor(
                candidate,
                query_date=query_date,
                source_kind=source_kind,
                target_kind=target_kind,
                config=config,
                query_target_assets=query_target_assets,
                root=root,
            )
            and _grids_match(query, candidate)
        ]
        selected_anchors = _select_latest_anchors(
            anchors,
            target_kind=target_kind,
            source_kind=source_kind,
            root=root,
            limit=config.max_anchors_per_query,
        )
        for anchor in selected_anchors:
            source_anchor_assets = _asset_paths(anchor, source_kind, root=root, include_mask=True)
            observations = _select_observations(
                records,
                query=query,
                query_date=query_date,
                source_kind=source_kind,
                query_target_assets=query_target_assets,
                source_anchor_assets=source_anchor_assets,
                config=config,
                root=root,
            )
            if len(observations) < config.min_observations:
                continue
            observation_dates = tuple(
                _acquired_date(record, source_kind) for record in observations
            )
            task_mode = _task_mode_for_observations(
                query_date,
                observation_dates,
                translation_max_delta_days=config.translation_max_delta_days,
            )
            if task_mode not in config.task_modes:
                continue
            samples.append(
                PairedTemporalSample(
                    sample_id=_paired_sample_id(config.direction, query, anchor, observations),
                    direction=config.direction,
                    task_mode=task_mode,
                    split=query.split,
                    tile=query.tile,
                    year=query.year,
                    orbit=query.orbit,
                    query_pair_id=query.pair_id,
                    anchor_pair_id=anchor.pair_id,
                    observation_pair_ids=tuple(record.pair_id for record in observations),
                    query_date=query_date.isoformat(),
                    source_anchor_date=_acquired_date(anchor, source_kind).isoformat(),
                    target_anchor_date=_acquired_date(anchor, target_kind).isoformat(),
                    observation_dates=tuple(value.isoformat() for value in observation_dates),
                )
            )
    return samples


@dataclass(frozen=True)
class _IndexedPair:
    """Per-direction manifest facts used repeatedly during one index build."""

    record: PairRecord
    source_date: date
    target_date: date
    grid: tuple[int, int, str, tuple[float, ...], float]
    source_assets: tuple[str, ...]
    target_assets: tuple[str, ...]


class _GroupSelectionCache:
    """Cached selection view for one split/tile/orbit group.

    This deliberately mirrors the reference helpers above.  Dates are indexed
    for the bounded causal horizon and asset identities are computed once per
    record, rather than once per candidate/query/anchor comparison.
    """

    def __init__(
        self,
        records: Sequence[PairRecord],
        *,
        config: PairedTemporalIndexConfig,
        root: Path,
        identity_cache: dict[str, str],
    ) -> None:
        self.config = config
        self.source_kind = source_modality(config.direction)
        self.target_kind = target_modality(config.direction)
        self.entries = tuple(
            _IndexedPair(
                record=record,
                source_date=_acquired_date(record, self.source_kind),
                target_date=_acquired_date(record, self.target_kind),
                grid=_grid_signature(record),
                source_assets=_asset_paths_cached(
                    record,
                    self.source_kind,
                    root=root,
                    identity_cache=identity_cache,
                ),
                target_assets=_asset_paths_cached(
                    record,
                    self.target_kind,
                    root=root,
                    identity_cache=identity_cache,
                ),
            )
            for record in records
        )
        self.queries = tuple(
            sorted(self.entries, key=lambda entry: (entry.target_date, entry.record.pair_id))
        )
        self._by_target_date = tuple(
            sorted(self.entries, key=lambda entry: (entry.target_date, entry.record.pair_id))
        )
        self._target_dates = tuple(entry.target_date for entry in self._by_target_date)
        self._by_source_date = tuple(
            sorted(self.entries, key=lambda entry: (entry.source_date, entry.record.pair_id))
        )
        self._source_dates = tuple(entry.source_date for entry in self._by_source_date)

    def build_all_samples(self) -> list[PairedTemporalSample]:
        samples: list[PairedTemporalSample] = []
        for query in self.queries:
            samples.extend(self.build_query_samples(query))
        return samples

    def build_query_samples(self, query: _IndexedPair) -> list[PairedTemporalSample]:
        query_target_assets = set(query.target_assets)
        observations = self._observations_for_query(query, query_target_assets)
        samples: list[PairedTemporalSample] = []
        for anchor in self._latest_anchors(query, query_target_assets):
            selected_observations = tuple(
                candidate
                for candidate in observations
                if candidate.source_assets != anchor.source_assets
            )
            if self.config.max_observations is not None:
                selected_observations = selected_observations[-self.config.max_observations :]
            if len(selected_observations) < self.config.min_observations:
                continue
            observation_dates = tuple(candidate.source_date for candidate in selected_observations)
            task_mode = _task_mode_for_observations(
                query.target_date,
                observation_dates,
                translation_max_delta_days=self.config.translation_max_delta_days,
            )
            if task_mode not in self.config.task_modes:
                continue
            samples.append(
                PairedTemporalSample(
                    sample_id=_paired_sample_id(
                        self.config.direction,
                        query.record,
                        anchor.record,
                        tuple(candidate.record for candidate in selected_observations),
                    ),
                    direction=self.config.direction,
                    task_mode=task_mode,
                    split=query.record.split,
                    tile=query.record.tile,
                    year=query.record.year,
                    orbit=query.record.orbit,
                    query_pair_id=query.record.pair_id,
                    anchor_pair_id=anchor.record.pair_id,
                    observation_pair_ids=tuple(
                        candidate.record.pair_id for candidate in selected_observations
                    ),
                    query_date=query.target_date.isoformat(),
                    source_anchor_date=anchor.source_date.isoformat(),
                    target_anchor_date=anchor.target_date.isoformat(),
                    observation_dates=tuple(value.isoformat() for value in observation_dates),
                )
            )
        return samples

    def _latest_anchors(
        self,
        query: _IndexedPair,
        query_target_assets: set[str],
    ) -> tuple[_IndexedPair, ...]:
        start = bisect_left(
            self._target_dates,
            query.target_date - timedelta(days=self.config.horizon_days),
        )
        stop = bisect_left(self._target_dates, query.target_date)
        candidates: list[_IndexedPair] = []
        for candidate in self._by_target_date[start:stop]:
            if not _grid_tuples_match(query.grid, candidate.grid):
                continue
            if candidate.source_date >= query.target_date:
                continue
            if (query.target_date - candidate.source_date).days > self.config.horizon_days:
                continue
            if abs((candidate.source_date - candidate.target_date).days) > self.config.anchor_max_delta_days:
                continue
            if abs(int(candidate.record.delta_days)) > self.config.anchor_max_delta_days:
                continue
            if query_target_assets.intersection(
                (*candidate.source_assets, *candidate.target_assets)
            ):
                continue
            candidates.append(candidate)

        seen: set[tuple[str, ...]] = set()
        selected: list[_IndexedPair] = []
        for candidate in reversed(candidates):
            identity = (*candidate.source_assets, *candidate.target_assets)
            if identity in seen:
                continue
            seen.add(identity)
            selected.append(candidate)
            if len(selected) == self.config.max_anchors_per_query:
                break
        return tuple(selected)

    def _observations_for_query(
        self,
        query: _IndexedPair,
        query_target_assets: set[str],
    ) -> tuple[_IndexedPair, ...]:
        start = bisect_left(
            self._source_dates,
            query.target_date - timedelta(days=self.config.horizon_days),
        )
        stop = bisect_right(self._source_dates, query.target_date)
        by_assets: dict[tuple[str, ...], _IndexedPair] = {}
        for candidate in self._by_source_date[start:stop]:
            if not _grid_tuples_match(query.grid, candidate.grid):
                continue
            if query_target_assets.intersection(candidate.source_assets):
                continue
            current = by_assets.get(candidate.source_assets)
            if current is None or (current.source_date, current.record.pair_id) < (
                candidate.source_date,
                candidate.record.pair_id,
            ):
                by_assets[candidate.source_assets] = candidate
        return tuple(
            sorted(by_assets.values(), key=lambda entry: (entry.source_date, entry.record.pair_id))
        )


def _eligible_anchor(
    candidate: PairRecord,
    *,
    query_date: date,
    source_kind: Modality,
    target_kind: Modality,
    config: PairedTemporalIndexConfig,
    query_target_assets: set[str],
    root: Path,
) -> bool:
    source_date = _acquired_date(candidate, source_kind)
    target_date = _acquired_date(candidate, target_kind)
    if not target_date < query_date or not source_date < query_date:
        return False
    if (query_date - target_date).days > config.horizon_days:
        return False
    if (query_date - source_date).days > config.horizon_days:
        return False
    if abs((source_date - target_date).days) > config.anchor_max_delta_days:
        return False
    if abs(int(candidate.delta_days)) > config.anchor_max_delta_days:
        return False
    source_assets = _asset_paths(candidate, source_kind, root=root, include_mask=True)
    target_assets = _asset_paths(candidate, target_kind, root=root, include_mask=True)
    return not query_target_assets.intersection((*source_assets, *target_assets))


def _select_latest_anchors(
    candidates: Sequence[PairRecord],
    *,
    target_kind: Modality,
    source_kind: Modality,
    root: Path,
    limit: int,
) -> tuple[PairRecord, ...]:
    seen: set[tuple[str, ...]] = set()
    selected: list[PairRecord] = []
    for candidate in sorted(
        candidates,
        key=lambda record: (_acquired_date(record, target_kind), record.pair_id),
        reverse=True,
    ):
        identity = (
            *_asset_paths(candidate, source_kind, root=root, include_mask=True),
            *_asset_paths(candidate, target_kind, root=root, include_mask=True),
        )
        if identity not in seen:
            seen.add(identity)
            selected.append(candidate)
            if len(selected) == limit:
                break
    return tuple(selected)


def _select_observations(
    records: Sequence[PairRecord],
    *,
    query: PairRecord,
    query_date: date,
    source_kind: Modality,
    query_target_assets: set[str],
    source_anchor_assets: tuple[str, ...],
    config: PairedTemporalIndexConfig,
    root: Path,
) -> tuple[PairRecord, ...]:
    by_assets: dict[tuple[str, ...], PairRecord] = {}
    for candidate in records:
        if not _grids_match(query, candidate):
            continue
        source_date = _acquired_date(candidate, source_kind)
        if source_date > query_date or (query_date - source_date).days > config.horizon_days:
            continue
        assets = _asset_paths(candidate, source_kind, root=root, include_mask=True)
        if tuple(assets) == tuple(source_anchor_assets):
            continue
        if query_target_assets.intersection(assets):
            continue
        current = by_assets.get(assets)
        if current is None or (_acquired_date(current, source_kind), current.pair_id) < (
            source_date,
            candidate.pair_id,
        ):
            by_assets[assets] = candidate
    selected = sorted(
        by_assets.values(),
        key=lambda record: (_acquired_date(record, source_kind), record.pair_id),
    )
    if config.max_observations is not None:
        selected = selected[-config.max_observations :]
    return tuple(selected)


def _coerce_index(
    index: str | Path | PairedTemporalIndex | Sequence[PairedTemporalSample],
) -> PairedTemporalIndex:
    if isinstance(index, PairedTemporalIndex):
        return index
    if isinstance(index, (str, Path)):
        return load_paired_temporal_index(index)
    samples = tuple(index)
    directions = {sample.direction for sample in samples}
    if len(directions) != 1:
        raise ValueError("PairedTemporalRasterDataset requires a non-empty single-direction index")
    return PairedTemporalIndex(
        config=PairedTemporalIndexConfig(direction=next(iter(directions))),
        samples=tuple(sorted(samples, key=_sample_sort_key)),
    )


def _coerce_records(
    records: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
) -> tuple[tuple[PairRecord, ...], Path | None]:
    if isinstance(records, (str, Path)):
        path = Path(_lexical_path(records))
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


def _config_to_dict(config: PairedTemporalIndexConfig) -> dict[str, object]:
    return {
        "direction": config.direction,
        "min_observations": config.min_observations,
        "max_observations": config.max_observations,
        "horizon_days": config.horizon_days,
        "anchor_max_delta_days": config.anchor_max_delta_days,
        "max_anchors_per_query": config.max_anchors_per_query,
        "translation_max_delta_days": config.translation_max_delta_days,
        "split": config.split,
        "orbit": config.orbit,
        "task_modes": list(config.task_modes),
        "max_samples": config.max_samples,
    }


def _config_dict_from_values(values: Mapping[str, object]) -> dict[str, object]:
    return _config_to_dict(_config_from_dict(values))


def _config_from_dict(values: Mapping[str, object]) -> PairedTemporalIndexConfig:
    try:
        return PairedTemporalIndexConfig(
            direction=str(values["direction"]),  # type: ignore[arg-type]
            min_observations=int(values["min_observations"]),
            max_observations=_optional_int(values.get("max_observations")),
            horizon_days=int(values["horizon_days"]),
            anchor_max_delta_days=int(values["anchor_max_delta_days"]),
            max_anchors_per_query=int(values.get("max_anchors_per_query", 1)),
            translation_max_delta_days=int(values.get("translation_max_delta_days", 1)),
            split=_optional_string(values.get("split")),
            orbit=_optional_string(values.get("orbit")),
            task_modes=tuple(str(value) for value in values["task_modes"]),  # type: ignore[arg-type]
            max_samples=_optional_int(values.get("max_samples")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid paired temporal-index index_config") from error


def _normalize_task_modes(task_modes: Iterable[TaskMode]) -> tuple[TaskMode, ...]:
    normalized = tuple(dict.fromkeys(str(task_mode) for task_mode in task_modes))
    if not normalized:
        raise ValueError("task_modes must be non-empty")
    invalid = tuple(task_mode for task_mode in normalized if task_mode not in ALL_TASK_MODES)
    if invalid:
        raise ValueError(f"unsupported paired temporal task modes: {invalid}")
    return tuple(normalized)  # type: ignore[return-value]


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


def _isolation_key(record: PairRecord) -> tuple[str, str, str]:
    """Return the leakage boundary while permitting causal cross-year history.

    Calendar year is metadata, not an isolation boundary.  The fixed dataset
    protocol already assigns years to disjoint splits; retaining ``split`` in
    this key prevents a 2022 train sequence from reaching a 2023 validation
    target while allowing a valid December-to-January train history.
    """

    return record.split, record.tile, record.orbit


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


def _asset_paths_cached(
    record: PairRecord,
    modality: Modality,
    *,
    root: Path,
    identity_cache: dict[str, str],
) -> tuple[str, ...]:
    if modality == "optical":
        values = (*[record.s2[channel] for channel in S2_CHANNEL_ORDER], record.scl)
    else:
        values = tuple(record.sar[channel] for channel in SAR_CHANNEL_ORDER)
    identities: list[str] = []
    for value in values:
        identity = identity_cache.get(value)
        if identity is None:
            identity = _asset_identity(value, root)
            identity_cache[value] = identity
        identities.append(identity)
    return tuple(identities)


def _asset_identity(value: str, root: Path) -> str:
    """Return a stable lexical identity without stat'ing a remote TIFF tree.

    Index construction needs path equality only.  ``Path.resolve(strict=False)``
    still walks ancestors and can issue network filesystem lookups for every
    comparison; the dataset performs its own real-path handling when it opens
    rasters.  ``abspath/normpath`` preserves the prior identity for ordinary
    absolute and manifest-relative paths without touching the filesystem.
    """

    expanded = os.path.expanduser(value)
    if not os.path.isabs(expanded):
        expanded = os.path.join(os.fspath(root), expanded)
    return _lexical_path(expanded)


def _resolve_asset_root(asset_root: str | Path | None, inferred_root: Path | None) -> Path:
    if asset_root is not None:
        return Path(_lexical_path(asset_root))
    if inferred_root is not None:
        return Path(_lexical_path(inferred_root))
    return Path(_lexical_path(Path.cwd()))


def _lexical_path(value: str | Path) -> str:
    """Normalize a manifest path without following symlinks or stat'ing it."""

    return os.path.abspath(os.path.normpath(os.path.expanduser(os.fspath(value))))


def _sample_sort_key(sample: PairedTemporalSample) -> tuple[str, str, str, str]:
    return sample.direction, sample.query_date, sample.query_pair_id, sample.sample_id


def _paired_sample_id(
    direction: Direction,
    query: PairRecord,
    anchor: PairRecord,
    observations: Sequence[PairRecord],
) -> str:
    observation_ids = ",".join(record.pair_id for record in observations)
    return (
        f"{direction}:{query.pair_id}:anchor={anchor.pair_id}:"
        f"observations={observation_ids}"
    )


def _task_mode_for_observations(
    query_date: date,
    observation_dates: Sequence[date],
    *,
    translation_max_delta_days: int,
) -> TaskMode:
    """Classify from the final chronological source observation.

    A one-day sensor-pair tolerance is intentional: a source acquisition from
    ``query_date - 1`` may still be a translation input for the target query
    pair.  ``observation_days`` always retains that actual ``-1`` offset.
    """

    if not observation_dates:
        raise ValueError("paired temporal samples need at least one observation")
    final_delta = (query_date - observation_dates[-1]).days
    return TRANSLATION if final_delta <= translation_max_delta_days else FORECAST


def _causal_failure(sample: PairedTemporalSample, message: str) -> None:
    raise AssertionError(f"paired temporal causality failure for {sample.sample_id}: {message}")


def _crop_seed(sample_id: str, seed: int, *, epoch: int = 0) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _require_tensor(value: object, key: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"paired temporal {key} must be a torch.Tensor")
    return value


def _validate_observation_tensors(
    values: torch.Tensor,
    valid: torch.Tensor,
    days: torch.Tensor,
    present: torch.Tensor,
) -> None:
    if values.ndim != 4 or valid.ndim != 4 or days.ndim != 1 or present.ndim != 1:
        raise ValueError("paired temporal observation tensors have invalid ranks")
    count = values.shape[0]
    if valid.shape[0] != count or days.shape[0] != count or present.shape[0] != count:
        raise ValueError("paired temporal observation tensors have inconsistent lengths")
    if valid.shape[1:] != (1, *values.shape[2:]):
        raise ValueError("paired temporal observation valid shape does not match values")


def _pad_observation_tensor(values: torch.Tensor, length: int) -> torch.Tensor:
    if values.shape[0] > length:
        raise ValueError("paired temporal observation tensor exceeds the requested batch length")
    if values.shape[0] == length:
        return values
    padding = values.new_zeros((length - values.shape[0], *values.shape[1:]))
    return torch.cat((values, padding), dim=0)


def _require_window_shape(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if values.shape != shape:
        raise RuntimeError(f"raster window has shape {values.shape}, expected {shape}")
    return values


def _clone_item(item: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in item.items()
    }


def _validate_slice(*, start: int, limit: int | None) -> None:
    if start < 0:
        raise ValueError("start must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when supplied")


def _slice_values(values: Sequence[Any], *, start: int, limit: int | None) -> list[Any]:
    _validate_slice(start=start, limit=limit)
    stop = None if limit is None else start + limit
    return list(values[start:stop])


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
