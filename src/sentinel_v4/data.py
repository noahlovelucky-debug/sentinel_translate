"""SOPAT V4 date-precision causal data contract and bidirectional batching.

V4 deliberately records the ambiguity of the available Sentinel manifest:
timestamps have date precision only.  The index therefore guarantees causal
ordering at date granularity, never an unsupported within-day ordering claim.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from sentinel_v3.dataset_builder import PairRecord, file_sha256
from sentinel_v3.paired_temporal_data import (
    ALL_DIRECTIONS,
    ALL_TASK_MODES,
    Direction,
    Modality,
    PairedTemporalIndex,
    PairedTemporalIndexConfig,
    PairedTemporalSample,
    TaskMode,
    assert_paired_temporal_causality,
    build_paired_temporal_index,
    collate_paired_temporal,
    load_pair_records,
    source_modality,
    target_modality,
)
from sentinel_v3.schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER

SOPAT_INDEX_FORMAT_VERSION = 1
SOPAT_TIME_PRECISION: Literal["date"] = "date"
SOPAT_CANONICAL_GSD_METERS = 10.0
SOPAT_NORMALIZATION_VERSION = "paired_temporal_v3_normalized"
SOPAT_CANONICALIZATION_VERSION = "canonical_10m_v1"
ObservationRole = Literal["history", "query_source"]


def sensor_schema_hash() -> str:
    """Stable hash of the currently supported Sentinel channel contract."""

    return _payload_hash(
        {
            "format_version": SOPAT_INDEX_FORMAT_VERSION,
            "optical": list(S2_CHANNEL_ORDER),
            "sar": list(SAR_CHANNEL_ORDER),
            "canonical_gsd_m": SOPAT_CANONICAL_GSD_METERS,
        }
    )


@dataclass(frozen=True)
class SOPATCanonicalGridV4:
    """Canonical target grid shared by all references in one example."""

    grid_id: str
    tile: str
    width: int
    height: int
    crs: str
    transform: tuple[float, ...]
    gsd_m: float

    def __post_init__(self) -> None:
        if not self.grid_id or not self.tile or self.width <= 0 or self.height <= 0:
            raise ValueError("canonical grid requires non-empty identity and positive dimensions")
        if len(self.transform) != 6:
            raise ValueError("canonical grid transform must contain six values")
        if abs(float(self.gsd_m) - SOPAT_CANONICAL_GSD_METERS) > 1e-9:
            raise ValueError("SOPAT V4 currently requires a canonical 10m grid")

    def to_dict(self) -> dict[str, object]:
        return {
            "grid_id": self.grid_id,
            "tile": self.tile,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "transform": list(self.transform),
            "gsd_m": self.gsd_m,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> SOPATCanonicalGridV4:
        return cls(
            grid_id=str(values["grid_id"]),
            tile=str(values["tile"]),
            width=int(values["width"]),
            height=int(values["height"]),
            crs=str(values["crs"]),
            transform=tuple(float(value) for value in _required_sequence(values, "transform")),
            gsd_m=float(values["gsd_m"]),
        )


@dataclass(frozen=True)
class SOPATMeasurementRefV4:
    """One sensor acquisition as a date-precision, grid-bound reference."""

    measurement_id: str
    record_id: str
    modality: Modality
    sensor_id: str
    time: str
    canonical_grid_id: str
    asset_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.measurement_id or not self.record_id or not self.sensor_id:
            raise ValueError("measurement reference identifiers must be non-empty")
        if self.modality not in {"optical", "sar"}:
            raise ValueError(f"unsupported measurement modality: {self.modality!r}")
        if not self.canonical_grid_id or not self.asset_ids:
            raise ValueError("measurement reference needs a grid and physical asset identities")
        if len(self.asset_ids) != len(set(self.asset_ids)):
            raise ValueError("measurement reference assets must be unique")
        date.fromisoformat(self.time)
        object.__setattr__(self, "asset_ids", tuple(str(value) for value in self.asset_ids))

    def to_dict(self) -> dict[str, object]:
        return {
            "measurement_id": self.measurement_id,
            "record_id": self.record_id,
            "modality": self.modality,
            "sensor_id": self.sensor_id,
            "time": self.time,
            "canonical_grid_id": self.canonical_grid_id,
            "asset_ids": list(self.asset_ids),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> SOPATMeasurementRefV4:
        return cls(
            measurement_id=str(values["measurement_id"]),
            record_id=str(values["record_id"]),
            modality=str(values["modality"]),  # type: ignore[arg-type]
            sensor_id=str(values["sensor_id"]),
            time=str(values["time"]),
            canonical_grid_id=str(values["canonical_grid_id"]),
            asset_ids=tuple(str(value) for value in _required_sequence(values, "asset_ids")),
        )


@dataclass(frozen=True)
class SOPATAnchorPairV4:
    """A registered historical source/target pair, explicit on both sides."""

    registration_id: str
    source: SOPATMeasurementRefV4
    target: SOPATMeasurementRefV4

    def __post_init__(self) -> None:
        if not self.registration_id:
            raise ValueError("anchor registration_id must be non-empty")
        if (
            self.registration_id != self.source.record_id
            or self.registration_id != self.target.record_id
        ):
            raise ValueError("SOPAT V4 anchor registration must name one paired record")
        if self.source.canonical_grid_id != self.target.canonical_grid_id:
            raise ValueError("registered anchor measurements must share a canonical grid")

    def to_dict(self) -> dict[str, object]:
        return {
            "registration_id": self.registration_id,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> SOPATAnchorPairV4:
        return cls(
            registration_id=str(values["registration_id"]),
            source=SOPATMeasurementRefV4.from_dict(_required_mapping(values, "source")),
            target=SOPATMeasurementRefV4.from_dict(_required_mapping(values, "target")),
        )


@dataclass(frozen=True)
class SOPATObservationRefV4:
    """One source observation in an unordered causal observation set."""

    measurement: SOPATMeasurementRefV4
    role: ObservationRole = "history"

    def __post_init__(self) -> None:
        if self.role not in {"history", "query_source"}:
            raise ValueError(f"unsupported observation role: {self.role!r}")

    def to_dict(self) -> dict[str, object]:
        return {"measurement": self.measurement.to_dict(), "role": self.role}

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> SOPATObservationRefV4:
        return cls(
            measurement=SOPATMeasurementRefV4.from_dict(_required_mapping(values, "measurement")),
            role=str(values.get("role", "history")),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SOPATProvenanceV4:
    """Protocol facts needed to interpret a V4 sample without hidden defaults."""

    time_precision: Literal["date"] = SOPAT_TIME_PRECISION
    source_protocol_hash: str = ""
    source_manifest_sha256: str = ""
    sensor_schema_hash: str = ""
    normalization_version: str = SOPAT_NORMALIZATION_VERSION
    canonicalization_version: str = SOPAT_CANONICALIZATION_VERSION

    def __post_init__(self) -> None:
        if self.time_precision != SOPAT_TIME_PRECISION:
            raise ValueError("available V4 Sentinel records have date precision only")
        if not self.source_protocol_hash or not self.sensor_schema_hash:
            raise ValueError("V4 provenance requires source protocol and sensor schema hashes")
        if self.sensor_schema_hash != sensor_schema_hash():
            raise ValueError("V4 provenance sensor schema hash does not match the active schema")
        if not self.normalization_version or not self.canonicalization_version:
            raise ValueError("V4 provenance versions must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> SOPATProvenanceV4:
        return cls(
            time_precision=str(values["time_precision"]),  # type: ignore[arg-type]
            source_protocol_hash=str(values["source_protocol_hash"]),
            source_manifest_sha256=str(values.get("source_manifest_sha256", "")),
            sensor_schema_hash=str(values["sensor_schema_hash"]),
            normalization_version=str(values["normalization_version"]),
            canonicalization_version=str(values["canonicalization_version"]),
        )


@dataclass(frozen=True)
class SOPATIndexConfigV4:
    """Global date-level constraints used for every V4 example in one index."""

    horizon_days: int = 180
    anchor_max_delta_days: int = 1
    translation_tolerance_days: int = 1
    canonical_gsd_m: float = SOPAT_CANONICAL_GSD_METERS
    time_precision: Literal["date"] = SOPAT_TIME_PRECISION

    def __post_init__(self) -> None:
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if not 0 <= self.anchor_max_delta_days <= 1:
            raise ValueError("anchor_max_delta_days must be in [0, 1] for date precision")
        if not 0 <= self.translation_tolerance_days <= 1:
            raise ValueError("translation_tolerance_days must be in [0, 1] for date precision")
        if abs(float(self.canonical_gsd_m) - SOPAT_CANONICAL_GSD_METERS) > 1e-9:
            raise ValueError("SOPAT V4 currently supports canonical 10m only")
        if self.time_precision != SOPAT_TIME_PRECISION:
            raise ValueError("SOPAT V4 currently supports date precision only")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> SOPATIndexConfigV4:
        return cls(
            horizon_days=int(values["horizon_days"]),
            anchor_max_delta_days=int(values["anchor_max_delta_days"]),
            translation_tolerance_days=int(values["translation_tolerance_days"]),
            canonical_gsd_m=float(values["canonical_gsd_m"]),
            time_precision=str(values["time_precision"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SOPATExampleV4:
    """One bidirectional SOPAT example with explicit source-query semantics."""

    sample_id: str
    direction: Direction
    task_mode: TaskMode
    split: str
    tile: str
    year: int
    orbit: str
    target: SOPATMeasurementRefV4
    anchor_pair: SOPATAnchorPairV4
    observations: tuple[SOPATObservationRefV4, ...]
    query_source_id: str | None
    canonical_grid: SOPATCanonicalGridV4
    provenance: SOPATProvenanceV4

    def __post_init__(self) -> None:
        if not self.sample_id or not self.split or not self.tile or not self.orbit:
            raise ValueError("SOPAT example identity fields must be non-empty")
        if self.direction not in ALL_DIRECTIONS or self.task_mode not in ALL_TASK_MODES:
            raise ValueError("SOPAT example direction or task mode is unsupported")
        if self.canonical_grid.tile != self.tile:
            raise ValueError("SOPAT example tile differs from its canonical grid")
        source = source_modality(self.direction)
        target = target_modality(self.direction)
        if self.target.modality != target:
            raise ValueError("target reference does not match the direction target modality")
        if self.anchor_pair.source.modality != source or self.anchor_pair.target.modality != target:
            raise ValueError("anchor pair modalities do not match the direction")
        references = (self.target, self.anchor_pair.source, self.anchor_pair.target)
        if any(reference.canonical_grid_id != self.canonical_grid.grid_id for reference in references):
            raise ValueError("SOPAT example references do not share the canonical grid")
        observations = tuple(sorted(self.observations, key=_observation_serialization_key))
        if not observations:
            raise ValueError("SOPAT example needs at least one source observation")
        if any(observation.measurement.modality != source for observation in observations):
            raise ValueError("SOPAT observations do not match the source modality")
        if any(
            observation.measurement.canonical_grid_id != self.canonical_grid.grid_id
            for observation in observations
        ):
            raise ValueError("SOPAT observations do not share the canonical grid")
        ids = tuple(observation.measurement.measurement_id for observation in observations)
        if len(ids) != len(set(ids)):
            raise ValueError("SOPAT observations must not repeat a physical measurement")
        object.__setattr__(self, "observations", observations)

    @property
    def source_modality(self) -> Modality:
        return source_modality(self.direction)

    @property
    def target_modality(self) -> Modality:
        return target_modality(self.direction)

    @property
    def target_time(self) -> str:
        return self.target.time

    @property
    def query_source(self) -> SOPATObservationRefV4 | None:
        if self.query_source_id is None:
            return None
        return next(
            (
                observation
                for observation in self.observations
                if observation.measurement.measurement_id == self.query_source_id
            ),
            None,
        )

    def observation_ordered_for_backend(self) -> tuple[SOPATObservationRefV4, ...]:
        """Stable chronological projection used only by the V3 raster backend."""

        return tuple(sorted(self.observations, key=_observation_backend_key))

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": SOPAT_INDEX_FORMAT_VERSION,
            "sample_id": self.sample_id,
            "direction": self.direction,
            "task_mode": self.task_mode,
            "split": self.split,
            "tile": self.tile,
            "year": self.year,
            "orbit": self.orbit,
            # Keep the forecast target time visible at the top level.  It is
            # redundant with target.time by design, and the loader verifies
            # that the two cannot silently disagree.
            "target_time": self.target_time,
            "target": self.target.to_dict(),
            "anchor_pair": self.anchor_pair.to_dict(),
            "observations": [observation.to_dict() for observation in self.observations],
            "query_source_id": self.query_source_id,
            "canonical_grid": self.canonical_grid.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> SOPATExampleV4:
        format_version = int(values.get("format_version", SOPAT_INDEX_FORMAT_VERSION))
        if format_version != SOPAT_INDEX_FORMAT_VERSION:
            raise ValueError(f"unsupported SOPAT index format: {format_version}")
        query_source_id = values.get("query_source_id")
        if query_source_id is not None and not isinstance(query_source_id, str):
            raise TypeError("query_source_id must be a string or null")
        target_time = values.get("target_time")
        if not isinstance(target_time, str):
            raise TypeError("target_time must be an ISO date string")
        result = cls(
            sample_id=str(values["sample_id"]),
            direction=str(values["direction"]),  # type: ignore[arg-type]
            task_mode=str(values["task_mode"]),  # type: ignore[arg-type]
            split=str(values["split"]),
            tile=str(values["tile"]),
            year=int(values["year"]),
            orbit=str(values["orbit"]),
            target=SOPATMeasurementRefV4.from_dict(_required_mapping(values, "target")),
            anchor_pair=SOPATAnchorPairV4.from_dict(_required_mapping(values, "anchor_pair")),
            observations=tuple(
                SOPATObservationRefV4.from_dict(value)
                for value in _required_mapping_sequence(values, "observations")
            ),
            query_source_id=query_source_id,
            canonical_grid=SOPATCanonicalGridV4.from_dict(
                _required_mapping(values, "canonical_grid")
            ),
            provenance=SOPATProvenanceV4.from_dict(_required_mapping(values, "provenance")),
        )
        if result.target_time != target_time:
            raise ValueError("SOPAT target_time differs from target.time")
        return result


@dataclass(frozen=True)
class SOPATIndexV4:
    """A single serializable V4 index that may contain both directions."""

    config: SOPATIndexConfigV4
    examples: tuple[SOPATExampleV4, ...]

    def __post_init__(self) -> None:
        examples = tuple(sorted(self.examples, key=_example_sort_key))
        seen: set[str] = set()
        for example in examples:
            if example.sample_id in seen:
                raise ValueError(f"SOPAT index contains duplicate sample_id: {example.sample_id}")
            seen.add(example.sample_id)
            if example.provenance.time_precision != self.config.time_precision:
                raise ValueError("SOPAT example time precision differs from its index")
            if abs(example.canonical_grid.gsd_m - self.config.canonical_gsd_m) > 1e-9:
                raise ValueError("SOPAT example grid differs from its index canonical GSD")
            _assert_example_causality(example, self.config)
        object.__setattr__(self, "examples", examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[SOPATExampleV4]:
        return iter(self.examples)

    @property
    def protocol_hash(self) -> str:
        """Hash the global selection protocol, excluding selected examples."""

        return _payload_hash(self.config.to_dict())

    @property
    def content_hash(self) -> str:
        """Hash the complete canonical index payload for checkpoint binding.

        ``protocol_hash`` answers whether two indexes use the same selection
        rules.  This hash additionally commits to every chosen sample,
        reference, role, grid, and provenance field.
        """

        return _payload_hash(
            {
                "format_version": SOPAT_INDEX_FORMAT_VERSION,
                "index_config": self.config.to_dict(),
                "examples": [example.to_dict() for example in self.examples],
            }
        )

    def select(self, *, direction: Direction, split: str) -> tuple[SOPATExampleV4, ...]:
        return tuple(
            example
            for example in self.examples
            if example.direction == direction and example.split == split
        )


def assert_sopat_v4_causality(
    index: SOPATIndexV4 | Sequence[SOPATExampleV4],
    *,
    config: SOPATIndexConfigV4 | None = None,
) -> None:
    """Validate date-level causal, anchor, role, and target-leakage rules."""

    if isinstance(index, SOPATIndexV4):
        resolved_config = index.config
        examples = index.examples
    else:
        if config is None:
            raise ValueError("config is required when validating bare SOPAT examples")
        resolved_config = config
        examples = tuple(index)
    seen: set[str] = set()
    for example in examples:
        if example.sample_id in seen:
            raise AssertionError(f"duplicate SOPAT sample_id: {example.sample_id}")
        seen.add(example.sample_id)
        _assert_example_causality(example, resolved_config)


def migrate_paired_temporal_index_v4(
    paired_index: PairedTemporalIndex,
    records: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
    *,
    provenance: SOPATProvenanceV4 | None = None,
    asset_root: str | Path | None = None,
) -> SOPATIndexV4:
    """Migrate a validated V3 one-direction index into explicit V4 roles."""

    record_map, manifest_path = _coerce_records(records)
    root = _resolve_asset_root(manifest_path, asset_root)
    assert_paired_temporal_causality(
        paired_index,
        record_map,
        asset_root=root,
    )
    resolved_provenance = provenance or _migration_provenance(paired_index, manifest_path)
    config = SOPATIndexConfigV4(
        horizon_days=paired_index.config.horizon_days,
        anchor_max_delta_days=paired_index.config.anchor_max_delta_days,
        translation_tolerance_days=paired_index.config.translation_max_delta_days,
    )
    examples = tuple(
        _migrate_sample(sample, record_map, root=root, provenance=resolved_provenance)
        for sample in paired_index
    )
    result = SOPATIndexV4(config=config, examples=examples)
    assert_sopat_v4_causality(result)
    return result


def build_sopat_v4_index(
    records: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
    *,
    splits: Sequence[str],
    directions: Sequence[Direction] = ALL_DIRECTIONS,
    min_observations: int = 1,
    max_observations: int | None = None,
    horizon_days: int = 180,
    anchor_max_delta_days: int = 1,
    max_anchors_per_query: int = 1,
    translation_tolerance_days: int = 1,
    orbit: str | None = "ascending",
    asset_root: str | Path | None = None,
) -> SOPATIndexV4:
    """Build one V4 dual-direction index from the existing V3 selector."""

    if not splits:
        raise ValueError("at least one split is required")
    record_map, manifest_path = _coerce_records(records)
    resolved_records = tuple(record_map.values())
    root = _resolve_asset_root(manifest_path, asset_root)
    config = SOPATIndexConfigV4(
        horizon_days=horizon_days,
        anchor_max_delta_days=anchor_max_delta_days,
        translation_tolerance_days=translation_tolerance_days,
    )
    all_examples: list[SOPATExampleV4] = []
    for direction in directions:
        for split in splits:
            paired = build_paired_temporal_index(
                resolved_records,
                direction=direction,
                min_observations=min_observations,
                max_observations=max_observations,
                horizon_days=horizon_days,
                anchor_max_delta_days=anchor_max_delta_days,
                max_anchors_per_query=max_anchors_per_query,
                translation_max_delta_days=translation_tolerance_days,
                split=split,
                orbit=orbit,
                task_modes=ALL_TASK_MODES,
                asset_root=root,
            )
            migrated = migrate_paired_temporal_index_v4(
                paired,
                records if manifest_path is not None else record_map,
                asset_root=root,
            )
            all_examples.extend(migrated.examples)
    result = SOPATIndexV4(config=config, examples=tuple(all_examples))
    assert_sopat_v4_causality(result)
    return result


def write_sopat_v4_index(path: str | Path, index: SOPATIndexV4) -> str:
    """Atomically write a stable V4 JSONL index and return its file SHA-256."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        metadata = {
            "record_type": "sopat_v4_index_metadata",
            "format_version": SOPAT_INDEX_FORMAT_VERSION,
            "index_config": index.config.to_dict(),
            "index_content_sha256": index.content_hash,
        }
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        for example in index.examples:
            row = example.to_dict()
            row["index_config"] = index.config.to_dict()
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return file_sha256(destination)


def load_sopat_v4_index(path: str | Path) -> SOPATIndexV4:
    """Load a V4 index and reject mixed formats or protocol configurations."""

    examples: list[SOPATExampleV4] = []
    index_config: SOPATIndexConfigV4 | None = None
    declared_content_hash: str | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid SOPAT V4 JSON on line {line_number}: {path}") from error
            if not isinstance(values, Mapping):
                raise TypeError(f"SOPAT V4 row {line_number} is not a mapping")
            if int(values.get("format_version", -1)) != SOPAT_INDEX_FORMAT_VERSION:
                raise ValueError(f"unsupported SOPAT V4 format on line {line_number}")
            raw_config = _required_mapping(values, "index_config")
            candidate = SOPATIndexConfigV4.from_dict(raw_config)
            if index_config is None:
                index_config = candidate
            elif index_config != candidate:
                raise ValueError("SOPAT V4 index rows have inconsistent index_config")
            declared = values.get("index_content_sha256")
            if declared is not None:
                if not isinstance(declared, str) or not declared:
                    raise ValueError("SOPAT V4 index has invalid index_content_sha256")
                if declared_content_hash is None:
                    declared_content_hash = declared
                elif declared_content_hash != declared:
                    raise ValueError("SOPAT V4 index rows have inconsistent content hashes")
            if values.get("record_type") == "sopat_v4_index_metadata":
                continue
            payload = dict(values)
            payload.pop("index_config", None)
            examples.append(SOPATExampleV4.from_dict(payload))
    if index_config is None:
        raise ValueError("SOPAT V4 index does not declare index_config")
    if declared_content_hash is None:
        raise ValueError("SOPAT V4 index is missing its required content hash metadata")
    result = SOPATIndexV4(config=index_config, examples=tuple(examples))
    if declared_content_hash is not None and declared_content_hash != result.content_hash:
        raise ValueError("SOPAT V4 index content hash does not match its rows")
    assert_sopat_v4_causality(result)
    return result


def sopat_v4_index_file_sha256(path: str | Path) -> str:
    """Return the exact persisted index digest for checkpoint metadata."""

    return file_sha256(Path(path))


def paired_temporal_index_from_sopat_v4(
    index: SOPATIndexV4,
    *,
    direction: Direction,
    split: str,
) -> PairedTemporalIndex:
    """Project one V4 direction/split onto the legacy raster reader contract."""

    examples = index.select(direction=direction, split=split)
    if not examples:
        raise ValueError(f"SOPAT V4 has no examples for {direction}/{split}")
    maximum_observations = max(len(example.observations) for example in examples)
    config = PairedTemporalIndexConfig(
        direction=direction,
        min_observations=1,
        max_observations=maximum_observations,
        horizon_days=index.config.horizon_days,
        anchor_max_delta_days=index.config.anchor_max_delta_days,
        max_anchors_per_query=1,
        translation_max_delta_days=index.config.translation_tolerance_days,
        split=split,
        orbit=examples[0].orbit,
        task_modes=ALL_TASK_MODES,
    )
    if any(example.orbit != config.orbit for example in examples):
        raise ValueError("legacy V3 projection requires one orbit per direction/split")
    samples: list[PairedTemporalSample] = []
    for example in examples:
        observations = example.observation_ordered_for_backend()
        samples.append(
            PairedTemporalSample(
                sample_id=example.sample_id,
                direction=example.direction,
                task_mode=example.task_mode,
                split=example.split,
                tile=example.tile,
                year=example.year,
                orbit=example.orbit,
                query_pair_id=example.target.record_id,
                anchor_pair_id=example.anchor_pair.registration_id,
                observation_pair_ids=tuple(
                    observation.measurement.record_id for observation in observations
                ),
                query_date=example.target.time,
                source_anchor_date=example.anchor_pair.source.time,
                target_anchor_date=example.anchor_pair.target.time,
                observation_dates=tuple(observation.measurement.time for observation in observations),
            )
        )
    return PairedTemporalIndex(config=config, samples=tuple(samples))


class SOPATDirectionDataset(Dataset[dict[str, object]]):
    """Wrap a direction-homogeneous V3 raster or chunk backend with V4 roles.

    The backend owns pixel decoding.  V4 only maps selected examples to it and
    performs a deterministic set permutation over real observation slots.  No
    source/target channel padding is performed across directions.
    """

    def __init__(
        self,
        index: SOPATIndexV4,
        backend: Dataset[dict[str, object]],
        *,
        direction: Direction,
        split: str,
        permutation_seed: int = 0,
        permute_observations: bool = True,
    ) -> None:
        examples = index.select(direction=direction, split=split)
        if not examples:
            raise ValueError(f"SOPAT V4 has no examples for {direction}/{split}")
        backend_samples = getattr(backend, "samples", None)
        if not isinstance(backend_samples, Sequence):
            raise TypeError("SOPAT V4 backend must expose its paired temporal samples")
        by_sample_id = {str(sample.sample_id): position for position, sample in enumerate(backend_samples)}
        if len(by_sample_id) != len(backend_samples):
            raise ValueError("SOPAT V4 backend has duplicate sample_id values")
        missing = [example.sample_id for example in examples if example.sample_id not in by_sample_id]
        if missing:
            raise ValueError(f"SOPAT V4 backend is missing sample {missing[0]}")
        expected_samples = paired_temporal_index_from_sopat_v4(
            index,
            direction=direction,
            split=split,
        )
        expected_by_id = {sample.sample_id: sample for sample in expected_samples.samples}
        for example in examples:
            backend_sample = backend_samples[by_sample_id[example.sample_id]]
            if backend_sample != expected_by_id[example.sample_id]:
                raise ValueError(
                    "SOPAT V4 backend sample disagrees with the routed V4 causal example: "
                    f"{example.sample_id}"
                )
        backend_windows = int(getattr(backend, "windows_per_sample", 1))
        if backend_windows <= 0:
            raise ValueError("SOPAT V4 backend windows_per_sample must be positive")
        expected_backend_length = len(backend_samples) * backend_windows
        if len(backend) != expected_backend_length:
            raise ValueError(
                "SOPAT V4 backend must use contiguous sample-times-window indexing"
            )
        self.index = index
        self.backend = backend
        self.examples = examples
        self.direction = direction
        self.split = split
        self.permutation_seed = int(permutation_seed)
        self.permute_observations = bool(permute_observations)
        self.epoch = 0
        self.backend_windows = backend_windows
        self._backend_positions = tuple(by_sample_id[example.sample_id] for example in examples)
        self._backend_observation_ids = {
            example.sample_id: tuple(
                observation.measurement.measurement_id
                for observation in example.observation_ordered_for_backend()
            )
            for example in examples
        }
        self._backend_observation_roles = {
            example.sample_id: tuple(
                observation.role for observation in example.observation_ordered_for_backend()
            )
            for example in examples
        }

    @classmethod
    def from_raster(
        cls,
        index: SOPATIndexV4,
        records: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord],
        *,
        direction: Direction,
        split: str,
        permutation_seed: int = 0,
        permute_observations: bool = True,
        **dataset_kwargs: object,
    ) -> SOPATDirectionDataset:
        """Construct a V4 wrapper around ``PairedTemporalRasterDataset``."""

        from sentinel_v3.paired_temporal_data import PairedTemporalRasterDataset

        legacy = paired_temporal_index_from_sopat_v4(index, direction=direction, split=split)
        backend = PairedTemporalRasterDataset(records, legacy, **dataset_kwargs)
        return cls(
            index,
            backend,
            direction=direction,
            split=split,
            permutation_seed=permutation_seed,
            permute_observations=permute_observations,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        setter = getattr(self.backend, "set_epoch", None)
        if callable(setter):
            setter(self.epoch)

    def __len__(self) -> int:
        return len(self.examples) * self.backend_windows

    def __getitem__(self, position: int) -> dict[str, object]:
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(position)
        example_position, window_position = divmod(position, self.backend_windows)
        example = self.examples[example_position]
        backend_position = self._backend_positions[example_position] * self.backend_windows + window_position
        item = dict(self.backend[backend_position])
        if item.get("direction") != self.direction or item.get("task_mode") != example.task_mode:
            raise RuntimeError("SOPAT V4 backend item disagrees with its routed example")
        order = self._observation_permutation(item, example, window_position)
        self._permute_observation_tensors(item, order)
        source_ids = self._backend_observation_ids[example.sample_id]
        source_roles = self._backend_observation_roles[example.sample_id]
        item.update(
            {
                "sopat_example_id": example.sample_id,
                "sopat_direction": self.direction,
                "sopat_grid_id": example.canonical_grid.grid_id,
                "sopat_time_precision": example.provenance.time_precision,
                "sopat_query_source_id": example.query_source_id,
                "sopat_observation_ids": tuple(source_ids[index] for index in order if index < len(source_ids)),
                "sopat_observation_roles": tuple(
                    source_roles[index] for index in order if index < len(source_roles)
                ),
            }
        )
        return item

    def _observation_permutation(
        self, item: Mapping[str, object], example: SOPATExampleV4, window: int
    ) -> tuple[int, ...]:
        present = item.get("observation_present")
        if not isinstance(present, torch.Tensor) or present.ndim != 1:
            raise TypeError("SOPAT V4 backend item lacks one-dimensional observation_present")
        count = int(present.bool().sum().item())
        if count != len(example.observations):
            raise RuntimeError("SOPAT V4 backend observation count disagrees with routed example")
        if not self.permute_observations or count < 2:
            return tuple(range(count))
        seed = _permutation_seed(self.permutation_seed, self.epoch, example.sample_id, window)
        generator = torch.Generator().manual_seed(seed)
        return tuple(int(value) for value in torch.randperm(count, generator=generator).tolist())

    @staticmethod
    def _permute_observation_tensors(item: dict[str, object], order: tuple[int, ...]) -> None:
        if not order:
            raise RuntimeError("SOPAT V4 cannot permute an empty observation set")
        present = item["observation_present"]
        if not isinstance(present, torch.Tensor):
            raise TypeError("observation_present must be a tensor")
        count = len(order)
        if not torch.equal(
            present.bool(),
            torch.arange(present.shape[0], device=present.device) < count,
        ):
            raise RuntimeError("SOPAT V4 backend must keep real observations before padding")
        full_order = (*order, *range(count, int(present.shape[0])))
        selector = torch.tensor(full_order, dtype=torch.long, device=present.device)
        for key in (
            "observation_values",
            "observation_valid",
            "observation_days",
            "observation_present",
        ):
            values = item.get(key)
            if not isinstance(values, torch.Tensor):
                raise TypeError(f"SOPAT V4 backend item lacks tensor {key}")
            item[key] = values.index_select(0, selector.to(values.device))


def collate_sopat_direction(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Collate one direction only, retaining the V3 core/training tensor keys."""

    if not samples:
        raise ValueError("collate_sopat_direction requires at least one sample")
    directions = {str(sample.get("sopat_direction", sample.get("direction", ""))) for sample in samples}
    if len(directions) != 1 or next(iter(directions)) not in ALL_DIRECTIONS:
        raise ValueError("SOPAT direction batches must be homogeneous")
    batch = collate_paired_temporal(samples)
    direction = next(iter(directions))
    batch["sopat_direction"] = direction
    for key in (
        "sopat_example_id",
        "sopat_grid_id",
        "sopat_time_precision",
        "sopat_query_source_id",
        "sopat_observation_ids",
        "sopat_observation_roles",
    ):
        batch[key] = [sample.get(key) for sample in samples]
    return batch


@dataclass(frozen=True)
class CoupledDirectionBatch:
    """One no-padding optimization step containing both homogeneous directions."""

    step: int
    sar_to_optical: Mapping[str, object]
    optical_to_sar: Mapping[str, object]

    def as_dict(self) -> dict[str, Mapping[str, object]]:
        return {
            "sar_to_optical": self.sar_to_optical,
            "optical_to_sar": self.optical_to_sar,
        }


class CoupledDirectionLoader(Iterable[CoupledDirectionBatch]):
    """Cycle the shorter directional loader so every step contains both tasks."""

    def __init__(self, loaders: Mapping[Direction, Iterable[Mapping[str, object]]]) -> None:
        if set(loaders) != set(ALL_DIRECTIONS):
            raise ValueError("CoupledDirectionLoader requires exactly both SOPAT directions")
        self.loaders = dict(loaders)
        self.epoch = 0
        self._lengths = {direction: _iterable_length(loader) for direction, loader in self.loaders.items()}
        if any(length <= 0 for length in self._lengths.values()):
            raise ValueError("CoupledDirectionLoader cannot cycle an empty loader")

    def __len__(self) -> int:
        return max(self._lengths.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        seen: set[int] = set()
        for loader in self.loaders.values():
            _set_epoch_if_supported(loader, self.epoch, seen)
            _set_epoch_if_supported(getattr(loader, "sampler", None), self.epoch, seen)
            batch_sampler = getattr(loader, "batch_sampler", None)
            _set_epoch_if_supported(batch_sampler, self.epoch, seen)
            _set_epoch_if_supported(getattr(batch_sampler, "sampler", None), self.epoch, seen)
            _set_epoch_if_supported(getattr(loader, "dataset", None), self.epoch, seen)

    def __iter__(self) -> Iterator[CoupledDirectionBatch]:
        iterators = {direction: iter(loader) for direction, loader in self.loaders.items()}
        for step in range(len(self)):
            batches: dict[Direction, Mapping[str, object]] = {}
            for direction in ALL_DIRECTIONS:
                try:
                    batch = next(iterators[direction])
                except StopIteration:
                    iterators[direction] = iter(self.loaders[direction])
                    try:
                        batch = next(iterators[direction])
                    except StopIteration as error:
                        raise RuntimeError("directional loader became empty while cycling") from error
                _assert_direction_batch(batch, direction)
                batches[direction] = batch
            yield CoupledDirectionBatch(
                step=step,
                sar_to_optical=batches["sar_to_optical"],
                optical_to_sar=batches["optical_to_sar"],
            )


def _migrate_sample(
    sample: PairedTemporalSample,
    records: Mapping[str, PairRecord],
    *,
    root: Path,
    provenance: SOPATProvenanceV4,
) -> SOPATExampleV4:
    try:
        query = records[sample.query_pair_id]
        anchor = records[sample.anchor_pair_id]
        observations = tuple(records[pair_id] for pair_id in sample.observation_pair_ids)
    except KeyError as error:
        raise ValueError(f"paired temporal sample references missing record: {error.args[0]}") from error
    direction = sample.direction
    source = source_modality(direction)
    target = target_modality(direction)
    grid = _canonical_grid(query)
    target_ref = _measurement_ref(query, target, grid, root)
    anchor_pair = SOPATAnchorPairV4(
        registration_id=anchor.pair_id,
        source=_measurement_ref(anchor, source, grid, root),
        target=_measurement_ref(anchor, target, grid, root),
    )
    ordered = tuple(sorted(observations, key=lambda record: (_record_time(record, source), record.pair_id)))
    query_source_id: str | None = None
    refs: list[SOPATObservationRefV4] = []
    if sample.task_mode == "translation":
        query_source_id = _measurement_id(ordered[-1], source)
    for record in ordered:
        measurement = _measurement_ref(record, source, grid, root)
        role: ObservationRole = "query_source" if measurement.measurement_id == query_source_id else "history"
        refs.append(SOPATObservationRefV4(measurement=measurement, role=role))
    return SOPATExampleV4(
        sample_id=sample.sample_id,
        direction=direction,
        task_mode=sample.task_mode,
        split=sample.split,
        tile=sample.tile,
        year=sample.year,
        orbit=sample.orbit,
        target=target_ref,
        anchor_pair=anchor_pair,
        observations=tuple(refs),
        query_source_id=query_source_id,
        canonical_grid=grid,
        provenance=provenance,
    )


def _assert_example_causality(example: SOPATExampleV4, config: SOPATIndexConfigV4) -> None:
    if example.provenance.time_precision != SOPAT_TIME_PRECISION:
        raise AssertionError(f"{example.sample_id}: V4 is only date-precision causal")
    if abs(example.canonical_grid.gsd_m - config.canonical_gsd_m) > 1e-9:
        raise AssertionError(f"{example.sample_id}: canonical GSD mismatch")
    target_day = _day(example.target.time)
    anchors = (example.anchor_pair.source, example.anchor_pair.target)
    for anchor in anchors:
        anchor_day = _day(anchor.time)
        if not anchor_day < target_day:
            raise AssertionError(f"{example.sample_id}: anchor must be strictly before target")
        if (target_day - anchor_day).days > config.horizon_days:
            raise AssertionError(f"{example.sample_id}: anchor exceeds horizon")
    if abs((_day(anchors[0].time) - _day(anchors[1].time)).days) > config.anchor_max_delta_days:
        raise AssertionError(f"{example.sample_id}: anchor pair exceeds registration tolerance")
    target_assets = set(example.target.asset_ids)
    input_assets = set(anchors[0].asset_ids).union(anchors[1].asset_ids)
    if target_assets.intersection(input_assets):
        raise AssertionError(f"{example.sample_id}: target asset leaks into anchor input")
    query_sources = [
        observation
        for observation in example.observations
        if observation.role == "query_source"
    ]
    seen_observation_assets: set[tuple[str, ...]] = set()
    for observation in example.observations:
        observation_day = _day(observation.measurement.time)
        if observation_day > target_day:
            raise AssertionError(f"{example.sample_id}: source observation is after target")
        if (target_day - observation_day).days > config.horizon_days:
            raise AssertionError(f"{example.sample_id}: source observation exceeds horizon")
        if target_assets.intersection(observation.measurement.asset_ids):
            raise AssertionError(f"{example.sample_id}: target asset leaks into observation input")
        if set(anchors[0].asset_ids).intersection(observation.measurement.asset_ids):
            raise AssertionError(f"{example.sample_id}: source anchor repeats as an observation")
        identity = tuple(observation.measurement.asset_ids)
        if identity in seen_observation_assets:
            raise AssertionError(f"{example.sample_id}: observations repeat a physical acquisition")
        seen_observation_assets.add(identity)
    if example.task_mode == "translation":
        if example.query_source_id is None or len(query_sources) != 1:
            raise AssertionError(f"{example.sample_id}: translation requires exactly one query source")
        query_source = query_sources[0]
        if query_source.measurement.measurement_id != example.query_source_id:
            raise AssertionError(f"{example.sample_id}: query source id does not match its role")
        age = (target_day - _day(query_source.measurement.time)).days
        if not 0 <= age <= config.translation_tolerance_days:
            raise AssertionError(f"{example.sample_id}: query source is outside translation tolerance")
        last = example.observation_ordered_for_backend()[-1]
        if last.measurement.measurement_id != example.query_source_id:
            raise AssertionError(
                f"{example.sample_id}: query source must be the final causal observation"
            )
    elif example.task_mode == "forecast":
        if example.query_source_id is not None or query_sources:
            raise AssertionError(f"{example.sample_id}: forecast cannot contain a query source")
        if any((target_day - _day(observation.measurement.time)).days <= 1 for observation in example.observations):
            raise AssertionError(f"{example.sample_id}: forecast contains a query-time source")
    else:
        raise AssertionError(f"{example.sample_id}: unsupported task mode")


def _canonical_grid(record: PairRecord) -> SOPATCanonicalGridV4:
    if abs(float(record.gsd) - SOPAT_CANONICAL_GSD_METERS) > 1e-9:
        raise ValueError(f"record {record.pair_id} is not on the canonical 10m grid")
    token = json.dumps(
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
    return SOPATCanonicalGridV4(
        grid_id="grid-" + hashlib.sha256(token.encode()).hexdigest()[:24],
        tile=record.tile,
        width=int(record.width),
        height=int(record.height),
        crs=str(record.crs),
        transform=tuple(float(value) for value in record.transform[:6]),
        gsd_m=float(record.gsd),
    )


def _measurement_ref(
    record: PairRecord, modality: Modality, grid: SOPATCanonicalGridV4, root: Path
) -> SOPATMeasurementRefV4:
    return SOPATMeasurementRefV4(
        measurement_id=_measurement_id(record, modality),
        record_id=record.pair_id,
        modality=modality,
        sensor_id="sentinel-2" if modality == "optical" else "sentinel-1",
        time=_record_time(record, modality).isoformat(),
        canonical_grid_id=grid.grid_id,
        asset_ids=_record_assets(record, modality, root),
    )


def _measurement_id(record: PairRecord, modality: Modality) -> str:
    return f"{record.pair_id}::{modality}"


def _record_time(record: PairRecord, modality: Modality) -> date:
    return _day(record.s2_date if modality == "optical" else record.s1_date)


def _record_assets(record: PairRecord, modality: Modality, root: Path) -> tuple[str, ...]:
    if modality == "optical":
        values = (*[record.s2[channel] for channel in S2_CHANNEL_ORDER], record.scl)
    else:
        values = tuple(record.sar[channel] for channel in SAR_CHANNEL_ORDER)
    return tuple(_asset_identity(value, root) for value in values)


def _migration_provenance(
    index: PairedTemporalIndex, manifest_path: Path | None
) -> SOPATProvenanceV4:
    source_hash = ""
    if manifest_path is not None and manifest_path.is_file():
        source_hash = file_sha256(manifest_path)
    return SOPATProvenanceV4(
        source_protocol_hash=_payload_hash(_paired_config_payload(index.config)),
        source_manifest_sha256=source_hash,
        sensor_schema_hash=sensor_schema_hash(),
    )


def paired_temporal_protocol_hash(config: PairedTemporalIndexConfig) -> str:
    """Expose the V3 selection-protocol digest used by V4 provenance."""

    return _payload_hash(_paired_config_payload(config))


def _paired_config_payload(config: PairedTemporalIndexConfig) -> dict[str, object]:
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


def _coerce_records(
    records: str | Path | Sequence[PairRecord] | Mapping[str, PairRecord]
) -> tuple[dict[str, PairRecord], Path | None]:
    manifest_path: Path | None = None
    if isinstance(records, (str, Path)):
        manifest_path = Path(records)
        values = load_pair_records(manifest_path)
    elif isinstance(records, Mapping):
        values = list(records.values())
    else:
        values = list(records)
    result: dict[str, PairRecord] = {}
    for record in values:
        if not isinstance(record, PairRecord):
            raise TypeError("SOPAT V4 records must be PairRecord values")
        if record.pair_id in result:
            raise ValueError(f"SOPAT V4 records contain duplicate pair_id: {record.pair_id}")
        result[record.pair_id] = record
    return result, manifest_path


def _resolve_asset_root(manifest_path: Path | None, asset_root: str | Path | None) -> Path:
    if asset_root is not None:
        return Path(os.path.abspath(os.path.normpath(os.fspath(asset_root))))
    if manifest_path is not None:
        return manifest_path.parent.resolve(strict=False)
    return Path.cwd()


def _observation_serialization_key(
    observation: SOPATObservationRefV4,
) -> tuple[str, str, str]:
    measurement = observation.measurement
    return measurement.measurement_id, measurement.record_id, observation.role


def _observation_backend_key(observation: SOPATObservationRefV4) -> tuple[str, str, str]:
    measurement = observation.measurement
    return measurement.time, measurement.record_id, measurement.measurement_id


def _example_sort_key(example: SOPATExampleV4) -> tuple[str, str, str, str]:
    return example.direction, example.split, example.target_time, example.sample_id


def _permutation_seed(seed: int, epoch: int, sample_id: str, window: int) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}:{sample_id}:{window}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _payload_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _asset_identity(value: str, root: Path) -> str:
    expanded = os.path.expanduser(value)
    if not os.path.isabs(expanded):
        expanded = os.path.join(os.fspath(root), expanded)
    return os.path.abspath(os.path.normpath(expanded))


def _day(value: str) -> date:
    return date.fromisoformat(value)


def _required_mapping(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping at {key}")
    return value


def _required_sequence(values: Mapping[str, object], key: str) -> Sequence[object]:
    value = values.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"expected sequence at {key}")
    return value


def _required_mapping_sequence(values: Mapping[str, object], key: str) -> Sequence[Mapping[str, object]]:
    sequence = _required_sequence(values, key)
    if not all(isinstance(value, Mapping) for value in sequence):
        raise TypeError(f"expected mapping entries at {key}")
    return sequence  # type: ignore[return-value]


def _iterable_length(values: Iterable[Mapping[str, object]]) -> int:
    try:
        result = len(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("CoupledDirectionLoader requires sized directional loaders") from error
    return int(result)


def _set_epoch_if_supported(value: object, epoch: int, seen: set[int]) -> None:
    if value is None or id(value) in seen:
        return
    seen.add(id(value))
    setter = getattr(value, "set_epoch", None)
    if callable(setter):
        setter(epoch)


def _assert_direction_batch(batch: Mapping[str, object], direction: Direction) -> None:
    actual = batch.get("sopat_direction")
    if actual != direction:
        raise ValueError(f"coupled loader expected {direction} batch, received {actual!r}")
