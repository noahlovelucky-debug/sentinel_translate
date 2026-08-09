"""Reproducible 2017--2024 raw-raster audit and V2 shard construction.

This module intentionally owns the raw-data contract.  It does not import the
older ``/data/sentinel_translate`` package, so an audit can be reproduced from
this repository and the immutable TIFF tree alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch

from .schema import CLEAR_SCL_CODES, S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER

DATASET_NAME = "sentinel_translate_v32_2017_2024"
DATASET_VERSION = "v2"
DEFAULT_YEARS: tuple[int, ...] = tuple(range(2017, 2025))
S2_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})_mosaic\.tiff?$")
SAR_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2})_(vv|vh)_(ascending|descending)\.tiff?$"
)
TILE_PATTERN = re.compile(r"^Beijing_r(\d+)_c(\d+)_")


@dataclass(frozen=True)
class BuildConfig:
    raw_root: Path
    output_root: Path
    years: tuple[int, ...] = DEFAULT_YEARS
    crop_size: int = 256
    patches_per_pair: int = 16
    crop_attempts: int = 24
    minimum_clear_fraction: float = 0.90
    minimum_crop_valid_fraction: float = 0.80
    thumbnail_size: int = 128
    pair_max_delta_days: int = 3
    train_years: tuple[int, ...] = tuple(range(2017, 2023))
    hf_years: tuple[int, ...] = tuple(range(2017, 2023))

    def __post_init__(self) -> None:
        if self.crop_size <= 0 or self.crop_size % 8:
            raise ValueError("crop_size must be positive and divisible by eight")
        if self.patches_per_pair <= 0 or self.crop_attempts <= 0:
            raise ValueError("patches_per_pair and crop_attempts must be positive")
        if not 0.0 <= self.minimum_clear_fraction <= 1.0:
            raise ValueError("minimum_clear_fraction must be in [0, 1]")
        if not 0.0 <= self.minimum_crop_valid_fraction <= 1.0:
            raise ValueError("minimum_crop_valid_fraction must be in [0, 1]")
        if self.thumbnail_size <= 0 or self.pair_max_delta_days < 0:
            raise ValueError("thumbnail_size must be positive and pair_max_delta_days non-negative")
        if not self.years or not self.train_years or not self.hf_years:
            raise ValueError("years, train_years, and hf_years must be non-empty")
        year_set = {int(year) for year in self.years}
        train_year_set = {int(year) for year in self.train_years}
        hf_year_set = {int(year) for year in self.hf_years}
        if any(year < 2017 or year > 2024 for year in year_set):
            raise ValueError("years must be within the fixed 2017--2024 corpus range")
        if not hf_year_set <= train_year_set:
            raise ValueError("hf_years must be a subset of train_years")
        if any(year > 2022 for year in train_year_set):
            raise ValueError("train_years must be core training years through 2022")


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    year: int
    tile: str
    tile_row: int
    tile_col: int
    split: str
    refit_split: str
    s2_date: str
    s1_date: str
    orbit: str
    delta_days: int
    s2: dict[str, str]
    scl: str
    sar: dict[str, str]
    clear_fraction: float
    valid_fraction: float
    width: int
    height: int
    crs: str
    transform: list[float]
    gsd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PairRecord:
        return cls(**values)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_manifest(path: Path, records: Sequence[PairRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.pair_id):
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    os.replace(temporary, path)


def fixed_split(year: int, row: int, col: int) -> str:
    """Return the fixed spatial/temporal split without any random patch split."""

    if row == 5 or col == 5:
        return "buffer"
    if row == 6 or col == 6:
        if year <= 2022:
            return "unused_spatial"
        if year == 2023:
            return "test_spatial"
        if year == 2024:
            return "test_joint"
        raise ValueError(f"unsupported dataset year: {year}")
    if year <= 2022:
        return "train"
    if year == 2023:
        return "validation_temporal"
    if year == 2024:
        return "test_temporal"
    raise ValueError(f"unsupported dataset year: {year}")


def _dated_mosaics(directory: Path) -> dict[str, Path]:
    records: dict[str, Path] = {}
    if not directory.is_dir():
        return records
    for path in sorted(directory.iterdir()):
        match = S2_PATTERN.match(path.name)
        if match:
            records[match.group(1)] = path
    return records


def _sar_frames(directory: Path) -> dict[tuple[str, str], dict[str, Path]]:
    frames: dict[tuple[str, str], dict[str, Path]] = {}
    if not directory.is_dir():
        return frames
    for path in sorted(directory.iterdir()):
        match = SAR_PATTERN.match(path.name)
        if match is None:
            continue
        day, polarization, orbit = match.groups()
        frames.setdefault((day, orbit), {})[polarization] = path
    return frames


def _grid_signature(path: Path) -> tuple[int, int, str, tuple[float, ...], float]:
    import rasterio

    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError(f"missing CRS: {path}")
        return (
            int(source.width),
            int(source.height),
            source.crs.to_string(),
            tuple(float(value) for value in source.transform[:6]),
            abs(float(source.transform.a)),
        )


def _all_same_grid(paths: Iterable[Path]) -> tuple[int, int, str, tuple[float, ...], float]:
    iterator = iter(paths)
    first = _grid_signature(next(iterator))
    for path in iterator:
        if _grid_signature(path)[:4] != first[:4]:
            raise ValueError(f"grid mismatch: {path}")
    return first


def _thumbnail(path: Path, size: int) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as source:
        height = min(int(source.height), size)
        width = min(int(source.width), size)
        return source.read(1, out_shape=(height, width), resampling=Resampling.nearest)


def _scene_quality(
    s2: dict[str, Path], scl: Path, sar: dict[str, Path], config: BuildConfig
) -> tuple[float, float]:
    scl_values = _thumbnail(scl, config.thumbnail_size)
    clear = np.isin(scl_values, CLEAR_SCL_CODES)
    clear_fraction = float(clear.mean())
    if clear_fraction < config.minimum_clear_fraction:
        return clear_fraction, 0.0
    s2_valid = np.ones_like(clear, dtype=bool)
    for path in s2.values():
        s2_valid &= _thumbnail(path, config.thumbnail_size) > 0
    sar_valid = np.ones_like(clear, dtype=bool)
    for path in sar.values():
        sar_valid &= _thumbnail(path, config.thumbnail_size) > 0
    return clear_fraction, float((clear & s2_valid & sar_valid).mean())


def _complete_s2_dates(tile_root: Path) -> tuple[dict[str, dict[str, Path]], Counter[str]]:
    by_band = {
        band: _dated_mosaics(tile_root / "data_raw" / band) for band in S2_CHANNEL_ORDER
    }
    scl = _dated_mosaics(tile_root / "data_raw" / "scl")
    union = set(scl)
    for values in by_band.values():
        union.update(values)
    complete: dict[str, dict[str, Path]] = {}
    rejected: Counter[str] = Counter()
    for day in sorted(union):
        missing = [band for band, values in by_band.items() if day not in values]
        if day not in scl:
            missing.append("scl")
        if missing:
            rejected["incomplete_s2_date"] += 1
            continue
        complete[day] = {**{band: by_band[band][day] for band in S2_CHANNEL_ORDER}, "scl": scl[day]}
    return complete, rejected


def _discover_tile(config: BuildConfig, year: int, tile_root: Path) -> tuple[list[PairRecord], Counter[str], Counter[str]]:
    import rasterio

    match = TILE_PATTERN.match(tile_root.name)
    if match is None:
        return [], Counter({"invalid_tile_name": 1}), Counter()
    row, col = (int(value) for value in match.groups())
    s2_dates, rejected = _complete_s2_dates(tile_root)
    frames = _sar_frames(tile_root / "data_sar_raw")
    complete_frames = {
        key: values
        for key, values in frames.items()
        if all(polarization in values for polarization in SAR_CHANNEL_ORDER)
    }
    discovered = Counter(
        {
            "complete_s2_dates": len(s2_dates),
            "complete_sar_frames": len(complete_frames),
        }
    )
    rejected["incomplete_sar_frame"] += len(frames) - len(complete_frames)
    records: list[PairRecord] = []
    for (s1_day, orbit), sar_paths in sorted(complete_frames.items()):
        s1_date = date.fromisoformat(s1_day)
        candidates = sorted(
            (
                abs((date.fromisoformat(s2_day) - s1_date).days),
                s2_day,
            )
            for s2_day in s2_dates
            if abs((date.fromisoformat(s2_day) - s1_date).days) <= config.pair_max_delta_days
        )
        if not candidates:
            rejected["no_s2_within_delta"] += 1
            continue
        selected: tuple[int, str, dict[str, Path], tuple[int, int, str, tuple[float, ...], float], float, float] | None = None
        candidate_failures: Counter[str] = Counter()
        for delta_days, s2_day in candidates:
            candidate = s2_dates[s2_day]
            s2_paths = {band: candidate[band] for band in S2_CHANNEL_ORDER}
            try:
                grid = _all_same_grid(
                    (candidate["scl"], *s2_paths.values(), *sar_paths.values())
                )
            except rasterio.errors.RasterioError:
                candidate_failures["grid_read_error"] += 1
                continue
            except (OSError, ValueError):
                candidate_failures["grid_or_crs_mismatch"] += 1
                continue
            try:
                clear_fraction, valid_fraction = _scene_quality(
                    s2_paths, candidate["scl"], sar_paths, config
                )
            except (OSError, rasterio.errors.RasterioError):
                candidate_failures["quality_read_error"] += 1
                continue
            if clear_fraction < config.minimum_clear_fraction:
                candidate_failures["clear_fraction"] += 1
                continue
            if valid_fraction <= 0.0:
                candidate_failures["no_joint_valid"] += 1
                continue
            selected = delta_days, s2_day, s2_paths, grid, clear_fraction, valid_fraction
            break
        rejected.update(candidate_failures)
        if selected is None:
            rejected["no_acceptable_s2_candidate"] += 1
            continue
        delta_days, s2_day, s2_paths, grid, clear_fraction, valid_fraction = selected
        split = fixed_split(year, row, col)
        records.append(
            PairRecord(
                pair_id=f"{year}:{tile_root.name}:{s1_day}:{orbit}:{s2_day}",
                year=year,
                tile=tile_root.name,
                tile_row=row,
                tile_col=col,
                split=split,
                refit_split="excluded",
                s2_date=s2_day,
                s1_date=s1_day,
                orbit=orbit,
                delta_days=delta_days,
                s2={key: str(value) for key, value in s2_paths.items()},
                scl=str(s2_dates[s2_day]["scl"]),
                sar={key: str(sar_paths[key]) for key in SAR_CHANNEL_ORDER},
                clear_fraction=clear_fraction,
                valid_fraction=valid_fraction,
                width=grid[0],
                height=grid[1],
                crs=grid[2],
                transform=list(grid[3]),
                gsd=grid[4],
            )
        )
    return records, rejected, discovered


def _discover_tile_worker(arguments: tuple[BuildConfig, int, str]) -> tuple[list[PairRecord], Counter[str], Counter[str]]:
    config, year, tile = arguments
    return _discover_tile(config, year, Path(tile))


def discover_pairs(config: BuildConfig, *, workers: int = 1) -> tuple[list[PairRecord], dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    tasks = [
        (config, year, str(tile))
        for year in config.years
        for tile in sorted((config.raw_root / str(year)).glob("Beijing_*"))
        if TILE_PATTERN.match(tile.name)
    ]
    records: list[PairRecord] = []
    rejected: Counter[str] = Counter()
    discovered: Counter[str] = Counter()

    def collect(results: Iterable[tuple[list[PairRecord], Counter[str], Counter[str]]]) -> None:
        for completed, (tile_records, tile_rejected, tile_discovered) in enumerate(results, start=1):
            records.extend(tile_records)
            rejected.update(tile_rejected)
            discovered.update(tile_discovered)
            if completed % 10 == 0 or completed == len(tasks):
                print(
                    json.dumps(
                        {
                            "audited_tiles": completed,
                            "total_tiles": len(tasks),
                            "accepted_pairs": len(records),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    if workers == 1:
        collect(_discover_tile_worker(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_discover_tile_worker, task) for task in tasks]
            collect(future.result() for future in as_completed(futures))
    records.sort(key=lambda record: record.pair_id)
    assert_split_leakage(records)
    summary = _audit_summary(records, rejected, discovered)
    return records, summary


def assert_split_leakage(records: Sequence[PairRecord]) -> None:
    """Assert the documented spatial and temporal separation rules."""

    pair_ids = [record.pair_id for record in records]
    if len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("manifest contains duplicate pair IDs")
    train_tiles: set[str] = set()
    spatial_holdout_tiles: set[str] = set()
    for record in records:
        expected = fixed_split(record.year, record.tile_row, record.tile_col)
        if record.split != expected:
            raise RuntimeError(f"split mismatch for {record.pair_id}: {record.split} != {expected}")
        if record.split == "train":
            if not 2017 <= record.year <= 2022:
                raise RuntimeError("train split contains a non-training year")
            train_tiles.add(record.tile)
        if record.split in {"unused_spatial", "test_spatial", "test_joint"}:
            spatial_holdout_tiles.add(record.tile)
        if record.split == "validation_temporal" and record.year != 2023:
            raise RuntimeError("validation_temporal must be 2023 core only")
        if record.split == "test_temporal" and record.year != 2024:
            raise RuntimeError("test_temporal must be 2024 core only")
    if train_tiles & spatial_holdout_tiles:
        raise RuntimeError("spatial holdout tile leaked into train")


def _count(records: Iterable[PairRecord], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter(str(getattr(record, key)) for record in records)
    return dict(sorted(counts.items()))


def _audit_summary(
    records: Sequence[PairRecord], rejected: Counter[str], discovered: Counter[str]
) -> dict[str, Any]:
    return {
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "records": len(records),
        "split_counts": _count(records, "split"),
        "year_counts": _count(records, "year"),
        "delta_day_counts": _count(records, "delta_days"),
        "orbit_counts": _count(records, "orbit"),
        "tile_counts": _count(records, "tile"),
        "rejected": dict(sorted(rejected.items())),
        "discovered": dict(sorted(discovered.items())),
    }


def write_audit_artifacts(config: BuildConfig, records: Sequence[PairRecord], summary: dict[str, Any]) -> dict[str, Any]:
    manifest = config.output_root / "manifests" / "pairs.jsonl"
    _atomic_manifest(manifest, records)
    manifest_hash = file_sha256(manifest)
    protocol_records = [record for record in records if record.split == "validation_temporal"]
    protocol = {
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "name": "validation_temporal_2023",
        "split": "validation_temporal",
        "expected_samples": len(protocol_records),
        "crop": {"kind": "center", "size": config.crop_size},
        "mask_scl_codes": list(CLEAR_SCL_CODES),
        "units": {
            "optical": "surface_reflectance_0_1",
            "sar": "decibel_backscatter",
        },
        "s2_channel_order": list(S2_CHANNEL_ORDER),
        "sar_channel_order": list(SAR_CHANNEL_ORDER),
        "train_years": list(config.train_years),
        "hf_years": list(config.hf_years),
        "manifest_sha256": manifest_hash,
    }
    _atomic_json(config.output_root / "manifests" / "validation_protocol.json", protocol)
    audit = {
        **summary,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "train_years": list(config.train_years),
        "hf_years": list(config.hf_years),
    }
    _atomic_json(config.output_root / "audit" / "audit.json", audit)
    return audit


def audit_dataset(config: BuildConfig, *, workers: int = 1) -> tuple[list[PairRecord], dict[str, Any]]:
    records, summary = discover_pairs(config, workers=workers)
    return records, write_audit_artifacts(config, records, summary)


def load_manifest(path: str | Path, *, split: str | None = None) -> list[PairRecord]:
    records: list[PairRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = PairRecord.from_dict(json.loads(line))
                if split is None or record.split == split:
                    records.append(record)
    return sorted(records, key=lambda record: record.pair_id)


def _metadata_vector(record: PairRecord) -> torch.Tensor:
    s1_day = date.fromisoformat(record.s1_date)
    s2_day = date.fromisoformat(record.s2_date)
    phase_s1 = 2.0 * math.pi * s1_day.timetuple().tm_yday / 366.0
    phase_s2 = 2.0 * math.pi * s2_day.timetuple().tm_yday / 366.0
    orbit = {"ascending": -1.0, "descending": 1.0}.get(record.orbit, 0.0)
    return torch.tensor(
        (
            record.delta_days / 3.0,
            orbit,
            math.sin(phase_s1),
            math.cos(phase_s1),
            math.sin(phase_s2),
            math.cos(phase_s2),
            math.log(max(record.gsd, 0.1)) / 4.0,
            1.0,
        ),
        dtype=torch.float32,
    )


def _window_seed(pair_id: str, patch_index: int) -> int:
    digest = hashlib.sha256(f"{pair_id}:{patch_index}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


class _PairRasterCache:
    """Open a record's immutable rasters once while selecting all of its crops."""

    def __init__(self, record: PairRecord) -> None:
        self.record = record
        self._stack = ExitStack()
        self.s2: tuple[Any, ...] = ()
        self.sar: tuple[Any, ...] = ()
        self.scl: Any | None = None

    def __enter__(self) -> Self:
        import rasterio

        try:
            self.s2 = tuple(
                self._stack.enter_context(rasterio.open(self.record.s2[channel]))
                for channel in S2_CHANNEL_ORDER
            )
            self.sar = tuple(
                self._stack.enter_context(rasterio.open(self.record.sar[channel]))
                for channel in SAR_CHANNEL_ORDER
            )
            self.scl = self._stack.enter_context(rasterio.open(self.record.scl))
        except BaseException:
            self._stack.close()
            raise
        return self

    def __exit__(self, *arguments: object) -> None:
        self._stack.close()


def _read_window(cache: _PairRasterCache, window: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    from rasterio.windows import Window

    col, row, width, height = window
    raster_window = Window(col, row, width, height)
    s2_values = [source.read(1, window=raster_window) for source in cache.s2]
    sar_values = [source.read(1, window=raster_window) for source in cache.sar]
    s2_raw = np.stack(s2_values)
    sar_raw = np.stack(sar_values)
    if cache.scl is None:
        raise RuntimeError("raster cache has not been opened")
    scl = cache.scl.read(1, window=raster_window)
    s2_valid = np.isin(scl, CLEAR_SCL_CODES) & np.all(s2_raw > 0, axis=0)
    sar_valid = np.all(sar_raw > 0, axis=0)
    joint_valid = s2_valid & sar_valid
    s2 = np.clip(s2_raw.astype(np.float32) / 10000.0, 0.0, 1.0) * 2.0 - 1.0
    sar_db = sar_raw.astype(np.float32) / 200.0 - 50.0
    minimum = np.asarray((-35.0, -45.0), dtype=np.float32)[:, None, None]
    maximum = np.asarray((5.0, -5.0), dtype=np.float32)[:, None, None]
    sar = np.clip(2.0 * (sar_db - minimum) / (maximum - minimum) - 1.0, -1.0, 1.0)
    s2[:, ~s2_valid] = 0.0
    sar[:, ~sar_valid] = 0.0
    return {
        "s2": s2,
        "sar": sar,
        "s2_valid": s2_valid[None].astype(np.uint8),
        "sar_valid": sar_valid[None].astype(np.uint8),
        "joint_valid": joint_valid[None].astype(np.uint8),
    }


def _select_windows(record: PairRecord, config: BuildConfig) -> tuple[list[tuple[int, int, int, int]], list[dict[str, np.ndarray]]]:
    if record.width < config.crop_size or record.height < config.crop_size:
        raise ValueError("raster_smaller_than_crop")
    windows: list[tuple[int, int, int, int]] = []
    payloads: list[dict[str, np.ndarray]] = []
    with _PairRasterCache(record) as cache:
        for patch_index in range(config.patches_per_pair):
            rng = np.random.default_rng(_window_seed(record.pair_id, patch_index))
            selected: tuple[tuple[int, int, int, int], dict[str, np.ndarray]] | None = None
            for _ in range(config.crop_attempts):
                row = int(rng.integers(0, record.height - config.crop_size + 1))
                col = int(rng.integers(0, record.width - config.crop_size + 1))
                window = (col, row, config.crop_size, config.crop_size)
                arrays = _read_window(cache, window)
                if float(arrays["joint_valid"].mean()) >= config.minimum_crop_valid_fraction:
                    selected = window, arrays
                    break
            if selected is None:
                raise ValueError("insufficient_valid_patches")
            window, arrays = selected
            windows.append(window)
            payloads.append(arrays)
    return windows, payloads


def _shard_path(output_root: Path, record: PairRecord) -> Path:
    digest = hashlib.sha256(record.pair_id.encode("utf-8")).hexdigest()[:16]
    return output_root / "shards" / "train" / f"{record.year}_{digest}.pt"


def _shard_payload(
    record: PairRecord,
    windows: Sequence[tuple[int, int, int, int]],
    arrays: Sequence[dict[str, np.ndarray]],
) -> dict[str, object]:
    return {
        "format_version": 2,
        "s2_channel_order": list(S2_CHANNEL_ORDER),
        "sar_channel_order": list(SAR_CHANNEL_ORDER),
        "s2": torch.from_numpy(np.stack([value["s2"] for value in arrays])).to(torch.float16),
        "sar": torch.from_numpy(np.stack([value["sar"] for value in arrays])).to(torch.float16),
        "s2_valid": torch.from_numpy(np.stack([value["s2_valid"] for value in arrays])),
        "sar_valid": torch.from_numpy(np.stack([value["sar_valid"] for value in arrays])),
        "joint_valid": torch.from_numpy(np.stack([value["joint_valid"] for value in arrays])),
        "metadata": _metadata_vector(record).expand(len(arrays), -1).clone(),
        "window": torch.tensor(windows, dtype=torch.int32),
        "pair_id": [record.pair_id] * len(arrays),
    }


def _validate_existing_shard(
    path: Path, record: PairRecord, windows: Sequence[tuple[int, int, int, int]], config: BuildConfig
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("format_version", 0)) != 2:
        raise RuntimeError(f"resume shard has wrong schema: {path}")
    if tuple(payload.get("s2_channel_order", ())) != S2_CHANNEL_ORDER:
        raise RuntimeError(f"resume shard has wrong S2 schema: {path}")
    if tuple(payload.get("sar_channel_order", ())) != SAR_CHANNEL_ORDER:
        raise RuntimeError(f"resume shard has wrong SAR schema: {path}")
    if payload.get("pair_id") != [record.pair_id] * config.patches_per_pair:
        raise RuntimeError(f"resume shard pair IDs do not match: {path}")
    expected_windows = torch.tensor(windows, dtype=torch.int32)
    if not torch.equal(payload.get("window"), expected_windows):
        raise RuntimeError(f"resume shard windows do not match: {path}")
    s2 = payload.get("s2")
    sar = payload.get("sar")
    if not isinstance(s2, torch.Tensor) or not isinstance(sar, torch.Tensor):
        raise TypeError(f"resume shard tensors are missing: {path}")
    if s2.shape != (config.patches_per_pair, len(S2_CHANNEL_ORDER), config.crop_size, config.crop_size):
        raise RuntimeError(f"resume shard S2 shape does not match: {path}")
    if sar.shape != (config.patches_per_pair, len(SAR_CHANNEL_ORDER), config.crop_size, config.crop_size):
        raise RuntimeError(f"resume shard SAR shape does not match: {path}")
    return {
        "path": str(path.resolve()),
        "pair_id": record.pair_id,
        "year": record.year,
        "delta_days": record.delta_days,
        "hf_candidate": record.year in config.hf_years and record.delta_days <= 1,
        "count": config.patches_per_pair,
        "windows_sha256": hashlib.sha256(expected_windows.numpy().tobytes()).hexdigest(),
        "checksum": file_sha256(path),
    }


def _write_shard(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _build_one_shard(arguments: tuple[PairRecord, BuildConfig, bool]) -> tuple[dict[str, object] | None, str | None]:
    import rasterio

    record, config, resume = arguments
    try:
        windows, arrays = _select_windows(record, config)
    except ValueError as error:
        return None, str(error)
    except (OSError, rasterio.errors.RasterioError):
        return None, "raster_read_error"
    path = _shard_path(config.output_root, record)
    if path.is_file() and resume:
        return _validate_existing_shard(path, record, windows, config), None
    if path.is_file() and not resume:
        raise RuntimeError(f"shard already exists; pass --resume to verify it: {path}")
    payload = _shard_payload(record, windows, arrays)
    _write_shard(path, payload)
    expected_windows = torch.tensor(windows, dtype=torch.int32)
    return {
        "path": str(path.resolve()),
        "pair_id": record.pair_id,
        "year": record.year,
        "delta_days": record.delta_days,
        "hf_candidate": record.year in config.hf_years and record.delta_days <= 1,
        "count": config.patches_per_pair,
        "windows_sha256": hashlib.sha256(expected_windows.numpy().tobytes()).hexdigest(),
        "checksum": file_sha256(path),
    }, None


def build_train_shards(
    config: BuildConfig,
    records: Sequence[PairRecord],
    *,
    workers: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    train_records = [record for record in records if record.split == "train"]
    if not train_records:
        raise RuntimeError("manifest contains no train records")
    tasks = [(record, config, resume) for record in sorted(train_records, key=lambda item: item.pair_id)]
    descriptors: list[dict[str, object]] = []
    rejected: Counter[str] = Counter()

    def collect(results: Iterable[tuple[dict[str, object] | None, str | None]]) -> None:
        for completed, (descriptor, reason) in enumerate(results, start=1):
            if descriptor is None:
                rejected[reason or "unknown"] += 1
            else:
                descriptors.append(descriptor)
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    json.dumps(
                        {
                            "built_pairs": completed,
                            "total_pairs": len(tasks),
                            "accepted_shards": len(descriptors),
                            "rejected_pairs": sum(rejected.values()),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    if workers == 1:
        collect(_build_one_shard(task) for task in tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_build_one_shard, task) for task in tasks]
            collect(future.result() for future in as_completed(futures))
    descriptors.sort(key=lambda item: str(item["pair_id"]))
    manifest = config.output_root / "manifests" / "pairs.jsonl"
    if not manifest.is_file():
        raise RuntimeError("manifest must be written before shards")
    index = {
        "format_version": 2,
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "split": "train",
        "crop_size": config.crop_size,
        "patches_per_pair": config.patches_per_pair,
        "s2_channel_order": list(S2_CHANNEL_ORDER),
        "sar_channel_order": list(SAR_CHANNEL_ORDER),
        "train_years": list(config.train_years),
        "hf_years": list(config.hf_years),
        "source_records": len(train_records),
        "rejected_pairs": dict(sorted(rejected.items())),
        "patches": sum(int(item["count"]) for item in descriptors),
        "shards": descriptors,
    }
    index_path = config.output_root / "shards" / "train" / "index.json"
    _atomic_json(index_path, index)
    eligible_indices: list[int] = []
    offset = 0
    for descriptor in descriptors:
        count = int(descriptor["count"])
        if bool(descriptor["hf_candidate"]):
            eligible_indices.extend(range(offset, offset + count))
        offset += count
    eligibility = {
        "format_version": 2,
        "source_index": str(index_path.resolve()),
        "source_index_sha256": file_sha256(index_path),
        "train_years": list(config.train_years),
        "hf_years": list(config.hf_years),
        "delta_weights": {"0": 1.0, "1": 0.25, "long_gap": 0.0},
        "registration_audited": False,
        "samples": offset,
        "eligible_samples": len(eligible_indices),
        "eligible_indices": eligible_indices,
    }
    _atomic_json(config.output_root / "hf_eligibility.json", eligibility)
    return {**index, "index": str(index_path.resolve())}


def run_build(config: BuildConfig, *, workers: int = 1, resume: bool = False) -> dict[str, Any]:
    records, audit = audit_dataset(config, workers=workers)
    index = build_train_shards(config, records, workers=workers, resume=resume)
    logs = config.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    result = {
        "audit": audit,
        "train_index": index["index"],
        "next_steps": [
            "run registration audit to replace hf_eligibility.json",
            "precompute temporal-prior shards from the audited training index",
        ],
    }
    _atomic_json(logs / "build.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the 2017-2024 Sentinel translation corpus")
    parser.add_argument("--raw-root", type=Path, default=Path("/data/data_disk/data_dir"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("/data/datasets/sentinel_translate_v32_2017_2024")
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--patches-per-pair", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--build", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = BuildConfig(
        raw_root=args.raw_root,
        output_root=args.output_root,
        patches_per_pair=args.patches_per_pair,
    )
    if args.audit_only:
        _, result = audit_dataset(config, workers=args.workers)
    else:
        result = run_build(config, workers=args.workers, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
