from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_sopat_v4_data import _record

from sentinel_v3.dataset_builder import PairRecord, file_sha256
from sentinel_v3.paired_temporal_data import write_paired_temporal_index
from sentinel_v3.schema import CLEAR_SCL_CODES, S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER
from sentinel_v4.cache import (
    SOPATChunkCachePreflightError,
    preflight_sopat_v4_chunk_cache,
    sopat_chunk_dataset_from_cache,
)
from sentinel_v4.data import (
    SOPATIndexV4,
    build_sopat_v4_index,
    paired_temporal_index_from_sopat_v4,
)


def _sopat_index() -> SOPATIndexV4:
    records = [
        _record(0, s1_date="2020-01-01", s2_date="2020-01-01"),
        _record(1, s1_date="2020-01-03", s2_date="2020-01-03"),
        _record(2, s1_date="2020-01-05", s2_date="2020-01-05"),
        _record(3, s1_date="2020-01-10", s2_date="2020-01-10"),
        _record(4, s1_date="2020-01-13", s2_date="2020-01-14"),
        _record(5, s1_date="2020-01-15", s2_date="2020-01-17"),
    ]
    return build_sopat_v4_index(records, splits=("train",), max_observations=3)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_published_plan(
    root: Path,
    *,
    config_sha256: str,
    source_manifest_sha256: str,
    crop_size: int = 256,
    windows_per_acquisition: int = 1,
    train_split: str = "train",
    validation_split: str = "validation_temporal",
) -> str:
    """Write the immutable plan payload V3 hashes into cache_index.json."""

    plan = {
        "format_version": 1,
        "config_path": "synthetic-config.yaml",
        "config_sha256": config_sha256,
        "source_manifest": "synthetic-manifest.jsonl",
        "source_manifest_sha256": source_manifest_sha256,
        "destination_root": str(root),
        "train_split": train_split,
        "validation_split": validation_split,
        "crop_size": crop_size,
        "windows_per_acquisition": windows_per_acquisition,
        "grids": [],
        "acquisitions": [],
        "indexes": [],
        "routing_path": "routing.json",
        "report": {},
    }
    _write_json(root / "plan.json", plan)
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _complete_cache(root: Path, index: SOPATIndexV4) -> None:
    routes: dict[str, dict[str, str]] = {}
    grids: dict[str, dict[str, object]] = {}
    acquisition_ids: set[str] = set()
    index_entries: list[dict[str, object]] = []
    for direction in ("sar_to_optical", "optical_to_sar"):
        legacy = paired_temporal_index_from_sopat_v4(index, direction=direction, split="train")
        index_path = root / "indexes" / direction / "train.jsonl"
        write_paired_temporal_index(index_path, legacy)
        index_entries.append(
            {
                "direction": direction,
                "split": "train",
                "relative_path": index_path.relative_to(root).as_posix(),
                "samples": len(legacy),
                "sha256": file_sha256(index_path),
            }
        )
        for sample in legacy:
            for pair_id in (sample.query_pair_id, sample.anchor_pair_id, *sample.observation_pair_ids):
                if pair_id in routes:
                    continue
                example = _example_for_pair(index, direction=direction, pair_id=pair_id)
                grid_id = example.canonical_grid.grid_id
                grids[grid_id] = {
                    "grid_id": grid_id,
                    "tile": example.canonical_grid.tile,
                    "width": example.canonical_grid.width,
                    "height": example.canonical_grid.height,
                    "crs": example.canonical_grid.crs,
                    "transform": list(example.canonical_grid.transform),
                    "gsd": example.canonical_grid.gsd_m,
                    "windows": [[0, 0, 256, 256]],
                    "center_window_index": 0,
                }
                optical = f"optical-{pair_id}"
                sar = f"sar-{pair_id}"
                routes[pair_id] = {
                    "optical_acquisition_id": optical,
                    "sar_acquisition_id": sar,
                    "grid_id": grid_id,
                    "split": "train",
                }
                acquisition_ids.update((optical, sar))
    routing_path = root / "routing.json"
    _write_json(routing_path, {"format_version": 1, "routes": routes})
    manifest_digest = next(iter(index.examples)).provenance.source_manifest_sha256
    provenance = {
        "format_version": 1,
        "source_manifest_sha256": manifest_digest,
        "normalization": "paired_temporal_v3_normalized",
        "optical_channels": list(S2_CHANNEL_ORDER),
        "sar_channels": list(SAR_CHANNEL_ORDER),
        "crop_size": 256,
        "windows_per_acquisition": 1,
        "splits": ["train", "validation_temporal"],
    }
    _write_json(root / "provenance.json", provenance)
    plan_sha256 = _write_published_plan(
        root,
        config_sha256="config-digest",
        source_manifest_sha256=manifest_digest,
    )
    cache_index = {
        "format_version": 1,
        "plan_sha256": plan_sha256,
        "config_sha256": "config-digest",
        "source_manifest_sha256": manifest_digest,
        "crop_size": 256,
        "windows_per_acquisition": 1,
        "train_split": "train",
        "validation_split": "validation_temporal",
        "grids": list(grids.values()),
        "routing_path": "routing.json",
        "routing_sha256": file_sha256(routing_path),
        "indexes": index_entries,
        "acquisitions": [
            {
                "acquisition_id": acquisition_id,
                "modality": "optical" if acquisition_id.startswith("optical-") else "sar",
                "grid_id": next(iter(grids)),
                "relative_directory": f"acquisitions/{acquisition_id}",
                "values": {"shape": [1, 10 if acquisition_id.startswith("optical-") else 2, 256, 256]},
                "valid": {"shape": [1, 1, 256, 256]},
            }
            for acquisition_id in sorted(acquisition_ids)
        ],
    }
    _write_json(root / "cache_index.json", cache_index)


def _example_for_pair(
    index: SOPATIndexV4, *, direction: str, pair_id: str
):
    for candidate in index.examples:
        if candidate.direction != direction:
            continue
        if candidate.target.record_id == pair_id or candidate.anchor_pair.registration_id == pair_id:
            return candidate
        if any(observation.measurement.record_id == pair_id for observation in candidate.observations):
            return candidate
    raise AssertionError(f"no SOPAT example contains {direction}/{pair_id}")


def _write_tiff(path: Path, values: np.ndarray) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs="EPSG:32650",
        transform=from_origin(500000.0, 4100000.0, 10.0, 10.0),
    ) as source:
        source.write(values, 1)


def _local_record(
    root: Path,
    number: int,
    *,
    s1_date: str,
    s2_date: str,
) -> PairRecord:
    size = 256
    record_root = root / "raw" / f"record-{number:03d}"
    optical = np.full((size, size), 3500 + number * 300, dtype=np.uint16)
    sar = np.full((size, size), 6000 + number * 100, dtype=np.uint16)
    scl = np.full((size, size), 4, dtype=np.uint8)
    s2: dict[str, str] = {}
    for channel in S2_CHANNEL_ORDER:
        path = record_root / "s2" / f"{channel}.tif"
        _write_tiff(path, optical)
        s2[channel] = str(path)
    scl_path = record_root / "scl.tif"
    _write_tiff(scl_path, scl)
    sar_paths: dict[str, str] = {}
    for channel in SAR_CHANNEL_ORDER:
        path = record_root / "sar" / f"{channel}.tif"
        _write_tiff(path, sar)
        sar_paths[channel] = str(path)
    return PairRecord(
        pair_id=f"2020:tile-local:train:{number:03d}:ascending",
        year=2020,
        tile="tile-local",
        tile_row=1,
        tile_col=1,
        split="train",
        refit_split="excluded",
        s2_date=s2_date,
        s1_date=s1_date,
        orbit="ascending",
        delta_days=abs(int(s2_date[-2:]) - int(s1_date[-2:])),
        s2=s2,
        scl=str(scl_path),
        sar=sar_paths,
        clear_fraction=1.0,
        valid_fraction=1.0,
        width=size,
        height=size,
        crs="EPSG:32650",
        transform=[10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0],
        gsd=10.0,
    )


def _local_records(root: Path) -> list[PairRecord]:
    return [
        _local_record(root, 0, s1_date="2020-01-01", s2_date="2020-01-01"),
        _local_record(root, 1, s1_date="2020-01-03", s2_date="2020-01-03"),
        _local_record(root, 2, s1_date="2020-01-05", s2_date="2020-01-05"),
        _local_record(root, 3, s1_date="2020-01-10", s2_date="2020-01-10"),
        _local_record(root, 4, s1_date="2020-01-13", s2_date="2020-01-14"),
        _local_record(root, 5, s1_date="2020-01-15", s2_date="2020-01-17"),
    ]


def _array_descriptor(path: Path, values: np.ndarray) -> dict[str, object]:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "dtype": np.dtype(values.dtype).str,
        "shape": list(values.shape),
    }


def _normalized_acquisition(record: PairRecord, modality: str) -> tuple[np.ndarray, np.ndarray]:
    rasterio = pytest.importorskip("rasterio")
    if modality == "optical":
        raw = np.stack(
            [rasterio.open(record.s2[channel]).read(1) for channel in S2_CHANNEL_ORDER]
        )
        with rasterio.open(record.scl) as source:
            scl = source.read(1)
        valid = np.isin(scl, CLEAR_SCL_CODES) & np.all(raw > 0, axis=0)
        values = np.clip(raw.astype(np.float32) / 10000.0, 0.0, 1.0) * 2.0 - 1.0
    else:
        raw = np.stack(
            [rasterio.open(record.sar[channel]).read(1) for channel in SAR_CHANNEL_ORDER]
        )
        valid = np.all(raw > 0, axis=0)
        values_db = raw.astype(np.float32) / 200.0 - 50.0
        minimum = np.asarray((-35.0, -45.0), dtype=np.float32)[:, None, None]
        maximum = np.asarray((5.0, -5.0), dtype=np.float32)[:, None, None]
        values = np.clip(2.0 * (values_db - minimum) / (maximum - minimum) - 1.0, -1.0, 1.0)
    values[:, ~valid] = 0.0
    return values[None].astype(np.float16), valid[None, None].astype(np.uint8)


def _write_full_completed_cache(
    root: Path,
    index: SOPATIndexV4,
    records: list[PairRecord],
    *,
    manifest_sha256: str,
) -> None:
    grid = index.examples[0].canonical_grid
    routes: dict[str, dict[str, str]] = {}
    acquisitions: list[dict[str, object]] = []
    for record in records:
        route = {
            "optical_acquisition_id": f"optical-{record.pair_id.replace(':', '_')}",
            "sar_acquisition_id": f"sar-{record.pair_id.replace(':', '_')}",
            "grid_id": grid.grid_id,
            "split": record.split,
        }
        routes[record.pair_id] = route
        for modality, acquisition_id in (
            ("optical", route["optical_acquisition_id"]),
            ("sar", route["sar_acquisition_id"]),
        ):
            values, valid = _normalized_acquisition(record, modality)
            relative_directory = Path("acquisitions") / modality / acquisition_id
            directory = root / relative_directory
            directory.mkdir(parents=True, exist_ok=True)
            values_path = directory / "values.npy"
            valid_path = directory / "valid.npy"
            np.save(values_path, values, allow_pickle=False)
            np.save(valid_path, valid, allow_pickle=False)
            chunk = {
                "format_version": 1,
                "acquisition_id": acquisition_id,
                "modality": modality,
                "grid_id": grid.grid_id,
                "relative_directory": relative_directory.as_posix(),
                "values": _array_descriptor(values_path, values),
                "valid": _array_descriptor(valid_path, valid),
            }
            _write_json(directory / "chunk.json", chunk)
            acquisitions.append(chunk)
    index_entries: list[dict[str, object]] = []
    for direction in ("sar_to_optical", "optical_to_sar"):
        legacy = paired_temporal_index_from_sopat_v4(index, direction=direction, split="train")
        path = root / "indexes" / direction / "train.jsonl"
        write_paired_temporal_index(path, legacy)
        index_entries.append(
            {
                "direction": direction,
                "split": "train",
                "relative_path": path.relative_to(root).as_posix(),
                "samples": len(legacy),
                "sha256": file_sha256(path),
            }
        )
    routing_path = root / "routing.json"
    _write_json(routing_path, {"format_version": 1, "routes": routes})
    _write_json(
        root / "provenance.json",
        {
            "format_version": 1,
            "source_manifest_sha256": manifest_sha256,
            "normalization": "paired_temporal_v3_normalized",
            "optical_channels": list(S2_CHANNEL_ORDER),
            "sar_channels": list(SAR_CHANNEL_ORDER),
            "crop_size": 256,
            "windows_per_acquisition": 1,
            "splits": ["train", "validation_temporal"],
        },
    )
    plan_sha256 = _write_published_plan(
        root,
        config_sha256="synthetic-config",
        source_manifest_sha256=manifest_sha256,
    )
    _write_json(
        root / "cache_index.json",
        {
            "format_version": 1,
            "plan_sha256": plan_sha256,
            "config_sha256": "synthetic-config",
            "source_manifest_sha256": manifest_sha256,
            "crop_size": 256,
            "windows_per_acquisition": 1,
            "train_split": "train",
            "validation_split": "validation_temporal",
            "grids": [
                {
                    "grid_id": grid.grid_id,
                    "tile": grid.tile,
                    "width": grid.width,
                    "height": grid.height,
                    "crs": grid.crs,
                    "transform": list(grid.transform),
                    "gsd": grid.gsd_m,
                    "windows": [[0, 0, 256, 256]],
                    "center_window_index": 0,
                }
            ],
            "routing_path": "routing.json",
            "routing_sha256": file_sha256(routing_path),
            "indexes": index_entries,
            "acquisitions": acquisitions,
        },
    )


def test_preflight_accepts_completed_local_cache_and_rejects_incomplete(
    tmp_path: Path,
) -> None:
    index = _sopat_index()
    root = tmp_path / "cache"
    _complete_cache(root, index)

    result = preflight_sopat_v4_chunk_cache(root, index)

    assert result.examples == len(index)
    assert result.crop_size == 256
    assert result.windows_per_acquisition == 1
    assert result.verified_chunks is False
    assert set(result.directions) == {"sar_to_optical", "optical_to_sar"}
    (root / "cache_index.json").unlink()
    with pytest.raises(SOPATChunkCachePreflightError, match="cannot read"):
        preflight_sopat_v4_chunk_cache(root, index)


def test_preflight_fails_closed_when_published_plan_is_missing_or_altered(
    tmp_path: Path,
) -> None:
    index = _sopat_index()
    root = tmp_path / "cache"
    _complete_cache(root, index)

    (root / "plan.json").unlink()
    with pytest.raises(SOPATChunkCachePreflightError, match="cannot read"):
        preflight_sopat_v4_chunk_cache(root, index)

    _complete_cache(root, index)
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["crop_size"] = 128
    _write_json(plan_path, plan)
    with pytest.raises(SOPATChunkCachePreflightError, match="plan digest differs"):
        preflight_sopat_v4_chunk_cache(root, index)


def test_preflight_rejects_protocol_route_and_target_leakage(tmp_path: Path) -> None:
    index = _sopat_index()
    root = tmp_path / "cache"
    _complete_cache(root, index)
    cache_index_path = root / "cache_index.json"
    cache_index = json.loads(cache_index_path.read_text(encoding="utf-8"))

    first = index.examples[0]
    route = json.loads((root / "routing.json").read_text(encoding="utf-8"))
    query = route["routes"][first.target.record_id]
    anchor = route["routes"][first.anchor_pair.registration_id]
    target_key = "optical_acquisition_id" if first.target_modality == "optical" else "sar_acquisition_id"
    source_key = "sar_acquisition_id" if first.source_modality == "sar" else "optical_acquisition_id"
    anchor[source_key] = query[target_key]
    _write_json(root / "routing.json", route)
    cache_index["routing_sha256"] = file_sha256(root / "routing.json")
    _write_json(cache_index_path, cache_index)
    with pytest.raises(SOPATChunkCachePreflightError, match="leaks"):
        preflight_sopat_v4_chunk_cache(root, index)

    _complete_cache(root, index)
    altered = list(index.examples)
    altered[0] = altered[0].__class__(
        **{**altered[0].__dict__, "provenance": altered[0].provenance.__class__(
            **{**altered[0].provenance.__dict__, "source_protocol_hash": "wrong"}
        )}
    )
    bad_index = SOPATIndexV4(config=index.config, examples=tuple(altered))
    with pytest.raises(SOPATChunkCachePreflightError, match="protocol differs"):
        preflight_sopat_v4_chunk_cache(root, bad_index)


def test_preflight_only_requires_materialized_sopat_roles(tmp_path: Path) -> None:
    index = _sopat_index()
    example = next(example for example in index.examples if example.task_mode == "forecast")
    single_example_index = SOPATIndexV4(config=index.config, examples=(example,))
    root = tmp_path / "cache"
    _complete_cache(root, index)

    routes = json.loads((root / "routing.json").read_text(encoding="utf-8"))["routes"]
    source_key = (
        "sar_acquisition_id" if example.source_modality == "sar" else "optical_acquisition_id"
    )
    target_key = (
        "optical_acquisition_id" if example.target_modality == "optical" else "sar_acquisition_id"
    )
    query_route = routes[example.target.record_id]
    anchor_route = routes[example.anchor_pair.registration_id]
    observation_routes = [
        routes[observation.measurement.record_id] for observation in example.observations
    ]
    input_ids = {
        anchor_route[source_key],
        anchor_route[target_key],
        *(route[source_key] for route in observation_routes),
    }
    target_id = query_route[target_key]
    required_ids = input_ids | {target_id}
    unused_query_source_id = query_route[source_key]
    unused_observation_target_ids = {route[target_key] for route in observation_routes}
    unused_ids = {unused_query_source_id, *unused_observation_target_ids}.difference(required_ids)
    assert unused_query_source_id not in required_ids
    assert unused_observation_target_ids.isdisjoint(required_ids)

    cache_index_path = root / "cache_index.json"
    cache_index = json.loads(cache_index_path.read_text(encoding="utf-8"))
    cache_index["acquisitions"] = [
        acquisition
        for acquisition in cache_index["acquisitions"]
        if acquisition["acquisition_id"] not in unused_ids
    ]
    _write_json(cache_index_path, cache_index)
    preflight_sopat_v4_chunk_cache(root, single_example_index)

    for missing_id in sorted(required_ids):
        _complete_cache(root, index)
        cache_index = json.loads(cache_index_path.read_text(encoding="utf-8"))
        cache_index["acquisitions"] = [
            acquisition
            for acquisition in cache_index["acquisitions"]
            if acquisition["acquisition_id"] != missing_id
        ]
        _write_json(cache_index_path, cache_index)
        with pytest.raises(SOPATChunkCachePreflightError, match="acquisition metadata is missing"):
            preflight_sopat_v4_chunk_cache(root, single_example_index)


def test_cache_wrapper_has_no_raw_source_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = _sopat_index()
    root = tmp_path / "cache"
    _complete_cache(root, index)

    def fail_if_constructed(*args: object, **kwargs: object) -> None:
        raise AssertionError("chunk dataset must not be built when cache chunks are absent")

    monkeypatch.setattr("sentinel_v4.cache.PairedTemporalChunkDataset", fail_if_constructed)
    with pytest.raises(SOPATChunkCachePreflightError, match="cache has no completed acquisitions"):
        empty = json.loads((root / "cache_index.json").read_text(encoding="utf-8"))
        empty["acquisitions"] = []
        _write_json(root / "cache_index.json", empty)
        sopat_chunk_dataset_from_cache(root, index, direction="sar_to_optical", split="train")


def test_local_raster_and_full_completed_chunk_backends_share_training_contract(
    tmp_path: Path,
) -> None:
    records = _local_records(tmp_path)
    manifest = tmp_path / "manifest" / "pairs.jsonl"
    from sentinel_v3.paired_temporal_data import write_pair_records
    from sentinel_v4.data import SOPATDirectionDataset

    write_pair_records(manifest, records)
    index = build_sopat_v4_index(manifest, splits=("train",), max_observations=3)
    cache_root = tmp_path / "full-complete-cache"
    _write_full_completed_cache(
        cache_root,
        index,
        records,
        manifest_sha256=file_sha256(manifest),
    )
    core_keys = (
        "source_anchor_values",
        "source_anchor_valid",
        "target_anchor_values",
        "target_anchor_valid",
        "observation_values",
        "observation_valid",
        "observation_days",
        "observation_present",
        "source_anchor_days",
        "target_anchor_days",
        "target_values",
        "target_valid",
        "high_frequency_valid",
        "high_frequency_eligible",
        "high_frequency_weight",
        "registration_shift_px",
        "registration_zero_ncc",
        "registration_best_ncc",
        "registration_evidence_supported",
        "anchor_days",
    )
    for direction in ("sar_to_optical", "optical_to_sar"):
        local = SOPATDirectionDataset.from_raster(
            index,
            manifest,
            direction=direction,
            split="train",
            crop_size=256,
            crop_attempts=1,
            minimum_valid_fraction=0.0,
            registration_audit=False,
            permute_observations=False,
        )
        chunk = sopat_chunk_dataset_from_cache(
            cache_root,
            index,
            direction=direction,
            split="train",
            window_mode="center",
            registration_audit=False,
            permute_observations=False,
            verify_chunks=True,
        )

        local_item = local[0]
        chunk_item = chunk[0]
        assert local_item["sopat_example_id"] == chunk_item["sopat_example_id"]
        assert local_item["sopat_direction"] == chunk_item["sopat_direction"] == direction
        for key in core_keys:
            assert isinstance(local_item[key], torch.Tensor)
            assert isinstance(chunk_item[key], torch.Tensor)
            assert local_item[key].shape == chunk_item[key].shape
            assert torch.allclose(local_item[key], chunk_item[key], atol=1e-3, equal_nan=True)


def test_full_chunk_preflight_fails_closed_without_publication_marker(tmp_path: Path) -> None:
    index = _sopat_index()
    cache_root = tmp_path / "partial"
    cache_root.mkdir()
    with pytest.raises(SOPATChunkCachePreflightError, match="cannot read"):
        preflight_sopat_v4_chunk_cache(cache_root, index)
