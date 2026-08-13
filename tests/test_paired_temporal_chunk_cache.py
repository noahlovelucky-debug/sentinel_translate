from __future__ import annotations

import os
import shutil
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import sentinel_v3.paired_temporal_chunk_cache as chunk_cache
from sentinel_v3.dataset_builder import PairRecord
from sentinel_v3.paired_temporal_chunk_cache import (
    ChunkCacheIntegrityError,
    PairedTemporalChunkDataset,
    assert_paired_temporal_chunk_cache_budget,
    build_paired_temporal_chunk_cache_plan,
    deterministic_chunk_windows,
    materialize_paired_temporal_chunk_cache,
    verify_paired_temporal_chunk_cache,
)
from sentinel_v3.paired_temporal_data import PairedTemporalRasterDataset, write_pair_records
from sentinel_v3.schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER


def _write_tiff(path: Path, values: np.ndarray, *, west: float = 500000.0) -> None:
    import rasterio
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
        transform=from_origin(west, 4100000, 10, 10),
    ) as source:
        source.write(values, 1)


def _record(
    root: Path,
    index: int,
    *,
    split: str,
    start: date,
    size: int = 1024,
    tile: str = "tile-chunk",
    source_offset: int = 0,
) -> PairRecord:
    s1_date = start + timedelta(days=index * 3)
    s2_date = s1_date + timedelta(days=index % 2)
    record_root = root / "raw" / split / f"record-{index:03d}"
    yy, xx = np.mgrid[:size, :size]
    optical = ((xx + yy + 1000 + index * 137) % 9000 + 1000).astype(np.uint16)
    sar = ((xx * 2 + yy + 6000 + index * 53) % 9000 + 1000).astype(np.uint16)
    scl = np.full((size, size), 4, dtype=np.uint8)
    if index % 5 == 0:
        scl[:32, :32] = 1
    s2: dict[str, str] = {}
    for channel in S2_CHANNEL_ORDER:
        path = record_root / "s2" / f"{channel}.tif"
        _write_tiff(path, optical)
        s2[channel] = os.path.relpath(path, root / "manifests")
    scl_path = record_root / "scl.tif"
    _write_tiff(scl_path, scl)
    sar_paths: dict[str, str] = {}
    for channel in SAR_CHANNEL_ORDER:
        path = record_root / "sar" / f"{channel}.tif"
        _write_tiff(path, sar)
        sar_paths[channel] = os.path.relpath(path, root / "manifests")
    return PairRecord(
        pair_id=f"2020:{tile}:{split}:{index:03d}:ascending",
        year=2020,
        tile=tile,
        tile_row=1,
        tile_col=1,
        split=split,
        refit_split="excluded",
        s2_date=s2_date.isoformat(),
        s1_date=s1_date.isoformat(),
        orbit="ascending",
        delta_days=index % 2,
        s2=s2,
        scl=os.path.relpath(scl_path, root / "manifests"),
        sar=sar_paths,
        clear_fraction=1.0,
        valid_fraction=1.0,
        width=size,
        height=size,
        crs="EPSG:32650",
        transform=[10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0],
        gsd=10.0,
    )


def _config(root: Path, manifest: Path) -> Path:
    path = root / "paired_temporal_v2_full.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "manifest": str(manifest),
                    "orbit": "ascending",
                    "anchor_pair_max_delta_days": 1,
                    "maximum_anchors_per_query": 1,
                    "horizon_days": 180,
                    "translation_max_delta_days": 1,
                    "minimum_observations": 1,
                    "maximum_observations": 3,
                    "crop_size": 256,
                    "train_split": "train",
                    "validation_split": "validation_temporal",
                    "task_modes": ["translation", "forecast"],
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def chunk_inputs(tmp_path: Path) -> tuple[Path, Path]:
    records = [
        *(
            _record(tmp_path, index, split="train", start=date(2020, 1, 1))
            for index in range(8)
        ),
        *(
            _record(
                tmp_path,
                index,
                split="validation_temporal",
                start=date(2021, 1, 1),
            )
            for index in range(8)
        ),
        _record(tmp_path, 99, split="test_temporal", start=date(2022, 1, 1)),
    ]
    manifest = tmp_path / "manifests" / "pairs.jsonl"
    write_pair_records(manifest, records)
    return _config(tmp_path, manifest), manifest


def _plan(config: Path, root: Path):
    return build_paired_temporal_chunk_cache_plan(
        config,
        destination_root=root,
        budget_bytes=10**12,
        minimum_free_bytes=0,
        windows_per_acquisition=4,
        free_bytes=10**12,
    )


def _resume_capacity_plan(config: Path, root: Path):
    return build_paired_temporal_chunk_cache_plan(
        config,
        destination_root=root,
        budget_bytes=10**12,
        minimum_free_bytes=1_000_000,
        windows_per_acquisition=4,
        free_bytes=10**12,
    )


def _metadata_bytes(root: Path, plan: object) -> dict[Path, bytes]:
    assert hasattr(plan, "indexes")
    index_paths = [root / entry.relative_path for entry in plan.indexes]  # type: ignore[union-attr]
    paths = [
        root / "cache_index.json",
        root / "routing.json",
        root / "plan.json",
        root / "provenance.json",
        *index_paths,
    ]
    return {path: path.read_bytes() for path in paths}


def test_deterministic_windows_keep_exact_sentinel_center_and_nonoverlap() -> None:
    first = deterministic_chunk_windows(tile="T31", width=2560, height=2560)
    second = deterministic_chunk_windows(tile="T31", width=2560, height=2560)

    assert first == second
    assert first[0].to_list() == [1152, 1152, 256, 256]
    assert len(first) == 64
    non_center = first[1:]
    assert len({(window.col, window.row) for window in non_center}) == 63
    for left_index, left in enumerate(non_center):
        for right in non_center[left_index + 1 :]:
            assert left.col + left.width <= right.col or right.col + right.width <= left.col or left.row + left.height <= right.row or right.row + right.height <= left.row


def test_plan_deduplicates_acquisitions_and_excludes_test(chunk_inputs: tuple[Path, Path], tmp_path: Path) -> None:
    config, _ = chunk_inputs
    plan = _plan(config, tmp_path / "cache")

    assert {(entry.direction, entry.split) for entry in plan.indexes} == {
        ("sar_to_optical", "train"),
        ("sar_to_optical", "validation_temporal"),
        ("optical_to_sar", "train"),
        ("optical_to_sar", "validation_temporal"),
    }
    assert all(route.split in {"train", "validation_temporal"} for route in plan.routes.values())
    assert all("test_temporal" not in path for acquisition in plan.acquisitions for path in acquisition.source_paths)
    assert len({acquisition.acquisition_id for acquisition in plan.acquisitions}) == len(plan.acquisitions)
    assert plan.report()["acquisitions"]["total"] == len(plan.acquisitions)  # type: ignore[index]


def test_plan_enforces_budget_and_reserve(chunk_inputs: tuple[Path, Path], tmp_path: Path) -> None:
    config, _ = chunk_inputs
    plan = build_paired_temporal_chunk_cache_plan(
        config,
        destination_root=tmp_path / "cache",
        budget_bytes=1,
        minimum_free_bytes=10,
        windows_per_acquisition=4,
        free_bytes=10,
    )

    assert not plan.allowed_to_materialize
    with pytest.raises(RuntimeError, match="exceeds budget"):
        assert_paired_temporal_chunk_cache_budget(plan)


def test_materialized_chunks_are_normalized_memmaps_and_resume_repairs_corruption(
    chunk_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    config, _ = chunk_inputs
    plan = _plan(config, tmp_path / "cache")
    result = materialize_paired_temporal_chunk_cache(plan, workers=1)

    assert result["copied_acquisitions"] == len(plan.acquisitions)
    assert verify_paired_temporal_chunk_cache(plan.destination_root)["valid"] is True
    acquisition = plan.acquisitions[0]
    values_path = plan.destination_root / acquisition.values_relative_path
    valid_path = plan.destination_root / acquisition.valid_relative_path
    values = np.load(values_path, mmap_mode="r", allow_pickle=False)
    valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
    assert isinstance(values, np.memmap)
    assert values.dtype == np.float16
    assert values.shape == (4, acquisition.channels, 256, 256)
    assert valid.dtype == np.uint8
    assert valid.shape == (4, 1, 256, 256)
    assert np.isfinite(values).all()
    assert np.abs(values).max() <= 1.0
    values_path.write_bytes(b"corrupt")

    repaired = materialize_paired_temporal_chunk_cache(plan, workers=1)
    assert repaired["copied_acquisitions"] >= 1
    assert verify_paired_temporal_chunk_cache(plan.destination_root)["valid"] is True

    resumed = materialize_paired_temporal_chunk_cache(plan, workers=1)
    assert resumed["copied_acquisitions"] == 0
    assert resumed["reused_acquisitions"] == len(plan.acquisitions)


def test_resume_capacity_charges_only_missing_chunks(
    chunk_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = chunk_inputs
    plan = _resume_capacity_plan(config, tmp_path / "cache")
    materialize_paired_temporal_chunk_cache(plan, workers=1)
    missing = plan.acquisitions[::2]
    for acquisition in missing:
        shutil.rmtree(plan.destination_root / acquisition.relative_directory)
    grids = {grid.grid_id: grid for grid in plan.grids}
    remaining = chunk_cache._remaining_materialization_bytes(missing, grids)
    current_free = plan.minimum_free_bytes + remaining
    assert current_free - plan.estimated_target_bytes < plan.minimum_free_bytes
    monkeypatch.setattr(
        chunk_cache.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=current_free),
    )

    result = materialize_paired_temporal_chunk_cache(plan, resume=True, workers=1)

    assert result["copied_acquisitions"] == len(missing)
    assert result["reused_acquisitions"] == len(plan.acquisitions) - len(missing)
    assert verify_paired_temporal_chunk_cache(plan.destination_root)["valid"] is True


def test_resume_capacity_rejection_preserves_completed_metadata(
    chunk_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = chunk_inputs
    plan = _resume_capacity_plan(config, tmp_path / "cache")
    materialize_paired_temporal_chunk_cache(plan, workers=1)
    metadata_before = _metadata_bytes(plan.destination_root, plan)
    missing = (plan.acquisitions[0],)
    shutil.rmtree(plan.destination_root / missing[0].relative_directory)
    grids = {grid.grid_id: grid for grid in plan.grids}
    remaining = chunk_cache._remaining_materialization_bytes(missing, grids)
    current_free = plan.minimum_free_bytes + remaining - 1
    monkeypatch.setattr(
        chunk_cache.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=current_free),
    )

    with pytest.raises(RuntimeError, match="required free-space reserve"):
        materialize_paired_temporal_chunk_cache(plan, resume=True, workers=1)

    assert _metadata_bytes(plan.destination_root, plan) == metadata_before


def test_no_resume_capacity_charges_every_acquisition_before_metadata_write(
    chunk_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = chunk_inputs
    plan = _resume_capacity_plan(config, tmp_path / "cache")
    materialize_paired_temporal_chunk_cache(plan, workers=1)
    metadata_before = _metadata_bytes(plan.destination_root, plan)
    grids = {grid.grid_id: grid for grid in plan.grids}
    reusable_remaining = chunk_cache._remaining_materialization_bytes((), grids)
    full_remaining = chunk_cache._remaining_materialization_bytes(plan.acquisitions, grids)
    current_free = plan.minimum_free_bytes + reusable_remaining
    assert current_free - full_remaining < plan.minimum_free_bytes
    monkeypatch.setattr(
        chunk_cache.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=current_free),
    )

    with pytest.raises(RuntimeError, match="required free-space reserve"):
        materialize_paired_temporal_chunk_cache(plan, resume=False, workers=1)

    assert _metadata_bytes(plan.destination_root, plan) == metadata_before


def test_chunk_dataset_both_directions_uses_local_memmap_without_target_leakage(
    chunk_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = chunk_inputs
    plan = _plan(config, tmp_path / "cache")
    materialize_paired_temporal_chunk_cache(plan, workers=1)
    original_load = np.load
    calls: list[tuple[object, object]] = []

    def checked_load(*args: object, **kwargs: object):
        calls.append((args[0], kwargs.get("mmap_mode")))
        assert kwargs.get("mmap_mode") == "r"
        return original_load(*args, **kwargs)

    monkeypatch.setattr(chunk_cache.np, "load", checked_load)
    for direction in ("sar_to_optical", "optical_to_sar"):
        dataset = PairedTemporalChunkDataset(
            plan.destination_root,
            direction=direction,  # type: ignore[arg-type]
            split="train",
            window_mode="all",
            registration_audit=False,
            pad_observations_to=3,
        )
        assert len(dataset) == len(dataset.samples) * 4
        item = dataset[0]
        assert item["source_anchor_values"].shape[-2:] == (256, 256)
        assert item["target_values"].shape[-2:] == (256, 256)
        assert item["observation_present"].dtype == torch.bool
        sample = dataset.samples[0]
        query = dataset.routes[sample.query_pair_id]
        anchor = dataset.routes[sample.anchor_pair_id]
        source_attr = "sar_acquisition_id" if direction == "sar_to_optical" else "optical_acquisition_id"
        target_attr = "optical_acquisition_id" if direction == "sar_to_optical" else "sar_acquisition_id"
        input_ids = {getattr(anchor, source_attr), getattr(anchor, target_attr)}
        input_ids.update(getattr(dataset.routes[pair_id], source_attr) for pair_id in sample.observation_pair_ids)
        assert getattr(query, target_attr) not in input_ids
    assert calls


def test_validation_defaults_center_only_and_no_raw_open(
    chunk_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = chunk_inputs
    plan = _plan(config, tmp_path / "cache")
    materialize_paired_temporal_chunk_cache(plan, workers=1)
    dataset = PairedTemporalChunkDataset(
        plan.destination_root,
        direction="sar_to_optical",
        split="validation_temporal",
        registration_audit=False,
    )
    assert dataset.windows_per_sample == 1
    assert len(dataset) == len(dataset.samples)
    monkeypatch.setitem(__import__("sys").modules, "rasterio", None)
    item = dataset[0]
    assert item["target_values"].shape == (10, 256, 256)


def test_chunk_center_window_matches_raw_center_contract(
    chunk_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    config, manifest = chunk_inputs
    plan = _plan(config, tmp_path / "cache")
    materialize_paired_temporal_chunk_cache(plan, workers=1)
    chunk = PairedTemporalChunkDataset(
        plan.destination_root,
        direction="sar_to_optical",
        split="validation_temporal",
        window_mode="center",
        registration_audit=False,
    )
    raw = PairedTemporalRasterDataset(
        manifest,
        chunk.index,
        crop_size=256,
        crop_mode="center",
        registration_audit=False,
    )

    chunk_item = chunk[0]
    raw_item = raw[0]
    for key in (
        "source_anchor_values",
        "source_anchor_valid",
        "target_anchor_values",
        "target_anchor_valid",
        "observation_values",
        "observation_valid",
        "target_values",
        "target_valid",
    ):
        assert torch.allclose(chunk_item[key], raw_item[key], atol=1e-3)  # type: ignore[arg-type]


def test_chunk_center_keeps_low_coverage_masks_and_rejects_zero_support(
    chunk_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    config, _ = chunk_inputs
    plan = _plan(config, tmp_path / "cache")
    materialize_paired_temporal_chunk_cache(plan, workers=1)
    dataset = PairedTemporalChunkDataset(
        plan.destination_root,
        direction="sar_to_optical",
        split="validation_temporal",
        window_mode="center",
        minimum_valid_fraction=1.0,
        registration_audit=False,
    )
    sample = dataset.samples[0]
    grid = dataset.grids[dataset.routes[sample.query_pair_id].grid_id]
    query_route = dataset.routes[sample.query_pair_id]
    target = dataset.acquisitions[query_route.optical_acquisition_id]
    valid_path = dataset.cache_root / target.valid_relative_path
    valid = np.load(valid_path, mmap_mode="r+", allow_pickle=False)
    valid[grid.center_window_index].fill(0)
    valid[grid.center_window_index, 0, 0, 0] = 1
    valid.flush()
    del valid

    low_coverage = dataset[0]
    assert int(low_coverage["target_valid"].sum()) == 1  # type: ignore[union-attr]
    assert int(low_coverage["target_anchor_valid"].sum()) == 256 * 256  # type: ignore[union-attr]
    assert int(
        (low_coverage["target_valid"] * low_coverage["target_anchor_valid"]).sum()
    ) == 1  # type: ignore[operator]
    dataset.close()

    valid = np.load(valid_path, mmap_mode="r+", allow_pickle=False)
    valid[grid.center_window_index].fill(0)
    valid.flush()
    del valid
    zero_support = PairedTemporalChunkDataset(
        plan.destination_root,
        direction="sar_to_optical",
        split="validation_temporal",
        window_mode="center",
        registration_audit=False,
    )
    with pytest.raises(ChunkCacheIntegrityError, match="no evaluable target/anchor pixels"):
        zero_support[0]


def test_mmap_cache_is_bounded_lru_and_closes_evicted_arrays(
    chunk_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = chunk_inputs
    plan = _plan(config, tmp_path / "cache")
    materialize_paired_temporal_chunk_cache(plan, workers=1)
    dataset = PairedTemporalChunkDataset(
        plan.destination_root,
        direction="sar_to_optical",
        split="train",
        registration_audit=False,
        max_mmap_arrays=2,
    )
    closed: list[np.ndarray] = []
    original_close = chunk_cache._close_memmap

    def checked_close(array: np.ndarray) -> None:
        closed.append(array)
        original_close(array)

    monkeypatch.setattr(chunk_cache, "_close_memmap", checked_close)
    paths = [
        plan.destination_root / acquisition.values_relative_path
        for acquisition in plan.acquisitions[:3]
    ]
    for path in paths:
        dataset._mmap(path.relative_to(plan.destination_root))
    assert len(dataset._mmap_cache) == 2
    assert len(closed) == 1
    # Touch the second array, then insert another: the first surviving array
    # must be evicted, proving ordered LRU rather than arbitrary deletion.
    dataset._mmap(paths[1].relative_to(plan.destination_root))
    dataset._mmap(plan.acquisitions[3].values_relative_path)
    assert len(dataset._mmap_cache) == 2
    assert len(closed) == 2
    state = dataset.__getstate__()
    assert not dataset._mmap_cache
    assert not state["_mmap_cache"]
    assert len(closed) == 4


def test_missing_chunk_hard_fails_and_train_validation_are_isolated(
    chunk_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    config, _ = chunk_inputs
    plan = _plan(config, tmp_path / "cache")
    materialize_paired_temporal_chunk_cache(plan, workers=1)
    train = PairedTemporalChunkDataset(
        plan.destination_root,
        direction="sar_to_optical",
        split="train",
        registration_audit=False,
    )
    validation = PairedTemporalChunkDataset(
        plan.destination_root,
        direction="sar_to_optical",
        split="validation_temporal",
        registration_audit=False,
    )
    assert {sample.split for sample in train.samples} == {"train"}
    assert {sample.split for sample in validation.samples} == {"validation_temporal"}
    sample = train.samples[0]
    route = train.routes[sample.anchor_pair_id]
    acquisition = train.acquisitions[route.sar_acquisition_id]
    (train.cache_root / acquisition.values_relative_path).unlink()
    with pytest.raises(ChunkCacheIntegrityError, match="missing local chunk"):
        train[0]
