from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from sentinel_v3 import paired_temporal_data as paired_temporal_data_module
from sentinel_v3.dataset_builder import PairRecord
from sentinel_v3.paired_temporal_data import (
    FORECAST,
    OPTICAL_TO_SAR,
    SAR_TO_OPTICAL,
    TRANSLATION,
    PairedTemporalIndex,
    PairedTemporalIndexConfig,
    PairedTemporalRasterDataset,
    assert_paired_temporal_causality,
    build_paired_temporal_index,
    collate_paired_temporal,
    load_paired_temporal_index,
    write_paired_temporal_index,
)
from sentinel_v3.paired_temporal_training import (
    forward_paired_temporal,
    paired_tensor_batch,
)
from sentinel_v3.paired_temporal_v2 import (
    PairedTemporalConfig,
    SparsePairedAnchorTransport,
)
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
    s1_date: str,
    s2_date: str,
    split: str = "train",
    tile: str = "tile-a",
    year: int = 2020,
    orbit: str = "ascending",
    size: int = 8,
    west: float = 500000.0,
    invalid_optical: bool = False,
    invalid_sar: bool = False,
    textured: bool = False,
) -> PairRecord:
    record_root = root / f"record-{index}"
    s2_values = np.full((size, size), 5000 + index * 100, dtype=np.uint16)
    sar_values = np.full((size, size), 7000 + index * 50, dtype=np.uint16)
    scl_values = np.full((size, size), 4, dtype=np.uint8)
    if textured:
        yy, xx = np.mgrid[:size, :size]
        structure = ((xx // 3 + yy // 5) % 2).astype(np.uint16)
        s2_values = 2500 + structure * 5000
        sar_values = 4000 + structure * 5000
    if invalid_optical:
        s2_values[0, 0] = 0
        scl_values[0, 0] = 1
    if invalid_sar:
        sar_values.fill(0)
    s2: dict[str, str] = {}
    for channel in S2_CHANNEL_ORDER:
        path = record_root / "s2" / f"{channel}.tif"
        _write_tiff(path, s2_values, west=west)
        s2[channel] = str(path)
    scl_path = record_root / "scl.tif"
    _write_tiff(scl_path, scl_values, west=west)
    sar: dict[str, str] = {}
    for channel in SAR_CHANNEL_ORDER:
        path = record_root / "sar" / f"{channel}.tif"
        _write_tiff(path, sar_values, west=west)
        sar[channel] = str(path)
    return PairRecord(
        pair_id=f"{year}:{tile}:{index:02d}:{orbit}",
        year=year,
        tile=tile,
        tile_row=1,
        tile_col=1,
        split=split,
        refit_split="excluded",
        s2_date=s2_date,
        s1_date=s1_date,
        orbit=orbit,
        delta_days=abs((np.datetime64(s2_date, "D") - np.datetime64(s1_date, "D")).astype(int)),
        s2=s2,
        scl=str(scl_path),
        sar=sar,
        clear_fraction=1.0,
        valid_fraction=1.0,
        width=size,
        height=size,
        crs="EPSG:32650",
        transform=[10.0, 0.0, west, 0.0, -10.0, 4100000.0],
        gsd=10.0,
    )


def _memory_record(
    index: int,
    *,
    s1_date: str,
    s2_date: str,
    tile: str = "tile-benchmark",
) -> PairRecord:
    """Manifest-only record for index selection and timing tests."""

    asset_prefix = f"remote-assets/{tile}/{index:04d}"
    return PairRecord(
        pair_id=f"2020:{tile}:{index:04d}:ascending",
        year=2020,
        tile=tile,
        tile_row=2,
        tile_col=2,
        split="train",
        refit_split="excluded",
        s2_date=s2_date,
        s1_date=s1_date,
        orbit="ascending",
        delta_days=abs((np.datetime64(s2_date, "D") - np.datetime64(s1_date, "D")).astype(int)),
        s2={
            channel: f"{asset_prefix}/s2-{channel}.tif"
            for channel in S2_CHANNEL_ORDER
        },
        scl=f"{asset_prefix}/scl.tif",
        sar={
            channel: f"{asset_prefix}/sar-{channel}.tif"
            for channel in SAR_CHANNEL_ORDER
        },
        clear_fraction=1.0,
        valid_fraction=1.0,
        width=256,
        height=256,
        crs="EPSG:32650",
        transform=[10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0],
        gsd=10.0,
    )


def _memory_records(count: int) -> list[PairRecord]:
    first_day = date(2020, 1, 1)
    records: list[PairRecord] = []
    for index in range(count):
        source_day = first_day + timedelta(days=index)
        target_day = source_day + timedelta(days=index % 2)
        records.append(
            _memory_record(
                index,
                s1_date=source_day.isoformat(),
                s2_date=target_day.isoformat(),
            )
        )
    return records


@pytest.fixture
def paired_records(tmp_path: Path) -> list[PairRecord]:
    return [
        _record(tmp_path, 0, s1_date="2020-01-01", s2_date="2020-01-01"),
        _record(tmp_path, 1, s1_date="2020-01-03", s2_date="2020-01-03"),
        _record(tmp_path, 2, s1_date="2020-01-05", s2_date="2020-01-05"),
        _record(
            tmp_path,
            3,
            s1_date="2020-01-10",
            s2_date="2020-01-10",
            invalid_optical=True,
        ),
        _record(tmp_path, 4, s1_date="2020-01-13", s2_date="2020-01-14"),
        _record(tmp_path, 5, s1_date="2020-01-15", s2_date="2020-01-17"),
    ]


def _sample_for(index: PairedTemporalIndex, query_pair_id: str):
    return next(sample for sample in index if sample.query_pair_id == query_pair_id)


def _set_optical_valid_mask(record: PairRecord, mask: np.ndarray) -> None:
    values = np.where(mask, 5000, 0).astype(np.uint16)
    scl = np.where(mask, 4, 1).astype(np.uint8)
    for channel in S2_CHANNEL_ORDER:
        _write_tiff(Path(record.s2[channel]), values)
    _write_tiff(Path(record.scl), scl)


def test_paired_index_builds_variable_sequences_and_task_modes(
    paired_records: list[PairRecord], tmp_path: Path
) -> None:
    other_split = _record(tmp_path, 20, s1_date="2020-01-09", s2_date="2020-01-09", split="test")
    other_orbit = _record(
        tmp_path,
        21,
        s1_date="2020-01-09",
        s2_date="2020-01-09",
        orbit="descending",
    )
    other_grid = _record(tmp_path, 22, s1_date="2020-01-09", s2_date="2020-01-09", west=500010)
    index = build_paired_temporal_index(
        [*paired_records, other_split, other_orbit, other_grid],
        direction=SAR_TO_OPTICAL,
        min_observations=1,
        max_observations=None,
    )

    translation = _sample_for(index, paired_records[3].pair_id)
    one_frame = _sample_for(index, paired_records[1].pair_id)
    assert one_frame.observation_count == 1
    assert one_frame.task_mode == TRANSLATION
    assert translation.anchor_pair_id == paired_records[2].pair_id
    assert translation.observation_pair_ids == (
        paired_records[0].pair_id,
        paired_records[1].pair_id,
        paired_records[3].pair_id,
    )
    assert translation.observation_dates == ("2020-01-01", "2020-01-03", "2020-01-10")
    assert translation.task_mode == TRANSLATION
    assert translation.source_anchor_date == "2020-01-05"
    assert translation.target_anchor_date == "2020-01-05"

    one_day_translation = _sample_for(index, paired_records[4].pair_id)
    assert one_day_translation.anchor_pair_id == paired_records[3].pair_id
    assert one_day_translation.observation_pair_ids == (
        paired_records[0].pair_id,
        paired_records[1].pair_id,
        paired_records[2].pair_id,
        paired_records[4].pair_id,
    )
    assert one_day_translation.observation_dates[-1] == "2020-01-13"
    assert one_day_translation.task_mode == TRANSLATION

    forecast = _sample_for(index, paired_records[5].pair_id)
    assert forecast.anchor_pair_id == paired_records[4].pair_id
    assert forecast.observation_dates[-1] == "2020-01-15"
    assert forecast.task_mode == FORECAST
    assert all(other.pair_id not in translation.observation_pair_ids for other in (other_split, other_orbit, other_grid))
    assert_paired_temporal_causality(index, [*paired_records, other_split, other_orbit, other_grid])


def test_paired_index_observation_limits_and_jsonl_round_trip(
    paired_records: list[PairRecord], tmp_path: Path
) -> None:
    index = build_paired_temporal_index(
        paired_records,
        direction=SAR_TO_OPTICAL,
        min_observations=2,
        max_observations=2,
        max_samples=4,
    )
    assert all(2 <= sample.observation_count <= 2 for sample in index)
    path = tmp_path / "indices" / "paired-temporal.jsonl"
    write_paired_temporal_index(path, index)
    loaded = load_paired_temporal_index(path)
    assert loaded.config == index.config
    assert loaded.samples == index.samples
    assert load_paired_temporal_index(path, start=1, limit=1).samples == index.samples[1:2]
    assert index.subset(start=1, limit=1).samples == index.samples[1:2]


def test_paired_index_can_train_multiple_registered_anchors_per_query(
    paired_records: list[PairRecord],
) -> None:
    index = build_paired_temporal_index(
        paired_records,
        direction=SAR_TO_OPTICAL,
        max_anchors_per_query=3,
    )
    samples = [sample for sample in index if sample.query_pair_id == paired_records[3].pair_id]
    assert {sample.anchor_pair_id for sample in samples} == {
        paired_records[2].pair_id,
        paired_records[1].pair_id,
        paired_records[0].pair_id,
    }


def test_paired_index_keeps_causal_history_across_calendar_years(tmp_path: Path) -> None:
    records = [
        _record(
            tmp_path,
            0,
            s1_date="2020-12-20",
            s2_date="2020-12-20",
            year=2020,
        ),
        _record(
            tmp_path,
            1,
            s1_date="2020-12-28",
            s2_date="2020-12-28",
            year=2020,
        ),
        _record(
            tmp_path,
            2,
            s1_date="2021-01-05",
            s2_date="2021-01-05",
            year=2021,
        ),
    ]
    index = build_paired_temporal_index(records, direction=SAR_TO_OPTICAL)
    sample = _sample_for(index, records[-1].pair_id)
    assert sample.year == 2021
    assert sample.anchor_pair_id == records[1].pair_id
    assert sample.observation_pair_ids == (records[0].pair_id, records[-1].pair_id)
    assert_paired_temporal_causality(index, records)


def test_paired_causality_rejects_query_target_asset_leakage(
    paired_records: list[PairRecord], tmp_path: Path
) -> None:
    index = build_paired_temporal_index(
        paired_records,
        direction=SAR_TO_OPTICAL,
        min_observations=1,
    )
    sample = _sample_for(index, paired_records[3].pair_id)
    leaked = _record(tmp_path, 30, s1_date="2020-01-09", s2_date="2020-01-09")
    leaked = replace(
        leaked,
        sar={"vv": paired_records[3].s2["blue"], "vh": paired_records[3].s2["green"]},
    )
    tampered = replace(
        sample,
        sample_id="leaked-query-target",
        observation_pair_ids=(leaked.pair_id,),
        observation_dates=(leaked.s1_date,),
        task_mode=TRANSLATION,
    )
    tampered_index = PairedTemporalIndex(config=index.config, samples=(tampered,))
    with pytest.raises(AssertionError, match="query target asset"):
        assert_paired_temporal_causality(tampered_index, [*paired_records, leaked])


@pytest.mark.parametrize(
    ("direction", "source_channels", "target_channels"),
    (
        (SAR_TO_OPTICAL, len(SAR_CHANNEL_ORDER), len(S2_CHANNEL_ORDER)),
        (OPTICAL_TO_SAR, len(S2_CHANNEL_ORDER), len(SAR_CHANNEL_ORDER)),
    ),
)
def test_paired_dataset_emits_padded_v3_units_and_directional_shapes(
    paired_records: list[PairRecord],
    direction: str,
    source_channels: int,
    target_channels: int,
) -> None:
    index = build_paired_temporal_index(
        paired_records,
        direction=direction,
        min_observations=1,
    )
    sample = _sample_for(index, paired_records[3].pair_id)
    dataset = PairedTemporalRasterDataset(
        paired_records,
        PairedTemporalIndex(config=index.config, samples=(sample,)),
        crop_size=4,
        minimum_valid_fraction=0.8,
        max_observations=4,
        cache_in_memory=True,
    )
    item = dataset[0]
    again = dataset[0]

    assert set(item) == {
        "source_anchor_values",
        "source_anchor_valid",
        "target_anchor_values",
        "target_anchor_valid",
        "observation_values",
        "observation_valid",
        "observation_days",
        "observation_present",
        "anchor_days",
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
        "sample_id",
        "direction",
        "task_mode",
    }
    assert item["direction"] == direction
    assert item["task_mode"] == TRANSLATION
    assert item["source_anchor_values"].shape == (source_channels, 4, 4)  # type: ignore[union-attr]
    assert item["source_anchor_valid"].shape == (1, 4, 4)  # type: ignore[union-attr]
    assert item["target_anchor_values"].shape == (target_channels, 4, 4)  # type: ignore[union-attr]
    assert item["target_anchor_valid"].shape == (1, 4, 4)  # type: ignore[union-attr]
    assert item["observation_values"].shape == (4, source_channels, 4, 4)  # type: ignore[union-attr]
    assert item["observation_valid"].shape == (4, 1, 4, 4)  # type: ignore[union-attr]
    assert item["target_values"].shape == (target_channels, 4, 4)  # type: ignore[union-attr]
    assert item["target_valid"].shape == (1, 4, 4)  # type: ignore[union-attr]
    assert item["source_anchor_values"].dtype == torch.float32  # type: ignore[union-attr]
    assert item["observation_present"].dtype == torch.bool  # type: ignore[union-attr]
    assert torch.equal(item["observation_present"], torch.tensor([True, True, True, False]))  # type: ignore[arg-type]
    assert torch.all(item["observation_days"] <= 0)  # type: ignore[operator]
    assert float(item["anchor_days"]) < 0  # type: ignore[arg-type]
    assert item["anchor_days"] == item["target_anchor_days"]  # type: ignore[comparison-overlap]
    assert float(item["source_anchor_days"]) < 0  # type: ignore[arg-type]
    assert torch.all(item["source_anchor_values"].abs() <= 1.0)  # type: ignore[union-attr]
    assert torch.all(item["target_anchor_values"].abs() <= 1.0)  # type: ignore[union-attr]
    assert torch.all(item["observation_values"].abs() <= 1.0)  # type: ignore[union-attr]
    assert torch.all(item["target_values"].abs() <= 1.0)  # type: ignore[union-attr]
    assert torch.equal(item["observation_values"], again["observation_values"])  # type: ignore[arg-type]
    item["observation_values"][0, 0, 0, 0] = 123.0  # type: ignore[index]
    assert dataset[0]["observation_values"][0, 0, 0, 0] != 123.0  # type: ignore[index]

    assert torch.all((item["target_valid"] == 0) | (item["target_valid"] == 1))  # type: ignore[operator]
    assert item["high_frequency_valid"].shape == (1, 4, 4)  # type: ignore[union-attr]
    assert not bool(item["high_frequency_eligible"])  # type: ignore[arg-type]
    assert float(item["high_frequency_weight"]) == 0.0  # type: ignore[arg-type]
    assert float(item["registration_shift_px"]) == 0.0  # type: ignore[arg-type]
    assert not bool(item["registration_evidence_supported"])  # type: ignore[arg-type]


def test_paired_dataset_rejects_non_fusion_crop_size(paired_records: list[PairRecord]) -> None:
    index = build_paired_temporal_index(paired_records, direction=SAR_TO_OPTICAL)
    with pytest.raises(ValueError, match="divisible by four"):
        PairedTemporalRasterDataset(paired_records, index, crop_size=6)


def test_center_crop_mode_is_fixed_and_ignores_random_retry(
    paired_records: list[PairRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    index = build_paired_temporal_index(paired_records, direction=SAR_TO_OPTICAL)
    sample = _sample_for(index, paired_records[3].pair_id)
    selected = PairedTemporalIndex(config=index.config, samples=(sample,))
    dataset = PairedTemporalRasterDataset(
        paired_records,
        selected,
        crop_size=4,
        crop_attempts=99,
        crop_mode="center",
        registration_audit=False,
    )
    windows: list[tuple[int, int, int, int]] = []
    original_read = dataset._read_modality_window

    def read_window(record: PairRecord, modality: str, window: tuple[int, int, int, int]):
        windows.append(window)
        return original_read(record, modality, window)  # type: ignore[arg-type]

    monkeypatch.setattr(dataset, "_read_modality_window", read_window)
    first = dataset[0]
    second = dataset[0]
    other_seed = PairedTemporalRasterDataset(
        paired_records,
        selected,
        crop_size=4,
        crop_attempts=1,
        crop_mode="center",
        seed=999,
        registration_audit=False,
    )[0]

    assert windows
    assert set(windows) == {(2, 2, 4, 4)}
    assert torch.equal(first["target_values"], second["target_values"])  # type: ignore[arg-type]
    assert torch.equal(first["target_values"], other_seed["target_values"])  # type: ignore[arg-type]


def test_random_valid_crop_changes_by_epoch_and_replays_deterministically(
    paired_records: list[PairRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    index = build_paired_temporal_index(paired_records, direction=SAR_TO_OPTICAL)
    sample = _sample_for(index, paired_records[3].pair_id)
    selected = PairedTemporalIndex(config=index.config, samples=(sample,))
    dataset = PairedTemporalRasterDataset(
        paired_records,
        selected,
        crop_size=4,
        crop_attempts=1,
        crop_mode="random_valid",
        seed=17,
        cache_in_memory=True,
        registration_audit=False,
    )
    windows: list[tuple[int, int, int, int]] = []
    original_read = dataset._read_modality_window

    def read_window(record: PairRecord, modality: str, window: tuple[int, int, int, int]):
        windows.append(window)
        return original_read(record, modality, window)  # type: ignore[arg-type]

    monkeypatch.setattr(dataset, "_read_modality_window", read_window)

    dataset.set_epoch(3)
    first = dataset[0]
    first_window = windows[-1]
    dataset.set_epoch(4)
    dataset[0]
    second_window = windows[-1]
    dataset.set_epoch(3)
    replay = dataset[0]
    replay_window = windows[-1]

    assert first_window != second_window
    assert first_window == replay_window
    assert torch.equal(first["target_values"], replay["target_values"])  # type: ignore[arg-type]


def test_center_crop_is_unchanged_across_epochs(
    paired_records: list[PairRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    index = build_paired_temporal_index(paired_records, direction=SAR_TO_OPTICAL)
    sample = _sample_for(index, paired_records[3].pair_id)
    selected = PairedTemporalIndex(config=index.config, samples=(sample,))
    dataset = PairedTemporalRasterDataset(
        paired_records,
        selected,
        crop_size=4,
        crop_mode="center",
        cache_in_memory=True,
        registration_audit=False,
    )
    windows: list[tuple[int, int, int, int]] = []
    original_read = dataset._read_modality_window

    def read_window(record: PairRecord, modality: str, window: tuple[int, int, int, int]):
        windows.append(window)
        return original_read(record, modality, window)  # type: ignore[arg-type]

    monkeypatch.setattr(dataset, "_read_modality_window", read_window)

    dataset.set_epoch(3)
    first = dataset[0]
    dataset.set_epoch(4)
    second = dataset[0]

    assert set(windows) == {(2, 2, 4, 4)}
    assert torch.equal(first["target_values"], second["target_values"])  # type: ignore[arg-type]


def test_persistent_worker_observes_shared_random_crop_epoch(
    paired_records: list[PairRecord],
) -> None:
    target = paired_records[3]
    coordinates = np.arange(1, 65, dtype=np.uint16).reshape(8, 8)
    for channel in S2_CHANNEL_ORDER:
        _write_tiff(Path(target.s2[channel]), coordinates)
    _write_tiff(Path(target.scl), np.full((8, 8), 4, dtype=np.uint8))
    index = build_paired_temporal_index(paired_records, direction=SAR_TO_OPTICAL)
    sample = _sample_for(index, target.pair_id)
    dataset = PairedTemporalRasterDataset(
        paired_records,
        PairedTemporalIndex(config=index.config, samples=(sample,)),
        crop_size=4,
        crop_attempts=1,
        crop_mode="random_valid",
        seed=17,
        registration_audit=False,
    )
    dataset.set_epoch(3)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    try:
        first = next(iter(loader))["target_values"].clone()
        dataset.set_epoch(4)
        second = next(iter(loader))["target_values"].clone()
        dataset.set_epoch(3)
        replay = next(iter(loader))["target_values"].clone()
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()

    assert not torch.equal(first, second)
    assert torch.equal(first, replay)


def test_center_crop_keeps_low_coverage_masks_for_masked_evaluation(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, index, s1_date=f"2020-01-{day:02d}", s2_date=f"2020-01-{day:02d}")
        for index, day in enumerate((1, 3, 5, 10))
    ]
    low_coverage = np.zeros((8, 8), dtype=bool)
    low_coverage[3, 3] = True
    _set_optical_valid_mask(records[-1], low_coverage)
    index = build_paired_temporal_index(records, direction=SAR_TO_OPTICAL)
    sample = _sample_for(index, records[-1].pair_id)
    selected = PairedTemporalIndex(config=index.config, samples=(sample,))

    center = PairedTemporalRasterDataset(
        records,
        selected,
        crop_size=4,
        crop_mode="center",
        minimum_valid_fraction=0.80,
        registration_audit=False,
    )
    item = center[0]

    assert int(item["target_valid"].sum()) == 1  # type: ignore[union-attr]
    assert int(item["target_anchor_valid"].sum()) == 16  # type: ignore[union-attr]
    assert int((item["target_valid"] * item["target_anchor_valid"]).sum()) == 1  # type: ignore[operator]
    with pytest.raises(RuntimeError, match="insufficient valid paired temporal crops"):
        PairedTemporalRasterDataset(
            records,
            selected,
            crop_size=4,
            crop_attempts=2,
            crop_mode="random_valid",
            minimum_valid_fraction=0.80,
            registration_audit=False,
        )[0]


def test_center_crop_rejects_zero_evaluable_target_anchor_support(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, index, s1_date=f"2020-01-{day:02d}", s2_date=f"2020-01-{day:02d}")
        for index, day in enumerate((1, 3, 5, 10))
    ]
    _set_optical_valid_mask(records[-1], np.zeros((8, 8), dtype=bool))
    index = build_paired_temporal_index(records, direction=SAR_TO_OPTICAL)
    sample = _sample_for(index, records[-1].pair_id)

    with pytest.raises(RuntimeError, match="no evaluable target/anchor pixels"):
        PairedTemporalRasterDataset(
            records,
            PairedTemporalIndex(config=index.config, samples=(sample,)),
            crop_size=4,
            crop_mode="center",
            minimum_valid_fraction=0.0,
            registration_audit=False,
        )[0]


def test_one_day_translation_keeps_its_actual_observation_day(
    paired_records: list[PairRecord],
) -> None:
    index = build_paired_temporal_index(paired_records, direction=SAR_TO_OPTICAL)
    sample = _sample_for(index, paired_records[4].pair_id)
    assert sample.task_mode == TRANSLATION
    dataset = PairedTemporalRasterDataset(
        paired_records,
        PairedTemporalIndex(config=index.config, samples=(sample,)),
        crop_size=4,
    )
    item = dataset[0]
    present_count = int(item["observation_present"].sum())  # type: ignore[union-attr]
    assert item["observation_days"][present_count - 1] == -1  # type: ignore[index]
    assert not bool(item["high_frequency_eligible"])  # type: ignore[arg-type]
    assert float(item["high_frequency_weight"]) == 0.0  # type: ignore[arg-type]


def test_forecast_is_never_a_deterministic_high_frequency_label(
    paired_records: list[PairRecord],
) -> None:
    index = build_paired_temporal_index(paired_records, direction=SAR_TO_OPTICAL)
    sample = _sample_for(index, paired_records[5].pair_id)
    assert sample.task_mode == FORECAST
    dataset = PairedTemporalRasterDataset(
        paired_records,
        PairedTemporalIndex(config=index.config, samples=(sample,)),
        crop_size=4,
    )
    item = dataset[0]
    assert not bool(item["high_frequency_eligible"])  # type: ignore[arg-type]
    assert float(item["high_frequency_weight"]) == 0.0  # type: ignore[arg-type]
    assert torch.all(item["high_frequency_valid"] == 0)  # type: ignore[operator]
    assert torch.isinf(item["registration_shift_px"])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source_date", "expected_weight"),
    (("2020-01-05", 1.0), ("2020-01-04", 0.25)),
)
def test_structurally_aligned_translation_receives_time_weighted_detail_label(
    tmp_path: Path,
    source_date: str,
    expected_weight: float,
) -> None:
    records = [
        _record(
            tmp_path,
            0,
            s1_date="2020-01-01",
            s2_date="2020-01-01",
            size=32,
            textured=True,
        ),
        _record(
            tmp_path,
            1,
            s1_date="2020-01-03",
            s2_date="2020-01-03",
            size=32,
            textured=True,
        ),
        _record(
            tmp_path,
            2,
            s1_date=source_date,
            s2_date="2020-01-05",
            size=32,
            textured=True,
        ),
    ]
    index = build_paired_temporal_index(records, direction=SAR_TO_OPTICAL)
    sample = _sample_for(index, records[-1].pair_id)
    dataset = PairedTemporalRasterDataset(
        records,
        PairedTemporalIndex(config=index.config, samples=(sample,)),
        crop_size=32,
    )
    item = dataset[0]
    assert bool(item["registration_evidence_supported"])  # type: ignore[arg-type]
    assert bool(item["high_frequency_eligible"])  # type: ignore[arg-type]
    assert float(item["high_frequency_weight"]) == expected_weight  # type: ignore[arg-type]


def test_paired_dataset_uses_any_observation_support_for_crop_selection(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, 0, s1_date="2020-01-01", s2_date="2020-01-01"),
        _record(
            tmp_path,
            1,
            s1_date="2020-01-03",
            s2_date="2020-01-03",
            invalid_sar=True,
        ),
        _record(tmp_path, 2, s1_date="2020-01-05", s2_date="2020-01-05"),
        _record(tmp_path, 3, s1_date="2020-01-10", s2_date="2020-01-10"),
    ]
    index = build_paired_temporal_index(records, direction=SAR_TO_OPTICAL)
    sample = _sample_for(index, records[3].pair_id)
    dataset = PairedTemporalRasterDataset(
        records,
        PairedTemporalIndex(config=index.config, samples=(sample,)),
        crop_size=4,
        minimum_valid_fraction=1.0,
    )
    item = dataset[0]
    assert torch.all(item["observation_valid"][1] == 0)  # type: ignore[index]
    assert torch.all(item["observation_present"][:3])  # type: ignore[index]


def test_collate_paired_temporal_pads_batch_local_sequence_lengths(
    paired_records: list[PairRecord],
) -> None:
    index = build_paired_temporal_index(
        paired_records,
        direction=SAR_TO_OPTICAL,
        min_observations=1,
    )
    short = _sample_for(index, paired_records[1].pair_id)
    long = _sample_for(index, paired_records[3].pair_id)
    assert short.observation_count == 1
    assert long.observation_count == 3
    short_dataset = PairedTemporalRasterDataset(
        paired_records,
        PairedTemporalIndex(config=index.config, samples=(short,)),
        crop_size=4,
    )
    long_dataset = PairedTemporalRasterDataset(
        paired_records,
        PairedTemporalIndex(config=index.config, samples=(long,)),
        crop_size=4,
    )
    short_item, long_item = short_dataset[0], long_dataset[0]
    batch = collate_paired_temporal((short_item, long_item))
    assert batch["observation_values"].shape == (2, 3, len(SAR_CHANNEL_ORDER), 4, 4)  # type: ignore[union-attr]
    assert batch["observation_valid"].shape == (2, 3, 1, 4, 4)  # type: ignore[union-attr]
    assert batch["observation_days"].shape == (2, 3)  # type: ignore[union-attr]
    assert batch["observation_present"].dtype == torch.bool  # type: ignore[union-attr]
    assert torch.equal(
        batch["observation_present"], torch.tensor([[True, False, False], [True, True, True]])
    )  # type: ignore[arg-type]
    assert torch.all(batch["observation_values"][0, 1:] == 0)  # type: ignore[index]
    assert torch.all(batch["observation_valid"][0, 1:] == 0)  # type: ignore[index]
    assert torch.all(batch["observation_days"][0, 1:] == 0)  # type: ignore[index]
    assert batch["sample_id"] == [short.sample_id, long.sample_id]
    assert batch["direction"] == [SAR_TO_OPTICAL, SAR_TO_OPTICAL]
    assert batch["task_mode"] == [short.task_mode, long.task_mode]


def test_data_collate_training_alias_and_model_form_an_end_to_end_contract(
    paired_records: list[PairRecord],
) -> None:
    index = build_paired_temporal_index(
        paired_records,
        direction=SAR_TO_OPTICAL,
        min_observations=1,
    )
    samples = (
        _sample_for(index, paired_records[1].pair_id),
        _sample_for(index, paired_records[3].pair_id),
    )
    items = []
    for sample in samples:
        dataset = PairedTemporalRasterDataset(
            paired_records,
            PairedTemporalIndex(config=index.config, samples=(sample,)),
            crop_size=4,
        )
        items.append(dataset[0])
    batch = collate_paired_temporal(items)
    tensors = paired_tensor_batch(batch, torch.device("cpu"))
    model = SparsePairedAnchorTransport(
        PairedTemporalConfig(width=16, latent_channels=4, attention_heads=4, flow_steps=1)
    )
    output = forward_paired_temporal(model, tensors, SAR_TO_OPTICAL)
    assert output.physical.shape == (2, len(S2_CHANNEL_ORDER), 4, 4)
    assert output.attention.shape == (2, 3, 1, 1, 1)
    torch.testing.assert_close(output.physical, tensors["target_anchor"])


@pytest.mark.parametrize("direction", (SAR_TO_OPTICAL, OPTICAL_TO_SAR))
def test_cached_index_selection_matches_reference_builder_byte_for_byte(
    direction: str,
) -> None:
    records = _memory_records(48)
    # Exercise source-asset de-duplication and more than one isolation group
    # while keeping the legacy builder available as an independent oracle.
    records[13] = replace(records[13], sar=records[12].sar)
    records.extend(
        _memory_record(
            100 + index,
            tile="tile-reference-other",
            s1_date=(date(2020, 1, 10) + timedelta(days=index)).isoformat(),
            s2_date=(date(2020, 1, 10) + timedelta(days=index % 2)).isoformat(),
        )
        for index in range(12)
    )
    config = PairedTemporalIndexConfig(
        direction=direction,
        min_observations=1,
        max_observations=6,
        horizon_days=30,
        max_anchors_per_query=3,
        max_samples=31,
    )
    root = Path("/manifest-root")
    groups: dict[tuple[str, str, str], list[PairRecord]] = {}
    for record in records:
        groups.setdefault(paired_temporal_data_module._isolation_key(record), []).append(record)
    reference = []
    for group_records in groups.values():
        reference.extend(
            paired_temporal_data_module._build_group_samples(
                group_records,
                config=config,
                root=root,
            )
        )
    reference = sorted(reference, key=paired_temporal_data_module._sample_sort_key)[
        : config.max_samples
    ]

    index = build_paired_temporal_index(
        records,
        direction=direction,
        min_observations=config.min_observations,
        max_observations=config.max_observations,
        horizon_days=config.horizon_days,
        max_anchors_per_query=config.max_anchors_per_query,
        max_samples=config.max_samples,
        asset_root=root,
    )
    expected_bytes = b"\n".join(
        json.dumps(sample.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        for sample in reference
    )
    actual_bytes = b"\n".join(
        json.dumps(sample.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        for sample in index.samples
    )
    assert actual_bytes == expected_bytes


def test_cached_index_build_avoids_per_asset_resolve_for_5325_record_pilot_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _memory_records(5325)
    resolve_calls = 0
    original_resolve = Path.resolve

    def tracked_resolve(path: Path, *args: object, **kwargs: object) -> Path:
        nonlocal resolve_calls
        resolve_calls += 1
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", tracked_resolve)
    started = perf_counter()
    index = build_paired_temporal_index(
        records,
        direction=SAR_TO_OPTICAL,
        max_observations=8,
        max_samples=64,
        asset_root="/manifest-root",
    )
    elapsed = perf_counter() - started

    assert len(index) == 64
    # Only the supplied root is resolved.  Candidate assets use lexical
    # normalization and are cached during index construction.
    assert resolve_calls == 0
    # This is intentionally generous for shared CI; the benchmark guards the
    # pilot-prefix path against returning to minute-scale per-query resolves.
    assert elapsed < 5.0
