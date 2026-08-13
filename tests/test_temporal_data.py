from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from sentinel_v3.dataset_builder import PairRecord
from sentinel_v3.schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER
from sentinel_v3.temporal_data import (
    OPTICAL_TO_SAR,
    SAR_TO_OPTICAL,
    TemporalIndex,
    TemporalIndexConfig,
    TemporalRasterDataset,
    TemporalSample,
    assert_strict_causality,
    build_temporal_index,
    load_pair_records,
    load_temporal_index,
    write_pair_records,
    write_temporal_index,
)


def _write_tiff(
    path: Path,
    values: np.ndarray,
    *,
    west: float = 500000.0,
) -> None:
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
    size: int = 4,
    west: float = 500000.0,
    invalid_target: bool = False,
) -> PairRecord:
    record_root = root / f"record-{index}"
    s2_values = np.full((size, size), 5000 + index * 1000, dtype=np.uint16)
    sar_values = np.full((size, size), 7000 + index * 200, dtype=np.uint16)
    scl = np.full((size, size), 4, dtype=np.uint8)
    if invalid_target:
        s2_values[0, 0] = 0
        scl[0, 0] = 1
    s2: dict[str, str] = {}
    for channel in S2_CHANNEL_ORDER:
        path = record_root / "s2" / f"{channel}.tif"
        _write_tiff(path, s2_values, west=west)
        s2[channel] = str(path)
    scl_path = record_root / "scl.tif"
    _write_tiff(scl_path, scl, west=west)
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
        delta_days=abs(
            (np.datetime64(s2_date, "D") - np.datetime64(s1_date, "D")).astype(int)
        ),
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


@pytest.fixture
def temporal_records(tmp_path: Path) -> list[PairRecord]:
    return [
        _record(tmp_path, 0, s1_date="2020-01-01", s2_date="2020-01-02"),
        _record(tmp_path, 1, s1_date="2020-01-05", s2_date="2020-01-06"),
        _record(tmp_path, 2, s1_date="2020-01-10", s2_date="2020-01-11"),
        _record(
            tmp_path,
            3,
            s1_date="2020-01-15",
            s2_date="2020-01-16",
            invalid_target=True,
        ),
    ]


def _sample_for(
    index: TemporalIndex, direction: str, query_pair_id: str
) -> TemporalSample:
    return next(
        sample
        for sample in index
        if sample.direction == direction and sample.query_pair_id == query_pair_id
    )


def test_temporal_index_uses_actual_modality_dates_and_fixed_isolation(
    temporal_records: list[PairRecord], tmp_path: Path
) -> None:
    other_split = _record(
        tmp_path,
        20,
        s1_date="2020-01-09",
        s2_date="2020-01-10",
        split="validation_temporal",
    )
    other_orbit = _record(
        tmp_path,
        21,
        s1_date="2020-01-09",
        s2_date="2020-01-10",
        orbit="descending",
    )
    other_year = _record(
        tmp_path,
        22,
        s1_date="2020-01-09",
        s2_date="2020-01-10",
        year=2021,
    )
    mismatch_grid = _record(
        tmp_path,
        23,
        s1_date="2020-01-09",
        s2_date="2020-01-10",
        west=500010.0,
    )
    index = build_temporal_index(
        [*temporal_records, other_split, other_orbit, other_year, mismatch_grid],
        source_frames=2,
    )

    sar_to_optical = _sample_for(index, SAR_TO_OPTICAL, temporal_records[2].pair_id)
    assert sar_to_optical.anchor_pair_id == temporal_records[1].pair_id
    assert sar_to_optical.source_pair_ids == (
        temporal_records[1].pair_id,
        temporal_records[2].pair_id,
    )
    assert sar_to_optical.query_date == "2020-01-11"
    assert sar_to_optical.anchor_date == "2020-01-06"
    assert sar_to_optical.source_dates == ("2020-01-05", "2020-01-10")

    optical_to_sar = _sample_for(index, OPTICAL_TO_SAR, temporal_records[2].pair_id)
    assert optical_to_sar.anchor_pair_id == temporal_records[1].pair_id
    assert optical_to_sar.source_pair_ids == (
        temporal_records[0].pair_id,
        temporal_records[1].pair_id,
    )
    assert optical_to_sar.query_date == "2020-01-10"
    assert optical_to_sar.anchor_date == "2020-01-05"
    assert optical_to_sar.source_dates == ("2020-01-02", "2020-01-06")

    assert all(sample.split == "train" for sample in index)
    assert all(sample.orbit == "ascending" for sample in index)
    assert all(sample.year == 2020 and sample.tile == "tile-a" for sample in index)
    assert_strict_causality(index, [*temporal_records, other_split, other_orbit, other_year, mismatch_grid])


def test_temporal_index_rejects_asset_leakage_and_invalid_time(
    temporal_records: list[PairRecord], tmp_path: Path
) -> None:
    query = temporal_records[3]
    anchor = temporal_records[2]
    leaked_source = _record(
        tmp_path,
        31,
        s1_date="2020-01-14",
        s2_date="2020-01-14",
    )
    leaked_source = replace(
        leaked_source,
        sar={
            "vv": query.s2["blue"],
            "vh": query.s2["green"],
        },
    )
    leaked = TemporalSample(
        sample_id="leaked",
        direction=SAR_TO_OPTICAL,
        split=query.split,
        tile=query.tile,
        year=query.year,
        orbit=query.orbit,
        query_pair_id=query.pair_id,
        anchor_pair_id=anchor.pair_id,
        source_pair_ids=(leaked_source.pair_id,),
        query_date=query.s2_date,
        anchor_date=anchor.s2_date,
        source_dates=(leaked_source.s1_date,),
    )
    leaked_index = TemporalIndex(
        config=TemporalIndexConfig(source_frames=1, directions=(SAR_TO_OPTICAL,)),
        samples=(leaked,),
    )
    with pytest.raises(AssertionError, match="target asset"):
        assert_strict_causality(leaked_index, [*temporal_records, leaked_source])

    future = replace(
        leaked,
        sample_id="future",
        source_pair_ids=(temporal_records[3].pair_id,),
        source_dates=("2020-01-20",),
    )
    future_index = TemporalIndex(
        config=TemporalIndexConfig(source_frames=1, directions=(SAR_TO_OPTICAL,)),
        samples=(future,),
    )
    with pytest.raises(AssertionError, match="stored source dates"):
        assert_strict_causality(future_index, temporal_records)

    late = _record(
        tmp_path,
        40,
        s1_date="2020-08-01",
        s2_date="2020-08-02",
    )
    assert not build_temporal_index(
        [temporal_records[0], late], source_frames=1, horizon_days=180
    ).samples


def test_temporal_jsonl_round_trip_and_slice_loader(
    temporal_records: list[PairRecord], tmp_path: Path
) -> None:
    manifest = tmp_path / "manifests" / "pairs.jsonl"
    write_pair_records(manifest, reversed(temporal_records))
    loaded_records = load_pair_records(manifest, start=1, limit=2)
    assert [record.pair_id for record in loaded_records] == sorted(
        record.pair_id for record in temporal_records
    )[1:3]

    index = build_temporal_index(
        manifest,
        source_frames=2,
        horizon_days=30,
        split="train",
        max_samples=3,
    )
    index_path = tmp_path / "indices" / "temporal.jsonl"
    write_temporal_index(index_path, index)
    reloaded = load_temporal_index(index_path)
    assert reloaded.config == index.config
    assert reloaded.samples == index.samples
    assert reloaded.subset(start=1, limit=1).samples == index.samples[1:2]
    assert load_temporal_index(index_path, start=1, limit=1).samples == index.samples[1:2]
    assert index[1:2].samples == index.samples[1:2]


def test_temporal_index_rejects_samples_outside_serialized_split_or_orbit(
    temporal_records: list[PairRecord],
) -> None:
    index = build_temporal_index(
        temporal_records,
        source_frames=1,
        split="train",
        orbit="ascending",
        directions=(SAR_TO_OPTICAL,),
    )
    sample = replace(index.samples[0], split="validation_temporal")
    with pytest.raises(ValueError, match="split"):
        TemporalIndex(config=index.config, samples=(sample,))
    sample = replace(index.samples[0], orbit="descending")
    with pytest.raises(ValueError, match="orbit"):
        TemporalIndex(config=index.config, samples=(sample,))


def test_temporal_raster_dataset_rejects_mixed_direction_index(
    temporal_records: list[PairRecord],
) -> None:
    index = build_temporal_index(temporal_records, source_frames=1)
    assert {sample.direction for sample in index} == {SAR_TO_OPTICAL, OPTICAL_TO_SAR}
    with pytest.raises(ValueError, match="single-direction"):
        TemporalRasterDataset(temporal_records, index, crop_size=4)


@pytest.mark.parametrize(
    ("direction", "source_channels", "target_channels"),
    (
        (SAR_TO_OPTICAL, len(SAR_CHANNEL_ORDER), len(S2_CHANNEL_ORDER)),
        (OPTICAL_TO_SAR, len(S2_CHANNEL_ORDER), len(SAR_CHANNEL_ORDER)),
    ),
)
def test_temporal_raster_dataset_emits_v3_units_and_cached_clones(
    temporal_records: list[PairRecord],
    direction: str,
    source_channels: int,
    target_channels: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = build_temporal_index(
        temporal_records,
        source_frames=2,
        directions=(direction,),
    )
    dataset = TemporalRasterDataset(
        temporal_records,
        index,
        crop_size=4,
        minimum_valid_fraction=0.8,
        cache_in_memory=True,
    )
    assert len(dataset) == len(index)

    import rasterio

    original_open = rasterio.open
    opens = 0

    def counted_open(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal opens
        opens += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(rasterio, "open", counted_open)
    item = dataset[-1]
    first_opens = opens
    again = dataset[-1]
    assert opens == first_opens

    assert set(item) == {
        "source_values",
        "source_valid",
        "anchor_values",
        "anchor_valid",
        "target_values",
        "target_valid",
        "source_days",
        "anchor_days",
        "sample_id",
        "direction",
    }
    assert item["direction"] == direction
    assert item["source_values"].shape == (2, source_channels, 4, 4)  # type: ignore[union-attr]
    assert item["source_valid"].shape == (2, 1, 4, 4)  # type: ignore[union-attr]
    assert item["anchor_values"].shape == (target_channels, 4, 4)  # type: ignore[union-attr]
    assert item["anchor_valid"].shape == (1, 4, 4)  # type: ignore[union-attr]
    assert item["target_values"].shape == (target_channels, 4, 4)  # type: ignore[union-attr]
    assert item["target_valid"].shape == (1, 4, 4)  # type: ignore[union-attr]
    assert item["source_values"].dtype == torch.float32  # type: ignore[union-attr]
    assert item["source_valid"].dtype == torch.float32  # type: ignore[union-attr]
    assert torch.all(item["source_days"] <= 0)  # type: ignore[operator]
    assert float(item["anchor_days"]) < 0  # type: ignore[arg-type]
    assert torch.all(item["source_values"].abs() <= 1.0)  # type: ignore[union-attr]
    assert torch.all(item["anchor_values"].abs() <= 1.0)  # type: ignore[union-attr]
    assert torch.all(item["target_values"].abs() <= 1.0)  # type: ignore[union-attr]

    if direction == SAR_TO_OPTICAL:
        target_valid = item["target_valid"]  # type: ignore[assignment]
        target_values = item["target_values"]  # type: ignore[assignment]
        assert target_valid[0, 0, 0] == 0
        assert torch.all(target_values[:, 0, 0] == 0)
        assert target_values[0, 1, 1] == pytest.approx(0.6)
    else:
        source_values = item["source_values"]  # type: ignore[assignment]
        assert source_values[0, 0, 1, 1] == pytest.approx(0.2)

    assert torch.equal(item["source_values"], again["source_values"])  # type: ignore[arg-type]
    item["source_values"][0, 0, 0, 0] = 123.0  # type: ignore[index]
    third = dataset[-1]
    assert third["source_values"][0, 0, 0, 0] != 123.0  # type: ignore[index]


def test_temporal_raster_dataset_detects_a_manifest_grid_lie(
    temporal_records: list[PairRecord],
) -> None:
    index = build_temporal_index(temporal_records, source_frames=1, directions=(SAR_TO_OPTICAL,))
    query = temporal_records[-1]
    lied = replace(query, transform=[10.0, 0.0, 499990.0, 0.0, -10.0, 4100000.0])
    records = [*temporal_records[:-1], lied]
    with pytest.raises(AssertionError, match="share the query grid"):
        TemporalRasterDataset(records, index, crop_size=4)


def test_temporal_raster_dataset_requires_four_pixel_fusion_grid(
    temporal_records: list[PairRecord],
) -> None:
    index = build_temporal_index(temporal_records, source_frames=1, directions=(SAR_TO_OPTICAL,))
    with pytest.raises(ValueError, match="divisible by four"):
        TemporalRasterDataset(temporal_records, index, crop_size=6)
