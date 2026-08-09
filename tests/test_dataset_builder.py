from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sentinel_v3.data import V2ShardDataset, high_frequency_eligible, time_weights
from sentinel_v3.dataset_builder import (
    BuildConfig,
    PairRecord,
    assert_split_leakage,
    audit_dataset,
    build_train_shards,
    discover_pairs,
    fixed_split,
    write_audit_artifacts,
)
from sentinel_v3.evaluation import ManifestCropDataset
from sentinel_v3.schema import (
    CLEAR_SCL_CODES,
    LEGACY_V1_S2_CHANNEL_ORDER,
    S2_CHANNEL_ORDER,
    SAR_CHANNEL_ORDER,
    channel_reorder_indices,
)
from sentinel_v3.temporal_prior import temporal_prior_config
from sentinel_v3.training import _load_high_frequency_eligibility
from sentinel_v3.validation import protocol_records, validation_protocol_for_manifest


def _write_tiff(path: Path, values: np.ndarray, *, west: float = 500000) -> None:
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


def _write_s2_scene(
    tile: Path,
    day: str,
    *,
    scl: int,
    size: int = 16,
    red_west: float = 500000,
    band_values: dict[str, int] | None = None,
) -> None:
    for channel in S2_CHANNEL_ORDER:
        _write_tiff(
            tile / "data_raw" / channel / f"{day}_mosaic.tiff",
            np.full((size, size), (band_values or {}).get(channel, 5000), dtype=np.uint16),
            west=red_west if channel == "red" else 500000,
        )
    _write_tiff(
        tile / "data_raw" / "scl" / f"{day}_mosaic.tiff",
        np.full((size, size), scl, dtype=np.uint8),
    )


def _write_sar_frame(tile: Path, day: str, orbit: str = "ascending", size: int = 16) -> None:
    for channel in SAR_CHANNEL_ORDER:
        _write_tiff(
            tile / "data_sar_raw" / f"{day}_{channel}_{orbit}.tiff",
            np.full((size, size), 7000, dtype=np.uint16),
        )


def _pair_record(
    *,
    year: int = 2021,
    tile: str = "Beijing_r0001_c0001_y000000_x000000_h16_w16",
    split: str = "train",
) -> PairRecord:
    return PairRecord(
        pair_id=f"{year}:{tile}:2021-01-02:ascending:2021-01-03",
        year=year,
        tile=tile,
        tile_row=1,
        tile_col=1,
        split=split,
        refit_split="excluded",
        s2_date="2021-01-03",
        s1_date="2021-01-02",
        orbit="ascending",
        delta_days=1,
        s2={channel: f"/{channel}.tiff" for channel in S2_CHANNEL_ORDER},
        scl="/scl.tiff",
        sar={channel: f"/{channel}.tiff" for channel in SAR_CHANNEL_ORDER},
        clear_fraction=1.0,
        valid_fraction=1.0,
        width=16,
        height=16,
        crs="EPSG:32650",
        transform=[10.0, 0.0, 0.0, 0.0, -10.0, 0.0],
        gsd=10.0,
    )


@pytest.mark.parametrize(
    ("year", "row", "col", "expected"),
    (
        (2021, 5, 1, "buffer"),
        (2024, 1, 5, "buffer"),
        (2022, 6, 1, "unused_spatial"),
        (2023, 1, 6, "test_spatial"),
        (2024, 6, 1, "test_joint"),
        (2022, 1, 1, "train"),
        (2023, 1, 1, "validation_temporal"),
        (2024, 1, 1, "test_temporal"),
    ),
)
def test_fixed_split_boundaries(year: int, row: int, col: int, expected: str) -> None:
    assert fixed_split(year, row, col) == expected


def test_discovery_falls_back_to_next_nearest_candidate_deterministically(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    tile = raw_root / "2021" / "Beijing_r0001_c0001_y000000_x000000_h16_w16"
    _write_s2_scene(tile, "2021-01-01", scl=1)
    _write_s2_scene(tile, "2021-01-03", scl=4)
    _write_sar_frame(tile, "2021-01-02")
    config = BuildConfig(raw_root=raw_root, output_root=tmp_path / "dataset", years=(2021,), crop_size=8)

    first, first_summary = discover_pairs(config)
    second, second_summary = discover_pairs(config)

    assert [record.to_dict() for record in first] == [record.to_dict() for record in second]
    assert first_summary == second_summary
    assert len(first) == 1
    assert first[0].s2_date == "2021-01-03"
    assert first[0].delta_days == 1
    assert first_summary["rejected"]["clear_fraction"] == 1


def test_discovery_falls_back_after_grid_failure(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    tile = raw_root / "2021" / "Beijing_r0001_c0001_y000000_x000000_h16_w16"
    _write_s2_scene(tile, "2021-01-01", scl=4, red_west=500010)
    _write_s2_scene(tile, "2021-01-03", scl=4)
    _write_sar_frame(tile, "2021-01-02")
    config = BuildConfig(raw_root=raw_root, output_root=tmp_path / "dataset", years=(2021,), crop_size=8)

    records, summary = discover_pairs(config)

    assert len(records) == 1
    assert records[0].s2_date == "2021-01-03"
    assert summary["rejected"]["grid_or_crs_mismatch"] == 1


def test_discovery_falls_back_after_corrupt_candidate(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    tile = raw_root / "2021" / "Beijing_r0001_c0001_y000000_x000000_h16_w16"
    _write_s2_scene(tile, "2021-01-01", scl=4)
    _write_s2_scene(tile, "2021-01-03", scl=4)
    _write_sar_frame(tile, "2021-01-02")
    (tile / "data_raw" / "red" / "2021-01-01_mosaic.tiff").write_bytes(b"broken")
    config = BuildConfig(raw_root=raw_root, output_root=tmp_path / "dataset", years=(2021,), crop_size=8)

    records, summary = discover_pairs(config)

    assert len(records) == 1
    assert records[0].s2_date == "2021-01-03"
    assert summary["rejected"]["grid_read_error"] == 1


def test_builder_writes_pair_homogeneous_resumable_v2_shards(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    tile = raw_root / "2021" / "Beijing_r0001_c0001_y000000_x000000_h16_w16"
    _write_s2_scene(tile, "2021-01-03", scl=4)
    _write_sar_frame(tile, "2021-01-02")
    config = BuildConfig(
        raw_root=raw_root,
        output_root=tmp_path / "dataset",
        years=(2021,),
        crop_size=8,
        patches_per_pair=2,
    )
    records, audit = audit_dataset(config)
    first = build_train_shards(config, records)
    second = build_train_shards(config, records, resume=True)

    assert audit["manifest_sha256"] == first["manifest_sha256"]
    assert first["shards"] == second["shards"]
    assert first["format_version"] == 2
    assert first["s2_channel_order"] == list(S2_CHANNEL_ORDER)
    assert first["train_years"] == list(range(2017, 2023))
    descriptor = first["shards"][0]
    shard = torch.load(descriptor["path"], map_location="cpu", weights_only=False)
    assert shard["pair_id"] == [records[0].pair_id, records[0].pair_id]
    assert shard["s2"].dtype is torch.float16
    assert shard["s2"].shape == (2, 10, 8, 8)
    assert shard["window"].shape == (2, 4)
    eligibility = json.loads((config.output_root / "hf_eligibility.json").read_text())
    assert eligibility["hf_years"] == list(range(2017, 2023))
    assert eligibility["eligible_indices"] == [0, 1]
    assert eligibility["registration_audited"] is False


def test_builder_reuses_record_scoped_raster_handles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_root = tmp_path / "raw"
    tile = raw_root / "2021" / "Beijing_r0001_c0001_y000000_x000000_h16_w16"
    _write_s2_scene(tile, "2021-01-03", scl=4)
    _write_sar_frame(tile, "2021-01-02")
    config = BuildConfig(
        raw_root=raw_root,
        output_root=tmp_path / "dataset",
        years=(2021,),
        crop_size=8,
        patches_per_pair=2,
    )
    records, _ = audit_dataset(config)
    import rasterio

    original_open = rasterio.open
    opens = 0

    def counted_open(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal opens
        opens += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(rasterio, "open", counted_open)
    build_train_shards(config, records)

    assert opens == len(S2_CHANNEL_ORDER) + len(SAR_CHANNEL_ORDER) + 1


def test_new_shard_and_loader_keep_canonical_channel_values(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    tile = raw_root / "2021" / "Beijing_r0001_c0001_y000000_x000000_h16_w16"
    values = {channel: 1000 + 500 * index for index, channel in enumerate(S2_CHANNEL_ORDER)}
    _write_s2_scene(tile, "2021-01-03", scl=4, band_values=values)
    _write_sar_frame(tile, "2021-01-02")
    config = BuildConfig(
        raw_root=raw_root,
        output_root=tmp_path / "dataset",
        years=(2021,),
        crop_size=8,
        patches_per_pair=1,
    )
    records, _ = audit_dataset(config)
    index = build_train_shards(config, records)
    item = V2ShardDataset(index["index"], augment=False, random_gsd=False)[0]

    expected = torch.tensor([values[channel] / 10000.0 for channel in S2_CHANNEL_ORDER])
    torch.testing.assert_close(item["s2"][:, 0, 0], expected, atol=4e-4, rtol=4e-4)


def test_v2_dataset_reorders_explicit_legacy_and_rejects_bad_schemas(tmp_path: Path) -> None:
    shard_path = tmp_path / "legacy.pt"
    source_values = torch.linspace(-1.0, 0.8, 10).view(1, 10, 1, 1).expand(1, 10, 8, 8)
    torch.save(
        {
            "s2": source_values.half(),
            "sar": torch.zeros(1, 2, 8, 8, dtype=torch.float16),
            "s2_valid": torch.ones(1, 1, 8, 8, dtype=torch.uint8),
            "sar_valid": torch.ones(1, 1, 8, 8, dtype=torch.uint8),
            "joint_valid": torch.ones(1, 1, 8, 8, dtype=torch.uint8),
            "metadata": torch.zeros(1, 8),
            "window": torch.tensor([[0, 0, 8, 8]], dtype=torch.int32),
            "pair_id": ["2018:tile:2018-01-01:ascending:2018-01-01"],
        },
        shard_path,
    )
    legacy_index = tmp_path / "legacy_index.json"
    legacy_index.write_text(json.dumps({"split": "train", "shards": [{"path": str(shard_path), "count": 1}]}))
    dataset = V2ShardDataset(legacy_index, augment=False, random_gsd=False)
    item = dataset[0]
    source_indices = channel_reorder_indices(LEGACY_V1_S2_CHANNEL_ORDER, S2_CHANNEL_ORDER)
    expected = (source_values[0, list(source_indices), 0, 0] + 1.0) * 0.5
    torch.testing.assert_close(item["s2"][:, 0, 0], expected, atol=2e-4, rtol=2e-4)

    bad_index = tmp_path / "bad_index.json"
    bad_index.write_text(
        json.dumps(
            {
                "format_version": 2,
                "split": "train",
                "s2_channel_order": list(S2_CHANNEL_ORDER[:-1]) + ["blue"],
                "shards": [{"path": str(shard_path), "count": 1}],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        V2ShardDataset(bad_index, augment=False)

    missing_index = tmp_path / "missing_index.json"
    missing_index.write_text(json.dumps({"format_version": 2, "split": "train", "shards": []}))
    with pytest.raises(RuntimeError, match="s2_channel_order"):
        V2ShardDataset(missing_index, augment=False)


@pytest.mark.parametrize(
    ("format_version", "s2_channel_order"),
    ((1, None), (2, list(S2_CHANNEL_ORDER))),
)
def test_loader_restores_joint_invalid_pixels_to_physical_zero(
    tmp_path: Path, format_version: int, s2_channel_order: list[str] | None
) -> None:
    shard_path = tmp_path / f"shard_{format_version}.pt"
    joint_valid = torch.ones(1, 1, 8, 8, dtype=torch.uint8)
    joint_valid[:, :, 0, 0] = 0
    payload: dict[str, object] = {
        "s2": torch.zeros(1, 10, 8, 8, dtype=torch.float16),
        "sar": torch.full((1, 2, 8, 8), 0.2, dtype=torch.float16),
        "s2_valid": torch.ones(1, 1, 8, 8, dtype=torch.uint8),
        "sar_valid": torch.ones(1, 1, 8, 8, dtype=torch.uint8),
        "joint_valid": joint_valid,
        "metadata": torch.zeros(1, 8),
        "window": torch.tensor([[0, 0, 8, 8]], dtype=torch.int32),
        "pair_id": ["2018:tile:2018-01-01:ascending:2018-01-01"],
    }
    if format_version == 2:
        payload.update(
            {
                "format_version": 2,
                "s2_channel_order": s2_channel_order,
                "sar_channel_order": list(SAR_CHANNEL_ORDER),
            }
        )
    torch.save(payload, shard_path)
    index: dict[str, object] = {
        "format_version": format_version,
        "split": "train",
        "shards": [{"path": str(shard_path), "count": 1}],
    }
    if format_version == 2:
        index.update(
            {
                "s2_channel_order": s2_channel_order,
                "sar_channel_order": list(SAR_CHANNEL_ORDER),
            }
        )
    index_path = tmp_path / f"index_{format_version}.json"
    index_path.write_text(json.dumps(index))
    item = V2ShardDataset(index_path, augment=False, random_gsd=False)[0]

    torch.testing.assert_close(item["s2"][:, 0, 0], torch.zeros(10))
    torch.testing.assert_close(item["sar"][:, 0, 0], torch.zeros(2))
    assert bool((item["s2"][:, 1, 1] > 0).all())
    assert bool((item["sar"][:, 1, 1] != 0).all())


def _minimal_v2_index(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "split": "train",
                "s2_channel_order": list(S2_CHANNEL_ORDER),
                "sar_channel_order": list(SAR_CHANNEL_ORDER),
                "train_years": [2021],
                "hf_years": [2021],
                "shards": [],
            }
        )
    )
    return path


def test_explicit_hf_sidecar_requires_completed_registration_audit(tmp_path: Path) -> None:
    index_path = _minimal_v2_index(tmp_path / "index.json")
    source_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
    sidecar_path = tmp_path / "hf.json"
    base = {
        "format_version": 2,
        "source_index": str(index_path.resolve()),
        "source_index_sha256": source_hash,
        "eligible_indices": [0, 2],
    }
    sidecar_path.write_text(json.dumps({**base, "registration_audited": False}))
    with pytest.raises(RuntimeError, match="registration audit"):
        _load_high_frequency_eligibility(sidecar_path, index_path)
    sidecar_path.write_text(json.dumps(base))
    with pytest.raises(RuntimeError, match="registration audit"):
        _load_high_frequency_eligibility(sidecar_path, index_path)
    sidecar_path.write_text(json.dumps({**base, "registration_audited": True}))
    assert _load_high_frequency_eligibility(sidecar_path, index_path) == [0, 2]
    sidecar_path.write_text(
        json.dumps({**base, "registration_audited": True, "source_index_sha256": "stale"})
    )
    with pytest.raises(RuntimeError, match="sha256"):
        _load_high_frequency_eligibility(sidecar_path, index_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "source_index": str(index_path.resolve()),
                "eligible_indices": [1],
            }
        )
    )
    assert _load_high_frequency_eligibility(sidecar_path, index_path) == [1]


def test_temporal_prior_index_must_match_source_index_and_schema(tmp_path: Path) -> None:
    index_path = _minimal_v2_index(tmp_path / "index.json")
    prior_path = tmp_path / "prior.json"
    base = {
        "format_version": 2,
        "source_index": str(index_path.resolve()),
        "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "s2_channel_order": list(S2_CHANNEL_ORDER),
        "sar_channel_order": list(SAR_CHANNEL_ORDER),
        "shards": [],
    }
    prior_path.write_text(json.dumps(base))
    V2ShardDataset(index_path, augment=False, temporal_prior_index=prior_path)
    prior_path.write_text(json.dumps({**base, "source_index_sha256": "stale"}))
    with pytest.raises(RuntimeError, match="source_index_sha256"):
        V2ShardDataset(index_path, augment=False, temporal_prior_index=prior_path)
    prior_path.write_text(
        json.dumps({**base, "s2_channel_order": list(reversed(S2_CHANNEL_ORDER))})
    )
    with pytest.raises(RuntimeError, match="canonical S2"):
        V2ShardDataset(index_path, augment=False, temporal_prior_index=prior_path)
    prior_path.write_text(json.dumps({"format_version": 1, "shards": []}))
    V2ShardDataset(index_path, augment=False, temporal_prior_index=prior_path)


def test_legacy_index_without_pair_metadata_keeps_all_high_frequency_shards() -> None:
    dataset = V2ShardDataset.__new__(V2ShardDataset)
    dataset.shards = [{"count": 2}, {"count": 2}]
    dataset.prior_shards = None
    assert dataset.high_frequency_shard_indices() == [0, 1]


def test_protocol_sidecar_and_legacy_463_fallback(tmp_path: Path) -> None:
    config = BuildConfig(raw_root=tmp_path / "raw", output_root=tmp_path / "dataset", years=(2023,))
    record = _pair_record(year=2023, split="validation_temporal")
    write_audit_artifacts(config, [record], {"records": 1, "rejected": {}, "discovered": {}})
    manifest = config.output_root / "manifests" / "pairs.jsonl"
    protocol = validation_protocol_for_manifest(manifest)
    assert protocol.expected_samples == 1
    assert protocol.optical_channels == S2_CHANNEL_ORDER
    assert len(protocol_records(manifest)) == 1

    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(json.dumps(record.to_dict()) + "\n")
    assert validation_protocol_for_manifest(legacy).expected_samples == 463
    with pytest.raises(RuntimeError, match="463"):
        protocol_records(legacy)


def test_sidecar_mask_is_used_and_sidecar_schema_is_strict(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    tile = raw_root / "2023" / "Beijing_r0001_c0001_y000000_x000000_h16_w16"
    _write_s2_scene(tile, "2021-01-03", scl=2, size=8)
    _write_sar_frame(tile, "2021-01-02", size=8)
    template = _pair_record(year=2023, tile=tile.name, split="validation_temporal")
    record = PairRecord(
        **{
            **template.to_dict(),
            "s2": {
                channel: str(tile / "data_raw" / channel / "2021-01-03_mosaic.tiff")
                for channel in S2_CHANNEL_ORDER
            },
            "scl": str(tile / "data_raw" / "scl" / "2021-01-03_mosaic.tiff"),
            "sar": {
                channel: str(tile / "data_sar_raw" / f"2021-01-02_{channel}_ascending.tiff")
                for channel in SAR_CHANNEL_ORDER
            },
            "width": 8,
            "height": 8,
        }
    )
    config = BuildConfig(raw_root=raw_root, output_root=tmp_path / "dataset", years=(2023,), crop_size=8)
    write_audit_artifacts(config, [record], {"records": 1, "rejected": {}, "discovered": {}})
    manifest = config.output_root / "manifests" / "pairs.jsonl"
    sidecar = config.output_root / "manifests" / "validation_protocol.json"
    values = json.loads(sidecar.read_text())

    dataset = ManifestCropDataset(manifest, "validation_temporal")
    item = dataset[0]
    assert dataset.protocol.mask_scl_codes == CLEAR_SCL_CODES
    assert float(item["valid"].mean()) == 1.0

    values["mask_scl_codes"] = [1]
    sidecar.write_text(json.dumps(values))
    with pytest.raises(RuntimeError, match="canonical clear SCL"):
        validation_protocol_for_manifest(manifest)
    values["mask_scl_codes"] = list(CLEAR_SCL_CODES)
    values["s2_channel_order"] = list(reversed(S2_CHANNEL_ORDER))
    sidecar.write_text(json.dumps(values))
    with pytest.raises(RuntimeError, match="canonical S2"):
        validation_protocol_for_manifest(manifest)
    values["s2_channel_order"] = list(S2_CHANNEL_ORDER)
    values.pop("units")
    sidecar.write_text(json.dumps(values))
    with pytest.raises(RuntimeError, match="optical units"):
        validation_protocol_for_manifest(manifest)


def test_builder_rejects_hf_years_outside_train_years(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hf_years"):
        BuildConfig(
            raw_root=tmp_path / "raw",
            output_root=tmp_path / "dataset",
            years=(2021,),
            train_years=(2021,),
            hf_years=(2022,),
        )


def test_hf_years_temporal_year_derivation_and_long_gap_zero_weight(tmp_path: Path) -> None:
    assert high_frequency_eligible(
        delta_days=1,
        year=2022,
        split="train",
        registration_shift_px=0.0,
        valid_fraction=1.0,
        cloud_shadow_fraction=0.0,
        train_years=range(2017, 2023),
    )
    assert not high_frequency_eligible(
        delta_days=2,
        year=2022,
        split="train",
        registration_shift_px=0.0,
        valid_fraction=1.0,
        cloud_shadow_fraction=0.0,
        train_years=range(2017, 2023),
    )
    values = torch.ones(1, requires_grad=True)
    high_frequency_weight = time_weights(torch.tensor([2]))[1]
    (values * high_frequency_weight).sum().backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))

    manifest = tmp_path / "pairs.jsonl"
    records = [_pair_record(year=2017), _pair_record(year=2022)]
    manifest.write_text("".join(json.dumps(record.to_dict()) + "\n" for record in records))
    config = temporal_prior_config(manifest)
    assert config.train_years == (2017, 2022)
    assert config.version == "train-seasonal-v3"


def test_split_leakage_rejects_spatial_tile_reuse() -> None:
    train = _pair_record()
    holdout = PairRecord(
        **{
            **train.to_dict(),
            "pair_id": "2023:tile:2023-01-01:ascending:2023-01-01",
            "year": 2023,
            "tile_row": 6,
            "tile_col": 1,
            "split": "test_spatial",
        }
    )
    with pytest.raises(RuntimeError, match="spatial holdout"):
        assert_split_leakage((train, holdout))
