from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

import sentinel_v3.paired_temporal_local_cache as local_cache
from sentinel_v3.dataset_builder import PairRecord
from sentinel_v3.paired_temporal_data import (
    assert_paired_temporal_causality,
    load_pair_records,
    load_paired_temporal_index,
    write_pair_records,
)
from sentinel_v3.paired_temporal_local_cache import (
    DEFAULT_BUDGET_BYTES,
    DEFAULT_COPY_RATE_BYTES_PER_SECOND,
    GIB,
    LocalCacheAsset,
    _copy_asset_atomically,
    _CopyRateLimiter,
    assert_paired_temporal_feasibility_cache_budget,
    build_paired_temporal_feasibility_cache_plan,
    materialize_paired_temporal_feasibility_cache,
)
from sentinel_v3.schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER


def _write_asset(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0.0
        self.sleeps.append(seconds)
        self.now += seconds


def _record(root: Path, index: int, *, split: str, start: date) -> PairRecord:
    source_date = start + timedelta(days=index)
    target_date = source_date + timedelta(days=index % 2)
    record_root = root / "raw" / split / f"record-{index:03d}"
    s2: dict[str, str] = {}
    for channel in S2_CHANNEL_ORDER:
        path = record_root / "s2" / f"{channel}.tif"
        _write_asset(path, f"s2:{split}:{index}:{channel}".encode())
        s2[channel] = os.path.relpath(path, root / "manifests")
    scl = record_root / "scl.tif"
    _write_asset(scl, f"scl:{split}:{index}".encode())
    sar: dict[str, str] = {}
    for channel in SAR_CHANNEL_ORDER:
        path = record_root / "sar" / f"{channel}.tif"
        _write_asset(path, f"sar:{split}:{index}:{channel}".encode())
        sar[channel] = os.path.relpath(path, root / "manifests")
    return PairRecord(
        pair_id=f"2020:tile-cache:{split}:{index:03d}:ascending",
        year=2020,
        tile="tile-cache",
        tile_row=1,
        tile_col=1,
        split=split,
        refit_split="excluded",
        s2_date=target_date.isoformat(),
        s1_date=source_date.isoformat(),
        orbit="ascending",
        delta_days=index % 2,
        s2=s2,
        scl=os.path.relpath(scl, root / "manifests"),
        sar=sar,
        clear_fraction=1.0,
        valid_fraction=1.0,
        width=64,
        height=64,
        crs="EPSG:32650",
        transform=[10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0],
        gsd=10.0,
    )


def _config(root: Path, manifest: Path) -> Path:
    path = root / "paired_temporal_v2_feasibility.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "manifest": str(manifest),
                    "orbit": "ascending",
                    "anchor_pair_max_delta_days": 1,
                    "maximum_anchors_per_query": 2,
                    "horizon_days": 180,
                    "translation_max_delta_days": 1,
                    "minimum_observations": 1,
                    "maximum_observations": 4,
                    "max_train_samples": 64,
                    "max_validation_samples": 64,
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
def feasibility_inputs(tmp_path: Path) -> tuple[Path, Path]:
    records = [
        *(
            _record(tmp_path, index, split="train", start=date(2020, 1, 1))
            for index in range(70)
        ),
        *(
            _record(
                tmp_path,
                index,
                split="validation_temporal",
                start=date(2021, 1, 1),
            )
            for index in range(70)
        ),
    ]
    manifest = tmp_path / "manifests" / "pairs.jsonl"
    write_pair_records(manifest, records)
    return _config(tmp_path, manifest), manifest


def test_local_feasibility_cache_plan_selects_four_64_sample_indexes_and_deduplicates(
    feasibility_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    config, manifest = feasibility_inputs
    plan = build_paired_temporal_feasibility_cache_plan(
        config,
        destination_root=tmp_path / "cache",
        budget_bytes=10**9,
        minimum_free_bytes=0,
    )

    assert {(entry.direction, entry.split, len(entry.index)) for entry in plan.indexes} == {
        ("sar_to_optical", "train", 64),
        ("sar_to_optical", "validation", 64),
        ("optical_to_sar", "train", 64),
        ("optical_to_sar", "validation", 64),
    }
    assert sum(asset.references for asset in plan.assets) > len(plan.assets)
    assert plan.logical_source_bytes >= plan.stat_source_bytes
    assert plan.stat_source_bytes == sum(asset.size_bytes for asset in plan.assets)
    assert plan.source_manifest == manifest
    assert plan.allowed_to_materialize
    report = plan.report()
    assert report["unique_files"] == len(plan.assets)
    assert report["allowed_to_materialize"] is True


def test_default_feasibility_cache_budget_is_thirty_gib() -> None:
    assert DEFAULT_BUDGET_BYTES == 30 * GIB
    assert DEFAULT_COPY_RATE_BYTES_PER_SECOND == 0


def test_dry_run_does_not_construct_or_sleep_rate_limiter(
    feasibility_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = feasibility_inputs

    def unexpected_rate_limiter(*args: object, **kwargs: object) -> object:
        raise AssertionError("dry-run must not initialize the copy rate limiter")

    monkeypatch.setattr(local_cache, "_CopyRateLimiter", unexpected_rate_limiter)
    plan = build_paired_temporal_feasibility_cache_plan(
        config,
        destination_root=tmp_path / "cache",
        budget_bytes=10**9,
        minimum_free_bytes=0,
    )

    assert plan.allowed_to_materialize


def test_copy_rate_limiter_is_global_across_atomic_asset_copies(tmp_path: Path) -> None:
    fake_clock = _FakeClock()
    limiter = _CopyRateLimiter(
        4,
        clock=fake_clock.monotonic,
        sleep=fake_clock.sleep,
    )
    first_source = tmp_path / "first-source.tif"
    second_source = tmp_path / "second-source.tif"
    _write_asset(first_source, b"abcdefgh")
    _write_asset(second_source, b"ijkl")
    first_asset = LocalCacheAsset(
        source=first_source,
        relative_destination=Path("assets") / "first.tif",
        size_bytes=8,
        allocated_bytes=8,
        references=1,
    )
    second_asset = LocalCacheAsset(
        source=second_source,
        relative_destination=Path("assets") / "second.tif",
        size_bytes=4,
        allocated_bytes=4,
        references=1,
    )

    first_destination = tmp_path / "cache" / first_asset.relative_destination
    second_destination = tmp_path / "cache" / second_asset.relative_destination
    _copy_asset_atomically(first_asset, first_destination, rate_limiter=limiter)
    _copy_asset_atomically(second_asset, second_destination, rate_limiter=limiter)

    assert first_destination.read_bytes() == b"abcdefgh"
    assert second_destination.read_bytes() == b"ijkl"
    assert fake_clock.sleeps == [1.0, 1.0]


def test_unlimited_copy_rate_limiter_never_sleeps() -> None:
    fake_clock = _FakeClock()
    limiter = _CopyRateLimiter(0, clock=fake_clock.monotonic, sleep=fake_clock.sleep)

    limiter.consume(10**6)

    assert fake_clock.sleeps == []


def test_local_feasibility_cache_budget_is_a_hard_stop(
    feasibility_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    config, _ = feasibility_inputs
    plan = build_paired_temporal_feasibility_cache_plan(
        config,
        destination_root=tmp_path / "cache",
        budget_bytes=1,
        minimum_free_bytes=0,
    )

    assert not plan.budget_ok
    with pytest.raises(RuntimeError, match="exceeds budget"):
        assert_paired_temporal_feasibility_cache_budget(plan)
    with pytest.raises(RuntimeError, match="exceeds budget"):
        materialize_paired_temporal_feasibility_cache(plan)


def test_local_feasibility_cache_copies_atomically_resumes_and_publishes_hash_manifest(
    feasibility_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    config, _ = feasibility_inputs
    destination = tmp_path / "cache"
    plan = build_paired_temporal_feasibility_cache_plan(
        config,
        destination_root=destination,
        budget_bytes=10**9,
        minimum_free_bytes=0,
    )
    result = materialize_paired_temporal_feasibility_cache(plan)

    assert result["copied_files"] == len(plan.assets)
    assert result["reused_files"] == 0
    assert result["rate_limit_bytes_per_second"] == 0.0
    local_records = load_pair_records(destination / "manifests" / "pairs.jsonl")
    assert len(local_records) == len(plan.local_records)
    for record in local_records:
        for value in (*record.s2.values(), record.scl, *record.sar.values()):
            assert (destination / "manifests" / value).is_file()
    for entry in plan.indexes:
        loaded = load_paired_temporal_index(destination / entry.relative_destination)
        assert loaded.samples == entry.index.samples
        assert_paired_temporal_causality(
            loaded,
            local_records,
            asset_root=destination / "manifests",
        )
    local_manifest_text = (destination / "manifests" / "pairs.jsonl").read_text(
        encoding="utf-8"
    )
    assert "/raw/" not in local_manifest_text
    checksum = json.loads((destination / "cache_manifest.json").read_text(encoding="utf-8"))
    assert len(checksum["assets"]) == len(plan.assets)
    assert len(checksum["local_manifest_sha256"]) == 64

    resumed = materialize_paired_temporal_feasibility_cache(plan)
    assert resumed["copied_files"] == 0
    assert resumed["reused_files"] == len(plan.assets)
