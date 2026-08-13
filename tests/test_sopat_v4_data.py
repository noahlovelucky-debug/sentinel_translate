from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from sentinel_v3.dataset_builder import PairRecord
from sentinel_v3.paired_temporal_data import (
    FORECAST,
    OPTICAL_TO_SAR,
    SAR_TO_OPTICAL,
    TRANSLATION,
    build_paired_temporal_index,
    write_pair_records,
)
from sentinel_v3.schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER
from sentinel_v4.data import (
    CoupledDirectionLoader,
    SOPATDirectionDataset,
    SOPATIndexConfigV4,
    SOPATIndexV4,
    assert_sopat_v4_causality,
    build_sopat_v4_index,
    collate_sopat_direction,
    load_sopat_v4_index,
    migrate_paired_temporal_index_v4,
    paired_temporal_index_from_sopat_v4,
    sopat_v4_index_file_sha256,
    write_sopat_v4_index,
)


def _record(
    number: int,
    *,
    s1_date: str,
    s2_date: str,
    split: str = "train",
    tile: str = "tile-v4",
) -> PairRecord:
    prefix = f"assets/{split}/{tile}/{number:03d}"
    return PairRecord(
        pair_id=f"2020:{tile}:{split}:{number:03d}:ascending",
        year=2020,
        tile=tile,
        tile_row=1,
        tile_col=1,
        split=split,
        refit_split="excluded",
        s2_date=s2_date,
        s1_date=s1_date,
        orbit="ascending",
        delta_days=abs(int(s2_date[-2:]) - int(s1_date[-2:])),
        s2={channel: f"{prefix}/s2-{channel}.tif" for channel in S2_CHANNEL_ORDER},
        scl=f"{prefix}/scl.tif",
        sar={channel: f"{prefix}/sar-{channel}.tif" for channel in SAR_CHANNEL_ORDER},
        clear_fraction=1.0,
        valid_fraction=1.0,
        width=256,
        height=256,
        crs="EPSG:32650",
        transform=[10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0],
        gsd=10.0,
    )


@pytest.fixture
def records() -> list[PairRecord]:
    return [
        _record(0, s1_date="2020-01-01", s2_date="2020-01-01"),
        _record(1, s1_date="2020-01-03", s2_date="2020-01-03"),
        _record(2, s1_date="2020-01-05", s2_date="2020-01-05"),
        _record(3, s1_date="2020-01-10", s2_date="2020-01-10"),
        _record(4, s1_date="2020-01-13", s2_date="2020-01-14"),
        _record(5, s1_date="2020-01-15", s2_date="2020-01-17"),
    ]


def _index(records: list[PairRecord]) -> SOPATIndexV4:
    return build_sopat_v4_index(
        records,
        splits=("train",),
        max_observations=3,
        horizon_days=180,
    )


def test_migration_preserves_both_directions_roles_and_date_causality(
    records: list[PairRecord],
) -> None:
    index = _index(records)

    assert {example.direction for example in index} == {SAR_TO_OPTICAL, OPTICAL_TO_SAR}
    assert {example.task_mode for example in index} == {TRANSLATION, FORECAST}
    for direction in (SAR_TO_OPTICAL, OPTICAL_TO_SAR):
        selected = build_paired_temporal_index(records, direction=direction, max_observations=3)
        assert len(index.select(direction=direction, split="train")) == len(selected)
    assert index.config.time_precision == "date"
    assert_sopat_v4_causality(index)
    for example in index:
        target_day = example.target.time
        assert example.target_time == target_day
        assert all(observation.measurement.time <= target_day for observation in example.observations)
        assert example.anchor_pair.source.time < target_day
        assert example.anchor_pair.target.time < target_day
        if example.task_mode == TRANSLATION:
            assert example.query_source is not None
            assert sum(role.role == "query_source" for role in example.observations) == 1
            assert 0 <= _age_days(target_day, example.query_source.measurement.time) <= 1
        else:
            assert example.query_source is None
            assert all(role.role == "history" for role in example.observations)
            assert all(_age_days(target_day, role.measurement.time) > 1 for role in example.observations)


def test_observation_set_serialization_is_order_invariant_and_splits_are_isolated(
    records: list[PairRecord],
) -> None:
    validation_records = [
        _record(
            number + 20,
            s1_date=f"2020-02-{number * 2 + 1:02d}",
            s2_date=f"2020-02-{number * 2 + 1:02d}",
            split="validation_temporal",
        )
        for number in range(5)
    ]
    index = build_sopat_v4_index(
        [*records, *validation_records],
        splits=("train", "validation_temporal"),
        max_observations=3,
    )
    example = next(item for item in index if len(item.observations) > 1)
    reordered = replace(example, observations=tuple(reversed(example.observations)))

    assert reordered.observations == example.observations
    for item in index:
        record_ids = (
            item.target.record_id,
            item.anchor_pair.registration_id,
            *(observation.measurement.record_id for observation in item.observations),
        )
        assert all(f":{item.split}:" in record_id for record_id in record_ids)


def test_manifest_migration_keeps_manifest_asset_root_and_provenance(
    records: list[PairRecord], tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest" / "pairs.jsonl"
    write_pair_records(manifest, records)
    v3 = build_paired_temporal_index(manifest, direction=SAR_TO_OPTICAL, max_observations=3)

    migrated = migrate_paired_temporal_index_v4(v3, manifest)

    assert migrated.examples[0].provenance.source_manifest_sha256
    assert all(str(manifest.parent) in asset for asset in migrated.examples[0].target.asset_ids)
    assert_sopat_v4_causality(migrated)


def test_index_round_trip_binds_content_and_file_digest(
    records: list[PairRecord], tmp_path: Path
) -> None:
    index = _index(records)
    path = tmp_path / "sopat-v4.jsonl"

    written = write_sopat_v4_index(path, index)
    loaded = load_sopat_v4_index(path)

    assert written == sopat_v4_index_file_sha256(path)
    assert loaded.content_hash == index.content_hash
    assert loaded.protocol_hash == index.protocol_hash
    rows = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(rows[1])
    tampered["target_time"] = "2020-02-01"
    rows[1] = json.dumps(tampered, sort_keys=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises((ValueError, AssertionError), match="target_time|content hash"):
        load_sopat_v4_index(path)


def test_future_target_asset_or_role_leakage_is_rejected(records: list[PairRecord]) -> None:
    example = next(item for item in _index(records) if item.task_mode == TRANSLATION)
    future_observation = replace(
        example.observations[0],
        measurement=replace(example.observations[0].measurement, time="2020-12-31"),
    )
    future = replace(example, observations=(future_observation, *example.observations[1:]))
    with pytest.raises(AssertionError, match="after target"):
        assert_sopat_v4_causality((future,), config=SOPATIndexConfigV4())

    leaked_measurement = replace(
        example.observations[0].measurement,
        asset_ids=example.target.asset_ids,
    )
    leaked = replace(
        example,
        observations=(replace(example.observations[0], measurement=leaked_measurement), *example.observations[1:]),
    )
    with pytest.raises(AssertionError, match="target asset leaks"):
        assert_sopat_v4_causality((leaked,), config=SOPATIndexConfigV4())

    repeated_anchor = replace(
        example.observations[0].measurement,
        asset_ids=example.anchor_pair.source.asset_ids,
    )
    with pytest.raises(AssertionError, match="source anchor repeats"):
        assert_sopat_v4_causality(
            (
                replace(
                    example,
                    observations=(
                        replace(example.observations[0], measurement=repeated_anchor),
                        *example.observations[1:],
                    ),
                ),
            ),
            config=SOPATIndexConfigV4(),
        )

    invalid_forecast = next(item for item in _index(records) if item.task_mode == FORECAST)
    query_role = replace(invalid_forecast.observations[0], role="query_source")
    role_leak = replace(
        invalid_forecast,
        observations=(query_role, *invalid_forecast.observations[1:]),
        query_source_id=query_role.measurement.measurement_id,
    )
    with pytest.raises(AssertionError, match="forecast cannot contain"):
        assert_sopat_v4_causality((role_leak,), config=SOPATIndexConfigV4())


class _SyntheticBackend(Dataset[dict[str, object]]):
    def __init__(self, samples: tuple[object, ...], *, windows_per_sample: int) -> None:
        self.samples = samples
        self.windows_per_sample = windows_per_sample

    def __len__(self) -> int:
        return len(self.samples) * self.windows_per_sample

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_position, window = divmod(index, self.windows_per_sample)
        sample = self.samples[sample_position]
        count = len(sample.observation_pair_ids)
        present = torch.tensor([True] * count + [False], dtype=torch.bool)
        values = torch.arange((count + 1) * 2, dtype=torch.float32).reshape(count + 1, 2, 1, 1)
        valid = torch.ones((count + 1, 1, 1, 1), dtype=torch.float32)
        valid[-1] = 0
        return {
            "source_anchor_values": torch.zeros((2, 1, 1)),
            "source_anchor_valid": torch.ones((1, 1, 1)),
            "target_anchor_values": torch.zeros((3, 1, 1)),
            "target_anchor_valid": torch.ones((1, 1, 1)),
            "observation_values": values + window * 100,
            "observation_valid": valid,
            "observation_days": torch.tensor([-float(count - item) for item in range(count)] + [0.0]),
            "observation_present": present,
            "source_anchor_days": torch.tensor(-2.0),
            "target_anchor_days": torch.tensor(-2.0),
            "anchor_days": torch.tensor(-2.0),
            "target_values": torch.zeros((3, 1, 1)),
            "target_valid": torch.ones((1, 1, 1)),
            "high_frequency_valid": torch.zeros((1, 1, 1)),
            "high_frequency_eligible": torch.tensor(False),
            "high_frequency_weight": torch.tensor(0.0),
            "registration_shift_px": torch.tensor(float("inf")),
            "registration_zero_ncc": torch.tensor(float("nan")),
            "registration_best_ncc": torch.tensor(float("nan")),
            "registration_evidence_supported": torch.tensor(False),
            "sample_id": sample.sample_id,
            "direction": sample.direction,
            "task_mode": sample.task_mode,
        }


def test_direction_dataset_permutation_collation_and_chunk_window_routing(
    records: list[PairRecord],
) -> None:
    index = _index(records)
    legacy = paired_temporal_index_from_sopat_v4(index, direction=SAR_TO_OPTICAL, split="train")
    backend = _SyntheticBackend(legacy.samples, windows_per_sample=2)
    dataset = SOPATDirectionDataset(
        index,
        backend,
        direction=SAR_TO_OPTICAL,
        split="train",
        permutation_seed=7,
    )

    item_zero = dataset[0]
    item_one = dataset[1]
    assert len(dataset) == len(legacy.samples) * 2
    assert item_zero["sopat_example_id"] == item_one["sopat_example_id"]
    assert item_zero["observation_values"][0, 0, 0, 0] + 100 == item_one["observation_values"][0, 0, 0, 0]
    assert set(item_zero["sopat_observation_roles"]) <= {"history", "query_source"}
    dataset.set_epoch(2)
    first = dataset[0]["sopat_observation_ids"]
    dataset.set_epoch(2)
    assert dataset[0]["sopat_observation_ids"] == first
    batch = collate_sopat_direction((item_zero, item_one))
    assert batch["sopat_direction"] == SAR_TO_OPTICAL
    assert batch["observation_values"].shape[0] == 2
    with pytest.raises(ValueError, match="homogeneous"):
        collate_sopat_direction((item_zero, {**item_one, "sopat_direction": OPTICAL_TO_SAR}))


class _LoaderProbe:
    def __init__(self, direction: str, length: int) -> None:
        self.direction = direction
        self.length = length
        self.epochs: list[int] = []

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        return iter({"sopat_direction": self.direction, "value": value} for value in range(self.length))

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


def test_coupled_loader_cycles_short_side_and_forwards_epoch() -> None:
    sar = _LoaderProbe(SAR_TO_OPTICAL, 2)
    optical = _LoaderProbe(OPTICAL_TO_SAR, 3)
    loader = CoupledDirectionLoader({SAR_TO_OPTICAL: sar, OPTICAL_TO_SAR: optical})

    loader.set_epoch(5)
    steps = list(loader)

    assert len(steps) == 3
    assert [step.sar_to_optical["value"] for step in steps] == [0, 1, 0]
    assert [step.optical_to_sar["value"] for step in steps] == [0, 1, 2]
    assert sar.epochs == [5]
    assert optical.epochs == [5]


def _age_days(target: str, source: str) -> int:
    return int(target[-2:]) - int(source[-2:])
