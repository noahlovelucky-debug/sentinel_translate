from __future__ import annotations

import hashlib
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
    GlobalCrossTileHardNegativePlan,
    SOPATDirectionDataset,
    SOPATExampleV4,
    SOPATIndexConfigV4,
    SOPATIndexV4,
    assert_sopat_v4_causality,
    build_global_cross_tile_hard_negative_plan,
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


def _cross_tile_records(*, tile_count: int = 4) -> list[PairRecord]:
    """Make independently gridded routes with matching SOPAT chronology."""

    result: list[PairRecord] = []
    for tile_number in range(tile_count):
        tile = f"tile-cf-{tile_number}"
        for number, day in enumerate((1, 3, 5, 10, 13, 17)):
            result.append(
                _record(
                    tile_number * 100 + number,
                    tile=tile,
                    s1_date=f"2020-01-{day:02d}",
                    s2_date=f"2020-01-{day:02d}",
                )
            )
    return result


def _cross_tile_index(*, tile_count: int = 4) -> SOPATIndexV4:
    return build_sopat_v4_index(
        _cross_tile_records(tile_count=tile_count),
        splits=("train",),
        max_observations=3,
        horizon_days=180,
    )


def _cross_tile_forecast_index(*, tile_count: int = 3) -> SOPATIndexV4:
    records: list[PairRecord] = []
    dates = (
        ("2020-01-01", "2020-01-01"),
        ("2020-01-03", "2020-01-03"),
        ("2020-01-05", "2020-01-05"),
        ("2020-01-10", "2020-01-10"),
        ("2020-01-13", "2020-01-14"),
        ("2020-01-15", "2020-01-17"),
    )
    for tile_number in range(tile_count):
        tile = f"tile-mixed-{tile_number}"
        for number, (s1_date, s2_date) in enumerate(dates):
            records.append(
                _record(
                    tile_number * 100 + number,
                    tile=tile,
                    s1_date=s1_date,
                    s2_date=s2_date,
                )
            )
    return build_sopat_v4_index(
        records,
        splits=("train",),
        max_observations=3,
        horizon_days=180,
    )


def _cross_tile_example(
    index: SOPATIndexV4,
    *,
    tile: str,
    task_mode: str,
    observation_count: int,
    target_time: str | None = None,
) -> SOPATExampleV4:
    return next(
        example
        for example in index.select(direction=SAR_TO_OPTICAL, split="train")
        if example.tile == tile
        and example.task_mode == task_mode
        and len(example.observations) == observation_count
        and (target_time is None or example.target_time == target_time)
    )


def _route_index(
    index: SOPATIndexV4, examples: tuple[SOPATExampleV4, ...]
) -> SOPATIndexV4:
    return SOPATIndexV4(config=index.config, examples=examples)


def _route_record_ids(example: SOPATExampleV4) -> set[str]:
    return {
        example.target.record_id,
        example.anchor_pair.registration_id,
        *(observation.measurement.record_id for observation in example.observations),
    }


def _route_measurement_ids(example: SOPATExampleV4) -> set[str]:
    return {
        observation.measurement.measurement_id for observation in example.observations
    }


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


def test_global_cross_tile_hard_negative_plan_is_deterministic_and_disjoint() -> None:
    index = _cross_tile_index()
    plan = build_global_cross_tile_hard_negative_plan(
        index,
        direction=SAR_TO_OPTICAL,
        split="train",
    )
    repeated = GlobalCrossTileHardNegativePlan.from_index(
        index,
        direction=SAR_TO_OPTICAL,
        split="train",
    )

    assert plan == repeated
    assert plan.hash == plan.plan_hash == repeated.plan_hash
    assert plan.coverage == 1.0
    assert plan.tier_counts == {
        "same_task_exact_n": len(plan.mappings),
        "same_task_n_bin": 0,
        "same_task": 0,
        "same_orbit": 0,
    }
    assert plan.mapping_metadata == {
        "format_version": 1,
        "version": 1,
        "planner": "global_cross_tile_hard_v1",
        "selection": "relative_source_anchor_l1_then_sha256_v1",
        "alignment": "chronological_normalized_nearest_v1",
        "direction": SAR_TO_OPTICAL,
        "split": "train",
        "index_content_sha256": index.content_hash,
        "coverage": 1.0,
        "cross_tile_coverage": 1.0,
        "mappings": len(plan.mappings),
        "tier_counts": plan.tier_counts,
        "plan_hash": plan.plan_hash,
    }
    serialized = plan.to_dict()
    assert serialized["mappings"] == [mapping.to_dict() for mapping in plan.mappings]

    by_id = {example.sample_id: example for example in index.select(direction=SAR_TO_OPTICAL, split="train")}
    for mapping in plan.mappings:
        recipient = by_id[mapping.recipient_sample_id]
        donor = by_id[mapping.donor_sample_id]
        assert recipient.sample_id != donor.sample_id
        assert recipient.tile != donor.tile
        assert recipient.direction == donor.direction == SAR_TO_OPTICAL
        assert recipient.split == donor.split == "train"
        assert recipient.orbit == donor.orbit
        assert recipient.anchor_pair.registration_id != donor.anchor_pair.registration_id
        assert recipient.target.record_id != donor.target.record_id
        assert _route_record_ids(recipient).isdisjoint(_route_record_ids(donor))
        assert _route_measurement_ids(recipient).isdisjoint(_route_measurement_ids(donor))


def test_global_cross_tile_hard_negative_plan_fails_closed_without_cross_tile_donor() -> None:
    index = _cross_tile_index(tile_count=1)

    with pytest.raises(ValueError, match="no eligible donor"):
        build_global_cross_tile_hard_negative_plan(
            index,
            direction=SAR_TO_OPTICAL,
            split="train",
        )

    legacy = paired_temporal_index_from_sopat_v4(index, direction=SAR_TO_OPTICAL, split="train")
    backend = _SyntheticBackend(legacy.samples, windows_per_sample=1)
    with pytest.raises(ValueError, match="requires a global cross-tile"):
        SOPATDirectionDataset(
            index,
            backend,
            direction=SAR_TO_OPTICAL,
            split="train",
            include_cf=True,
        )


def test_global_cross_tile_hard_negative_plan_uses_tier_priority_and_same_orbit() -> None:
    full = _cross_tile_index()
    exact_n = _cross_tile_example(
        full,
        tile="tile-cf-0",
        task_mode=TRANSLATION,
        observation_count=2,
    )
    n_bin = _cross_tile_example(
        full,
        tile="tile-cf-1",
        task_mode=TRANSLATION,
        observation_count=3,
        target_time="2020-01-10",
    )
    same_task = _cross_tile_example(
        full,
        tile="tile-cf-2",
        task_mode=TRANSLATION,
        observation_count=1,
    )
    tier_index = _route_index(full, (exact_n, n_bin, same_task))
    tier_plan = build_global_cross_tile_hard_negative_plan(
        tier_index,
        direction=SAR_TO_OPTICAL,
        split="train",
    )

    recipient_mapping = tier_plan.donor_for(exact_n.sample_id)
    assert recipient_mapping.donor_sample_id == n_bin.sample_id
    assert recipient_mapping.tier == "same_task_n_bin"
    assert tier_plan.donor_for(same_task.sample_id).tier == "same_task"

    mixed = _cross_tile_forecast_index()
    translation = _cross_tile_example(
        mixed,
        tile="tile-mixed-0",
        task_mode=TRANSLATION,
        observation_count=1,
    )
    forecast_one = _cross_tile_example(
        mixed,
        tile="tile-mixed-1",
        task_mode=FORECAST,
        observation_count=3,
    )
    forecast_two = _cross_tile_example(
        mixed,
        tile="tile-mixed-2",
        task_mode=FORECAST,
        observation_count=3,
    )
    orbit_index = _route_index(mixed, (translation, forecast_one, forecast_two))
    orbit_plan = build_global_cross_tile_hard_negative_plan(
        orbit_index,
        direction=SAR_TO_OPTICAL,
        split="train",
    )

    assert orbit_plan.donor_for(translation.sample_id).tier == "same_orbit"
    assert orbit_plan.donor_for(forecast_one.sample_id).tier == "same_task_exact_n"


def test_global_cross_tile_hard_negative_plan_prefers_time_matched_donor_before_sha() -> None:
    full = _cross_tile_index()
    examples = tuple(
        _cross_tile_example(
            full,
            tile=f"tile-cf-{tile_number}",
            task_mode=TRANSLATION,
            observation_count=3,
            target_time="2020-01-10",
        )
        for tile_number in range(4)
    )
    recipient, *candidates = examples
    sha_first = min(
        candidates,
        key=lambda donor: (
            hashlib.sha256(
                f"global_cross_tile_hard_v1:{recipient.sample_id}:{donor.sample_id}".encode()
            ).hexdigest(),
            donor.sample_id,
        ),
    )
    history_dates = iter(("2020-01-08", "2020-01-09"))
    timing_mismatch = replace(
        sha_first,
        observations=tuple(
            replace(
                observation,
                measurement=replace(observation.measurement, time=next(history_dates)),
            )
            if observation.role == "history"
            else observation
            for observation in sha_first.observations
        ),
    )
    timing_index = _route_index(
        full,
        tuple(
            timing_mismatch if example.sample_id == sha_first.sample_id else example
            for example in examples
        ),
    )
    plan = build_global_cross_tile_hard_negative_plan(
        timing_index,
        direction=SAR_TO_OPTICAL,
        split="train",
    )
    expected = min(
        (candidate for candidate in candidates if candidate.sample_id != sha_first.sample_id),
        key=lambda donor: (
            hashlib.sha256(
                f"global_cross_tile_hard_v1:{recipient.sample_id}:{donor.sample_id}".encode()
            ).hexdigest(),
            donor.sample_id,
        ),
    )

    assert plan.donor_for(recipient.sample_id).tier == "same_task_exact_n"
    assert plan.donor_for(recipient.sample_id).donor_sample_id == expected.sample_id


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


class _TaggedSyntheticBackend(_SyntheticBackend):
    """Expose sample identity in pixels so donor leakage is observable in tests."""

    def __getitem__(self, index: int) -> dict[str, object]:
        item = super().__getitem__(index)
        sample_position, _ = divmod(index, self.windows_per_sample)
        tag = float((sample_position + 1) * 1000)
        observation_values = item["observation_values"]
        source_anchor_values = item["source_anchor_values"]
        target_anchor_values = item["target_anchor_values"]
        target_values = item["target_values"]
        assert isinstance(observation_values, torch.Tensor)
        assert isinstance(source_anchor_values, torch.Tensor)
        assert isinstance(target_anchor_values, torch.Tensor)
        assert isinstance(target_values, torch.Tensor)
        item["observation_values"] = observation_values + tag
        item["source_anchor_values"] = torch.full_like(source_anchor_values, tag + 1)
        item["target_anchor_values"] = torch.full_like(target_anchor_values, tag + 2)
        item["target_values"] = torch.full_like(target_values, tag + 3)
        return item


def _fallback_counterfactual_dataset():
    full = _cross_tile_index()
    recipient = _cross_tile_example(
        full,
        tile="tile-cf-0",
        task_mode=TRANSLATION,
        observation_count=2,
    )
    donor = _cross_tile_example(
        full,
        tile="tile-cf-1",
        task_mode=TRANSLATION,
        observation_count=3,
        target_time="2020-01-10",
    )
    same_task = _cross_tile_example(
        full,
        tile="tile-cf-2",
        task_mode=TRANSLATION,
        observation_count=1,
    )
    index = _route_index(full, (recipient, donor, same_task))
    plan = build_global_cross_tile_hard_negative_plan(
        index,
        direction=SAR_TO_OPTICAL,
        split="train",
    )
    assert plan.donor_for(recipient.sample_id).donor_sample_id == donor.sample_id
    assert plan.donor_for(recipient.sample_id).tier == "same_task_n_bin"
    legacy = paired_temporal_index_from_sopat_v4(index, direction=SAR_TO_OPTICAL, split="train")
    backend = _TaggedSyntheticBackend(legacy.samples, windows_per_sample=2)
    baseline = SOPATDirectionDataset(
        index,
        backend,
        direction=SAR_TO_OPTICAL,
        split="train",
        permutation_seed=17,
    )
    counterfactual = SOPATDirectionDataset(
        index,
        backend,
        direction=SAR_TO_OPTICAL,
        split="train",
        permutation_seed=17,
        hard_negative_plan=plan,
        include_cf=True,
    )
    return recipient, donor, plan, backend, legacy, baseline, counterfactual


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


def test_global_counterfactual_uses_donor_history_only_and_preserves_recipient_order() -> None:
    recipient, donor, plan, backend, legacy, baseline, counterfactual = (
        _fallback_counterfactual_dataset()
    )
    recipient_position = next(
        position
        for position, example in enumerate(counterfactual.examples)
        if example.sample_id == recipient.sample_id
    )
    position = recipient_position * counterfactual.backend_windows + 1
    unchanged = baseline[position]
    item = counterfactual[position]
    donor_position = next(
        position
        for position, sample in enumerate(legacy.samples)
        if sample.sample_id == donor.sample_id
    )
    donor_item = backend[donor_position * counterfactual.backend_windows + 1]
    donor_values = donor_item["observation_values"]
    donor_valid = donor_item["observation_valid"]
    donor_target = donor_item["target_values"]
    donor_source_anchor = donor_item["source_anchor_values"]
    donor_target_anchor = donor_item["target_anchor_values"]
    assert isinstance(donor_values, torch.Tensor)
    assert isinstance(donor_valid, torch.Tensor)
    assert isinstance(donor_target, torch.Tensor)
    assert isinstance(donor_source_anchor, torch.Tensor)
    assert isinstance(donor_target_anchor, torch.Tensor)

    for key in (
        "source_anchor_values",
        "source_anchor_valid",
        "target_anchor_values",
        "target_anchor_valid",
        "observation_values",
        "observation_valid",
        "observation_days",
        "observation_present",
        "target_values",
        "target_valid",
    ):
        assert torch.equal(item[key], unchanged[key])
    assert item["sopat_tile"] == recipient.tile
    assert item["sopat_cf_donor_sample_id"] == donor.sample_id
    assert item["sopat_cf_donor_tile"] == donor.tile
    assert item["sopat_cf_donor_grid_id"] == donor.canonical_grid.grid_id
    assert item["sopat_cf_tier"] == "same_task_n_bin"
    assert item["sopat_cf_plan_hash"] == plan.plan_hash

    recipient_ids = tuple(
        observation.measurement.measurement_id
        for observation in recipient.observation_ordered_for_backend()
    )
    item_ids = item["sopat_observation_ids"]
    assert isinstance(item_ids, tuple)
    recipient_order = torch.tensor(
        [recipient_ids.index(measurement_id) for measurement_id in item_ids],
        dtype=torch.long,
    )
    chronological = donor_values[: len(donor.observations)].index_select(
        0,
        torch.tensor([0, 2], dtype=torch.long),
    )
    chronological_valid = donor_valid[: len(donor.observations)].index_select(
        0,
        torch.tensor([0, 2], dtype=torch.long),
    )
    expected_values = chronological.index_select(0, recipient_order)
    expected_valid = chronological_valid.index_select(0, recipient_order)
    counterfactual_values = item["counterfactual_observation_values"]
    counterfactual_valid = item["counterfactual_observation_valid"]
    assert isinstance(counterfactual_values, torch.Tensor)
    assert isinstance(counterfactual_valid, torch.Tensor)
    assert torch.equal(counterfactual_values, expected_values)
    assert torch.equal(counterfactual_valid, expected_valid)

    # Tagged tensors make it explicit that donor targets and anchors never enter the item.
    assert not torch.equal(item["target_values"], donor_target)
    assert not torch.equal(item["source_anchor_values"], donor_source_anchor)
    assert not torch.equal(item["target_anchor_values"], donor_target_anchor)


def test_global_counterfactual_collation_pads_real_recipient_lengths_independently() -> None:
    _, _, _, _, _, _, dataset = _fallback_counterfactual_dataset()
    item_one = next(
        dataset[position * dataset.backend_windows]
        for position, example in enumerate(dataset.examples)
        if len(example.observations) == 1
    )
    item_three = next(
        dataset[position * dataset.backend_windows]
        for position, example in enumerate(dataset.examples)
        if len(example.observations) == 3
    )
    one_values = item_one["counterfactual_observation_values"]
    one_valid = item_one["counterfactual_observation_valid"]
    three_values = item_three["counterfactual_observation_values"]
    three_valid = item_three["counterfactual_observation_valid"]
    assert isinstance(one_values, torch.Tensor)
    assert isinstance(one_valid, torch.Tensor)
    assert isinstance(three_values, torch.Tensor)
    assert isinstance(three_valid, torch.Tensor)

    batch = collate_sopat_direction((item_one, item_three))
    batch_values = batch["counterfactual_observation_values"]
    batch_valid = batch["counterfactual_observation_valid"]
    observation_values = batch["observation_values"]
    observation_valid = batch["observation_valid"]
    assert isinstance(batch_values, torch.Tensor)
    assert isinstance(batch_valid, torch.Tensor)
    assert isinstance(observation_values, torch.Tensor)
    assert isinstance(observation_valid, torch.Tensor)
    assert batch_values.shape == observation_values.shape
    assert batch_valid.shape == observation_valid.shape
    assert torch.equal(batch_values[0, : one_values.shape[0]], one_values)
    assert torch.equal(batch_valid[0, : one_valid.shape[0]], one_valid)
    assert torch.equal(batch_values[1, : three_values.shape[0]], three_values)
    assert torch.equal(batch_valid[1, : three_valid.shape[0]], three_valid)
    assert not batch_values[0, one_values.shape[0] :].bool().any()
    assert not batch_valid[0, one_valid.shape[0] :].bool().any()
    assert batch["sopat_tile"] == [item_one["sopat_tile"], item_three["sopat_tile"]]
    assert batch["sopat_cf_donor_sample_id"] == [
        item_one["sopat_cf_donor_sample_id"],
        item_three["sopat_cf_donor_sample_id"],
    ]
    for batch_index, item in enumerate((item_one, item_three)):
        for key in (
            "observation_values",
            "observation_valid",
            "observation_days",
            "observation_present",
        ):
            values = item[key]
            assert isinstance(values, torch.Tensor)
            assert torch.equal(batch[key][batch_index, : values.shape[0]], values)


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
