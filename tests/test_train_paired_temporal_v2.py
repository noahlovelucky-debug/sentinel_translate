from __future__ import annotations

import importlib.util
import math
import os
from datetime import date, timedelta
from pathlib import Path

import pytest
import torch

from sentinel_v3.dataset_builder import PairRecord
from sentinel_v3.paired_temporal_data import (
    OPTICAL_TO_SAR,
    SAR_TO_OPTICAL,
    build_paired_temporal_index,
    write_pair_records,
    write_paired_temporal_index,
)
from sentinel_v3.paired_temporal_training import PairedTemporalTrainConfig
from sentinel_v3.paired_temporal_v2 import PairedTemporalConfig, SparsePairedAnchorTransport
from sentinel_v3.schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER


def _runner():  # type: ignore[no-untyped-def]
    path = Path(__file__).parents[1] / "scripts" / "train_paired_temporal_v2.py"
    spec = importlib.util.spec_from_file_location("train_paired_temporal_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics() -> dict[str, dict[str, float]]:
    values = {
        "physical_rmse": 0.2,
        "detail_mae": 0.1,
        "detail_zero_mae": 0.2,
        "detail_improvement_percent": 50.0,
        "physical_anchor_improvement_percent": 10.0,
        "source_evidence_improvement_percent": 5.0,
        "flow_objective": 0.3,
        "visual_frequency": 0.1,
        "visual_frequency_improvement_percent": 10.0,
        "visual_frequency_improvement_fraction": 1.0,
        "visual_over_physical": 1.02,
        "visual_rmse_regression_fraction": 0.0,
        "pre_projection_violation": 0.0,
    }
    return {"translation/one": dict(values), "forecast/one": dict(values)}


def test_validation_score_requires_deployment_one_frame_gates() -> None:
    runner = _runner()
    config = PairedTemporalTrainConfig(direction="sar_to_optical", visual_rmse_budget=1.05)
    assert math.isfinite(runner._validation_score(_metrics(), "physical", config))
    incomplete = _metrics()
    incomplete.pop("forecast/one")
    assert math.isinf(runner._validation_score(incomplete, "physical", config))
    no_source = _metrics()
    no_source["translation/one"]["source_evidence_improvement_percent"] = 0.0
    assert math.isinf(runner._validation_score(no_source, "physical", config))
    weak_anchor = _metrics()
    weak_anchor["forecast/one"]["physical_anchor_improvement_percent"] = 4.99
    assert math.isinf(runner._validation_score(weak_anchor, "physical", config))
    weak_source = _metrics()
    weak_source["forecast/one"]["source_evidence_improvement_percent"] = 0.99
    assert math.isinf(runner._validation_score(weak_source, "physical", config))
    no_detail = _metrics()
    no_detail["translation/one"]["detail_improvement_percent"] = 0.0
    assert math.isinf(runner._validation_score(no_detail, "detail", config))
    over_budget = _metrics()
    over_budget["forecast/one"]["visual_over_physical"] = 1.06
    assert math.isinf(runner._validation_score(over_budget, "balance", config))
    scene_regression = _metrics()
    scene_regression["forecast/one"]["visual_rmse_regression_fraction"] = 0.2
    assert math.isinf(runner._validation_score(scene_regression, "balance", config))
    missing_regime_improvement = _metrics()
    missing_regime_improvement["forecast/one"]["source_evidence_improvement_percent"] = 0.0
    assert math.isinf(runner._validation_score(missing_regime_improvement, "physical", config))


def test_protocol_hash_binds_the_full_canonical_config() -> None:
    runner = _runner()
    config = {
        "data": {"horizon_days": 180},
        "model": {"width": 32},
        "training": {"observation_dropout": 0.35},
        "validation": {"fixed_seed": 71},
    }
    first = runner._protocol_hash(config, "sar_to_optical")
    changed = {**config, "training": {"observation_dropout": 0.1}}
    assert runner._protocol_hash(changed, "sar_to_optical") != first


def test_balance_release_sync_is_a_noop_for_one_rank() -> None:
    runner = _runner()
    model = SparsePairedAnchorTransport(
        PairedTemporalConfig(width=16, latent_channels=4, attention_heads=4, flow_steps=1)
    )
    before = (model.detail_scale.detach().clone(), model.visual_scale.detach().clone())
    runner._synchronize_balance_release(model, world_size=1)
    torch.testing.assert_close(model.detail_scale, before[0])
    torch.testing.assert_close(model.visual_scale, before[1])


def _local_record(root: Path, split: str, index: int, start: date) -> PairRecord:
    record_root = root / "assets" / split / f"record-{index:03d}"
    source_date = start + timedelta(days=index)
    target_date = source_date + timedelta(days=index % 2)
    manifest_root = root / "manifests"
    s2 = {
        channel: os.path.relpath(record_root / "s2" / f"{channel}.tif", manifest_root)
        for channel in S2_CHANNEL_ORDER
    }
    sar = {
        channel: os.path.relpath(record_root / "sar" / f"{channel}.tif", manifest_root)
        for channel in SAR_CHANNEL_ORDER
    }
    scl = os.path.relpath(record_root / "scl.tif", manifest_root)
    for path in (*s2.values(), *sar.values(), scl):
        target = root / "manifests" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    return PairRecord(
        pair_id=f"2020:local:{split}:{index:03d}:ascending",
        year=2020,
        tile="local-tile",
        tile_row=1,
        tile_col=1,
        split=split,
        refit_split="excluded",
        s2_date=target_date.isoformat(),
        s1_date=source_date.isoformat(),
        orbit="ascending",
        delta_days=index % 2,
        s2=s2,
        scl=scl,
        sar=sar,
        clear_fraction=1.0,
        valid_fraction=1.0,
        width=64,
        height=64,
        crs="EPSG:32650",
        transform=[10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0],
        gsd=10.0,
    )


def _local_explicit_config(root: Path, *, direction_mapping: bool = False) -> dict[str, object]:
    records = [
        *(
            _local_record(root, "train", index, date(2020, 1, 1))
            for index in range(5)
        ),
        *(
            _local_record(root, "validation_temporal", index, date(2021, 1, 1))
            for index in range(5)
        ),
    ]
    manifest = root / "manifests" / "pairs.jsonl"
    write_pair_records(manifest, records)
    index_root = root / "indexes"
    for direction in (SAR_TO_OPTICAL, OPTICAL_TO_SAR):
        for label, split in (("train", "train"), ("validation", "validation_temporal")):
            index = build_paired_temporal_index(
                records,
                direction=direction,
                min_observations=1,
                max_observations=4,
                horizon_days=180,
                anchor_max_delta_days=1,
                max_anchors_per_query=2,
                translation_max_delta_days=1,
                split=split,
                orbit="ascending",
                task_modes=("translation", "forecast"),
                max_samples=2,
                asset_root=manifest.parent,
            )
            assert len(index) == 2
            write_paired_temporal_index(index_root / direction / f"{label}.jsonl", index)
    data: dict[str, object] = {
        "manifest": str(manifest),
        "orbit": "ascending",
        "anchor_pair_max_delta_days": 1,
        "maximum_anchors_per_query": 2,
        "horizon_days": 180,
        "translation_max_delta_days": 1,
        "minimum_observations": 1,
        "maximum_observations": 4,
        "max_train_samples": 2,
        "max_validation_samples": 2,
        "train_split": "train",
        "validation_split": "validation_temporal",
        "task_modes": ["translation", "forecast"],
    }
    if direction_mapping:
        data["train_index"] = {
            direction: str(index_root / direction / "train.jsonl")
            for direction in (SAR_TO_OPTICAL, OPTICAL_TO_SAR)
        }
        data["validation_index"] = {
            direction: str(index_root / direction / "validation.jsonl")
            for direction in (SAR_TO_OPTICAL, OPTICAL_TO_SAR)
        }
    else:
        data["train_index"] = str(index_root / "{direction}" / "train.jsonl")
        data["validation_index"] = str(index_root / "{direction}" / "validation.jsonl")
    return {"data": data, "model": {"width": 32}}


@pytest.mark.parametrize("direction", (SAR_TO_OPTICAL, OPTICAL_TO_SAR))
def test_prepare_indexes_loads_explicit_local_indexes_without_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direction: str
) -> None:
    runner = _runner()
    config = _local_explicit_config(tmp_path)

    def no_rebuild(*args: object, **kwargs: object) -> object:
        raise AssertionError("explicit local indexes must not invoke the index builder")

    monkeypatch.setattr(runner, "build_paired_temporal_index", no_rebuild)
    prepared = runner._prepare_indexes(
        config, direction, tmp_path / "output", rank=0, world_size=1
    )

    assert prepared.explicit
    assert prepared.train_path == tmp_path / "indexes" / direction / "train.jsonl"
    assert prepared.validation_path == tmp_path / "indexes" / direction / "validation.jsonl"
    assert len(prepared.protocol_hash) == 64


def test_prepare_indexes_accepts_direction_mapping_and_hashes_artifacts(tmp_path: Path) -> None:
    runner = _runner()
    config = _local_explicit_config(tmp_path, direction_mapping=True)
    first = runner._prepare_indexes(config, SAR_TO_OPTICAL, tmp_path / "output", 0, 1)
    index_path = tmp_path / "indexes" / SAR_TO_OPTICAL / "train.jsonl"
    index_path.write_text(index_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    second = runner._prepare_indexes(config, SAR_TO_OPTICAL, tmp_path / "output", 0, 1)

    assert first.protocol_hash != second.protocol_hash
    assert second.explicit


def test_prepare_indexes_rejects_explicit_wrong_direction_and_count(tmp_path: Path) -> None:
    runner = _runner()
    config = _local_explicit_config(tmp_path)
    data = config["data"]
    assert isinstance(data, dict)
    optical_train = tmp_path / "indexes" / OPTICAL_TO_SAR / "train.jsonl"
    data["train_index"] = str(optical_train)
    with pytest.raises(ValueError, match="requested direction differs"):
        runner._prepare_indexes(config, SAR_TO_OPTICAL, tmp_path / "output", 0, 1)

    config = _local_explicit_config(tmp_path / "count")
    data = config["data"]
    assert isinstance(data, dict)
    data["max_train_samples"] = 3
    with pytest.raises(ValueError, match="has 2 samples, expected 3"):
        runner._prepare_indexes(config, SAR_TO_OPTICAL, tmp_path / "output", 0, 1)


def test_prepare_indexes_rejects_missing_local_asset(tmp_path: Path) -> None:
    runner = _runner()
    config = _local_explicit_config(tmp_path)
    data = config["data"]
    assert isinstance(data, dict)
    manifest = Path(str(data["manifest"]))
    records = {record.pair_id: record for record in runner.load_pair_records(manifest)}
    index = runner.load_paired_temporal_index(
        tmp_path / "indexes" / SAR_TO_OPTICAL / "train.jsonl"
    )
    # Query optical is always a selected label asset, so its removal must be
    # detected before a Dataset is created.
    missing = manifest.parent / records[index.samples[0].query_pair_id].s2[S2_CHANNEL_ORDER[0]]
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="missing local asset"):
        runner._prepare_indexes(config, SAR_TO_OPTICAL, tmp_path / "output", 0, 1)


def test_prepare_indexes_rejects_missing_explicit_index_file(tmp_path: Path) -> None:
    runner = _runner()
    config = _local_explicit_config(tmp_path)
    (tmp_path / "indexes" / SAR_TO_OPTICAL / "validation.jsonl").unlink()

    with pytest.raises(FileNotFoundError, match="explicit validation paired temporal index is missing"):
        runner._prepare_indexes(config, SAR_TO_OPTICAL, tmp_path / "output", 0, 1)


def test_prepare_indexes_keeps_automatic_build_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner()
    config = _local_explicit_config(tmp_path)
    data = config["data"]
    assert isinstance(data, dict)
    data.pop("train_index")
    data.pop("validation_index")
    calls = 0
    original = runner.build_paired_temporal_index

    def tracked(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "build_paired_temporal_index", tracked)
    prepared = runner._prepare_indexes(config, SAR_TO_OPTICAL, tmp_path / "output", 0, 1)

    assert calls == 2
    assert not prepared.explicit
    assert prepared.train_path.is_file()
    assert prepared.validation_path.is_file()
