"""Focused protocol tests for the leakage-safe downstream cache layer."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest
import torch

import sentinel_v3.downstream_data as downstream
from sentinel_v3.downstream_probe import ProbeCache
from sentinel_v3.schema import S2_CHANNEL_ORDER
from sentinel_v3.sensors import SENTINEL1, SENTINEL2


def _record(
    pair_id: str,
    split: str,
    *,
    delta_days: int = 0,
    width: int = 512,
    height: int = 512,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "split": split,
        "delta_days": delta_days,
        "tile": f"tile-{pair_id}",
        "s1_date": "2020-05-01",
        "s2_date": "2020-12-31",
        "orbit": "ascending",
        "gsd": 10.0,
        "width": width,
        "height": height,
        "sar": {"vv": "vv.tif", "vh": "vh.tif"},
        "s2": {str(band): f"forbidden-{band}.tif" for band in S2_CHANNEL_ORDER},
        "scl": "forbidden-scl.tif",
    }


def _write_manifest(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest


def _plan(
    tmp_path: Path,
    records: list[dict[str, Any]],
    *,
    crop_size: int = 256,
    index: dict[str, Any] | None = None,
) -> downstream.CachePlan:
    manifest = _write_manifest(tmp_path, records)
    checkpoint = tmp_path / "best_physical.pt"
    checkpoint.write_bytes(b"test checkpoint")
    config = tmp_path / "downstream.yaml"
    config.write_text("protocol: test\n", encoding="utf-8")
    train_shards = tmp_path / "train-index.json"
    payload = {
        "split": "train",
        "manifest_sha256": downstream.file_sha256(manifest),
        "shards": [],
    }
    if index is not None:
        payload.update(index)
    train_shards.write_text(json.dumps(payload), encoding="utf-8")
    return downstream.CachePlan(
        manifest=manifest,
        train_shards=train_shards,
        checkpoint=checkpoint,
        checkpoint_sha256=downstream.file_sha256(checkpoint),
        cache_root=tmp_path / "cache",
        config_path=config,
        config_sha256=downstream.file_sha256(config),
        crop_size=crop_size,
    )


def _sample(
    record: dict[str, Any], *, crop_size: int = 2, partition: str | None = None
) -> downstream.CropSample:
    resolved_partition = partition or str(record["split"])
    return downstream.CropSample(
        sample_id=f"{resolved_partition}:{record['pair_id']}:0:0:{crop_size}:{crop_size}",
        partition=resolved_partition,
        pair_id=str(record["pair_id"]),
        tile=str(record["tile"]),
        s1_date=str(record["s1_date"]),
        s2_date=str(record["s2_date"]),
        orbit=str(record["orbit"]),
        gsd=float(record["gsd"]),
        window=(0, 0, crop_size, crop_size),
        record=record,
    )


def test_scl_proxy_labels_fold_only_preregistered_classes() -> None:
    labels = downstream.scl_proxy_labels(
        np.array([[0, 2, 4, 5], [6, 7, 9, 11]], dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        labels,
        np.array([[-1, -1, 1, 0], [0, -1, -1, -1]], dtype=np.int64),
    )


@pytest.mark.parametrize(
    "split",
    ("validation_temporal", "test_spatial", "test_temporal", "test_joint", "test_other"),
)
def test_split_guard_rejects_closed_or_unknown_splits(split: str) -> None:
    with pytest.raises(ValueError, match="split"):
        downstream.assert_allowed_split(split)


def test_prepare_rejects_a_manually_supplied_closed_split_sample(tmp_path: Path) -> None:
    record = _record("closed", "test_spatial", width=2, height=2)
    plan = _plan(tmp_path, [record], crop_size=2)

    with pytest.raises(ValueError, match="split"):
        downstream.prepare_cache(plan, [_sample(record)])


def test_train_uses_exactly_sixteen_canonical_windows_per_same_day_pair(tmp_path: Path) -> None:
    train = _record("train-0", "train", width=1024, height=1024)
    skipped = _record("train-not-same-day", "train", delta_days=1)
    windows = torch.tensor([[index * 8, 0, 256, 256] for index in range(16)])
    shard = tmp_path / "fixed-train.pt"
    torch.save({"pair_id": ["train-0"] * 16, "window": windows}, shard)
    plan = _plan(
        tmp_path,
        [skipped, train, _record("heldout", "unused_spatial")],
        index={"shards": [{"pair_id": "train-0", "delta_days": 0, "path": str(shard)}]},
    )

    samples = downstream.train_fixed_window_samples(plan)

    assert len(samples) == 16
    assert [sample.window for sample in samples] == [tuple(row.tolist()) for row in windows]
    assert {sample.partition for sample in samples} == {"train"}


def test_unused_spatial_uses_only_delta_zero_center_crop(tmp_path: Path) -> None:
    heldout = _record("heldout-0", "unused_spatial", width=768, height=512)
    skipped = _record("heldout-not-same-day", "unused_spatial", delta_days=2)
    plan = _plan(tmp_path, [heldout, skipped])

    samples = downstream.heldout_center_samples(plan)

    assert len(samples) == 1
    assert samples[0].partition == "unused_spatial"
    assert samples[0].window == (256, 128, 256, 256)


def test_rank_shards_are_disjoint_and_deterministic() -> None:
    records = [_record(f"pair-{index}", "train") for index in range(5)]
    samples = [_sample(record) for record in records]
    shards = [downstream.rank_shard(samples, rank, 3) for rank in range(3)]

    assigned = [sample.sample_id for shard in shards for sample in shard]
    assert set(assigned) == {sample.sample_id for sample in samples}
    assert len(set(assigned)) == len(samples)
    assert downstream.rank_shard(samples, 1, 3) == shards[1]


def test_sar_reader_never_opens_optical_or_scl_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    rasters = {
        "vv.tif": np.array([[200, 0], [400, 600]], dtype=np.uint16),
        "vh.tif": np.array([[300, 500], [0, 700]], dtype=np.uint16),
    }

    class FakeWindow:
        def __init__(self, *values: int) -> None:
            self.values = values

    class FakeDataset:
        def __init__(self, path: str) -> None:
            self.path = path

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self, index: int, *, window: FakeWindow) -> np.ndarray:
            assert index == 1
            assert window.values == (0, 0, 2, 2)
            return rasters[self.path]

    def fake_open(path: str) -> FakeDataset:
        opened.append(path)
        if path not in rasters:
            raise AssertionError(f"non-SAR asset opened by generator: {path}")
        return FakeDataset(path)

    rasterio = types.ModuleType("rasterio")
    rasterio.open = fake_open  # type: ignore[attr-defined]
    rasterio_windows = types.ModuleType("rasterio.windows")
    rasterio_windows.Window = FakeWindow  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rasterio", rasterio)
    monkeypatch.setitem(sys.modules, "rasterio.windows", rasterio_windows)

    values, valid = downstream.read_sar_raw_valid(_record("pair-0", "train"), (0, 0, 2, 2))

    assert opened == ["vv.tif", "vh.tif"]
    np.testing.assert_array_equal(valid, np.array([[True, False], [False, True]]))
    np.testing.assert_allclose(values[:, 0, 0], np.array([-49.0, -48.5]))
    np.testing.assert_array_equal(values[:, ~valid], np.zeros((2, 2), dtype=np.float32))


def test_checkpoint_loader_uses_ema_and_disables_temporal_prior(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_physical.pt"
    checkpoint.write_bytes(b"ema checkpoint")
    calls: dict[str, object] = {}

    class FakeModel:
        temporal_prior: object | None = object()

        def configure_temporal_prior(self, config: object) -> None:
            calls["prior"] = config
            self.temporal_prior = None

        def eval(self) -> FakeModel:
            calls["eval"] = True
            return self

    model = FakeModel()

    def loader(path: Path, device: torch.device, *, use_ema: bool) -> FakeModel:
        calls.update(path=path, device=device, use_ema=use_ema)
        return model

    resolved = downstream.load_frozen_physical_model(
        checkpoint,
        downstream.file_sha256(checkpoint),
        torch.device("cpu"),
        loader=loader,
    )

    assert resolved is model
    assert calls["use_ema"] is True
    assert calls["prior"] is None
    assert calls["eval"] is True


def test_physical_generator_uses_direct_physical_and_s1_target_metadata() -> None:
    record = _record("pair-0", "train")
    sample = _sample(record)
    calls: dict[str, object] = {}

    class FakeModel:
        temporal_prior = None

        def physical(
            self,
            values: torch.Tensor,
            source: object,
            target: object,
            valid: torch.Tensor,
            **kwargs: object,
        ) -> tuple[torch.Tensor, torch.Tensor, object]:
            calls.update(values=values, source=source, target=target, valid=valid, **kwargs)
            output = torch.full((1, 10, 2, 2), 0.5)
            return output, torch.empty(0), object()

    output = downstream.generate_physical_optical(
        FakeModel(),
        sample,
        np.full((2, 2, 2), -12.0, dtype=np.float32),
        np.array([[True, False], [True, True]]),
        torch.device("cpu"),
    )

    assert output.shape == (10, 2, 2)
    assert calls["source"] is SENTINEL1
    assert calls["target"] is SENTINEL2
    assert torch.equal(calls["valid"], torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]))
    metadata = calls["metadata"]
    assert isinstance(metadata, torch.Tensor)
    assert metadata[0, 0].item() == 0.0
    assert metadata[0, 2].item() == pytest.approx(metadata[0, 4].item())
    assert metadata[0, 3].item() == pytest.approx(metadata[0, 5].item())


def test_cache_resumes_only_after_float16_checksum_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record("pair-0", "train", width=2, height=2)
    plan = _plan(tmp_path, [record], crop_size=2)
    sample = _sample(record)
    downstream.prepare_cache(plan, [sample])
    calls = {"generated": 0}

    class FakeModel:
        temporal_prior = None

    def fake_loader(*_: object) -> FakeModel:
        return FakeModel()

    def fake_sar_reader(*_: object) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((2, 2, 2), dtype=np.float32), np.ones((2, 2), dtype=bool)

    def fake_generator(*_: object) -> torch.Tensor:
        calls["generated"] += 1
        return torch.full((10, 2, 2), 0.5)

    monkeypatch.setattr(downstream, "read_sar_raw_valid", fake_sar_reader)
    monkeypatch.setattr(downstream, "generate_physical_optical", fake_generator)

    first = downstream.cache_rank_samples(
        plan,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        model_loader=fake_loader,
    )
    second = downstream.cache_rank_samples(
        plan,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        model_loader=fake_loader,
    )
    path = downstream.sample_cache_path(plan.cache_root, sample)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["tensor_sha256"] = "corrupt"
    torch.save(payload, path)
    repaired = downstream.cache_rank_samples(
        plan,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        model_loader=fake_loader,
    )
    manifest = downstream.finalize_cache(plan)

    assert first == {"assigned": 1, "generated": 1, "reused": 0}
    assert second == {"assigned": 1, "generated": 0, "reused": 1}
    assert repaired == {"assigned": 1, "generated": 1, "reused": 0}
    assert calls["generated"] == 2
    assert len(manifest["entries"]) == 1


def test_materialize_probe_cache_writes_probe_contract_chunks_after_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_record = _record("train", "train", width=2, height=2)
    dev_record = _record("dev", "train", width=2, height=2)
    test_record = _record("test", "unused_spatial", width=2, height=2)
    dev_record["tile"] = "tile-dev"
    test_record["tile"] = "tile-test"
    plan = _plan(tmp_path, [train_record, dev_record, test_record], crop_size=2)
    samples = [_sample(record) for record in (train_record, dev_record, test_record)]
    downstream.prepare_cache(plan, samples)

    class FakeModel:
        temporal_prior = None

    def fake_loader(*_: object) -> FakeModel:
        return FakeModel()

    def generation_sar_reader(*_: object) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((2, 2, 2), dtype=np.float32), np.ones((2, 2), dtype=bool)

    def fake_generator(*_: object) -> torch.Tensor:
        return torch.full((10, 2, 2), 0.75)

    monkeypatch.setattr(downstream, "read_sar_raw_valid", generation_sar_reader)
    monkeypatch.setattr(downstream, "generate_physical_optical", fake_generator)
    downstream.cache_rank_samples(
        plan,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        model_loader=fake_loader,
    )
    with pytest.raises(RuntimeError, match="finalize_cache"):
        downstream.materialize_probe_cache(plan, dev_tiles=("tile-dev",), chunk_size=2)
    downstream.finalize_cache(plan)

    materialized_reads = {"sar": 0, "optical": 0, "scl": 0}

    def materialized_sar_reader(*_: object) -> tuple[np.ndarray, np.ndarray]:
        materialized_reads["sar"] += 1
        return np.full((2, 2, 2), -15.0, dtype=np.float32), np.ones((2, 2), dtype=bool)

    def real_optical_reader(*_: object) -> np.ndarray:
        materialized_reads["optical"] += 1
        return np.full((10, 2, 2), 0.25, dtype=np.float32)

    def label_reader(*_: object) -> np.ndarray:
        materialized_reads["scl"] += 1
        return np.array([[1, 0], [0, -1]], dtype=np.int64)

    monkeypatch.setattr(downstream, "read_sar_raw_valid", materialized_sar_reader)
    monkeypatch.setattr(downstream, "read_real_optical", real_optical_reader)
    monkeypatch.setattr(downstream, "read_scl_proxy_label", label_reader)
    manifest = downstream.materialize_probe_cache(plan, dev_tiles=("tile-dev",), chunk_size=2)

    observed_splits: list[str] = []
    for entry in manifest["entries"]:
        payload = torch.load(entry["path"], map_location="cpu", weights_only=False)
        assert payload["sar"].dtype == torch.float16
        assert payload["real_optical"].dtype == torch.float16
        assert payload["synthetic_optical"].dtype == torch.float16
        assert payload["sar_valid"].dtype == torch.bool
        assert payload["label"].dtype == torch.int8
        cache = ProbeCache.from_mapping(payload)
        assert len(cache) >= 1
        observed_splits.extend(cache.split)
    assert sorted(observed_splits) == ["dev", "test", "train"]
    assert materialized_reads == {"sar": 3, "optical": 3, "scl": 3}

    reused = downstream.materialize_probe_cache(plan, dev_tiles=("tile-dev",), chunk_size=2)
    assert reused == manifest
    assert materialized_reads == {"sar": 3, "optical": 3, "scl": 3}
