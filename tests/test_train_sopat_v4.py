from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def runner_module():
    path = Path(__file__).parents[1] / "scripts" / "train_sopat_v4.py"
    spec = importlib.util.spec_from_file_location("train_sopat_v4_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _minimal_config(tmp_path: Path, *, cache_root: str | None = None) -> dict[str, object]:
    data: dict[str, object] = {
        "train_split": "train",
        "validation_split": "validation_temporal",
        "crop_size": 8,
        "maximum_observations": 2,
    }
    if cache_root is not None:
        data.update(
            {
                "chunk_cache_root": cache_root,
                "sopat_v4_index": "route.jsonl",
            }
        )
    return {
        "data": data,
        "model": {},
        "training": {
            "precision": "bfloat16",
            "validate_every": 1,
            "full_validate_every": 1,
            "early_stop_full_validations": 1,
            "stages": {"factorizer": {"steps": 1}, "physical": {"steps": 1}},
        },
        "validation": {"selection": {}},
    }


def test_chunk_route_requires_v4_role_index_before_raw_fallback(tmp_path: Path, runner_module) -> None:
    config = _minimal_config(tmp_path, cache_root="chunks")

    with pytest.raises(ValueError, match="sopat_v4_index"):
        runner_module._prepare_data_on_rank_zero(
            {**config, "data": {**config["data"], "sopat_v4_index": None}},
            output=tmp_path / "output",
            config_base=tmp_path,
        )


def test_full_chunk_configuration_loads_with_eight_rank_contract(runner_module) -> None:
    config = runner_module._load_config(
        Path(__file__).parents[1] / "configs" / "sopat_v4_full_chunk.yaml"
    )

    runner_module._validate_run_config(config, world_size=8)
    model = runner_module._model_config(config)
    physical = runner_module._train_config(config, stage="physical")

    assert model.architecture == "sopat_v4"
    assert physical.source_shuffle_weight == 0.25
    assert physical.structural_pool_kernel == 5


def test_chunk_preflight_missing_cache_is_clear_and_fail_closed(tmp_path: Path, runner_module) -> None:
    config = _minimal_config(tmp_path, cache_root="chunks")
    index_path = tmp_path / "route.jsonl"
    index_path.write_text("not a v4 index\n", encoding="utf-8")

    # The role index is checked before any dataset construction.  Invalid
    # cache publication must never fall back to `from_raster`.
    with pytest.raises(Exception, match="SOPAT|index|header"):
        runner_module._prepare_data_on_rank_zero(
            config,
            output=tmp_path / "output",
            config_base=tmp_path,
        )


def test_cache_artifacts_delegates_complete_preflight_and_binds_direction_indexes(
    tmp_path: Path, runner_module, monkeypatch
) -> None:
    cache_root = tmp_path / "chunks"
    cache_root.mkdir()
    train_sar = cache_root / "indexes" / "sar-train.jsonl"
    validation_sar = cache_root / "indexes" / "sar-validation.jsonl"
    train_optical = cache_root / "indexes" / "optical-train.jsonl"
    validation_optical = cache_root / "indexes" / "optical-validation.jsonl"
    for path in (train_sar, validation_sar, train_optical, validation_optical):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    entries = [
        {"direction": "sar_to_optical", "split": "train", "relative_path": "indexes/sar-train.jsonl"},
        {
            "direction": "sar_to_optical",
            "split": "validation_temporal",
            "relative_path": "indexes/sar-validation.jsonl",
        },
        {"direction": "optical_to_sar", "split": "train", "relative_path": "indexes/optical-train.jsonl"},
        {
            "direction": "optical_to_sar",
            "split": "validation_temporal",
            "relative_path": "indexes/optical-validation.jsonl",
        },
    ]
    for entry in entries:
        entry["sha256"] = runner_module.file_sha256(cache_root / entry["relative_path"])
    (cache_root / "cache_index.json").write_text(
        json.dumps({"indexes": entries}), encoding="utf-8"
    )
    calls: list[tuple[Path, Path, bool]] = []

    def preflight(root, index, *, verify_chunks):
        calls.append((Path(root), Path(index), verify_chunks))
        return SimpleNamespace(
            cache_index_sha256="a" * 64,
            cache_plan_sha256="b" * 64,
            index_content_sha256="c" * 64,
        )

    monkeypatch.setattr(runner_module, "preflight_sopat_v4_chunk_cache", preflight)
    artifacts = runner_module._cache_index_artifacts(
        cache_root,
        tmp_path / "route.jsonl",
        train_split="train",
        validation_split="validation_temporal",
        verify_chunks=True,
    )

    assert calls == [(cache_root, tmp_path / "route.jsonl", True)]
    assert artifacts["sar_to_optical"]["cache_train_index"] == entries[0]["sha256"]
    assert artifacts["optical_to_sar"]["cache_validation_index"] == entries[3]["sha256"]


def test_datasets_chunk_route_uses_cache_adapter_not_raster(tmp_path: Path, runner_module, monkeypatch) -> None:
    config = _minimal_config(tmp_path, cache_root="chunks")
    prepared = runner_module._PreparedData(
        route="chunk_cache",
        index_path=tmp_path / "route.jsonl",
        cache_root=tmp_path / "chunks",
        manifest_path=None,
        train_split="train",
        validation_split="validation_temporal",
        protocol_hashes={
            "sar_to_optical": {"index": "a" * 64},
            "optical_to_sar": {"index": "b" * 64},
        },
    )
    index = object()
    calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(runner_module, "load_sopat_v4_index", lambda _path: index)
    monkeypatch.setattr(runner_module, "_validate_v4_index", lambda *_args, **_kwargs: None)

    def adapter(_root, _index, *, direction, split, window_mode, **_kwargs):
        calls.append((direction, split, window_mode))
        return SimpleNamespace(direction=direction, split=split)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("raw RasterDataset route must not run for chunk cache")

    monkeypatch.setattr(runner_module, "sopat_chunk_dataset_from_cache", adapter)
    monkeypatch.setattr(runner_module.SOPATDirectionDataset, "from_raster", forbidden)
    train, validation = runner_module._datasets(config, prepared, seed=3, stage="physical")

    assert set(train) == {"sar_to_optical", "optical_to_sar"}
    assert set(validation) == {"sar_to_optical", "optical_to_sar"}
    assert calls == [
        ("sar_to_optical", "train", "all"),
        ("sar_to_optical", "validation_temporal", "center"),
        ("optical_to_sar", "train", "all"),
        ("optical_to_sar", "validation_temporal", "center"),
    ]


def test_datasets_raw_route_uses_random_train_and_center_validation(
    tmp_path: Path, runner_module, monkeypatch
) -> None:
    config = _minimal_config(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    prepared = runner_module._PreparedData(
        route="raw_v4_index",
        index_path=tmp_path / "route.jsonl",
        cache_root=None,
        manifest_path=manifest,
        train_split="train",
        validation_split="validation_temporal",
        protocol_hashes={
            "sar_to_optical": {"index": "a" * 64},
            "optical_to_sar": {"index": "b" * 64},
        },
    )
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(runner_module, "load_sopat_v4_index", lambda _path: object())
    monkeypatch.setattr(runner_module, "_validate_v4_index", lambda *_args, **_kwargs: None)

    def raster(_index, _manifest, *, direction, split, crop_mode, **_kwargs):
        calls.append((direction, split, crop_mode))
        return SimpleNamespace(direction=direction, split=split)

    monkeypatch.setattr(runner_module.SOPATDirectionDataset, "from_raster", raster)
    runner_module._datasets(config, prepared, seed=3, stage="physical")

    assert calls == [
        ("sar_to_optical", "train", "random_valid"),
        ("sar_to_optical", "validation_temporal", "center"),
        ("optical_to_sar", "train", "random_valid"),
        ("optical_to_sar", "validation_temporal", "center"),
    ]


def test_activation_checkpointing_and_log_validation(tmp_path: Path, runner_module) -> None:
    config = _minimal_config(tmp_path)
    config["training"]["activation_checkpointing"] = "yes"  # type: ignore[index]
    with pytest.raises(TypeError, match="activation_checkpointing"):
        runner_module._validate_run_config(config, world_size=1)

    config = _minimal_config(tmp_path)
    config["training"]["log_every"] = 0  # type: ignore[index]
    with pytest.raises(ValueError, match="log_every"):
        runner_module._validate_run_config(config, world_size=1)


def test_validation_batch_size_is_independent_and_checked(tmp_path: Path, runner_module) -> None:
    config = _minimal_config(tmp_path)
    config["training"]["batch_size"] = 1  # type: ignore[index]
    config["validation"]["batch_size"] = 4  # type: ignore[index]
    runner_module._validate_run_config(config, world_size=1)

    config["validation"]["batch_size"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="validation.batch_size.*at least 2"):
        runner_module._validate_run_config(config, world_size=1)


@pytest.mark.parametrize(
    ("sample_count", "batch_size", "expected"),
    [
        (2, 4, [[0, 1]]),
        (5, 4, [[0, 1, 2], [3, 4]]),
        (9, 4, [[0, 1, 2, 3], [4, 5, 6], [7, 8]]),
    ],
)
def test_no_singleton_validation_batch_sampler_covers_every_sample_once(
    runner_module, sample_count: int, batch_size: int, expected: list[list[int]]
) -> None:
    sampler = runner_module.NoSingletonBatchSampler(sample_count, batch_size)
    batches = list(sampler)

    assert batches == expected
    flattened = [sample for batch in batches for sample in batch]
    assert flattened == list(range(sample_count))
    assert len(flattened) == len(set(flattened))
    assert all(2 <= len(batch) <= batch_size for batch in batches)


def test_no_singleton_validation_batch_sampler_fails_closed_for_impossible_sizes(
    runner_module,
) -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        runner_module.NoSingletonBatchSampler(1, 4)
    with pytest.raises(ValueError, match="odd sample count.*batch_size=2"):
        runner_module.NoSingletonBatchSampler(3, 2)


def test_validation_loaders_accept_explicit_batch_samplers(runner_module, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _LengthOnlyDataset:
        def __len__(self) -> int:
            return 5

    class _Loader:
        def __len__(self) -> int:
            return 2

    def fake_loader(*_args: object, **kwargs: object) -> _Loader:
        calls.append(dict(kwargs))
        return _Loader()

    monkeypatch.setattr(runner_module, "DataLoader", fake_loader)
    samplers = {
        direction: runner_module.NoSingletonBatchSampler(5, 4)
        for direction in ("sar_to_optical", "optical_to_sar")
    }
    loaders = runner_module._loaders(
        {direction: _LengthOnlyDataset() for direction in samplers},
        batch_size=4,
        num_workers=0,
        device=runner_module.torch.device("cpu"),
        rank=0,
        world_size=1,
        training=False,
        seed=7,
        batch_samplers=samplers,
    )

    assert set(loaders) == set(samplers)
    assert len(calls) == 2
    assert all(call["batch_sampler"] is samplers[direction] for call, direction in zip(calls, samplers))
    assert all("batch_size" not in call and "drop_last" not in call for call in calls)
