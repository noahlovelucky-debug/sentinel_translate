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
    assert physical.counterfactual_confidence_margin == pytest.approx(0.10)


def test_calibration_cli_exposes_ema_and_learning_rate_overrides(runner_module) -> None:
    parsed = runner_module._parser().parse_args(
        [
            "--config",
            "config.yaml",
            "--stage",
            "physical",
            "--output",
            "output",
            "--init-checkpoint",
            "quality.pt",
            "--init-use-ema",
            "--learning-rate",
            "1e-5",
            "--encoder-learning-rate",
            "2e-7",
            "--trainable-scope",
            "confidence_only",
        ]
    )

    assert parsed.init_use_ema is True
    assert parsed.learning_rate == pytest.approx(1.0e-5)
    assert parsed.encoder_learning_rate == pytest.approx(2.0e-7)
    assert parsed.trainable_scope == "confidence_only"


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
    monkeypatch.setattr(
        runner_module,
        "build_global_cross_tile_hard_negative_plan",
        lambda *_args, direction, split: f"plan:{direction}:{split}",
    )

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


def test_physical_datasets_build_and_attach_global_plans(tmp_path: Path, runner_module, monkeypatch) -> None:
    config = _minimal_config(tmp_path, cache_root="chunks")
    prepared = runner_module._PreparedData(
        route="chunk_cache",
        index_path=tmp_path / "route.jsonl",
        cache_root=tmp_path / "chunks",
        manifest_path=None,
        train_split="train",
        validation_split="validation_temporal",
        protocol_hashes={direction: {} for direction in ("sar_to_optical", "optical_to_sar")},
    )
    class _Index:
        def select(self, **_kwargs):
            return ()

    index = _Index()
    calls: list[tuple[str, str, object, bool]] = []
    plans: list[tuple[str, str]] = []

    def plan_builder(_index, *, direction, split):
        plans.append((direction, split))
        return SimpleNamespace(
            mapping_metadata={
                "plan_hash": f"{direction}:{split}",
                "coverage": 1.0,
                "cross_tile_coverage": 1.0,
                "tier_counts": {"same_task_exact_n": 1},
            }
        )

    def adapter(_root, _index, *, direction, split, hard_negative_plan, include_cf, **_kwargs):
        calls.append((direction, split, hard_negative_plan, include_cf))
        return SimpleNamespace(direction=direction, split=split, hard_negative_plan=hard_negative_plan)

    monkeypatch.setattr(runner_module, "load_sopat_v4_index", lambda _path: index)
    monkeypatch.setattr(runner_module, "_validate_v4_index", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner_module, "build_global_cross_tile_hard_negative_plan", plan_builder)
    monkeypatch.setattr(runner_module, "sopat_chunk_dataset_from_cache", adapter)

    train, validation = runner_module._datasets(config, prepared, seed=3, stage="physical")

    assert plans == [
        ("sar_to_optical", "train"),
        ("sar_to_optical", "validation_temporal"),
        ("optical_to_sar", "train"),
        ("optical_to_sar", "validation_temporal"),
    ]
    assert all(include_cf is True and plan is not None for *_rest, plan, include_cf in calls)
    state = runner_module._global_counterfactual_data_state(
        train_datasets=train,
        validation_datasets=validation,
        train_split="train",
        validation_split="validation_temporal",
    )
    counterfactual = state["global_counterfactual"]
    assert counterfactual["train"]["plans"]["sar_to_optical"]["plan_hash"] == "sar_to_optical:train"
    assert (
        counterfactual["validation"]["plans"]["optical_to_sar"]["plan_hash"]
        == "optical_to_sar:validation_temporal"
    )
    missing_plan = {
        direction: SimpleNamespace(hard_negative_plan=None)
        for direction in ("sar_to_optical", "optical_to_sar")
    }
    with pytest.raises(TypeError, match="requires global counterfactual plan metadata"):
        runner_module._global_counterfactual_data_state(
            train_datasets=missing_plan,
            validation_datasets=validation,
            train_split="train",
            validation_split="validation_temporal",
        )


def test_factorizer_datasets_do_not_build_global_plans(tmp_path: Path, runner_module, monkeypatch) -> None:
    config = _minimal_config(tmp_path, cache_root="chunks")
    prepared = runner_module._PreparedData(
        route="chunk_cache",
        index_path=tmp_path / "route.jsonl",
        cache_root=tmp_path / "chunks",
        manifest_path=None,
        train_split="train",
        validation_split="validation_temporal",
        protocol_hashes={direction: {} for direction in ("sar_to_optical", "optical_to_sar")},
    )
    monkeypatch.setattr(runner_module, "load_sopat_v4_index", lambda _path: SimpleNamespace())
    monkeypatch.setattr(runner_module, "_validate_v4_index", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner_module,
        "build_global_cross_tile_hard_negative_plan",
        lambda *_args, **_kwargs: pytest.fail("factorizer must not build a donor plan"),
    )
    seen: list[tuple[object, bool]] = []

    def adapter(*_args, hard_negative_plan, include_cf, **_kwargs):
        seen.append((hard_negative_plan, include_cf))
        return SimpleNamespace(direction="sar_to_optical", split="train")

    monkeypatch.setattr(runner_module, "sopat_chunk_dataset_from_cache", adapter)
    runner_module._datasets(config, prepared, seed=3, stage="factorizer")

    assert seen and all(plan is None and include_cf is False for plan, include_cf in seen)


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
    index = object()
    monkeypatch.setattr(runner_module, "load_sopat_v4_index", lambda _path: index)
    monkeypatch.setattr(runner_module, "_validate_v4_index", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner_module,
        "build_global_cross_tile_hard_negative_plan",
        lambda *_args, direction, split: f"plan:{direction}:{split}",
    )

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
    runner_module._validate_run_config(config, world_size=1)

    config["validation"]["batch_size"] = 0  # type: ignore[index]
    with pytest.raises(ValueError, match="validation.batch_size.*positive"):
        runner_module._validate_run_config(config, world_size=1)


def test_runner_pins_global_counterfactual_selection_policy_v3(tmp_path: Path, runner_module) -> None:
    config = _minimal_config(tmp_path)
    config["validation"]["selection"] = {"policy_version": "sopat_v4_quality_gate_v2"}  # type: ignore[index]

    selection = runner_module._selection_config(config, phase="feasibility")

    assert selection.policy_version == "sopat_v4_quality_gate_v3"


def test_validation_loader_uses_normal_batch_size_for_global_plan(runner_module, monkeypatch) -> None:
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
    loaders = runner_module._loaders(
        {direction: _LengthOnlyDataset() for direction in ("sar_to_optical", "optical_to_sar")},
        batch_size=1,
        num_workers=0,
        device=runner_module.torch.device("cpu"),
        rank=0,
        world_size=1,
        training=False,
        seed=7,
    )

    assert set(loaders) == {"sar_to_optical", "optical_to_sar"}
    assert len(calls) == 2
    assert all(call["batch_size"] == 1 for call in calls)
    assert all(call["shuffle"] is False for call in calls)
    assert all("batch_sampler" not in call for call in calls)
