"""Train one Sparse Paired-Anchor Transport V2 stage, with optional DDP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from sentinel_v3.paired_temporal_data import (
    PairedTemporalIndex,
    PairedTemporalRasterDataset,
    assert_paired_temporal_causality,
    build_paired_temporal_index,
    collate_paired_temporal,
    load_pair_records,
    load_paired_temporal_index,
    source_modality,
    target_modality,
    write_paired_temporal_index,
)
from sentinel_v3.paired_temporal_training import (
    PAIRED_TEMPORAL_STAGES,
    PairedTemporalTrainConfig,
    PairedTemporalTrainingModule,
    apply_observation_dropout,
    evaluate_paired_temporal_batches,
    load_paired_temporal_checkpoint,
    paired_tensor_batch,
    save_paired_temporal_checkpoint,
    set_paired_temporal_stage,
    validation_release_search,
)
from sentinel_v3.paired_temporal_v2 import PairedTemporalConfig, SparsePairedAnchorTransport
from sentinel_v3.schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER


class _PreparedIndexes:
    """The validated index files and checkpoint protocol for one direction."""

    __slots__ = ("explicit", "protocol_hash", "train_path", "validation_path")

    def __init__(
        self,
        *,
        train_path: Path,
        validation_path: Path,
        protocol_hash: str,
        explicit: bool,
    ) -> None:
        self.train_path = train_path
        self.validation_path = validation_path
        self.protocol_hash = protocol_hash
        self.explicit = explicit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--direction", choices=("sar_to_optical", "optical_to_sar"), required=True)
    parser.add_argument("--stage", choices=PAIRED_TEMPORAL_STAGES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Rank zero performs fixed-seed full validation and release search while
        # the other ranks wait. Keep long research validation from tripping the
        # backend's much shorter default collective timeout.
        dist.init_process_group(backend=backend, timeout=timedelta(hours=6))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def _load_config(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("paired temporal configuration must be a mapping")
    return values


def _protocol_hash(
    config: dict[str, Any],
    direction: str,
    artifact_sha256: Mapping[str, str] | None = None,
) -> str:
    """Hash the training contract, including explicit local-cache artifacts.

    The canonical auto-built route retains its historic config-only identity.
    Explicit-index runs additionally bind the local manifest and both serialized
    index files, so a cache update cannot silently resume an incompatible
    checkpoint.
    """

    payload = {"config": config, "direction": direction}
    if artifact_sha256 is not None:
        payload["artifact_sha256"] = {
            str(name): str(value) for name, value in sorted(artifact_sha256.items())
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seed(seed: int, rank: int) -> None:
    resolved = seed + rank * 100003
    random.seed(resolved)
    np.random.seed(resolved % (2**32))
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)


def _index_paths(output: Path, direction: str, protocol_hash: str) -> tuple[Path, Path]:
    root = output / "indices"
    suffix = protocol_hash[:12]
    return (
        root / f"train_{direction}_{suffix}.jsonl",
        root / f"validation_{direction}_{suffix}.jsonl",
    )


def _prepare_indexes(
    config: dict[str, Any],
    direction: str,
    output: Path,
    rank: int,
    world_size: int,
    device: torch.device | None = None,
) -> _PreparedIndexes:
    """Prepare auto-built indexes or validate an explicit local-cache pair.

    Explicit ``data.train_index``/``data.validation_index`` values are a hard
    routing boundary: rank zero loads and validates the supplied files, but
    never calls the index builder.  Its result or failure is broadcast before
    any worker tries to construct a dataset, preventing one failed rank from
    leaving the remaining DDP workers waiting at a later collective.
    """

    prepared: _PreparedIndexes | None = None
    preparation_error: Exception | None = None
    if rank == 0:
        try:
            prepared = _prepare_indexes_on_rank_zero(config, direction, output)
        except Exception as error:  # noqa: BLE001 - propagate rank-0 build failure to all ranks
            preparation_error = error
    return _broadcast_prepared_indexes(
        prepared,
        preparation_error,
        rank=rank,
        world_size=world_size,
        device=device,
    )


def _prepare_indexes_on_rank_zero(
    config: dict[str, Any], direction: str, output: Path
) -> _PreparedIndexes:
    data = _data_mapping(config)
    explicit_paths = _explicit_index_paths(data, direction)
    if explicit_paths is not None:
        train_path, validation_path = explicit_paths
        manifest_path = _configured_path(data, "manifest")
        records = _load_explicit_manifest(manifest_path)
        record_map = {record.pair_id: record for record in records}
        if len(record_map) != len(records):
            raise ValueError(f"explicit paired temporal manifest has duplicate pair_id values: {manifest_path}")
        _validate_explicit_index(
            train_path,
            label="train",
            direction=direction,
            expected_split=_required_data_string(data, "train_split"),
            expected_count=_optional_data_count(data, "max_train_samples"),
            data=data,
            records=records,
            record_map=record_map,
            manifest_root=manifest_path.parent,
        )
        _validate_explicit_index(
            validation_path,
            label="validation",
            direction=direction,
            expected_split=_required_data_string(data, "validation_split"),
            expected_count=_optional_data_count(data, "max_validation_samples"),
            data=data,
            records=records,
            record_map=record_map,
            manifest_root=manifest_path.parent,
        )
        artifact_sha256 = {
            "manifest": _file_sha256(manifest_path),
            "train_index": _file_sha256(train_path),
            "validation_index": _file_sha256(validation_path),
        }
        return _PreparedIndexes(
            train_path=train_path,
            validation_path=validation_path,
            protocol_hash=_protocol_hash(config, direction, artifact_sha256),
            explicit=True,
        )

    protocol_hash = _protocol_hash(config, direction)
    train_path, validation_path = _index_paths(output, direction, protocol_hash)
    if not train_path.is_file() or not validation_path.is_file():
        records = load_pair_records(_configured_path(data, "manifest"))
        common = {
            "direction": direction,
            "min_observations": _required_data_int(data, "minimum_observations"),
            "max_observations": _required_data_int(data, "maximum_observations"),
            "horizon_days": _required_data_int(data, "horizon_days"),
            "anchor_max_delta_days": _required_data_int(data, "anchor_pair_max_delta_days"),
            "max_anchors_per_query": _optional_data_count(
                data, "maximum_anchors_per_query"
            )
            or 1,
            "translation_max_delta_days": _required_data_int(
                data, "translation_max_delta_days"
            ),
            "orbit": _required_data_string(data, "orbit"),
            "task_modes": _configured_task_modes(data),
        }
        train_limit = _optional_data_count(data, "max_train_samples")
        validation_limit = _optional_data_count(data, "max_validation_samples")
        train = build_paired_temporal_index(
            records,
            split=_required_data_string(data, "train_split"),
            max_samples=train_limit,
            **common,
        )
        validation = build_paired_temporal_index(
            records,
            split=_required_data_string(data, "validation_split"),
            max_samples=validation_limit,
            **common,
        )
        if not train.samples or not validation.samples:
            raise RuntimeError("paired temporal protocol produced an empty train or validation index")
        write_paired_temporal_index(train_path, train)
        write_paired_temporal_index(validation_path, validation)
    if not train_path.is_file() or not validation_path.is_file():
        raise RuntimeError("paired temporal index creation did not complete")
    return _PreparedIndexes(
        train_path=train_path,
        validation_path=validation_path,
        protocol_hash=protocol_hash,
        explicit=False,
    )


def _broadcast_prepared_indexes(
    prepared: _PreparedIndexes | None,
    preparation_error: Exception | None,
    *,
    rank: int,
    world_size: int,
    device: torch.device | None,
) -> _PreparedIndexes:
    """Share rank-zero preparation status before further DDP work begins."""

    if world_size <= 1:
        if preparation_error is not None:
            raise preparation_error
        if prepared is None:
            raise RuntimeError("rank 0 paired temporal index preparation produced no result")
        return prepared

    payload: list[object] = [None]
    if rank == 0:
        if preparation_error is not None:
            payload[0] = {
                "ok": False,
                "error": f"{type(preparation_error).__name__}: {preparation_error}",
            }
        elif prepared is not None:
            payload[0] = {
                "ok": True,
                "train_path": str(prepared.train_path),
                "validation_path": str(prepared.validation_path),
                "protocol_hash": prepared.protocol_hash,
                "explicit": prepared.explicit,
            }
        else:
            payload[0] = {"ok": False, "error": "rank 0 produced no prepared index result"}
    dist.broadcast_object_list(payload, src=0, device=_collective_device(device))
    received = payload[0]
    if not isinstance(received, dict) or not isinstance(received.get("ok"), bool):
        raise TypeError("invalid paired temporal index preparation broadcast")
    if received["ok"] is not True:
        if preparation_error is not None:
            raise preparation_error
        raise RuntimeError(
            "rank 0 paired temporal index preparation failed: "
            f"{received.get('error', 'unknown error')}"
        )
    try:
        train_path = Path(str(received["train_path"]))
        validation_path = Path(str(received["validation_path"]))
        protocol_hash = str(received["protocol_hash"])
        explicit = received["explicit"]
    except KeyError as error:
        raise RuntimeError("incomplete paired temporal index preparation broadcast") from error
    if len(protocol_hash) != 64 or not isinstance(explicit, bool):
        raise RuntimeError("invalid paired temporal index preparation broadcast fields")
    return _PreparedIndexes(
        train_path=train_path,
        validation_path=validation_path,
        protocol_hash=protocol_hash,
        explicit=explicit,
    )


def _data_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise TypeError("paired temporal configuration requires a data mapping")
    return data


def _explicit_index_paths(
    data: Mapping[str, Any], direction: str
) -> tuple[Path, Path] | None:
    """Resolve string-template or direction-map explicit index configuration."""

    has_train = data.get("train_index") is not None
    has_validation = data.get("validation_index") is not None
    if not has_train and not has_validation:
        return None
    if has_train != has_validation:
        raise ValueError(
            "data.train_index and data.validation_index must be supplied together for explicit routing"
        )
    return (
        _configured_direction_path(data["train_index"], "train_index", direction),
        _configured_direction_path(data["validation_index"], "validation_index", direction),
    )


def _configured_direction_path(value: object, name: str, direction: str) -> Path:
    selected = value
    if isinstance(value, Mapping):
        if direction not in value:
            raise ValueError(f"data.{name} has no entry for direction={direction}")
        selected = value[direction]
    if not isinstance(selected, (str, Path)):
        raise TypeError(
            f"data.{name} must be a path string/template or a direction-to-path mapping"
        )
    rendered = os.fspath(selected)
    if "{" in rendered or "}" in rendered:
        remaining = rendered.replace("{direction}", "")
        if rendered.count("{direction}") != 1 or "{" in remaining or "}" in remaining:
            raise ValueError(f"data.{name} only supports the {{direction}} path template")
        rendered = rendered.replace("{direction}", direction)
    return _lexical_path(rendered)


def _configured_path(data: Mapping[str, Any], name: str) -> Path:
    value = data.get(name)
    if not isinstance(value, (str, Path)):
        raise TypeError(f"data.{name} must be a path string")
    return _lexical_path(value)


def _lexical_path(value: str | Path) -> Path:
    rendered = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    return Path(os.path.abspath(os.path.normpath(rendered)))


def _required_data_string(data: Mapping[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"data.{name} must be a non-empty string")
    return value


def _required_data_int(data: Mapping[str, Any], name: str) -> int:
    value = data.get(name)
    if value is None or isinstance(value, bool):
        raise TypeError(f"data.{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"data.{name} must be an integer") from error


def _optional_data_count(data: Mapping[str, Any], name: str) -> int | None:
    value = data.get(name)
    if value is None:
        return None
    result = _required_data_int(data, name)
    if result <= 0:
        raise ValueError(f"data.{name} must be positive when supplied")
    return result


def _configured_task_modes(data: Mapping[str, Any]) -> tuple[str, ...]:
    values = data.get("task_modes", ("translation", "forecast"))
    if not isinstance(values, (list, tuple)):
        raise TypeError("data.task_modes must be a sequence")
    result = tuple(str(value) for value in values)
    if not result:
        raise ValueError("data.task_modes must not be empty")
    return result


def _load_explicit_manifest(manifest_path: Path) -> list[Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"explicit paired temporal manifest is missing: {manifest_path}")
    return load_pair_records(manifest_path)


def _validate_explicit_index(
    index_path: Path,
    *,
    label: str,
    direction: str,
    expected_split: str,
    expected_count: int | None,
    data: Mapping[str, Any],
    records: list[Any],
    record_map: Mapping[str, Any],
    manifest_root: Path,
) -> PairedTemporalIndex:
    if not index_path.is_file():
        raise FileNotFoundError(f"explicit {label} paired temporal index is missing: {index_path}")
    index = load_paired_temporal_index(index_path, direction=direction)  # type: ignore[arg-type]
    if not index.samples:
        raise ValueError(f"explicit {label} paired temporal index is empty: {index_path}")
    if index.config.direction != direction:
        raise ValueError(
            f"explicit {label} paired temporal index direction is {index.config.direction}, expected {direction}"
        )
    _validate_explicit_index_constraints(index, data, expected_split, label)
    if expected_count is not None and len(index) != expected_count:
        raise ValueError(
            f"explicit {label} paired temporal index has {len(index)} samples, expected {expected_count}"
        )
    if expected_count is not None and index.config.max_samples != expected_count:
        raise ValueError(
            f"explicit {label} paired temporal index max_samples={index.config.max_samples!r}, "
            f"expected {expected_count}"
        )
    assert_paired_temporal_causality(index, records, asset_root=manifest_root)
    _assert_explicit_asset_paths(index, record_map, manifest_root)
    return index


def _validate_explicit_index_constraints(
    index: PairedTemporalIndex,
    data: Mapping[str, Any],
    expected_split: str,
    label: str,
) -> None:
    expected = {
        "min_observations": _required_data_int(data, "minimum_observations"),
        "max_observations": _required_data_int(data, "maximum_observations"),
        "horizon_days": _required_data_int(data, "horizon_days"),
        "anchor_max_delta_days": _required_data_int(data, "anchor_pair_max_delta_days"),
        "max_anchors_per_query": _optional_data_count(data, "maximum_anchors_per_query")
        or 1,
        "translation_max_delta_days": _required_data_int(data, "translation_max_delta_days"),
        "orbit": _required_data_string(data, "orbit"),
        "split": expected_split,
    }
    for name, expected_value in expected.items():
        if getattr(index.config, name) != expected_value:
            raise ValueError(
                f"explicit {label} paired temporal index {name}={getattr(index.config, name)!r}, "
                f"expected {expected_value!r}"
            )
    expected_modes = _configured_task_modes(data)
    if set(index.config.task_modes) != set(expected_modes):
        raise ValueError(
            f"explicit {label} paired temporal index task_modes={index.config.task_modes!r}, "
            f"expected {expected_modes!r}"
        )


def _assert_explicit_asset_paths(
    index: PairedTemporalIndex,
    records: Mapping[str, Any],
    manifest_root: Path,
) -> None:
    required_paths: set[Path] = set()
    for sample in index:
        query = records[sample.query_pair_id]
        anchor = records[sample.anchor_pair_id]
        observations = tuple(records[pair_id] for pair_id in sample.observation_pair_ids)
        required_paths.update(
            _modality_asset_paths(query, target_modality(sample.direction), manifest_root)
        )
        required_paths.update(
            _modality_asset_paths(anchor, source_modality(sample.direction), manifest_root)
        )
        required_paths.update(
            _modality_asset_paths(anchor, target_modality(sample.direction), manifest_root)
        )
        for observation in observations:
            required_paths.update(
                _modality_asset_paths(observation, source_modality(sample.direction), manifest_root)
            )
    missing = [path for path in sorted(required_paths, key=os.fspath) if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f"explicit paired temporal indexes reference {len(missing)} missing local asset file(s): "
            f"{preview}"
        )


def _modality_asset_paths(record: Any, modality: str, manifest_root: Path) -> tuple[Path, ...]:
    if modality == "optical":
        values = tuple(record.s2[channel] for channel in S2_CHANNEL_ORDER) + (record.scl,)
    elif modality == "sar":
        values = tuple(record.sar[channel] for channel in SAR_CHANNEL_ORDER)
    else:
        raise ValueError(f"unsupported paired temporal modality: {modality}")
    return tuple(_manifest_asset_path(manifest_root, value) for value in values)


def _manifest_asset_path(manifest_root: Path, value: str) -> Path:
    expanded = os.path.expanduser(value)
    path = expanded if os.path.isabs(expanded) else os.path.join(os.fspath(manifest_root), expanded)
    return Path(os.path.abspath(os.path.normpath(path)))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _datasets(
    config: dict[str, Any],
    train_path: Path,
    validation_path: Path,
    seed: int,
    stage: str,
) -> tuple[PairedTemporalRasterDataset, PairedTemporalRasterDataset]:
    data = config["data"]
    registration_audit = bool(data.get("registration_audit", True)) and stage == "detail"
    options = {
        "crop_size": int(data["crop_size"]),
        "minimum_valid_fraction": float(data["minimum_valid_fraction"]),
        "max_observations": int(data["maximum_observations"]),
        "registration_audit": registration_audit,
        "maximum_registration_shift_px": float(
            data.get("maximum_registration_shift_px", 0.5)
        ),
    }
    return (
        PairedTemporalRasterDataset(
            data["manifest"], load_paired_temporal_index(train_path), seed=seed, **options
        ),
        PairedTemporalRasterDataset(
            data["manifest"],
            load_paired_temporal_index(validation_path),
            seed=seed,
            **options,
        ),
    )


def _model_config(config: dict[str, Any]) -> PairedTemporalConfig:
    values = config["model"]
    return PairedTemporalConfig(
        width=int(values["width"]),
        latent_channels=int(values["latent_channels"]),
        attention_heads=int(values["attention_heads"]),
        maximum_horizon_days=int(config["data"]["horizon_days"]),
        translation_max_delta_days=int(config["data"]["translation_max_delta_days"]),
        flow_steps=int(values["flow_steps"]),
        deterministic_detail_limit=float(values["deterministic_detail_limit"]),
        visual_residual_limit=float(values["visual_residual_limit"]),
        texture_block_size=int(values["texture_block_size"]),
        architecture=str(values["architecture"]),
    )


def _train_config(
    config: dict[str, Any], direction: str, stage: str
) -> PairedTemporalTrainConfig:
    training = config["training"]
    stage_values = training["stages"][stage]
    validation = config["validation"]
    count_probabilities = training.get("observation_count_probabilities", {})
    if not isinstance(count_probabilities, dict):
        raise TypeError("training.observation_count_probabilities must be a mapping")
    return PairedTemporalTrainConfig(
        direction=direction,
        stage=stage,
        learning_rate=float(stage_values["learning_rate"]),
        observation_dropout=float(training["observation_dropout"]),
        query_observation_dropout=float(training["query_observation_dropout"]),
        one_frame_probability=float(count_probabilities.get("one", 1.0 / 3.0)),
        two_to_three_frame_probability=float(
            count_probabilities.get("two_to_three", 1.0 / 3.0)
        ),
        four_plus_frame_probability=float(count_probabilities.get("four_plus", 1.0 / 3.0)),
        translation_max_delta_days=int(config["data"]["translation_max_delta_days"]),
        visual_rmse_budget=float(validation["visual_rmse_over_physical_maximum"]),
        minimum_physical_anchor_improvement_percent=float(
            validation["minimum_physical_anchor_improvement_percent"]
        ),
        minimum_source_evidence_improvement_percent=float(
            validation["minimum_source_evidence_improvement_percent"]
        ),
        pre_projection_violation_maximum=float(
            validation["pre_projection_violation_maximum"]
        ),
        minimum_scene_improvement_fraction=float(
            validation["minimum_scene_improvement_fraction"]
        ),
        maximum_scene_rmse_regression_fraction=float(
            validation["maximum_scene_rmse_regression_fraction"]
        ),
        visual_seed=int(validation["fixed_seed"]),
    )


def _validation_score(
    metrics: dict[str, dict[str, float]],
    stage: str,
    config: PairedTemporalTrainConfig,
) -> float:
    if not metrics:
        return float("inf")
    required_one_frame = {"translation/one", "forecast/one"}
    if not required_one_frame.issubset(metrics):
        return float("inf")
    if stage == "physical":
        if any(
            values["physical_anchor_improvement_percent"]
            < config.minimum_physical_anchor_improvement_percent
            or values["source_evidence_improvement_percent"]
            < config.minimum_source_evidence_improvement_percent
            for values in metrics.values()
        ):
            return float("inf")
        return max(values["physical_rmse"] for values in metrics.values())
    if stage == "detail":
        audited_translation = [
            values
            for key, values in metrics.items()
            if key.startswith("translation/") and values["detail_zero_mae"] > 1e-8
        ]
        if not audited_translation or any(
            values["detail_improvement_percent"] <= 0.0 for values in audited_translation
        ):
            return float("inf")
        return max(values["detail_mae"] for values in audited_translation)
    if stage == "flow":
        return max(values["flow_objective"] for values in metrics.values())
    if any(
        values["visual_over_physical"] > config.visual_rmse_budget
        or values["visual_frequency_improvement_percent"] <= 0.0
        or values["pre_projection_violation"] > config.pre_projection_violation_maximum
        or values["visual_frequency_improvement_fraction"]
        < config.minimum_scene_improvement_fraction
        or values["visual_rmse_regression_fraction"]
        > config.maximum_scene_rmse_regression_fraction
        for values in metrics.values()
    ):
        return float("inf")
    return max(values["visual_frequency"] for values in metrics.values())


def _collective_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    if dist.is_initialized() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _validate_protocol_config(config: dict[str, Any], world_size: int) -> None:
    training = config["training"]
    validation = config["validation"]
    release = config.get("release", {})
    if str(training.get("precision")) != "bfloat16":
        raise ValueError("paired temporal runner requires training.precision=bfloat16")
    if int(training.get("world_size", -1)) != world_size:
        raise ValueError("training.world_size must match the launched WORLD_SIZE")
    validate_every = int(training["validate_every"])
    full_validate_every = int(training["full_validate_every"])
    patience = int(training["early_stop_full_validations"])
    if validate_every <= 0 or full_validate_every <= 0 or full_validate_every % validate_every:
        raise ValueError("full_validate_every must be a positive multiple of validate_every")
    if patience <= 0:
        raise ValueError("early_stop_full_validations must be positive")
    if validation.get("no_best_of_k") is not True:
        raise ValueError("validation.no_best_of_k must be true")
    if not isinstance(release, dict) or release.get("validation_selected_release_only") is not True:
        raise ValueError("release.validation_selected_release_only must be true")
    required_release_guards = (
        "require_translation_and_forecast_gates",
        "require_one_frame_gate",
        "physical_checkpoint_frozen_before_detail",
    )
    if any(release.get(name) is not True for name in required_release_guards):
        raise ValueError("all paired temporal release safety guards must be true")


def _best_score_from_metrics(metrics: object) -> float:
    if not isinstance(metrics, dict):
        return float("inf")
    candidate = metrics.get("best_score", metrics.get("validation_score", float("inf")))
    try:
        return float(candidate)
    except (TypeError, ValueError):
        return float("inf")


def _no_improvement_count_from_metrics(metrics: object) -> int:
    if not isinstance(metrics, dict):
        return 0
    try:
        return max(0, int(metrics.get("no_improvement_full_validations", 0)))
    except (TypeError, ValueError):
        return 0


def _synchronize_balance_release(
    model: SparsePairedAnchorTransport, *, world_size: int
) -> None:
    """Copy rank-zero's fixed validation release to every DDP replica."""

    if world_size <= 1:
        return
    dist.broadcast(model.detail_scale.data, src=0)
    dist.broadcast(model.visual_scale.data, src=0)


def main() -> None:
    args = _parser().parse_args()
    if args.resume is not None and args.init_checkpoint is not None:
        raise SystemExit("use either --resume or --init-checkpoint, not both")
    if args.stage != "physical" and args.resume is None and args.init_checkpoint is None:
        raise SystemExit("non-physical stages require --init-checkpoint or --resume")
    rank, local_rank, world_size, device = _distributed()
    del local_rank
    try:
        config = _load_config(args.config)
        _validate_protocol_config(config, world_size)
        _seed(args.seed, rank)
        prepared_indexes = _prepare_indexes(
            config,
            args.direction,
            args.output,
            rank,
            world_size,
            device,
        )
        train_index = prepared_indexes.train_path
        validation_index = prepared_indexes.validation_path
        protocol_hash = prepared_indexes.protocol_hash
        train_dataset, validation_dataset = _datasets(
            config, train_index, validation_index, args.seed, args.stage
        )
        model = SparsePairedAnchorTransport(_model_config(config))
        train_config = _train_config(config, args.direction, args.stage)
        previous = {"detail": ("physical",), "flow": ("detail",), "balance": ("flow",)}
        if args.init_checkpoint is not None:
            load_paired_temporal_checkpoint(
                args.init_checkpoint,
                model,
                direction=args.direction,
                allowed_stages=previous.get(args.stage, PAIRED_TEMPORAL_STAGES),
                expected_protocol_sha256=protocol_hash,
            )
        set_paired_temporal_stage(model, args.stage)
        model.to(device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = AdamW(
            trainable,
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        start_step = 0
        best_score = float("inf")
        no_improvement_full_validations = 0
        if args.resume is not None:
            payload = load_paired_temporal_checkpoint(
                args.resume,
                model,
                direction=args.direction,
                allowed_stages=(args.stage,),
                expected_protocol_sha256=protocol_hash,
            )
            optimizer_state = payload.get("optimizer")
            if not isinstance(optimizer_state, dict):
                raise RuntimeError("resume checkpoint has no optimizer state")
            optimizer.load_state_dict(optimizer_state)
            start_step = int(payload["step"])
            best_score = _best_score_from_metrics(payload.get("metrics"))
            no_improvement_full_validations = _no_improvement_count_from_metrics(
                payload.get("metrics")
            )
        training_module = PairedTemporalTrainingModule(model, train_config).to(device)
        distributed_module: torch.nn.Module = training_module
        if world_size > 1:
            distributed_module = DistributedDataParallel(
                training_module,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
            if world_size > 1
            else None
        )
        loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=device.type == "cuda",
            drop_last=True,
            collate_fn=collate_paired_temporal,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(args.num_workers, 2),
            persistent_workers=args.num_workers > 0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_paired_temporal,
        )
        configured_steps = int(config["training"]["stages"][args.stage]["steps"])
        maximum_steps = args.max_steps or configured_steps
        validate_every = int(config["training"]["validate_every"])
        full_validate_every = int(config["training"]["full_validate_every"])
        full_validation_patience = int(config["training"]["early_stop_full_validations"])
        pilot_validation_samples = 32
        pilot_validation_batches = max(
            1,
            (pilot_validation_samples + args.batch_size - 1) // args.batch_size,
        )
        epoch = 0
        iterator = iter(loader)
        output_dir = args.output / args.direction / args.stage
        output_dir.mkdir(parents=True, exist_ok=True)
        for step in range(start_step + 1, maximum_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            tensors = paired_tensor_batch(batch, device)
            tensors = apply_observation_dropout(
                tensors,
                frame_probability=train_config.observation_dropout,
                query_probability=train_config.query_observation_dropout,
                translation_max_delta_days=train_config.translation_max_delta_days,
                one_frame_probability=train_config.one_frame_probability,
                two_to_three_frame_probability=train_config.two_to_three_frame_probability,
                four_plus_frame_probability=train_config.four_plus_frame_probability,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss, stage_metrics, _ = distributed_module(tensors)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite paired temporal loss at step {step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, train_config.gradient_clip
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(f"non-finite paired temporal gradient at step {step}")
            optimizer.step()
            if step % validate_every == 0 or step == maximum_steps:
                if world_size > 1:
                    dist.barrier()
                validation_error: Exception | None = None
                stop_after_validation = False
                is_full_validation = step % full_validate_every == 0 or step == maximum_steps
                validation_scope = "full" if is_full_validation else "pilot_32"
                try:
                    if rank == 0:
                        release_selection: dict[str, float | bool] | None = None
                        if args.stage == "balance" and is_full_validation:
                            release_selection = validation_release_search(
                                model,
                                validation_loader,
                                train_config,
                                device=device,
                            )
                        metrics = evaluate_paired_temporal_batches(
                            model,
                            validation_loader,
                            train_config,
                            device=device,
                            limit_batches=None if is_full_validation else pilot_validation_batches,
                        )
                        score = _validation_score(metrics, args.stage, train_config)
                        is_best = is_full_validation and score < best_score
                        if is_full_validation:
                            if is_best:
                                best_score = score
                                no_improvement_full_validations = 0
                            else:
                                no_improvement_full_validations += 1
                        stop_after_validation = (
                            is_full_validation
                            and no_improvement_full_validations >= full_validation_patience
                        )
                        report = {
                            "step": step,
                            "loss": float(loss.detach()),
                            "gradient_norm": float(gradient_norm),
                            "train_metrics": {
                                name: float(value) for name, value in stage_metrics.items()
                            },
                            "validation": metrics,
                            "validation_score": score,
                            "best_score": best_score,
                            "full_validation": is_full_validation,
                            "validation_scope": validation_scope,
                            "no_improvement_full_validations": no_improvement_full_validations,
                            "early_stop": stop_after_validation,
                            "release_selection": release_selection,
                            "protocol_hash": protocol_hash,
                        }
                        save_paired_temporal_checkpoint(
                            output_dir / "latest.pt",
                            model=model,
                            config=train_config,
                            step=step,
                            optimizer=optimizer,
                            metrics=report,
                            protocol={"sha256": protocol_hash},
                        )
                        if is_best:
                            save_paired_temporal_checkpoint(
                                output_dir / f"best_{args.stage}.pt",
                                model=model,
                                config=train_config,
                                step=step,
                                optimizer=optimizer,
                                metrics=report,
                                protocol={"sha256": protocol_hash},
                            )
                        report_path = output_dir / "latest_report.json"
                        temporary = report_path.with_name(
                            f".{report_path.name}.{os.getpid()}.tmp"
                        )
                        temporary.write_text(
                            json.dumps(report, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                        os.replace(temporary, report_path)
                except Exception as error:  # noqa: BLE001 - broadcast rank-0 validation failure
                    validation_error = error
                failure = torch.tensor(
                    [1 if validation_error is not None else 0], device=device, dtype=torch.int32
                )
                if world_size > 1:
                    dist.broadcast(failure, src=0)
                if int(failure.item()):
                    if validation_error is not None:
                        raise validation_error
                    raise RuntimeError("rank 0 paired temporal validation failed")
                if args.stage == "balance" and is_full_validation:
                    _synchronize_balance_release(model, world_size=world_size)
                stop = torch.tensor(
                    [1 if stop_after_validation else 0], device=device, dtype=torch.int32
                )
                if world_size > 1:
                    dist.broadcast(stop, src=0)
                if int(stop.item()):
                    break
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
