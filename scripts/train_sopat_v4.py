"""Train one bidirectional SOPAT V4 checkpoint with DDP-safe coupled batches.

Launch on eight GPUs with ``torchrun --nproc_per_node=8``.  Each global step
contains one SAR-to-Optical and one Optical-to-SAR homogeneous microbatch;
there is never a channel-padded mixed-direction batch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from sentinel_v3.dataset_builder import file_sha256
from sentinel_v3.paired_temporal_data import load_paired_temporal_index
from sentinel_v4.cache import (
    preflight_sopat_v4_chunk_cache,
    sopat_chunk_dataset_from_cache,
)
from sentinel_v4.data import (
    CoupledDirectionLoader,
    SOPATDirectionDataset,
    SOPATIndexV4,
    collate_sopat_direction,
    load_sopat_v4_index,
    migrate_paired_temporal_index_v4,
    write_sopat_v4_index,
)
from sentinel_v4.evaluation import (
    SELECTION_POLICY_VERSION,
    NoSingletonBatchSampler,
    SOPATSelectionConfig,
    SOPATSelectionDecision,
    SOPATVariantConfig,
    evaluate_sopat_loaders,
    export_sopat_prediction_samples,
    is_better_sopat_candidate,
    select_sopat_candidate,
)
from sentinel_v4.model import SOPAT, SOPATConfig
from sentinel_v4.training import (
    DIRECTIONS,
    ModelEMA,
    SOPATTrainConfig,
    SOPATTrainingModule,
    canonical_json_sha256,
    capture_rng_state,
    configure_sopat_stage,
    evaluate_factorizer_loaders,
    gather_rng_states,
    initialize_from_sopat_checkpoint,
    initialize_from_v3_checkpoint,
    load_sopat_checkpoint,
    save_sopat_checkpoint,
    train_coupled_step,
)


@dataclass(frozen=True)
class _PreparedData:
    """Resolved V4 index and immutable artifacts bound into a checkpoint."""

    route: str
    index_path: Path
    cache_root: Path | None
    manifest_path: Path | None
    train_split: str
    validation_split: str
    protocol_hashes: dict[str, dict[str, str]]

    def payload(self) -> dict[str, object]:
        return {
            "route": self.route,
            "index_path": str(self.index_path),
            "cache_root": str(self.cache_root) if self.cache_root is not None else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "train_split": self.train_split,
            "validation_split": self.validation_split,
            "protocol_hashes": self.protocol_hashes,
        }

    @classmethod
    def from_payload(cls, values: Mapping[str, object]) -> _PreparedData:
        route = values.get("route")
        index_path = values.get("index_path")
        cache_root = values.get("cache_root")
        manifest_path = values.get("manifest_path")
        train_split = values.get("train_split")
        validation_split = values.get("validation_split")
        protocol_hashes = values.get("protocol_hashes")
        if not all(isinstance(value, str) for value in (route, index_path, train_split, validation_split)):
            raise TypeError("SOPAT prepared-data broadcast is incomplete")
        if cache_root is not None and not isinstance(cache_root, str):
            raise TypeError("SOPAT prepared-data cache root is invalid")
        if manifest_path is not None and not isinstance(manifest_path, str):
            raise TypeError("SOPAT prepared-data manifest path is invalid")
        if not isinstance(protocol_hashes, Mapping):
            raise TypeError("SOPAT prepared-data protocol hashes are invalid")
        normalized: dict[str, dict[str, str]] = {}
        for direction in DIRECTIONS:
            artifacts = protocol_hashes.get(direction)
            if not isinstance(artifacts, Mapping):
                raise TypeError("SOPAT prepared-data direction protocol is invalid")
            normalized[direction] = {str(name): str(value) for name, value in artifacts.items()}
        return cls(
            route=route,
            index_path=Path(index_path),
            cache_root=Path(cache_root) if cache_root is not None else None,
            manifest_path=Path(manifest_path) if manifest_path is not None else None,
            train_split=train_split,
            validation_split=validation_split,
            protocol_hashes=normalized,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("factorizer", "physical"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--init-use-ema",
        action="store_true",
        help="initialize model weights from the checkpoint EMA instead of raw model weights",
    )
    parser.add_argument("--init-v3", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--encoder-learning-rate", type=float)
    parser.add_argument("--trainable-scope", choices=("full", "confidence_only"))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
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
        raise TypeError("SOPAT V4 configuration must be a mapping")
    return values


def _resolve_path(value: object, *, base: Path, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"SOPAT data.{field} must be a path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _mapping(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = values.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"SOPAT configuration requires a {name} mapping")
    return value


def _data_splits(data: Mapping[str, Any]) -> tuple[str, str]:
    train_split = data.get("train_split")
    validation_split = data.get("validation_split")
    if not isinstance(train_split, str) or not isinstance(validation_split, str):
        raise TypeError("SOPAT data.train_split and data.validation_split must be strings")
    if train_split == validation_split:
        raise ValueError("SOPAT train and validation splits must differ")
    return train_split, validation_split


def _direction_path(value: object, *, field: str, direction: str, base: Path) -> Path:
    selected = value
    if isinstance(value, Mapping):
        if direction not in value:
            raise ValueError(f"SOPAT data.{field} has no {direction} path")
        selected = value[direction]
    if not isinstance(selected, (str, Path)):
        raise TypeError(f"SOPAT data.{field} must be a path/template or direction mapping")
    rendered = os.fspath(selected)
    try:
        rendered = rendered.format(direction=direction)
    except KeyError as error:
        raise ValueError(f"SOPAT data.{field} has an unsupported format key") from error
    return _resolve_path(rendered, base=base, field=field)


def _explicit_v3_paths(data: Mapping[str, Any], *, base: Path) -> dict[str, dict[str, Path]]:
    train = data.get("train_index")
    validation = data.get("validation_index")
    if train is None or validation is None:
        raise ValueError(
            "raw SOPAT route requires data.sopat_v4_index or both explicit "
            "data.train_index/data.validation_index; automatic raw index building is disabled"
        )
    return {
        direction: {
            "train": _direction_path(train, field="train_index", direction=direction, base=base),
            "validation": _direction_path(
                validation, field="validation_index", direction=direction, base=base
            ),
        }
        for direction in DIRECTIONS
    }


def _cache_index_artifacts(
    cache_root: Path,
    index_path: Path,
    *,
    train_split: str,
    validation_split: str,
    verify_chunks: bool,
) -> dict[str, dict[str, str]]:
    preflight = preflight_sopat_v4_chunk_cache(
        cache_root,
        index_path,
        verify_chunks=verify_chunks,
    )
    cache_index = cache_root / "cache_index.json"
    if not cache_index.is_file():
        raise FileNotFoundError(
            "SOPAT chunk-cache preflight failed: cache_index.json is missing at "
            f"{cache_index}; materialization must complete before training"
        )
    try:
        payload = json.loads(cache_index.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"SOPAT chunk-cache cache_index.json is invalid: {cache_index}") from error
    if not isinstance(payload, Mapping):
        raise TypeError("SOPAT chunk-cache cache_index.json must be a mapping")
    entries = payload.get("indexes")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("SOPAT chunk-cache cache_index.json lacks indexes")
    result: dict[str, dict[str, str]] = {}
    for direction in DIRECTIONS:
        artifacts = {
            "cache_index": preflight.cache_index_sha256,
            "cache_plan": preflight.cache_plan_sha256,
            "cache_v4_index_content": preflight.index_content_sha256,
        }
        for split, label in ((train_split, "train"), (validation_split, "validation")):
            matches = [
                entry
                for entry in entries
                if isinstance(entry, Mapping)
                and entry.get("direction") == direction
                and entry.get("split") == split
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"SOPAT chunk-cache must have exactly one cached index for {direction}/{split}"
                )
            entry = matches[0]
            relative = entry.get("relative_path")
            digest = entry.get("sha256")
            if not isinstance(relative, str) or not isinstance(digest, str):
                raise TypeError("SOPAT chunk-cache index entry lacks path/hash")
            index_path = cache_root / relative
            if not index_path.is_file() or file_sha256(index_path) != digest:
                raise RuntimeError(f"SOPAT chunk-cache index is missing or corrupt: {index_path}")
            artifacts[f"cache_{label}_index"] = digest
        result[direction] = artifacts
    return result


def _combine_migrated_indexes(indexes: Sequence[SOPATIndexV4]) -> SOPATIndexV4:
    if not indexes:
        raise ValueError("SOPAT V4 migration needs at least one index")
    config = indexes[0].config
    if any(index.config != config for index in indexes[1:]):
        raise ValueError("migrated SOPAT V4 direction indexes have incompatible causal configuration")
    examples = tuple(example for index in indexes for example in index.examples)
    return SOPATIndexV4(config=config, examples=examples)


def _validate_v4_index(index: SOPATIndexV4, *, train_split: str, validation_split: str) -> None:
    for direction in DIRECTIONS:
        for split in (train_split, validation_split):
            if not index.select(direction=direction, split=split):
                raise ValueError(f"SOPAT V4 index has no examples for {direction}/{split}")


def _prepare_data_on_rank_zero(
    config: Mapping[str, Any], *, output: Path, config_base: Path
) -> _PreparedData:
    data = _mapping(config, "data")
    train_split, validation_split = _data_splits(data)
    config_hash = canonical_json_sha256(dict(config))
    cache_root_value = data.get("chunk_cache_root")
    v4_index_value = data.get("sopat_v4_index")
    if cache_root_value is not None:
        cache_root = _resolve_path(cache_root_value, base=config_base, field="chunk_cache_root")
        if v4_index_value is None:
            raise ValueError(
                "SOPAT chunk-cache route requires data.sopat_v4_index. The V4 role index is "
                "the causal/provenance contract and must not be reconstructed from raw assets."
            )
        index_path = _resolve_path(v4_index_value, base=config_base, field="sopat_v4_index")
        if not index_path.is_file():
            raise FileNotFoundError(f"SOPAT V4 index is missing: {index_path}")
        index = load_sopat_v4_index(index_path)
        _validate_v4_index(index, train_split=train_split, validation_split=validation_split)
        protocol = _cache_index_artifacts(
            cache_root,
            index_path,
            train_split=train_split,
            validation_split=validation_split,
            verify_chunks=bool(data.get("verify_chunk_cache", False)),
        )
        index_digest = file_sha256(index_path)
        for direction in DIRECTIONS:
            protocol[direction].update(
                {
                    "config": config_hash,
                    "sopat_v4_index": index_digest,
                    "sopat_index_content": index.content_hash,
                    "sopat_index_protocol": index.protocol_hash,
                }
            )
        return _PreparedData(
            "chunk_cache", index_path, cache_root, None, train_split, validation_split, protocol
        )

    manifest_value = data.get("manifest")
    if manifest_value is None:
        raise ValueError("SOPAT raw route requires data.manifest")
    manifest = _resolve_path(manifest_value, base=config_base, field="manifest")
    if not manifest.is_file():
        raise FileNotFoundError(f"SOPAT raw manifest is missing: {manifest}")
    manifest_digest = file_sha256(manifest)
    protocol: dict[str, dict[str, str]] = {}
    if v4_index_value is not None:
        index_path = _resolve_path(v4_index_value, base=config_base, field="sopat_v4_index")
        if not index_path.is_file():
            raise FileNotFoundError(f"SOPAT V4 index is missing: {index_path}")
        index = load_sopat_v4_index(index_path)
        _validate_v4_index(index, train_split=train_split, validation_split=validation_split)
        for direction in DIRECTIONS:
            protocol[direction] = {
                "config": config_hash,
                "manifest": manifest_digest,
                "sopat_v4_index": file_sha256(index_path),
                "sopat_index_content": index.content_hash,
                "sopat_index_protocol": index.protocol_hash,
            }
        return _PreparedData(
            "raw_v4_index", index_path, None, manifest, train_split, validation_split, protocol
        )

    paths = _explicit_v3_paths(data, base=config_base)
    migrated: list[SOPATIndexV4] = []
    for direction in DIRECTIONS:
        protocol[direction] = {"config": config_hash, "manifest": manifest_digest}
        for split, label in ((train_split, "train"), (validation_split, "validation")):
            path = paths[direction][label]
            if not path.is_file():
                raise FileNotFoundError(f"SOPAT explicit {label} index is missing: {path}")
            legacy = load_paired_temporal_index(path, direction=direction)
            if any(sample.split != split for sample in legacy):
                raise ValueError(f"SOPAT explicit {label} index has a mismatched split: {path}")
            migrated.append(migrate_paired_temporal_index_v4(legacy, manifest))
            protocol[direction][f"{label}_v3_index"] = file_sha256(path)
    index = _combine_migrated_indexes(migrated)
    _validate_v4_index(index, train_split=train_split, validation_split=validation_split)
    generated = output / "indices" / f"sopat_v4_{config_hash[:16]}.jsonl"
    write_sopat_v4_index(generated, index)
    index_digest = file_sha256(generated)
    for direction in DIRECTIONS:
        protocol[direction].update(
            {
                "sopat_v4_index": index_digest,
                "sopat_index_content": index.content_hash,
                "sopat_index_protocol": index.protocol_hash,
            }
        )
    return _PreparedData(
        "raw_explicit_v3", generated, None, manifest, train_split, validation_split, protocol
    )


def _prepare_data(
    config: Mapping[str, Any],
    *,
    output: Path,
    config_base: Path,
    rank: int,
    world_size: int,
    device: torch.device,
) -> _PreparedData:
    prepared: _PreparedData | None = None
    failure: Exception | None = None
    if rank == 0:
        try:
            prepared = _prepare_data_on_rank_zero(config, output=output, config_base=config_base)
        except Exception as error:  # noqa: BLE001 - synchronize rank-zero preflight errors
            failure = error
    if world_size <= 1:
        if failure is not None:
            raise failure
        if prepared is None:
            raise RuntimeError("SOPAT rank-zero data preflight produced no route")
        return prepared
    payload: list[object] = [None]
    if rank == 0:
        payload[0] = (
            {"ok": True, "prepared": prepared.payload()}
            if prepared is not None
            else {"ok": False, "error": f"{type(failure).__name__}: {failure}"}
        )
    dist.broadcast_object_list(payload, src=0, device=device)
    message = payload[0]
    if not isinstance(message, Mapping) or not isinstance(message.get("ok"), bool):
        raise TypeError("SOPAT data-preflight broadcast is invalid")
    if message["ok"] is not True:
        if failure is not None:
            raise failure
        raise RuntimeError(f"SOPAT rank-zero data preflight failed: {message.get('error')}")
    serialized = message.get("prepared")
    if not isinstance(serialized, Mapping):
        raise TypeError("SOPAT data-preflight broadcast lacks prepared data")
    return _PreparedData.from_payload(serialized)


def _dataset_options(data: Mapping[str, Any], *, stage: str) -> dict[str, object]:
    try:
        crop_size = int(data["crop_size"])
        minimum_valid = float(data.get("minimum_valid_fraction", 0.80))
        maximum_observations = int(data["maximum_observations"])
    except KeyError as error:
        raise ValueError(f"SOPAT data configuration is missing {error.args[0]}") from error
    return {
        "crop_size": crop_size,
        "minimum_valid_fraction": minimum_valid,
        "max_observations": maximum_observations,
        "registration_audit": bool(data.get("registration_audit", True)) and stage == "physical",
        "maximum_registration_shift_px": float(data.get("maximum_registration_shift_px", 0.5)),
    }


def _datasets(
    config: Mapping[str, Any], prepared: _PreparedData, *, seed: int, stage: str
) -> tuple[dict[str, SOPATDirectionDataset], dict[str, SOPATDirectionDataset]]:
    data = _mapping(config, "data")
    index = load_sopat_v4_index(prepared.index_path)
    _validate_v4_index(index, train_split=prepared.train_split, validation_split=prepared.validation_split)
    options = _dataset_options(data, stage=stage)
    train: dict[str, SOPATDirectionDataset] = {}
    validation: dict[str, SOPATDirectionDataset] = {}
    if prepared.route == "chunk_cache":
        cache_root = prepared.cache_root
        if cache_root is None:
            raise RuntimeError("SOPAT chunk-cache prepared route has no resolved cache root")
        for direction in DIRECTIONS:
            train[direction] = sopat_chunk_dataset_from_cache(
                cache_root,
                index,
                direction=direction,
                split=prepared.train_split,
                window_mode="all",
                permutation_seed=seed,
                minimum_valid_fraction=float(options["minimum_valid_fraction"]),
                max_observations=int(options["max_observations"]),
                registration_audit=bool(options["registration_audit"]),
                maximum_registration_shift_px=float(options["maximum_registration_shift_px"]),
            )
            validation[direction] = sopat_chunk_dataset_from_cache(
                cache_root,
                index,
                direction=direction,
                split=prepared.validation_split,
                window_mode="center",
                permutation_seed=seed,
                minimum_valid_fraction=float(options["minimum_valid_fraction"]),
                max_observations=int(options["max_observations"]),
                registration_audit=bool(options["registration_audit"]),
                maximum_registration_shift_px=float(options["maximum_registration_shift_px"]),
            )
    else:
        manifest = prepared.manifest_path
        if manifest is None:
            raise RuntimeError("SOPAT raw prepared route has no resolved manifest path")
        train_options = {**options, "crop_mode": "random_valid"}
        validation_options = {**options, "crop_mode": "center"}
        for direction in DIRECTIONS:
            train[direction] = SOPATDirectionDataset.from_raster(
                index,
                manifest,
                direction=direction,
                split=prepared.train_split,
                permutation_seed=seed,
                **train_options,
            )
            validation[direction] = SOPATDirectionDataset.from_raster(
                index,
                manifest,
                direction=direction,
                split=prepared.validation_split,
                permutation_seed=seed,
                **validation_options,
            )
    return train, validation


def _model_config(config: Mapping[str, Any]) -> SOPATConfig:
    values = dict(_mapping(config, "model"))
    allowed = {field.name for field in fields(SOPATConfig)}
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"unknown SOPAT model setting(s): {', '.join(unknown)}")
    data = _mapping(config, "data")
    values.setdefault("max_horizon_days", int(data.get("horizon_days", 180)))
    values.setdefault("translation_tolerance_days", int(data.get("translation_max_delta_days", 1)))
    return SOPATConfig(**values)


def _train_config(config: Mapping[str, Any], *, stage: str) -> SOPATTrainConfig:
    training = _mapping(config, "training")
    stages = training.get("stages")
    if not isinstance(stages, Mapping) or not isinstance(stages.get(stage), Mapping):
        raise TypeError(f"SOPAT training.stages.{stage} must be a mapping")
    allowed = {field.name for field in fields(SOPATTrainConfig)}
    values: dict[str, object] = {}
    objective = training.get("objective", {})
    if not isinstance(objective, Mapping):
        raise TypeError("SOPAT training.objective must be a mapping")
    values.update(objective)
    for name in allowed:
        if name in training:
            values[name] = training[name]
    values.update(stages[stage])
    values.pop("steps", None)
    values["stage"] = stage
    return SOPATTrainConfig.from_mapping(values)


def _selection_config(config: Mapping[str, Any], *, phase: str) -> SOPATSelectionConfig:
    validation = _mapping(config, "validation")
    values = validation.get("selection", {})
    if not isinstance(values, Mapping):
        raise TypeError("SOPAT validation.selection must be a mapping")
    allowed = {field.name for field in fields(SOPATSelectionConfig)}
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(f"unknown SOPAT selection setting(s): {', '.join(unknown)}")
    payload = dict(values)
    payload["phase"] = phase
    if "required_observation_counts" in payload:
        counts = payload["required_observation_counts"]
        if not isinstance(counts, Sequence) or isinstance(counts, (str, bytes)):
            raise TypeError("SOPAT selection required_observation_counts must be a sequence")
        payload["required_observation_counts"] = tuple(counts)
    if "required_tasks" in payload:
        tasks = payload["required_tasks"]
        if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
            raise TypeError("SOPAT selection required_tasks must be a sequence")
        payload["required_tasks"] = tuple(str(task) for task in tasks)
    return SOPATSelectionConfig(**payload)


def _validate_run_config(config: Mapping[str, Any], *, world_size: int) -> None:
    training = _mapping(config, "training")
    configured_world_size = training.get("world_size")
    if configured_world_size is not None and int(configured_world_size) != world_size:
        raise ValueError("SOPAT training.world_size must match launched WORLD_SIZE")
    precision = training.get("precision", "bfloat16")
    if precision != "bfloat16":
        raise ValueError("SOPAT runner currently requires training.precision=bfloat16")
    for name in ("validate_every", "full_validate_every", "early_stop_full_validations"):
        if int(training.get(name, 0)) <= 0:
            raise ValueError(f"SOPAT training.{name} must be positive")
    if int(training["full_validate_every"]) % int(training["validate_every"]):
        raise ValueError("SOPAT full_validate_every must be a multiple of validate_every")
    activation_checkpointing = training.get("activation_checkpointing", False)
    if not isinstance(activation_checkpointing, bool):
        raise TypeError("SOPAT training.activation_checkpointing must be a boolean")
    log_every = int(training.get("log_every", 10))
    if log_every <= 0:
        raise ValueError("SOPAT training.log_every must be positive")
    validation = _mapping(config, "validation")
    validation_batch_size = validation.get("batch_size", training.get("batch_size", 1))
    if int(validation_batch_size) < 2:
        raise ValueError(
            "SOPAT validation.batch_size must be at least 2 for source_shuffle evaluation"
        )


def _seed(seed: int, rank: int) -> None:
    resolved = int(seed) + int(rank) * 1_000_003
    random.seed(resolved)
    np.random.seed(resolved % (2**32))
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)


def _loaders(
    datasets: Mapping[str, SOPATDirectionDataset],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    rank: int,
    world_size: int,
    training: bool,
    seed: int,
    batch_samplers: Mapping[str, Sampler[list[int]]] | None = None,
) -> dict[str, DataLoader[dict[str, object]]]:
    if training and batch_samplers is not None:
        raise ValueError("SOPAT training loaders cannot use validation batch samplers")
    result: dict[str, DataLoader[dict[str, object]]] = {}
    for direction in DIRECTIONS:
        dataset = datasets[direction]
        if not training and batch_samplers is None:
            batch_sampler: Sampler[list[int]] = NoSingletonBatchSampler(len(dataset), batch_size)
        elif not training:
            try:
                batch_sampler = batch_samplers[direction]  # type: ignore[index]
            except KeyError as error:
                raise ValueError(f"missing validation batch sampler for {direction}") from error
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=training,
                seed=seed,
                drop_last=training,
            )
            if world_size > 1 and training
            else None
        )
        if training:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                pin_memory=device.type == "cuda",
                drop_last=True,
                collate_fn=collate_sopat_direction,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                pin_memory=device.type == "cuda",
                collate_fn=collate_sopat_direction,
            )
        if len(loader) <= 0:
            raise RuntimeError(
                f"SOPAT {direction} {'train' if training else 'validation'} loader is empty; "
                "reduce batch size or check cache/index sample counts"
            )
        result[direction] = loader
    return result


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _optimizer_parameter_groups(model: SOPAT, config: SOPATTrainConfig) -> list[dict[str, object]]:
    """Separate the transferred encoder from V4-specific trainable modules.

    A physical full run can preserve low-level sensor features with a much
    smaller encoder learning rate while allowing the newly initialized
    factorizer/transport/rendering heads to adapt.  The groups intentionally
    keep the checkpoint's normal AdamW state representation unchanged.
    """

    encoder = getattr(model, "encoder", None)
    encoder_ids = {
        id(parameter)
        for parameter in encoder.parameters()
        if parameter.requires_grad
    } if isinstance(encoder, torch.nn.Module) else set()
    encoder_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) in encoder_ids
    ]
    other_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in encoder_ids
    ]
    groups: list[dict[str, object]] = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": config.encoder_learning_rate})
    if other_parameters:
        groups.append({"params": other_parameters, "lr": config.learning_rate})
    if not groups:
        raise RuntimeError("SOPAT selected stage has no optimizer parameters")
    return groups


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _best_from_checkpoint(payload: Mapping[str, object]) -> SOPATSelectionDecision | None:
    metrics = payload.get("best_metrics")
    if not isinstance(metrics, Mapping):
        return None
    decision = metrics.get("selection")
    if not isinstance(decision, Mapping):
        return None
    try:
        return SOPATSelectionDecision(
            eligible=bool(decision["eligible"]),
            score=float(decision["score"]),
            failures=tuple(str(value) for value in decision.get("failures", ())),
            phase=str(decision["phase"]),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError):
        return None


def _best_factorizer_loss(payload: Mapping[str, object]) -> float | None:
    metrics = payload.get("best_metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get("factorizer_weighted_loss")
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _evaluate_validation(
    model: SOPAT,
    ema: ModelEMA,
    loaders: Mapping[str, DataLoader[dict[str, object]]],
    *,
    device: torch.device,
    full: bool,
    batch_size: int,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, object], SOPATSelectionDecision]:
    validation = _mapping(config, "validation")
    pilot_samples = int(validation.get("pilot_samples", 32))
    limit_batches = None if full else max(1, math.ceil(pilot_samples / batch_size))
    phase = str(validation.get("selection_phase", "full" if full else "feasibility"))
    if not full:
        phase = "feasibility"
    selection = _selection_config(config, phase=phase)
    with ema.average_parameters(model):
        sopat = evaluate_sopat_loaders(
            model,
            loaders,
            variant=SOPATVariantConfig("sopat", seed=seed),
            device=device,
            limit_batches=limit_batches,
        )
        anchor = evaluate_sopat_loaders(
            None,
            loaders,
            variant=SOPATVariantConfig("anchor_copy", seed=seed),
            device=device,
            limit_batches=limit_batches,
        )
        shuffle = evaluate_sopat_loaders(
            model,
            loaders,
            variant=SOPATVariantConfig("source_shuffle", seed=seed),
            device=device,
            limit_batches=limit_batches,
        )
    decision = select_sopat_candidate(
        sopat,
        selection,
        source_shuffle_report=shuffle,
    )
    report = {
        "scope": "full" if full else "pilot",
        "input_matched": True,
        "sopat": sopat,
        "anchor_copy": anchor,
        "source_shuffle": shuffle,
        "selection": decision.to_dict(),
        "selection_policy": {
            "version": SELECTION_POLICY_VERSION,
            "effective": asdict(selection),
        },
        "external_baselines": {
            "note": "V2/V3 external checkpoints require an explicit input-matched adapter; "
            "they are not fabricated by this runner."
        },
    }
    return report, decision


def _evaluate_factorizer_validation(
    model: SOPAT,
    ema: ModelEMA,
    loaders: Mapping[str, DataLoader[dict[str, object]]],
    *,
    device: torch.device,
    full: bool,
    batch_size: int,
    config: Mapping[str, Any],
    train_config: SOPATTrainConfig,
) -> dict[str, object]:
    """Validate factorization using only its paired-anchor objectives."""

    validation = _mapping(config, "validation")
    pilot_samples = int(validation.get("pilot_samples", 32))
    limit_batches = None if full else max(1, math.ceil(pilot_samples / batch_size))
    with ema.average_parameters(model):
        result = evaluate_factorizer_loaders(
            model,
            loaders,
            train_config,
            device=device,
            limit_batches=limit_batches,
        )
    return {
        "scope": "full" if full else "pilot",
        "input_matched": True,
        "factorizer": result.to_dict(),
        "selection": {
            "kind": "factorizer_anchor_objective",
            "weighted_loss": result.weighted_loss,
            "directions": dict(result.direction_losses),
        },
    }


def main() -> None:
    args = _parser().parse_args()
    initializers = sum(value is not None for value in (args.resume, args.init_checkpoint, args.init_v3))
    if initializers > 1:
        raise SystemExit("use only one of --resume, --init-checkpoint, or --init-v3")
    if args.init_use_ema and args.init_checkpoint is None:
        raise SystemExit("--init-use-ema requires --init-checkpoint")
    rank, local_rank, world_size, device = _distributed()
    del local_rank
    try:
        config = _load_config(args.config)
        _validate_run_config(config, world_size=world_size)
        _seed(args.seed, rank)
        prepared = _prepare_data(
            config,
            output=args.output,
            config_base=args.config.parent.resolve(),
            rank=rank,
            world_size=world_size,
            device=device,
        )
        train_datasets, validation_datasets = _datasets(config, prepared, seed=args.seed, stage=args.stage)
        training = _mapping(config, "training")
        batch_size = int(args.batch_size or training.get("batch_size", 1))
        validation = _mapping(config, "validation")
        validation_batch_size = int(validation.get("batch_size", batch_size))
        num_workers = int(args.num_workers if args.num_workers is not None else training.get("num_workers", 4))
        if batch_size <= 0 or validation_batch_size < 2 or num_workers < 0:
            raise ValueError(
                "SOPAT training.batch_size must be positive, validation.batch_size at least 2, "
                "and num_workers non-negative"
            )
        train_loaders = _loaders(
            train_datasets,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            rank=rank,
            world_size=world_size,
            training=True,
            seed=args.seed,
        )
        validation_loaders = _loaders(
            validation_datasets,
            batch_size=validation_batch_size,
            num_workers=min(num_workers, 2),
            device=device,
            rank=0,
            world_size=1,
            training=False,
            seed=args.seed,
        )
        coupled = CoupledDirectionLoader(train_loaders)  # type: ignore[arg-type]
        model_config = _model_config(config)
        train_config = _train_config(config, stage=args.stage)
        learning_rate = (
            train_config.learning_rate if args.learning_rate is None else args.learning_rate
        )
        encoder_learning_rate = (
            train_config.encoder_learning_rate
            if args.encoder_learning_rate is None
            else args.encoder_learning_rate
        )
        train_config = replace(
            train_config,
            learning_rate=learning_rate,
            encoder_learning_rate=encoder_learning_rate,
            trainable_scope=(
                train_config.trainable_scope
                if args.trainable_scope is None
                else args.trainable_scope
            ),
        )
        model = SOPAT(model_config)
        initialization: dict[str, object] | None = None
        if args.init_checkpoint is not None:
            initialization = initialize_from_sopat_checkpoint(
                model,
                args.init_checkpoint,
                model_config=asdict(model_config),
                protocol_hashes=prepared.protocol_hashes,
                use_ema=args.init_use_ema,
            )
        elif args.init_v3 is not None:
            initialization = initialize_from_v3_checkpoint(model, args.init_v3)
        configure_sopat_stage(
            model, args.stage, trainable_scope=train_config.trainable_scope
        )
        activation_checkpointing = bool(training.get("activation_checkpointing", False))
        set_activation_checkpointing = getattr(model, "set_activation_checkpointing", None)
        if callable(set_activation_checkpointing):
            set_activation_checkpointing(activation_checkpointing)
        elif activation_checkpointing:
            raise TypeError(
                "SOPAT training.activation_checkpointing requires a core "
                "model.set_activation_checkpointing(bool) hook"
            )
        model.to(device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("SOPAT selected stage has no trainable parameters")
        optimizer = AdamW(
            _optimizer_parameter_groups(model, train_config),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        configured_steps = int(_mapping(training, "stages")[args.stage]["steps"])
        maximum_steps = int(args.max_steps or configured_steps)
        if maximum_steps <= 0:
            raise ValueError("SOPAT maximum steps must be positive")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=maximum_steps)
        ema = ModelEMA.create(model, train_config.ema_decay)
        training_module = SOPATTrainingModule(model, train_config).to(device)
        distributed_module: torch.nn.Module = training_module
        if world_size > 1:
            distributed_module = DistributedDataParallel(
                training_module,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        start_step = 0
        best: SOPATSelectionDecision | None = None
        best_factorizer_loss: float | None = None
        no_improvement = 0
        if args.resume is not None:
            payload = load_sopat_checkpoint(
                args.resume,
                model=distributed_module,
                optimizer=optimizer,
                ema=ema,
                scheduler=scheduler,
                model_config=asdict(model_config),
                train_config=train_config,
                protocol_hashes=prepared.protocol_hashes,
                restore_rng=True,
            )
            start_step = int(payload["global_step"])
            if args.stage == "physical":
                best = _best_from_checkpoint(payload)
            else:
                best_factorizer_loss = _best_factorizer_loss(payload)
            best_metrics = payload.get("best_metrics")
            if isinstance(best_metrics, Mapping):
                no_improvement = max(0, int(best_metrics.get("no_improvement_full_validations", 0)))
        validate_every = int(training["validate_every"])
        full_validate_every = int(training["full_validate_every"])
        patience = int(training["early_stop_full_validations"])
        log_every = int(training.get("log_every", 10))
        output_dir = args.output / args.stage
        output_dir.mkdir(parents=True, exist_ok=True)
        steps_per_epoch = len(coupled)
        epoch = start_step // steps_per_epoch
        skip = start_step % steps_per_epoch
        global_step = start_step
        stop = False
        while global_step < maximum_steps and not stop:
            coupled.set_epoch(epoch)
            for coupled_batch in coupled:
                if skip:
                    skip -= 1
                    continue
                global_step += 1
                result = train_coupled_step(
                    distributed_module,
                    optimizer,
                    coupled_batch.as_dict(),
                    train_config,
                    ema=ema,
                    generator=_device_generator(device, args.seed + rank * 100_003 + global_step),
                )
                scheduler.step()
                if rank == 0 and (global_step % log_every == 0 or global_step == 1):
                    print(
                        json.dumps(
                            {
                                "event": "train_step",
                                "global_step": global_step,
                                "stage": args.stage,
                                "total_loss": result.total_loss,
                                "gradient_norm": result.gradient_norm,
                                "direction_losses": dict(result.direction_losses),
                                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                                "activation_checkpointing": activation_checkpointing,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                boundary = global_step % validate_every == 0 or global_step == maximum_steps
                if boundary:
                    if world_size > 1:
                        dist.barrier()
                    failure: Exception | None = None
                    validation_report: dict[str, object] | None = None
                    decision: SOPATSelectionDecision | None = None
                    factorizer_loss: float | None = None
                    became_best = False
                    full = global_step % full_validate_every == 0 or global_step == maximum_steps
                    if rank == 0:
                        try:
                            if args.stage == "factorizer":
                                validation_report = _evaluate_factorizer_validation(
                                    model,
                                    ema,
                                    validation_loaders,
                                    device=device,
                                    full=full,
                                    batch_size=validation_batch_size,
                                    config=config,
                                    train_config=train_config,
                                )
                                selection = validation_report["selection"]
                                if not isinstance(selection, Mapping):
                                    raise TypeError("factorizer validation has no selection payload")
                                raw_loss = selection.get("weighted_loss")
                                if not isinstance(raw_loss, (int, float)) or not math.isfinite(raw_loss):
                                    raise FloatingPointError("factorizer validation weighted loss is invalid")
                                factorizer_loss = float(raw_loss)
                            else:
                                validation_report, decision = _evaluate_validation(
                                    model,
                                    ema,
                                    validation_loaders,
                                    device=device,
                                    full=full,
                                    batch_size=validation_batch_size,
                                    config=config,
                                    seed=args.seed,
                                )
                            if full:
                                if args.stage == "factorizer":
                                    assert factorizer_loss is not None
                                    if best_factorizer_loss is None or factorizer_loss < best_factorizer_loss:
                                        best_factorizer_loss = factorizer_loss
                                        no_improvement = 0
                                        became_best = True
                                    else:
                                        no_improvement += 1
                                else:
                                    assert decision is not None
                                    if is_better_sopat_candidate(decision, best):
                                        best = decision
                                        no_improvement = 0
                                        became_best = True
                                    else:
                                        no_improvement += 1
                                stop = no_improvement >= patience
                            report = {
                                "global_step": global_step,
                                "stage": args.stage,
                                "train": {
                                    "total_loss": result.total_loss,
                                    "gradient_norm": result.gradient_norm,
                                    "direction_losses": dict(result.direction_losses),
                                    "metrics": dict(result.metrics),
                                    "learning_rate": optimizer.param_groups[0]["lr"],
                                },
                                "validation": validation_report,
                                "best_metrics": {
                                    "selection": best.to_dict() if best is not None else None,
                                    "factorizer_weighted_loss": best_factorizer_loss,
                                    "no_improvement_full_validations": no_improvement,
                                },
                                "protocol_hashes": prepared.protocol_hashes,
                                "initialization": initialization,
                            }
                        except Exception as error:  # noqa: BLE001 - broadcast rank-zero validation failure
                            failure = error
                    status = torch.tensor([1 if failure is not None else 0], device=device, dtype=torch.int32)
                    if world_size > 1:
                        dist.broadcast(status, src=0)
                    if int(status.item()):
                        if failure is not None:
                            raise failure
                        raise RuntimeError("SOPAT rank-zero validation failed")
                    # All ranks must participate in this collective.  Keeping
                    # rank-local RNG streams records a reproducible DDP resume
                    # without allowing a rank-zero-only all_gather deadlock.
                    rng_states = gather_rng_states(capture_rng_state())
                    if rank == 0:
                        assert validation_report is not None
                        if args.stage == "physical":
                            assert decision is not None
                        else:
                            assert factorizer_loss is not None
                        report = {
                            "global_step": global_step,
                            "stage": args.stage,
                            "train": {
                                "total_loss": result.total_loss,
                                "gradient_norm": result.gradient_norm,
                                "direction_losses": dict(result.direction_losses),
                                "metrics": dict(result.metrics),
                                "learning_rate": optimizer.param_groups[0]["lr"],
                            },
                            "validation": validation_report,
                            "best_metrics": {
                                "selection": best.to_dict() if best is not None else None,
                                "factorizer_weighted_loss": best_factorizer_loss,
                                "no_improvement_full_validations": no_improvement,
                            },
                            "protocol_hashes": prepared.protocol_hashes,
                            "initialization": initialization,
                        }
                        save_sopat_checkpoint(
                            output_dir / "latest.pt",
                            model=distributed_module,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            ema=ema,
                            model_config=asdict(model_config),
                            train_config=train_config,
                            protocol_hashes=prepared.protocol_hashes,
                            global_step=global_step,
                            best_metrics=report["best_metrics"],
                            rng_state=rng_states,
                            data_state={"epoch": epoch, "step_in_epoch": coupled_batch.step},
                        )
                        if full and became_best:
                            save_sopat_checkpoint(
                                output_dir / f"best_{args.stage}.pt",
                                model=distributed_module,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                ema=ema,
                                model_config=asdict(model_config),
                                train_config=train_config,
                                protocol_hashes=prepared.protocol_hashes,
                                global_step=global_step,
                                best_metrics=report["best_metrics"],
                                rng_state=rng_states,
                                data_state={"epoch": epoch, "step_in_epoch": coupled_batch.step},
                            )
                            if args.stage == "physical":
                                panel_count = int(_mapping(config, "validation").get("panel_samples", 16))
                                with ema.average_parameters(model):
                                    export_sopat_prediction_samples(
                                        model,
                                        validation_loaders,
                                        output_dir / "panels",
                                        device=device,
                                        limit_per_direction=panel_count,
                                    )
                        _write_json(output_dir / "latest_report.json", report)
                    if world_size > 1:
                        stop_tensor = torch.tensor([1 if stop else 0], device=device, dtype=torch.int32)
                        dist.broadcast(stop_tensor, src=0)
                        stop = bool(stop_tensor.item())
                if global_step >= maximum_steps or stop:
                    break
            epoch += 1
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
