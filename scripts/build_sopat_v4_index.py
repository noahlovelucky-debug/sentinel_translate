"""Build and publish one immutable, cache-compatible SOPAT V4 role index.

The V4 role index and the legacy paired-temporal indexes consumed by the
memory-mapped cache are a single publication.  Validation is screened at the
fixed center *before* V4 migration; training rows are never screened by target
pixels.  This prevents a zero-support validation row from reaching a long
training run and prevents a V4/V3 sample-set mismatch from reaching cache
preflight.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from sentinel_v3.dataset_builder import file_sha256
from sentinel_v3.paired_temporal_data import (
    ALL_DIRECTIONS,
    PairedTemporalIndex,
    build_paired_temporal_index,
    write_paired_temporal_index,
)
from sentinel_v4.data import (
    SOPATIndexV4,
    migrate_paired_temporal_index_v4,
    paired_temporal_index_from_sopat_v4,
    paired_temporal_protocol_hash,
    write_sopat_v4_index,
)

DEFAULT_OUTPUT = Path("/data/datasets/sopat_v4_2017_2024/index.jsonl")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/sopat_v4_full_index_source.yaml"
PUBLICATION_FORMAT_VERSION = 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--paired-index-root",
        type=Path,
        help="destination for exact V3 direction/split indexes; defaults beside --output",
    )
    parser.add_argument(
        "--publication",
        type=Path,
        help="content-hashed final publication marker; defaults beside --output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the immutable publication marker remains valid",
    )
    return parser


def _absolute(path: str | Path, *, base: Path | None = None) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() and base is not None:
        value = base / value
    return Path(os.path.abspath(value))


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"SOPAT index config requires a {name} mapping")
    return value  # type: ignore[return-value]


def _required_string(data: Mapping[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"SOPAT index config data.{name} must be a non-empty string")
    return value


def _required_int(data: Mapping[str, Any], name: str) -> int:
    value = data.get(name)
    if isinstance(value, bool):
        raise TypeError(f"SOPAT index config data.{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"SOPAT index config data.{name} must be an integer") from error
    if result <= 0:
        raise ValueError(f"SOPAT index config data.{name} must be positive")
    return result


def _task_modes(data: Mapping[str, Any]) -> tuple[str, ...]:
    value = data.get("task_modes", ("translation", "forecast"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("SOPAT index config data.task_modes must be a sequence")
    result = tuple(str(item) for item in value)
    if not result:
        raise ValueError("SOPAT index config data.task_modes cannot be empty")
    return result


def _load_config(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(f"cannot read SOPAT index config: {path}") from error
    return _mapping(value, "top-level")


def _configured_indexes(
    data: Mapping[str, Any], *, manifest: Path
) -> dict[tuple[str, str], PairedTemporalIndex]:
    train_split = _required_string(data, "train_split")
    validation_split = _required_string(data, "validation_split")
    if train_split == validation_split:
        raise ValueError("SOPAT train and validation splits must differ")
    kwargs: dict[str, object] = {
        "min_observations": _required_int(data, "minimum_observations"),
        "max_observations": _required_int(data, "maximum_observations"),
        "horizon_days": _required_int(data, "horizon_days"),
        "anchor_max_delta_days": _required_int(data, "anchor_pair_max_delta_days"),
        "max_anchors_per_query": _required_int(data, "maximum_anchors_per_query"),
        "translation_max_delta_days": _required_int(data, "translation_max_delta_days"),
        "orbit": _required_string(data, "orbit"),
        "task_modes": _task_modes(data),
    }
    result: dict[tuple[str, str], PairedTemporalIndex] = {}
    for direction in ALL_DIRECTIONS:
        for split in (train_split, validation_split):
            index = build_paired_temporal_index(
                manifest,
                direction=direction,
                split=split,
                **kwargs,  # type: ignore[arg-type]
            )
            if not index:
                raise RuntimeError(f"SOPAT selector produced no samples: {direction}/{split}")
            result[(direction, split)] = index
    return result


def _combine_migrated_indexes(
    indexes: Mapping[tuple[str, str], PairedTemporalIndex], *, manifest: Path
) -> SOPATIndexV4:
    migrated = [migrate_paired_temporal_index_v4(index, manifest) for index in indexes.values()]
    config = migrated[0].config
    if any(index.config != config for index in migrated[1:]):
        raise ValueError("SOPAT V4 direction indexes have incompatible global configuration")
    result = SOPATIndexV4(
        config=config,
        examples=tuple(example for index in migrated for example in index.examples),
    )
    _assert_exact_projection(result, indexes)
    return result


def _assert_exact_projection(
    index: SOPATIndexV4, indexes: Mapping[tuple[str, str], PairedTemporalIndex]
) -> None:
    """Prove that every final V4 role is backed by the final V3 cache row."""

    for (direction, split), expected in indexes.items():
        actual = paired_temporal_index_from_sopat_v4(index, direction=direction, split=split)
        expected_by_id = {sample.sample_id: sample for sample in expected.samples}
        actual_by_id = {sample.sample_id: sample for sample in actual.samples}
        if actual_by_id != expected_by_id:
            raise RuntimeError(f"V4/V3 role projection differs for {direction}/{split}")
        expected_protocol = paired_temporal_protocol_hash(expected.config)
        examples = index.select(direction=direction, split=split)
        if len(examples) != len(expected_by_id):
            raise RuntimeError(f"V4/V3 sample count differs for {direction}/{split}")
        if any(example.provenance.source_protocol_hash != expected_protocol for example in examples):
            raise RuntimeError(f"V4 provenance protocol differs for {direction}/{split}")


def _publication_payload(
    *,
    config: Path,
    manifest: Path,
    output: Path,
    paired_root: Path,
    index: SOPATIndexV4,
    indexes: Mapping[tuple[str, str], PairedTemporalIndex],
    staged_index: Path,
    staged_paired_root: Path,
    filter_report: Mapping[str, object],
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for direction, split in sorted(indexes):
        path = staged_paired_root / direction / f"{split}.jsonl"
        entries.append(
            {
                "direction": direction,
                "split": split,
                "path": str(paired_root / direction / f"{split}.jsonl"),
                "file_sha256": file_sha256(path),
                "samples": len(indexes[(direction, split)]),
                "source_protocol_sha256": paired_temporal_protocol_hash(
                    indexes[(direction, split)].config
                ),
            }
        )
    return {
        "format_version": PUBLICATION_FORMAT_VERSION,
        "config_path": str(config),
        "config_sha256": file_sha256(config),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "v4_index_path": str(output),
        "v4_index_file_sha256": file_sha256(staged_index),
        "v4_index_content_sha256": index.content_hash,
        "v4_index_protocol_sha256": index.protocol_hash,
        "paired_indexes": entries,
        "validation_center_filter": dict(filter_report),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> Mapping[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _publication_is_valid(
    publication: Path, *, config: Path, manifest: Path, output: Path, paired_root: Path
) -> bool:
    payload = _read_json(publication)
    if payload is None or payload.get("format_version") != PUBLICATION_FORMAT_VERSION:
        return False
    if payload.get("config_sha256") != file_sha256(config):
        return False
    if payload.get("manifest_sha256") != file_sha256(manifest):
        return False
    if payload.get("v4_index_path") != str(output) or not output.is_file():
        return False
    if payload.get("v4_index_file_sha256") != file_sha256(output):
        return False
    entries = payload.get("paired_indexes")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return False
    required = {(direction, split) for direction in ALL_DIRECTIONS for split in ("train", "validation_temporal")}
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        direction, split = entry.get("direction"), entry.get("split")
        if not isinstance(direction, str) or not isinstance(split, str):
            return False
        key = direction, split
        path = paired_root / direction / f"{split}.jsonl"
        if key in seen or not path.is_file() or entry.get("path") != str(path):
            return False
        if entry.get("file_sha256") != file_sha256(path):
            return False
        seen.add(key)
    return seen == required


def build_and_publish(
    config_path: Path,
    output: Path,
    *,
    paired_index_root: Path | None = None,
    publication: Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Build the V4/V3 index publication or reuse an exact prior publication."""

    config = _absolute(config_path)
    values = _load_config(config)
    data = _mapping(values.get("data"), "data")
    manifest = _absolute(_required_string(data, "manifest"), base=config.parent)
    if not manifest.is_file():
        raise FileNotFoundError(f"SOPAT source manifest is missing: {manifest}")
    destination = _absolute(output)
    paired_root = _absolute(paired_index_root or destination.parent / "paired_indexes")
    publication_path = _absolute(publication or destination.parent / "index_publication.json")
    if not force and _publication_is_valid(
        publication_path,
        config=config,
        manifest=manifest,
        output=destination,
        paired_root=paired_root,
    ):
        payload = _read_json(publication_path)
        assert payload is not None
        return {"reused": True, "publication": payload}

    # Running via ``python scripts/...`` places ``scripts`` on sys.path,
    # whereas importlib-based unit tests do not.  Keep this local CLI import
    # out of package installation concerns.
    script_directory = Path(__file__).resolve().parent
    if str(script_directory) not in sys.path:
        sys.path.insert(0, str(script_directory))
    from filter_sopat_v4_center_evaluable import filter_validation_index

    selected = _configured_indexes(data, manifest=manifest)
    validation_split = _required_string(data, "validation_split")
    crop_size = _required_int(data, "crop_size")
    maximum_observations = _required_int(data, "maximum_observations")
    filter_report: dict[str, object] = {
        "protocol": "sopat_v4_fixed_center_evaluable_v2",
        "crop_size": crop_size,
        "train_pixel_filtered": False,
        "directions": {},
    }
    final_indexes: dict[tuple[str, str], PairedTemporalIndex] = {}
    for (direction, split), candidate in selected.items():
        if split != validation_split:
            final_indexes[(direction, split)] = candidate
            continue
        filtered, dropped = filter_validation_index(
            manifest,
            candidate,
            crop_size=crop_size,
            maximum_observations=maximum_observations,
        )
        if len(filtered) < 2:
            raise RuntimeError(
                f"fixed-center validation has fewer than two samples: {direction}/{split}"
            )
        final_indexes[(direction, split)] = filtered
        filter_report["directions"] = {
            **_mapping(filter_report["directions"], "filter directions"),
            direction: {
                "input_samples": len(candidate),
                "output_samples": len(filtered),
                "dropped_sample_ids": list(dropped),
            },
        }
    index = _combine_migrated_indexes(final_indexes, manifest=manifest)

    stage_token = uuid.uuid4().hex
    stage_index = destination.with_name(f".{destination.name}.{stage_token}.tmp")
    stage_paired_root = paired_root.with_name(f".{paired_root.name}.{stage_token}.tmp")
    try:
        write_sopat_v4_index(stage_index, index)
        for (direction, split), legacy in final_indexes.items():
            write_paired_temporal_index(stage_paired_root / direction / f"{split}.jsonl", legacy)
        payload = _publication_payload(
            config=config,
            manifest=manifest,
            output=destination,
            paired_root=paired_root,
            index=index,
            indexes=final_indexes,
            staged_index=stage_index,
            staged_paired_root=stage_paired_root,
            filter_report=filter_report,
        )
        # Publish every legacy cache index before the role index.  The final
        # publication marker is written last, so launchers only accept a fully
        # hash-bound set even if the process is interrupted between replaces.
        for direction, split in sorted(final_indexes):
            final_path = paired_root / direction / f"{split}.jsonl"
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage_paired_root / direction / f"{split}.jsonl", final_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_index, destination)
        _atomic_json(publication_path, payload)
    finally:
        stage_index.unlink(missing_ok=True)
        if stage_paired_root.exists():
            for path in sorted(stage_paired_root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            stage_paired_root.rmdir()
    return {"reused": False, "publication": payload}


def main() -> None:
    args = _parser().parse_args()
    result = build_and_publish(
        args.config,
        args.output,
        paired_index_root=args.paired_index_root,
        publication=args.publication,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
