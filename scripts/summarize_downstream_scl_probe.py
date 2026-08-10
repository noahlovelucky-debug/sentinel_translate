"""Validate and merge six downstream SCL proxy group reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from sentinel_v3.downstream_probe import (
    PROBE_GROUPS,
    cache_contract,
    summarize_probe_statistics,
)

CANONICAL_CHECKPOINT_SHA256 = "5c26e96ee639609624d350f4ab4eff272a94b9f799fc9ce51579ee1420881363"
EXPECTED_SEEDS = (13, 17, 29)
EXPECTED_PROTOCOL: dict[str, object] = {
    "augment": True,
    "optimizer": "AdamW",
    "epochs": 12,
    "steps_per_epoch": 100,
    "batch_size": 8,
    "eval_batch_size": 16,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "width": 16,
    "mixed_real_probability": 0.5,
    "mixed_evaluation_route": "sar_synthetic_optical",
}


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _read_json(path: Path, name: str) -> dict[str, object]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), name)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {name}: {path}: {error}") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expect_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} does not match the registered protocol")


def _configured_cache_root(config_path: Path) -> tuple[Path, str, str]:
    try:
        config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"cannot read cache config {config_path}: {error}") from error
    paths = _mapping(config.get("paths"), "config.paths")
    cache_root_value = paths.get("cache_root")
    if not isinstance(cache_root_value, str) or not cache_root_value:
        raise RuntimeError("config.paths.cache_root must be a non-empty string")
    checkpoint = _mapping(paths.get("checkpoint"), "config.paths.checkpoint")
    checkpoint_sha256 = checkpoint.get("sha256")
    if checkpoint_sha256 != CANONICAL_CHECKPOINT_SHA256:
        raise RuntimeError("config does not bind the canonical best_physical SHA-256")
    checkpoint_path = checkpoint.get("path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise RuntimeError("config.paths.checkpoint.path must be a non-empty string")
    return Path(cache_root_value).resolve(), str(checkpoint_sha256), _file_sha256(config_path)


def expected_cache_provenance(config_path: Path, materialized_root: Path) -> dict[str, str]:
    """Validate the immutable cache chain and return the required report stamp."""

    cache_root, checkpoint_sha256, config_sha256 = _configured_cache_root(config_path)
    root = materialized_root.resolve()
    if root != (cache_root / "materialized").resolve():
        raise RuntimeError("materialized root does not belong to config.paths.cache_root")
    materialized_provenance = _read_json(root / "provenance.json", "materialized provenance")
    materialized_manifest_path = root / "manifest.json"
    materialized_manifest = _read_json(materialized_manifest_path, "materialized manifest")
    materialized_provenance_sha256 = _canonical_json_sha256(materialized_provenance)
    _expect_equal(
        materialized_manifest.get("materialized_provenance_sha256"),
        materialized_provenance_sha256,
        "materialized manifest provenance",
    )
    _expect_equal(
        materialized_provenance.get("source_cache_root"),
        str(cache_root),
        "materialized source cache root",
    )
    _expect_equal(
        materialized_provenance.get("source_config_sha256"),
        config_sha256,
        "materialized source config SHA-256",
    )
    _expect_equal(
        materialized_provenance.get("source_checkpoint_sha256"),
        checkpoint_sha256,
        "materialized source checkpoint SHA-256",
    )
    _expect_equal(
        materialized_provenance.get("probe_cache_contract"),
        cache_contract(),
        "materialized probe cache contract",
    )

    source_provenance = _read_json(cache_root / "provenance.json", "source cache provenance")
    source_provenance_sha256 = _canonical_json_sha256(source_provenance)
    source_manifest_path = cache_root / "cache_manifest.json"
    source_manifest_sha256 = _file_sha256(source_manifest_path)
    _expect_equal(
        materialized_provenance.get("source_cache_provenance_sha256"),
        source_provenance_sha256,
        "source cache provenance SHA-256",
    )
    _expect_equal(
        materialized_provenance.get("source_cache_manifest_sha256"),
        source_manifest_sha256,
        "source cache manifest SHA-256",
    )
    _expect_equal(
        source_provenance.get("config_sha256"),
        config_sha256,
        "source cache config SHA-256",
    )
    _expect_equal(
        source_provenance.get("checkpoint_sha256"),
        checkpoint_sha256,
        "source cache checkpoint SHA-256",
    )
    return {
        "materialized_provenance_sha256": materialized_provenance_sha256,
        "materialized_manifest_sha256": _file_sha256(materialized_manifest_path),
        "source_cache_provenance_sha256": source_provenance_sha256,
        "source_cache_manifest_sha256": source_manifest_sha256,
        "source_config_sha256": config_sha256,
        "source_checkpoint_sha256": checkpoint_sha256,
    }


def _as_seed_tuple(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a seed sequence")
    try:
        return tuple(int(seed) for seed in value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} contains an invalid seed") from error


def _finite_score(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a numeric macroIoU or null")
    score = float(value)
    if not math.isfinite(score):
        raise RuntimeError(f"{name} must be finite when present")
    return score


@dataclass(frozen=True)
class ValidatedGroupReport:
    group: str
    path: Path
    stats: dict[str, object]
    protocol: dict[str, object]
    parameter_count: int
    scene_scores: dict[str, float]


def _single_group_name(payload: Mapping[str, object], path: Path) -> str:
    suite = _mapping(payload.get("suite"), f"suite in {path}")
    groups = _mapping(suite.get("groups"), f"suite.groups in {path}")
    if len(groups) != 1:
        raise RuntimeError(f"{path} must contain exactly one selected probe group")
    group = next(iter(groups))
    if group not in PROBE_GROUPS:
        raise RuntimeError(f"{path} has an unknown probe group {group!r}")
    selected = payload.get("groups")
    if selected != [group]:
        raise RuntimeError(f"{path} top-level group selection does not match suite.groups")
    return group


def validate_group_report(
    path: Path,
    cache_provenance: Mapping[str, str],
    *,
    expected_group: str | None = None,
    require_stamp: bool,
) -> ValidatedGroupReport:
    """Validate one serialized group run and recover paired test scene scores."""

    payload = _read_json(path, "group report")
    if payload.get("format_version") != 1:
        raise RuntimeError(f"{path} has an unsupported group report format")
    _expect_equal(payload.get("cache_contract"), cache_contract(), f"cache contract in {path}")
    _expect_equal(_as_seed_tuple(payload.get("seeds"), f"seeds in {path}"), EXPECTED_SEEDS, "seeds")
    group = _single_group_name(payload, path)
    if expected_group is not None and group != expected_group:
        raise RuntimeError(f"{path} belongs to {group!r}, not expected group {expected_group!r}")
    if require_stamp:
        _expect_equal(payload.get("cache_provenance"), dict(cache_provenance), f"cache provenance in {path}")

    suite = _mapping(payload.get("suite"), f"suite in {path}")
    stats = _mapping(suite.get("stats"), f"suite.stats in {path}")
    protocol = _mapping(suite.get("protocol"), f"suite.protocol in {path}")
    for key, expected in EXPECTED_PROTOCOL.items():
        _expect_equal(protocol.get(key), expected, f"protocol.{key} in {path}")
    groups = _mapping(suite.get("groups"), f"suite.groups in {path}")
    raw_results = groups[group]
    if not isinstance(raw_results, list) or len(raw_results) != len(EXPECTED_SEEDS):
        raise RuntimeError(f"{path} must contain exactly three seed results")

    expected_evaluation_input = "sar_synthetic_optical" if group == "sar_mixed_optical" else group
    seed_scene_scores: list[dict[str, float]] = []
    parameter_counts: set[int] = set()
    actual_seeds: list[int] = []
    for result_index, raw_result in enumerate(raw_results):
        result = _mapping(raw_result, f"seed result {result_index} in {path}")
        _expect_equal(result.get("group"), group, f"result group in {path}")
        _expect_equal(
            result.get("evaluation_input_group"),
            expected_evaluation_input,
            f"evaluation route in {path}",
        )
        seed = result.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed result {result_index} in {path} has an invalid seed")
        actual_seeds.append(seed)
        count = result.get("parameter_count")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"seed result {result_index} in {path} has an invalid parameter count")
        if count <= 0:
            raise RuntimeError(f"seed result {result_index} in {path} has an invalid parameter count")
        parameter_counts.add(count)
        evaluations = _mapping(result.get("evaluations"), f"evaluations in {path}")
        test = _mapping(evaluations.get("test"), f"test evaluation in {path}")
        per_scene = _mapping(test.get("per_scene"), f"test per_scene in {path}")
        scores: dict[str, float] = {}
        for scene_id, raw_metrics in per_scene.items():
            metrics = _mapping(raw_metrics, f"metrics for {scene_id!r} in {path}")
            score = _finite_score(metrics.get("macro_iou"), f"macro_iou for {scene_id!r} in {path}")
            if score is not None:
                scores[scene_id] = score
        if not scores:
            raise RuntimeError(f"{path} has no finite test scene macroIoU values")
        seed_scene_scores.append(scores)
    _expect_equal(tuple(actual_seeds), EXPECTED_SEEDS, f"seed order in {path}")
    if len(parameter_counts) != 1:
        raise RuntimeError(f"{path} changes parameter count between seeds")
    reference_scenes = set(seed_scene_scores[0])
    if any(set(scores) != reference_scenes for scores in seed_scene_scores[1:]):
        raise RuntimeError(f"{path} does not evaluate the same finite test scenes for every seed")
    scene_scores = {
        scene_id: sum(scores[scene_id] for scores in seed_scene_scores) / len(seed_scene_scores)
        for scene_id in sorted(reference_scenes)
    }
    return ValidatedGroupReport(
        group=group,
        path=path,
        stats=stats,
        protocol=protocol,
        parameter_count=next(iter(parameter_counts)),
        scene_scores=scene_scores,
    )


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stamp_group_report(path: Path, cache_provenance: Mapping[str, str], expected_group: str) -> None:
    """Attach the cache chain after validating a freshly written single-group report."""

    validate_group_report(
        path,
        cache_provenance,
        expected_group=expected_group,
        require_stamp=False,
    )
    payload = _read_json(path, "group report")
    payload["cache_provenance"] = dict(cache_provenance)
    payload["provenance_stamp_version"] = 1
    _atomic_json(path, payload)


def merge_reports(
    reports: Sequence[ValidatedGroupReport], cache_provenance: Mapping[str, str]
) -> dict[str, object]:
    if len(reports) != len(PROBE_GROUPS):
        raise RuntimeError("final summary requires exactly six group reports")
    by_group = {report.group: report for report in reports}
    if set(by_group) != set(PROBE_GROUPS):
        raise RuntimeError("final summary must contain every required probe group exactly once")
    reference = reports[0]
    reference_stats = _canonical_json_sha256(reference.stats)
    reference_protocol = _canonical_json_sha256(reference.protocol)
    parameter_counts = {report.parameter_count for report in reports}
    for report in reports[1:]:
        if _canonical_json_sha256(report.stats) != reference_stats:
            raise RuntimeError(f"stats differ across reports; refusing to merge {report.path}")
        if _canonical_json_sha256(report.protocol) != reference_protocol:
            raise RuntimeError(f"protocol differs across reports; refusing to merge {report.path}")
    if len(parameter_counts) != 1:
        raise RuntimeError("parameter count differs between probe groups")
    scene_scores = {group: by_group[group].scene_scores for group in PROBE_GROUPS}
    statistics = summarize_probe_statistics(scene_scores)
    return {
        "format_version": 1,
        "cache_provenance": dict(cache_provenance),
        "stats": reference.stats,
        "protocol": reference.protocol,
        "parameter_count": parameter_counts.pop(),
        "seeds": list(EXPECTED_SEEDS),
        "groups": {
            group: {
                "report": str(by_group[group].path.resolve()),
                "scene_scores": by_group[group].scene_scores,
            }
            for group in PROBE_GROUPS
        },
        "scene_scores": scene_scores,
        "statistics": statistics,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="downstream_scl_proxy YAML")
    parser.add_argument("--materialized-root", required=True)
    parser.add_argument("--report", action="append", required=True, help="group JSON report")
    parser.add_argument("--output", help="final merged JSON destination")
    parser.add_argument("--expected-group", choices=PROBE_GROUPS)
    parser.add_argument("--stamp", action="store_true", help="stamp one freshly written group report")
    parser.add_argument("--verify-only", action="store_true", help="validate report(s) without writing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stamp and args.verify_only:
        raise ValueError("--stamp and --verify-only are mutually exclusive")
    config_path = Path(args.config).resolve()
    materialized_root = Path(args.materialized_root).resolve()
    cache_provenance = expected_cache_provenance(config_path, materialized_root)
    report_paths = [Path(path).resolve() for path in args.report]
    if args.stamp:
        if len(report_paths) != 1 or args.expected_group is None or args.output is not None:
            raise ValueError("--stamp requires one --report and --expected-group, without --output")
        stamp_group_report(report_paths[0], cache_provenance, args.expected_group)
        print(json.dumps({"stamped": str(report_paths[0]), "group": args.expected_group}))
        return

    if args.expected_group is not None and len(report_paths) != 1:
        raise ValueError("--expected-group is only valid with one --report")
    validated = [
        validate_group_report(
            path,
            cache_provenance,
            expected_group=args.expected_group,
            require_stamp=True,
        )
        for path in report_paths
    ]
    if args.verify_only:
        if args.output is not None:
            raise ValueError("--verify-only does not accept --output")
        print(json.dumps({"verified_reports": [str(path) for path in report_paths]}))
        return
    if args.output is None:
        raise ValueError("--output is required when merging reports")
    _atomic_json(Path(args.output), merge_reports(validated, cache_provenance))


if __name__ == "__main__":
    main()
