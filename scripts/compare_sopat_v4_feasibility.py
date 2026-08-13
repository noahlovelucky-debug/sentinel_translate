"""Compare SOPAT V4 with explicit causal V2 checkpoint adapters.

The comparison deliberately uses the SOPAT V4 validation loaders, whose
validation crops are fixed at the center.  The V2 route receives the complete
V4 source observation set and the same registered anchor pair.  ``latest``
in its result name refers to the caller-selected V2 checkpoint, never to a
frame-selection operation.  V3.2 checkpoints may be recorded as an
input-mismatched reference only and are never loaded here.

Ground-truth targets are read only after all model forwards through the V4
causal whitelist have completed.  This script is evaluation-only and never
starts or resumes training.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from sentinel_v3.paired_temporal_training import (
    PAIRED_TEMPORAL_STAGES,
    load_paired_temporal_checkpoint,
)
from sentinel_v3.paired_temporal_v2 import PairedTemporalConfig, SparsePairedAnchorTransport
from sentinel_v4.evaluation import SOPATVariantConfig, evaluate_sopat_loaders
from sentinel_v4.model import SOPAT
from sentinel_v4.training import (
    DIRECTIONS,
    ModelEMA,
    SOPATTrainConfig,
    forward_direction,
    forward_input_tensors,
    load_sopat_checkpoint,
    output_tensor,
    supervision_tensors,
)

# Keep preparation and loader creation on the exact V4 evaluator path.  Direct
# script execution exposes this directory; module-style test execution does not.
try:
    from train_sopat_v4 import (  # type: ignore[import-not-found]
        _datasets,
        _device_generator,
        _load_config,
        _loaders,
        _model_config,
        _prepare_data_on_rank_zero,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by module invocation
    from scripts.train_sopat_v4 import (
        _datasets,
        _device_generator,
        _load_config,
        _loaders,
        _model_config,
        _prepare_data_on_rank_zero,
    )


_LOWER_IS_BETTER = frozenset(
    {
        "rmse",
        "mae",
        "bias",
        "pre_projection_violation",
        "psd_log_l1",
        "sam_deg",
        "ndvi_mae",
        "ndvi_bias",
        "sar_db_rmse",
        "sar_db_mae",
        "sar_db_bias",
        "sar_psd_log_l1",
    }
)
_HIGHER_IS_BETTER = frozenset({"edge_f1", "scene_improved_fraction", "sar_db_corr"})


@dataclass(frozen=True)
class V2CheckpointOutput:
    """Minimal physical output accepted by the V4 evaluator."""

    physical: Tensor
    pre_projection_violation: Tensor | None


class V2CheckpointAdapter(nn.Module):
    """Expose one old V2 checkpoint through the target-free V4 contract.

    This adapter intentionally does no temporal resampling, frame selection,
    or padding cleanup.  It passes the complete V4 BxT source set, its masks,
    timestamps, and availability flags to V2 so both learned routes see the
    same causal source evidence and registered anchors.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        *,
        observations: Tensor,
        observation_valid: Tensor,
        observation_days: Tensor,
        observation_present: Tensor,
        source_anchor: Tensor,
        source_anchor_valid: Tensor,
        target_anchor: Tensor,
        target_anchor_valid: Tensor,
        source_anchor_days: Tensor,
        target_anchor_days: Tensor,
        source_sensor: object,
        target_sensor: object,
    ) -> V2CheckpointOutput:
        if observations.ndim != 5:
            raise ValueError("V2 checkpoint adapter observations must have shape BxTxCxHxW")
        batch, frames, _, height, width = observations.shape
        if frames < 1 or observation_days.shape != (batch, frames):
            raise ValueError("V2 checkpoint adapter observation_days does not match observations")
        if observation_present.shape != (batch, frames):
            raise ValueError("V2 checkpoint adapter observation_present does not match observations")
        if observation_valid.shape != (batch, frames, 1, height, width):
            raise ValueError("V2 checkpoint adapter observation_valid does not match observations")
        if bool((observation_present.bool().sum(dim=1) < 1).any()):
            raise ValueError("V2 checkpoint adapter requires at least one present observation per sample")
        physical = getattr(self.model, "physical", None)
        if not callable(physical):
            raise TypeError("V2 checkpoint adapter model must expose a physical method")
        result = physical(
            observations,
            observation_valid,
            observation_days,
            observation_present,
            source_anchor,
            source_anchor_valid,
            target_anchor,
            target_anchor_valid,
            source_sensor=source_sensor,
            target_sensor=target_sensor,
            source_anchor_days=source_anchor_days,
            target_anchor_days=target_anchor_days,
        )
        values = output_tensor(result, "physical")
        violation = output_tensor(result, "pre_projection_violation", required=False)
        assert values is not None
        return V2CheckpointOutput(values, violation)


class BidirectionalV2CheckpointAdapter(nn.Module):
    """Select one direction-specific V2 checkpoint from the causal sensors."""

    def __init__(self, sar_to_optical: V2CheckpointAdapter, optical_to_sar: V2CheckpointAdapter) -> None:
        super().__init__()
        self.sar_to_optical = sar_to_optical
        self.optical_to_sar = optical_to_sar

    def forward(
        self,
        *,
        observations: Tensor,
        observation_valid: Tensor,
        observation_days: Tensor,
        observation_present: Tensor,
        source_anchor: Tensor,
        source_anchor_valid: Tensor,
        target_anchor: Tensor,
        target_anchor_valid: Tensor,
        source_anchor_days: Tensor,
        target_anchor_days: Tensor,
        source_sensor: object,
        target_sensor: object,
    ) -> V2CheckpointOutput:
        source_name = getattr(source_sensor, "name", None)
        target_name = getattr(target_sensor, "name", None)
        if (source_name, target_name) == ("sentinel-1", "sentinel-2"):
            adapter = self.sar_to_optical
        elif (source_name, target_name) == ("sentinel-2", "sentinel-1"):
            adapter = self.optical_to_sar
        else:
            raise ValueError(
                "V2 checkpoint adapter only supports the canonical Sentinel-1/Sentinel-2 directions"
            )
        return adapter(
            observations=observations,
            observation_valid=observation_valid,
            observation_days=observation_days,
            observation_present=observation_present,
            source_anchor=source_anchor,
            source_anchor_valid=source_anchor_valid,
            target_anchor=target_anchor,
            target_anchor_valid=target_anchor_valid,
            source_anchor_days=source_anchor_days,
            target_anchor_days=target_anchor_days,
            source_sensor=source_sensor,
            target_sensor=target_sensor,
        )


def load_v2_checkpoint_adapter(
    checkpoint: str | Path,
    *,
    direction: str,
    device: torch.device,
) -> tuple[V2CheckpointAdapter, dict[str, object]]:
    """Instantiate a V2 model from its own strict direction-bound checkpoint."""

    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("V2 checkpoint payload must be a mapping")
    model_values = payload.get("model_config")
    if not isinstance(model_values, Mapping):
        raise TypeError("V2 checkpoint is missing model_config")
    model_config = PairedTemporalConfig(**dict(model_values))
    model = SparsePairedAnchorTransport(model_config)
    loaded = load_paired_temporal_checkpoint(
        path,
        model,
        direction=direction,
        allowed_stages=PAIRED_TEMPORAL_STAGES,
    )
    adapter = V2CheckpointAdapter(model).to(device)
    adapter.eval()
    return adapter, {
        "checkpoint": str(path),
        "checkpoint_step": int(loaded.get("step", 0)),
        "checkpoint_stage": str(loaded.get("stage", "unknown")),
        "checkpoint_direction": str(loaded.get("direction", "unknown")),
        "model_config": asdict(model_config),
        "input_adapter": {
            "name": "v2_full_v4_observation_set",
            "checkpoint_label": "latest",
            "uses_full_v4_observation_set": True,
            "uses_registered_v4_anchor_pair": True,
            "target_label_forwarded": False,
        },
    }


def _device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_train_config(path: Path) -> SOPATTrainConfig:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("SOPAT V4 checkpoint payload must be a mapping")
    values = payload.get("train_config")
    if not isinstance(values, Mapping):
        raise TypeError("SOPAT V4 checkpoint is missing train_config")
    return SOPATTrainConfig.from_mapping(values)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_sample_id(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return safe[:96] or "sample"


def _sample_ids(batch: Mapping[str, object], batch_size: int) -> list[str]:
    values = batch.get("sopat_example_id", batch.get("sample_id"))
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        result = [str(value) for value in values]
        if len(result) == batch_size:
            return result
    if isinstance(values, str):
        return [values] * batch_size
    return [f"sample-{index:06d}" for index in range(batch_size)]


def _task_modes(batch: Mapping[str, object], batch_size: int) -> list[str]:
    values = batch.get("task_mode")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        result = [str(value) for value in values]
        if len(result) == batch_size:
            return result
    if isinstance(values, str):
        return [values] * batch_size
    return ["unknown"] * batch_size


def _observation_count_bin(value: int) -> str:
    if value <= 0:
        raise ValueError("comparison panel requires at least one source observation")
    if value == 1:
        return "one"
    if value <= 3:
        return "two_to_three"
    return "four_plus"


@contextlib.contextmanager
def _evaluation_mode(*models: nn.Module) -> Iterator[None]:
    states = [model.training for model in models]
    for model in models:
        model.eval()
    try:
        yield
    finally:
        for model, was_training in zip(models, states, strict=True):
            model.train(was_training)


def export_feasibility_panel_payloads(
    v4_model: nn.Module,
    v2_model: nn.Module,
    loaders: Mapping[str, Iterable[Mapping[str, object]]],
    output_root: str | Path,
    *,
    device: torch.device | None,
    limit_per_direction: int = 16,
) -> dict[str, object]:
    """Write shared-batch V2/V4 payloads for the separate honest renderer.

    Targets are deliberately fetched only after both target-free calls to
    ``forward_direction``.  The payload preserves normalized tensors rather
    than making a color interpretation at evaluation time.
    """

    if set(loaders) != set(DIRECTIONS):
        raise ValueError("feasibility panel export requires both V4 direction loaders")
    if limit_per_direction <= 0:
        raise ValueError("feasibility panel export limit_per_direction must be positive")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    with _evaluation_mode(v4_model, v2_model):
        for direction in DIRECTIONS:
            written = 0
            for batch in loaders[direction]:
                if not isinstance(batch, Mapping):
                    raise TypeError("feasibility panel loader must yield mapping batches")
                with torch.inference_mode():
                    # Do not move or inspect target labels until these two causal
                    # model calls have returned.  This is intentionally more
                    # explicit than relying on a convention at each call site.
                    v4_output = forward_direction(v4_model, batch, direction, device=device)
                    v2_output = forward_direction(v2_model, batch, direction, device=device)
                    v4_physical = output_tensor(v4_output, "physical")
                    v2_physical = output_tensor(v2_output, "physical")
                    assert v4_physical is not None
                    assert v2_physical is not None
                    inputs = forward_input_tensors(batch, device=device)
                    labels = supervision_tensors(batch, device=device)
                    if v4_physical.shape != labels["target"].shape:
                        raise ValueError("V4 panel prediction does not match its target tensor")
                    if v2_physical.shape != labels["target"].shape:
                        raise ValueError("V2 checkpoint panel prediction does not match its target tensor")
                    valid = labels["target_valid"] * inputs["target_anchor_valid"].to(
                        labels["target_valid"]
                    )
                    sample_ids = _sample_ids(batch, labels["target"].shape[0])
                    task_modes = _task_modes(batch, labels["target"].shape[0])
                    observation_counts = inputs["observation_present"].sum(dim=1)
                    for sample_index in range(labels["target"].shape[0]):
                        if written >= limit_per_direction:
                            break
                        filename = (
                            f"{direction}_{written:03d}_{_safe_sample_id(sample_ids[sample_index])}.pt"
                        )
                        destination = root / filename
                        item = {
                            "family": "sopat_v4_feasibility_panel",
                            "schema_version": 1,
                            "direction": direction,
                            "sample_id": sample_ids[sample_index],
                            "task_mode": task_modes[sample_index],
                            "observation_count_bin": _observation_count_bin(
                                int(observation_counts[sample_index])
                            ),
                            "normalization": "[-1, 1]",
                            "target_label_forwarded": False,
                            "source_anchor": inputs["source_anchor"][sample_index].detach().cpu(),
                            "source_valid": inputs["source_anchor_valid"][sample_index].detach().cpu(),
                            "target_anchor": inputs["target_anchor"][sample_index].detach().cpu(),
                            "v2_latest_checkpoint": v2_physical[sample_index].detach().cpu(),
                            "v4_ema": v4_physical[sample_index].detach().cpu(),
                            "target": labels["target"][sample_index].detach().cpu(),
                            "valid": valid[sample_index].detach().cpu(),
                        }
                        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
                        torch.save(item, temporary)
                        os.replace(temporary, destination)
                        entries.append(
                            {
                                "direction": direction,
                                "sample_id": sample_ids[sample_index],
                                "task_mode": task_modes[sample_index],
                                "observation_count_bin": item["observation_count_bin"],
                                "file": filename,
                            }
                        )
                        written += 1
                if written >= limit_per_direction:
                    break
    manifest = {
        "family": "sopat_v4_feasibility_panel",
        "schema_version": 1,
        "normalization": "[-1, 1]",
        "target_label_forwarded": False,
        "entries": entries,
    }
    _write_json(root / "panel_manifest.json", manifest)
    return manifest


def _metric_value(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_comparison(
    candidate: Mapping[str, object], baseline: Mapping[str, object]
) -> dict[str, object]:
    """Compute positive-is-better improvement values for one metric bucket."""

    result: dict[str, object] = {}
    for name in sorted(_LOWER_IS_BETTER | _HIGHER_IS_BETTER):
        candidate_value = _metric_value(candidate.get(name))
        baseline_value = _metric_value(baseline.get(name))
        if candidate_value is None or baseline_value is None:
            continue
        lower_is_better = name in _LOWER_IS_BETTER
        candidate_score = abs(candidate_value) if name.endswith("bias") else candidate_value
        baseline_score = abs(baseline_value) if name.endswith("bias") else baseline_value
        improvement = (
            baseline_score - candidate_score
            if lower_is_better
            else candidate_score - baseline_score
        )
        result[name] = {
            "candidate": candidate_value,
            "baseline": baseline_value,
            "absolute_improvement": improvement,
            "relative_improvement": (
                improvement / abs(baseline_score) if abs(baseline_score) > 1e-12 else None
            ),
            "positive_means_better": True,
            "metric_orientation": "lower_is_better" if lower_is_better else "higher_is_better",
        }
    return result


def _relative_tree(candidate: object, baseline: object) -> object:
    if not isinstance(candidate, Mapping) or not isinstance(baseline, Mapping):
        return {}
    if "samples" in candidate and "samples" in baseline:
        return _metric_comparison(candidate, baseline)
    return {
        str(name): _relative_tree(candidate[name], baseline[name])
        for name in sorted(set(candidate).intersection(baseline))
        if isinstance(name, str)
    }


def relative_improvements(
    candidate_report: Mapping[str, object], baseline_report: Mapping[str, object]
) -> dict[str, object]:
    """Compare all common V4 evaluator direction/task/N/change buckets."""

    candidate_directions = candidate_report.get("directions")
    baseline_directions = baseline_report.get("directions")
    if not isinstance(candidate_directions, Mapping) or not isinstance(baseline_directions, Mapping):
        raise TypeError("comparison reports must contain V4 evaluator directions")
    return {
        "definition": "positive relative_improvement means the candidate is better",
        "directions": _relative_tree(candidate_directions, baseline_directions),
    }


def _method_result(
    report: Mapping[str, object],
    *,
    input_protocol: Mapping[str, object],
) -> dict[str, object]:
    directions = report.get("directions")
    if not isinstance(directions, Mapping):
        raise TypeError("V4 evaluator report has no directions")
    return {
        "variant": report.get("variant"),
        "changed_threshold_normalized": report.get("changed_threshold_normalized"),
        "input_protocol": dict(input_protocol),
        "directions": dict(directions),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="SOPAT V4 data/model configuration")
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--v2-sar-to-optical-checkpoint", type=Path, required=True)
    parser.add_argument("--v2-optical-to-sar-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--v3-2-best-reference",
        "--v3-best-reference",
        dest="v3_2_best_reference",
        type=Path,
        help="Recorded only as an input-mismatched reference; never evaluated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--panel-samples", type=int, default=16)
    parser.add_argument("--changed-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def evaluate(
    config: Mapping[str, Any],
    *,
    v4_checkpoint: Path,
    v2_sar_to_optical_checkpoint: Path,
    v2_optical_to_sar_checkpoint: Path,
    output: Path,
    config_base: Path,
    device: torch.device,
    limit_batches: int | None,
    panel_samples: int,
    changed_threshold: float,
    seed: int,
    v3_2_best_reference: Path | None = None,
) -> dict[str, object]:
    """Evaluate all declared routes on one fixed-center SOPAT V4 protocol."""

    if limit_batches is not None and limit_batches <= 0:
        raise ValueError("--limit-batches must be positive")
    if panel_samples <= 0:
        raise ValueError("--panel-samples must be positive")
    if not math.isfinite(changed_threshold) or changed_threshold < 0.0:
        raise ValueError("--changed-threshold must be finite and non-negative")
    prepared = _prepare_data_on_rank_zero(
        config,
        output=output / "prepared",
        config_base=config_base,
    )
    model_config = _model_config(config)
    train_config = _checkpoint_train_config(v4_checkpoint)
    v4_model = SOPAT(model_config).to(device)
    ema = ModelEMA.create(v4_model, train_config.ema_decay)
    v4_payload = load_sopat_checkpoint(
        v4_checkpoint,
        model=v4_model,
        optimizer=None,
        ema=ema,
        scheduler=None,
        model_config=asdict(model_config),
        train_config=train_config,
        protocol_hashes=prepared.protocol_hashes,
        restore_rng=False,
    )
    v2_sar_to_optical, v2_sar_to_optical_metadata = load_v2_checkpoint_adapter(
        v2_sar_to_optical_checkpoint,
        direction="sar_to_optical",
        device=device,
    )
    v2_optical_to_sar, v2_optical_to_sar_metadata = load_v2_checkpoint_adapter(
        v2_optical_to_sar_checkpoint,
        direction="optical_to_sar",
        device=device,
    )
    v2_model = BidirectionalV2CheckpointAdapter(v2_sar_to_optical, v2_optical_to_sar).to(device)

    train_datasets, validation_datasets = _datasets(config, prepared, seed=seed, stage="physical")
    del train_datasets
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("SOPAT feasibility configuration requires training mapping")
    validation = config.get("validation")
    if not isinstance(validation, Mapping):
        raise TypeError("SOPAT feasibility configuration requires validation mapping")
    batch_size = int(validation.get("batch_size", training.get("batch_size", 1)))
    if batch_size < 2:
        raise ValueError("SOPAT validation.batch_size must be at least 2 for source_shuffle")
    num_workers = int(training.get("num_workers", 4))
    validation_loaders = _loaders(
        validation_datasets,
        batch_size=batch_size,
        num_workers=max(0, min(num_workers, 2)),
        device=device,
        rank=0,
        world_size=1,
        training=False,
        seed=seed,
    )

    anchor_copy_report = evaluate_sopat_loaders(
        None,
        validation_loaders,
        variant=SOPATVariantConfig(name="anchor_copy", seed=seed),
        device=device,
        changed_threshold=changed_threshold,
        limit_batches=limit_batches,
    )
    v2_report = evaluate_sopat_loaders(
        v2_model,
        validation_loaders,
        variant=SOPATVariantConfig(name="sopat", seed=seed),
        device=device,
        changed_threshold=changed_threshold,
        limit_batches=limit_batches,
    )
    v2_report["variant"] = {
        "name": "v2_latest_checkpoint",
        "route_kind": "old_paired_temporal_full_set_checkpoint_adapter",
        "external_reproduction": False,
    }
    with ema.average_parameters(v4_model):
        v4_report = evaluate_sopat_loaders(
            v4_model,
            validation_loaders,
            variant=SOPATVariantConfig(name="sopat", seed=seed),
            device=device,
            changed_threshold=changed_threshold,
            limit_batches=limit_batches,
        )
        source_shuffle_report = evaluate_sopat_loaders(
            v4_model,
            validation_loaders,
            variant=SOPATVariantConfig(name="source_shuffle", seed=seed),
            device=device,
            changed_threshold=changed_threshold,
            limit_batches=limit_batches,
            generator=_device_generator(device, seed + 1_000_003),
        )
        panel_manifest = export_feasibility_panel_payloads(
            v4_model,
            v2_model,
            validation_loaders,
            output / "panel_payloads",
            device=device,
            limit_per_direction=panel_samples,
        )
    v4_report["variant"] = {
        "name": "v4_ema",
        "route_kind": "trained_sopat_ema",
        "external_reproduction": False,
    }

    methods = {
        "anchor_copy": _method_result(
            anchor_copy_report,
            input_protocol={
                "name": "registered_target_anchor_copy",
                "uses_full_v4_observation_set": False,
                "target_label_forwarded": False,
            },
        ),
        "v4_ema": _method_result(
            v4_report,
            input_protocol={
                "name": "sopat_v4_ema",
                "uses_full_v4_observation_set": True,
                "target_label_forwarded": False,
            },
        ),
        "v2_latest_checkpoint": _method_result(
            v2_report,
            input_protocol={
                "name": "v2_full_v4_observation_set",
                "checkpoint_label": "latest",
                "uses_full_v4_observation_set": True,
                "uses_registered_v4_anchor_pair": True,
                "target_label_forwarded": False,
            },
        ),
        "source_shuffle": _method_result(
            source_shuffle_report,
            input_protocol={
                "name": "v4_source_shuffle_counterfactual",
                "shuffle_scope": "within_validation_batch",
                "target_label_forwarded": False,
            },
        ),
    }
    result: dict[str, object] = {
        "family": "sopat_v4_feasibility_comparison",
        "schema_version": 1,
        "canonical_grid": {"gsd_m": 10.0, "claim": "canonical_10m_only"},
        "validation_protocol": {
            "source": "sopat_v4_validation_loaders",
            "crop": "fixed_center",
            "target_label_forwarded_to_any_model": False,
            "changed_threshold_normalized": changed_threshold,
        },
        "checkpoints": {
            "v4_ema": {
                "checkpoint": str(v4_checkpoint),
                "checkpoint_global_step": int(v4_payload.get("global_step", 0)),
                "checkpoint_train_stage": train_config.stage,
                "protocol_hashes": prepared.protocol_hashes,
            },
            "v2_latest_checkpoint": {
                "sar_to_optical": v2_sar_to_optical_metadata,
                "optical_to_sar": v2_optical_to_sar_metadata,
            },
            "v3_2_best_reference": {
                "checkpoint": str(v3_2_best_reference) if v3_2_best_reference is not None else None,
                "status": "input_mismatched_reference_not_evaluated",
                "reason": "V3.2 best is not fed the SOPAT V4 paired-anchor input contract.",
            },
        },
        "methods": methods,
        "relative_improvements": {
            "v4_ema_vs_anchor_copy": relative_improvements(v4_report, anchor_copy_report),
            "v2_latest_checkpoint_vs_anchor_copy": relative_improvements(v2_report, anchor_copy_report),
            "v4_ema_vs_v2_latest_checkpoint": relative_improvements(v4_report, v2_report),
            "v4_ema_vs_source_shuffle": relative_improvements(v4_report, source_shuffle_report),
        },
        "panel_payloads": panel_manifest,
    }
    _write_json(output / "comparison.json", result)
    return result


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args.config)
    result = evaluate(
        config,
        v4_checkpoint=args.v4_checkpoint,
        v2_sar_to_optical_checkpoint=args.v2_sar_to_optical_checkpoint,
        v2_optical_to_sar_checkpoint=args.v2_optical_to_sar_checkpoint,
        output=args.output,
        config_base=args.config.parent.resolve(),
        device=_device(args.device),
        limit_batches=args.limit_batches,
        panel_samples=args.panel_samples,
        changed_threshold=args.changed_threshold,
        seed=args.seed,
        v3_2_best_reference=args.v3_2_best_reference,
    )
    print(json.dumps({"output": str(args.output / "comparison.json"), "methods": tuple(result["methods"])}))


if __name__ == "__main__":
    main()
