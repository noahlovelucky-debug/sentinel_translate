"""Evaluation and model-selection utilities for bidirectional SOPAT V4.

The evaluator keeps model inputs causal.  Ground-truth targets are retrieved
only after a prediction is produced, then used to stratify metrics by task,
available observation count, and target-anchor change.  This makes a reported
``changed`` score an evaluation diagnostic rather than an input feature.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Sampler

from .training import (
    DIRECTIONS,
    Direction,
    SOPATForwardProtocol,
    forward_direction,
    forward_input_tensors,
    latest_only_batch,
    null_change_batch,
    output_tensor,
    supervision_tensors,
)

VariantName = Literal[
    "sopat",
    "anchor_copy",
    "latest_only",
    "mean_pool",
    "concat",
    "source_shuffle",
    "source_null",
]
SelectionPhase = Literal["feasibility", "full"]
SELECTION_POLICY_VERSION = "sopat_v4_quality_gate_v2"

VARIANT_NAMES: frozenset[str] = frozenset(
    {
        "sopat",
        "anchor_copy",
        "latest_only",
        "mean_pool",
        "concat",
        "source_shuffle",
        "source_null",
    }
)
_SAR_DB_MINIMUM = (-35.0, -45.0)
_SAR_DB_MAXIMUM = (5.0, -5.0)
_SAR_HISTOGRAM_MINIMUM_DB = -60.0
_SAR_HISTOGRAM_MAXIMUM_DB = 20.0
_SAR_HISTOGRAM_BINS = 320
_STRUCTURAL_BOX_SIZE = 9
_SOURCE_SHUFFLE_PLANNER = "stable_cyclic_offset_v1"


class NoSingletonBatchSampler(Sampler[list[int]]):
    """Yield a fixed full-validation partition with no singleton batches.

    Source-history shuffling is only meaningful when every evaluated batch
    contains at least two independent scenes.  This sampler preserves the
    complete validation protocol by visiting each index exactly once in
    ascending order.  When a singleton tail would occur, one item is moved
    from the preceding full batch to make ``batch_size - 1`` and ``2``.

    A batch size of two cannot partition an odd number of samples without a
    singleton, duplication, or an oversized batch.  That case fails closed so
    callers can choose a compatible validation batch size (for example three).
    """

    def __init__(self, sample_count: int, batch_size: int) -> None:
        if sample_count < 2:
            raise ValueError(
                "SOPAT validation requires at least two samples per direction "
                "for source_shuffle"
            )
        if batch_size < 2:
            raise ValueError("SOPAT validation.batch_size must be at least 2")
        if batch_size == 2 and sample_count % 2:
            raise ValueError(
                "SOPAT validation cannot partition an odd sample count with "
                "batch_size=2 without a singleton; use validation.batch_size >= 3"
            )
        self.sample_count = int(sample_count)
        self.batch_size = int(batch_size)
        self._batches = self._build_batches()

    def _build_batches(self) -> tuple[tuple[int, ...], ...]:
        full_batches, remainder = divmod(self.sample_count, self.batch_size)
        if remainder != 1:
            return tuple(
                tuple(range(start, min(start + self.batch_size, self.sample_count)))
                for start in range(0, self.sample_count, self.batch_size)
            )

        # There is at least one full batch here: sample_count >= batch_size + 1.
        prefix_full_batches = full_batches - 1
        prefix_end = prefix_full_batches * self.batch_size
        borrowed_batch_end = prefix_end + self.batch_size - 1
        batches = [
            tuple(range(start, start + self.batch_size))
            for start in range(0, prefix_end, self.batch_size)
        ]
        batches.append(tuple(range(prefix_end, borrowed_batch_end)))
        batches.append(tuple(range(borrowed_batch_end, self.sample_count)))
        return tuple(batches)

    def __iter__(self) -> Iterator[list[int]]:
        return iter([list(batch) for batch in self._batches])

    def __len__(self) -> int:
        return len(self._batches)


@dataclass(frozen=True)
class SOPATVariantConfig:
    """One reproducible SOPAT evaluation route.

    ``mean_pool`` and ``concat`` are architecture/training ablation labels.
    They call the supplied trained model and are not asserted to reproduce an
    external baseline.  A checkpoint must itself have been trained with the
    corresponding aggregation configuration.
    """

    name: VariantName = "sopat"
    seed: int = 71

    def __post_init__(self) -> None:
        if self.name not in VARIANT_NAMES:
            raise ValueError(f"unsupported SOPAT evaluation variant: {self.name}")

    @property
    def route_kind(self) -> str:
        if self.name == "anchor_copy":
            return "nonlearned_anchor_baseline"
        if self.name in {"mean_pool", "concat"}:
            return "trained_ablation_interface"
        if self.name in {"source_shuffle", "source_null"}:
            return "counterfactual"
        return "trained_sopat"

    def metadata(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "seed": self.seed,
            "route_kind": self.route_kind,
            "external_reproduction": False,
        }
        if self.name == "source_shuffle":
            result.update(
                {
                    "shuffle_planner": _SOURCE_SHUFFLE_PLANNER,
                    "shuffle_key": ("variant.seed", "direction", "batch_index"),
                    "shuffle_generator": "ignored_for_reproducibility",
                }
            )
        return result


@dataclass(frozen=True)
class SOPATSelectionConfig:
    """Directional selection gates for feasibility and full validation.

    The feasibility phase requires both directions and all declared key
    regimes, while allowing a limited bucket regression.  It deliberately
    does not impose an old fixed percentage improvement on every bucket.  The
    full phase adds the published physical thresholds and requires no anchor
    regression in each key regime.
    """

    phase: SelectionPhase = "feasibility"
    required_tasks: tuple[str, ...] = ("translation", "forecast")
    required_observation_counts: tuple[int | str, ...] = ("one",)
    feasibility_overall_anchor_ratio: float = 1.10
    feasibility_bucket_anchor_ratio: float = 1.25
    full_overall_anchor_ratio: float = 1.0
    full_bucket_anchor_ratio: float = 1.0
    full_optical_rmse_max: float = 0.03909
    full_optical_sam_deg_max: float = 5.716
    full_sar_db_rmse_max: float = 5.0
    full_sar_db_bias_abs_max: float = 0.5
    feasibility_scene_improved_fraction_min: float = 0.50
    full_scene_improved_fraction_min: float = 0.70
    feasibility_source_shuffle_min_degradation: float = 0.01
    full_source_shuffle_min_degradation: float = 0.02
    optical_sam_anchor_delta_max: float = 0.0
    optical_ndvi_mae_anchor_delta_max: float = 0.0
    optical_edge_f1_anchor_delta_min: float = 0.0
    sar_edge_f1_anchor_delta_min: float = -0.02

    def __post_init__(self) -> None:
        if self.phase not in {"feasibility", "full"}:
            raise ValueError("SOPAT selection phase must be feasibility or full")
        if any(not task for task in self.required_tasks):
            raise ValueError("SOPAT selection tasks must be non-empty strings")
        allowed_counts = {"one", "two_to_three", "four_plus"}
        for count in self.required_observation_counts:
            if isinstance(count, int) and count > 0:
                continue
            if isinstance(count, str) and count in allowed_counts:
                continue
            raise ValueError(
                "SOPAT selection observation counts must be positive integers "
                "or one/two_to_three/four_plus"
            )
        ratios = (
            self.feasibility_overall_anchor_ratio,
            self.feasibility_bucket_anchor_ratio,
            self.full_overall_anchor_ratio,
            self.full_bucket_anchor_ratio,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
            raise ValueError("SOPAT selection anchor ratios must be finite and positive")
        thresholds = (
            self.full_optical_rmse_max,
            self.full_optical_sam_deg_max,
            self.full_sar_db_rmse_max,
            self.full_sar_db_bias_abs_max,
            self.feasibility_scene_improved_fraction_min,
            self.full_scene_improved_fraction_min,
            self.feasibility_source_shuffle_min_degradation,
            self.full_source_shuffle_min_degradation,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
            raise ValueError("SOPAT full selection thresholds must be finite and non-negative")
        signed_thresholds = (
            self.optical_sam_anchor_delta_max,
            self.optical_ndvi_mae_anchor_delta_max,
            self.optical_edge_f1_anchor_delta_min,
            self.sar_edge_f1_anchor_delta_min,
        )
        if any(not math.isfinite(value) for value in signed_thresholds):
            raise ValueError("SOPAT anchor-relative thresholds must be finite")


@dataclass(frozen=True)
class SOPATSelectionDecision:
    """Serializable result of applying joint two-direction selection gates."""

    eligible: bool
    score: float
    failures: tuple[str, ...]
    phase: SelectionPhase

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_DEFAULT_VARIANT = SOPATVariantConfig()
_DEFAULT_SELECTION = SOPATSelectionConfig()


@dataclass(frozen=True)
class _Prediction:
    values: Tensor
    pre_projection_violation: Tensor | None


class _MetricAccumulator:
    """Pixel-weighted metric sums for one direction/regime/change bucket."""

    def __init__(self, direction: Direction) -> None:
        self.direction = direction
        self.samples = 0
        self.pixels = 0.0
        self.squared_error = 0.0
        self.absolute_error = 0.0
        self.error_sum = 0.0
        self.anchor_squared_error = 0.0
        self.anchor_absolute_error = 0.0
        self.anchor_error_sum = 0.0
        self.structural_squared_error = 0.0
        self.anchor_structural_squared_error = 0.0
        self.structural_pixels = 0.0
        self.sam_sum = 0.0
        self.anchor_sam_sum = 0.0
        self.sam_pixels = 0.0
        self.ndvi_absolute_error = 0.0
        self.anchor_ndvi_absolute_error = 0.0
        self.ndvi_error_sum = 0.0
        self.ndvi_pixels = 0.0
        self.sar_squared_error = 0.0
        self.sar_absolute_error = 0.0
        self.sar_error_sum = 0.0
        self.sar_anchor_squared_error = 0.0
        self.sar_count = 0.0
        self.sar_prediction_sum = 0.0
        self.sar_target_sum = 0.0
        self.sar_prediction_square_sum = 0.0
        self.sar_target_square_sum = 0.0
        self.sar_product_sum = 0.0
        self.pre_projection_sum = 0.0
        self.pre_projection_samples = 0
        self.edge_true_positive = 0.0
        self.edge_false_positive = 0.0
        self.edge_false_negative = 0.0
        self.anchor_edge_true_positive = 0.0
        self.anchor_edge_false_positive = 0.0
        self.anchor_edge_false_negative = 0.0
        self.psd_log_l1_sum = 0.0
        self.psd_samples = 0
        self.scene_improved = 0
        self.scene_total = 0
        self.sar_prediction_histogram = torch.zeros(_SAR_HISTOGRAM_BINS, dtype=torch.float64)
        self.sar_target_histogram = torch.zeros(_SAR_HISTOGRAM_BINS, dtype=torch.float64)

    def add(
        self,
        prediction: Tensor,
        target: Tensor,
        anchor: Tensor,
        valid: Tensor,
        *,
        pre_projection_violation: float | None,
    ) -> None:
        if prediction.shape != target.shape or target.shape != anchor.shape:
            raise ValueError("SOPAT evaluation prediction, target, and anchor shapes must match")
        if prediction.shape[0] != 1:
            raise ValueError("SOPAT metric accumulation expects one sample at a time")
        if valid.shape != (1, 1, *prediction.shape[-2:]):
            raise ValueError("SOPAT evaluation valid mask must be 1x1xHxW")
        mask = valid.to(prediction).expand_as(prediction)
        count = float(mask.sum().detach().cpu())
        if count <= 0.0:
            return
        error = prediction - target
        anchor_error = anchor - target
        self.samples += 1
        self.pixels += count
        self.squared_error += float((error.square() * mask).sum().detach().cpu())
        self.absolute_error += float((error.abs() * mask).sum().detach().cpu())
        self.error_sum += float((error * mask).sum().detach().cpu())
        self.anchor_squared_error += float((anchor_error.square() * mask).sum().detach().cpu())
        self.anchor_absolute_error += float((anchor_error.abs() * mask).sum().detach().cpu())
        self.anchor_error_sum += float((anchor_error * mask).sum().detach().cpu())
        structural_prediction, structural_support = _masked_box_lowpass(prediction, valid)
        structural_target, _ = _masked_box_lowpass(target, valid)
        structural_anchor, _ = _masked_box_lowpass(anchor, valid)
        structural_mask = (
            (valid.to(prediction) > 0.0).to(prediction) * structural_support
        ).expand_as(prediction)
        structural_count = float(structural_mask.sum().detach().cpu())
        if structural_count > 0.0:
            self.structural_squared_error += float(
                ((structural_prediction - structural_target).square() * structural_mask)
                .sum()
                .detach()
                .cpu()
            )
            self.anchor_structural_squared_error += float(
                ((structural_anchor - structural_target).square() * structural_mask)
                .sum()
                .detach()
                .cpu()
            )
            self.structural_pixels += structural_count
        prediction_scene = _scene_rmse(prediction, target, valid)
        anchor_scene = _scene_rmse(anchor, target, valid)
        self.scene_improved += int(prediction_scene < anchor_scene)
        self.scene_total += 1
        self._add_edge_and_spectrum(prediction, target, anchor, valid)
        if pre_projection_violation is not None:
            self.pre_projection_sum += float(pre_projection_violation)
            self.pre_projection_samples += 1
        if self.direction == "sar_to_optical":
            self._add_optical(prediction, target, anchor, valid)
        else:
            self._add_sar(prediction, target, anchor, valid)

    def _add_edge_and_spectrum(
        self, prediction: Tensor, target: Tensor, anchor: Tensor, valid: Tensor
    ) -> None:
        edge_prediction = _edge_mask(prediction)
        edge_anchor = _edge_mask(anchor)
        edge_target = _edge_mask(target)
        edge_valid = _interior_valid(valid).to(torch.bool)
        true_positive = edge_prediction & edge_target & edge_valid
        false_positive = edge_prediction & ~edge_target & edge_valid
        false_negative = ~edge_prediction & edge_target & edge_valid
        self.edge_true_positive += float(true_positive.sum().detach().cpu())
        self.edge_false_positive += float(false_positive.sum().detach().cpu())
        self.edge_false_negative += float(false_negative.sum().detach().cpu())
        self.anchor_edge_true_positive += float(
            (edge_anchor & edge_target & edge_valid).sum().detach().cpu()
        )
        self.anchor_edge_false_positive += float(
            (edge_anchor & ~edge_target & edge_valid).sum().detach().cpu()
        )
        self.anchor_edge_false_negative += float(
            (~edge_anchor & edge_target & edge_valid).sum().detach().cpu()
        )
        self.psd_log_l1_sum += float(_psd_log_l1(prediction, target, valid).detach().cpu())
        self.psd_samples += 1

    def _add_optical(
        self, prediction: Tensor, target: Tensor, anchor: Tensor, valid: Tensor
    ) -> None:
        if prediction.shape[1] < 7:
            raise ValueError("SAR-to-optical evaluation requires canonical optical channels")
        valid_pixels = valid[:, 0].to(prediction)
        cosine = (prediction * target).sum(dim=1) / (
            prediction.square().sum(dim=1).sqrt() * target.square().sum(dim=1).sqrt()
        ).clamp_min(1e-6)
        angles = torch.acos(cosine.clamp(-1.0, 1.0)) * (180.0 / math.pi)
        anchor_cosine = (anchor * target).sum(dim=1) / (
            anchor.square().sum(dim=1).sqrt() * target.square().sum(dim=1).sqrt()
        ).clamp_min(1e-6)
        anchor_angles = torch.acos(anchor_cosine.clamp(-1.0, 1.0)) * (180.0 / math.pi)
        pixels = float(valid_pixels.sum().detach().cpu())
        self.sam_sum += float((angles * valid_pixels).sum().detach().cpu())
        self.anchor_sam_sum += float((anchor_angles * valid_pixels).sum().detach().cpu())
        self.sam_pixels += pixels
        predicted_reflectance = (prediction + 1.0) * 0.5
        target_reflectance = (target + 1.0) * 0.5
        anchor_reflectance = (anchor + 1.0) * 0.5
        predicted_ndvi = (
            predicted_reflectance[:, 6] - predicted_reflectance[:, 2]
        ) / (predicted_reflectance[:, 6] + predicted_reflectance[:, 2] + 1e-4)
        target_ndvi = (target_reflectance[:, 6] - target_reflectance[:, 2]) / (
            target_reflectance[:, 6] + target_reflectance[:, 2] + 1e-4
        )
        anchor_ndvi = (anchor_reflectance[:, 6] - anchor_reflectance[:, 2]) / (
            anchor_reflectance[:, 6] + anchor_reflectance[:, 2] + 1e-4
        )
        ndvi_error = predicted_ndvi - target_ndvi
        self.ndvi_absolute_error += float((ndvi_error.abs() * valid_pixels).sum().detach().cpu())
        self.anchor_ndvi_absolute_error += float(
            ((anchor_ndvi - target_ndvi).abs() * valid_pixels).sum().detach().cpu()
        )
        self.ndvi_error_sum += float((ndvi_error * valid_pixels).sum().detach().cpu())
        self.ndvi_pixels += pixels

    def _add_sar(self, prediction: Tensor, target: Tensor, anchor: Tensor, valid: Tensor) -> None:
        prediction_db = _sar_normalized_to_db(prediction)
        target_db = _sar_normalized_to_db(target)
        anchor_db = _sar_normalized_to_db(anchor)
        mask = valid.to(prediction_db).expand_as(prediction_db)
        error = prediction_db - target_db
        anchor_error = anchor_db - target_db
        self.sar_squared_error += float((error.square() * mask).sum().detach().cpu())
        self.sar_absolute_error += float((error.abs() * mask).sum().detach().cpu())
        self.sar_error_sum += float((error * mask).sum().detach().cpu())
        self.sar_anchor_squared_error += float((anchor_error.square() * mask).sum().detach().cpu())
        self.sar_count += float(mask.sum().detach().cpu())
        self.sar_prediction_sum += float((prediction_db * mask).sum().detach().cpu())
        self.sar_target_sum += float((target_db * mask).sum().detach().cpu())
        self.sar_prediction_square_sum += float((prediction_db.square() * mask).sum().detach().cpu())
        self.sar_target_square_sum += float((target_db.square() * mask).sum().detach().cpu())
        self.sar_product_sum += float((prediction_db * target_db * mask).sum().detach().cpu())
        values_prediction = prediction_db.masked_select(mask.bool()).detach().float().cpu()
        values_target = target_db.masked_select(mask.bool()).detach().float().cpu()
        self.sar_prediction_histogram += torch.histc(
            values_prediction,
            bins=_SAR_HISTOGRAM_BINS,
            min=_SAR_HISTOGRAM_MINIMUM_DB,
            max=_SAR_HISTOGRAM_MAXIMUM_DB,
        ).to(dtype=torch.float64)
        self.sar_target_histogram += torch.histc(
            values_target,
            bins=_SAR_HISTOGRAM_BINS,
            min=_SAR_HISTOGRAM_MINIMUM_DB,
            max=_SAR_HISTOGRAM_MAXIMUM_DB,
        ).to(dtype=torch.float64)

    def report(self) -> dict[str, float | int | None]:
        if self.pixels <= 0.0:
            return _empty_report(self.direction)
        result: dict[str, float | int | None] = {
            "samples": self.samples,
            "valid_channel_pixels": self.pixels,
            "rmse": math.sqrt(self.squared_error / self.pixels),
            "mae": self.absolute_error / self.pixels,
            "bias": self.error_sum / self.pixels,
            "anchor_rmse": math.sqrt(self.anchor_squared_error / self.pixels),
            "anchor_mae": self.anchor_absolute_error / self.pixels,
            "anchor_bias": self.anchor_error_sum / self.pixels,
            "structural_rmse": (
                math.sqrt(self.structural_squared_error / self.structural_pixels)
                if self.structural_pixels > 0.0
                else None
            ),
            "anchor_structural_rmse": (
                math.sqrt(self.anchor_structural_squared_error / self.structural_pixels)
                if self.structural_pixels > 0.0
                else None
            ),
            "pre_projection_violation": (
                self.pre_projection_sum / self.pre_projection_samples
                if self.pre_projection_samples
                else None
            ),
            "edge_f1": _edge_f1(
                self.edge_true_positive,
                self.edge_false_positive,
                self.edge_false_negative,
            ),
            "anchor_edge_f1": _edge_f1(
                self.anchor_edge_true_positive,
                self.anchor_edge_false_positive,
                self.anchor_edge_false_negative,
            ),
            "psd_log_l1": self.psd_log_l1_sum / max(self.psd_samples, 1),
            "scene_improved_fraction": self.scene_improved / self.scene_total
            if self.scene_total
            else None,
        }
        if self.direction == "sar_to_optical":
            result.update(
                {
                    "sam_deg": self.sam_sum / max(self.sam_pixels, 1.0),
                    "anchor_sam_deg": self.anchor_sam_sum / max(self.sam_pixels, 1.0),
                    "ndvi_mae": self.ndvi_absolute_error / max(self.ndvi_pixels, 1.0),
                    "anchor_ndvi_mae": self.anchor_ndvi_absolute_error
                    / max(self.ndvi_pixels, 1.0),
                    "ndvi_bias": self.ndvi_error_sum / max(self.ndvi_pixels, 1.0),
                }
            )
        else:
            correlation = _correlation_from_sums(
                self.sar_count,
                self.sar_prediction_sum,
                self.sar_target_sum,
                self.sar_prediction_square_sum,
                self.sar_target_square_sum,
                self.sar_product_sum,
            )
            result.update(
                {
                    "sar_db_rmse": math.sqrt(self.sar_squared_error / max(self.sar_count, 1.0)),
                    "sar_db_mae": self.sar_absolute_error / max(self.sar_count, 1.0),
                    "sar_db_bias": self.sar_error_sum / max(self.sar_count, 1.0),
                    "sar_db_anchor_rmse": math.sqrt(
                        self.sar_anchor_squared_error / max(self.sar_count, 1.0)
                    ),
                    "sar_db_corr": correlation,
                    "sar_psd_log_l1": self.psd_log_l1_sum / max(self.psd_samples, 1),
                    "sar_db_prediction_p01": _histogram_quantile(
                        self.sar_prediction_histogram, 0.01
                    ),
                    "sar_db_prediction_p99": _histogram_quantile(
                        self.sar_prediction_histogram, 0.99
                    ),
                    "sar_db_target_p01": _histogram_quantile(self.sar_target_histogram, 0.01),
                    "sar_db_target_p99": _histogram_quantile(self.sar_target_histogram, 0.99),
                }
            )
        return result


def predict_sopat_variant(
    model: SOPATForwardProtocol | nn.Module | None,
    batch: Mapping[str, object],
    direction: Direction,
    variant: SOPATVariantConfig | str = _DEFAULT_VARIANT,
    *,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    batch_index: int = 0,
) -> _Prediction:
    """Produce one physical prediction through a named, auditable route."""

    resolved = _coerce_variant(variant)
    if resolved.name == "anchor_copy":
        inputs = forward_input_tensors(batch, device=device)
        return _Prediction(inputs["target_anchor"], None)
    if model is None:
        raise TypeError(f"SOPAT variant {resolved.name} requires a trained model")
    routed_batch: Mapping[str, object]
    if resolved.name == "latest_only":
        routed_batch = latest_only_batch(batch)
    elif resolved.name == "source_shuffle":
        routed_batch = _source_shuffle_batch(
            batch,
            seed=resolved.seed,
            direction=direction,
            batch_index=batch_index,
            generator=generator,
        )
    elif resolved.name == "source_null":
        routed_batch = null_change_batch(batch)
    else:
        # ``mean_pool`` and ``concat`` are trained architecture variants.  The
        # model config selects their aggregation implementation; evaluation
        # must not silently substitute a handcrafted non-equivalent operation.
        routed_batch = batch
    output = forward_direction(model, routed_batch, direction, device=device)
    physical = output_tensor(output, "physical")
    assert physical is not None
    violation = output_tensor(output, "pre_projection_violation", required=False)
    return _Prediction(physical, violation)


def evaluate_sopat_loaders(
    model: SOPATForwardProtocol | nn.Module | None,
    loaders: Mapping[Direction, Iterable[Mapping[str, object]]],
    *,
    variant: SOPATVariantConfig | str = _DEFAULT_VARIANT,
    device: torch.device | None = None,
    changed_threshold: float = 0.05,
    limit_batches: int | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, object]:
    """Evaluate both directions with direction/task/N/change stratification."""

    if set(loaders) != set(DIRECTIONS):
        raise ValueError("SOPAT evaluation requires exactly both direction loaders")
    if not math.isfinite(changed_threshold) or changed_threshold < 0.0:
        raise ValueError("SOPAT changed_threshold must be finite and non-negative")
    if limit_batches is not None and limit_batches <= 0:
        raise ValueError("SOPAT limit_batches must be positive when supplied")
    resolved = _coerce_variant(variant)
    was_training = isinstance(model, nn.Module) and model.training
    if isinstance(model, nn.Module):
        model.eval()
    try:
        directions = {
            direction: _evaluate_direction_loader(
                model,
                loaders[direction],
                direction,
                resolved,
                device=device,
                changed_threshold=changed_threshold,
                limit_batches=limit_batches,
                generator=generator,
            )
            for direction in DIRECTIONS
        }
    finally:
        if isinstance(model, nn.Module) and was_training:
            model.train()
    return {
        "family": "sopat_v4_evaluation",
        "variant": resolved.metadata(),
        "changed_threshold_normalized": float(changed_threshold),
        "directions": directions,
    }


def export_sopat_prediction_samples(
    model: SOPATForwardProtocol | nn.Module | None,
    loaders: Mapping[Direction, Iterable[Mapping[str, object]]],
    output_root: str | Path,
    *,
    variant: SOPATVariantConfig | str = _DEFAULT_VARIANT,
    device: torch.device | None = None,
    limit_per_direction: int = 16,
    generator: torch.Generator | None = None,
) -> dict[str, object]:
    """Write a deterministic fixed-panel payload and JSON manifest.

    Each ``.pt`` item contains prediction, target anchor, target, valid mask,
    and causal metadata.  This intentionally preserves normalized tensors for
    a separate renderer, avoiding an implicit RGB visualization that could
    hide multispectral or SAR-unit choices.
    """

    if set(loaders) != set(DIRECTIONS):
        raise ValueError("SOPAT panel export requires both direction loaders")
    if limit_per_direction <= 0:
        raise ValueError("SOPAT panel export limit_per_direction must be positive")
    resolved = _coerce_variant(variant)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    was_training = isinstance(model, nn.Module) and model.training
    if isinstance(model, nn.Module):
        model.eval()
    entries: list[dict[str, object]] = []
    try:
        for direction in DIRECTIONS:
            written = 0
            for batch_index, batch in enumerate(loaders[direction]):
                if not isinstance(batch, Mapping):
                    raise TypeError("SOPAT panel loader must yield mapping batches")
                with torch.inference_mode():
                    prediction = predict_sopat_variant(
                        model,
                        batch,
                        direction,
                        resolved,
                        device=device,
                        generator=generator,
                        batch_index=batch_index,
                    )
                    inputs = forward_input_tensors(batch, device=device)
                    labels = supervision_tensors(batch, device=device)
                    task_modes = _task_modes(batch, labels["target"].shape[0])
                    sample_ids = _sample_ids(batch, labels["target"].shape[0])
                    counts = inputs["observation_present"].sum(dim=1).to(dtype=torch.int64)
                    for sample_index in range(labels["target"].shape[0]):
                        if written >= limit_per_direction:
                            break
                        filename = f"{direction}_{written:03d}_{_safe_sample_id(sample_ids[sample_index])}.pt"
                        destination = root / filename
                        item = {
                            "family": "sopat_v4_fixed_panel",
                            "direction": direction,
                            "variant": resolved.name,
                            "sample_id": sample_ids[sample_index],
                            "task_mode": task_modes[sample_index],
                            "observation_count_bin": _observation_count_bin(
                                int(counts[sample_index])
                            ),
                            "prediction": prediction.values[sample_index].detach().cpu(),
                            "target_anchor": inputs["target_anchor"][sample_index].detach().cpu(),
                            "target": labels["target"][sample_index].detach().cpu(),
                            "valid": (
                                labels["target_valid"][sample_index]
                                * inputs["target_anchor_valid"][sample_index].to(
                                    labels["target_valid"]
                                )
                            )
                            .detach()
                            .cpu(),
                        }
                        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
                        torch.save(item, temporary)
                        os.replace(temporary, destination)
                        entries.append(
                            {
                                "direction": direction,
                                "sample_id": sample_ids[sample_index],
                                "task_mode": task_modes[sample_index],
                                "observation_count_bin": _observation_count_bin(
                                    int(counts[sample_index])
                                ),
                                "file": filename,
                            }
                        )
                        written += 1
                if written >= limit_per_direction:
                    break
    finally:
        if isinstance(model, nn.Module) and was_training:
            model.train()
    manifest = {
        "family": "sopat_v4_fixed_panel",
        "variant": resolved.metadata(),
        "input_matched": True,
        "entries": entries,
    }
    destination = root / "panel_manifest.json"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    import json

    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return manifest


def select_sopat_candidate(
    report: Mapping[str, object],
    config: SOPATSelectionConfig = _DEFAULT_SELECTION,
    *,
    source_shuffle_report: Mapping[str, object] | None = None,
) -> SOPATSelectionDecision:
    """Require joint directional quality and effective source-shuffle evidence.

    The source-shuffle counterfactual is a causal-input dependence check.  A
    candidate can only become best when replacing each scene's source history
    with a different scene's history measurably degrades mask-aware structural
    RMSE in both directions.  Missing or invalid shuffle evidence fails closed.
    """

    directions = report.get("directions")
    if not isinstance(directions, Mapping):
        raise TypeError("SOPAT selection report is missing directions")
    failures: list[str] = []
    scores: list[float] = []
    overall_ratio, bucket_ratio = _selection_ratios(config)
    for direction in DIRECTIONS:
        directional = directions.get(direction)
        if not isinstance(directional, Mapping):
            failures.append(f"missing_direction:{direction}")
            continue
        overall = _bucket_metrics(directional, "all")
        ratio = _anchor_ratio(overall, direction)
        if ratio is None:
            failures.append(f"missing_overall_metrics:{direction}")
        elif ratio > overall_ratio:
            failures.append(f"overall_anchor_regression:{direction}:{ratio:.6f}")
        else:
            scores.append(ratio)
        _full_direction_gates(overall, direction, config, failures)
        _source_shuffle_gate(
            overall,
            _overall_metrics(
                source_shuffle_report,
                direction,
                required_variant="source_shuffle",
            ),
            direction,
            config,
            failures,
        )
        for task in config.required_tasks:
            for count in config.required_observation_counts:
                key = f"{task}/n={_count_bin_label(count)}"
                regimes = directional.get("regimes")
                regime = regimes.get(key) if isinstance(regimes, Mapping) else None
                metrics = _change_bucket_metrics(regime, "all")
                regime_ratio = _anchor_ratio(metrics, direction)
                if regime_ratio is None:
                    failures.append(f"missing_regime:{direction}:{key}")
                elif regime_ratio > bucket_ratio:
                    failures.append(
                        f"regime_anchor_regression:{direction}:{key}:{regime_ratio:.6f}"
                    )
                else:
                    scores.append(regime_ratio)
    if not scores:
        failures.append("no_valid_joint_direction_score")
    return SOPATSelectionDecision(
        eligible=not failures,
        score=max(scores) if scores and not failures else float("inf"),
        failures=tuple(failures),
        phase=config.phase,
    )


def is_better_sopat_candidate(
    candidate: SOPATSelectionDecision,
    current_best: SOPATSelectionDecision | None,
) -> bool:
    """Only a jointly eligible two-direction candidate can become best."""

    if not candidate.eligible or not math.isfinite(candidate.score):
        return False
    return current_best is None or not current_best.eligible or candidate.score < current_best.score


def _evaluate_direction_loader(
    model: SOPATForwardProtocol | nn.Module | None,
    loader: Iterable[Mapping[str, object]],
    direction: Direction,
    variant: SOPATVariantConfig,
    *,
    device: torch.device | None,
    changed_threshold: float,
    limit_batches: int | None,
    generator: torch.Generator | None,
) -> dict[str, object]:
    buckets: dict[tuple[str, str], _MetricAccumulator] = {}

    def accumulator(group: str, change: str) -> _MetricAccumulator:
        key = (group, change)
        current = buckets.get(key)
        if current is None:
            current = _MetricAccumulator(direction)
            buckets[key] = current
        return current

    for batch_index, batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        if not isinstance(batch, Mapping):
            raise TypeError("SOPAT evaluation loader must yield mapping batches")
        with torch.inference_mode():
            prediction = predict_sopat_variant(
                model,
                batch,
                direction,
                variant,
                device=device,
                generator=generator,
                batch_index=batch_index,
            )
            inputs = forward_input_tensors(batch, device=device)
            labels = supervision_tensors(batch, device=device)
            valid = labels["target_valid"] * inputs["target_anchor_valid"].to(labels["target_valid"])
            target = labels["target"]
            anchor = inputs["target_anchor"]
            if prediction.values.shape != target.shape:
                raise ValueError("SOPAT evaluation physical prediction does not match the target")
            if anchor.shape != target.shape:
                raise ValueError("SOPAT evaluation target anchor does not match the target")
            task_modes = _task_modes(batch, target.shape[0])
            counts = inputs["observation_present"].sum(dim=1).to(dtype=torch.int64)
            changed = _changed_mask(target, anchor, valid, changed_threshold)
            for sample_index in range(target.shape[0]):
                sample_valid = valid[sample_index : sample_index + 1]
                sample_changed = changed[sample_index : sample_index + 1]
                pre_projection = _sample_violation(prediction.pre_projection_violation, sample_index)
                groups = (
                    "all",
                    f"task={task_modes[sample_index]}",
                    f"n={_observation_count_bin(int(counts[sample_index]))}",
                    f"{task_modes[sample_index]}/n={_observation_count_bin(int(counts[sample_index]))}",
                )
                for group in groups:
                    accumulator(group, "all").add(
                        prediction.values[sample_index : sample_index + 1],
                        target[sample_index : sample_index + 1],
                        anchor[sample_index : sample_index + 1],
                        sample_valid,
                        pre_projection_violation=pre_projection,
                    )
                    accumulator(group, "changed").add(
                        prediction.values[sample_index : sample_index + 1],
                        target[sample_index : sample_index + 1],
                        anchor[sample_index : sample_index + 1],
                        sample_valid * sample_changed,
                        pre_projection_violation=pre_projection,
                    )
                    accumulator(group, "unchanged").add(
                        prediction.values[sample_index : sample_index + 1],
                        target[sample_index : sample_index + 1],
                        anchor[sample_index : sample_index + 1],
                        sample_valid * (1.0 - sample_changed),
                        pre_projection_violation=pre_projection,
                    )
    return _direction_report(buckets, direction)


def _direction_report(
    buckets: Mapping[tuple[str, str], _MetricAccumulator], direction: Direction
) -> dict[str, object]:
    all_metrics = _group_report(buckets, "all", direction)
    tasks: dict[str, object] = {}
    counts: dict[str, object] = {}
    regimes: dict[str, object] = {}
    for group in sorted({name for name, _ in buckets}):
        if group.startswith("task="):
            tasks[group.removeprefix("task=")] = _group_report(buckets, group, direction)
        elif group.startswith("n="):
            counts[group.removeprefix("n=")] = _group_report(buckets, group, direction)
        elif "/n=" in group:
            regimes[group] = _group_report(buckets, group, direction)
    return {
        "all": all_metrics,
        "by_task": tasks,
        "by_observation_count": counts,
        "by_observation_count_bin": counts,
        "regimes": regimes,
    }


def _group_report(
    buckets: Mapping[tuple[str, str], _MetricAccumulator], group: str, direction: Direction
) -> dict[str, dict[str, float | int | None]]:
    return {
        change: buckets.get((group, change), _MetricAccumulator(direction)).report()
        for change in ("all", "changed", "unchanged")
    }


def _empty_report(direction: Direction) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "samples": 0,
        "valid_channel_pixels": 0.0,
        "rmse": None,
        "mae": None,
        "bias": None,
        "anchor_rmse": None,
        "anchor_mae": None,
        "anchor_bias": None,
        "structural_rmse": None,
        "anchor_structural_rmse": None,
        "pre_projection_violation": None,
        "edge_f1": None,
        "anchor_edge_f1": None,
        "psd_log_l1": None,
        "scene_improved_fraction": None,
    }
    if direction == "sar_to_optical":
        result.update(
            {
                "sam_deg": None,
                "anchor_sam_deg": None,
                "ndvi_mae": None,
                "anchor_ndvi_mae": None,
                "ndvi_bias": None,
            }
        )
    else:
        result.update(
            {
                "sar_db_rmse": None,
                "sar_db_mae": None,
                "sar_db_bias": None,
                "sar_db_anchor_rmse": None,
                "sar_db_corr": None,
                "sar_psd_log_l1": None,
                "sar_db_prediction_p01": None,
                "sar_db_prediction_p99": None,
                "sar_db_target_p01": None,
                "sar_db_target_p99": None,
            }
        )
    return result


def _coerce_variant(value: SOPATVariantConfig | str) -> SOPATVariantConfig:
    return value if isinstance(value, SOPATVariantConfig) else SOPATVariantConfig(name=value)  # type: ignore[arg-type]


def _source_shuffle_batch(
    batch: Mapping[str, object],
    *,
    seed: int = _DEFAULT_VARIANT.seed,
    direction: Direction = "sar_to_optical",
    batch_index: int = 0,
    generator: torch.Generator | None = None,
) -> dict[str, object]:
    """Counterfactually exchange source histories with a derangement.

    A batch of one cannot establish source dependence.  Returning it unchanged
    would falsely make the counterfactual look valid, so the route fails
    closed.  For every larger batch, a deterministic cyclic shift gives a
    derangement by construction: no sample retains its own history.  The
    shift is a stable function of variant seed, direction, and loader batch
    ordinal.  ``generator`` remains accepted for public-call compatibility,
    but is intentionally ignored: using a CUDA generator for CPU batches used
    to fall back to global RNG and made reports irreproducible.
    """

    inputs = forward_input_tensors(batch)
    batch_size = inputs["observations"].shape[0]
    if batch_size < 2:
        raise ValueError(
            "source_shuffle evaluation requires batch_size >= 2; "
            "configure validation.batch_size accordingly"
        )
    if direction not in DIRECTIONS:
        raise ValueError(f"unsupported SOPAT source-shuffle direction: {direction}")
    if batch_index < 0:
        raise ValueError("source_shuffle batch_index must be non-negative")
    device = inputs["observations"].device
    # A non-zero cyclic offset is a derangement by construction.  In contrast,
    # rolling an arbitrary ``randperm`` can reintroduce fixed points.
    del generator
    offset = _source_shuffle_offset(
        seed=seed,
        direction=direction,
        batch_index=batch_index,
        batch_size=batch_size,
    )
    order = (torch.arange(batch_size, device=device) + offset) % batch_size
    if bool((order == torch.arange(batch_size, device=device)).any()):
        raise RuntimeError("source_shuffle derangement construction failed")
    result = dict(batch)
    for name in ("observations", "observation_valid", "observation_days", "observation_present"):
        result[name] = inputs[name].index_select(0, order)
    return result


def _source_shuffle_offset(
    *, seed: int, direction: Direction, batch_index: int, batch_size: int
) -> int:
    """Return a non-zero reproducible cyclic source-history offset."""

    if batch_size < 2:
        raise ValueError("source_shuffle batch_size must be at least 2")
    key = f"{_SOURCE_SHUFFLE_PLANNER}|{int(seed)}|{direction}|{int(batch_index)}"
    value = int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:8], "big")
    return 1 + value % (batch_size - 1)


def _sar_normalized_to_db(values: Tensor) -> Tensor:
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("SOPAT SAR metrics require Bx2xHxW normalized Sentinel-1 tensors")
    minimum = values.new_tensor(_SAR_DB_MINIMUM).reshape(1, 2, 1, 1)
    maximum = values.new_tensor(_SAR_DB_MAXIMUM).reshape(1, 2, 1, 1)
    return (values + 1.0) * 0.5 * (maximum - minimum) + minimum


def _observation_count_bin(count: int) -> str:
    if count <= 0:
        raise ValueError("SOPAT evaluation requires at least one observation")
    if count == 1:
        return "one"
    if count <= 3:
        return "two_to_three"
    return "four_plus"


def _count_bin_label(value: int | str) -> str:
    if isinstance(value, int):
        return _observation_count_bin(value)
    return value


def _scene_rmse(prediction: Tensor, target: Tensor, valid: Tensor) -> float:
    mask = valid.to(prediction).expand_as(prediction)
    denominator = mask.sum().clamp_min(1.0)
    return float((((prediction - target).square() * mask).sum() / denominator).sqrt().detach().cpu())


def _masked_box_lowpass(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Apply a strictly masked 9x9 box lowpass in normalized image units.

    A source-shuffle counterfactual must measure coherent scene degradation,
    not pixel-scale SAR speckle.  Values outside the mask are replaced before
    multiplication, so even non-finite or extreme invalid pixels cannot enter
    the pooled numerator.  Pooling the numerator and mask denominator with
    identical geometry keeps borders and irregular masks unbiased.
    """

    if values.ndim != 4:
        raise ValueError("SOPAT structural metric values must have shape BxCxHxW")
    if valid.shape != (values.shape[0], 1, *values.shape[-2:]):
        raise ValueError("SOPAT structural metric valid mask must have shape Bx1xHxW")
    mask = (valid.to(values) > 0.0).to(values)
    masked_values = torch.where(mask.bool(), values, torch.zeros_like(values))
    padding = _STRUCTURAL_BOX_SIZE // 2
    numerator = F.avg_pool2d(
        masked_values * mask,
        _STRUCTURAL_BOX_SIZE,
        stride=1,
        padding=padding,
        count_include_pad=True,
    )
    denominator = F.avg_pool2d(
        mask,
        _STRUCTURAL_BOX_SIZE,
        stride=1,
        padding=padding,
        count_include_pad=True,
    )
    support = denominator > 0.0
    epsilon = torch.finfo(values.dtype).eps
    lowpass = torch.where(
        support,
        numerator / denominator.clamp_min(epsilon),
        torch.zeros_like(numerator),
    )
    return lowpass, support.to(values)


def _interior_valid(valid: Tensor) -> Tensor:
    if valid.ndim != 4 or valid.shape[1] != 1:
        raise ValueError("SOPAT edge metric valid mask must have shape Bx1xHxW")
    result = valid.clone()
    result[..., -1, :] = 0.0
    result[..., :, -1] = 0.0
    return result


def _edge_mask(values: Tensor, threshold: float = 0.05) -> Tensor:
    if values.ndim != 4:
        raise ValueError("SOPAT edge metric values must have shape BxCxHxW")
    if values.shape[-2] < 2 or values.shape[-1] < 2:
        return torch.zeros_like(values[:, :1], dtype=torch.bool)
    luminance = values.mean(dim=1, keepdim=True)
    dx = luminance[..., :, 1:] - luminance[..., :, :-1]
    dy = luminance[..., 1:, :] - luminance[..., :-1, :]
    magnitude = torch.zeros_like(luminance)
    magnitude[..., :-1, :-1] = torch.sqrt(
        dx[..., :-1, :].square() + dy[..., :, :-1].square()
    )
    return magnitude >= threshold


def _psd_log_l1(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    mask = valid.to(prediction).expand_as(prediction)
    prediction_gray = (prediction * mask).mean(dim=1)
    target_gray = (target * mask).mean(dim=1)
    prediction_power = torch.fft.rfft2(prediction_gray, norm="ortho").abs().square()
    target_power = torch.fft.rfft2(target_gray, norm="ortho").abs().square()
    return (torch.log1p(prediction_power) - torch.log1p(target_power)).abs().mean()


def _edge_f1(true_positive: float, false_positive: float, false_negative: float) -> float:
    denominator = 2.0 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator > 0.0 else 0.0


def _histogram_quantile(histogram: Tensor, quantile: float) -> float | None:
    total = float(histogram.sum())
    if total <= 0.0:
        return None
    target = max(0.0, min(1.0, quantile)) * total
    index = int(torch.searchsorted(histogram.cumsum(0), histogram.new_tensor(target)).item())
    index = min(max(index, 0), _SAR_HISTOGRAM_BINS - 1)
    width = (_SAR_HISTOGRAM_MAXIMUM_DB - _SAR_HISTOGRAM_MINIMUM_DB) / _SAR_HISTOGRAM_BINS
    return _SAR_HISTOGRAM_MINIMUM_DB + (index + 0.5) * width


def _correlation_from_sums(
    count: float,
    prediction_sum: float,
    target_sum: float,
    prediction_square_sum: float,
    target_square_sum: float,
    product_sum: float,
) -> float:
    if count <= 1.0:
        return 0.0
    covariance = count * product_sum - prediction_sum * target_sum
    prediction_variance = count * prediction_square_sum - prediction_sum**2
    target_variance = count * target_square_sum - target_sum**2
    denominator = math.sqrt(max(prediction_variance * target_variance, 0.0))
    return covariance / denominator if denominator > 1e-12 else 0.0


def _task_modes(batch: Mapping[str, object], batch_size: int) -> list[str]:
    values = batch.get("task_mode")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        result = [str(value) for value in values]
        if len(result) == batch_size:
            return result
    if isinstance(values, str):
        return [values] * batch_size
    return ["unknown"] * batch_size


def _sample_ids(batch: Mapping[str, object], batch_size: int) -> list[str]:
    values = batch.get("sopat_example_id", batch.get("sample_id"))
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        result = [str(value) for value in values]
        if len(result) == batch_size:
            return result
    if isinstance(values, str):
        return [values] * batch_size
    return [f"sample-{index:06d}" for index in range(batch_size)]


def _safe_sample_id(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return safe[:96] or "sample"


def _changed_mask(target: Tensor, anchor: Tensor, valid: Tensor, threshold: float) -> Tensor:
    # GT target-anchor is deliberately used only after prediction.  It never
    # crosses ``forward_direction`` and is not a model conditioning feature.
    magnitude = (target - anchor).square().mean(dim=1, keepdim=True).sqrt()
    return (magnitude > threshold).to(valid)


def _sample_violation(values: Tensor | None, index: int) -> float | None:
    if values is None:
        return None
    if values.ndim == 0:
        return float(values.detach().cpu())
    if values.shape[0] <= index:
        raise ValueError("SOPAT pre_projection_violation batch dimension is invalid")
    return float(values[index].detach().mean().cpu())


def _selection_ratios(config: SOPATSelectionConfig) -> tuple[float, float]:
    if config.phase == "full":
        return config.full_overall_anchor_ratio, config.full_bucket_anchor_ratio
    return config.feasibility_overall_anchor_ratio, config.feasibility_bucket_anchor_ratio


def _bucket_metrics(directional: Mapping[str, object], name: str) -> Mapping[str, object] | None:
    return _change_bucket_metrics(directional.get(name), "all")


def _overall_metrics(
    report: Mapping[str, object] | None,
    direction: Direction,
    *,
    required_variant: VariantName | None = None,
) -> Mapping[str, object] | None:
    if not isinstance(report, Mapping):
        return None
    if required_variant is not None:
        variant = report.get("variant")
        if not isinstance(variant, Mapping) or variant.get("name") != required_variant:
            return None
    directions = report.get("directions")
    if not isinstance(directions, Mapping):
        return None
    directional = directions.get(direction)
    return _bucket_metrics(directional, "all") if isinstance(directional, Mapping) else None


def _change_bucket_metrics(value: object, change: str) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    nested = value.get(change)
    return nested if isinstance(nested, Mapping) else None


def _anchor_ratio(metrics: Mapping[str, object] | None, direction: Direction) -> float | None:
    if metrics is None:
        return None
    if direction == "optical_to_sar":
        candidate = _metric_float(metrics, "sar_db_rmse")
        anchor = _metric_float(metrics, "sar_db_anchor_rmse")
    else:
        candidate = _metric_float(metrics, "rmse")
        anchor = _metric_float(metrics, "anchor_rmse")
    if candidate is None or anchor is None or anchor <= 0.0:
        return None
    return candidate / anchor


def _source_shuffle_gate(
    candidate: Mapping[str, object] | None,
    shuffle: Mapping[str, object] | None,
    direction: Direction,
    config: SOPATSelectionConfig,
    failures: list[str],
) -> None:
    candidate_rmse = _metric_float(candidate, "structural_rmse")
    shuffle_rmse = _metric_float(shuffle, "structural_rmse")
    if candidate_rmse is None or shuffle_rmse is None:
        failures.append(f"missing_source_shuffle_structural_metrics:{direction}")
        return
    if candidate_rmse <= 0.0:
        failures.append(
            f"invalid_source_shuffle_structural_rmse:{direction}:{candidate_rmse}"
        )
        return
    degradation = shuffle_rmse / candidate_rmse - 1.0
    threshold = (
        config.full_source_shuffle_min_degradation
        if config.phase == "full"
        else config.feasibility_source_shuffle_min_degradation
    )
    if degradation < threshold:
        failures.append(
            "source_shuffle_insufficient_structural_degradation:"
            f"{direction}:{degradation:.6f}<{threshold:.6f}"
        )


def _metric_float(metrics: Mapping[str, object] | None, name: str) -> float | None:
    if metrics is None:
        return None
    value = metrics.get(name)
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _full_direction_gates(
    metrics: Mapping[str, object] | None,
    direction: Direction,
    config: SOPATSelectionConfig,
    failures: list[str],
) -> None:
    scene_minimum = (
        config.full_scene_improved_fraction_min
        if config.phase == "full"
        else config.feasibility_scene_improved_fraction_min
    )
    if direction == "sar_to_optical":
        rmse = _metric_float(metrics, "rmse")
        sam = _metric_float(metrics, "sam_deg")
        anchor_sam = _metric_float(metrics, "anchor_sam_deg")
        ndvi = _metric_float(metrics, "ndvi_mae")
        anchor_ndvi = _metric_float(metrics, "anchor_ndvi_mae")
        edge = _metric_float(metrics, "edge_f1")
        anchor_edge = _metric_float(metrics, "anchor_edge_f1")
        if config.phase == "full":
            if rmse is None or rmse > config.full_optical_rmse_max:
                failures.append(f"optical_rmse_gate:{rmse}")
            if sam is None or sam > config.full_optical_sam_deg_max:
                failures.append(f"optical_sam_gate:{sam}")
        if sam is None or anchor_sam is None or (
            sam - anchor_sam > config.optical_sam_anchor_delta_max
        ):
            failures.append(f"optical_sam_anchor_gate:{sam}:{anchor_sam}")
        if ndvi is None or anchor_ndvi is None or (
            ndvi - anchor_ndvi > config.optical_ndvi_mae_anchor_delta_max
        ):
            failures.append(f"optical_ndvi_anchor_gate:{ndvi}:{anchor_ndvi}")
        if edge is None or anchor_edge is None or (
            edge - anchor_edge < config.optical_edge_f1_anchor_delta_min
        ):
            failures.append(f"optical_edge_anchor_gate:{edge}:{anchor_edge}")
    else:
        rmse = _metric_float(metrics, "sar_db_rmse")
        bias = _metric_float(metrics, "sar_db_bias")
        edge = _metric_float(metrics, "edge_f1")
        anchor_edge = _metric_float(metrics, "anchor_edge_f1")
        if config.phase == "full" and (rmse is None or rmse > config.full_sar_db_rmse_max):
            failures.append(f"sar_rmse_gate:{rmse}")
        if bias is None or abs(bias) > config.full_sar_db_bias_abs_max:
            failures.append(f"sar_bias_gate:{bias}")
        if edge is None or anchor_edge is None or (
            edge - anchor_edge < config.sar_edge_f1_anchor_delta_min
        ):
            failures.append(f"sar_edge_anchor_gate:{edge}:{anchor_edge}")
    fraction = _metric_float(metrics, "scene_improved_fraction")
    if fraction is None or fraction < scene_minimum:
        failures.append(f"scene_improvement_gate:{direction}:{fraction}")
