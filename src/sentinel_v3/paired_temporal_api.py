"""Public inference contract for sparse paired-anchor image translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .paired_temporal_v2 import PairedTemporalOutput, SparsePairedAnchorTransport
from .sensors import SensorSpec

PairedMode = Literal["physical", "visual"]


@dataclass(frozen=True)
class PairedAnchorBatch:
    """A registered sensor pair and one-to-many causal source observations.

    `observation_present` is the only availability authority.  Padded tensor
    values are ignored.  Days are relative to the requested target time, which
    is zero; the registered pair must have a negative `anchor_days` value.
    """

    observations: Tensor
    observation_valid: Tensor
    observation_days: Tensor
    observation_present: Tensor
    source_anchor: Tensor
    source_anchor_valid: Tensor
    target_anchor: Tensor
    target_anchor_valid: Tensor
    anchor_days: Tensor
    source_sensor: SensorSpec
    target_sensor: SensorSpec
    source_anchor_days: Tensor | None = None
    target_anchor_days: Tensor | None = None


@dataclass
class PairedTranslationResult:
    output: Tensor
    physical: Tensor
    mode: PairedMode
    task_is_translation: Tensor
    attention: Tensor
    log_variance: Tensor
    deterministic_detail: Tensor
    detail_confidence: Tensor
    stochastic_residual: Tensor | None
    residual_amplitude: Tensor | None
    pre_projection_violation: Tensor


@torch.no_grad()
def translate_paired(
    model: SparsePairedAnchorTransport,
    batch: PairedAnchorBatch,
    *,
    mode: PairedMode = "physical",
    seed: int = 0,
    steps: int | None = None,
) -> PairedTranslationResult:
    """Translate or forecast with one API; timing determines the task regime."""

    if mode not in {"physical", "visual"}:
        raise ValueError("mode must be physical or visual")
    was_training = model.training
    model.eval()
    try:
        base = model(
            batch.observations,
            batch.observation_valid,
            batch.observation_days,
            batch.observation_present,
            batch.source_anchor,
            batch.source_anchor_valid,
            batch.target_anchor,
            batch.target_anchor_valid,
            batch.anchor_days,
            source_sensor=batch.source_sensor,
            target_sensor=batch.target_sensor,
            source_anchor_days=batch.source_anchor_days,
            target_anchor_days=batch.target_anchor_days,
        )
        resolved: PairedTemporalOutput
        if mode == "visual":
            resolved = model.sample_visual(
                base,
                batch.target_anchor_valid,
                batch.target_sensor,
                seed=seed,
                steps=steps,
            )
            if resolved.visual is None:
                raise RuntimeError("visual translation did not return an image")
            output = resolved.visual
        else:
            resolved = base
            output = base.physical
    finally:
        if was_training:
            model.train()
    return PairedTranslationResult(
        output=output,
        physical=resolved.physical,
        mode=mode,
        task_is_translation=resolved.task_is_translation,
        attention=resolved.attention,
        log_variance=resolved.log_variance,
        deterministic_detail=resolved.deterministic_detail,
        detail_confidence=resolved.detail_confidence,
        stochastic_residual=resolved.stochastic_residual,
        residual_amplitude=resolved.residual_amplitude,
        pre_projection_violation=resolved.pre_projection_violation,
    )
