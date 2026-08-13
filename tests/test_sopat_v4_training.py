from __future__ import annotations

import math
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from sentinel_v4 import evaluation as sopat_evaluation
from sentinel_v4.evaluation import SOPATVariantConfig, evaluate_sopat_loaders
from sentinel_v4.model import SOPAT, SOPATConfig
from sentinel_v4.training import (
    ModelEMA,
    SOPATTrainConfig,
    SOPATTrainingModule,
    _structural_error,
    _utility_oracle,
    configure_sopat_stage,
    evaluate_factorizer_loaders,
    initialize_from_sopat_checkpoint,
    initialize_from_v3_checkpoint,
    load_sopat_checkpoint,
    save_sopat_checkpoint,
    sopat_direction_objective,
    source_shuffle_batch,
    train_coupled_step,
)


def _batch(
    source_channels: int,
    target_channels: int,
    *,
    batch_size: int = 1,
    observations: int = 2,
    changed: bool = True,
) -> dict[str, object]:
    height = width = 8
    source_anchor = torch.full((batch_size, source_channels, height, width), -0.2)
    target_anchor = torch.full((batch_size, target_channels, height, width), 0.1)
    target = target_anchor.clone()
    if changed:
        target[-1] += 0.2
    present = torch.zeros((batch_size, observations), dtype=torch.bool)
    present[:, : max(1, observations)] = True
    days = torch.full((batch_size, observations), -3.0)
    return {
        "observations": source_anchor[:, None].expand(-1, observations, -1, -1, -1).clone(),
        "observation_valid": torch.ones((batch_size, observations, 1, height, width)),
        "observation_days": days,
        "observation_present": present,
        "source_anchor": source_anchor,
        "source_anchor_valid": torch.ones((batch_size, 1, height, width)),
        "target_anchor": target_anchor,
        "target_anchor_valid": torch.ones((batch_size, 1, height, width)),
        "source_anchor_days": torch.full((batch_size,), -5.0),
        "target_anchor_days": torch.full((batch_size,), -5.0),
        "target": target,
        "target_valid": torch.ones((batch_size, 1, height, width)),
    }


class _TwoHeadPhysicalModel(nn.Module):
    """Small public-contract model whose both render heads must be in one graph."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.tensor(0.05))
        self.optical_renderer = nn.Parameter(torch.tensor(0.10))
        self.sar_renderer = nn.Parameter(torch.tensor(-0.10))

    def forward(self, **inputs: object) -> SimpleNamespace:
        anchor = inputs["target_anchor"]
        assert isinstance(anchor, torch.Tensor)
        renderer = self.optical_renderer if anchor.shape[1] == 10 else self.sar_renderer
        delta = (self.shared + renderer) * torch.ones_like(anchor)
        return SimpleNamespace(physical=anchor + delta, log_variance=torch.zeros_like(anchor))


class _CountingCoupledModule(SOPATTrainingModule):
    def __init__(self, model: nn.Module, config: SOPATTrainConfig) -> None:
        super().__init__(model, config)
        self.calls = 0

    def forward(self, *args: object, **kwargs: object) -> tuple[torch.Tensor, dict[str, object]]:
        self.calls += 1
        return super().forward(*args, **kwargs)  # type: ignore[arg-type]


def test_coupled_step_uses_one_ddp_visible_graph_and_updates_both_heads() -> None:
    config = SOPATTrainConfig(
        stage="physical",
        autocast_bfloat16=False,
        physical_null_change_probability=0.0,
        physical_permutation_probability=0.0,
    )
    model = _TwoHeadPhysicalModel()
    module = _CountingCoupledModule(model, config)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2)
    before_optical = model.optical_renderer.detach().clone()
    before_sar = model.sar_renderer.detach().clone()

    result = train_coupled_step(
        module,
        optimizer,
        {
            "sar_to_optical": _batch(2, 10),
            "optical_to_sar": _batch(10, 2),
        },
        config,
        ema=ModelEMA.create(model, 0.9),
    )

    assert module.calls == 1
    assert set(result.direction_losses) == {"sar_to_optical", "optical_to_sar"}
    assert torch.isfinite(torch.tensor(result.gradient_norm))
    assert not torch.equal(model.optical_renderer.detach(), before_optical)
    assert not torch.equal(model.sar_renderer.detach(), before_sar)


def _ddp_coupled_worker(rank: int, world_size: int, init_method: str, queue: object) -> None:
    """Run in a fresh process so DDP sees the complete coupled graph."""

    assert hasattr(queue, "put")
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    try:
        config = SOPATTrainConfig(
            stage="physical",
            autocast_bfloat16=False,
            physical_null_change_probability=0.0,
            physical_permutation_probability=0.0,
        )
        model = _TwoHeadPhysicalModel()
        wrapped = DistributedDataParallel(SOPATTrainingModule(model, config), find_unused_parameters=False)
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
        train_coupled_step(
            wrapped,
            optimizer,
            {
                "sar_to_optical": _batch(2, 10),
                "optical_to_sar": _batch(10, 2),
            },
            config,
        )
        queue.put((rank, model.optical_renderer.item(), model.sar_renderer.item()))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_gloo_keeps_both_direction_heads_synchronized(tmp_path) -> None:
    # A file rendezvous keeps this test self-contained and avoids using a TCP
    # port shared with another developer's process.
    rendezvous = tmp_path / f"sopat-ddp-{os.getpid()}"
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.spawn(
        _ddp_coupled_worker,
        args=(2, f"file://{rendezvous}", queue),
        nprocs=2,
        join=True,
    )
    values = sorted(queue.get() for _ in range(2))
    assert values[0][1:] == pytest.approx(values[1][1:])
    # Both renderer heads participated in the single coupled DDP graph.
    assert values[0][1] != pytest.approx(0.10)
    assert values[0][2] != pytest.approx(-0.10)


class _SourceAwareToyModel(nn.Module):
    """A minimal source-dependent physical route for cross-rank ranking tests."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, **inputs: object) -> SimpleNamespace:
        observations = inputs["observations"]
        anchor = inputs["target_anchor"]
        assert isinstance(observations, torch.Tensor)
        assert isinstance(anchor, torch.Tensor)
        source_code = observations.mean(dim=(1, 2), keepdim=False)[:, None]
        source_code = source_code.expand(-1, anchor.shape[1], -1, -1)
        candidate = anchor + self.scale * source_code
        return SimpleNamespace(
            physical=candidate,
            candidate_physical=candidate,
            transport_confidence=torch.ones_like(anchor[:, :1]),
            log_variance=torch.zeros_like(anchor),
        )


def _ddp_singleton_source_shuffle_worker(
    rank: int, world_size: int, init_method: str, queue: object
) -> None:
    assert hasattr(queue, "put")
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    try:
        config = SOPATTrainConfig(
            stage="physical",
            autocast_bfloat16=False,
            physical_charbonnier_weight=1e-12,
            physical_gradient_weight=0.0,
            physical_optical_spectral_weight=0.0,
            physical_optical_ndvi_weight=0.0,
            physical_sar_statistics_weight=0.0,
            physical_anchor_delta_weight=0.0,
            physical_null_change_weight=0.0,
            physical_null_change_probability=0.0,
            physical_nll_weight=0.0,
            physical_permutation_weight=0.0,
            physical_permutation_probability=0.0,
            physical_anchor_regret_weight=0.0,
            candidate_weight=0.0,
            utility_weight=0.0,
            source_shuffle_weight=1.0,
            source_shuffle_probability=1.0,
            source_shuffle_margin=0.005,
        )
        batch = _batch(2, 10, batch_size=1)
        code = 0.1 if rank == 0 else 0.5
        observations = batch["observations"]
        target = batch["target"]
        anchor = batch["target_anchor"]
        assert isinstance(observations, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(anchor, torch.Tensor)
        observations.fill_(code)
        target.copy_(anchor + code)
        model = _SourceAwareToyModel()
        loss, metrics = sopat_direction_objective(model, batch, "sar_to_optical", config)
        loss.backward()
        assert model.scale.grad is not None
        queue.put((rank, float(metrics["physical_source_shuffle"]), float(model.scale.grad)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_singleton_source_shuffle_has_counterfactual_loss_and_gradient(tmp_path) -> None:
    rendezvous = tmp_path / f"sopat-source-shuffle-{os.getpid()}"
    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.spawn(
        _ddp_singleton_source_shuffle_worker,
        args=(2, f"file://{rendezvous}", queue),
        nprocs=2,
        join=True,
    )
    values = sorted(queue.get() for _ in range(2))
    assert all(counterfactual > 0.0 for _rank, counterfactual, _gradient in values)
    assert all(abs(gradient) > 1e-6 for _rank, _counterfactual, gradient in values)


class _ExactNullModel(nn.Module):
    """Returns an exact target anchor on the null-change causal route."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, **inputs: object) -> SimpleNamespace:
        observations = inputs["observations"]
        source_anchor = inputs["source_anchor"]
        target_anchor = inputs["target_anchor"]
        assert isinstance(observations, torch.Tensor)
        assert isinstance(source_anchor, torch.Tensor)
        assert isinstance(target_anchor, torch.Tensor)
        null = torch.equal(observations, source_anchor[:, None].expand_as(observations))
        scale = self.scale * (0.0 if null else 1.0)
        physical = target_anchor + scale * torch.ones_like(target_anchor)
        return SimpleNamespace(physical=physical, log_variance=torch.zeros_like(physical))


def test_exact_null_identity_has_finite_backward() -> None:
    config = SOPATTrainConfig(
        stage="physical",
        autocast_bfloat16=False,
        physical_null_change_probability=1.0,
        physical_permutation_probability=0.0,
    )
    model = _ExactNullModel()
    loss, metrics = sopat_direction_objective(model, _batch(2, 10), "sar_to_optical", config)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["physical_null_change"])
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)


class _GatedToyModel(nn.Module):
    """Expose independent candidate/gate parameters for objective tests."""

    def __init__(self) -> None:
        super().__init__()
        self.candidate = nn.Parameter(torch.tensor(0.25))
        self.confidence = nn.Parameter(torch.tensor(-4.0))

    def forward(self, **inputs: object) -> SimpleNamespace:
        anchor = inputs["target_anchor"]
        assert isinstance(anchor, torch.Tensor)
        candidate = anchor + self.candidate * torch.ones_like(anchor)
        confidence_logits = self.confidence * torch.ones_like(anchor[:, :1])
        confidence = torch.sigmoid(confidence_logits)
        physical = anchor + confidence * (candidate - anchor)
        return SimpleNamespace(
            physical=physical,
            candidate_physical=candidate,
            transport_confidence=confidence,
            transport_confidence_logits=confidence_logits,
            transport_evidence=torch.ones_like(confidence),
            log_variance=torch.zeros_like(anchor),
        )


class _CounterfactualConfidenceToyModel(nn.Module):
    """Emit a learnable open gate only for a shuffled source history."""

    def __init__(
        self,
        *,
        wrong_logit: float,
        correct_logit: float = 2.0,
        evidence: float = 1.0,
        logits: bool = True,
    ) -> None:
        super().__init__()
        self.wrong_logit = nn.Parameter(torch.tensor(wrong_logit))
        self.correct_logit = nn.Parameter(torch.tensor(correct_logit))
        self.evidence = float(evidence)
        self.logits = bool(logits)

    def forward(self, **inputs: object) -> SimpleNamespace:
        observations = inputs["observations"]
        source_anchor = inputs["source_anchor"]
        target_anchor = inputs["target_anchor"]
        assert isinstance(observations, torch.Tensor)
        assert isinstance(source_anchor, torch.Tensor)
        assert isinstance(target_anchor, torch.Tensor)
        # ``source_shuffle_batch`` exchanges observations but leaves this
        # recipient's anchor in place.  Correct histories are an exact anchor
        # copy in this toy, while shuffled histories activate the parameter.
        matches_recipient = (observations[:, 0] - source_anchor).abs().flatten(1).amax(dim=1) == 0.0
        wrong_mask = (~matches_recipient).to(target_anchor).view(-1, 1, 1, 1)
        confidence_logits = (
            wrong_mask * self.wrong_logit + (1.0 - wrong_mask) * self.correct_logit
        ) * torch.ones_like(target_anchor[:, :1])
        confidence = torch.sigmoid(confidence_logits)
        output = SimpleNamespace(
            physical=target_anchor,
            candidate_physical=target_anchor,
            transport_confidence=confidence,
            transport_evidence=torch.full_like(confidence, self.evidence),
            log_variance=torch.zeros_like(target_anchor),
        )
        if self.logits:
            output.transport_confidence_logits = confidence_logits
        return output


def _counterfactual_confidence_batch(*, batch_size: int = 2) -> dict[str, object]:
    batch = _batch(2, 10, batch_size=batch_size, changed=False)
    source_anchor = batch["source_anchor"]
    observations = batch["observations"]
    assert isinstance(source_anchor, torch.Tensor)
    assert isinstance(observations, torch.Tensor)
    source_anchor[:, 0] = torch.arange(batch_size, dtype=source_anchor.dtype).view(-1, 1, 1)
    source_anchor[:, 1] = -source_anchor[:, 0]
    batch["observations"] = source_anchor[:, None].expand_as(observations).clone()
    return batch


def _counterfactual_confidence_config(*, autocast_bfloat16: bool = False) -> SOPATTrainConfig:
    return SOPATTrainConfig(
        stage="physical",
        autocast_bfloat16=autocast_bfloat16,
        physical_charbonnier_weight=1e-12,
        physical_gradient_weight=0.0,
        physical_optical_spectral_weight=0.0,
        physical_optical_ndvi_weight=0.0,
        physical_sar_statistics_weight=0.0,
        physical_anchor_delta_weight=0.0,
        physical_null_change_weight=0.0,
        physical_null_change_probability=0.0,
        physical_nll_weight=0.0,
        physical_permutation_weight=0.0,
        physical_permutation_probability=0.0,
        physical_anchor_regret_weight=0.0,
        candidate_weight=0.0,
        utility_weight=0.0,
        source_shuffle_weight=0.0,
        source_shuffle_probability=1.0,
        counterfactual_confidence_weight=1.0,
    )


def test_counterfactual_confidence_has_symmetric_authenticity_gradients() -> None:
    model = _CounterfactualConfidenceToyModel(wrong_logit=4.0)
    loss, metrics = sopat_direction_objective(
        model,
        _counterfactual_confidence_batch(),
        "sar_to_optical",
        _counterfactual_confidence_config(),
        generator=torch.Generator().manual_seed(3),
    )
    loss.backward()

    assert metrics["physical_counterfactual_confidence_binary"].item() > 2.0
    assert metrics["physical_counterfactual_confidence_margin"].item() > 0.19
    assert metrics["physical_counterfactual_confidence"].item() == pytest.approx(
        metrics["physical_counterfactual_confidence_binary"].item()
        + metrics["physical_counterfactual_confidence_margin"].item()
    )
    assert model.wrong_logit.grad is not None
    assert model.wrong_logit.grad.item() > 0.0
    assert model.correct_logit.grad is not None
    assert model.correct_logit.grad.item() < 0.0


def test_counterfactual_confidence_rejects_both_closed_logits() -> None:
    model = _CounterfactualConfidenceToyModel(wrong_logit=-20.0, correct_logit=-20.0)
    loss, metrics = sopat_direction_objective(
        model,
        _counterfactual_confidence_batch(),
        "sar_to_optical",
        _counterfactual_confidence_config(),
        generator=torch.Generator().manual_seed(3),
    )
    loss.backward()

    assert metrics["physical_counterfactual_confidence_binary"].item() > 9.9
    assert metrics["physical_counterfactual_confidence_margin"].item() == pytest.approx(0.10)
    assert metrics["physical_counterfactual_confidence"].item() > 10.0
    assert model.correct_logit.grad is not None and model.correct_logit.grad.item() < 0.0
    assert model.wrong_logit.grad is not None and model.wrong_logit.grad.item() > 0.0


def test_counterfactual_confidence_is_near_zero_for_well_separated_logits() -> None:
    model = _CounterfactualConfidenceToyModel(wrong_logit=-20.0, correct_logit=20.0)
    _loss, metrics = sopat_direction_objective(
        model,
        _counterfactual_confidence_batch(),
        "sar_to_optical",
        _counterfactual_confidence_config(),
        generator=torch.Generator().manual_seed(3),
    )

    assert metrics["physical_counterfactual_confidence_binary"].item() < 1.0e-7
    assert metrics["physical_counterfactual_confidence_margin"].item() == 0.0
    assert metrics["physical_counterfactual_confidence"].item() < 1.0e-7


def test_counterfactual_confidence_skips_legacy_wrong_route_without_gate_fields() -> None:
    model = _TwoHeadPhysicalModel()
    _loss, metrics = sopat_direction_objective(
        model,
        _counterfactual_confidence_batch(),
        "sar_to_optical",
        _counterfactual_confidence_config(),
        generator=torch.Generator().manual_seed(3),
    )

    assert metrics["physical_counterfactual_confidence"].item() == 0.0


@pytest.mark.parametrize("zero_support", ("evidence", "target_valid", "source_anchor_valid"))
def test_counterfactual_confidence_zero_support_has_exact_zero_loss(zero_support: str) -> None:
    model = _CounterfactualConfidenceToyModel(
        wrong_logit=4.0, evidence=0.0 if zero_support == "evidence" else 1.0
    )
    batch = _counterfactual_confidence_batch()
    if zero_support != "evidence":
        support = batch[zero_support]
        assert isinstance(support, torch.Tensor)
        support.zero_()
    loss, metrics = sopat_direction_objective(
        model,
        batch,
        "sar_to_optical",
        _counterfactual_confidence_config(),
        generator=torch.Generator().manual_seed(3),
    )
    loss.backward()

    assert metrics["physical_counterfactual_confidence"].item() == 0.0
    assert metrics["physical_counterfactual_confidence_binary"].item() == 0.0
    assert metrics["physical_counterfactual_confidence_margin"].item() == 0.0
    assert model.wrong_logit.grad is not None
    assert model.wrong_logit.grad.item() == 0.0
    assert model.correct_logit.grad is not None
    assert model.correct_logit.grad.item() == 0.0


def test_counterfactual_confidence_probability_fallback_is_cpu_bf16_safe() -> None:
    model = _CounterfactualConfidenceToyModel(wrong_logit=4.0, logits=False)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, metrics = sopat_direction_objective(
            model,
            _counterfactual_confidence_batch(),
            "sar_to_optical",
            _counterfactual_confidence_config(autocast_bfloat16=True),
            generator=torch.Generator().manual_seed(3),
        )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["physical_counterfactual_confidence"])
    assert model.wrong_logit.grad is not None and torch.isfinite(model.wrong_logit.grad)


def test_counterfactual_confidence_rejects_wrong_output_shapes() -> None:
    class _MalformedEvidenceModel(_CounterfactualConfidenceToyModel):
        def forward(self, **inputs: object) -> SimpleNamespace:
            output = super().forward(**inputs)
            observations = inputs["observations"]
            assert isinstance(observations, torch.Tensor)
            if not torch.equal(observations, inputs["source_anchor"][:, None].expand_as(observations)):
                output.transport_evidence = output.transport_evidence[:, :, :-1, :]
            return output

    with pytest.raises(ValueError, match="source-shuffle transport_evidence"):
        sopat_direction_objective(
            _MalformedEvidenceModel(wrong_logit=4.0),
            _counterfactual_confidence_batch(),
            "sar_to_optical",
            _counterfactual_confidence_config(),
            generator=torch.Generator().manual_seed(3),
        )


@pytest.mark.parametrize(
    "config_name",
    (
        "sopat_v4_feasibility_local.yaml",
        "sopat_v4_full_chunk.yaml",
        "sopat_v4_full_raw.yaml",
        "sopat_v4_smoke.yaml",
    ),
)
def test_counterfactual_confidence_weight_is_explicit_in_all_sopat_configs(config_name: str) -> None:
    path = Path(__file__).parents[1] / "configs" / config_name
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(values, dict)
    training = values["training"]
    assert isinstance(training, dict)
    objective = training["objective"]
    stages = training["stages"]
    assert isinstance(objective, dict)
    assert isinstance(stages, dict)
    physical_stage = stages["physical"]
    assert isinstance(physical_stage, dict)
    payload = {**objective, **physical_stage, "stage": "physical"}
    payload.pop("steps")
    config = SOPATTrainConfig.from_mapping(payload)
    assert config.counterfactual_confidence_weight == pytest.approx(0.10)
    assert config.counterfactual_confidence_binary_weight == pytest.approx(1.0)
    assert config.counterfactual_confidence_margin == pytest.approx(0.10)


@pytest.mark.parametrize("value", (-1.0, float("inf"), float("nan")))
def test_counterfactual_confidence_weight_requires_a_finite_nonnegative_value(value: float) -> None:
    with pytest.raises(ValueError, match="counterfactual_confidence_weight|loss weights"):
        SOPATTrainConfig(counterfactual_confidence_weight=value)


@pytest.mark.parametrize("value", (-1.0, float("inf"), float("nan")))
def test_counterfactual_confidence_binary_weight_requires_a_finite_nonnegative_value(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="counterfactual_confidence_binary_weight|loss weights"):
        SOPATTrainConfig(counterfactual_confidence_binary_weight=value)


def test_confidence_only_scope_freezes_everything_except_two_confidence_heads() -> None:
    model = SOPAT(
        SOPATConfig(
            width=8,
            hidden=32,
            encoder_depth=1,
            heads=4,
            adapter_rank=8,
            transport_heads=4,
            anchor_window_size=2,
        )
    )

    configure_sopat_stage(model, "physical", trainable_scope="confidence_only")

    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable == {
        "renderers.optical.confidence.weight",
        "renderers.optical.confidence.bias",
        "renderers.sar.confidence.weight",
        "renderers.sar.confidence.bias",
    }


def test_confidence_only_scope_rejects_factorizer_stage() -> None:
    with pytest.raises(ValueError, match="physical stage"):
        SOPATTrainConfig(stage="factorizer", trainable_scope="confidence_only")


def test_candidate_auxiliary_and_utility_are_finite_with_closed_gate() -> None:
    config = SOPATTrainConfig(
        stage="physical",
        autocast_bfloat16=False,
        physical_null_change_probability=0.0,
        physical_permutation_probability=0.0,
        source_shuffle_probability=0.0,
    )
    model = _GatedToyModel()
    batch = _batch(2, 10, batch_size=2)
    loss, metrics = sopat_direction_objective(model, batch, "sar_to_optical", config)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["physical_candidate"])
    assert torch.isfinite(metrics["physical_utility"])
    assert model.candidate.grad is not None and model.candidate.grad.abs().item() > 0.0
    assert model.confidence.grad is not None and torch.isfinite(model.confidence.grad)


def test_utility_logits_are_amp_safe_and_no_evidence_pixels_are_excluded() -> None:
    config = SOPATTrainConfig(
        stage="physical",
        autocast_bfloat16=True,
        physical_null_change_probability=0.0,
        physical_permutation_probability=0.0,
        source_shuffle_probability=0.0,
    )
    model = _GatedToyModel()
    batch = _batch(2, 10, batch_size=2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, metrics = sopat_direction_objective(model, batch, "sar_to_optical", config)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["physical_utility"])
    assert model.confidence.grad is not None and torch.isfinite(model.confidence.grad)

    # An all-zero evidence mask must remove every utility-supervision pixel;
    # the result stays finite and produces no confidence gradient from BCE.
    class _NoEvidenceToyModel(_GatedToyModel):
        def forward(self, **inputs: object) -> SimpleNamespace:
            output = super().forward(**inputs)
            anchor = inputs["target_anchor"]
            assert isinstance(anchor, torch.Tensor)
            # Keep every non-utility term independent of the gate parameter;
            # any remaining confidence gradient would therefore prove that
            # no-evidence pixels still participate in BCE supervision.
            output.physical = anchor
            output.transport_evidence = torch.zeros_like(output.transport_confidence)
            return output

    no_evidence = _NoEvidenceToyModel()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        no_evidence_loss, no_evidence_metrics = sopat_direction_objective(
            no_evidence, batch, "sar_to_optical", config
        )
    no_evidence_loss.backward()
    assert torch.isfinite(no_evidence_loss)
    assert torch.isfinite(no_evidence_metrics["physical_utility"])
    assert no_evidence.confidence.grad is not None
    assert no_evidence.confidence.grad.abs().item() == 0.0


def test_legacy_probability_only_utility_fallback_is_cpu_autocast_safe() -> None:
    class _LegacyProbabilityToyModel(_GatedToyModel):
        def forward(self, **inputs: object) -> SimpleNamespace:
            output = super().forward(**inputs)
            del output.transport_confidence_logits
            del output.transport_evidence
            return output

    config = SOPATTrainConfig(
        stage="physical",
        autocast_bfloat16=True,
        physical_null_change_probability=0.0,
        physical_permutation_probability=0.0,
        source_shuffle_probability=0.0,
    )
    model = _LegacyProbabilityToyModel()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, metrics = sopat_direction_objective(
            model, _batch(2, 10, batch_size=2), "sar_to_optical", config
        )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["physical_utility"])
    assert model.confidence.grad is not None and torch.isfinite(model.confidence.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 autocast is unavailable")
def test_utility_logits_are_cuda_bf16_safe_when_available() -> None:
    config = SOPATTrainConfig(
        stage="physical",
        autocast_bfloat16=True,
        physical_null_change_probability=0.0,
        physical_permutation_probability=0.0,
        source_shuffle_probability=0.0,
    )
    model = _GatedToyModel().cuda()
    batch = {
        name: value.cuda() if isinstance(value, torch.Tensor) else value
        for name, value in _batch(2, 10, batch_size=2).items()
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, metrics = sopat_direction_objective(model, batch, "sar_to_optical", config)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["physical_utility"])
    assert model.confidence.grad is not None and torch.isfinite(model.confidence.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 autocast is unavailable")
def test_legacy_probability_utility_fallback_is_cuda_bf16_safe() -> None:
    class _LegacyProbabilityToyModel(_GatedToyModel):
        def forward(self, **inputs: object) -> SimpleNamespace:
            output = super().forward(**inputs)
            del output.transport_confidence_logits
            del output.transport_evidence
            return output

    config = SOPATTrainConfig(
        stage="physical",
        autocast_bfloat16=True,
        physical_null_change_probability=0.0,
        physical_permutation_probability=0.0,
        source_shuffle_probability=0.0,
    )
    model = _LegacyProbabilityToyModel().cuda()
    batch = {
        name: value.cuda() if isinstance(value, torch.Tensor) else value
        for name, value in _batch(2, 10, batch_size=2).items()
    }
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, metrics = sopat_direction_objective(model, batch, "sar_to_optical", config)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["physical_utility"])
    assert model.confidence.grad is not None and torch.isfinite(model.confidence.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 autocast is unavailable")
def test_native_sopat_physical_coupled_step_is_cuda_bf16_safe() -> None:
    model = SOPAT(
        SOPATConfig(
            width=8,
            hidden=32,
            encoder_depth=1,
            heads=4,
            adapter_rank=8,
            transport_heads=4,
            anchor_window_size=2,
        )
    ).cuda()
    model.set_training_stage("physical")
    config = SOPATTrainConfig(
        stage="physical",
        autocast_bfloat16=True,
        physical_null_change_probability=0.0,
        physical_permutation_probability=0.0,
        source_shuffle_probability=0.0,
    )
    batches = {
        direction: {
            name: value.cuda() if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
        }
        for direction, batch in {
            # H/8 is 1x1 for this compact fixture, so retain two examples for
            # GroupNorm's training-mode support requirement.
            "sar_to_optical": _batch(2, 10, batch_size=2),
            "optical_to_sar": _batch(10, 2, batch_size=2),
        }.items()
    }
    for batch in batches.values():
        observations = batch["observations"]
        assert isinstance(observations, torch.Tensor)
        observations.add_(0.05)
    module = SOPATTrainingModule(model, config)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)

    result = train_coupled_step(module, optimizer, batches, config)

    assert math.isfinite(result.total_loss)
    assert math.isfinite(result.gradient_norm)
    assert all(math.isfinite(value) for value in result.direction_losses.values())


def test_structural_losses_ignore_invalid_nan_and_extreme_pixels() -> None:
    valid = torch.ones(1, 1, 9, 9)
    valid[..., 4, 4] = 0.0
    valid[..., 0, 0] = 0.0
    target = torch.zeros(1, 2, 9, 9)
    prediction = target.clone()
    candidate = target.clone()
    anchor = target.clone()
    prediction[..., 4, 4] = float("nan")
    candidate[..., 4, 4] = float("nan")
    anchor[..., 4, 4] = float("nan")
    prediction[..., 0, 0] = 1.0e30
    candidate[..., 0, 0] = -1.0e30
    anchor[..., 0, 0] = 1.0e30

    structural = _structural_error(prediction, target, valid, kernel_size=5)
    oracle = _utility_oracle(candidate, anchor, target, valid, temperature=0.02, kernel_size=5)

    assert torch.isfinite(structural).all()
    assert torch.isfinite(oracle).all()
    torch.testing.assert_close(structural, torch.full_like(structural, 1.0e-4), atol=1.0e-7, rtol=0.0)


def test_source_shuffle_batch_is_a_local_derangement_and_singleton_counterfactual_is_zero() -> None:
    batch = _batch(2, 10, batch_size=4)
    values = batch["observations"]
    assert isinstance(values, torch.Tensor)
    values[:, :, :, 0, 0] = torch.arange(4)[:, None, None]
    shuffled = source_shuffle_batch(batch, generator=torch.Generator().manual_seed(3))
    shuffled_values = shuffled["observations"]
    assert isinstance(shuffled_values, torch.Tensor)
    assert not torch.equal(shuffled_values[:, 0, 0, 0, 0], values[:, 0, 0, 0, 0])

    config = SOPATTrainConfig(
        stage="physical",
        autocast_bfloat16=False,
        physical_null_change_probability=0.0,
        physical_permutation_probability=0.0,
        source_shuffle_probability=1.0,
    )
    singleton = _batch(2, 10, batch_size=1)
    model = _GatedToyModel()
    _loss, metrics = sopat_direction_objective(model, singleton, "sar_to_optical", config)
    assert metrics["physical_source_shuffle"].item() == 0.0
    assert metrics["physical_counterfactual_confidence"].item() == 0.0


class _FactorizerSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.calls: list[set[str]] = []

    def forward(self, **_inputs: object) -> object:
        raise AssertionError("factorizer validation must not invoke the full observation forward")

    def factorize_anchors(self, **inputs: object) -> SimpleNamespace:
        self.calls.append(set(inputs))
        source = inputs["source_anchor"]
        target = inputs["target_anchor"]
        assert isinstance(source, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        batch_size, _, height, width = source.shape
        common = self.scale * torch.ones((batch_size, 4, height, width), device=source.device)
        private = (1.0 - self.scale) * torch.ones_like(common)
        return SimpleNamespace(
            source_anchor_reconstruction=source * self.scale,
            target_anchor_reconstruction=target * self.scale,
            common_source=common,
            common_target=common * 0.5,
            private_source=private,
            private_target=private * 0.5,
        )


def test_factorizer_shortcut_and_validation_never_send_labels_or_observations() -> None:
    config = SOPATTrainConfig(stage="factorizer", autocast_bfloat16=False)
    model = _FactorizerSpy()
    batch = _batch(2, 10)
    loss, _ = sopat_direction_objective(model, batch, "sar_to_optical", config)
    loss.backward()
    validation = evaluate_factorizer_loaders(
        model,
        {
            "sar_to_optical": [batch],
            "optical_to_sar": [_batch(10, 2)],
        },
        config,
    )

    expected = {
        "source_anchor",
        "source_anchor_valid",
        "target_anchor",
        "target_anchor_valid",
        "source_sensor",
        "target_sensor",
    }
    assert model.calls and all(keys == expected for keys in model.calls)
    assert "target" not in model.calls[0]
    assert "observations" not in model.calls[0]
    assert validation.weighted_loss > 0.0
    assert set(validation.direction_losses) == {"sar_to_optical", "optical_to_sar"}


class _EncoderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)


def _protocol() -> dict[str, dict[str, str]]:
    return {
        "sar_to_optical": {"index": "a" * 64},
        "optical_to_sar": {"index": "b" * 64},
    }


def test_checkpoint_is_bidirectional_strict_and_v3_encoder_initialization(tmp_path) -> None:
    model = _EncoderModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = SOPATTrainConfig(stage="factorizer", autocast_bfloat16=False)
    checkpoint = tmp_path / "sopat.pt"
    save_sopat_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        ema=ModelEMA.create(model, 0.9),
        model_config={"architecture": "test"},
        train_config=config,
        protocol_hashes=_protocol(),
        global_step=7,
        best_metrics={},
    )
    restored = _EncoderModel()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = load_sopat_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        ema=ModelEMA.create(restored, 0.9),
        model_config={"architecture": "test"},
        train_config=config,
        protocol_hashes=_protocol(),
    )
    assert payload["global_step"] == 7
    with pytest.raises(RuntimeError, match="protocol"):
        load_sopat_checkpoint(
            checkpoint,
            model=restored,
            optimizer=None,
            ema=None,
            model_config={"architecture": "test"},
            train_config=config,
            protocol_hashes={
                "sar_to_optical": {"index": "c" * 64},
                "optical_to_sar": {"index": "b" * 64},
            },
        )

    source = _EncoderModel()
    with torch.no_grad():
        source.encoder.weight.fill_(0.75)
    v3 = tmp_path / "v3.pt"
    ema_state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    ema_state["encoder.weight"].fill_(0.55)
    torch.save({"model": source.state_dict(), "ema": {"state": ema_state}}, v3)
    initialized = _EncoderModel()
    summary = initialize_from_v3_checkpoint(initialized, v3)
    assert any(name.startswith("encoder.") for name in summary["loaded"])
    assert summary["use_ema"] is True
    assert "encoder.weight" in summary["ema_overlaid"]
    assert torch.equal(initialized.encoder.weight, ema_state["encoder.weight"])
    assert not torch.equal(initialized.head.weight, source.head.weight)


def test_legacy_v4_initialization_allows_only_new_confidence_heads(tmp_path) -> None:
    config = SOPATConfig(
        width=8,
        hidden=32,
        encoder_depth=1,
        heads=4,
        adapter_rank=8,
        transport_heads=4,
        anchor_window_size=2,
    )
    source = SOPAT(config)
    legacy_state = {
        name: value
        for name, value in source.state_dict().items()
        if ".confidence." not in name
    }
    path = tmp_path / "legacy-v4.pt"
    torch.save(
        {
            "sopat_v4_format": 1,
            "family": "sopat_v4",
            "directions": ["sar_to_optical", "optical_to_sar"],
            "model_config": asdict(config),
            "protocol_hashes": _protocol(),
            "model": legacy_state,
        },
        path,
    )
    restored = SOPAT(config)
    summary = initialize_from_sopat_checkpoint(
        restored,
        path,
        model_config=asdict(config),
        protocol_hashes=_protocol(),
    )

    assert summary["initialized_missing_keys"] == [
        "renderers.optical.confidence.bias",
        "renderers.optical.confidence.weight",
        "renderers.sar.confidence.bias",
        "renderers.sar.confidence.weight",
    ]
    assert restored.renderers["optical"].confidence.bias.item() == pytest.approx(-2.0)


def test_factorizer_initialization_allows_only_pre_contrast_model_config(tmp_path) -> None:
    config = SOPATConfig(
        width=8,
        hidden=32,
        encoder_depth=1,
        heads=4,
        adapter_rank=8,
        transport_heads=4,
        anchor_window_size=2,
    )
    old_config = asdict(config)
    old_config.pop("transport_parameterization")

    def write_checkpoint(path, *, stage: str, model_config: dict[str, object]) -> None:
        torch.save(
            {
                "sopat_v4_format": 1,
                "family": "sopat_v4",
                "directions": ["sar_to_optical", "optical_to_sar"],
                "model_config": model_config,
                "train_config": {"stage": stage},
                "protocol_hashes": _protocol(),
                "model": SOPAT(config).state_dict(),
            },
            path,
        )

    factorizer_path = tmp_path / "pre-contrast-factorizer.pt"
    write_checkpoint(factorizer_path, stage="factorizer", model_config=old_config)
    initialized = SOPAT(config)
    summary = initialize_from_sopat_checkpoint(
        initialized,
        factorizer_path,
        model_config=asdict(config),
        protocol_hashes=_protocol(),
    )
    assert summary["source_train_stage"] == "factorizer"
    assert summary["initialized_missing_keys"] == []

    physical_path = tmp_path / "pre-contrast-physical.pt"
    write_checkpoint(physical_path, stage="physical", model_config=old_config)
    with pytest.raises(RuntimeError, match="model configuration"):
        initialize_from_sopat_checkpoint(
            SOPAT(config),
            physical_path,
            model_config=asdict(config),
            protocol_hashes=_protocol(),
        )
    with pytest.raises(RuntimeError, match="model configuration"):
        load_sopat_checkpoint(
            physical_path,
            model=SOPAT(config),
            optimizer=None,
            ema=None,
            model_config=asdict(config),
            train_config=SOPATTrainConfig(stage="physical", autocast_bfloat16=False),
            protocol_hashes=_protocol(),
        )

    exact_path = tmp_path / "contrastive-factorizer.pt"
    write_checkpoint(exact_path, stage="factorizer", model_config=asdict(config))
    exact_summary = initialize_from_sopat_checkpoint(
        SOPAT(config),
        exact_path,
        model_config=asdict(config),
        protocol_hashes=_protocol(),
    )
    assert exact_summary["initialized_missing_keys"] == []


def test_sopat_initialization_can_select_checkpoint_ema_weights(tmp_path) -> None:
    config = SOPATConfig(
        width=8,
        hidden=32,
        encoder_depth=1,
        heads=4,
        adapter_rank=8,
        transport_heads=4,
        anchor_window_size=2,
    )
    source = SOPAT(config)
    raw_state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    ema_state = {
        name: (value.detach().clone() + 0.25 if torch.is_floating_point(value) else value.detach().clone())
        for name, value in raw_state.items()
    }
    path = tmp_path / "ema-init.pt"
    torch.save(
        {
            "sopat_v4_format": 1,
            "family": "sopat_v4",
            "directions": ["sar_to_optical", "optical_to_sar"],
            "model_config": asdict(config),
            "train_config": {"stage": "physical"},
            "protocol_hashes": _protocol(),
            "model": raw_state,
            "ema": {"decay": 0.999, "state": ema_state},
        },
        path,
    )
    restored = SOPAT(config)

    summary = initialize_from_sopat_checkpoint(
        restored,
        path,
        model_config=asdict(config),
        protocol_hashes=_protocol(),
        use_ema=True,
    )

    assert summary["initialized_weight_source"] == "ema"
    for name, value in restored.state_dict().items():
        assert torch.equal(value, ema_state[name])


def test_sopat_ema_initialization_fails_without_complete_ema_state(tmp_path) -> None:
    config = SOPATConfig(
        width=8,
        hidden=32,
        encoder_depth=1,
        heads=4,
        adapter_rank=8,
        transport_heads=4,
        anchor_window_size=2,
    )
    source = SOPAT(config)
    path = tmp_path / "missing-ema.pt"
    torch.save(
        {
            "sopat_v4_format": 1,
            "family": "sopat_v4",
            "directions": ["sar_to_optical", "optical_to_sar"],
            "model_config": asdict(config),
            "train_config": {"stage": "physical"},
            "protocol_hashes": _protocol(),
            "model": source.state_dict(),
        },
        path,
    )
    with pytest.raises(RuntimeError, match="no EMA state"):
        initialize_from_sopat_checkpoint(
            SOPAT(config),
            path,
            model_config=asdict(config),
            protocol_hashes=_protocol(),
            use_ema=True,
        )


def test_evaluation_is_stratified_and_has_both_direction_schemas() -> None:
    optical = _batch(2, 10, batch_size=2, observations=2, changed=True)
    optical["target"][0] = optical["target_anchor"][0]  # type: ignore[index]
    optical["observation_present"][0, 1] = False  # type: ignore[index]
    optical["task_mode"] = ["translation", "forecast"]
    sar = _batch(10, 2, batch_size=2, observations=2, changed=True)
    sar["target"][0] = sar["target_anchor"][0]  # type: ignore[index]
    sar["observation_present"][0, 1] = False  # type: ignore[index]
    sar["task_mode"] = ["translation", "forecast"]

    report = evaluate_sopat_loaders(
        None,
        {"sar_to_optical": [optical], "optical_to_sar": [sar]},
        variant=SOPATVariantConfig("anchor_copy"),
    )
    optical_metrics = report["directions"]["sar_to_optical"]["all"]["all"]  # type: ignore[index]
    sar_metrics = report["directions"]["optical_to_sar"]["all"]["all"]  # type: ignore[index]
    assert optical_metrics["sam_deg"] is not None
    assert optical_metrics["scene_improved_fraction"] is not None
    assert sar_metrics["sar_db_rmse"] is not None
    assert sar_metrics["sar_db_prediction_p01"] is not None
    regimes = report["directions"]["sar_to_optical"]["regimes"]  # type: ignore[index]
    assert "translation/n=one" in regimes
    assert "forecast/n=two_to_three" in regimes


def test_source_shuffle_rejects_singleton_and_has_no_fixed_points() -> None:
    singleton = _batch(2, 10, batch_size=1)
    with pytest.raises(ValueError, match="batch_size >= 2"):
        sopat_evaluation._source_shuffle_batch(singleton)

    batch = _batch(2, 10, batch_size=4)
    observations = batch["observations"]
    assert isinstance(observations, torch.Tensor)
    tagged = observations.clone()
    tagged[:, :, :, 0, 0] = torch.arange(4)[:, None, None]
    tagged_batch = {**batch, "observations": tagged}
    for seed in range(8):
        # Each input scene is deliberately encoded by a unique constant in the
        # first source-observation pixel; no row may retain its own history.
        tagged_shuffled = sopat_evaluation._source_shuffle_batch(
            tagged_batch,
            generator=torch.Generator().manual_seed(seed),
        )
        output = tagged_shuffled["observations"]
        assert isinstance(output, torch.Tensor)
        assert not torch.equal(output[:, 0, 0, 0, 0], tagged[:, 0, 0, 0, 0])


def _selection_report(
    *,
    optical_rmse: float,
    sar_rmse: float,
    optical_structural: float = 0.1,
    sar_structural: float = 0.2,
    variant: str | None = None,
) -> dict[str, object]:
    optical = {
        "rmse": optical_rmse,
        "anchor_rmse": 1.0,
        "sam_deg": 1.0,
        "anchor_sam_deg": 1.1,
        "ndvi_mae": 0.1,
        "anchor_ndvi_mae": 0.2,
        "edge_f1": 0.8,
        "anchor_edge_f1": 0.7,
        "scene_improved_fraction": 1.0,
        "structural_rmse": optical_structural,
    }
    sar = {
        "sar_db_rmse": sar_rmse,
        "sar_db_anchor_rmse": 10.0,
        "sar_db_bias": 0.0,
        "edge_f1": 0.8,
        "anchor_edge_f1": 0.81,
        "scene_improved_fraction": 1.0,
        "structural_rmse": sar_structural,
    }
    report: dict[str, object] = {
        "directions": {
            "sar_to_optical": {
                "all": {"all": optical},
                "regimes": {"translation/n=one": {"all": optical}},
            },
            "optical_to_sar": {
                "all": {"all": sar},
                "regimes": {"translation/n=one": {"all": sar}},
            },
        }
    }
    if variant is not None:
        report["variant"] = {"name": variant}
    return report


def test_source_shuffle_selection_gate_requires_both_direction_degradation() -> None:
    config = sopat_evaluation.SOPATSelectionConfig(
        phase="feasibility",
        required_tasks=("translation",),
        required_observation_counts=("one",),
        feasibility_overall_anchor_ratio=2.0,
        feasibility_bucket_anchor_ratio=2.0,
        feasibility_source_shuffle_min_degradation=0.01,
    )
    candidate = _selection_report(optical_rmse=0.5, sar_rmse=2.0)
    passing_shuffle = _selection_report(
        optical_rmse=0.51,
        sar_rmse=2.03,
        optical_structural=0.102,
        sar_structural=0.205,
        variant="source_shuffle",
    )
    passing = sopat_evaluation.select_sopat_candidate(
        candidate,
        config,
        source_shuffle_report=passing_shuffle,
    )
    assert passing.eligible

    failing_shuffle = _selection_report(
        optical_rmse=0.51,
        sar_rmse=2.0,
        optical_structural=0.102,
        sar_structural=0.2,
        variant="source_shuffle",
    )
    failing = sopat_evaluation.select_sopat_candidate(
        candidate,
        config,
        source_shuffle_report=failing_shuffle,
    )
    assert not failing.eligible
    assert any(
        "source_shuffle_insufficient_structural_degradation:optical_to_sar" in item
        for item in failing.failures
    )

    missing = sopat_evaluation.select_sopat_candidate(candidate, config)
    assert not missing.eligible
    assert any("missing_source_shuffle_structural_metrics" in item for item in missing.failures)

    invalid = sopat_evaluation.select_sopat_candidate(
        candidate,
        config,
        source_shuffle_report=_selection_report(
            optical_rmse=0.51,
            sar_rmse=2.03,
            variant="anchor_copy",
        ),
    )
    assert not invalid.eligible
    assert any("missing_source_shuffle_structural_metrics" in item for item in invalid.failures)
