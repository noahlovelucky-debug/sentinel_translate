from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sentinel_v3.paired_temporal_training import (
    PairedTemporalTrainConfig,
    apply_observation_dropout,
    count_stratified_observation_subsampling,
    effective_valid,
    evaluate_paired_temporal_batches,
    forward_paired_temporal,
    load_paired_temporal_checkpoint,
    paired_temporal_objective,
    regime_labels,
    save_paired_temporal_checkpoint,
    set_paired_temporal_stage,
    validation_regime_variants,
)
from sentinel_v3.paired_temporal_v2 import (
    PairedTemporalConfig,
    SparsePairedAnchorTransport,
)


def _model() -> SparsePairedAnchorTransport:
    return SparsePairedAnchorTransport(
        PairedTemporalConfig(width=32, latent_channels=8, attention_heads=4, flow_steps=1)
    )


def _tensors() -> dict[str, torch.Tensor]:
    return {
        "observations": torch.randn(2, 4, 2, 32, 32).tanh(),
        "observation_valid": torch.ones(2, 4, 1, 32, 32),
        "observation_days": torch.tensor([[-30.0, -15.0, -5.0, 0.0]]).expand(2, -1).clone(),
        "observation_present": torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        ),
        "source_anchor": torch.randn(2, 2, 32, 32).tanh(),
        "source_anchor_valid": torch.ones(2, 1, 32, 32),
        "target_anchor": torch.randn(2, 10, 32, 32).tanh(),
        "target_anchor_valid": torch.ones(2, 1, 32, 32),
        "anchor_days": torch.full((2,), -40.0),
        "target": torch.randn(2, 10, 32, 32).tanh(),
        "target_valid": torch.ones(2, 1, 32, 32),
    }


def test_observation_dropout_never_resurrects_padding_or_removes_all_frames() -> None:
    tensors = _tensors()
    generator = torch.Generator().manual_seed(5)
    sparse = apply_observation_dropout(
        tensors,
        frame_probability=0.99,
        query_probability=1.0,
        generator=generator,
    )
    assert torch.all(sparse["observation_present"].sum(dim=1) >= 1)
    assert not bool(sparse["observation_present"][1, 2:].any())
    absent = ~sparse["observation_present"]
    assert torch.all(sparse["observations"][absent] == 0)
    assert torch.all(sparse["observation_valid"][absent] == 0)


def test_regime_labels_separate_count_and_task_mode() -> None:
    tensors = _tensors()
    bins, translation = regime_labels(tensors)
    assert bins.tolist() == [2, 1]
    assert translation.tolist() == [True, False]


@pytest.mark.parametrize(
    ("weights", "minimum", "maximum"),
    [((1.0, 0.0, 0.0), 1, 1), ((0.0, 1.0, 0.0), 2, 3), ((0.0, 0.0, 1.0), 4, 8)],
)
def test_count_stratified_subsampling_is_replayable_and_keeps_real_frames(
    weights: tuple[float, float, float], minimum: int, maximum: int
) -> None:
    present = torch.tensor(
        [[True, True, True, True, True, True, True, True, False]], dtype=torch.bool
    )
    first = count_stratified_observation_subsampling(
        present,
        one_frame_probability=weights[0],
        two_to_three_frame_probability=weights[1],
        four_plus_frame_probability=weights[2],
        generator=torch.Generator().manual_seed(19),
    )
    second = count_stratified_observation_subsampling(
        present,
        one_frame_probability=weights[0],
        two_to_three_frame_probability=weights[1],
        four_plus_frame_probability=weights[2],
        generator=torch.Generator().manual_seed(19),
    )
    assert torch.equal(first, second)
    assert minimum <= int(first.sum()) <= maximum
    assert not bool(first[0, -1])


def test_query_dropout_treats_negative_one_day_observations_as_query_time() -> None:
    tensors = _tensors()
    tensors["observation_days"] = torch.tensor(
        [[-8.0, -1.0, 0.0, 0.0], [-8.0, -2.0, 0.0, 0.0]]
    )
    tensors["observation_present"] = torch.tensor(
        [[True, True, False, False], [True, True, False, False]]
    )
    sparse = apply_observation_dropout(
        tensors,
        frame_probability=0.0,
        query_probability=1.0,
        translation_max_delta_days=1,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(
        sparse["observation_present"],
        torch.tensor(
            [[True, False, False, False], [True, True, False, False]]
        ),
    )
    _, translation = regime_labels(tensors)
    assert translation.tolist() == [True, False]


@pytest.mark.parametrize("stage", ("physical", "detail", "flow", "balance"))
def test_stage_objectives_are_finite_and_freeze_prior_stages(stage: str) -> None:
    model = _model()
    set_paired_temporal_stage(model, stage)
    tensors = _tensors()
    output = forward_paired_temporal(model, tensors, "sar_to_optical")
    config = PairedTemporalTrainConfig(direction="sar_to_optical", stage=stage)
    loss, metrics = paired_temporal_objective(model, output, tensors, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics and all(torch.isfinite(value) for value in metrics.values())
    physical_trainable = any(
        parameter.requires_grad for parameter in model.physical_head.parameters()
    )
    assert physical_trainable == (stage == "physical")
    if stage == "balance":
        assert float(torch.tanh(model.visual_scale.detach())) > 0.0
        assert not any(parameter.requires_grad for parameter in model.detail_head.parameters())
        assert model.detail_scale.requires_grad
        assert model.visual_scale.requires_grad


def test_zero_high_frequency_weight_has_zero_detail_loss_and_gradient() -> None:
    model = _model()
    set_paired_temporal_stage(model, "detail")
    tensors = _tensors()
    tensors["high_frequency_valid"] = torch.ones_like(tensors["target_valid"])
    tensors["high_frequency_weight"] = torch.zeros(2)
    output = forward_paired_temporal(model, tensors, "sar_to_optical")
    loss, _ = paired_temporal_objective(
        model,
        output,
        tensors,
        PairedTemporalTrainConfig(direction="sar_to_optical", stage="detail"),
    )
    assert loss.item() == 0.0
    loss.backward()
    for parameter in model.detail_head.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_query_dropout_removes_high_frequency_detail_supervision() -> None:
    model = _model()
    set_paired_temporal_stage(model, "detail")
    tensors = _tensors()
    tensors["high_frequency_valid"] = torch.ones_like(tensors["target_valid"])
    tensors["high_frequency_weight"] = torch.ones(2)
    sparse = apply_observation_dropout(
        tensors,
        frame_probability=0.0,
        query_probability=1.0,
        translation_max_delta_days=1,
        generator=torch.Generator().manual_seed(11),
    )
    output = forward_paired_temporal(model, sparse, "sar_to_optical")
    loss, _ = paired_temporal_objective(
        model,
        output,
        sparse,
        PairedTemporalTrainConfig(direction="sar_to_optical", stage="detail"),
    )
    assert loss.item() == 0.0


def test_effective_valid_excludes_dropped_observation_only_support() -> None:
    tensors = _tensors()
    tensors["observation_valid"].zero_()
    tensors["observation_valid"][:, 0, :, :, :16] = 1.0
    tensors["observation_present"] = torch.tensor(
        [[True, False, False, False], [True, False, False, False]]
    )
    valid = effective_valid(tensors)
    assert torch.all(valid[..., :, :16] == 1)
    assert torch.all(valid[..., :, 16:] == 0)
    tensors["observation_present"][:, 0] = False
    assert torch.count_nonzero(effective_valid(tensors)) == 0


def test_validation_regime_variants_are_causal_and_never_restore_padding() -> None:
    tensors = _tensors()
    tensors["observation_days"] = torch.tensor([[-20.0, -12.0, -6.0, -2.0, 0.0]]).expand(2, -1)
    tensors["observations"] = torch.cat(
        (tensors["observations"], torch.zeros(2, 1, 2, 32, 32)), dim=1
    )
    tensors["observation_valid"] = torch.cat(
        (tensors["observation_valid"], torch.zeros(2, 1, 1, 32, 32)), dim=1
    )
    tensors["observation_days"] = torch.cat(
        (tensors["observation_days"], torch.zeros(2, 0)), dim=1
    )
    tensors["observation_present"] = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, True]]
    )
    variants = dict(validation_regime_variants(tensors, translation_max_delta_days=1))
    assert set(variants) == {
        "translation/one",
        "translation/two_to_three",
        "translation/four_plus",
        "forecast/one",
        "forecast/two_to_three",
        "forecast/four_plus",
    }
    for key, variant in variants.items():
        present = variant["observation_present"]
        days = variant["observation_days"]
        assert torch.all(present <= tensors["observation_present"][0:1])
        if key.startswith("translation/"):
            assert bool((present & (days.abs() <= 1)).any())
        else:
            assert not bool((present & (days.abs() <= 1)).any())


def test_evaluator_expands_rich_sequences_and_is_deterministic() -> None:
    model = _model().train()
    tensors = _tensors()
    tensors["observations"] = torch.cat(
        (tensors["observations"], torch.randn(2, 1, 2, 32, 32).tanh()), dim=1
    )
    tensors["observation_valid"] = torch.cat(
        (tensors["observation_valid"], torch.ones(2, 1, 1, 32, 32)), dim=1
    )
    tensors["observation_days"] = torch.tensor(
        [[-20.0, -12.0, -6.0, -2.0, 0.0]]
    ).expand(2, -1)
    tensors["observation_present"] = torch.ones(2, 5, dtype=torch.bool)
    first = evaluate_paired_temporal_batches(
        model,
        [tensors],
        PairedTemporalTrainConfig(direction="sar_to_optical", stage="physical"),
    )
    second = evaluate_paired_temporal_batches(
        model,
        [tensors],
        PairedTemporalTrainConfig(direction="sar_to_optical", stage="physical"),
    )
    assert model.training
    assert set(first) == {
        "translation/one",
        "translation/two_to_three",
        "translation/four_plus",
        "forecast/one",
        "forecast/two_to_three",
        "forecast/four_plus",
    }
    assert first == second
    assert all(values["flow_objective"] == 0.0 for values in first.values())


def test_checkpoint_rejects_direction_and_model_config(tmp_path: Path) -> None:
    model = _model()
    config = PairedTemporalTrainConfig(direction="sar_to_optical")
    path = save_paired_temporal_checkpoint(
        tmp_path / "physical.pt",
        model=model,
        config=config,
        step=3,
    )
    load_paired_temporal_checkpoint(path, _model(), direction="sar_to_optical")
    with pytest.raises(RuntimeError, match="direction"):
        load_paired_temporal_checkpoint(path, _model(), direction="optical_to_sar")
    other = SparsePairedAnchorTransport(
        PairedTemporalConfig(width=64, latent_channels=8, attention_heads=4)
    )
    with pytest.raises(RuntimeError, match="configuration"):
        load_paired_temporal_checkpoint(path, other, direction="sar_to_optical")


def test_checkpoint_binds_validation_protocol(tmp_path: Path) -> None:
    model = _model()
    config = PairedTemporalTrainConfig(direction="sar_to_optical")
    path = save_paired_temporal_checkpoint(
        tmp_path / "bound.pt",
        model=model,
        config=config,
        step=1,
        protocol={"sha256": "expected"},
    )
    load_paired_temporal_checkpoint(
        path,
        _model(),
        direction="sar_to_optical",
        expected_protocol_sha256="expected",
    )
    with pytest.raises(RuntimeError, match="protocol hash"):
        load_paired_temporal_checkpoint(
            path,
            _model(),
            direction="sar_to_optical",
            expected_protocol_sha256="different",
        )
