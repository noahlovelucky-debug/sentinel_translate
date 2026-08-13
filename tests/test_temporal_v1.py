from __future__ import annotations

import pytest
import torch

from sentinel_v3.sensors import SENTINEL1, SENTINEL2
from sentinel_v3.temporal_training import (
    TemporalPilotConfig,
    calibrate_temporal_visual_release,
    evaluate_pilot,
    load_temporal_checkpoint,
    physical_objective,
    save_temporal_checkpoint,
    set_temporal_stage,
    train_temporal_pilot,
)
from sentinel_v3.temporal_v1 import CausalAnchorDeltaTransport, TemporalModelConfig


def _batch(
    *,
    batch: int = 2,
    frames: int = 4,
    size: int = 32,
) -> dict[str, torch.Tensor]:
    return {
        "source_values": torch.randn(batch, frames, 2, size, size),
        "source_valid": torch.ones(batch, frames, 1, size, size),
        "anchor_values": torch.randn(batch, 10, size, size).tanh(),
        "anchor_valid": torch.ones(batch, 1, size, size),
        "target_values": torch.randn(batch, 10, size, size).tanh(),
        "target_valid": torch.ones(batch, 1, size, size),
        "source_days": torch.tensor([[-30.0, -20.0, -10.0, 0.0]]).expand(batch, -1),
        "anchor_days": torch.full((batch,), -40.0),
    }


def _model() -> CausalAnchorDeltaTransport:
    return CausalAnchorDeltaTransport(
        TemporalModelConfig(width=32, latent_channels=8, temporal_heads=4, flow_steps=2)
    )


def test_physical_anchor_initialization_and_shapes() -> None:
    model = _model()
    batch = _batch()
    output = model(
        batch["source_values"],
        batch["source_valid"],
        batch["anchor_values"],
        batch["anchor_valid"],
        batch["source_days"],
        batch["anchor_days"],
        source_sensor=SENTINEL1,
        target_sensor=SENTINEL2,
    )
    assert output.physical.shape == batch["anchor_values"].shape
    assert output.log_variance.shape == output.physical.shape
    assert output.attention.shape == (2, 4, 1, 8, 8)
    torch.testing.assert_close(output.physical, batch["anchor_values"])
    assert output.deterministic_detail is not None
    assert output.visual_base is not None
    torch.testing.assert_close(output.deterministic_detail, torch.zeros_like(output.physical))
    torch.testing.assert_close(output.visual_base, output.physical)
    assert float(output.physical.max()) <= 1.0
    assert float(output.physical.min()) >= -1.0


@pytest.mark.parametrize(
    ("source_days", "anchor_days", "match"),
    [
        (torch.tensor([[-20.0, -10.0, 1.0, 0.0]]), torch.tensor([-30.0]), "future"),
        (torch.tensor([[-20.0, -10.0, 0.0, 0.0]]), torch.tensor([0.0]), "anchor"),
        (torch.tensor([[-181.0, -10.0, 0.0, 0.0]]), torch.tensor([-30.0]), "horizon"),
    ],
)
def test_temporal_model_rejects_noncausal_inputs(
    source_days: torch.Tensor, anchor_days: torch.Tensor, match: str
) -> None:
    model = _model()
    batch = _batch(batch=1)
    with pytest.raises(ValueError, match=match):
        model(
            batch["source_values"],
            batch["source_valid"],
            batch["anchor_values"],
            batch["anchor_valid"],
            source_days,
            anchor_days,
            source_sensor=SENTINEL1,
            target_sensor=SENTINEL2,
        )


def test_bounded_anchor_update_reports_preprojection_violation() -> None:
    anchor = torch.full((1, 2, 4, 4), 0.9)
    delta = torch.full_like(anchor, 10.0)
    bounded, violation = CausalAnchorDeltaTransport.bounded_anchor_update(anchor, delta)
    assert float(bounded.max()) <= 1.0
    assert float(bounded.min()) >= -1.0
    assert float(violation) == 1.0


def test_temporal_model_requires_four_aligned_spatial_grid() -> None:
    model = _model()
    with pytest.raises(ValueError, match="divisible by four"):
        model(
            torch.randn(1, 4, 2, 30, 30),
            torch.ones(1, 4, 1, 30, 30),
            torch.randn(1, 10, 30, 30).tanh(),
            torch.ones(1, 1, 30, 30),
            torch.tensor([[-30.0, -20.0, -10.0, 0.0]]),
            torch.tensor([-40.0]),
            source_sensor=SENTINEL1,
            target_sensor=SENTINEL2,
        )


def test_physical_loss_has_finite_backward() -> None:
    model = _model()
    batch = _batch()
    output = model(
        batch["source_values"],
        batch["source_valid"],
        batch["anchor_values"],
        batch["anchor_valid"],
        batch["source_days"],
        batch["anchor_days"],
        source_sensor=SENTINEL1,
        target_sensor=SENTINEL2,
    )
    loss, metrics = physical_objective(
        output,
        batch["target_values"],
        batch["target_valid"],
        gradient_weight=0.2,
        uncertainty_weight=0.05,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert model.physical_head.delta.features[-1].weight.grad is not None


def test_visual_sampling_is_seeded_and_does_not_mutate_physical() -> None:
    model = _model().eval()
    batch = _batch(batch=1)
    with torch.no_grad():
        output = model(
            batch["source_values"],
            batch["source_valid"],
            batch["anchor_values"],
            batch["anchor_valid"],
            batch["source_days"],
            batch["anchor_days"],
            source_sensor=SENTINEL1,
            target_sensor=SENTINEL2,
        )
        first = model.sample_visual(output, batch["anchor_valid"], SENTINEL2, seed=7)
        second = model.sample_visual(output, batch["anchor_valid"], SENTINEL2, seed=7)
    assert first.visual is not None
    assert first.stochastic_residual is not None
    torch.testing.assert_close(first.physical, output.physical)
    torch.testing.assert_close(first.visual, second.visual)
    torch.testing.assert_close(first.visual, output.physical)


def test_temporal_attention_handles_a_locally_missing_source_sequence() -> None:
    model = _model()
    batch = _batch(batch=1)
    batch["source_valid"][..., :4, :4] = 0.0
    output = model(
        batch["source_values"],
        batch["source_valid"],
        batch["anchor_values"],
        batch["anchor_valid"],
        batch["source_days"],
        batch["anchor_days"],
        source_sensor=SENTINEL1,
        target_sensor=SENTINEL2,
    )
    assert torch.isfinite(output.physical).all()
    assert torch.isfinite(output.attention).all()
    torch.testing.assert_close(
        output.attention.sum(dim=1), torch.ones_like(output.attention[:, 0])
    )


def test_optical_to_sar_arbitrary_direction_shape() -> None:
    model = _model()
    batch = _batch(batch=1)
    source = torch.randn(1, 4, 10, 32, 32).tanh()
    anchor = torch.randn(1, 2, 32, 32).tanh()
    output = model(
        source,
        batch["source_valid"],
        anchor,
        batch["anchor_valid"],
        batch["source_days"],
        batch["anchor_days"],
        source_sensor=SENTINEL2,
        target_sensor=SENTINEL1,
    )
    assert output.physical.shape == anchor.shape
    assert TemporalPilotConfig(direction="optical_to_sar").direction == "optical_to_sar"


@pytest.mark.parametrize(
    ("stage", "trainable", "frozen"),
    [
        ("physical", "physical_head", "detail_head"),
        ("detail", "detail_head", "physical_head"),
        ("flow", "bridge", "physical_head"),
        ("balance", "detail_head", "physical_head"),
    ],
)
def test_stage_freezing_keeps_physical_frozen_after_first_stage(
    stage: str, trainable: str, frozen: str
) -> None:
    model = _model()
    set_temporal_stage(model, stage)
    assert any(parameter.requires_grad for parameter in getattr(model, trainable).parameters())
    assert not any(parameter.requires_grad for parameter in getattr(model, frozen).parameters())
    if stage == "balance":
        assert model.detail_scale.requires_grad
        assert model.visual_scale.requires_grad


def test_temporal_checkpoint_rejects_cross_direction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = _model()
    config = TemporalPilotConfig(direction="sar_to_optical", amp=False)
    path = save_temporal_checkpoint(
        tmp_path / "physical.pt",
        model=model,
        optimizer=None,
        config=config,
        step=0,
        report={"test": True},
    )
    reloaded = _model()
    load_temporal_checkpoint(path, reloaded, direction="sar_to_optical")
    with pytest.raises(RuntimeError, match="direction"):
        load_temporal_checkpoint(path, _model(), direction="optical_to_sar")


def test_visual_release_calibration_closes_harmful_texture() -> None:
    class _NoisyTransport(CausalAnchorDeltaTransport):
        def sample_visual(self, output, anchor_valid, target_sensor, **kwargs):  # type: ignore[no-untyped-def]
            sampled = super().sample_visual(output, anchor_valid, target_sensor, **kwargs)
            assert sampled.visual is not None
            if float(torch.tanh(self.visual_scale)) > 0.0:
                sampled.visual = (sampled.visual + 0.5).clamp(-1.0, 1.0)
            return sampled

    model = _NoisyTransport(TemporalModelConfig(width=32, latent_channels=8, temporal_heads=4))
    dataset = _SyntheticTemporalDataset(2)
    config = TemporalPilotConfig(direction="sar_to_optical", batch_size=2, amp=False)
    result = calibrate_temporal_visual_release(
        model,
        dataset,
        config,
        torch.device("cpu"),
        detail_candidates=(0.0,),
        texture_candidates=(0.0, 1.0),
    )
    assert result["budget_satisfied"]
    assert result["texture_release"] == 0.0


def test_flow_loss_is_finite_and_balance_sees_release_parameters() -> None:
    model = _model()
    set_temporal_stage(model, "flow")
    batch = _batch()
    output = model(
        batch["source_values"],
        batch["source_valid"],
        batch["anchor_values"],
        batch["anchor_valid"],
        batch["source_days"],
        batch["anchor_days"],
        source_sensor=SENTINEL1,
        target_sensor=SENTINEL2,
    )
    losses = model.visual_flow_loss(
        output, batch["target_values"], batch["target_valid"], SENTINEL2
    )
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total)
    assert model.bridge.body[-1].weight.grad is not None
    set_temporal_stage(model, "balance")
    assert model.detail_scale.requires_grad and model.visual_scale.requires_grad


class _SyntheticTemporalDataset(torch.utils.data.Dataset[dict[str, object]]):
    """A deterministic temporal signal where the anchor alone is insufficient."""

    def __init__(self, count: int, *, seed: int = 5) -> None:
        self.count = count
        generator = torch.Generator().manual_seed(seed)
        self.anchor = torch.rand(count, 10, 32, 32, generator=generator) * 1.2 - 0.6
        # The source's newest frame carries a scene-level observed change.  It
        # is intentionally independent from the anchor, so copying the anchor
        # cannot solve the task, while a causal source encoder can.
        driver = torch.rand(count, 1, 1, 1, generator=generator) * 0.8 - 0.4
        driver = driver.expand(-1, -1, 32, 32)
        weights = torch.linspace(0.18, 0.34, 10).reshape(1, 10, 1, 1)
        self.target = (self.anchor + weights * driver).clamp(-0.95, 0.95)
        self.source = torch.stack(
            (
                driver.expand(-1, 2, -1, -1) * 0.25,
                driver.expand(-1, 2, -1, -1) * 0.5,
                driver.expand(-1, 2, -1, -1) * 0.75,
                driver.expand(-1, 2, -1, -1),
            ),
            dim=1,
        )

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "source_values": self.source[index],
            "source_valid": torch.ones(4, 1, 32, 32),
            "anchor_values": self.anchor[index],
            "anchor_valid": torch.ones(1, 32, 32),
            "target_values": self.target[index],
            "target_valid": torch.ones(1, 32, 32),
            "source_days": torch.tensor([-30.0, -20.0, -10.0, 0.0]),
            "anchor_days": torch.tensor(-40.0),
            "sample_id": f"synthetic-{index}",
            "direction": "sar_to_optical",
        }


def test_pilot_uses_temporal_source_on_a_known_identifiable_signal(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = _model()
    train = _SyntheticTemporalDataset(12)
    validation = _SyntheticTemporalDataset(6, seed=11)
    config = TemporalPilotConfig(
        direction="sar_to_optical",
        max_steps=400,
        batch_size=2,
        learning_rate=5e-3,
        amp=False,
        log_every=400,
    )
    report = train_temporal_pilot(
        model, train, validation, config, output_dir=tmp_path, device="cpu"
    )
    assert report["feasibility"]["train_overfit_improves_anchor"]
    assert report["final_train"]["anchor_improvement_percent"] > 10.0
    assert report["final_train"]["source_ablation_penalty_percent"] > 1.0
    assert (tmp_path / "temporal_pilot_last.pt").is_file()
    assert (tmp_path / "temporal_pilot_report.json").is_file()
    metrics = evaluate_pilot(model, validation, config, torch.device("cpu"))
    assert metrics["physical_rmse"] < metrics["anchor_rmse"]
