from __future__ import annotations

import torch

from sentinel_v3.losses import high_frequency_loss, highpass, low_frequency_loss
from sentinel_v3.model import SentinelV3
from sentinel_v3.training import JointObjective, _load_compatible_state


def test_highpass_removes_constant_low_frequency() -> None:
    values = torch.ones(2, 3, 32, 32) * 4.0
    residual = highpass(values)
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    assert float(low_frequency_loss(residual, torch.ones(2, 1, 32, 32))) == 0.0


def test_sar_frequency_loss_is_finite() -> None:
    prediction = highpass(torch.randn(2, 2, 32, 32))
    target = highpass(torch.randn(2, 2, 32, 32))
    loss, metrics = high_frequency_loss(
        prediction, target, torch.ones(2, 1, 32, 32), "sar"
    )
    assert torch.isfinite(loss)
    assert "speckle_scale" in metrics


def test_visual_stage_trains_residual_branch(tiny_model: SentinelV3) -> None:
    objective = JointObjective(tiny_model, [0.35, 0.35, 0.15, 0.15])
    batch_size = 4
    batch: dict[str, object] = {
        "s2": torch.rand(batch_size, 10, 32, 32),
        "sar": torch.randn(batch_size, 2, 32, 32) * 4 - 15,
        "s2_view": torch.rand(batch_size, 10, 32, 32),
        "sar_view": torch.randn(batch_size, 2, 32, 32) * 4 - 15,
        "s2_target": torch.rand(batch_size, 10, 32, 32),
        "sar_target": torch.randn(batch_size, 2, 32, 32) * 4 - 15,
        "valid": torch.ones(batch_size, 1, 32, 32),
        "metadata": torch.zeros(batch_size, 8),
        "input_gsd": torch.full((batch_size,), 10.0),
        "target_gsd": torch.full((batch_size,), 10.0),
        "delta_days": torch.zeros(batch_size, dtype=torch.long),
    }
    loss, metrics = objective(batch, "visual")
    loss.backward()
    assert torch.isfinite(loss)
    assert "sar2opt/hf_spectrum" in metrics
    assert "opt2sar/speckle_scale" in metrics
    assert any(parameter.grad is not None for parameter in tiny_model.residual_dit.parameters())
    assert all(parameter.grad is None for parameter in tiny_model.encoder.parameters())


def test_v3_state_migration_skips_new_or_incompatible_tensors(tiny_model: SentinelV3) -> None:
    state = tiny_model.state_dict()
    first_name = next(iter(state))
    partial = {first_name: state[first_name].clone(), "unknown.weight": torch.ones(1)}
    loaded, initialized = _load_compatible_state(tiny_model, partial)
    assert loaded == 1
    assert initialized == len(state) - 1


def test_balance_stage_updates_physical_and_visual(tiny_model: SentinelV3) -> None:
    objective = JointObjective(tiny_model, [0.35, 0.35, 0.15, 0.15])
    objective.visual_joint = True
    batch_size = 4
    batch: dict[str, object] = {
        "s2": torch.rand(batch_size, 10, 32, 32),
        "sar": torch.randn(batch_size, 2, 32, 32) * 4 - 15,
        "s2_view": torch.rand(batch_size, 10, 32, 32),
        "sar_view": torch.randn(batch_size, 2, 32, 32) * 4 - 15,
        "s2_target": torch.rand(batch_size, 10, 32, 32),
        "sar_target": torch.randn(batch_size, 2, 32, 32) * 4 - 15,
        "valid": torch.ones(batch_size, 1, 32, 32),
        "metadata": torch.zeros(batch_size, 8),
        "input_gsd": torch.full((batch_size,), 10.0),
        "target_gsd": torch.full((batch_size,), 10.0),
        "delta_days": torch.zeros(batch_size, dtype=torch.long),
    }
    loss, _ = objective(batch, "balance")
    loss.backward()
    assert any(parameter.grad is not None for parameter in tiny_model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in tiny_model.residual_dit.parameters())


def test_physical_stage_only_trains_translation_tasks(tiny_model: SentinelV3) -> None:
    objective = JointObjective(
        tiny_model, [0.35, 0.35, 0.15, 0.15], physical_alignment_samples=2
    )
    batch_size = 4
    batch: dict[str, object] = {
        "s2": torch.rand(batch_size, 10, 32, 32),
        "sar": torch.randn(batch_size, 2, 32, 32) * 4 - 15,
        "s2_view": torch.rand(batch_size, 10, 32, 32),
        "sar_view": torch.randn(batch_size, 2, 32, 32) * 4 - 15,
        "s2_target": torch.rand(batch_size, 10, 32, 32),
        "sar_target": torch.randn(batch_size, 2, 32, 32) * 4 - 15,
        "valid": torch.ones(batch_size, 1, 32, 32),
        "metadata": torch.zeros(batch_size, 8),
        "input_gsd": torch.full((batch_size,), 10.0),
        "target_gsd": torch.full((batch_size,), 10.0),
        "delta_days": torch.zeros(batch_size, dtype=torch.long),
    }
    loss, metrics = objective(batch, "physical")
    loss.backward()
    assert torch.isfinite(loss)
    assert "sar2opt/rmse" in metrics
    assert "opt2sar/rmse" in metrics
    assert "opt_self/rmse" not in metrics
    assert "sar_self/rmse" not in metrics
    assert "latent/info_nce" in metrics
