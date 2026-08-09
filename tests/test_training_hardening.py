from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import sentinel_v3.training as training_module
from sentinel_v3.config import DEFAULT, validate_config
from sentinel_v3.model import SceneEncoder
from sentinel_v3.training import (
    JointObjective,
    _accumulate_pcgrad_corrections,
    _data_loader_worker_options,
    _validate_protocol_binding,
)


def test_scene_encoder_checkpointing_preserves_forward_and_backward() -> None:
    torch.manual_seed(7)
    baseline = SceneEncoder(width=4, hidden=16, depth=2, heads=4, adapter_rank=2).train()
    checkpointed = copy.deepcopy(baseline).train()
    checkpointed.set_activation_checkpointing(True)
    values = torch.randn(2, 2, 16, 16, requires_grad=True)
    checkpoint_values = values.detach().clone().requires_grad_(True)
    descriptors = torch.randn(2, 8)
    valid = torch.ones(2, 1, 16, 16)
    condition = torch.randn(2, 11)

    baseline_output = baseline(values, descriptors, valid, condition, "optical")
    checkpoint_output = checkpointed(
        checkpoint_values, descriptors, valid, condition, "optical"
    )
    for left, right in zip(baseline_output, checkpoint_output, strict=True):
        torch.testing.assert_close(left, right)
    sum(value.square().mean() for value in baseline_output).backward()
    sum(value.square().mean() for value in checkpoint_output).backward()
    torch.testing.assert_close(values.grad, checkpoint_values.grad)
    for (name, left), (_, right) in zip(
        baseline.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            assert right.grad is not None
            torch.testing.assert_close(left.grad, right.grad)


def test_pcgrad_corrections_accumulate_each_microstep_without_double_averaging() -> None:
    first = [torch.tensor([1.0, -2.0]), None, torch.tensor([3.0])]
    second = [torch.tensor([4.0, 5.0]), torch.tensor([6.0]), None]
    accumulated = _accumulate_pcgrad_corrections(None, first)
    accumulated = _accumulate_pcgrad_corrections(accumulated, second)

    torch.testing.assert_close(accumulated[0], torch.tensor([5.0, 3.0]))
    torch.testing.assert_close(accumulated[1], torch.tensor([6.0]))
    torch.testing.assert_close(accumulated[2], torch.tensor([3.0]))
    single = _accumulate_pcgrad_corrections(None, first)
    for expected, actual in zip(first, single, strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(expected, actual)


def test_worker_loader_options_skip_persistent_settings_without_workers() -> None:
    config: dict[str, Any] = {
        "num_workers": 0,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }
    assert _data_loader_worker_options(config) == {"num_workers": 0}
    config["num_workers"] = 2
    assert _data_loader_worker_options(config) == {
        "num_workers": 2,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }


def test_train_runtime_option_validation() -> None:
    config = copy.deepcopy(DEFAULT)
    config["train"]["activation_checkpointing"] = "yes"
    with pytest.raises(TypeError, match="activation_checkpointing"):
        validate_config(config)
    config = copy.deepcopy(DEFAULT)
    config["train"]["persistent_workers"] = 1
    with pytest.raises(TypeError, match="persistent_workers"):
        validate_config(config)
    config = copy.deepcopy(DEFAULT)
    config["train"]["prefetch_factor"] = 0
    with pytest.raises(ValueError, match="prefetch_factor"):
        validate_config(config)
    config = copy.deepcopy(DEFAULT)
    config["train"]["physical_alignment_every"] = 0
    with pytest.raises(ValueError, match="physical_alignment_every"):
        validate_config(config)


def test_protocol_binding_preserves_physical_weight_init_and_rejects_downstream_mismatch() -> None:
    checkpoint = {"format_version": 4, "validation_protocol_hash": "old"}
    assert not _validate_protocol_binding(
        checkpoint, "current", stage="physical", resume=False
    )
    with pytest.raises(RuntimeError, match="validation_protocol_hash"):
        _validate_protocol_binding(checkpoint, "current", stage="flow", resume=False)
    with pytest.raises(RuntimeError, match="validation_protocol_hash"):
        _validate_protocol_binding(checkpoint, "current", stage="physical", resume=True)
    assert _validate_protocol_binding(
        {"format_version": 4, "validation_protocol_hash": "current"},
        "current",
        stage="flow",
        resume=False,
    )


class _AlignmentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def encode(self, values: Tensor, *args: object, **kwargs: object) -> tuple[Tensor]:
        del args, kwargs
        return (self.weight.expand(values.shape[0], 1, 2, 2),)


def test_sparse_physical_alignment_runs_on_interval_and_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _AlignmentModel()
    objective = JointObjective(
        model,  # type: ignore[arg-type]
        physical_alignment_samples=2,
        physical_alignment_weight=0.2,
        physical_alignment_every=4,
    )

    def physical_direction(*args: object, **kwargs: object) -> tuple[Tensor, dict[str, Tensor], tuple[Tensor, ...]]:
        del args, kwargs
        return model.weight * 0.0, {}, ()

    def alignment(*args: object, **kwargs: object) -> tuple[Tensor, dict[str, Tensor]]:
        del args, kwargs
        return model.weight, {"info_nce": model.weight.detach()}

    objective._physical_direction = physical_direction  # type: ignore[method-assign]
    monkeypatch.setattr(training_module, "latent_alignment", alignment)
    batch: dict[str, object] = {
        "s2": torch.zeros(2, 10, 8, 8),
        "sar": torch.zeros(2, 2, 8, 8),
        "s2_view": torch.zeros(2, 10, 8, 8),
        "sar_view": torch.zeros(2, 2, 8, 8),
        "s2_target": torch.zeros(2, 10, 8, 8),
        "sar_target": torch.zeros(2, 2, 8, 8),
        "valid": torch.ones(2, 1, 8, 8),
        "delta_days": torch.zeros(2, dtype=torch.long),
        "input_gsd": torch.full((2,), 10.0),
        "target_gsd": torch.full((2,), 10.0),
        "metadata": torch.zeros(2, 8),
    }
    objective.set_progress(1, 8)
    skipped, skipped_metrics = objective(batch, "physical")
    assert float(skipped.detach()) == 0.0
    assert not any(name.startswith("latent/") for name in skipped_metrics)
    objective.set_progress(4, 8)
    applied, applied_metrics = objective(batch, "physical")
    torch.testing.assert_close(applied, torch.tensor(0.8))
    assert "latent/info_nce" in applied_metrics
