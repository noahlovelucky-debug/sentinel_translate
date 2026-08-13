from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from sentinel_v3.paired_temporal_training import (
    PairedTemporalTrainConfig,
    save_paired_temporal_checkpoint,
)
from sentinel_v3.paired_temporal_v2 import PairedTemporalConfig, SparsePairedAnchorTransport
from sentinel_v3.sensors import SENTINEL1, SENTINEL2


@pytest.fixture(scope="module")
def comparison_module():
    root = Path(__file__).parents[1]
    scripts = root / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "compare_sopat_v4_feasibility.py"
        spec = importlib.util.spec_from_file_location("compare_sopat_v4_feasibility_under_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


class _RecordingV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def physical(
        self,
        observations: Tensor,
        observation_valid: Tensor,
        observation_days: Tensor,
        observation_present: Tensor,
        source_anchor: Tensor,
        source_anchor_valid: Tensor,
        target_anchor: Tensor,
        target_anchor_valid: Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        assert "target" not in kwargs
        assert "target_valid" not in kwargs
        self.calls.append(
            {
                "observations": observations.detach().clone(),
                "observation_valid": observation_valid.detach().clone(),
                "observation_days": observation_days.detach().clone(),
                "observation_present": observation_present.detach().clone(),
                "source_anchor": source_anchor.detach().clone(),
                "source_anchor_valid": source_anchor_valid.detach().clone(),
                "target_anchor": target_anchor.detach().clone(),
                "target_anchor_valid": target_anchor_valid.detach().clone(),
                "kwargs": kwargs,
            }
        )
        return SimpleNamespace(physical=target_anchor, pre_projection_violation=torch.zeros(1))


class _RejectingForwardModel(nn.Module):
    """Fails immediately if labels cross the V4 causal forwarding boundary."""

    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(delta))
        self.calls: list[set[str]] = []

    def forward(self, **inputs: object) -> SimpleNamespace:
        assert "target" not in inputs
        assert "target_valid" not in inputs
        expected = {
            "observations",
            "observation_valid",
            "observation_days",
            "observation_present",
            "source_anchor",
            "source_anchor_valid",
            "target_anchor",
            "target_anchor_valid",
            "source_anchor_days",
            "target_anchor_days",
            "source_sensor",
            "target_sensor",
        }
        assert set(inputs) == expected
        self.calls.append(set(inputs))
        target_anchor = inputs["target_anchor"]
        assert isinstance(target_anchor, Tensor)
        return SimpleNamespace(
            physical=target_anchor + self.delta * torch.ones_like(target_anchor),
            pre_projection_violation=torch.zeros(target_anchor.shape[0], device=target_anchor.device),
        )


def _adapter_inputs() -> dict[str, Tensor]:
    batch, frames, channels, height, width = 2, 3, 2, 8, 8
    observations = torch.empty(batch, frames, channels, height, width)
    observations[0, 0].fill_(1.0)
    observations[0, 1].fill_(2.0)
    observations[0, 2].fill_(77.0)  # Deliberate padded garbage.
    observations[1, 0].fill_(3.0)
    observations[1, 1].fill_(88.0)
    observations[1, 2].fill_(99.0)
    return {
        "observations": observations,
        "observation_valid": torch.ones(batch, frames, 1, height, width),
        "observation_days": torch.tensor([[-8.0, -1.0, 1234.0], [-2.0, 456.0, 789.0]]),
        "observation_present": torch.tensor([[True, True, False], [True, False, False]]),
        "source_anchor": torch.zeros(batch, channels, height, width),
        "source_anchor_valid": torch.ones(batch, 1, height, width),
        "target_anchor": torch.zeros(batch, 10, height, width),
        "target_anchor_valid": torch.ones(batch, 1, height, width),
        "source_anchor_days": torch.full((batch,), -10.0),
        "target_anchor_days": torch.full((batch,), -9.0),
    }


def _panel_batch(direction: str) -> dict[str, object]:
    source_channels, target_channels = (2, 10) if direction == "sar_to_optical" else (10, 2)
    height = width = 8
    source_anchor = torch.full((1, source_channels, height, width), -0.2)
    target_anchor = torch.full((1, target_channels, height, width), 0.1)
    return {
        "observations": source_anchor[:, None].clone(),
        "observation_valid": torch.ones(1, 1, 1, height, width),
        "observation_days": torch.tensor([[-1.0]]),
        "observation_present": torch.tensor([[True]]),
        "source_anchor": source_anchor,
        "source_anchor_valid": torch.ones(1, 1, height, width),
        "target_anchor": target_anchor,
        "target_anchor_valid": torch.ones(1, 1, height, width),
        "source_anchor_days": torch.tensor([-4.0]),
        "target_anchor_days": torch.tensor([-3.0]),
        "target": target_anchor + 0.2,
        "target_valid": torch.ones(1, 1, height, width),
        "sopat_example_id": [f"{direction}-example"],
        "task_mode": ["translation"],
    }


def test_v2_checkpoint_adapter_preserves_full_v4_observation_set(
    comparison_module: object,
) -> None:
    adapter_type = comparison_module.V2CheckpointAdapter
    v2 = _RecordingV2()
    adapter = adapter_type(v2)
    inputs = _adapter_inputs()

    output = adapter(
        **inputs,
        source_sensor=SENTINEL1,
        target_sensor=SENTINEL2,
    )

    assert torch.equal(output.physical, inputs["target_anchor"])
    assert len(v2.calls) == 1
    call = v2.calls[0]
    for name in ("observations", "observation_valid", "observation_days", "observation_present"):
        forwarded = call[name]
        assert isinstance(forwarded, Tensor)
        assert torch.equal(forwarded, inputs[name])
    forwarded_observations = call["observations"]
    assert isinstance(forwarded_observations, Tensor)
    # The padded image values are preserved instead of being selected, zeroed,
    # or otherwise rewritten by the comparison adapter.
    assert float(forwarded_observations[0, 2, 0, 0, 0]) == 77.0
    assert float(forwarded_observations[1, 2, 0, 0, 0]) == 99.0


def test_v2_checkpoint_adapter_loads_direction_bound_model(
    tmp_path: Path, comparison_module: object
) -> None:
    config = PairedTemporalConfig(width=8, latent_channels=4, attention_heads=2, flow_steps=1)
    model = SparsePairedAnchorTransport(config)
    checkpoint = save_paired_temporal_checkpoint(
        tmp_path / "v2-sar-to-optical.pt",
        model=model,
        config=PairedTemporalTrainConfig(direction="sar_to_optical"),
        step=17,
    )

    adapter, metadata = comparison_module.load_v2_checkpoint_adapter(
        checkpoint,
        direction="sar_to_optical",
        device=torch.device("cpu"),
    )

    assert isinstance(adapter.model, SparsePairedAnchorTransport)
    assert metadata["checkpoint_step"] == 17
    assert metadata["checkpoint_direction"] == "sar_to_optical"
    input_adapter = metadata["input_adapter"]
    assert isinstance(input_adapter, dict)
    assert input_adapter["checkpoint_label"] == "latest"
    assert input_adapter["uses_full_v4_observation_set"] is True
    with pytest.raises(RuntimeError, match="direction"):
        comparison_module.load_v2_checkpoint_adapter(
            checkpoint,
            direction="optical_to_sar",
            device=torch.device("cpu"),
        )


def test_panel_export_keeps_target_labels_out_of_both_model_forwards(
    tmp_path: Path, comparison_module: object
) -> None:
    v4 = _RejectingForwardModel(0.05)
    v2 = _RejectingForwardModel(-0.05)
    loaders = {
        "sar_to_optical": [_panel_batch("sar_to_optical")],
        "optical_to_sar": [_panel_batch("optical_to_sar")],
    }

    manifest = comparison_module.export_feasibility_panel_payloads(
        v4,
        v2,
        loaders,
        tmp_path / "payloads",
        device=torch.device("cpu"),
        limit_per_direction=1,
    )

    assert manifest["target_label_forwarded"] is False
    assert len(manifest["entries"]) == 2
    assert len(v4.calls) == len(v2.calls) == 2
    for entry in manifest["entries"]:
        assert isinstance(entry, dict)
        payload = torch.load(tmp_path / "payloads" / str(entry["file"]), weights_only=False)
        assert payload["target_label_forwarded"] is False
        assert payload["source_valid"].shape == (1, 8, 8)
        assert payload["target"].shape == payload["v4_ema"].shape
        assert payload["target"].shape == payload["v2_latest_checkpoint"].shape


def test_relative_improvements_preserve_metric_orientation_for_nested_buckets(
    comparison_module: object,
) -> None:
    baseline = {
        "directions": {
            "sar_to_optical": {
                "all": {
                    "all": {
                        "samples": 2,
                        "rmse": 0.4,
                        "edge_f1": 0.5,
                        "psd_log_l1": 0.3,
                        "scene_improved_fraction": 0.25,
                    }
                },
                "by_task": {},
                "by_observation_count": {},
                "regimes": {},
            }
        }
    }
    candidate = {
        "directions": {
            "sar_to_optical": {
                "all": {
                    "all": {
                        "samples": 2,
                        "rmse": 0.2,
                        "edge_f1": 0.75,
                        "psd_log_l1": 0.15,
                        "scene_improved_fraction": 0.5,
                    }
                },
                "by_task": {},
                "by_observation_count": {},
                "regimes": {},
            }
        }
    }

    comparison = comparison_module.relative_improvements(candidate, baseline)
    directions = comparison["directions"]
    assert isinstance(directions, dict)
    metrics = directions["sar_to_optical"]["all"]["all"]
    assert metrics["rmse"]["relative_improvement"] == pytest.approx(0.5)
    assert metrics["edge_f1"]["relative_improvement"] == pytest.approx(0.5)
    assert metrics["psd_log_l1"]["metric_orientation"] == "lower_is_better"
    assert metrics["scene_improved_fraction"]["metric_orientation"] == "higher_is_better"
