from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sentinel_v3.config import load_config
from sentinel_v3.training import train


def test_checkpoint_is_complete_and_resumable(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["paths"]["output"] = str(tmp_path)
    config["train"]["max_steps"] = 1
    config["train"]["save_every"] = 1
    config["train"]["batch_size"] = 2
    train(config, limit=4)
    checkpoint_path = tmp_path / "latest.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload.keys() >= {
        "model",
        "ema",
        "optimizer_states",
        "scheduler_states",
        "rank_states",
        "config",
        "step",
        "codec_version",
        "validation_protocol_hash",
        "best_metrics",
    }
    assert payload["format_version"] == 4
    assert payload["step"] == 1
    config["train"]["max_steps"] = 2
    train(config, resume=str(checkpoint_path), limit=4)
    resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert resumed["step"] == 2


def test_old_checkpoint_cannot_resume(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["paths"]["output"] = str(tmp_path / "output")
    config["train"]["max_steps"] = 1
    legacy = tmp_path / "legacy.pt"
    torch.save({"format_version": 3}, legacy)
    with pytest.raises(RuntimeError, match="--init-model"):
        train(config, resume=str(legacy), limit=4)
