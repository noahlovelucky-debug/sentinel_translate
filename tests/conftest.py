from __future__ import annotations

import pytest

from sentinel_v3.model import ModelConfig, SentinelV3


@pytest.fixture
def tiny_model() -> SentinelV3:
    return SentinelV3(
        ModelConfig(width=8, hidden=32, encoder_depth=1, dit_depth=1, heads=4, flow_steps=2)
    )
