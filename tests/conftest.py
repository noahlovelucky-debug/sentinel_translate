from __future__ import annotations

import pytest

from sentinel_v3.model import ModelConfig, SentinelV3


@pytest.fixture
def tiny_model() -> SentinelV3:
    return SentinelV3(
        ModelConfig(
            width=8,
            hidden=32,
            encoder_depth=1,
            heads=4,
            adapter_rank=8,
            dit_hidden=32,
            dit_depth=1,
            dit_heads=4,
            codec_width=8,
            codec_latent_channels=4,
            flow_steps=2,
        )
    )
