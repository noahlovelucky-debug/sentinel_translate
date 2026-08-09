from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from sentinel_v3.evaluation import (
    _PERCEPTUAL_CACHE,
    ManifestCropDataset,
    load_checkpoint,
    manifest_metadata,
    perceptual_evaluators,
)
from sentinel_v3.sensors import SENTINEL1, SENTINEL2
from sentinel_v3.validation import (
    ValidationProtocol,
    protocol_records,
    validation_protocol_hash,
)


def test_missing_perceptual_weights_fail_before_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _PERCEPTUAL_CACHE.clear()
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    with pytest.raises(RuntimeError, match="mandatory"):
        perceptual_evaluators(torch.device("cpu"))


def test_validation_protocol_is_fixed_to_463_pairs() -> None:
    manifest = "/data/sentinel_translate/data/manifests/pairs.jsonl"
    records = protocol_records(manifest)
    assert len(records) == ValidationProtocol().expected_samples == 463
    assert validation_protocol_hash(manifest) == validation_protocol_hash(manifest)


def test_quick_validation_is_deterministically_stratified() -> None:
    manifest = "/data/sentinel_translate/data/manifests/pairs.jsonl"
    full = ManifestCropDataset(manifest, "validation_temporal")
    quick = ManifestCropDataset(manifest, "validation_temporal", limit=32)
    assert len(quick) == 32
    assert quick.records[0]["pair_id"] == full.records[0]["pair_id"]
    assert quick.records[-1]["pair_id"] == full.records[-1]["pair_id"]
    assert [record["pair_id"] for record in quick.records] != [
        record["pair_id"] for record in full.records[:32]
    ]


def test_manifest_metadata_matches_training_encoding() -> None:
    item = {
        "s1_date": "2019-04-04",
        "s2_date": "2019-04-03",
        "delta_days": 1,
        "orbit": "ascending",
        "gsd": 10.0,
    }
    metadata = manifest_metadata(item, torch.device("cpu"))
    assert metadata.shape == (1, 8)
    assert metadata[0, 0] == pytest.approx(1 / 3)
    assert metadata[0, 1] == -1
    assert metadata[0, 6] == pytest.approx(torch.log(torch.tensor(10.0)).item() / 4)
    assert metadata[0, 7] == 1


def test_zero_radiometric_extensions_preserve_v4_loading(
    tmp_path: Path, tiny_model: torch.nn.Module
) -> None:
    prefixes = (
        "decoder.radiometric_kernel.",
        "decoder.radiometric_condition.",
        "decoder.radiometric_descriptor.",
        "decoder.radiometric_bias.",
        "decoder.full_resolution_fusion.",
        "decoder.optical_direction_kernel.",
        "decoder.optical_amplitude_head.",
        "decoder.sar_spatial_kernel.",
        "decoder.sar_mean_condition.",
        "decoder.sar_mean_descriptor.",
        "decoder.sar_mean_head.",
    )
    legacy_state = {
        name: value for name, value in tiny_model.state_dict().items() if not name.startswith(prefixes)
    }
    checkpoint = tmp_path / "legacy_v4.pt"
    torch.save(
        {
            "format_version": 4,
            "config": {"model": asdict(tiny_model.config)},
            "model": legacy_state,
        },
        checkpoint,
    )
    values = torch.randn(1, 2, 32, 32)
    valid = torch.ones(1, 1, 32, 32)
    tiny_model.eval()
    with torch.inference_mode():
        expected = tiny_model.physical(values, SENTINEL1, SENTINEL2, valid)[0]
        actual = load_checkpoint(checkpoint, torch.device("cpu")).physical(
            values, SENTINEL1, SENTINEL2, valid
        )[0]
    torch.testing.assert_close(actual, expected)


def test_id_bridge_extensions_are_optional_for_legacy_v4_checkpoints(
    tmp_path: Path, tiny_model: torch.nn.Module
) -> None:
    prefixes = (
        "id_bridge_origin.",
        "residual_dit.origin_projection.",
        "residual_dit.id_bridge_field_projection.",
        "residual_dit.id_bridge_anchor_projection.",
        "phase_transport_head.",
    )
    legacy_state = {
        name: value for name, value in tiny_model.state_dict().items() if not name.startswith(prefixes)
    }
    legacy_config = asdict(tiny_model.config)
    for name in (
        "id_bridge_state",
        "id_bridge_state_channels",
        "id_bridge_optical_state_scale",
        "id_bridge_sar_state_scale",
        "id_bridge_anchor_origin",
        "id_bridge_optical_innovation_scale",
        "id_bridge_sar_innovation_scale",
        "phase_transport_enabled",
        "phase_transport_hidden",
        "phase_transport_gain_caps",
        "phase_transport_offset_caps_px",
        "phase_transport_initial_gate",
        "phase_transport_null_calibrated",
        "phase_transport_null_quantile",
        "phase_transport_support_epsilon",
    ):
        legacy_config.pop(name)
    checkpoint = tmp_path / "legacy_v4_without_id_bridge.pt"
    torch.save(
        {
            "format_version": 4,
            "config": {"model": legacy_config},
            "model": legacy_state,
        },
        checkpoint,
    )

    loaded = load_checkpoint(checkpoint, torch.device("cpu"))
    assert isinstance(loaded, type(tiny_model))


def test_checkpoint_loader_rejects_unknown_missing_v4_parameter(
    tmp_path: Path, tiny_model: torch.nn.Module
) -> None:
    state = dict(tiny_model.state_dict())
    unknown_key = next(name for name in state if name.startswith("encoder."))
    state.pop(unknown_key)
    checkpoint = tmp_path / "legacy_v4_unknown_missing.pt"
    torch.save(
        {
            "format_version": 4,
            "config": {"model": asdict(tiny_model.config)},
            "model": state,
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="incompatible checkpoint"):
        load_checkpoint(checkpoint, torch.device("cpu"))
