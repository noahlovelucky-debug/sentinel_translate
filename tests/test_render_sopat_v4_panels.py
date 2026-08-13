from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.fixture(scope="module")
def renderer_module():
    path = Path(__file__).parents[1] / "scripts" / "render_sopat_v4_panels.py"
    spec = importlib.util.spec_from_file_location("render_sopat_v4_panels_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _payload(direction: str, *, height: int = 4, width: int = 5) -> dict[str, object]:
    source_channels, target_channels = (2, 10) if direction == "sar_to_optical" else (10, 2)
    target = torch.linspace(-1.0, 1.0, target_channels * height * width).reshape(
        target_channels, height, width
    )
    valid = torch.ones(1, height, width)
    valid[..., 0, 0] = 0.0
    source_valid = torch.ones(1, height, width)
    source_valid[..., 0, -1] = 0.0
    return {
        "family": "sopat_v4_feasibility_panel",
        "schema_version": 1,
        "direction": direction,
        "sample_id": f"{direction}-fixed-scale",
        "task_mode": "translation",
        "observation_count_bin": "one",
        "source_anchor": torch.linspace(-1.0, 1.0, source_channels * height * width).reshape(
            source_channels, height, width
        ),
        "source_valid": source_valid,
        "target_anchor": target * 0.5,
        "v2_latest_checkpoint": target * 0.75,
        "v4_ema": target * 0.9,
        "target": target,
        "valid": valid,
    }


def test_fixed_physical_color_scales_are_not_image_adaptive(renderer_module: object) -> None:
    reflectance = renderer_module.normalized_to_reflectance(torch.tensor([-1.0, 0.0, 1.0]))
    assert reflectance.tolist() == pytest.approx([0.0, 0.5, 1.0])
    sar = renderer_module.normalized_to_sar_db(torch.tensor([[[-1.0, 1.0]], [[-1.0, 1.0]]]))
    assert sar[0, 0].tolist() == pytest.approx([-35.0, 5.0])
    assert sar[1, 0].tolist() == pytest.approx([-45.0, -5.0])
    metadata = renderer_module.color_scale_metadata()
    optical = metadata["optical"]
    sar_metadata = metadata["sar"]
    assert isinstance(optical, dict) and optical["reflectance_range"] == renderer_module.OPTICAL_REFLECTANCE_RANGE
    assert isinstance(sar_metadata, dict) and sar_metadata["vv_db_range"] == renderer_module.SAR_DB_RANGES[0]


def test_optical_mask_is_rendered_as_exact_black(renderer_module: object) -> None:
    values = torch.ones(10, 2, 2)
    valid = torch.tensor([[True, False], [True, True]])
    image = renderer_module.optical_rgb(values, valid)
    pixels = torch.from_numpy(__import__("numpy").asarray(image).copy())
    assert torch.equal(pixels[0, 1], torch.zeros(3, dtype=torch.uint8))
    assert torch.equal(pixels[0, 0], torch.full((3,), 255, dtype=torch.uint8))


@pytest.mark.parametrize("direction", ("sar_to_optical", "optical_to_sar"))
def test_renderer_writes_honest_png_and_manifest(
    tmp_path: Path, direction: str, renderer_module: object
) -> None:
    payload = _payload(direction)
    payload_root = tmp_path / "payloads"
    payload_root.mkdir()
    torch.save(payload, payload_root / f"{direction}.pt")

    manifest = renderer_module.render_panels(payload_root, tmp_path / "rendered")

    assert len(manifest["entries"]) == 1
    entry = manifest["entries"][0]
    assert isinstance(entry, dict)
    png = tmp_path / "rendered" / str(entry["file"])
    assert png.is_file()
    with Image.open(png) as image:
        assert image.format == "PNG"
        assert image.width > 0 and image.height > 0
        assert image.getbbox() is not None
    render_metadata = manifest["color_scales"]
    assert isinstance(render_metadata, dict)
    assert render_metadata == renderer_module.color_scale_metadata()


def test_renderer_rejects_missing_source_mask_and_wrong_target_channels(renderer_module: object) -> None:
    missing_mask = _payload("sar_to_optical")
    del missing_mask["source_valid"]
    with pytest.raises(TypeError, match="valid"):
        renderer_module.validate_panel_payload(missing_mask)

    wrong_channels = _payload("optical_to_sar")
    wrong_channels["target"] = torch.zeros(10, 4, 5)
    with pytest.raises(ValueError, match="match|channels"):
        renderer_module.validate_panel_payload(wrong_channels)
