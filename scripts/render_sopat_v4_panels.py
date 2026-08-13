"""Render fixed-scale feasibility panels from SOPAT V4 comparison payloads.

This script is a post-forward visualization step.  It reads the normalized
payloads emitted by ``compare_sopat_v4_feasibility.py`` and never loads a
model or calls a model forward.  Optical panels use the canonical S2
B04/B03/B02 RGB mapping at fixed reflectance limits.  SAR panels show fixed
VV/VH dB rows, not an arbitrary image-normalized contrast stretch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

OPTICAL_RGB_CHANNELS = (2, 1, 0)  # Canonical Sentinel-2 B04/B03/B02.
OPTICAL_REFLECTANCE_RANGE = (0.0, 1.0)
OPTICAL_ERROR_RANGE = (0.0, 1.0)
SAR_DB_RANGES = ((-35.0, 5.0), (-45.0, -5.0))  # VV, VH.
SAR_ERROR_RANGE = (0.0, 40.0)
_TILE_GAP = 2
_LABEL_HEIGHT = 18
_HEADER_HEIGHT = 22
_MIN_TILE_EDGE = 128


def normalized_to_reflectance(values: Tensor) -> Tensor:
    """Convert canonical normalized S2 data to fixed [0, 1] reflectance."""

    return ((values.float() + 1.0) * 0.5).clamp(*OPTICAL_REFLECTANCE_RANGE)


def normalized_to_sar_db(values: Tensor) -> Tensor:
    """Convert canonical normalized S1 VV/VH values to fixed physical dB."""

    if values.ndim < 3 or values.shape[-3] != 2:
        raise ValueError("SAR panel values must have two channels ordered VV, VH")
    minimum = values.new_tensor((SAR_DB_RANGES[0][0], SAR_DB_RANGES[1][0])).reshape(
        *((1,) * (values.ndim - 3)), 2, 1, 1
    )
    maximum = values.new_tensor((SAR_DB_RANGES[0][1], SAR_DB_RANGES[1][1])).reshape(
        *((1,) * (values.ndim - 3)), 2, 1, 1
    )
    return (values.float() + 1.0) * 0.5 * (maximum - minimum) + minimum


def color_scale_metadata() -> dict[str, object]:
    """Return the exact non-adaptive display limits recorded in render manifests."""

    return {
        "normalization": "input tensors are canonical normalized [-1, 1]",
        "optical": {
            "target_rgb_channels": ("B04", "B03", "B02"),
            "reflectance_range": OPTICAL_REFLECTANCE_RANGE,
            "absolute_error_reflectance_range": OPTICAL_ERROR_RANGE,
        },
        "sar": {
            "channel_order": ("VV", "VH"),
            "vv_db_range": SAR_DB_RANGES[0],
            "vh_db_range": SAR_DB_RANGES[1],
            "absolute_error_db_range": SAR_ERROR_RANGE,
        },
    }


def _as_image(values: object, name: str) -> Tensor:
    if not isinstance(values, Tensor):
        raise TypeError(f"panel field {name} must be a tensor")
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 3:
        raise ValueError(f"panel field {name} must have shape CxHxW")
    if not torch.is_floating_point(values):
        raise TypeError(f"panel field {name} must be floating point")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"panel field {name} contains non-finite values")
    return values.detach().cpu()


def _as_valid(values: object, height: int, width: int) -> Tensor:
    if not isinstance(values, Tensor):
        raise TypeError("panel field valid must be a tensor")
    if values.ndim == 4 and values.shape[:2] == (1, 1):
        values = values[0]
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.shape != (height, width):
        raise ValueError("panel valid mask must have shape 1xHxW")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("panel valid mask contains non-finite values")
    return values.detach().cpu() > 0.0


def validate_panel_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate one comparison payload before any visual conversion."""

    if payload.get("family") != "sopat_v4_feasibility_panel":
        raise ValueError("not a SOPAT V4 feasibility panel payload")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported SOPAT V4 feasibility panel schema")
    direction = payload.get("direction")
    if direction not in {"sar_to_optical", "optical_to_sar"}:
        raise ValueError("panel direction must be sar_to_optical or optical_to_sar")
    required = ("source_anchor", "target_anchor", "v2_latest_checkpoint", "v4_ema", "target")
    images = {name: _as_image(payload.get(name), name) for name in required}
    target_shape = images["target"].shape
    if any(values.shape != target_shape for name, values in images.items() if name != "source_anchor"):
        raise ValueError("target-anchor, V2, V4, and target panel tensors must match")
    target_channels, height, width = target_shape
    source_channels = images["source_anchor"].shape[0]
    if images["source_anchor"].shape[-2:] != (height, width):
        raise ValueError("source anchor panel grid must match target panel grid")
    expected = (2, 10) if direction == "sar_to_optical" else (10, 2)
    if (source_channels, target_channels) != expected:
        raise ValueError(
            f"panel direction {direction} requires source/target channels {expected}, "
            f"got {(source_channels, target_channels)}"
        )
    valid = _as_valid(payload.get("valid"), height, width)
    source_valid = _as_valid(payload.get("source_valid"), height, width)
    result = dict(payload)
    result.update(images)
    result["valid"] = valid
    result["source_valid"] = source_valid
    return result


def _to_uint8(values: Tensor, minimum: float, maximum: float) -> Tensor:
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
        raise ValueError("color scale range must be finite and increasing")
    return ((values.float() - minimum) / (maximum - minimum)).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)


def _apply_valid(rgb: Tensor, valid: Tensor) -> Tensor:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB display tensor must have shape HxWx3")
    if valid.shape != rgb.shape[:2]:
        raise ValueError("panel valid mask does not match RGB tile")
    return torch.where(valid[..., None], rgb, torch.zeros_like(rgb))


def _image_from_rgb(rgb: Tensor, valid: Tensor) -> Image.Image:
    masked = _apply_valid(rgb, valid).contiguous().cpu().numpy()
    return Image.fromarray(masked, mode="RGB")


def optical_rgb(values: Tensor, valid: Tensor) -> Image.Image:
    """Render S2 B04/B03/B02 from one normalized canonical image."""

    if values.shape[0] != 10:
        raise ValueError("optical RGB panel requires canonical 10-channel Sentinel-2 data")
    reflectance = normalized_to_reflectance(values)
    rgb = _to_uint8(reflectance[list(OPTICAL_RGB_CHANNELS)].permute(1, 2, 0), *OPTICAL_REFLECTANCE_RANGE)
    return _image_from_rgb(rgb, valid)


def sar_composite(values: Tensor, valid: Tensor) -> Image.Image:
    """Render SAR's fixed VV/VH dB scales in a labeled two-channel composite."""

    db = normalized_to_sar_db(values)
    vv = _to_uint8(db[0], *SAR_DB_RANGES[0])
    vh = _to_uint8(db[1], *SAR_DB_RANGES[1])
    rgb = torch.stack((vv, vh, vh), dim=-1)
    return _image_from_rgb(rgb, valid)


def sar_band(values: Tensor, valid: Tensor, channel: int) -> Image.Image:
    if channel not in (0, 1):
        raise ValueError("SAR channel must be VV=0 or VH=1")
    db = normalized_to_sar_db(values)
    gray = _to_uint8(db[channel], *SAR_DB_RANGES[channel])
    return _image_from_rgb(torch.stack((gray, gray, gray), dim=-1), valid)


def error_heat(values: Tensor, valid: Tensor, maximum: float) -> Image.Image:
    """Render a fixed black-red-yellow absolute-error scale."""

    normalized = (values.float() / maximum).clamp(0.0, 1.0)
    red = (normalized * 2.0).clamp(0.0, 1.0)
    green = ((normalized - 0.5) * 2.0).clamp(0.0, 1.0)
    blue = torch.zeros_like(normalized)
    rgb = torch.stack((red, green, blue), dim=-1).mul(255.0).round().to(torch.uint8)
    return _image_from_rgb(rgb, valid)


def _resize_for_display(image: Image.Image) -> Image.Image:
    minimum = min(image.width, image.height)
    if minimum >= _MIN_TILE_EDGE:
        return image
    scale = max(1, math.ceil(_MIN_TILE_EDGE / max(1, minimum)))
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _labeled_tile(image: Image.Image, label: str) -> Image.Image:
    image = _resize_for_display(image)
    tile = Image.new("RGB", (image.width, image.height + _LABEL_HEIGHT), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((3, 3), label, fill="black", font=ImageFont.load_default())
    tile.paste(image, (0, _LABEL_HEIGHT))
    return tile


def _blank_tile(width: int, height: int, label: str) -> Image.Image:
    tile = Image.new("RGB", (width, height + _LABEL_HEIGHT), "white")
    ImageDraw.Draw(tile).text((3, 3), label, fill="black", font=ImageFont.load_default())
    return tile


def _join_row(tiles: Sequence[Image.Image]) -> Image.Image:
    if not tiles:
        raise ValueError("panel row needs at least one tile")
    height = max(tile.height for tile in tiles)
    width = sum(tile.width for tile in tiles) + _TILE_GAP * (len(tiles) - 1)
    row = Image.new("RGB", (width, height), "white")
    offset = 0
    for tile in tiles:
        row.paste(tile, (offset, 0))
        offset += tile.width + _TILE_GAP
    return row


def _panel_with_header(header: str, rows: Sequence[Image.Image]) -> Image.Image:
    if not rows:
        raise ValueError("panel needs at least one row")
    width = max(row.width for row in rows)
    height = _HEADER_HEIGHT + sum(row.height for row in rows) + _TILE_GAP * (len(rows) - 1)
    panel = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(panel).text((3, 4), header, fill="black", font=ImageFont.load_default())
    offset = _HEADER_HEIGHT
    for row in rows:
        panel.paste(row, (0, offset))
        offset += row.height + _TILE_GAP
    return panel


def _render_optical(payload: Mapping[str, object]) -> Image.Image:
    valid = payload["valid"]
    source_valid = payload["source_valid"]
    source_anchor = payload["source_anchor"]
    target_anchor = payload["target_anchor"]
    v2_latest_checkpoint = payload["v2_latest_checkpoint"]
    v4_ema = payload["v4_ema"]
    target = payload["target"]
    assert all(
        isinstance(value, Tensor)
        for value in (
            valid,
            source_valid,
            source_anchor,
            target_anchor,
            v2_latest_checkpoint,
            v4_ema,
            target,
        )
    )
    source_image = sar_composite(source_anchor, source_valid)
    # Normalized [-1, 1] differences span two reflectance units, while an
    # absolute error is displayed in physical reflectance units.
    v4_error = ((v4_ema - target).abs() * 0.5).clamp(*OPTICAL_ERROR_RANGE).mean(dim=0)
    tiles = (
        _labeled_tile(source_image, "Source anchor S1 (VV/VH dB)"),
        _labeled_tile(optical_rgb(target_anchor, valid), "Target anchor S2 RGB"),
        _labeled_tile(optical_rgb(v2_latest_checkpoint, valid), "V2 latest checkpoint S2 RGB"),
        _labeled_tile(optical_rgb(v4_ema, valid), "V4 EMA S2 RGB"),
        _labeled_tile(optical_rgb(target, valid), "Ground truth S2 RGB"),
        _labeled_tile(error_heat(v4_error, valid, OPTICAL_ERROR_RANGE[1]), "|V4 - GT| reflectance"),
    )
    header = (
        f"{payload.get('sample_id', 'sample')} | S1 -> S2 | B04/B03/B02 reflectance [0.00, 1.00] "
        "| V4 error [0.00, 1.00]"
    )
    return _panel_with_header(header, (_join_row(tiles),))


def _render_sar(payload: Mapping[str, object]) -> Image.Image:
    valid = payload["valid"]
    source_valid = payload["source_valid"]
    source_anchor = payload["source_anchor"]
    target_anchor = payload["target_anchor"]
    v2_latest_checkpoint = payload["v2_latest_checkpoint"]
    v4_ema = payload["v4_ema"]
    target = payload["target"]
    assert all(
        isinstance(value, Tensor)
        for value in (
            valid,
            source_valid,
            source_anchor,
            target_anchor,
            v2_latest_checkpoint,
            v4_ema,
            target,
        )
    )
    source_tile = _labeled_tile(optical_rgb(source_anchor, source_valid), "Source anchor S2 RGB")
    width = source_tile.width
    height = source_tile.height - _LABEL_HEIGHT
    v4_db = normalized_to_sar_db(v4_ema)
    target_db = normalized_to_sar_db(target)
    rows: list[Image.Image] = []
    for channel, name in enumerate(("VV", "VH")):
        error = (v4_db[channel] - target_db[channel]).abs()
        source = source_tile if channel == 0 else _blank_tile(width, height, "Source anchor shown above")
        tiles = (
            source,
            _labeled_tile(sar_band(target_anchor, valid, channel), f"Target anchor S1 {name}"),
            _labeled_tile(
                sar_band(v2_latest_checkpoint, valid, channel),
                f"V2 latest checkpoint S1 {name}",
            ),
            _labeled_tile(sar_band(v4_ema, valid, channel), f"V4 EMA S1 {name}"),
            _labeled_tile(sar_band(target, valid, channel), f"Ground truth S1 {name}"),
            _labeled_tile(error_heat(error, valid, SAR_ERROR_RANGE[1]), f"|V4 - GT| {name} dB"),
        )
        rows.append(_join_row(tiles))
    header = (
        f"{payload.get('sample_id', 'sample')} | S2 -> S1 | VV [-35, 5] dB, VH [-45, -5] dB "
        "| V4 error [0, 40] dB"
    )
    return _panel_with_header(header, rows)


def render_panel(payload: Mapping[str, object]) -> Image.Image:
    """Render one validated comparison payload with modality-fixed scales."""

    values = validate_panel_payload(payload)
    if values["direction"] == "sar_to_optical":
        return _render_optical(values)
    return _render_sar(values)


def _payload_paths(source: Path) -> list[Path]:
    if source.is_file():
        manifest = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise TypeError("panel manifest must be a mapping")
        entries = manifest.get("entries")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise TypeError("panel manifest entries must be a sequence")
        result: list[Path] = []
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("file"), str):
                raise TypeError("panel manifest entry is invalid")
            result.append(source.parent / entry["file"])
        return result
    if not source.is_dir():
        raise FileNotFoundError(f"panel input does not exist: {source}")
    manifest = source / "panel_manifest.json"
    return _payload_paths(manifest) if manifest.is_file() else sorted(source.glob("*.pt"))


def render_panels(
    source: str | Path,
    output: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """Render all selected panel payloads and write a reproducible manifest."""

    if limit is not None and limit <= 0:
        raise ValueError("panel render limit must be positive")
    input_path = Path(source)
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _payload_paths(input_path)
    if limit is not None:
        paths = paths[:limit]
    entries: list[dict[str, object]] = []
    for index, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError(f"panel payload must be a mapping: {path}")
        validated = validate_panel_payload(payload)
        image = render_panel(validated)
        filename = f"{index:03d}_{validated['direction']}_{_safe_filename(str(validated.get('sample_id', index)))}.png"
        destination = output_root / filename
        temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.png")
        image.save(temporary, format="PNG")
        os.replace(temporary, destination)
        entries.append(
            {
                "source_payload": str(path),
                "file": filename,
                "direction": str(validated["direction"]),
                "sample_id": str(validated.get("sample_id", "unknown")),
                "task_mode": str(validated.get("task_mode", "unknown")),
                "observation_count_bin": str(validated.get("observation_count_bin", "unknown")),
            }
        )
    manifest = {
        "family": "sopat_v4_feasibility_render",
        "schema_version": 1,
        "color_scales": color_scale_metadata(),
        "entries": entries,
    }
    destination = output_root / "render_manifest.json"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return manifest


def _safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return safe[:96] or "sample"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Panel payload directory or panel_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = render_panels(args.input, args.output, limit=args.limit)
    print(json.dumps({"output": str(args.output), "panels": len(result["entries"])}))


if __name__ == "__main__":
    main()
