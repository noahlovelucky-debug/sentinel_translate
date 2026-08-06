from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from sentinel_v3.evaluation import ManifestCropDataset, _image, load_checkpoint
from sentinel_v3.losses import masked_mean
from sentinel_v3.sensors import SENTINEL1, SENTINEL2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--split", default="validation_temporal")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device, use_ema=False)
    dataset = ManifestCropDataset(args.manifest, args.split, limit=args.samples)
    tile_size = 192
    header = 30
    columns = 8
    canvas = Image.new("RGB", (columns * tile_size, len(dataset) * (tile_size + header)), "white")
    draw = ImageDraw.Draw(canvas)

    for row, item in enumerate(dataset):
        s2 = item["s2"].unsqueeze(0).to(device)
        sar = item["sar"].unsqueeze(0).to(device)
        valid = item["valid"].unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            s2_mean, _, _ = model.physical(sar, SENTINEL1, SENTINEL2, valid)
            sar_mean, _, _ = model.physical(s2, SENTINEL2, SENTINEL1, valid)
        optical_rmse = float(torch.sqrt(masked_mean((s2_mean - s2).square(), valid)))
        sar_rmse = float(torch.sqrt(masked_mean((sar_mean - sar).square(), valid)))
        panels = [
            ("Input SAR (VV)", _image(sar[0], "sar", tile_size)),
            (f"Pred RGB | RMSE {optical_rmse:.3f}", _image(s2_mean[0], "rgb", tile_size)),
            ("Reference RGB", _image(s2[0], "rgb", tile_size)),
            ("Optical abs error", _image(s2_mean[0] - s2[0], "map", tile_size)),
            ("Input RGB", _image(s2[0], "rgb", tile_size)),
            (f"Pred SAR | RMSE {sar_rmse:.2f} dB", _image(sar_mean[0], "sar", tile_size)),
            ("Reference SAR (VV)", _image(sar[0], "sar", tile_size)),
            ("SAR abs error", _image(sar_mean[0] - sar[0], "map", tile_size)),
        ]
        y = row * (tile_size + header)
        for column, (title, panel) in enumerate(panels):
            x = column * tile_size
            draw.text((x + 4, y + 8), title, fill="black")
            canvas.paste(panel, (x, y + header))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
