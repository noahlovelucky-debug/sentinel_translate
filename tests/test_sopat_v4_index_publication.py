from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from sentinel_v3.dataset_builder import PairRecord
from sentinel_v3.paired_temporal_data import (
    load_paired_temporal_index,
    write_pair_records,
)
from sentinel_v3.schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER
from sentinel_v4.data import load_sopat_v4_index, paired_temporal_index_from_sopat_v4


@pytest.fixture(scope="module")
def index_builder_module():
    path = Path(__file__).parents[1] / "scripts" / "build_sopat_v4_index.py"
    spec = importlib.util.spec_from_file_location("build_sopat_v4_index_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiff(path: Path, values: np.ndarray) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs="EPSG:32650",
        transform=from_origin(500000.0, 4100000.0, 10.0, 10.0),
    ) as source:
        source.write(values, 1)


def _record(root: Path, *, split: str, number: int, invalid_optical: bool = False) -> PairRecord:
    size = 256
    acquired = date(2020, 1, 1) + timedelta(days=number * 3)
    record_root = root / "raw" / split / f"record-{number:03d}"
    optical = np.full((size, size), 3000 + number * 100, dtype=np.uint16)
    sar = np.full((size, size), 6000 + number * 100, dtype=np.uint16)
    scl = np.full((size, size), 1 if invalid_optical else 4, dtype=np.uint8)
    s2: dict[str, str] = {}
    for channel in S2_CHANNEL_ORDER:
        path = record_root / "s2" / f"{channel}.tif"
        _write_tiff(path, optical)
        s2[channel] = str(path.relative_to(root))
    scl_path = record_root / "scl.tif"
    _write_tiff(scl_path, scl)
    sar_paths: dict[str, str] = {}
    for channel in SAR_CHANNEL_ORDER:
        path = record_root / "sar" / f"{channel}.tif"
        _write_tiff(path, sar)
        sar_paths[channel] = str(path.relative_to(root))
    return PairRecord(
        pair_id=f"2020:tile-v4:{split}:{number:03d}:ascending",
        year=2020,
        tile="tile-v4",
        tile_row=1,
        tile_col=1,
        split=split,
        refit_split="excluded",
        s2_date=acquired.isoformat(),
        s1_date=acquired.isoformat(),
        orbit="ascending",
        delta_days=0,
        s2=s2,
        scl=str(scl_path.relative_to(root)),
        sar=sar_paths,
        clear_fraction=0.0 if invalid_optical else 1.0,
        valid_fraction=1.0,
        width=size,
        height=size,
        crs="EPSG:32650",
        transform=[10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0],
        gsd=10.0,
    )


def _config(path: Path, manifest: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "manifest": str(manifest),
                    "orbit": "ascending",
                    "anchor_pair_max_delta_days": 1,
                    "maximum_anchors_per_query": 2,
                    "horizon_days": 180,
                    "translation_max_delta_days": 1,
                    "minimum_observations": 1,
                    "maximum_observations": 4,
                    "crop_size": 256,
                    "train_split": "train",
                    "validation_split": "validation_temporal",
                    "task_modes": ["translation", "forecast"],
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _inputs(root: Path) -> tuple[Path, Path]:
    records = [
        *(_record(root, split="train", number=number) for number in range(6)),
        *(
            _record(
                root,
                split="validation_temporal",
                number=number + 20,
                invalid_optical=number == 2,
            )
            for number in range(6)
        ),
    ]
    manifest = root / "pairs.jsonl"
    write_pair_records(manifest, records)
    return _config(root / "index-source.yaml", manifest), manifest


def test_publication_rebinds_center_filtered_v4_roles_to_exact_v3_indexes(
    tmp_path: Path, index_builder_module
) -> None:
    config, _ = _inputs(tmp_path)
    output = tmp_path / "published" / "index.jsonl"
    paired_root = output.parent / "paired_indexes"
    publication = output.parent / "index_publication.json"

    result = index_builder_module.build_and_publish(
        config,
        output,
        paired_index_root=paired_root,
        publication=publication,
    )

    assert result["reused"] is False
    marker = json.loads(publication.read_text(encoding="utf-8"))
    assert marker["v4_index_file_sha256"]
    assert marker["validation_center_filter"]["train_pixel_filtered"] is False
    assert marker["validation_center_filter"]["directions"]["sar_to_optical"][
        "output_samples"
    ] < marker["validation_center_filter"]["directions"]["sar_to_optical"]["input_samples"]

    role_index = load_sopat_v4_index(output)
    for direction in ("sar_to_optical", "optical_to_sar"):
        for split in ("train", "validation_temporal"):
            published = load_paired_temporal_index(paired_root / direction / f"{split}.jsonl")
            projected = paired_temporal_index_from_sopat_v4(
                role_index,
                direction=direction,
                split=split,
            )
            assert {sample.sample_id: sample for sample in projected.samples} == {
                sample.sample_id: sample for sample in published.samples
            }
            assert len(projected) == len(published)


def test_valid_publication_reuses_without_reopening_rasters(
    tmp_path: Path, index_builder_module, monkeypatch
) -> None:
    config, _ = _inputs(tmp_path)
    output = tmp_path / "published" / "index.jsonl"
    paired_root = output.parent / "paired_indexes"
    publication = output.parent / "index_publication.json"
    index_builder_module.build_and_publish(
        config,
        output,
        paired_index_root=paired_root,
        publication=publication,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a valid index publication must not rebuild or read raster pixels")

    monkeypatch.setattr(index_builder_module, "_configured_indexes", forbidden)
    reused = index_builder_module.build_and_publish(
        config,
        output,
        paired_index_root=paired_root,
        publication=publication,
    )

    assert reused["reused"] is True
