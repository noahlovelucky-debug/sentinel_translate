from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch

from sentinel_v3.evaluation import aggregate_scene_bias
from sentinel_v3.model import ModelConfig, SentinelV3
from sentinel_v3.schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER
from sentinel_v3.sensors import SENTINEL2
from sentinel_v3.temporal_prior import TemporalPriorStore, temporal_prior_config


def _record(split: str, year: int, tile: str, pair_id: str, day: str) -> dict[str, object]:
    return {
        "split": split,
        "year": year,
        "tile": tile,
        "pair_id": pair_id,
        "s1_date": day,
        "s2_date": day,
        "orbit": "descending",
    }


def test_temporal_index_is_strictly_train_2017_2018(tmp_path: Path) -> None:
    manifest = tmp_path / "pairs.jsonl"
    records = [
        _record("train", 2017, "known", "eligible", "2017-01-02"),
        _record("validation_temporal", 2019, "leak", "validation", "2019-01-02"),
        _record("train", 2019, "late", "wrong-year", "2019-01-02"),
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    store = TemporalPriorStore(temporal_prior_config(manifest))
    assert store.locations == frozenset({"known"})
    assert store._nearest("unknown", date(2020, 1, 1), "optical", "unknown") == []


def test_temporal_composition_separates_spectrum_and_amplitude(tmp_path: Path) -> None:
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        json.dumps(_record("train", 2018, "known", "eligible", "2018-01-02")) + "\n"
    )
    store = TemporalPriorStore(temporal_prior_config(manifest))
    physical = torch.tensor([[[[0.2]], [[0.4]]]])
    prior = torch.tensor([[[[0.6]], [[0.8]]]])
    output, violation = store.compose(
        physical, prior, torch.ones(1, 1, 1, 1), "optical"
    )
    expected_direction = prior / torch.linalg.vector_norm(prior, dim=1, keepdim=True)
    torch.testing.assert_close(
        output / torch.linalg.vector_norm(output, dim=1, keepdim=True), expected_direction
    )
    assert float(violation) == 0.0
    fallback, _ = store.compose(
        physical, prior, torch.zeros(1, 1, 1, 1), "optical"
    )
    torch.testing.assert_close(fallback, physical)


def test_hard_bias_is_absolute_mean_signed_scene_bias() -> None:
    hard_bias, scene_abs_bias = aggregate_scene_bias([0.8, -0.6])
    assert hard_bias == pytest.approx(0.1)
    assert scene_abs_bias == pytest.approx(0.7)


def test_temporal_prior_receives_the_training_spatial_transform() -> None:
    class FakeStore:
        def query(self, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
            prior = torch.arange(40, dtype=torch.float32).reshape(1, 10, 2, 2)
            return prior, torch.ones(1, 1, 2, 2)

        def compose(
            self,
            physical: torch.Tensor,
            prior: torch.Tensor,
            coverage: torch.Tensor,
            modality: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del physical, coverage, modality
            return prior, prior.new_zeros(())

    model = SentinelV3(
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
        )
    )
    model.temporal_prior = FakeStore()  # type: ignore[assignment]
    physical = torch.zeros(1, 10, 2, 2)
    output = model.apply_temporal_prior(
        physical,
        SENTINEL2,
        acquired="2020-01-01",
        location_id="known",
        pixel_window=(0, 0, 2, 2),
        spatial_transform=(True, False, 1),
    )[0]
    original = torch.arange(40, dtype=torch.float32).reshape(1, 10, 2, 2)
    torch.testing.assert_close(output, torch.rot90(torch.flip(original, (-1,)), 1, (-2, -1)))


def _write_tiff(path: Path, values: np.ndarray) -> None:
    import rasterio
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
        transform=from_origin(500000, 4100000, 10, 10),
    ) as source:
        source.write(values, 1)


def _raw_record(root: Path, day: str, pair_id: str, value: int) -> dict[str, object]:
    s2 = {}
    for index, channel in enumerate(S2_CHANNEL_ORDER):
        path = root / day / f"{channel}.tiff"
        _write_tiff(path, np.full((8, 8), value + index, dtype=np.uint16))
        s2[channel] = str(path)
    scl = root / day / "scl.tiff"
    _write_tiff(scl, np.full((8, 8), 2, dtype=np.uint8))
    sar = {}
    for index, channel in enumerate(SAR_CHANNEL_ORDER):
        path = root / day / f"{channel}.tiff"
        _write_tiff(path, np.full((8, 8), 6000 + index, dtype=np.uint16))
        sar[channel] = str(path)
    return {
        **_record("train", 2018, "tile", pair_id, day),
        "s2": s2,
        "scl": str(scl),
        "sar": sar,
        "width": 8,
        "height": 8,
    }


def test_windows_prior_matches_full_scene_crops_and_reuses_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_id = "2018:tile:2018-01-02:descending:2018-01-02"
    neighbor_id = "2018:tile:2018-01-04:descending:2018-01-04"
    records = [
        _raw_record(tmp_path / "raw", "2018-01-02", target_id, 1000),
        _raw_record(tmp_path / "raw", "2018-01-04", neighbor_id, 2000),
    ]
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    store = TemporalPriorStore(temporal_prior_config(manifest))
    windows = [(0, 0, 4, 4), (4, 0, 4, 4)]
    import rasterio

    original_open = rasterio.open
    opens = 0

    def counted_open(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal opens
        opens += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(rasterio, "open", counted_open)
    optical, optical_coverage = store.windows_prior(
        location_id="tile",
        acquired="2018-01-02",
        modality="optical",
        orbit="descending",
        windows=windows,
        exclude_pair_id=target_id,
    )
    sar, sar_coverage = store.windows_prior(
        location_id="tile",
        acquired="2018-01-02",
        modality="sar",
        orbit="descending",
        windows=windows,
        exclude_pair_id=target_id,
    )
    assert opens == len(S2_CHANNEL_ORDER) + 1 + len(SAR_CHANNEL_ORDER)
    monkeypatch.setattr(rasterio, "open", original_open)

    full_optical, full_optical_coverage = store.full_scene_prior(
        location_id="tile",
        acquired="2018-01-02",
        modality="optical",
        orbit="descending",
        exclude_pair_id=target_id,
    )
    full_sar, full_sar_coverage = store.full_scene_prior(
        location_id="tile",
        acquired="2018-01-02",
        modality="sar",
        orbit="descending",
        exclude_pair_id=target_id,
    )
    expected_optical = np.stack(
        [full_optical[:, row : row + height, col : col + width] for col, row, width, height in windows]
    )
    expected_optical_coverage = np.stack(
        [
            full_optical_coverage[:, row : row + height, col : col + width]
            for col, row, width, height in windows
        ]
    )
    expected_sar = np.stack(
        [full_sar[:, row : row + height, col : col + width] for col, row, width, height in windows]
    )
    expected_sar_coverage = np.stack(
        [
            full_sar_coverage[:, row : row + height, col : col + width]
            for col, row, width, height in windows
        ]
    )
    np.testing.assert_allclose(optical, expected_optical)
    np.testing.assert_array_equal(optical_coverage, expected_optical_coverage)
    np.testing.assert_allclose(sar, expected_sar)
    np.testing.assert_array_equal(sar_coverage, expected_sar_coverage)
