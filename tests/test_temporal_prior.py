from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import torch

from sentinel_v3.evaluation import aggregate_scene_bias
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
