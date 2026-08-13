from __future__ import annotations

import math

import torch

from sentinel_v4.evaluation import (
    SOPATSelectionConfig,
    SOPATVariantConfig,
    _masked_box_lowpass,
    _source_shuffle_batch,
    evaluate_sopat_loaders,
    select_sopat_candidate,
)


def _batch(direction: str, *, batch_size: int = 2) -> dict[str, object]:
    source_channels, target_channels = (2, 10) if direction == "sar_to_optical" else (10, 2)
    frames, height, width = 2, 12, 12
    source_anchor = torch.full((batch_size, source_channels, height, width), -0.2)
    target_anchor = torch.full((batch_size, target_channels, height, width), 0.1)
    target = target_anchor.clone()
    target[0] += 0.2
    present = torch.ones(batch_size, frames, dtype=torch.bool)
    present[0, -1] = False
    return {
        "observations": source_anchor[:, None].expand(-1, frames, -1, -1, -1).clone(),
        "observation_valid": torch.ones(batch_size, frames, 1, height, width),
        "observation_days": torch.stack(
            (
                torch.full((batch_size,), -1.0),
                torch.full((batch_size,), -3.0),
            ),
            dim=1,
        ),
        "observation_present": present,
        "source_anchor": source_anchor,
        "source_anchor_valid": torch.ones(batch_size, 1, height, width),
        "target_anchor": target_anchor,
        "target_anchor_valid": torch.ones(batch_size, 1, height, width),
        "source_anchor_days": torch.full((batch_size,), -5.0),
        "target_anchor_days": torch.full((batch_size,), -4.0),
        "target": target,
        "target_valid": torch.ones(batch_size, 1, height, width),
        "task_mode": ["translation" if index % 2 == 0 else "forecast" for index in range(batch_size)],
    }


def _tagged_source_shuffle_batch(*, batch_size: int = 5) -> tuple[dict[str, object], torch.Tensor]:
    batch = _batch("sar_to_optical", batch_size=batch_size)
    observations = batch["observations"]
    assert isinstance(observations, torch.Tensor)
    tagged = observations.clone()
    tags = torch.arange(batch_size, dtype=tagged.dtype)
    tagged[:, :, 0, 0, 0] = tags[:, None]
    return {**batch, "observations": tagged}, tags


def _source_shuffle_mapping(
    batch: dict[str, object],
    *,
    seed: int,
    direction: str = "sar_to_optical",
    batch_index: int = 0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    shuffled = _source_shuffle_batch(
        batch,
        seed=seed,
        direction=direction,  # type: ignore[arg-type]
        batch_index=batch_index,
        generator=generator,
    )
    observations = shuffled["observations"]
    assert isinstance(observations, torch.Tensor)
    return observations[:, 0, 0, 0, 0].clone()


def _selection_report(
    *,
    optical_structural: float | None,
    sar_structural: float | None,
    optical_rmse: float = 0.5,
    sar_rmse: float = 2.0,
    variant: str | None = None,
) -> dict[str, object]:
    optical: dict[str, float] = {
        "rmse": optical_rmse,
        "anchor_rmse": 1.0,
        "sam_deg": 1.0,
        "scene_improved_fraction": 1.0,
    }
    sar: dict[str, float] = {
        "sar_db_rmse": sar_rmse,
        "sar_db_anchor_rmse": 10.0,
        "sar_db_bias": 0.0,
        "scene_improved_fraction": 1.0,
    }
    if optical_structural is not None:
        optical["structural_rmse"] = optical_structural
    if sar_structural is not None:
        sar["structural_rmse"] = sar_structural
    report: dict[str, object] = {
        "directions": {
            "sar_to_optical": {
                "all": {"all": optical},
                "regimes": {"translation/n=one": {"all": optical}},
            },
            "optical_to_sar": {
                "all": {"all": sar},
                "regimes": {"translation/n=one": {"all": sar}},
            },
        }
    }
    if variant is not None:
        report["variant"] = {"name": variant}
    return report


def test_masked_structural_lowpass_ignores_invalid_nan_and_extreme_pixels() -> None:
    valid = torch.ones(1, 1, 11, 11)
    valid[..., 5, 5] = 0.0
    reference = torch.zeros(1, 2, 11, 11)
    contaminated = reference.clone()
    contaminated[..., 5, 5] = float("nan")
    contaminated[..., 0, 0] = 1.0e30
    valid[..., 0, 0] = 0.0

    reference_lowpass, reference_support = _masked_box_lowpass(reference, valid)
    contaminated_lowpass, contaminated_support = _masked_box_lowpass(contaminated, valid)

    expected_mask = valid.bool().expand_as(reference)
    assert torch.equal(contaminated_support, reference_support)
    assert not bool(torch.isnan(contaminated_lowpass[expected_mask]).any())
    torch.testing.assert_close(
        contaminated_lowpass[expected_mask],
        reference_lowpass[expected_mask],
        rtol=0.0,
        atol=0.0,
    )


def test_structural_metrics_exist_for_all_change_task_and_observation_buckets() -> None:
    report = evaluate_sopat_loaders(
        None,
        {
            "sar_to_optical": [_batch("sar_to_optical")],
            "optical_to_sar": [_batch("optical_to_sar")],
        },
        variant=SOPATVariantConfig("anchor_copy"),
    )

    directions = report["directions"]
    assert isinstance(directions, dict)
    for direction in ("sar_to_optical", "optical_to_sar"):
        directional = directions[direction]
        assert isinstance(directional, dict)
        reports = [
            directional["all"],
            directional["by_task"]["translation"],
            directional["by_task"]["forecast"],
            directional["by_observation_count"]["one"],
            directional["by_observation_count"]["two_to_three"],
        ]
        for grouped in reports:
            assert isinstance(grouped, dict)
            for change in ("all", "changed", "unchanged"):
                metrics = grouped[change]
                assert isinstance(metrics, dict)
                assert "structural_rmse" in metrics
                assert "anchor_structural_rmse" in metrics
        overall = directional["all"]["all"]
        assert isinstance(overall, dict)
        assert math.isfinite(float(overall["structural_rmse"]))
        assert math.isfinite(float(overall["anchor_structural_rmse"]))


def test_source_shuffle_is_seeded_by_variant_direction_and_batch_ordinal() -> None:
    batch, tags = _tagged_source_shuffle_batch()

    first = _source_shuffle_mapping(batch, seed=71, batch_index=3)
    repeated = _source_shuffle_mapping(batch, seed=71, batch_index=3)
    different_seed = _source_shuffle_mapping(batch, seed=72, batch_index=3)
    different_direction = _source_shuffle_mapping(
        batch,
        seed=71,
        direction="optical_to_sar",
        batch_index=3,
    )
    different_batch = _source_shuffle_mapping(batch, seed=71, batch_index=4)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different_seed)
    assert not torch.equal(first, different_direction)
    assert not torch.equal(first, different_batch)
    assert not torch.equal(first, tags)


def test_source_shuffle_does_not_depend_on_global_torch_rng_or_generator_device() -> None:
    batch, _ = _tagged_source_shuffle_batch()
    torch.manual_seed(1)
    _ = torch.rand(31)
    first = _source_shuffle_mapping(batch, seed=71, batch_index=2)
    torch.manual_seed(999)
    _ = torch.rand(127)
    second = _source_shuffle_mapping(batch, seed=71, batch_index=2)

    assert torch.equal(first, second)

    if torch.cuda.is_available():
        cuda_generator = torch.Generator(device="cuda").manual_seed(123)
        cuda_generator_mapping = _source_shuffle_mapping(
            batch,
            seed=71,
            batch_index=2,
            generator=cuda_generator,
        )
        assert torch.equal(first, cuda_generator_mapping)


def test_source_shuffle_metadata_records_reproducible_planner() -> None:
    metadata = SOPATVariantConfig("source_shuffle", seed=71).metadata()

    assert metadata["seed"] == 71
    assert metadata["shuffle_planner"] == "stable_cyclic_offset_v1"
    assert metadata["shuffle_key"] == ("variant.seed", "direction", "batch_index")
    assert metadata["shuffle_generator"] == "ignored_for_reproducibility"


def test_source_shuffle_gate_uses_structural_degradation_in_both_directions() -> None:
    config = SOPATSelectionConfig(
        phase="feasibility",
        required_tasks=("translation",),
        required_observation_counts=("one",),
        feasibility_overall_anchor_ratio=2.0,
        feasibility_bucket_anchor_ratio=2.0,
        feasibility_source_shuffle_min_degradation=0.01,
    )
    candidate = _selection_report(optical_structural=0.1, sar_structural=0.2)
    passing_shuffle = _selection_report(
        optical_structural=0.102,
        sar_structural=0.205,
        optical_rmse=0.0,
        sar_rmse=0.0,
        variant="source_shuffle",
    )
    passing = select_sopat_candidate(
        candidate,
        config,
        source_shuffle_report=passing_shuffle,
    )
    assert passing.eligible

    failing_shuffle = _selection_report(
        optical_structural=0.102,
        sar_structural=0.2,
        optical_rmse=99.0,
        sar_rmse=99.0,
        variant="source_shuffle",
    )
    failing = select_sopat_candidate(
        candidate,
        config,
        source_shuffle_report=failing_shuffle,
    )
    assert not failing.eligible
    assert any(
        "source_shuffle_insufficient_structural_degradation:optical_to_sar" in item
        for item in failing.failures
    )


def test_source_shuffle_gate_fails_closed_when_structural_metric_is_missing() -> None:
    config = SOPATSelectionConfig(
        phase="feasibility",
        required_tasks=("translation",),
        required_observation_counts=("one",),
        feasibility_overall_anchor_ratio=2.0,
        feasibility_bucket_anchor_ratio=2.0,
    )
    candidate = _selection_report(optical_structural=0.1, sar_structural=0.2)
    missing = _selection_report(
        optical_structural=None,
        sar_structural=0.25,
        variant="source_shuffle",
    )

    result = select_sopat_candidate(candidate, config, source_shuffle_report=missing)

    assert not result.eligible
    assert any(
        "missing_source_shuffle_structural_metrics:sar_to_optical" in item
        for item in result.failures
    )


def test_full_sar_gate_still_uses_sar_db_rmse_not_structural_rmse() -> None:
    config = SOPATSelectionConfig(
        phase="full",
        required_tasks=("translation",),
        required_observation_counts=("one",),
        full_overall_anchor_ratio=1.0,
        full_bucket_anchor_ratio=1.0,
        full_optical_rmse_max=1.0,
        full_optical_sam_deg_max=2.0,
        full_sar_db_rmse_max=5.0,
        full_sar_db_bias_abs_max=1.0,
        full_scene_improved_fraction_min=0.5,
        full_source_shuffle_min_degradation=0.01,
    )
    candidate = _selection_report(
        optical_structural=0.1,
        sar_structural=0.01,
        optical_rmse=0.1,
        sar_rmse=5.5,
    )
    shuffle = _selection_report(
        optical_structural=0.102,
        sar_structural=0.011,
        variant="source_shuffle",
    )

    result = select_sopat_candidate(candidate, config, source_shuffle_report=shuffle)

    assert not result.eligible
    assert any("sar_rmse_gate:5.5" in item for item in result.failures)
    assert not any("source_shuffle_insufficient_structural_degradation" in item for item in result.failures)
