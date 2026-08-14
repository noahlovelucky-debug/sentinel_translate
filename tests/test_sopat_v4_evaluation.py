from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from sentinel_v4.evaluation import (
    SOPATSelectionConfig,
    SOPATVariantConfig,
    _global_cross_tile_counterfactual_batch,
    _masked_box_lowpass,
    _source_shuffle_batch,
    evaluate_sopat_loaders,
    predict_sopat_variant,
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
        "sopat_example_id": [f"recipient-{index}" for index in range(batch_size)],
        "sopat_tile": [f"tile-{index}" for index in range(batch_size)],
        "sopat_grid_id": [f"grid-{index}" for index in range(batch_size)],
    }


def _global_counterfactual_batch(direction: str, *, batch_size: int = 2) -> dict[str, object]:
    batch = _batch(direction, batch_size=batch_size)
    observations = batch["observations"]
    valid = batch["observation_valid"]
    assert isinstance(observations, torch.Tensor)
    assert isinstance(valid, torch.Tensor)
    donor = observations.clone()
    donor[:, :, 0, 0, 0] = torch.arange(batch_size, dtype=donor.dtype)[:, None] + 10.0
    return {
        **batch,
        "counterfactual_observation_values": donor,
        "counterfactual_observation_valid": valid.clone(),
        "sopat_cf_donor_sample_id": [f"donor-{index}" for index in range(batch_size)],
        "sopat_cf_donor_tile": [f"donor-tile-{index}" for index in range(batch_size)],
        "sopat_cf_donor_grid_id": [f"donor-grid-{index}" for index in range(batch_size)],
        "sopat_cf_tier": ["same_task_exact_n"] * batch_size,
        "sopat_cf_plan_hash": ["a" * 64] * batch_size,
    }


class _EchoCandidateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forward(self, **inputs: object) -> dict[str, torch.Tensor]:
        self.calls.append(dict(inputs))
        observations = inputs["observations"]
        target_anchor = inputs["target_anchor"]
        assert isinstance(observations, torch.Tensor)
        assert isinstance(target_anchor, torch.Tensor)
        source_effect = observations[:, 0].mean(dim=(1, 2, 3), keepdim=True)
        value = target_anchor + source_effect
        return {
            "physical": value,
            "candidate_physical": value + 0.1,
            "pre_projection_violation": torch.zeros(value.shape[0], device=value.device),
        }


class _PlanLoader:
    def __init__(self, batches: list[dict[str, object]], *, direction: str) -> None:
        self._batches = batches
        self.dataset = SimpleNamespace(
            hard_negative_plan=SimpleNamespace(
                mapping_metadata={
                    "planner": "global_cross_tile_hard_v1",
                    "direction": direction,
                    "split": "validation_temporal",
                    "plan_hash": ("a" if direction == "sar_to_optical" else "b") * 64,
                    "coverage": 1.0,
                    "cross_tile_coverage": 1.0,
                    "tier_counts": {"same_task_exact_n": 3},
                }
            )
        )

    def __iter__(self):
        return iter(self._batches)


def _select_batch(batch: dict[str, object], order: list[int]) -> dict[str, object]:
    indices = torch.tensor(order)
    batch_size = len(order)
    result: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.shape[0] >= batch_size:
            result[key] = value.index_select(0, indices)
        elif isinstance(value, list) and len(value) >= batch_size:
            result[key] = [value[index] for index in order]
        else:
            result[key] = value
    return result


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
        "anchor_sam_deg": 1.1,
        "ndvi_mae": 0.1,
        "anchor_ndvi_mae": 0.2,
        "edge_f1": 0.8,
        "anchor_edge_f1": 0.7,
        "scene_improved_fraction": 1.0,
    }
    sar: dict[str, float] = {
        "sar_db_rmse": sar_rmse,
        "sar_db_anchor_rmse": 10.0,
        "sar_db_bias": 0.0,
        "edge_f1": 0.8,
        "anchor_edge_f1": 0.81,
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


def _global_selection_report(*, candidate_structural: tuple[float, float] = (0.1, 0.2)) -> dict[str, object]:
    report = _selection_report(optical_structural=0.1, sar_structural=0.2)
    directions = report["directions"]
    assert isinstance(directions, dict)
    candidate_directions: dict[str, object] = {}
    for direction, structural in zip(("sar_to_optical", "optical_to_sar"), candidate_structural, strict=True):
        candidate = _selection_report(
            optical_structural=structural if direction == "sar_to_optical" else 0.1,
            sar_structural=structural if direction == "optical_to_sar" else 0.2,
        )["directions"][direction]
        assert isinstance(candidate, dict)
        candidate_directions[direction] = candidate
    report["candidate_directions"] = candidate_directions
    return report


def _global_counterfactual_report(*, candidate_structural: tuple[float, float] = (0.102, 0.205)) -> dict[str, object]:
    report = _selection_report(
        optical_structural=0.102,
        sar_structural=0.205,
        variant="global_cross_tile",
    )
    report["variant"].update(  # type: ignore[index]
        {
            "planner": "global_cross_tile_hard_v1",
            "plan_hash": "b" * 64,
            "coverage": 1.0,
            "cross_tile_coverage": 1.0,
            "tier_counts": {"same_task_exact_n": 2},
        }
    )
    directions = report["directions"]
    assert isinstance(directions, dict)
    candidate_directions: dict[str, object] = {}
    for direction, structural in zip(("sar_to_optical", "optical_to_sar"), candidate_structural, strict=True):
        candidate = _selection_report(
            optical_structural=structural if direction == "sar_to_optical" else 0.1,
            sar_structural=structural if direction == "optical_to_sar" else 0.2,
        )["directions"][direction]
        assert isinstance(candidate, dict)
        candidate_directions[direction] = candidate
    report["candidate_directions"] = candidate_directions
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


def test_global_counterfactual_metadata_records_reproducible_planner() -> None:
    metadata = SOPATVariantConfig("global_cross_tile", seed=71).metadata()

    assert metadata["seed"] == 71
    assert metadata["planner"] == "global_cross_tile_hard_v1"
    assert metadata["counterfactual_source"] == "dataset_global_plan"


def test_global_counterfactual_overrides_only_donor_values_and_valid() -> None:
    batch = _global_counterfactual_batch("sar_to_optical")
    observations = batch["observations"]
    days = batch["observation_days"]
    present = batch["observation_present"]
    assert isinstance(observations, torch.Tensor)
    assert isinstance(days, torch.Tensor)
    assert isinstance(present, torch.Tensor)

    routed = _global_cross_tile_counterfactual_batch(batch)
    replacement = batch["counterfactual_observation_values"]
    replacement_valid = batch["counterfactual_observation_valid"]
    assert isinstance(replacement, torch.Tensor)
    assert isinstance(replacement_valid, torch.Tensor)
    torch.testing.assert_close(routed["observations"], replacement)
    torch.testing.assert_close(routed["observation_valid"], replacement_valid)
    torch.testing.assert_close(routed["observation_days"], days)
    torch.testing.assert_close(routed["observation_present"], present)
    assert routed["target"] is batch["target"]
    assert routed["target_anchor"] is batch["target_anchor"]


def test_global_counterfactual_is_batch_order_invariant_and_target_free() -> None:
    batch = _global_counterfactual_batch("sar_to_optical", batch_size=3)
    model = _EchoCandidateModel()
    first = predict_sopat_variant(model, batch, "sar_to_optical", "global_cross_tile")
    order = [2, 0, 1]
    reordered = _select_batch(batch, order)
    second = predict_sopat_variant(model, reordered, "sar_to_optical", "global_cross_tile")
    torch.testing.assert_close(second.values, first.values.index_select(0, torch.tensor(order)))
    torch.testing.assert_close(
        second.candidate_values,
        first.candidate_values.index_select(0, torch.tensor(order)),  # type: ignore[union-attr]
    )
    forward_inputs = model.calls[0]
    assert "target" not in forward_inputs
    assert "target_valid" not in forward_inputs
    target_anchor = batch["target_anchor"]
    donor_values = batch["counterfactual_observation_values"]
    assert isinstance(target_anchor, torch.Tensor)
    assert isinstance(donor_values, torch.Tensor)
    torch.testing.assert_close(forward_inputs["target_anchor"], target_anchor)  # type: ignore[arg-type]
    torch.testing.assert_close(forward_inputs["observations"], donor_values)  # type: ignore[arg-type]


def test_global_counterfactual_evaluation_is_invariant_to_batch_size_and_order() -> None:
    full_batches = {
        direction: _global_counterfactual_batch(direction, batch_size=3)
        for direction in ("sar_to_optical", "optical_to_sar")
    }
    full_loaders = {
        direction: _PlanLoader([batch], direction=direction)
        for direction, batch in full_batches.items()
    }
    reordered_loaders = {
        direction: _PlanLoader(
            [_select_batch(batch, [2]), _select_batch(batch, [0, 1])], direction=direction
        )
        for direction, batch in full_batches.items()
    }

    first = evaluate_sopat_loaders(
        _EchoCandidateModel(), full_loaders, variant=SOPATVariantConfig("global_cross_tile")
    )
    second = evaluate_sopat_loaders(
        _EchoCandidateModel(), reordered_loaders, variant=SOPATVariantConfig("global_cross_tile")
    )

    assert "candidate" not in first["directions"]["sar_to_optical"]  # type: ignore[index]
    assert set(first["candidate_directions"]) == {"sar_to_optical", "optical_to_sar"}  # type: ignore[arg-type]
    variant = first["variant"]
    assert isinstance(variant, dict)
    assert variant["planner"] == "global_cross_tile_hard_v1"
    assert variant["coverage"] == 1.0
    assert variant["cross_tile_coverage"] == 1.0
    assert set(variant["plan_hashes"]) == {"sar_to_optical", "optical_to_sar"}  # type: ignore[arg-type]
    for direction in ("sar_to_optical", "optical_to_sar"):
        for report_key in ("directions", "candidate_directions"):
            first_metrics = first[report_key][direction]["all"]["all"]  # type: ignore[index]
            second_metrics = second[report_key][direction]["all"]["all"]  # type: ignore[index]
            for metric in ("rmse", "anchor_rmse", "structural_rmse"):
                assert float(second_metrics[metric]) == pytest.approx(float(first_metrics[metric]))


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


def test_policy_v3_requires_global_plan_metadata_and_candidate_degradation() -> None:
    config = SOPATSelectionConfig(
        phase="feasibility",
        policy_version="sopat_v4_quality_gate_v3",
        required_tasks=("translation",),
        required_observation_counts=("one",),
        feasibility_overall_anchor_ratio=2.0,
        feasibility_bucket_anchor_ratio=2.0,
        feasibility_source_shuffle_min_degradation=0.01,
        feasibility_candidate_source_shuffle_min_degradation=0.01,
    )
    passing = select_sopat_candidate(
        _global_selection_report(), config, source_shuffle_report=_global_counterfactual_report()
    )
    assert passing.eligible

    candidate_failure = select_sopat_candidate(
        _global_selection_report(),
        config,
        source_shuffle_report=_global_counterfactual_report(candidate_structural=(0.1, 0.2)),
    )
    assert not candidate_failure.eligible
    assert any("global_counterfactual_candidate_insufficient" in item for item in candidate_failure.failures)

    legacy = _global_counterfactual_report()
    legacy["variant"]["name"] = "source_shuffle"  # type: ignore[index]
    legacy_failure = select_sopat_candidate(_global_selection_report(), config, source_shuffle_report=legacy)
    assert not legacy_failure.eligible
    assert "legacy_global_counterfactual_planner" in legacy_failure.failures

    incomplete = _global_counterfactual_report()
    incomplete["variant"]["coverage"] = 0.99  # type: ignore[index]
    incomplete_failure = select_sopat_candidate(
        _global_selection_report(), config, source_shuffle_report=incomplete
    )
    assert not incomplete_failure.eligible
    assert any("incomplete_global_counterfactual_coverage" in item for item in incomplete_failure.failures)


def test_policy_v3_keeps_feasibility_quality_gates_enabled() -> None:
    config = SOPATSelectionConfig(
        phase="feasibility",
        policy_version="sopat_v4_quality_gate_v3",
        required_tasks=("translation",),
        required_observation_counts=("one",),
        feasibility_overall_anchor_ratio=2.0,
        feasibility_bucket_anchor_ratio=2.0,
    )
    report = _global_selection_report()
    directions = report["directions"]
    assert isinstance(directions, dict)
    sar_to_optical = directions["sar_to_optical"]
    assert isinstance(sar_to_optical, dict)
    metrics = sar_to_optical["all"]["all"]
    assert isinstance(metrics, dict)
    metrics["scene_improved_fraction"] = 0.49

    result = select_sopat_candidate(
        report,
        config,
        source_shuffle_report=_global_counterfactual_report(),
    )

    assert not result.eligible
    assert any("scene_improvement_gate:sar_to_optical" in item for item in result.failures)


@pytest.mark.parametrize(
    ("direction", "field", "value", "failure"),
    [
        ("sar_to_optical", "scene_improved_fraction", 0.49, "scene_improvement_gate"),
        ("sar_to_optical", "sam_deg", 1.11, "optical_sam_anchor_gate"),
        ("sar_to_optical", "ndvi_mae", 0.21, "optical_ndvi_anchor_gate"),
        ("sar_to_optical", "edge_f1", 0.69, "optical_edge_anchor_gate"),
        ("optical_to_sar", "sar_db_bias", 0.51, "sar_bias_gate"),
        ("optical_to_sar", "edge_f1", 0.78, "sar_edge_anchor_gate"),
    ],
)
def test_feasibility_quality_gates_fail_closed(
    direction: str, field: str, value: float, failure: str
) -> None:
    candidate = _selection_report(optical_structural=0.1, sar_structural=0.2)
    directional = candidate["directions"][direction]  # type: ignore[index]
    directional["all"]["all"][field] = value
    shuffle = _selection_report(
        optical_structural=0.102,
        sar_structural=0.205,
        variant="source_shuffle",
    )
    result = select_sopat_candidate(
        candidate,
        SOPATSelectionConfig(
            phase="feasibility",
            required_tasks=("translation",),
            required_observation_counts=("one",),
            feasibility_overall_anchor_ratio=2.0,
            feasibility_bucket_anchor_ratio=2.0,
        ),
        source_shuffle_report=shuffle,
    )

    assert not result.eligible
    assert any(failure in item for item in result.failures)


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
