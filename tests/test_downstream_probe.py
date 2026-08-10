from __future__ import annotations

import math
import runpy
from pathlib import Path

import torch

from sentinel_v3.downstream_probe import (
    PROBE_GROUPS,
    ProbeCache,
    ProbeTrainConfig,
    TwoStreamLightUNet,
    binary_confusion,
    evaluate_probe_model,
    evaluate_scene_predictions,
    fit_probe_stats,
    fixed_group_parameter_counts,
    holm_adjust,
    normalize_optical,
    normalize_sar,
    paired_bootstrap_scene_delta,
    paired_permutation_test,
    route_probe_inputs,
    summarize_probe_statistics,
    train_probe_seed,
)


def _cache() -> ProbeCache:
    batch_size, height, width = 6, 8, 8
    sar = torch.empty(batch_size, 2, height, width)
    real_optical = torch.empty(batch_size, 10, height, width)
    synthetic_optical = torch.empty(batch_size, 10, height, width)
    for index in range(batch_size):
        sar[index].fill_(float(index + 1))
        real_optical[index].fill_(float(10 + 2 * index))
        synthetic_optical[index].fill_(float(100 + 10 * index))
    sar_valid = torch.ones(batch_size, 1, height, width, dtype=torch.bool)
    sar_valid[0, :, 0, 0] = False
    sar[0, :, 0, 0] = 999.0
    label = torch.zeros(batch_size, height, width, dtype=torch.long)
    label[:, :, width // 2 :] = 1
    label[0, 0, 0] = -1
    return ProbeCache.from_mapping(
        {
            "sample_id": [f"sample-{index}" for index in range(batch_size)],
            "scene_id": ["train-a", "train-b", "dev-a", "dev-b", "test-a", "test-b"],
            "tile": ["T31" for _ in range(batch_size)],
            "split": ["train", "train", "dev", "dev", "test", "test"],
            "sar": sar,
            "real_optical": real_optical,
            "synthetic_optical": synthetic_optical,
            "label": label,
            "sar_valid": sar_valid,
        }
    )


def test_stats_are_train_only_and_synthetic_uses_real_optical_stats() -> None:
    cache = _cache()
    cache.label[2:].fill_(1)
    stats = fit_probe_stats(cache)
    expected_sar_mean = (191.0 / 127.0, 191.0 / 127.0)
    torch.testing.assert_close(stats.sar_mean.flatten(), torch.tensor(expected_sar_mean))
    torch.testing.assert_close(stats.optical_mean.flatten(), torch.full((10,), 11.0))
    torch.testing.assert_close(stats.optical_std.flatten(), torch.ones(10))
    torch.testing.assert_close(stats.class_counts, torch.tensor([63, 64]))
    expected_weights = torch.tensor([63.0, 64.0]).rsqrt()
    expected_weights /= expected_weights.mean()
    torch.testing.assert_close(stats.class_weights, expected_weights)
    assert stats.label_pixels == 127
    normalized_sar = normalize_sar(cache.sar, cache.sar_valid, stats)
    torch.testing.assert_close(normalized_sar[0, :, 0, 0], torch.zeros(2))
    normalized_synthetic = normalize_optical(cache.synthetic_optical, stats)
    torch.testing.assert_close(normalized_synthetic[0, :, 0, 0], torch.full((10,), 89.0))


def test_all_groups_route_fixed_width_and_missing_streams_are_zero() -> None:
    cache = _cache()
    stats = fit_probe_stats(cache)
    routed = {group: route_probe_inputs(cache, stats, group) for group in PROBE_GROUPS}
    for values in routed.values():
        assert values.sar.shape == (6, 12, 8, 8)
        assert values.optical.shape == (6, 12, 8, 8)
        assert int(torch.count_nonzero(values.sar[:, 2:])) == 0
        assert int(torch.count_nonzero(values.optical[:, 10:])) == 0
    assert int(torch.count_nonzero(routed["sar_only"].optical)) == 0
    assert int(torch.count_nonzero(routed["optical_only"].sar)) == 0
    assert int(torch.count_nonzero(routed["synthetic_optical_only"].sar)) == 0
    torch.testing.assert_close(
        routed["sar_real_optical"].optical,
        routed["optical_only"].optical,
    )
    torch.testing.assert_close(
        routed["sar_synthetic_optical"].optical,
        routed["synthetic_optical_only"].optical,
    )
    mixed_mask = torch.tensor([True, False, True, False, True, False])
    mixed = route_probe_inputs(cache, stats, "sar_mixed_optical", mixed_real_mask=mixed_mask)
    torch.testing.assert_close(mixed.optical[0], routed["sar_real_optical"].optical[0])
    torch.testing.assert_close(mixed.optical[1], routed["sar_synthetic_optical"].optical[1])


def test_mixed_group_evaluation_always_uses_synthetic_optical() -> None:
    class CaptureModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.optical_inputs: list[torch.Tensor] = []

        def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
            self.optical_inputs.append(optical.detach().cpu().clone())
            return sar.new_zeros((sar.shape[0], 2, sar.shape[2], sar.shape[3])) + self.anchor * 0

    cache = _cache()
    stats = fit_probe_stats(cache)
    dev = cache.for_split("dev")
    model = CaptureModel()
    evaluate_probe_model(model, dev, stats, "sar_mixed_optical", batch_size=len(dev))
    expected = route_probe_inputs(dev, stats, "sar_synthetic_optical").optical
    assert len(model.optical_inputs) == 1
    torch.testing.assert_close(model.optical_inputs[0], expected)


def test_every_group_uses_the_same_parameter_count_and_model_shape() -> None:
    counts = fixed_group_parameter_counts(width=4)
    assert len(set(counts.values())) == 1
    model = TwoStreamLightUNet(width=4)
    logits = model(torch.zeros(2, 12, 8, 8), torch.zeros(2, 12, 8, 8))
    assert logits.shape == (2, 2, 8, 8)


def test_scene_metrics_keep_per_scene_confusion_and_scene_equal_iou() -> None:
    label = torch.tensor(
        [
            [[0, 1], [1, -1]],
            [[0, 1], [0, 1]],
        ]
    )
    prediction = torch.tensor(
        [
            [[0, 1], [0, 1]],
            [[0, 1], [0, 1]],
        ]
    )
    confusion = binary_confusion(prediction, label)
    torch.testing.assert_close(confusion, torch.tensor([[3, 0], [1, 3]]))
    evaluation = evaluate_scene_predictions(prediction, label, ("scene-a", "scene-b"))
    assert evaluation.per_scene["scene-a"]["confusion"] == [[1, 0], [1, 1]]
    assert evaluation.per_scene["scene-b"]["confusion"] == [[2, 0], [0, 2]]
    assert math.isclose(float(evaluation.per_scene["scene-a"]["macro_f1"]), 2.0 / 3.0)
    assert math.isclose(float(evaluation.per_scene["scene-a"]["balanced_accuracy"]), 0.75)
    assert math.isclose(float(evaluation.per_scene["scene-a"]["valid_coverage"]), 0.75)
    assert math.isclose(float(evaluation.pooled["valid_coverage"]), 7.0 / 8.0)
    expected = 0.75
    assert evaluation.scene_equal_macro_iou == expected


def test_paired_bootstrap_and_permutation_are_deterministic() -> None:
    baseline = {"a": 0.2, "b": 0.3, "c": 0.4}
    candidate = {"a": 0.3, "b": 0.4, "c": 0.5}
    first = paired_bootstrap_scene_delta(candidate, baseline, resamples=10_000, seed=9)
    second = paired_bootstrap_scene_delta(candidate, baseline, resamples=10_000, seed=9)
    assert first == second
    assert math.isclose(first.estimate, 0.1)
    assert math.isclose(first.ci_lower, 0.1)
    assert math.isclose(first.ci_upper, 0.1)
    permutation_one = paired_permutation_test(candidate, baseline, permutations=10_000, seed=9)
    permutation_two = paired_permutation_test(candidate, baseline, permutations=10_000, seed=999)
    assert permutation_one == permutation_two
    assert permutation_one.exact


def test_holm_adjustment_and_gate_with_oracle_recovery() -> None:
    adjusted = holm_adjust({"first": 0.01, "second": 0.04, "third": 0.03})
    assert adjusted == {"first": 0.03, "third": 0.06, "second": 0.06}
    scores = {
        "sar_only": {"a": 0.20, "b": 0.20, "c": 0.20},
        "sar_real_optical": {"a": 0.25, "b": 0.25, "c": 0.25},
        "sar_synthetic_optical": {"a": 0.24, "b": 0.24, "c": 0.24},
        "sar_mixed_optical": {"a": 0.23, "b": 0.23, "c": 0.23},
    }
    summary = summarize_probe_statistics(scores, bootstrap_resamples=200, permutation_samples=200)
    assert summary["gate"]["passed"] is True
    assert math.isclose(summary["oracle_headroom_recovery"]["recovery"], 0.8)
    blocked = {group: dict(values) for group, values in scores.items()}
    blocked["sar_real_optical"] = {"a": 0.19, "b": 0.19, "c": 0.19}
    blocked_summary = summarize_probe_statistics(
        blocked, bootstrap_resamples=200, permutation_samples=200
    )
    assert blocked_summary["gate"]["passed"] is False


def test_probe_training_is_seed_deterministic_and_selects_dev_epoch() -> None:
    cache = _cache()
    stats = fit_probe_stats(cache)
    config = ProbeTrainConfig(
        epochs=2,
        steps_per_epoch=1,
        batch_size=2,
        eval_batch_size=2,
        width=4,
        augment=True,
    )
    first = train_probe_seed(cache, stats, "sar_only", seed=123, config=config)
    second = train_probe_seed(cache, stats, "sar_only", seed=123, config=config)
    assert first.selected_epoch in {1, 2}
    assert first.dev_history == second.dev_history
    assert first.selected_epoch == second.selected_epoch
    assert math.isfinite(first.selected_dev_scene_equal_macro_iou)


def test_cli_accepts_group_and_seed_subsets_with_skip_summary() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_downstream_scl_probe.py"
    parse_args = runpy.run_path(str(script), run_name="downstream_probe_cli_test")["parse_args"]
    args = parse_args(
        [
            "--cache",
            "cache.pt",
            "--output",
            "report.json",
            "--groups",
            "sar_only",
            "sar_mixed_optical",
            "--seeds",
            "11",
            "29",
            "--skip-summary",
        ]
    )
    assert args.groups == ["sar_only", "sar_mixed_optical"]
    assert args.seeds == [11, 29]
    assert args.skip_summary is True
