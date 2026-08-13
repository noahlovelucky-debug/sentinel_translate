# SOPAT V4 Feasibility Comparison

## Scope

- Validation source: SOPAT V4 validation loaders with fixed-center crops.
- Canonical grid: 10 m only. This report makes no arbitrary-GSD claim.
- Models compared on the same V4 validation batches:
  - `anchor_copy`: registered target-anchor copy.
  - `v4_ema`: SOPAT V4 EMA checkpoint.
  - `v2_latest_checkpoint`: direction-specific paired-temporal V2 `latest` checkpoint using the complete V4 source observation set and the same registered V4 anchor pair.
  - `source_shuffle`: V4 EMA source-observation shuffle counterfactual.
- `v3_2_best_reference` is recorded only as an input-mismatched reference. It is not loaded, evaluated, or included in relative-improvement claims.
- The target image and target validity mask are passed only to post-forward metric/panel code, never to a model forward.

## Reproducible Invocation

```bash
PYTHONPATH=src python scripts/compare_sopat_v4_feasibility.py \
  --config configs/sopat_v4_feasibility_local.yaml \
  --v4-checkpoint RESULTS/V4_PHYSICAL_BEST.pt \
  --v2-sar-to-optical-checkpoint RESULTS/V2_SAR_TO_OPTICAL.pt \
  --v2-optical-to-sar-checkpoint RESULTS/V2_OPTICAL_TO_SAR.pt \
  --v3-2-best-reference RESULTS/V3_2_BEST.pt \
  --output RESULTS/sopat_v4_feasibility \
  --device cuda

PYTHONPATH=src python scripts/render_sopat_v4_panels.py \
  --input RESULTS/sopat_v4_feasibility/panel_payloads \
  --output RESULTS/sopat_v4_feasibility/panels
```

## Checkpoint Record

| Route | Checkpoint | Direction | Input protocol | Status |
| --- | --- | --- | --- | --- |
| V4 EMA | `TODO` | both | full SOPAT V4 observation set | `TODO` |
| V2 latest checkpoint | `TODO` | S1 to S2 | complete V4 source observation set | `TODO` |
| V2 latest checkpoint | `TODO` | S2 to S1 | complete V4 source observation set | `TODO` |
| V3.2 best | `TODO` | `TODO` | input-mismatched reference | not evaluated |

## Results

Use `comparison.json` as the source of truth. It retains `overall`, `by_task`, `by_observation_count`, `changed`, `unchanged`, and joint task/count regimes for both directions. Positive `relative_improvement` always means the named candidate is better under the metric's stated orientation.

| Direction | Route | Overall physical RMSE | Edge F1 | PSD log L1 | Changed RMSE | Scene improved fraction | Source-shuffle RMSE | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S1 to S2 | anchor copy | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | n/a | registered target anchor |
| S1 to S2 | V2 latest checkpoint | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | n/a | full-set checkpoint adapter |
| S1 to S2 | V4 EMA | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | full V4 input set |
| S2 to S1 | anchor copy | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | n/a | registered target anchor |
| S2 to S1 | V2 latest checkpoint | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | n/a | full-set checkpoint adapter |
| S2 to S1 | V4 EMA | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | `TODO` | full V4 input set |

## Panel Scales

- Optical target images: canonical S2 RGB channels B04/B03/B02, fixed reflectance range `[0.00, 1.00]`.
- Optical V4 absolute error: fixed reflectance range `[0.00, 1.00]`.
- SAR target images: VV fixed at `[-35, 5]` dB and VH fixed at `[-45, -5]` dB.
- SAR V4 absolute error: fixed range `[0, 40]` dB.
- Invalid pixels are black. No per-panel contrast stretching is applied.
