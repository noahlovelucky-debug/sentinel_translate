# Sentinel Translate V3.2

The next bidirectional paired-anchor research model is specified in
[docs/SOPAT_V4_RESEARCH_AND_IMPLEMENTATION_ZH.md](docs/SOPAT_V4_RESEARCH_AND_IMPLEMENTATION_ZH.md).
SOPAT V4 uses one shared checkpoint, a registered historical S1/S2 anchor pair,
and an unordered set of 1--N causal source observations. Its first release is
limited to Sentinel-1/Sentinel-2 on the canonical 10 m grid.

The sparse registered-pair temporal image-to-image research track is documented in
[docs/PAIRED_TEMPORAL_V2_DESIGN_ZH.md](docs/PAIRED_TEMPORAL_V2_DESIGN_ZH.md). It
supports one-to-many causal source observations plus one registered SAR/Optical
anchor pair without changing the released single-frame V3.2 API.

Sentinel Translate V3.2 is a bidirectional Sentinel-1/Sentinel-2 conditional image generation
model. It separates a deterministic, radiometrically constrained `physical` output from a
perceptual `visual` output:

```text
physical = deterministic low-error translation
visual   = bounded(physical + observable_detail + sampled_innovation)
```

The current canonical model was trained on the 2017–2022 portion of the local 2017–2024 dataset.
It passes all physical gates and the SAR visual gate on the fixed 141-pair 2023 validation split.
Optical visual passes its RMSE, DISTS, edge, PSD, bounds, and scene-risk constraints, but improves
LPIPS by 3.71% rather than the required 5%. Consequently, the repository currently publishes only
`best_physical.pt`; it does not claim a finished Optical visual model or SOTA test result.

## Documentation

- [SOPAT V4 research hypothesis, model, causal protocol, training, and acceptance](docs/SOPAT_V4_RESEARCH_AND_IMPLEMENTATION_ZH.md)
- [Sparse registered-pair temporal V2 design, data contract, training, and feasibility](docs/PAIRED_TEMPORAL_V2_DESIGN_ZH.md)
- [Current model, labels, bridge formulation, results, and visual interpretation](docs/V32_CURRENT_PIPELINE_ZH.md)
- [Canonical 2017–2024 training and acceptance report](docs/V32_CANONICAL_2017_2024_TRAINING_REPORT_ZH.md)
- [Acceptance criteria](docs/V32_ACCEPTANCE.md)
- [Temporal V1 causal anchor-delta research track](docs/TEMPORAL_V1_CAUSAL_ANCHOR_DELTA_ZH.md)

## Canonical data

```text
root:                    /data/datasets/sentinel_translate_v32_2017_2024
train candidates:        2,050 pairs, 2017-2022
accepted train:          1,947 pairs / 31,152 patches
high-frequency eligible: 14,622 patches
validation_temporal:     141 pairs, 2023
test_spatial:             39 pairs, 2023, closed
test_temporal:           131 pairs, 2024, closed
test_joint:               62 pairs, 2024, closed
patch/crop size:         256 x 256 on a 10 m grid
protocol hash:           f72deee58e7c421bd6af9d96164a272717564f94b7c227e4b38fa4e915f61606
```

Sentinel-2 uses
`blue,green,red,rededge1,rededge2,rededge3,nir,nir08,swir16,swir22`; Sentinel-1 uses `vv,vh`.
High-frequency supervision is restricted to registration-audited train patches with
`delta_days=0/1`. Longer temporal gaps have exact zero residual gradient.

## Model

- Shared 12-layer, 768-wide Transformer physical encoder with rank-64 direction adapters at
  layers 3/6/9/12 and direction-specific radiometric decoders.
- Four-scale FPN conditioning for deterministic detail and residual generation.
- A 4x, 16-channel standardized residual codec for the legacy/SAR path.
- An 8-layer, 512-wide Residual-DiT trained as a conditional residual rectified-flow bridge.
- A two-level, 48-channel Haar packet state for the Optical phase-identifiability path; LL-to-LL
  coefficients are fixed to zero so generated detail cannot rewrite verified low frequency.
- A zero-initialized, null-calibrated orthogonal phase carrier under validation. It admits source
  structure only where observed coherence exceeds a cyclic-shift null hypothesis.

This is not whole-image generation from pure noise. The bridge starts from a source/physical
conditioned origin `mu + sigma(q) * epsilon` and transports only the target residual unexplained by
the frozen physical and deterministic-detail branches.

## Current validation

| Metric | Result | Gate |
| --- | ---: | ---: |
| SAR -> Optical physical RMSE | `0.0326609` | `<=0.03909` |
| SAR -> Optical physical SAM | `5.58075 deg` | `<=5.716 deg` |
| Optical -> SAR physical RMSE | `4.50149 dB` | `<=5.0 dB` |
| Optical -> SAR physical bias | `0.01087 dB` | `<=0.5 dB` |
| Optical visual/physical RGB RMSE | `1.01180x` | `<=1.05x` |
| Optical LPIPS improvement | `3.7111%` | `>=5%`, not passed |
| Optical DISTS improvement | `6.7778%` | `>=5%` |
| Optical pre-projection violation | `0.016716%` | `<=0.1%` |

SAR visual improves PSD distance `0.65475 -> 0.28154`, ENL error `0.18116 -> 0.07391`, histogram
distance `0.004823 -> 0.001350`, P01 error `4.1113 -> 1.6178 dB`, and P99 error
`5.3851 -> 2.9792 dB`.

Authoritative local artifacts:

```text
checkpoints_v32_canonical_2017_2024/final_calibrated.pt
checkpoints_v32_canonical_2017_2024/best_physical.pt
reports_v32_canonical_2017_2024/final_validation.json
reports_v32_canonical_2017_2024/final_validation_panels/
```

The three closed test splits must not be evaluated until validation selection produces a valid
`best_joint.pt`.

## Verify

```bash
cd /data/code/sentinel_translat/v3.2
PYTHONPATH=src pytest -q
ruff check .
git diff --check
```

Re-evaluate the canonical validation checkpoint:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src python -m sentinel_v3.cli \
  --config configs/canonical_2017_2024_phase_transport.yaml \
  evaluate \
  --checkpoint checkpoints_v32_canonical_2017_2024/final_calibrated.pt \
  --split validation_temporal \
  --output reports_v32_canonical_2017_2024/final_validation.json
```

Checkpoint format v4 stores the residual-state and codec versions, validation protocol hash,
best metrics, EMA state, and optimizer/scheduler state. Older checkpoints may be loaded only as
model initialization; their optimizer and scheduler states are not resumed.

## Scope

SAR-to-Optical and Optical-to-SAR are many-to-many mappings. The model targets input-constrained,
distributionally plausible detail; it cannot uniquely recover target-only color, texture, speckle,
or extreme scatterers. Support for a new sensor or resolution requires explicit channel physics,
unit and PSF/MTF calibration, paired training data, and independent validation.
