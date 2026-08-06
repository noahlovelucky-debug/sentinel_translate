# Sentinel Translate V3.2

V3.2 separates low-error deterministic translation from predictable detail and stochastic
texture. It is an independent repository at `/data/code/sentinel_translat/v3.2`; V3.1 and
V3.1.1 checkpoints are never modified.

## Output contract

`translate(..., mode="physical")` returns only the deterministic radiometric prediction.
This is the primary RMSE/SAM result. `mode="visual"` returns:

```text
physical + deterministic_detail + alpha * stochastic_texture
```

Optical composition is performed in logit space and mapped through sigmoid. The result also
reports `deterministic_detail`, `stochastic_residual`, `residual_amplitude`, and
`pre_projection_violation`. A Visual checkpoint is publishable only when its complete
validation report passes the 5% RMSE guardrail and all perceptual, edge, PSD, bounds, and SAR
gates.

## Architecture

- The physical encoder retains the shared 12-layer Transformer and adds rank-64 direction
  adapters after layers 3/6/9/12 plus direction-specific radiometric correction.
- `MultiscaleDetailHead` consumes H/1, H/2, H/4, and H/8 features. It owns predictable roads,
  roofs, and field boundaries.
- The shared residual codec has optical RGB and SAR VV/VH I/O heads, 4x compression, and a
  standardized 16-channel latent.
- Residual-DiT is 512 wide with 8 layers and 8 heads. Every block receives all four pyramid
  levels through zero-initialized gates.
- Scene-conditioned 4x4-block robust RMS predicts texture amplitude. A validation calibration
  command finds the largest alpha that satisfies the RMSE/bias guardrail.
- `delta_days=0/1` high-frequency weights are `1.0/0.25`; gaps 2/3 and patches failing the
  registration/cloud/validity audit have exact zero residual gradient.

## Verify

```bash
cd /data/code/sentinel_translat/v3.2
PYTHONPATH=src pytest -q
ruff check .
PYTHONPATH=src python -m sentinel_v3.cli --config configs/sentinel_v3.yaml validation-protocol
```

The fixed protocol contains exactly 463 sorted `validation_temporal` pairs. Its hash binds the
pair IDs, center crop, SCL mask, channel order, and reflectance/dB units. Reports with different
hashes cannot be combined for checkpoint selection.

## Physical foundation

Re-evaluate V1 Mean, V2 Refiner, and V3.1 physical steps 4k/6k/8k/10k/12k with one protocol:

```bash
bash scripts/evaluate_physical_candidates.sh
```

The physical hard gates are SAR-to-optical RMSE <= 0.03909, SAM <= 5.716 degrees,
optical-to-SAR RMSE <= 5.0 dB, and absolute bias <= 0.5 dB. The recovery run uses
`configs/physical_recovery.yaml`: shared encoder LR `2e-6`, physical decoder LR `1e-5`, and
direction adapter LR `1e-4`. It evaluates full candidates at 4k/6k/8k/10k/12k and stops after
five comparable validations without improvement. Older checkpoints may be used only with
`--init-model`; `--resume` accepts V3.2 format-v4 checkpoints from the same stage.

The selected physical predictor also has a train-only seasonal memory for locations observed
in 2017-2018. Optical uses six unique acquisitions with an exponential 30-day seasonal kernel
for spectral direction and a `0.75` amplitude blend. SAR uses eight same-orbit acquisitions
with a uniform robust seasonal mean and a `0.80` blend. The manifest hash and all calibration
values are stored in the checkpoint. `Observation.location_id` and `pixel_window` opt into this
path; unseen spatial tiles fall back exactly to the neural physical model.

```bash
PYTHONPATH=src python -m sentinel_v3.cli --config configs/physical_recovery.yaml \
  configure-temporal-prior \
  --checkpoint checkpoints_v32_condition_pilot/physical/step_0004000.pt \
  --output checkpoints_v32_temporal/physical_candidate.pt
```

## Staged training

The required local order is physical -> codec 20k -> detail 20k -> flow pilots/full 40k ->
balance 5k. Run the 64-patch connectivity test first:

```bash
bash scripts/connectivity_64.sh
```

The full eight-GPU launcher resumes each stage automatically. It stops after the 1k and 5k
pilots unless automated checks and explicit manual panel review pass:

```bash
MANUAL_VISUAL_PASS=1 bash scripts/launch_8gpu.sh
```

Validation runs every 1k on 32 fixed samples and at the configured full milestones on all 463
samples. Selection evaluates EMA weights, matching normal checkpoint loading. Reports and panels
are written under `reports_v32_recovery`; format-v4 checkpoints and `latest.pt`,
`best_physical_candidate.pt`, `best_physical.pt`, `best_visual.pt`, and `best_joint.pt` are
confined to `checkpoints_v32_recovery`. A candidate symlink records progress but cannot unlock
high-frequency training; only `best_physical.pt` can do that.

Codec and deterministic-detail gates are also computed on fixed manifest crops, not the most
recent training batch. Flow requires both gates in its initializer. The detail gate requires at
least 30% MAE improvement over zero output independently for Optical and SAR.

Calibrate a trained flow checkpoint before final selection:

```bash
PYTHONPATH=src python -m sentinel_v3.cli --config configs/flow.yaml calibrate-alpha \
  --checkpoint checkpoints_v32_recovery/flow/step_0040000.pt \
  --output checkpoints_v32_recovery/flow/step_0040000_calibrated.pt
```

Only after validation selection may the three closed test splits run:

```bash
CHECKPOINT=checkpoints_v32_recovery/best_joint.pt bash scripts/evaluate_closed_splits.sh
```

## Evaluation weights

LPIPS and DISTS are cached once per process. Validation fails immediately if either package or
its weights are unavailable; it never emits `null` metrics. Install the eval extra and run
`scripts/download_eval_weights.sh` before Visual validation.

## DiffusionSat comparison

DiffusionSat ControlNet is a research-only comparison. Its multispectral-conditioned RGB model
and public 512x512 checkpoint must not enter V3.2 training, the main checkpoint, or redistributed
artifacts. The paper is [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/16c3c941409d0581286eff49b180930f-Abstract-Conference.html)
and the comparison checkpoint is on [Zenodo](https://zenodo.org/records/13756246). The checkpoint
license is not declared and must be reported as a risk.

V3.2 defines realism as a plausible conditional distribution with structure constrained by the
input. It does not claim that a stochastic sample recovers the unique true optical image.
