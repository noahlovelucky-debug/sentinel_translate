# Sentinel Translate V3.2

> 当前完整中文技术说明、模型框架、监督标签、训练流程、真实指标和可视化：
> [docs/V32_CURRENT_PIPELINE_ZH.md](docs/V32_CURRENT_PIPELINE_ZH.md)

V3.2 separates low-error deterministic translation from predictable detail and stochastic
texture. It is an independent repository at `/data/code/sentinel_translat/v3.2`; V3.1 and
V3.1.1 checkpoints are never modified.

## Output contract

`translate(..., mode="physical")` returns only the deterministic radiometric prediction.
This is the primary RMSE/SAM result. `mode="visual"` returns:

```text
physical + observable_pixel_detail + sampled_orthogonal_innovation
```

Optical composition is performed in logit space and mapped through sigmoid. The result also
reports `deterministic_detail`, `stochastic_residual`, `residual_amplitude`, and
`pre_projection_violation`. A Visual checkpoint is publishable only when its complete
validation report passes the 5% RMSE guardrail and all perceptual, edge, PSD, bounds, and SAR
gates.

## Current architecture

- The physical encoder retains the shared 12-layer Transformer and adds rank-64 direction
  adapters after layers 3/6/9/12 plus direction-specific radiometric correction.
- The currently best fully validated Optical detail is a source-aware, physical-frequency
  anchor. It keeps input-supported edges in pixel space instead of asking a random branch to
  reconstruct them.
- The experimental id bridge uses an exact two-level Haar packet state. LL-to-LL coefficients
  are forced to zero, so generated innovations cannot rewrite the verified physical low
  frequency. This route replaces the rejected learned-codec bridge that produced off-manifold
  checkerboard artifacts.
- A source/physical-conditioned origin predicts `mu`, `log_sigma`, and three-band
  identifiability `q`. Residual-DiT is 512 wide with 8 layers and 8 heads and transports
  `mu + sigma(q) * noise` to the paired Haar residual endpoint using rectified flow.
- Optical currently publishes the deterministic anchor while innovation transport remains
  disabled pending a measured gain. SAR publishes stochastic residuals and improves PSD, ENL,
  histogram, and tail statistics on the complete validation set.
- `delta_days=0/1` high-frequency weights are `1.0/0.25`; gaps 2/3 and patches failing the
  registration/cloud/validity audit have exact zero residual gradient.

The authoritative document linked above contains the exact labels, bridge equations, accepted
and rejected experiments, current 463-sample metrics, visualizations, and paper acceptance
criteria. Older codec/detail/flow commands below are retained for reproducibility and ablation;
they are not claims about the current best visual model.

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

## 2017-2024 Dataset Build

The repository-owned builder creates a reproducible manifest and normalized patch shards from
`/data/data_disk/data_dir` without copying the source TIFF tree. It uses the canonical S2 order
`blue,green,red,rededge1,rededge2,rededge3,nir,nir08,swir16,swir22` and SAR `vv,vh`.
All temporary and atomic-write files are created next to their destination under the output root.

```bash
cd /data/code/sentinel_translat/v3.2
PYTHONPATH=src python scripts/build_dataset_2017_2024.py \
  --raw-root /data/data_disk/data_dir \
  --output-root /data/datasets/sentinel_translate_v32_2017_2024 \
  --workers 8 --audit-only

PYTHONPATH=src python scripts/build_dataset_2017_2024.py \
  --raw-root /data/data_disk/data_dir \
  --output-root /data/datasets/sentinel_translate_v32_2017_2024 \
  --workers 8 --patches-per-pair 16 --build --resume
```

The build writes `manifests/pairs.jsonl`, a dataset-owned validation protocol sidecar,
`audit/audit.json`, one homogeneous V2 shard per training pair, a candidate
`hf_eligibility.json`, and logs. The builder-side eligibility file is explicitly marked
`registration_audited: false` and must not be used for high-frequency training. The formal
order is build, registration audit (which atomically replaces that sidecar), then temporal-prior
precomputation:

```bash
PYTHONPATH=src python scripts/audit_high_frequency.py \
  --index /data/datasets/sentinel_translate_v32_2017_2024/shards/train/index.json \
  --output /data/datasets/sentinel_translate_v32_2017_2024/hf_eligibility.json \
  --workers 8

PYTHONPATH=src python scripts/precompute_temporal_prior_shards.py \
  --shard-index /data/datasets/sentinel_translate_v32_2017_2024/shards/train/index.json \
  --manifest /data/datasets/sentinel_translate_v32_2017_2024/manifests/pairs.jsonl \
  --output /data/datasets/sentinel_translate_v32_2017_2024/temporal_prior \
  --workers 8
```

The canonical output is expected to contain normalized patch shards, not duplicated source rasters.

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

Before detail/flow, build the train-only leave-one-out temporal-prior sidecars once. Each source
pair is aggregated at full-tile resolution and then cropped to exactly the windows recorded in
the immutable training shards. This keeps GeoTIFF access outside the optimizer loop and stores
float16 priors plus coverage masks; reruns reuse completed sidecars.

```bash
PYTHONPATH=src python scripts/precompute_temporal_prior_shards.py \
  --shard-index /data/sentinel_translate/data/shards_v2/train/index.json \
  --manifest /data/sentinel_translate/data/manifests/pairs.jsonl \
  --output /data/sentinel_translate/data/shards_v32_temporal_prior \
  --workers 8
```

The DataLoader verifies every sidecar's pair IDs and windows against its source shard, and applies
the exact same flip/rotation augmentation to imagery, prior, and coverage mask.

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
