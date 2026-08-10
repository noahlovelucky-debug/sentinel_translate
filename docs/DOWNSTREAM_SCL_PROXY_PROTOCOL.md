# Downstream SCL Proxy Protocol

This protocol measures a narrow downstream segmentation proxy. It does not
establish land-cover accuracy, ecological validity, or performance on an
independent semantic ground-truth dataset.

## Target

The target is derived from the Sentinel-2 Scene Classification Layer (SCL):

- SCL code `4` is the positive vegetation proxy label (`1`).
- SCL codes `5` and `6` are the negative non-vegetation proxy label (`0`).
- Every other SCL code is ignored (`-1`).

SCL remains a sensor product and pseudo-label. All reported values must be
described as SCL-proxy segmentation results, not land-cover results.

## Frozen Inputs

`configs/downstream_scl_proxy.yaml` binds the run to:

- the canonical 2017--2024 pair manifest;
- the canonical fixed-window train shard index;
- a dedicated downstream cache root;
- crop size `256`;
- `checkpoints_v32_canonical_2017_2024/best_physical.pt` with SHA-256
  `5c26e96ee639609624d350f4ab4eff272a94b9f799fc9ce51579ee1420881363`.

The materialized cache records the source config, source cache provenance,
cache-manifest digest, checkpoint digest, chunk layout, and probe cache
contract. Group reports are stamped with this provenance chain before they can
be merged.

## Split

Only same-day (`delta_days == 0`) records are eligible.

- `train`: canonical train samples whose tile is not in the registered dev set.
- `dev`: canonical train samples from exactly these five tiles:
  - `(0, 0)` `Beijing_r0000_c0000_y000000_x000000_h256_w256`
  - `(0, 4)` `Beijing_r0000_c0004_y000000_x001024_h256_w256`
  - `(2, 2)` `Beijing_r0002_c0002_y000512_x000512_h256_w256`
  - `(4, 0)` `Beijing_r0004_c0000_y001024_x000000_h256_w256`
  - `(4, 4)` `Beijing_r0004_c0004_y001024_x001024_h256_w256`
- `test`: all canonical `unused_spatial` same-day center crops.

The translation validation and closed test splits are never opened by this
protocol: `validation_temporal`, `test_temporal`, `test_spatial`, and
`test_joint` remain closed.

## Leakage Controls

Synthetic optical cache generation reads only SAR rasters and a SAR raw-valid
mask. It calls the frozen checkpoint's physical path with the temporal prior
disabled. Synthetic cache entries are checksum-verified before materialization.

Only after cache finalization does materialization read real SAR, real optical,
and SCL for the downstream payload. The payload contract contains `sample_id`,
`scene_id`, `tile`, `split`, `sar`, `real_optical`, `synthetic_optical`,
`label`, and `sar_valid`.

Normalization uses train-only real SAR and real optical moments. Synthetic
optical uses the real-optical moments. Train labels define fixed inverse-square-
root class weights, normalized to mean one; those weights are shared by every
group and seed.

## Probe Groups

Every group uses the same fixed-capacity light U-Net with two input streams of
12 channels each. A missing stream is an exact zero tensor.

- `sar_only`
- `optical_only` using real optical inputs
- `sar_real_optical`
- `sar_synthetic_optical`
- `synthetic_optical_only`
- `sar_mixed_optical`

The mixed group uses a 50:50 real/synthetic optical choice only during training.
For dev and test it is evaluated through the synthetic-optical route while
retaining the `sar_mixed_optical` result name.

All groups use the same augmentation, AdamW settings, learning rate, batch
size, width, epochs, and steps. The registered run uses seeds `13`, `17`, and
`29`, 12 epochs, 100 steps per epoch, batch size 8, and width 16. Each seed's
epoch is selected by dev scene-equal macroIoU.

## Reporting And Gate

Each seed reports per-scene and pooled confusion matrices, macroIoU, macro F1,
balanced accuracy, and valid-label coverage. Final statistics average each
test-scene macroIoU across the three seeds, then use scenes as paired units.

The final summary reports 10,000-resample paired bootstrap intervals,
two-sided paired sign-flip permutation tests, and Holm-adjusted p-values for:

- `C-A`: `sar_synthetic_optical - sar_only`
- `B-A`: `sar_real_optical - sar_only`
- `C-B`: `sar_synthetic_optical - sar_real_optical`
- `mixed-C`: `sar_mixed_optical - sar_synthetic_optical`

The registered gate passes only when `B-A > 0` and the lower bound of the 95%
paired-bootstrap interval for `C-A` is greater than `0.02`. Oracle headroom
recovery is `(C-A) / (B-A)` at the scene-equal mean.

## Execution And Resume

`scripts/launch_downstream_scl_proxy_8gpu.sh` runs prepare, 8-rank cache
generation, finalize, and materialization. It then launches one group per GPU
on GPUs 0--5, stamps each verified group report, and merges all six reports.

Cache preparation, per-rank generation, and materialization reuse existing
artifacts only when their recorded provenance matches. An existing group report
is reused only after the summarizer verifies its cache provenance, protocol,
seed set, parameter count, and group identity. Any mismatch or failed command
terminates the launch rather than replacing an artifact.
