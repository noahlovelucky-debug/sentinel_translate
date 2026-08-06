# V3.2 Physical And High-Frequency Recovery

## Confirmed Failure Modes

- The 10k full validation report failed all four physical gates: optical RMSE
  `0.06973`, SAM `13.36 deg`, SAR RMSE `7.524 dB`, and SAR bias `2.28 dB`.
- Validation omitted the eight-value date/orbit/GSD condition that is always present
  during training. All checkpoint comparisons were therefore evaluated out of
  distribution.
- Every rank-64 adapter was inactive. Both its output projection and trainable scale
  were initialized to zero, so neither received a gradient. All eight scales remained
  exactly zero at step 9k.
- Only one sixth of the previous random GSD combinations was the 10-to-10 m task used
  by the physical acceptance protocol.
- The physical objective did not directly optimize the SAR bias gate, and latent
  alignment used zero metadata even when translation used real metadata.
- Quick 32-sample and full 463-sample scores shared one early-stop state, despite being
  incomparable sample populations.
- The historical `0.03909 / 5.716 deg` optical reference came from 256 patches under
  the legacy four-crops-per-pair evaluator. Under the immutable 463-pair center-crop
  protocol, V1 Mean scores `0.07916 / 16.75 deg` and V2 Refiner scores
  `0.07844 / 16.87 deg`. Keep the requested hard gate, but do not present the legacy
  number as a unified-protocol baseline.

## Physical Recovery

### Protocol Repair

Use one metadata builder for V1, V2, V3.1, V3.2, visual evaluation, and amplitude
calibration. The physical gate remains unchanged. Reports must also include slices by
`delta_days`, orbit, band or polarization, and valid-pixel fraction so an aggregate
improvement cannot hide a failed regime.

Evaluate these immutable candidates before a new long run:

1. V1 Mean and V2 Refiner.
2. V3.1 physical at 4k, 6k, 8k, 10k, and 12k.
3. V3.2 10k with and without metadata, for diagnosis only.

### Recovery Optimization

Initialize from the best compatible physical checkpoint, never its optimizer. Preserve
the imported function with a zero output projection while using a non-zero adapter
multiplier so the first backward pass updates the projection.

Train with separate parameter groups:

- shared encoder: `2e-6`;
- direction adapters, modality adapters, and direction-specific radiometric kernels:
  `1e-4`;
- shared physical decoder: `1e-5`.

The radiometric kernel is descriptor-conditioned and corrects the Optical logits or
the SAR pre-projection value. A second global kernel combines each target channel
descriptor with the complete 11-value date/orbit/GSD scene condition, making observed
orbit-specific bias directly correctable. Both final projections are zero-initialized,
so importing a checkpoint is function-preserving while they receive a gradient on the
first step. Optical corrections are bounded to 2 logit units and SAR corrections to
5 dB-equivalent pre-projection units.

Use 80% native 10-to-10 m examples and 20% multiscale examples. High-frequency stages
use native 10-to-10 m only. Weight physical temporal buckets `1.0/1.0/0.75/0.5`; retain
strict `1.0/0.25/0/0` for high frequency.

Optimize the acceptance quantities directly: optical MSE and SAM, SAR MSE, per-scene
global bias, per-polarization bias, gradients, and structure. Normalize SAM by its
`5.716 deg` hard gate so it remains material beside normalized RMSE. Keep uncertainty
NLL at a small auxiliary weight. Pass the real condition to latent alignment and limit
its weight to `0.02` until the hard metrics stabilize.

Select candidates by the worst normalized gate ratio, not by an unscaled sum. A
`best_physical_candidate.pt` may track progress without authorizing high-frequency
training. Only a full 463-sample pass may create `best_physical.pt`.

### Experiment Ladder

1. Gradient test: adapter output is initially unchanged and its output projection has
   a non-zero gradient on the first backward pass.
2. 64-patch overfit: both directions improve; SAR bias decreases; fixed seed is exact.
3. 1k pilot: fixed 32-sample validation improves the worst normalized gate ratio over
   the imported checkpoint.
4. Full validations at 4k, 6k, 8k, 10k, and 12k. Stop after five comparable validations
   without improvement.
5. If adapters and heads plateau, unfreeze only transformer blocks 9-12. Unfreeze the
   complete shared encoder only if gradient-conflict logs and validation slices show a
   shared representation failure.

## High-Frequency Recovery

Freeze the first physical checkpoint that passes all four gates. Do not let detail,
codec, flow, or balance update it before their own acceptance tests.

### Eligible Data

Use only 2017-2018 train patches with `delta_days <= 1`, registration shift at most
`0.5 px`, valid fraction at least `0.8`, and cloud-shadow fraction at most `0.2`.
Ineligible samples must not update gradients, codec statistics, or amplitude buffers.
Precompute the eligible index instead of repeatedly auditing rejected patches.

### Deterministic Detail

Train the full FPN detail head on `H(target - stopgrad(physical))`. Require at least 30%
high-frequency MAE improvement over zero output before flow training. This branch owns
edges and repeatable structures. Track Edge F1, tiled local spectrum, and low-frequency
leakage in addition to reconstruction loss.

### Codec

Train independent optical/SAR I/O heads with a shared 4x trunk and 16-channel latent.
Use tiled FFT spectra rather than spectra of tile averages. Optical reconstruction uses
real DISTS features; SAR uses radial PSD, local variance, and speckle scale. After codec
weights freeze, recompute exact latent mean/std over the eligible corpus once and freeze
those statistics before flow starts.

### Rectified Flow

Generate only `texture_gt = target_detail - stopgrad(det_detail)`. Inject all four FPN
levels through gated projections. Train velocity, one-step endpoint, gradient, DISTS,
local spectrum, and robust 4x4 amplitude losses. Keep optical chroma energy and SAR
radiometric bias explicit in reports to reject colored white noise and biased speckle.

The 1k pilot must satisfy Visual RMSE at most `1.10 x Physical`; the 5k full pilot must
satisfy `1.05 x`, improve LPIPS and DISTS by at least 3%, and improve both Edge F1 and
optical PSD. A failed 5k pilot stops the flow run. Final selection requires 5%
LPIPS/DISTS improvement, all physical gates, projection violation at most 0.1%, and the
SAR PSD/ENL/histogram gates. Fixed-seed single samples are used for the main result;
best-of-K remains excluded.
