# High-Frequency Failure Analysis

Status: diagnosis based on `visual/step_0040000.pt`, eight fixed validation
samples, and the stopped balance run. This is an implementation diagnosis, not
a claim that a proposed fix has already been validated.

V3.1.1 implements the corrective mechanics described below: full temporal
gating, endpoint reconstruction, scene-conditioned block amplitude, bounded
optical composition, and low-rate Balance optimization. Their quality impact
still requires a new training run and closed validation.

## Evidence

The visual checkpoint improves SAR PSD, ENL, and histogram statistics relative
to the deterministic mean, but the optical visual output fails its safety and
fidelity gates:

| Observation | Value |
| --- | ---: |
| Physical RGB RMSE | 0.0675 |
| Visual RGB RMSE | 0.1585 |
| Physical / visual optical PSD distance | 0.0117 / 0.0457 |
| Visual optical out-of-bounds fraction | 41.46% |
| Physical / visual Edge F1 | 0.3098 / 0.3323 |

The visual samples visibly contain mostly unconditioned fine noise rather than
scene-aligned detail. See [the panel](assets/validation_000_sar2opt.png).

## Primary causes

1. Invalid temporal supervision enters the residual decoder. In
   `JointObjective._visual_task`, `high_frequency_weights` weights only flow
   loss. `high_frequency_loss(decoded, residual, ...)` is still applied for
   `delta_days` 2 and 3, although the protocol explicitly excludes those pairs
   from high-frequency training. Temporal land-cover and illumination changes
   are therefore learned as texture.

2. The residual generator has no calibrated amplitude target. The decoder is
   supervised to reconstruct an encoded target residual, while sampling starts
   from Gaussian noise and integrates a separate flow field. There is no loss
   that checks the sampled residual RMS, radial PSD, or block-level amplitude.
   A low autoencoder reconstruction loss does not establish a good sampled
   distribution.

3. The output safety constraint is incomplete. `sample_residual` bounds the
   residual with `tanh`, but visual optical output is `physical_rgb + residual`
   without projection back to the reflectance range. The measured 41.46%
   out-of-bounds fraction confirms this is material, not cosmetic.

4. The Balance stage uses the base learning rates. `train()` reduces physical
   learning rates in the Visual phase but not in Balance. It reinitializes the
   optimizer from the Visual checkpoint with decoder LR `1e-4` instead of the
   intended joint-finetuning scale `1e-5`. The observed physical NLL explosion
   is consistent with this tenfold jump.

## Fix order

1. Gate *all* residual losses to `delta_days <= 1`, including reconstruction,
   gradient, spectrum, and SAR speckle losses. Keep physical losses for all
   temporal gaps. Add a regression test proving no residual-branch gradient is
   produced for gaps two and three.

2. Produce an explicit residual-amplitude prediction from the physical scene
   latent. Train it against per-block robust RMS of the true residual, then
   scale sampled residuals by this prediction. Add sampled-residual RMS and
   radial-PSD losses, computed from a short fixed-step sample during validation.

3. Enforce optical validity at the composition boundary. Use a bounded residual
   parameterization, for example predict a logit-space residual and map through
   sigmoid, or project `physical + residual` to `[0, 1]` while reporting the
   pre-projection violation. Clamping alone is a guardrail, not the quality fix.

4. Resume joint training from `visual/step_0040000.pt` with physical encoder and
   decoder LR `1e-5`, residual-DiT LR `1e-4`, gradient clipping per optimizer
   group, and a 1k-step frozen-physical warmup. Keep uncertainty-head gradients
   out of visual-only losses.

5. Select checkpoints only after fixed-seed validation gates pass: visual RGB
   RMSE within 5% of physical, bounds <= 0.1%, improved optical PSD and edge F1,
   SAR bias <= 0.5 dB, and improved SAR PSD/ENL/histogram.

## Fast validation sequence

Use 64 patches first: overfit the two directions with only `delta_days <= 1`,
then sample fixed seeds and inspect amplitude, PSD, bounds, and spatial detail.
Run an eight-sample fixed validation panel next. Only then launch a full closed
validation and long joint fine-tune.
