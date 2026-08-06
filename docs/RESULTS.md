# V3.1 Experimental Results

Status date: 2026-08-06

## Status

The following checkpoint is the last stable V3.1 visual checkpoint:

```text
checkpoints/visual/step_0040000.pt
```

The physical stage completed at 12,000 steps and the visual stage completed at
40,000 steps. The subsequent 10,000-step balance stage became numerically
unstable and was stopped near step 9,900. In the final balance logs, SAR NLL
reached roughly `3e5-6e5` and total loss roughly `2e4-3e4`; therefore those
balance checkpoints are invalid and are deliberately not reported as results.

## Fixed-Seed Diagnostic

This is a small diagnostic on eight `validation_temporal` samples, not the
463-sample closed validation report and not a test-set claim.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m sentinel_v3.cli \
  --config configs/sentinel_v3.yaml evaluate \
  --checkpoint checkpoints/visual/step_0040000.pt \
  --split validation_temporal --output /tmp/v31_visual_validation8.json \
  --limit 8 --seed 42
```

| Metric | Result |
| --- | ---: |
| SAR-to-optical physical RMSE | 0.1164 |
| SAR-to-optical SAM | 0.6330 |
| Optical-to-SAR mean RMSE (dB) | 6.4489 |
| Optical-to-SAR visual bias (dB) | 0.6134 |
| Physical RGB RMSE | 0.0675 |
| Visual RGB RMSE | 0.1585 |
| Physical / visual edge F1 | 0.3098 / 0.3323 |
| Physical / visual optical PSD distance | 0.0117 / 0.0457 |
| Optical visual out-of-bounds fraction | 0.4146 |
| SAR mean / visual PSD distance | 1.0609 / 0.7379 |
| SAR mean / visual ENL error | 0.8601 / 0.6248 |
| SAR mean / visual histogram distance | 0.0133 / 0.0098 |
| Cross-modal Recall@1 / Recall@5 | 0.125 / 0.625 |

The complete machine-readable diagnostic is in
[validation_temporal_8samples.json](results/validation_temporal_8samples.json).

## Interpretation

The visual SAR branch improves SAR PSD, ENL, and histogram statistics relative
to its deterministic mean. The optical random residual is not acceptable yet:
it improves Edge F1 in this small diagnostic, but it worsens RGB RMSE and
optical PSD distance and violates the `0.1%` out-of-bounds safety gate by a
large margin. Perceptual LPIPS/DISTS values were unavailable in this run, so
no perceptual-quality claim is made.

The panels below are included to make the failure mode inspectable. They show
the deterministic physical output, three fixed-seed visual samples, reference,
error, uncertainty, and edge representation. The high-frequency optical samples
contain excessive noise and should not be presented as a successful result.

![SAR-to-optical diagnostic](assets/validation_000_sar2opt.png)

![Optical-to-SAR diagnostic](assets/validation_000_opt2sar.png)

## Next Work

1. Diagnose and stabilize the balance-stage joint objective before resuming it.
2. Constrain or recalibrate the optical residual before measuring visual quality.
3. Run the full closed validation set, then the temporal, spatial, and joint
   test sets only from a selected stable checkpoint.
4. Compare against V1 Mean, V1 DiT, and V2 Refiner under the same closed splits.

Until those checks pass, V3.1 is a technical prototype and not evidence for a
paper or deployment claim.
