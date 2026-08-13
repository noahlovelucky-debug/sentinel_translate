# Temporal V1: Causal Anchor-Delta Transport

## Scope

This is a research track separate from the V3.2 single-frame checkpoint.  It
targets two causal tasks on a common spatial grid:

1. Multi-temporal SAR plus one earlier real Optical anchor produces an Optical query.
2. Multi-temporal Optical plus one earlier real SAR anchor produces a SAR query.

For query time `t`, every source frame has acquisition time `<= t`.  The only
target-modality input is a real anchor strictly before `t`; the query target is
never passed as an input.  The maximum temporal horizon is 180 days.

## Why This Architecture

The old single-frame translator has to infer both slow radiometric state and
fast modality-specific detail from one observation.  The temporal task exposes
an additional fact: a past observation in the target modality is a much better
radiometric reference than a synthesized global mean.

```text
past real target anchor + bounded physical delta = deterministic physical output
                                           + sampled residual bridge = visual output
```

The physical delta has two terms:

```text
delta = observable_source_carrier(source sequence, time attention) + free_residual_head
```

The source carrier projects actual, attention-weighted source measurements to
the target channel descriptors.  It is zero-initialized, so an untrained model
is exactly the real-anchor baseline.  The residual head learns cross-modal
nonlinearities that cannot be explained by the direct carrier.  Bounded signed
composition prevents the Optical normalized output from leaving `[-1, 1]`
without using a final clamp to hide a failure.

The visual branch is a residual latent bridge: it starts from a
physical-conditioned origin and transports only the residual unexplained by
the physical output.  It is not whole-image sampling from noise.

## Sensor And Resolution Path

`DescriptorSensorAdapter` and `DynamicChannelProjection` consume channel
descriptors (`reflectance/backscatter`, wavelength/frequency, polarization,
native/grid GSD, PSF).  The temporal core therefore does not hard-code the
Sentinel 2/10 channel counts.  A new sensor still needs paired calibration and
validation, but later few-shot adaptation can tune the shallow source carrier
and I/O adapters while preserving the causal fusion core.

## Data Contract

`src/sentinel_v3/temporal_data.py` creates a JSONL index from the immutable
pair manifest.  It rejects any sequence that crosses split, tile, year, orbit,
or grid.  It also checks physical asset identities, preventing the query target
TIFF from appearing in an anchor or source input.

For the current ascending-only, same tile/year/split protocol with four source
frames and a 180-day horizon, the existing manifest contains:

| Direction | Train queries | 2023 validation queries |
| --- | ---: | ---: |
| SAR -> Optical | 753 | 56 |
| Optical -> SAR | 832 | 61 |

Raster values retain V3 units: Optical is normalized surface reflectance and
SAR is per-polarization normalized dB, both in `[-1, 1]`.  SCL controls Optical
validity; each sample uses a deterministic aligned crop and requires a common
valid fraction.

## Go/No-Go Pilot

Run separately for each direction:

```bash
PYTHONPATH=src python scripts/run_temporal_v1_feasibility.py \
  --direction sar_to_optical \
  --output reports_temporal_v1/sar_to_optical \
  --source-frames 4 --crop-size 128 \
  --train-samples 64 --validation-samples 32 --steps 400
```

The runner writes separate train and 2023 validation measurements.  It only
recommends scale-up when validation improves on copying the real anchor by at
least 5% RMSE and blanking the causal source sequence causes at least 1%
degradation.  It writes `temporal_pilot_last.pt`, but this is a new V1 research
checkpoint, never a V3.2 format-v4 checkpoint.

## Current Feasibility Evidence

The controlled local test is deliberately constructed so the real anchor lacks
the target change while the newest causal source frame observes it.  After 400
CPU steps, the held-out anchor RMSE is `0.04498`; Temporal V1 reaches `0.01895`
(57.9% improvement).  Zeroing source frames restores RMSE to `0.04478`.

This establishes that the causal fusion plus observable carrier can learn and
use an identifiable temporal change.  It is not a claim of Sentinel
performance.  The real TIFF pilot remains required before any 8-GPU training
or SOTA claim.

## Current Limitation

On 2026-08-13 the shared `/data` NFS mount entered uninterruptible read waits.
The real TIFF smoke was stopped before GPU computation, and direct
`pytest` reads of the NFS-backed repository could not complete.  Static checks,
the in-memory model test suite, synthetic causal test, and data-layer static
checks completed.  Rerun the real pilot only after the mount returns healthy.
