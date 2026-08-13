# Paired Temporal V2 Feasibility Cache

Prepare the bounded local feasibility corpus before the pilot run:

```bash
PYTHONPATH=src python scripts/build_paired_temporal_local_cache.py
```

The default command is a dry-run. It builds the fixed four indexes from
`configs/paired_temporal_v2_feasibility.yaml`:

- `sar_to_optical` train and validation, 64 samples each
- `optical_to_sar` train and validation, 64 samples each

Its report includes unique source-file count, logical reference bytes, actual
deduplicated `stat` bytes, allocation bytes, destination budget, and free-space
gate. It refuses materialization above 30 GiB or when the target would retain
less than 80 GiB free by default.

After reviewing an allowed report, copy with resume support:

```bash
PYTHONPATH=src python scripts/build_paired_temporal_local_cache.py --execute
```

The first-stage default budget is 30 GiB, sized above the measured 64-sample
four-index union while retaining the 80 GiB free-space floor. Copies are
unlimited by default (`--rate-limit-mib-per-second 0`). An explicit positive
value enables one serial application-level token bucket shared across asset
boundaries. Dry-runs never copy or wait for the rate limiter.

The cache is written to
`/home/noah/datasets/sentinel_translate_paired_v2_feasibility` unless
`--destination` is supplied. Files are copied through temporary paths and
atomically published after size checks. The final `cache_manifest.json` binds
the local manifest, four indexes, and every copied asset to SHA-256 hashes.

This tool only copies raw files. It does not decode TIFFs and does not start
training.

## Training From The Local Cache

After a successful materialization, use the separate local configuration.  It
does not change the canonical remote feasibility configuration:

```bash
cd /data/code/sentinel_translat/v3.2
CONFIG_PATH=configs/paired_temporal_v2_feasibility_local.yaml \
OUTPUT_ROOT=checkpoints_paired_temporal_v2_feasibility_local \
DIRECTION=sar_to_optical \
./scripts/launch_paired_temporal_v2_pipeline_8gpu.sh
```

Set `DIRECTION=optical_to_sar` for the reverse direction.  The launcher already
accepts `CONFIG_PATH`, so no launcher edit is required.

`data.train_index` and `data.validation_index` are optional paired settings.
When both are absent, the runner retains the existing automatic-index route.
When both are present, they may be either a path containing exactly one
`{direction}` template or a mapping with a path for each direction.  In this
explicit route rank zero only loads and validates the supplied local indexes;
it does not rebuild indexes and therefore does not consult the remote
manifest.  Validation checks the index direction, configured split, expected
count, serialized selection constraints, strict causal/no-leakage rules, and
the existence of every selected local raw asset before DDP workers create a
dataset.

The checkpoint protocol hash includes SHA-256 values for the local manifest,
train index, and validation index.  Editing any of those files makes existing
checkpoint resume incompatible by design.  The raw TIFF contents are covered
by the cache's `cache_manifest.json`; the runner deliberately checks their
existence rather than re-hashing multi-gigabyte image assets at startup.
