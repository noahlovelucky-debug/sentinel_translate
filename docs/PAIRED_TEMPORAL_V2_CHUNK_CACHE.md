# Paired Temporal V2 Acquisition Chunk Cache

The second local-data stage replaces repeated NFS TIFF window reads with one
normalized, acquisition-deduplicated local cache.  It is a training data
format, not a model checkpoint and not a new split protocol.

## Scope

The plan is built only from the `train` and `validation_temporal` indexes of
`configs/paired_temporal_v2_full.yaml`, on the ascending orbit.  It constructs
both directions independently while deduplicating the underlying acquisitions:

- SAR to Optical: source history and query Optical label;
- Optical to SAR: source history and query SAR label.

No test, buffer, or unused-spatial record can enter the plan.  Index selection
runs before TIFF decoding and retains the existing causal checks.  A query
target can be cached because it is a label, but the route table rejects any
sample that would use that same cached acquisition as an anchor or observation
input.

## Layout

```text
sentinel_translate_paired_v2_chunks/
  plan.json
  provenance.json
  routing.json
  indexes/<direction>/<split>.jsonl
  acquisitions/
    optical/<content-id>/values.npy
    optical/<content-id>/valid.npy
    optical/<content-id>/chunk.json
    sar/<content-id>/values.npy
    sar/<content-id>/valid.npy
    sar/<content-id>/chunk.json
  cache_index.json
```

`cache_index.json` is the completion marker and is written only after all
chunks verify.  Consumers require it; an interrupted cache is not usable.
Each per-acquisition `chunk.json` records the `values.npy` and `valid.npy`
sizes, dtypes, shapes, and SHA-256 hashes.  Partial writes use a sibling
temporary directory and `os.replace`; corrupt or incomplete chunks are rebuilt
on resume.

The value arrays use the exact V3 paired-raster units:

- Optical `[W, 10, 256, 256] float16`: reflectance normalized to `[-1, 1]`.
- SAR `[W, 2, 256, 256] float16`: VV/VH dB normalized to `[-1, 1]`.
- Validity `[W, 1, 256, 256] uint8`: optical SCL clear classes plus positive
  bands, or positive SAR values.

At runtime `PairedTemporalChunkDataset` opens these arrays with
`np.load(..., mmap_mode="r")` and indexes one window before converting it to a
tensor.  It never opens a raw TIFF or deserializes a whole acquisition.

## Windows

Every acquisition on the same tile/grid uses the same fixed 64-window table.
The first entry is the exact central Sentinel window:

```text
(col=1152, row=1152, width=256, height=256)
```

For a 2560-square scene, the remaining 63 entries are selected deterministically
by SHA-256 from 99 non-overlapping cells of the 10x10 256-pixel lattice.  The
center can overlap neighboring lattice cells; the 63 selected lattice windows
remain mutually non-overlapping.  Training uses all windows; validation defaults
to the center window only.  The dataset length is therefore
`causal_samples * selected_windows`.

## Capacity and operation

The default maximum cache budget is 180 GiB and the hard free-space floor is
80 GiB.  The plan calculates normalized array bytes before any source raster is
opened.  It estimates roughly 128.9 GiB of values/masks for the full ascending
train plus validation selection; the precise dry-run report is authoritative.

```bash
cd /data/code/sentinel_translat/v3.2
PYTHONPATH=src python scripts/build_paired_temporal_chunk_cache.py
```

This prints a dry-run report only.  It does not create the cache, decode TIFFs,
copy data, or launch training.  After reviewing the report:

```bash
PYTHONPATH=src python scripts/build_paired_temporal_chunk_cache.py --execute --workers 4
PYTHONPATH=src python scripts/build_paired_temporal_chunk_cache.py --verify
```

`--workers` controls concurrent acquisition conversion.  There is deliberately
no copy-rate option: the cache directly decodes TIFF windows and the requested
default is unrestricted local materialization.  `--no-resume` rebuilds all
chunks; ordinary `--execute` only repairs missing or SHA-invalid chunks.

The cache builder must not be pointed at a test-inclusive configuration.  It
does not start a training process.  A training launcher must explicitly opt in
to `PairedTemporalChunkDataset`; raw `PairedTemporalRasterDataset` behavior is
unchanged.
