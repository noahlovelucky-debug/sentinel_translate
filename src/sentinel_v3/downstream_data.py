"""Leakage-safe data and physical-cache utilities for the SCL proxy benchmark.

This module deliberately does not reuse :mod:`sentinel_v3.evaluation` datasets.
Those datasets form a joint Sentinel-1/Sentinel-2 validity mask, which is correct
for translation evaluation but would leak Optical/SCL information into a SAR-only
downstream experiment.  The cache path below opens only SAR rasters, constructs a
SAR raw-valid mask, disables the checkpoint temporal prior, and calls
``model.physical`` directly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .evaluation import load_checkpoint
from .schema import S2_CHANNEL_ORDER, SAR_CHANNEL_ORDER
from .sensors import SENTINEL1, SENTINEL2

CACHE_FORMAT_VERSION = 1
MATERIALIZED_CACHE_FORMAT_VERSION = 1
PROTOCOL_VERSION = "downstream-scl-proxy-v1"
IGNORE_INDEX = -1
VEGETATION_LABEL = 1
NONVEGETATION_LABEL = 0
ALLOWED_SPLITS = frozenset(("train", "unused_spatial"))
CLOSED_SPLITS = frozenset(
    ("validation_temporal", "test_spatial", "test_temporal", "test_joint")
)


def file_sha256(path: str | Path) -> str:
    """Return a stable file digest without loading the complete file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(values: Tensor) -> str:
    array = values.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def assert_allowed_split(split: str) -> None:
    """Reject all translation validation and closed-test splits for this benchmark."""

    if split in CLOSED_SPLITS:
        raise ValueError(f"downstream benchmark must not read closed split: {split}")
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"unsupported downstream split: {split}; allowed={sorted(ALLOWED_SPLITS)}"
        )


def scl_proxy_labels(scl: np.ndarray) -> np.ndarray:
    """Map SCL to the preregistered binary proxy target.

    Sentinel-2 SCL is a pseudo-label rather than independent land-cover truth.
    Code 4 maps to vegetation, codes 5/6 map to non-vegetation, and every other
    value is ignored.  In particular, class 2 remains ignored despite being used
    by the translation valid-mask protocol.
    """

    labels = np.full(scl.shape, IGNORE_INDEX, dtype=np.int64)
    labels[scl == 4] = VEGETATION_LABEL
    labels[np.isin(scl, (5, 6))] = NONVEGETATION_LABEL
    return labels


@dataclass(frozen=True)
class CachePlan:
    """Immutable inputs that bind every synthetic cache to one protocol."""

    manifest: Path
    train_shards: Path
    checkpoint: Path
    checkpoint_sha256: str
    cache_root: Path
    config_path: Path
    config_sha256: str
    crop_size: int = 256

    def __post_init__(self) -> None:
        if self.crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if len(self.checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a SHA-256 hex digest")
        if len(self.config_sha256) != 64:
            raise ValueError("config_sha256 must be a SHA-256 hex digest")


@dataclass(frozen=True)
class CropSample:
    """One deterministic crop with the raw manifest record needed to read SAR."""

    sample_id: str
    partition: str
    pair_id: str
    tile: str
    s1_date: str
    s2_date: str
    orbit: str
    gsd: float
    window: tuple[int, int, int, int]
    record: dict[str, Any]

    def index_value(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "partition": self.partition,
            "pair_id": self.pair_id,
            "tile": self.tile,
            "s1_date": self.s1_date,
            "s2_date": self.s2_date,
            "orbit": self.orbit,
            "gsd": self.gsd,
            "window": list(self.window),
            "record": self.record,
        }

    @classmethod
    def from_index_value(cls, value: Mapping[str, object]) -> CropSample:
        window = tuple(int(item) for item in value["window"])  # type: ignore[index]
        if len(window) != 4:
            raise ValueError("crop window must contain four values")
        return cls(
            sample_id=str(value["sample_id"]),
            partition=str(value["partition"]),
            pair_id=str(value["pair_id"]),
            tile=str(value["tile"]),
            s1_date=str(value["s1_date"]),
            s2_date=str(value["s2_date"]),
            orbit=str(value["orbit"]),
            gsd=float(value["gsd"]),
            window=window,
            record=dict(value["record"]),  # type: ignore[arg-type,index]
        )


def _sample_id(pair_id: str, partition: str, window: Sequence[int]) -> str:
    serialized = ":".join(str(int(value)) for value in window)
    return f"{partition}:{pair_id}:{serialized}"


def _crop_sample(
    record: Mapping[str, Any], partition: str, window: tuple[int, int, int, int]
) -> CropSample:
    pair_id = str(record["pair_id"])
    return CropSample(
        sample_id=_sample_id(pair_id, partition, window),
        partition=partition,
        pair_id=pair_id,
        tile=str(record["tile"]),
        s1_date=str(record["s1_date"]),
        s2_date=str(record["s2_date"]),
        orbit=str(record.get("orbit", "unknown")),
        gsd=float(record.get("gsd", 10.0)),
        window=window,
        record=dict(record),
    )


def _validate_indexed_sample(sample: CropSample) -> None:
    """Reject a prepared index that was not derived from the allowed manifest slices."""

    assert_allowed_split(sample.partition)
    if str(sample.record.get("split")) != sample.partition:
        raise RuntimeError("downstream sample partition does not match its manifest record")
    if int(sample.record.get("delta_days", -1)) != 0:
        raise RuntimeError("downstream sample is not a delta_days=0 record")
    if sample.sample_id != _sample_id(sample.pair_id, sample.partition, sample.window):
        raise RuntimeError("downstream sample ID does not bind its partition, pair, and window")
    col, row, width, height = sample.window
    if col < 0 or row < 0 or width <= 0 or height <= 0:
        raise RuntimeError("downstream sample has an invalid crop window")


def _validate_indexed_samples(samples: Sequence[CropSample]) -> None:
    if not samples:
        raise ValueError("downstream cache must contain at least one sample")
    sample_ids: set[str] = set()
    for sample in samples:
        _validate_indexed_sample(sample)
        if sample.sample_id in sample_ids:
            raise RuntimeError(
                f"downstream cache index has duplicate sample ID: {sample.sample_id}"
            )
        sample_ids.add(sample.sample_id)


def load_manifest_records(manifest: str | Path, split: str) -> list[dict[str, Any]]:
    """Load only an explicitly permitted, same-day manifest split."""

    assert_allowed_split(split)
    records: list[dict[str, Any]] = []
    with Path(manifest).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("split")) != split:
                continue
            if int(record.get("delta_days", -1)) != 0:
                continue
            records.append(record)
    records.sort(key=lambda item: str(item["pair_id"]))
    if not records:
        raise ValueError(f"no delta_days=0 records in downstream split {split}")
    return records


def _load_shard_windows(
    path: str | Path, pair_id: str, crop_size: int
) -> list[tuple[int, int, int, int]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored_ids = [str(value) for value in payload.get("pair_id", ())]
    if stored_ids != [pair_id] * 16:
        raise RuntimeError(
            f"train shard does not contain exactly sixteen windows for {pair_id}"
        )
    windows = payload.get("window")
    if not isinstance(windows, Tensor) or tuple(windows.shape) != (16, 4):
        raise RuntimeError(f"train shard has invalid windows for {pair_id}")
    resolved = [tuple(int(value) for value in row.tolist()) for row in windows]
    if any(window[2:] != (crop_size, crop_size) for window in resolved):
        raise RuntimeError(f"train shard window size differs from protocol for {pair_id}")
    return resolved


def train_fixed_window_samples(plan: CachePlan) -> list[CropSample]:
    """Return the canonical train shard's fixed sixteen crops for every same-day pair."""

    records = {
        record["pair_id"]: record for record in load_manifest_records(plan.manifest, "train")
    }
    index = json.loads(plan.train_shards.read_text(encoding="utf-8"))
    if str(index.get("split")) != "train":
        raise RuntimeError("downstream train shard index must have split=train")
    indexed_manifest_sha = index.get("manifest_sha256")
    if indexed_manifest_sha != file_sha256(plan.manifest):
        raise RuntimeError("downstream train shard index does not match the manifest")

    samples: list[CropSample] = []
    for descriptor in sorted(index.get("shards", ()), key=lambda item: str(item["pair_id"])):
        pair_id = str(descriptor["pair_id"])
        record = records.get(pair_id)
        if record is None:
            continue
        if int(descriptor.get("delta_days", record["delta_days"])) != 0:
            continue
        windows = _load_shard_windows(descriptor["path"], pair_id, plan.crop_size)
        samples.extend(_crop_sample(record, "train", window) for window in windows)
    if not samples:
        raise ValueError("no same-day fixed train shard samples are available")
    return samples


def heldout_center_samples(plan: CachePlan) -> list[CropSample]:
    """Use exactly one center crop per real, spatially held-out same-day pair."""

    samples: list[CropSample] = []
    for record in load_manifest_records(plan.manifest, "unused_spatial"):
        width = int(record["width"])
        height = int(record["height"])
        if width < plan.crop_size or height < plan.crop_size:
            raise RuntimeError(f"held-out raster smaller than crop: {record['pair_id']}")
        window = (
            (width - plan.crop_size) // 2,
            (height - plan.crop_size) // 2,
            plan.crop_size,
            plan.crop_size,
        )
        samples.append(_crop_sample(record, "unused_spatial", window))
    return samples


def benchmark_samples(plan: CachePlan) -> list[CropSample]:
    """Return the only two permitted benchmark partitions in deterministic order."""

    return [*train_fixed_window_samples(plan), *heldout_center_samples(plan)]


def rank_shard(samples: Sequence[CropSample], rank: int, world_size: int) -> list[CropSample]:
    """Deterministically assign disjoint cache samples to one distributed rank."""

    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    return [sample for index, sample in enumerate(samples) if index % world_size == rank]


def read_sar_raw_valid(
    record: Mapping[str, Any], window: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Read only SAR inputs and derive validity from raw SAR values.

    This function intentionally never opens S2 or SCL assets.  Invalid SAR pixels
    are set to zero only after the SAR-only validity mask is derived.
    """

    import rasterio
    from rasterio.windows import Window

    raster_window = Window(*window)
    encoded = []
    sar_paths = record["sar"]
    for polarization in SAR_CHANNEL_ORDER:
        with rasterio.open(sar_paths[polarization]) as source:
            encoded.append(source.read(1, window=raster_window))
    raw = np.stack(encoded).astype(np.float32)
    valid = np.all(raw > 0, axis=0)
    values_db = raw / 200.0 - 50.0
    values_db[:, ~valid] = 0.0
    return values_db, valid


def read_real_optical(
    record: Mapping[str, Any], window: tuple[int, int, int, int]
) -> np.ndarray:
    """Read real ten-band reflectance for a downstream input after generation.

    This is intentionally separate from :func:`read_sar_raw_valid` and never
    participates in physical cache generation.  Nodata remains zero after the
    reflectance conversion; no SCL, S2-valid, or joint-valid mask is applied.
    """

    import rasterio
    from rasterio.windows import Window

    raster_window = Window(*window)
    encoded = []
    optical_paths = record["s2"]
    for channel in S2_CHANNEL_ORDER:
        with rasterio.open(optical_paths[channel]) as source:
            encoded.append(source.read(1, window=raster_window))
    raw = np.stack(encoded).astype(np.float32)
    return np.clip(raw / 10000.0, 0.0, 1.0)


def read_scl_proxy_label(
    record: Mapping[str, Any], window: tuple[int, int, int, int]
) -> np.ndarray:
    """Read the proxy target separately from all generator inputs."""

    import rasterio
    from rasterio.windows import Window

    with rasterio.open(record["scl"]) as source:
        scl = source.read(1, window=Window(*window))
    return scl_proxy_labels(scl)


def same_day_sar_metadata(sample: CropSample, device: torch.device) -> Tensor:
    """Build physical metadata using S1 acquisition time for both source and target."""

    acquired = date.fromisoformat(sample.s1_date)
    phase = 2.0 * math.pi * acquired.timetuple().tm_yday / 366.0
    orbit = {"ascending": -1.0, "descending": 1.0, "unknown": 0.0}.get(sample.orbit, 0.0)
    return torch.tensor(
        [
            [
                0.0,
                orbit,
                math.sin(phase),
                math.cos(phase),
                math.sin(phase),
                math.cos(phase),
                math.log(max(sample.gsd, 0.1)) / 4.0,
                1.0,
            ]
        ],
        device=device,
        dtype=torch.float32,
    )


def load_frozen_physical_model(
    checkpoint: str | Path,
    expected_sha256: str,
    device: torch.device,
    *,
    loader: Callable[..., Any] = load_checkpoint,
) -> Any:
    """Load EMA weights and make temporal Optical memory impossible to use."""

    actual_sha256 = file_sha256(checkpoint)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )
    model = loader(checkpoint, device, use_ema=True)
    model.configure_temporal_prior(None)
    if getattr(model, "temporal_prior", None) is not None:
        raise RuntimeError("downstream synthetic cache requires temporal prior to be disabled")
    return model.eval()


def generate_physical_optical(
    model: Any,
    sample: CropSample,
    sar_db: np.ndarray,
    sar_valid: np.ndarray,
    device: torch.device,
) -> Tensor:
    """Generate one physical 10-band Optical crop from SAR-only inputs."""

    if getattr(model, "temporal_prior", None) is not None:
        raise RuntimeError("temporal prior must be disabled before downstream cache generation")
    if sar_db.shape != (2, sample.window[3], sample.window[2]):
        raise ValueError("SAR shape does not match the sample window")
    if sar_valid.shape != sar_db.shape[1:]:
        raise ValueError("SAR validity mask shape does not match SAR values")
    values = torch.from_numpy(sar_db).unsqueeze(0).to(device=device, dtype=torch.float32)
    valid = torch.from_numpy(sar_valid.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    metadata = same_day_sar_metadata(sample, device)
    with torch.inference_mode():
        physical, _, _ = model.physical(
            values,
            SENTINEL1,
            SENTINEL2,
            valid,
            input_gsd=sample.gsd,
            target_gsd=sample.gsd,
            metadata=metadata,
        )
    output = physical.squeeze(0).detach().float().cpu()
    expected_shape = (len(S2_CHANNEL_ORDER), sample.window[3], sample.window[2])
    if tuple(output.shape) != expected_shape or not torch.isfinite(output).all():
        raise RuntimeError("physical checkpoint returned an invalid synthetic Optical crop")
    if float(output.min()) < -1e-5 or float(output.max()) > 1.00001:
        raise RuntimeError("physical synthetic Optical is outside reflectance [0, 1]")
    return output


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sample_cache_path(cache_root: str | Path, sample: CropSample) -> Path:
    digest = hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()
    return Path(cache_root) / "samples" / f"{digest}.pt"


def cache_provenance(plan: CachePlan, samples: Sequence[CropSample]) -> dict[str, object]:
    """Return auditable cache provenance before any generator is loaded."""

    index_hash = canonical_json_sha256([sample.index_value() for sample in samples])
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "manifest": str(plan.manifest.resolve()),
        "manifest_sha256": file_sha256(plan.manifest),
        "train_shards": str(plan.train_shards.resolve()),
        "train_shards_sha256": file_sha256(plan.train_shards),
        "checkpoint": str(plan.checkpoint.resolve()),
        "checkpoint_sha256": plan.checkpoint_sha256,
        "config": str(plan.config_path.resolve()),
        "config_sha256": plan.config_sha256,
        "crop_size": plan.crop_size,
        "samples": len(samples),
        "sample_index_sha256": index_hash,
        "generator": {
            "entrypoint": "model.physical",
            "use_ema": True,
            "mode": "physical",
            "temporal_prior": "disabled",
            "valid_mask": "sar_raw_valid_only",
            "target_metadata_date": "s1_date",
            "output_channels": list(S2_CHANNEL_ORDER),
            "output_dtype": "float16",
        },
    }


def prepare_cache(plan: CachePlan, samples: Sequence[CropSample]) -> dict[str, object]:
    """Atomically bind a cache root to one immutable sample index and provenance."""

    _validate_indexed_samples(samples)
    provenance = cache_provenance(plan, samples)
    provenance_path = plan.cache_root / "provenance.json"
    index_path = plan.cache_root / "sample_index.json"
    if provenance_path.exists():
        current = json.loads(provenance_path.read_text(encoding="utf-8"))
        if current != provenance:
            raise RuntimeError("existing downstream cache provenance does not match this run")
    else:
        _atomic_json(provenance_path, provenance)
    index = {
        "format_version": CACHE_FORMAT_VERSION,
        "provenance_sha256": canonical_json_sha256(provenance),
        "samples": [sample.index_value() for sample in samples],
    }
    if index_path.exists():
        current_index = json.loads(index_path.read_text(encoding="utf-8"))
        if current_index != index:
            raise RuntimeError("existing downstream cache sample index does not match this run")
    else:
        _atomic_json(index_path, index)
    return provenance


def load_prepared_samples(cache_root: str | Path) -> tuple[dict[str, object], list[CropSample]]:
    root = Path(cache_root)
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    index = json.loads((root / "sample_index.json").read_text(encoding="utf-8"))
    if index.get("provenance_sha256") != canonical_json_sha256(provenance):
        raise RuntimeError("cache sample index does not bind to provenance")
    samples = [CropSample.from_index_value(value) for value in index.get("samples", ())]
    _validate_indexed_samples(samples)
    return provenance, samples


def _cache_payload_is_valid(path: Path, sample: CropSample, provenance_sha256: str) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError):
        return False
    values = payload.get("synthetic_s2")
    if not isinstance(values, Tensor):
        return False
    expected_shape = (len(S2_CHANNEL_ORDER), sample.window[3], sample.window[2])
    return bool(
        payload.get("format_version") == CACHE_FORMAT_VERSION
        and payload.get("sample_id") == sample.sample_id
        and payload.get("provenance_sha256") == provenance_sha256
        and values.dtype == torch.float16
        and tuple(values.shape) == expected_shape
        and torch.isfinite(values).all()
        and payload.get("tensor_sha256") == tensor_sha256(values)
    )


def cache_rank_samples(
    plan: CachePlan,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    resume: bool = True,
    model_loader: Callable[..., Any] = load_frozen_physical_model,
) -> dict[str, int]:
    """Generate this rank's disjoint float16 physical cache entries.

    ``prepare_cache`` must run first.  Each rank writes only content-addressed
    sample files, so ranks do not share mutable output files.
    """

    provenance, samples = load_prepared_samples(plan.cache_root)
    expected = cache_provenance(plan, samples)
    if provenance != expected:
        raise RuntimeError("prepared cache provenance differs from requested plan")
    provenance_sha256 = canonical_json_sha256(provenance)
    assigned = rank_shard(samples, rank, world_size)
    model = model_loader(plan.checkpoint, plan.checkpoint_sha256, device)
    generated = 0
    reused = 0
    for sample in assigned:
        destination = sample_cache_path(plan.cache_root, sample)
        if (
            destination.is_file()
            and resume
            and _cache_payload_is_valid(destination, sample, provenance_sha256)
        ):
            reused += 1
            continue
        sar_db, sar_valid = read_sar_raw_valid(sample.record, sample.window)
        synthetic = generate_physical_optical(model, sample, sar_db, sar_valid, device).to(
            torch.float16
        )
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "sample_id": sample.sample_id,
            "provenance_sha256": provenance_sha256,
            "synthetic_s2": synthetic,
            "tensor_sha256": tensor_sha256(synthetic),
        }
        _atomic_torch_save(destination, payload)
        if not _cache_payload_is_valid(destination, sample, provenance_sha256):
            raise RuntimeError(f"synthetic cache checksum verification failed: {destination}")
        generated += 1
    return {"assigned": len(assigned), "generated": generated, "reused": reused}


def finalize_cache(plan: CachePlan) -> dict[str, object]:
    """Verify every rank output and write the cache checksum manifest."""

    provenance, samples = load_prepared_samples(plan.cache_root)
    provenance_sha256 = canonical_json_sha256(provenance)
    entries: list[dict[str, str]] = []
    for sample in samples:
        path = sample_cache_path(plan.cache_root, sample)
        if not _cache_payload_is_valid(path, sample, provenance_sha256):
            raise RuntimeError(f"missing or invalid synthetic cache entry: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        entries.append(
            {
                "sample_id": sample.sample_id,
                "path": str(path.resolve()),
                "file_sha256": file_sha256(path),
                "tensor_sha256": str(payload["tensor_sha256"]),
            }
        )
    result = {
        "format_version": CACHE_FORMAT_VERSION,
        "provenance_sha256": provenance_sha256,
        "entries": entries,
    }
    _atomic_json(plan.cache_root / "cache_manifest.json", result)
    return result


def _load_verified_synthetic(
    cache_root: str | Path, sample: CropSample, provenance_sha256: str
) -> Tensor:
    path = sample_cache_path(cache_root, sample)
    if not _cache_payload_is_valid(path, sample, provenance_sha256):
        raise RuntimeError(f"invalid downstream synthetic cache entry: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    values = payload["synthetic_s2"]
    if not isinstance(values, Tensor):  # Defensive; validity check already establishes this.
        raise TypeError(f"synthetic cache tensor is absent: {path}")
    return values


def cached_synthetic(cache_root: str | Path, sample: CropSample) -> Tensor:
    """Load a verified synthetic crop for a downstream consumer."""

    provenance, _ = load_prepared_samples(cache_root)
    provenance_sha256 = canonical_json_sha256(provenance)
    return _load_verified_synthetic(cache_root, sample, provenance_sha256).float()


@dataclass(frozen=True)
class _MaterializedRow:
    sample_id: str
    scene_id: str
    tile: str
    split: str
    sar: Tensor
    real_optical: Tensor
    synthetic_optical: Tensor
    label: Tensor
    sar_valid: Tensor


def _load_finalized_cache(
    plan: CachePlan,
) -> tuple[dict[str, object], str, list[CropSample], dict[str, object]]:
    """Require a complete, checksum-verified physical cache before materializing."""

    provenance, samples = load_prepared_samples(plan.cache_root)
    expected_provenance = cache_provenance(plan, samples)
    if provenance != expected_provenance:
        raise RuntimeError(
            "prepared cache provenance differs from requested materialization plan"
        )
    provenance_sha256 = canonical_json_sha256(provenance)
    manifest_path = plan.cache_root / "cache_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("run finalize_cache before materializing downstream probe inputs")
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        cache_manifest.get("format_version") != CACHE_FORMAT_VERSION
        or cache_manifest.get("provenance_sha256") != provenance_sha256
    ):
        raise RuntimeError("synthetic cache manifest does not match the prepared cache")
    raw_entries = cache_manifest.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != len(samples):
        raise RuntimeError("synthetic cache manifest has an invalid entry count")
    entries: dict[str, Mapping[str, object]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("synthetic cache manifest has a malformed entry")
        sample_id = str(raw_entry.get("sample_id"))
        if sample_id in entries:
            raise RuntimeError(f"synthetic cache manifest repeats sample ID: {sample_id}")
        entries[sample_id] = raw_entry
    for sample in samples:
        entry = entries.get(sample.sample_id)
        path = sample_cache_path(plan.cache_root, sample)
        if entry is None or str(entry.get("path")) != str(path.resolve()):
            raise RuntimeError(
                f"synthetic cache manifest does not bind sample: {sample.sample_id}"
            )
        if str(entry.get("file_sha256")) != file_sha256(path):
            raise RuntimeError(f"synthetic cache manifest checksum mismatch: {path}")
        values = _load_verified_synthetic(plan.cache_root, sample, provenance_sha256)
        if str(entry.get("tensor_sha256")) != tensor_sha256(values):
            raise RuntimeError(f"synthetic cache tensor checksum mismatch: {path}")
    return provenance, provenance_sha256, samples, cache_manifest


def _materialized_root(plan: CachePlan) -> Path:
    return plan.cache_root / "materialized"


def _materialized_chunk_path(root: Path, index: int) -> Path:
    return root / "chunks" / f"chunk-{index:05d}.pt"


def _materialized_provenance(
    plan: CachePlan,
    samples: Sequence[CropSample],
    cache_provenance_sha256: str,
    cache_manifest_sha256: str,
    dev_tiles: Sequence[str],
    chunk_size: int,
) -> dict[str, object]:
    from .downstream_probe import cache_contract

    return {
        "format_version": MATERIALIZED_CACHE_FORMAT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "source_cache_root": str(plan.cache_root.resolve()),
        "source_cache_provenance_sha256": cache_provenance_sha256,
        "source_cache_manifest_sha256": cache_manifest_sha256,
        "source_config_sha256": plan.config_sha256,
        "source_checkpoint_sha256": plan.checkpoint_sha256,
        "sample_index_sha256": canonical_json_sha256(
            [sample.index_value() for sample in samples]
        ),
        "samples": len(samples),
        "chunk_size": chunk_size,
        "dev_tiles": list(dev_tiles),
        "splits": {
            "train": "canonical train samples whose tile is not in dev_tiles",
            "dev": "canonical train samples whose tile is in dev_tiles",
            "test": "all canonical unused_spatial samples",
        },
        "inputs": {
            "sar": "float16 raw-SAR-decibel values with SAR-only invalid pixels zeroed",
            "real_optical": "float16 raw S2 reflectance clipped to [0, 1]",
            "synthetic_optical": "float16 verified physical cache output",
            "sar_valid": "bool SAR raw-valid mask",
            "label": "int8 SCL proxy: 4=>1, 5/6=>0, other=>-1",
        },
        "probe_cache_contract": cache_contract(),
    }


def _write_or_validate_materialized_provenance(
    root: Path, provenance: Mapping[str, object]
) -> str:
    path = root / "provenance.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != provenance:
            raise RuntimeError("existing materialized cache provenance does not match this run")
    else:
        _atomic_json(path, provenance)
    return canonical_json_sha256(provenance)


def _probe_split(sample: CropSample, dev_tiles: frozenset[str]) -> str:
    if sample.partition == "unused_spatial":
        return "test"
    if sample.partition != "train":  # _validate_indexed_samples makes this defensive.
        raise RuntimeError(f"unsupported materialized sample partition: {sample.partition}")
    return "dev" if sample.tile in dev_tiles else "train"


def _materialize_row(
    sample: CropSample,
    *,
    cache_root: Path,
    cache_provenance_sha256: str,
    dev_tiles: frozenset[str],
) -> _MaterializedRow:
    """Join an already-verified synthetic crop with independently read real inputs."""

    sar_values, sar_valid = read_sar_raw_valid(sample.record, sample.window)
    real_optical = read_real_optical(sample.record, sample.window)
    label = read_scl_proxy_label(sample.record, sample.window)
    synthetic_optical = _load_verified_synthetic(cache_root, sample, cache_provenance_sha256)
    height, width = sample.window[3], sample.window[2]
    if tuple(sar_values.shape) != (len(SAR_CHANNEL_ORDER), height, width):
        raise RuntimeError(f"materialized SAR shape does not match sample: {sample.sample_id}")
    if tuple(real_optical.shape) != (len(S2_CHANNEL_ORDER), height, width):
        raise RuntimeError(f"materialized S2 shape does not match sample: {sample.sample_id}")
    if tuple(label.shape) != (height, width):
        raise RuntimeError(f"materialized SCL shape does not match sample: {sample.sample_id}")
    if tuple(synthetic_optical.shape) != (len(S2_CHANNEL_ORDER), height, width):
        raise RuntimeError(
            f"materialized synthetic shape does not match sample: {sample.sample_id}"
        )
    if sar_valid.shape != (height, width):
        raise RuntimeError(
            f"materialized SAR valid shape does not match sample: {sample.sample_id}"
        )
    if not np.isfinite(sar_values).all() or not np.isfinite(real_optical).all():
        raise RuntimeError(
            f"materialized real input contains non-finite values: {sample.sample_id}"
        )
    if not np.isin(label, (IGNORE_INDEX, NONVEGETATION_LABEL, VEGETATION_LABEL)).all():
        raise RuntimeError(f"materialized label has invalid proxy codes: {sample.sample_id}")
    return _MaterializedRow(
        sample_id=sample.sample_id,
        scene_id=f"{sample.tile}:{sample.s2_date}",
        tile=sample.tile,
        split=_probe_split(sample, dev_tiles),
        sar=torch.from_numpy(np.ascontiguousarray(sar_values)).to(torch.float16),
        real_optical=torch.from_numpy(np.ascontiguousarray(real_optical)).to(torch.float16),
        synthetic_optical=synthetic_optical.to(torch.float16).contiguous(),
        label=torch.from_numpy(np.ascontiguousarray(label.astype(np.int8, copy=False))),
        sar_valid=torch.from_numpy(np.ascontiguousarray(sar_valid.astype(np.bool_))).unsqueeze(
            0
        ),
    )


def _materialized_chunk_payload(
    rows: Sequence[_MaterializedRow], materialized_provenance_sha256: str
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot write an empty materialized probe chunk")
    return {
        "format_version": MATERIALIZED_CACHE_FORMAT_VERSION,
        "materialized_provenance_sha256": materialized_provenance_sha256,
        "sample_id": [row.sample_id for row in rows],
        "scene_id": [row.scene_id for row in rows],
        "tile": [row.tile for row in rows],
        "split": [row.split for row in rows],
        "sar": torch.stack([row.sar for row in rows]).to(torch.float16).contiguous(),
        "real_optical": torch.stack([row.real_optical for row in rows])
        .to(torch.float16)
        .contiguous(),
        "synthetic_optical": torch.stack([row.synthetic_optical for row in rows])
        .to(torch.float16)
        .contiguous(),
        "label": torch.stack([row.label for row in rows]).to(torch.int8).contiguous(),
        "sar_valid": torch.stack([row.sar_valid for row in rows]).bool().contiguous(),
    }


def _validate_materialized_chunk(
    path: Path,
    *,
    sample_ids: Sequence[str],
    materialized_provenance_sha256: str,
) -> bool:
    """Validate both on-disk dtypes and the downstream probe's public contract."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError):
        return False
    if not isinstance(payload, Mapping):
        return False
    if (
        payload.get("format_version") != MATERIALIZED_CACHE_FORMAT_VERSION
        or payload.get("materialized_provenance_sha256") != materialized_provenance_sha256
        or payload.get("sample_id") != list(sample_ids)
    ):
        return False
    tensor_dtypes = {
        "sar": torch.float16,
        "real_optical": torch.float16,
        "synthetic_optical": torch.float16,
        "label": torch.int8,
        "sar_valid": torch.bool,
    }
    if any(
        not isinstance(payload.get(name), Tensor) or payload[name].dtype != dtype
        for name, dtype in tensor_dtypes.items()
    ):
        return False
    try:
        from .downstream_probe import ProbeCache

        cache = ProbeCache.from_mapping(payload)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return False
    return cache.sample_id == tuple(sample_ids)


def _existing_materialized_manifest(
    root: Path,
    *,
    materialized_provenance_sha256: str,
    layouts: Sequence[Sequence[CropSample]],
) -> dict[str, object] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    entries = manifest.get("entries")
    if (
        manifest.get("format_version") != MATERIALIZED_CACHE_FORMAT_VERSION
        or manifest.get("materialized_provenance_sha256") != materialized_provenance_sha256
        or not isinstance(entries, list)
        or len(entries) != len(layouts)
    ):
        return None
    for index, (entry, layout) in enumerate(zip(entries, layouts, strict=True)):
        if not isinstance(entry, Mapping):
            return None
        path = _materialized_chunk_path(root, index)
        sample_ids = [sample.sample_id for sample in layout]
        if not path.is_file():
            return None
        if (
            str(entry.get("path")) != str(path.resolve())
            or str(entry.get("file_sha256")) != file_sha256(path)
            or str(entry.get("sample_ids_sha256")) != canonical_json_sha256(sample_ids)
            or not _validate_materialized_chunk(
                path,
                sample_ids=sample_ids,
                materialized_provenance_sha256=materialized_provenance_sha256,
            )
        ):
            return None
    return manifest


def materialize_probe_cache(
    plan: CachePlan,
    dev_tiles: Sequence[str],
    *,
    chunk_size: int = 32,
) -> dict[str, object]:
    """Write verified, chunked probe inputs after physical cache finalization.

    Generation and materialization are deliberately separate phases.  The former
    receives SAR-only values and a SAR-only valid mask.  This function first
    verifies every finalized synthetic artifact, then independently reads real
    SAR, real S2, and SCL for the downstream-only probe payload.
    """

    if isinstance(dev_tiles, str) or not dev_tiles:
        raise ValueError("materialization requires a non-empty fixed dev_tiles sequence")
    if chunk_size <= 0:
        raise ValueError("materialized chunk_size must be positive")
    registered_dev_tiles = tuple(str(tile) for tile in dev_tiles)
    if len(set(registered_dev_tiles)) != len(registered_dev_tiles):
        raise ValueError("materialization dev_tiles must not contain duplicates")
    provenance, cache_provenance_sha256, samples, _ = _load_finalized_cache(plan)
    del provenance
    train_tiles = {sample.tile for sample in samples if sample.partition == "train"}
    unknown_dev_tiles = sorted(set(registered_dev_tiles).difference(train_tiles))
    if unknown_dev_tiles:
        raise ValueError(
            f"materialization dev_tiles are not canonical train tiles: {unknown_dev_tiles}"
        )
    layouts = [
        samples[start : start + chunk_size] for start in range(0, len(samples), chunk_size)
    ]
    root = _materialized_root(plan)
    cache_manifest_sha256 = file_sha256(plan.cache_root / "cache_manifest.json")
    materialized_provenance = _materialized_provenance(
        plan,
        samples,
        cache_provenance_sha256,
        cache_manifest_sha256,
        registered_dev_tiles,
        chunk_size,
    )
    materialized_provenance_sha256 = _write_or_validate_materialized_provenance(
        root, materialized_provenance
    )
    existing = _existing_materialized_manifest(
        root,
        materialized_provenance_sha256=materialized_provenance_sha256,
        layouts=layouts,
    )
    if existing is not None:
        return existing

    dev_tile_set = frozenset(registered_dev_tiles)
    entries: list[dict[str, object]] = []
    for index, layout in enumerate(layouts):
        rows = [
            _materialize_row(
                sample,
                cache_root=plan.cache_root,
                cache_provenance_sha256=cache_provenance_sha256,
                dev_tiles=dev_tile_set,
            )
            for sample in layout
        ]
        payload = _materialized_chunk_payload(rows, materialized_provenance_sha256)
        path = _materialized_chunk_path(root, index)
        _atomic_torch_save(path, payload)
        sample_ids = [sample.sample_id for sample in layout]
        if not _validate_materialized_chunk(
            path,
            sample_ids=sample_ids,
            materialized_provenance_sha256=materialized_provenance_sha256,
        ):
            raise RuntimeError(f"materialized probe chunk failed contract verification: {path}")
        entries.append(
            {
                "index": index,
                "path": str(path.resolve()),
                "samples": len(layout),
                "sample_ids_sha256": canonical_json_sha256(sample_ids),
                "file_sha256": file_sha256(path),
            }
        )
    result = {
        "format_version": MATERIALIZED_CACHE_FORMAT_VERSION,
        "materialized_provenance_sha256": materialized_provenance_sha256,
        "source_cache_provenance_sha256": cache_provenance_sha256,
        "entries": entries,
    }
    _atomic_json(root / "manifest.json", result)
    return result
