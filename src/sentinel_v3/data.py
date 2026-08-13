from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .physics import normalized_s2_to_reflectance, normalized_sar_to_db, physical_resample
from .schema import (
    LEGACY_V1_S2_CHANNEL_ORDER,
    S2_CHANNEL_ORDER,
    SAR_CHANNEL_ORDER,
    channel_reorder_indices,
    require_channel_order,
)

REGISTRATION_AUDIT_METHOD = "local_structure_ncc"
REGISTRATION_AUDIT_VERSION = 2
REGISTRATION_SEARCH_RADIUS_PX = 2
REGISTRATION_MIN_NCC = 0.10
REGISTRATION_MIN_IMPROVEMENT = 0.05


@dataclass(frozen=True)
class RegistrationShiftAudit:
    shift_px: Tensor
    zero_ncc: Tensor
    best_ncc: Tensor
    improvement: Tensor
    evidence_supported: bool


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_sidecar_source(path: str | Path, sidecar_path: Path) -> Path:
    source = Path(path)
    return (sidecar_path.parent / source).resolve() if not source.is_absolute() else source.resolve()


def estimate_registration_shift(
    s2: Tensor,
    sar: Tensor,
    *,
    valid: Tensor | None = None,
    maximum_shift_px: int = REGISTRATION_SEARCH_RADIUS_PX,
    minimum_ncc: float = REGISTRATION_MIN_NCC,
    minimum_improvement: float = REGISTRATION_MIN_IMPROVEMENT,
) -> Tensor:
    """Return a nonzero shift only when local cross-modal structure supports it."""

    return registration_shift_audit(
        s2,
        sar,
        valid=valid,
        maximum_shift_px=maximum_shift_px,
        minimum_ncc=minimum_ncc,
        minimum_improvement=minimum_improvement,
    ).shift_px


def registration_shift_audit(
    s2: Tensor,
    sar: Tensor,
    *,
    valid: Tensor | None = None,
    maximum_shift_px: int = REGISTRATION_SEARCH_RADIUS_PX,
    minimum_ncc: float = REGISTRATION_MIN_NCC,
    minimum_improvement: float = REGISTRATION_MIN_IMPROVEMENT,
) -> RegistrationShiftAudit:
    """Estimate displacement and distinguish alignment from absent evidence."""

    if s2.ndim != 3 or sar.ndim != 3 or s2.shape[-2:] != sar.shape[-2:]:
        raise ValueError("s2 and sar must be CxHxW tensors on the same grid")
    if maximum_shift_px < 1 or minimum_ncc < -1.0 or minimum_improvement < 0.0:
        raise ValueError("invalid registration-audit thresholds")
    height, width = s2.shape[-2:]
    if height <= 2 * maximum_shift_px or width <= 2 * maximum_shift_px:
        zero = s2.new_zeros(())
        return RegistrationShiftAudit(zero, zero, zero, zero, False)
    if valid is None:
        joint_valid = torch.ones((height, width), device=s2.device, dtype=torch.bool)
    else:
        if valid.shape not in {(height, width), (1, height, width)}:
            raise ValueError("valid must have shape HxW or 1xHxW")
        joint_valid = valid.reshape(height, width).to(device=s2.device).bool()

    def structure(values: Tensor) -> Tensor:
        gray = values.float().mean(dim=0, keepdim=True).unsqueeze(0)
        smooth = F.avg_pool2d(gray, 3, stride=1, padding=1)
        dx = F.pad((smooth[..., :, 2:] - smooth[..., :, :-2]) * 0.5, (1, 1, 0, 0))
        dy = F.pad((smooth[..., 2:, :] - smooth[..., :-2, :]) * 0.5, (0, 0, 1, 1))
        magnitude = torch.sqrt(dx.square() + dy.square()).squeeze(0).squeeze(0)
        local_mean = F.avg_pool2d(magnitude[None, None], 9, stride=1, padding=4)
        local_energy = F.avg_pool2d(magnitude.square()[None, None], 9, stride=1, padding=4)
        local_std = (local_energy - local_mean.square()).clamp_min(0.0).sqrt().clamp_min(1e-6)
        return torch.nan_to_num(((magnitude[None, None] - local_mean) / local_std).squeeze())

    optical = structure(s2)
    radar = structure(sar)

    def overlap_ncc(dy: int, dx: int) -> Tensor:
        source_y = slice(max(dy, 0), min(height + dy, height))
        target_y = slice(max(-dy, 0), min(height - dy, height))
        source_x = slice(max(dx, 0), min(width + dx, width))
        target_x = slice(max(-dx, 0), min(width - dx, width))
        mask = joint_valid[source_y, source_x] & joint_valid[target_y, target_x]
        if int(mask.sum()) < 16:
            return optical.new_tensor(-1.0)
        source = optical[source_y, source_x][mask]
        target = radar[target_y, target_x][mask]
        source = source - source.mean()
        target = target - target.mean()
        denominator = source.square().sum().sqrt() * target.square().sum().sqrt()
        return torch.nan_to_num((source * target).sum() / denominator.clamp_min(1e-6))

    zero_ncc = overlap_ncc(0, 0)
    best_ncc = optical.new_tensor(-1.0)
    best_shift = (0, 0)
    for dy in range(-maximum_shift_px, maximum_shift_px + 1):
        for dx in range(-maximum_shift_px, maximum_shift_px + 1):
            if dx == 0 and dy == 0:
                continue
            candidate = overlap_ncc(dy, dx)
            if bool(candidate > best_ncc):
                best_ncc = candidate
                best_shift = (dy, dx)
    improvement = best_ncc - zero_ncc
    shifted = bool(best_ncc >= minimum_ncc and improvement >= minimum_improvement)
    aligned = bool(zero_ncc >= minimum_ncc and improvement < minimum_improvement)
    shift = s2.new_tensor(math.hypot(*best_shift)) if shifted else s2.new_zeros(())
    return RegistrationShiftAudit(
        shift_px=shift,
        zero_ncc=zero_ncc,
        best_ncc=best_ncc,
        improvement=improvement,
        evidence_supported=shifted or aligned,
    )


def high_frequency_eligible(
    *,
    delta_days: int | Tensor,
    year: int,
    split: str,
    registration_shift_px: float | Tensor,
    valid_fraction: float | Tensor,
    cloud_shadow_fraction: float | Tensor,
    maximum_shift_px: float = 0.5,
    minimum_valid_fraction: float = 0.8,
    maximum_cloud_shadow_fraction: float = 0.2,
    train_years: Iterable[int] | None = None,
) -> bool:
    eligible_years = frozenset((2017, 2018) if train_years is None else train_years)
    return (
        int(delta_days) <= 1
        and year in eligible_years
        and split == "train"
        and float(registration_shift_px) <= maximum_shift_px
        and float(valid_fraction) >= minimum_valid_fraction
        and float(cloud_shadow_fraction) <= maximum_cloud_shadow_fraction
    )


class V2ShardDataset(Dataset[dict[str, object]]):
    """Read existing shards without copying or rewriting the 61 GB corpus."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        augment: bool = True,
        random_gsd: bool = True,
        native_gsd_probability: float = 0.0,
        audit_high_frequency: bool = False,
        temporal_prior_index: str | Path | None = None,
        hf_years: Iterable[int] | None = None,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        format_version = int(self.index.get("format_version", 1))
        stored_s2_order = self.index.get("s2_channel_order")
        if stored_s2_order is None:
            if format_version >= 2:
                raise RuntimeError("v2 shard indexes must declare s2_channel_order")
            stored_s2_order = LEGACY_V1_S2_CHANNEL_ORDER
        self.s2_channel_order = require_channel_order(stored_s2_order)
        self._s2_reorder = channel_reorder_indices(self.s2_channel_order, S2_CHANNEL_ORDER)
        stored_sar_order = self.index.get("sar_channel_order", SAR_CHANNEL_ORDER)
        self.sar_channel_order = require_channel_order(stored_sar_order, SAR_CHANNEL_ORDER)
        self._sar_reorder = channel_reorder_indices(self.sar_channel_order, SAR_CHANNEL_ORDER)
        default_years = (2017, 2018)
        index_train_years = self.index.get("train_years", default_years)
        index_hf_years = self.index.get("hf_years", index_train_years)
        selected_hf_years = index_hf_years if hf_years is None else hf_years
        self.train_years = tuple(sorted({int(year) for year in index_train_years}))
        self.hf_years = tuple(sorted({int(year) for year in selected_hf_years}))
        if not self.train_years or not self.hf_years:
            raise RuntimeError("shard index train_years and hf_years must be non-empty")
        if not set(self.hf_years) <= set(self.train_years):
            raise RuntimeError("shard index hf_years must be a subset of train_years")
        self.shards = list(self.index["shards"])
        self.prior_shards: list[dict[str, object]] | None = None
        if temporal_prior_index is not None:
            prior_path = Path(temporal_prior_index).resolve()
            prior_index = json.loads(prior_path.read_text(encoding="utf-8"))
            prior_format = int(prior_index.get("format_version", 1))
            if prior_format >= 2:
                source_index = prior_index.get("source_index")
                if source_index is None:
                    raise RuntimeError("v2 temporal-prior index must declare source_index")
                if _resolved_sidecar_source(str(source_index), prior_path) != self.index_path:
                    raise RuntimeError("temporal-prior index belongs to a different training index")
                if prior_index.get("source_index_sha256") != _file_sha256(self.index_path):
                    raise RuntimeError("temporal-prior index source_index_sha256 does not match")
                if tuple(prior_index.get("s2_channel_order", ())) != S2_CHANNEL_ORDER:
                    raise RuntimeError("v2 temporal-prior index must use canonical S2 channel order")
                if tuple(prior_index.get("sar_channel_order", ())) != SAR_CHANNEL_ORDER:
                    raise RuntimeError("v2 temporal-prior index must use canonical SAR channel order")
            self.prior_shards = list(prior_index["shards"])
            if len(self.prior_shards) != len(self.shards):
                raise RuntimeError("temporal-prior and training shard counts differ")
        self.ends: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.ends.append(total)
        self.total = total
        self.augment = augment
        self.random_gsd = random_gsd
        if not 0.0 <= native_gsd_probability <= 1.0:
            raise ValueError("native_gsd_probability must be in [0, 1]")
        self.native_gsd_probability = native_gsd_probability
        self.audit_high_frequency = audit_high_frequency
        self.split = str(self.index.get("split", "unknown"))
        self._cache_index = -1
        self._cache: dict[str, object] | None = None
        self._prior_cache: dict[str, object] | None = None

    def __len__(self) -> int:
        return self.total

    def high_frequency_shard_indices(self) -> list[int]:
        if self.prior_shards is None and not any(
            "pair_id" in shard or "year" in shard for shard in self.shards
        ):
            # Historic indexes did not carry per-pair metadata, so retain their
            # all-shard behavior instead of silently dropping the corpus.
            return list(range(len(self.shards)))
        eligible = []
        descriptors = self.shards if self.prior_shards is None else self.prior_shards
        hf_years = frozenset(getattr(self, "hf_years", (2017, 2018)))
        for index, shard in enumerate(descriptors):
            parts = str(shard.get("pair_id", "")).split(":")
            try:
                year = int(shard.get("year", parts[0]))
                delta_days = int(
                    shard.get(
                        "delta_days",
                        abs((date.fromisoformat(parts[-1]) - date.fromisoformat(parts[-3])).days),
                    )
                )
            except (ValueError, IndexError):
                continue
            candidate = bool(shard.get("hf_candidate", delta_days <= 1 and year in hf_years))
            if candidate and year in hf_years and delta_days <= 1:
                eligible.append(index)
        if not eligible:
            raise RuntimeError("no delta-t <= 1 training shards are available")
        return eligible

    def _load(
        self, index: int
    ) -> tuple[dict[str, object], dict[str, object] | None, int]:
        shard_index = bisect.bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        if shard_index != self._cache_index:
            self._cache = torch.load(
                self.shards[shard_index]["path"], map_location="cpu", weights_only=False
            )
            self._prior_cache = (
                torch.load(
                    self.prior_shards[shard_index]["path"],
                    map_location="cpu",
                    weights_only=False,
                )
                if self.prior_shards is not None
                else None
            )
            if self._prior_cache is not None:
                if self._prior_cache["pair_id"] != self._cache["pair_id"]:
                    raise RuntimeError("temporal-prior shard pair IDs do not match training data")
                if not torch.equal(self._prior_cache["window"], self._cache["window"]):
                    raise RuntimeError("temporal-prior windows do not match training data")
            self._cache_index = shard_index
        assert self._cache is not None
        return self._cache, self._prior_cache, index - start

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += self.total
        if not 0 <= index < self.total:
            raise IndexError(index)
        shard, prior_shard, local = self._load(index)
        encoded_s2 = shard["s2"][local].float().unsqueeze(0)
        encoded_sar = shard["sar"][local].float().unsqueeze(0)
        if encoded_s2.shape[1] != len(self._s2_reorder):
            raise RuntimeError("shard S2 tensor does not match its declared channel order")
        if encoded_sar.shape[1] != len(self._sar_reorder):
            raise RuntimeError("shard SAR tensor does not match its declared channel order")
        s2 = normalized_s2_to_reflectance(encoded_s2[:, self._s2_reorder])
        sar = normalized_sar_to_db(encoded_sar[:, self._sar_reorder])
        valid = shard["joint_valid"][local].float().unsqueeze(0)
        # Stored invalid pixels use normalized zero as a compact placeholder.
        # Restore the raw-evaluator contract before any geometric resampling.
        s2 = torch.where(valid.bool(), s2, torch.zeros_like(s2))
        sar = torch.where(valid.bool(), sar, torch.zeros_like(sar))
        optical_prior = (
            prior_shard["optical"][local].unsqueeze(0)
            if prior_shard is not None
            else None
        )
        optical_coverage = (
            prior_shard["optical_coverage"][local].unsqueeze(0)
            if prior_shard is not None
            else None
        )
        sar_prior = (
            prior_shard["sar"][local].unsqueeze(0)
            if prior_shard is not None
            else None
        )
        sar_coverage = (
            prior_shard["sar_coverage"][local].unsqueeze(0)
            if prior_shard is not None
            else None
        )
        flip_x = False
        flip_y = False
        rotations = 0
        if self.augment:
            flip_x = bool(torch.rand(()) < 0.5)
            if flip_x:
                s2, sar, valid = (torch.flip(value, (-1,)) for value in (s2, sar, valid))
                if optical_prior is not None:
                    optical_prior, optical_coverage, sar_prior, sar_coverage = (
                        torch.flip(value, (-1,))
                        for value in (
                            optical_prior,
                            optical_coverage,
                            sar_prior,
                            sar_coverage,
                        )
                    )
            flip_y = bool(torch.rand(()) < 0.5)
            if flip_y:
                s2, sar, valid = (torch.flip(value, (-2,)) for value in (s2, sar, valid))
                if optical_prior is not None:
                    optical_prior, optical_coverage, sar_prior, sar_coverage = (
                        torch.flip(value, (-2,))
                        for value in (
                            optical_prior,
                            optical_coverage,
                            sar_prior,
                            sar_coverage,
                        )
                    )
            rotations = int(torch.randint(0, 4, ()))
            if rotations:
                s2, sar, valid = (
                    torch.rot90(value, rotations, (-2, -1)) for value in (s2, sar, valid)
                )
                if optical_prior is not None:
                    optical_prior, optical_coverage, sar_prior, sar_coverage = (
                        torch.rot90(value, rotations, (-2, -1))
                        for value in (
                            optical_prior,
                            optical_coverage,
                            sar_prior,
                            sar_coverage,
                        )
                    )
        use_native_gsd = not self.random_gsd or bool(
            torch.rand(()) < self.native_gsd_probability
        )
        input_gsd = (
            10.0 if use_native_gsd else float((10, 20, 40)[int(torch.randint(0, 3, ()))])
        )
        target_gsd = 10.0 if use_native_gsd else float((10, 20)[int(torch.randint(0, 2, ()))])
        s2_view = physical_resample(
            s2,
            modality="optical",
            source_gsd_m=10.0,
            target_gsd_m=input_gsd,
            valid=valid,
        )
        sar_view = physical_resample(
            sar,
            modality="sar",
            source_gsd_m=10.0,
            target_gsd_m=input_gsd,
            valid=valid,
        )
        s2_target = physical_resample(
            s2,
            modality="optical",
            source_gsd_m=10.0,
            target_gsd_m=target_gsd,
            valid=valid,
        )
        sar_target = physical_resample(
            sar,
            modality="sar",
            source_gsd_m=10.0,
            target_gsd_m=target_gsd,
            valid=valid,
        )
        metadata = shard["metadata"][local].float()
        delta_days = round(abs(float(metadata[0])) * 3.0)
        pair_id = str(shard["pair_id"][local])
        try:
            year = int(pair_id.split(":", 1)[0])
        except ValueError:
            year = -1
        valid_fraction = float(valid.mean())
        s2_valid = shard.get("s2_valid", shard["joint_valid"])[local].float()
        cloud_shadow_fraction = float(1.0 - s2_valid.mean())
        registration_shift = (
            estimate_registration_shift(s2.squeeze(0), sar.squeeze(0), valid=valid.squeeze(0))
            if self.audit_high_frequency and delta_days <= 1
            else torch.tensor(float("inf"))
        )
        eligible = (
            high_frequency_eligible(
                delta_days=delta_days,
                year=year,
                split=self.split,
                registration_shift_px=registration_shift,
                valid_fraction=valid_fraction,
                cloud_shadow_fraction=cloud_shadow_fraction,
                train_years=self.hf_years,
            )
            if self.audit_high_frequency
            else delta_days <= 1 and year in self.hf_years and self.split == "train"
        )
        result = {
            "s2": s2.squeeze(0),
            "sar": sar.squeeze(0),
            "s2_view": s2_view.squeeze(0),
            "sar_view": sar_view.squeeze(0),
            "s2_target": s2_target.squeeze(0),
            "sar_target": sar_target.squeeze(0),
            "valid": valid.squeeze(0),
            "metadata": metadata,
            "input_gsd": torch.tensor(input_gsd),
            "target_gsd": torch.tensor(target_gsd),
            "delta_days": torch.tensor(delta_days),
            "pair_id": pair_id,
            "window": shard["window"][local],
            "year": torch.tensor(year),
            "registration_shift_px": registration_shift,
            "valid_fraction": torch.tensor(valid_fraction),
            "cloud_shadow_fraction": torch.tensor(cloud_shadow_fraction),
            "hf_eligible": torch.tensor(eligible),
            "augmentation": torch.tensor((int(flip_x), int(flip_y), rotations)),
        }
        if optical_prior is not None:
            result.update(
                {
                    "optical_temporal_prior": optical_prior.squeeze(0),
                    "optical_temporal_coverage": optical_coverage.squeeze(0),
                    "sar_temporal_prior": sar_prior.squeeze(0),
                    "sar_temporal_coverage": sar_coverage.squeeze(0),
                }
            )
        return result

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_cache_index"] = -1
        state["_cache"] = None
        state["_prior_cache"] = None
        return state


class StatefulShardSampler(Sampler[int]):
    def __init__(
        self,
        dataset: V2ShardDataset,
        *,
        replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        high_frequency_only: bool = False,
    ) -> None:
        self.dataset = dataset
        self.replicas = replicas
        self.rank = rank
        self.seed = seed
        self.shard_indices = (
            dataset.high_frequency_shard_indices()
            if high_frequency_only
            else list(range(len(dataset.shards)))
        )
        self.epoch = 0
        self.offset = 0
        eligible_samples = sum(
            int(dataset.shards[index]["count"]) for index in self.shard_indices
        )
        self.num_samples = (eligible_samples + replicas - 1) // replicas

    def _indices(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.shard_indices), generator=generator).tolist()
        shard_order = [self.shard_indices[index] for index in order]
        indices: list[int] = []
        for shard_index in shard_order[self.rank :: self.replicas]:
            start = 0 if shard_index == 0 else self.dataset.ends[shard_index - 1]
            count = int(self.dataset.shards[shard_index]["count"])
            member_order = torch.randperm(count, generator=generator).tolist()
            indices.extend(start + member for member in member_order)
        if not indices:
            raise RuntimeError(f"rank {self.rank} was assigned no shards")
        if len(indices) < self.num_samples:
            repeats = (self.num_samples + len(indices) - 1) // len(indices)
            indices = (indices * repeats)[: self.num_samples]
        return indices[: self.num_samples]

    def __iter__(self):
        indices = self._indices()
        for position in range(self.offset, len(indices)):
            self.offset = position + 1
            yield indices[position]
        self.epoch += 1
        self.offset = 0

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state["epoch"])
        self.offset = int(state["offset"])


class StatefulIndexSampler(Sampler[int]):
    """Distributed resumable sampler over an audited subset of dataset indices."""

    def __init__(
        self,
        indices: list[int],
        *,
        replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ) -> None:
        if not indices:
            raise ValueError("audited sampler requires at least one eligible index")
        self.indices = indices
        self.replicas = replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.offset = 0
        self.num_samples = (len(indices) + replicas - 1) // replicas

    def _indices(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.indices), generator=generator).tolist()
        total_size = self.num_samples * self.replicas
        if len(order) < total_size:
            order.extend(order[: total_size - len(order)])
        return [self.indices[position] for position in order[self.rank : total_size : self.replicas]]

    def __iter__(self):
        indices = self._indices()
        for position in range(self.offset, len(indices)):
            self.offset = position + 1
            yield indices[position]
        self.epoch += 1
        self.offset = 0

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state["epoch"])
        self.offset = int(state["offset"])


def time_weights(delta_days: Tensor) -> tuple[Tensor, Tensor]:
    # Validation scores every temporal bucket equally; keep noisy long gaps useful
    # without allowing them to dominate exact or one-day pairs.
    physical = delta_days.new_tensor((1.0, 1.0, 0.75, 0.5), dtype=torch.float32)
    high_frequency = delta_days.new_tensor((1.0, 0.25, 0.0, 0.0), dtype=torch.float32)
    indices = delta_days.long().clamp(0, 3)
    return physical[indices], high_frequency[indices]
