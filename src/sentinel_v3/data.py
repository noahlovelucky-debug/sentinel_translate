from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .physics import normalized_s2_to_reflectance, normalized_sar_to_db, physical_resample


def estimate_registration_shift(s2: Tensor, sar: Tensor, maximum_size: int = 64) -> Tensor:
    """Estimate translation from cross-modal gradient phase correlation."""
    optical = s2[[2, 1, 0]].mean(0, keepdim=True).unsqueeze(0)
    radar = sar.mean(0, keepdim=True).unsqueeze(0)
    optical = F.interpolate(optical, size=(maximum_size, maximum_size), mode="area")
    radar = F.interpolate(radar, size=(maximum_size, maximum_size), mode="area")

    def gradient(values: Tensor) -> Tensor:
        dy = F.pad(values[..., 1:, :] - values[..., :-1, :], (0, 0, 0, 1))
        dx = F.pad(values[..., :, 1:] - values[..., :, :-1], (0, 1))
        magnitude = torch.sqrt(dx.square() + dy.square() + 1e-8)
        return (magnitude - magnitude.mean()) / magnitude.std().clamp_min(1e-6)

    optical_fft = torch.fft.rfft2(gradient(optical).float())
    radar_fft = torch.fft.rfft2(gradient(radar).float())
    cross = optical_fft * radar_fft.conj()
    correlation = torch.fft.irfft2(cross / cross.abs().clamp_min(1e-8))
    peak = int(correlation.flatten().argmax())
    row, col = divmod(peak, maximum_size)
    row = row if row <= maximum_size // 2 else row - maximum_size
    col = col if col <= maximum_size // 2 else col - maximum_size
    scale_y = s2.shape[-2] / maximum_size
    scale_x = s2.shape[-1] / maximum_size
    return s2.new_tensor(math.hypot(row * scale_y, col * scale_x))


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
) -> bool:
    return (
        int(delta_days) <= 1
        and year in {2017, 2018}
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
    ) -> None:
        self.index_path = Path(index_path)
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.shards = list(self.index["shards"])
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

    def __len__(self) -> int:
        return self.total

    def _load(self, index: int) -> tuple[dict[str, object], int]:
        shard_index = bisect.bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        if shard_index != self._cache_index:
            self._cache = torch.load(
                self.shards[shard_index]["path"], map_location="cpu", weights_only=False
            )
            self._cache_index = shard_index
        assert self._cache is not None
        return self._cache, index - start

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += self.total
        if not 0 <= index < self.total:
            raise IndexError(index)
        shard, local = self._load(index)
        s2 = normalized_s2_to_reflectance(shard["s2"][local].float().unsqueeze(0))
        sar = normalized_sar_to_db(shard["sar"][local].float().unsqueeze(0))
        valid = shard["joint_valid"][local].float().unsqueeze(0)
        flip_x = False
        flip_y = False
        rotations = 0
        if self.augment:
            flip_x = bool(torch.rand(()) < 0.5)
            if flip_x:
                s2, sar, valid = (torch.flip(value, (-1,)) for value in (s2, sar, valid))
            flip_y = bool(torch.rand(()) < 0.5)
            if flip_y:
                s2, sar, valid = (torch.flip(value, (-2,)) for value in (s2, sar, valid))
            rotations = int(torch.randint(0, 4, ()))
            if rotations:
                s2, sar, valid = (
                    torch.rot90(value, rotations, (-2, -1)) for value in (s2, sar, valid)
                )
        use_native_gsd = not self.random_gsd or bool(
            torch.rand(()) < self.native_gsd_probability
        )
        input_gsd = (
            10.0 if use_native_gsd else float((10, 20, 40)[int(torch.randint(0, 3, ()))])
        )
        target_gsd = 10.0 if use_native_gsd else float((10, 20)[int(torch.randint(0, 2, ()))])
        s2_view = physical_resample(
            s2, modality="optical", source_gsd_m=10.0, target_gsd_m=input_gsd
        )
        sar_view = physical_resample(
            sar, modality="sar", source_gsd_m=10.0, target_gsd_m=input_gsd
        )
        s2_target = physical_resample(
            s2, modality="optical", source_gsd_m=10.0, target_gsd_m=target_gsd
        )
        sar_target = physical_resample(
            sar, modality="sar", source_gsd_m=10.0, target_gsd_m=target_gsd
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
            estimate_registration_shift(s2.squeeze(0), sar.squeeze(0))
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
            )
            if self.audit_high_frequency
            else delta_days <= 1 and year in {2017, 2018} and self.split == "train"
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
        return result

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_cache_index"] = -1
        state["_cache"] = None
        return state


class StatefulShardSampler(Sampler[int]):
    def __init__(
        self, dataset: V2ShardDataset, *, replicas: int = 1, rank: int = 0, seed: int = 42
    ) -> None:
        self.dataset = dataset
        self.replicas = replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.offset = 0
        self.num_samples = (len(dataset) + replicas - 1) // replicas

    def _indices(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(len(self.dataset.shards), generator=generator).tolist()
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


def time_weights(delta_days: Tensor) -> tuple[Tensor, Tensor]:
    # Validation scores every temporal bucket equally; keep noisy long gaps useful
    # without allowing them to dominate exact or one-day pairs.
    physical = delta_days.new_tensor((1.0, 1.0, 0.75, 0.5), dtype=torch.float32)
    high_frequency = delta_days.new_tensor((1.0, 0.25, 0.0, 0.0), dtype=torch.float32)
    indices = delta_days.long().clamp(0, 3)
    return physical[indices], high_frequency[indices]
