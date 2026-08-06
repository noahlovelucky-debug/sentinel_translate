from __future__ import annotations

import bisect
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .physics import normalized_s2_to_reflectance, normalized_sar_to_db, physical_resample


class V2ShardDataset(Dataset[dict[str, object]]):
    """Read existing shards without copying or rewriting the 61 GB corpus."""

    def __init__(self, index_path: str | Path, *, augment: bool = True, random_gsd: bool = True) -> None:
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
        self._cache_index = -1
        self._cache: dict[str, object] | None = None

    def __len__(self) -> int:
        return self.total

    def _load(self, index: int) -> tuple[dict[str, object], int]:
        shard_index = bisect.bisect_right(self.ends, index)
        start = 0 if shard_index == 0 else self.ends[shard_index - 1]
        if shard_index != self._cache_index:
            self._cache = torch.load(self.shards[shard_index]["path"], map_location="cpu", weights_only=False)
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
        if self.augment:
            if torch.rand(()) < 0.5:
                s2, sar, valid = (torch.flip(value, (-1,)) for value in (s2, sar, valid))
            if torch.rand(()) < 0.5:
                s2, sar, valid = (torch.flip(value, (-2,)) for value in (s2, sar, valid))
            rotations = int(torch.randint(0, 4, ()))
            if rotations:
                s2, sar, valid = (torch.rot90(value, rotations, (-2, -1)) for value in (s2, sar, valid))
        input_gsd = float((10, 20, 40)[int(torch.randint(0, 3, ()))]) if self.random_gsd else 10.0
        target_gsd = float((10, 20)[int(torch.randint(0, 2, ()))]) if self.random_gsd else 10.0
        s2_view = physical_resample(s2, modality="optical", source_gsd_m=10.0, target_gsd_m=input_gsd)
        sar_view = physical_resample(sar, modality="sar", source_gsd_m=10.0, target_gsd_m=input_gsd)
        s2_target = physical_resample(s2, modality="optical", source_gsd_m=10.0, target_gsd_m=target_gsd)
        sar_target = physical_resample(sar, modality="sar", source_gsd_m=10.0, target_gsd_m=target_gsd)
        metadata = shard["metadata"][local].float()
        delta_days = round(abs(float(metadata[0])) * 3.0)
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
            "pair_id": shard["pair_id"][local],
            "window": shard["window"][local],
        }
        return result

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_cache_index"] = -1
        state["_cache"] = None
        return state


class StatefulShardSampler(Sampler[int]):
    def __init__(self, dataset: V2ShardDataset, *, replicas: int = 1, rank: int = 0, seed: int = 42) -> None:
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
    physical = delta_days.new_tensor((1.0, 0.75, 0.4, 0.2), dtype=torch.float32)
    high_frequency = delta_days.new_tensor((1.0, 0.5, 0.0, 0.0), dtype=torch.float32)
    indices = delta_days.long().clamp(0, 3)
    return physical[indices], high_frequency[indices]
