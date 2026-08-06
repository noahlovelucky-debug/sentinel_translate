from __future__ import annotations

from itertools import pairwise
from typing import ClassVar

from sentinel_v3.data import StatefulShardSampler


class _Shards:
    shards: ClassVar[list[dict[str, int]]] = [
        {"count": 4}, {"count": 4}, {"count": 4}, {"count": 4}
    ]
    ends: ClassVar[list[int]] = [4, 8, 12, 16]

    def __len__(self) -> int:
        return 16


def test_sampler_keeps_shard_members_contiguous() -> None:
    sampler = StatefulShardSampler(_Shards(), replicas=1, rank=0, seed=7)  # type: ignore[arg-type]
    indices = list(sampler)
    shard_ids = [index // 4 for index in indices]
    transitions = sum(left != right for left, right in pairwise(shard_ids))
    assert transitions == 3
    assert sorted(indices) == list(range(16))


def test_distributed_ranks_have_equal_length_and_disjoint_shards() -> None:
    first = StatefulShardSampler(_Shards(), replicas=2, rank=0, seed=7)  # type: ignore[arg-type]
    second = StatefulShardSampler(_Shards(), replicas=2, rank=1, seed=7)  # type: ignore[arg-type]
    first_indices = list(first)
    second_indices = list(second)
    assert len(first_indices) == len(second_indices) == 8
    assert {index // 4 for index in first_indices}.isdisjoint(
        {index // 4 for index in second_indices}
    )
