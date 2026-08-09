"""Canonical channel schemas shared by raw data, shards, and evaluation."""

from __future__ import annotations

from collections.abc import Sequence

S2_CHANNEL_ORDER: tuple[str, ...] = (
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "swir16",
    "swir22",
)
SAR_CHANNEL_ORDER: tuple[str, ...] = ("vv", "vh")
CLEAR_SCL_CODES: tuple[int, ...] = (2, 4, 5, 6, 7)

# V1 shards were written before the V3 descriptor order was standardized.
LEGACY_V1_S2_CHANNEL_ORDER: tuple[str, ...] = (
    "red",
    "blue",
    "green",
    "nir",
    "nir08",
    "rededge1",
    "rededge2",
    "rededge3",
    "swir16",
    "swir22",
)

S2_RGB_INDICES: tuple[int, int, int] = tuple(
    S2_CHANNEL_ORDER.index(channel) for channel in ("red", "green", "blue")
)


def channel_reorder_indices(
    source_order: Sequence[str],
    target_order: Sequence[str] = S2_CHANNEL_ORDER,
) -> tuple[int, ...]:
    """Return the strict source indices needed to produce ``target_order``."""

    source = tuple(str(channel) for channel in source_order)
    target = tuple(str(channel) for channel in target_order)
    if len(source) != len(set(source)):
        raise ValueError(f"duplicate channels in source order: {source}")
    if len(target) != len(set(target)):
        raise ValueError(f"duplicate channels in target order: {target}")
    if set(source) != set(target):
        missing = tuple(channel for channel in target if channel not in source)
        unexpected = tuple(channel for channel in source if channel not in target)
        raise ValueError(
            f"channel order mismatch; missing={missing}, unexpected={unexpected}"
        )
    return tuple(source.index(channel) for channel in target)


def require_channel_order(
    channels: Sequence[str], target_order: Sequence[str] = S2_CHANNEL_ORDER
) -> tuple[str, ...]:
    """Validate that a stored schema is an exact permutation of the canonical schema."""

    normalized = tuple(str(channel) for channel in channels)
    channel_reorder_indices(normalized, target_order)
    return normalized
