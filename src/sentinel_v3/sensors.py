from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Modality = Literal["optical", "sar"]


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    kind: Literal["reflectance", "backscatter_db", "backscatter_intensity"]
    native_gsd_m: float
    grid_gsd_m: float
    wavelength_nm: float | None = None
    frequency_ghz: float | None = None
    polarization: str | None = None
    psf_sigma_pixels: float = 0.5

    def descriptor(self) -> tuple[float, ...]:
        polarization = {None: 0.0, "VV": 1.0, "VH": -1.0}.get(self.polarization, 0.0)
        return (
            1.0 if self.kind == "reflectance" else 0.0,
            1.0 if self.kind.startswith("backscatter") else 0.0,
            0.0 if self.wavelength_nm is None else self.wavelength_nm / 2500.0,
            0.0 if self.frequency_ghz is None else self.frequency_ghz / 10.0,
            polarization,
            self.native_gsd_m / 40.0,
            self.grid_gsd_m / 40.0,
            self.psf_sigma_pixels / 2.0,
        )


@dataclass(frozen=True)
class SensorSpec:
    name: str
    modality: Modality
    channels: tuple[ChannelSpec, ...]
    units: str

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)


SENTINEL2 = SensorSpec(
    name="sentinel-2",
    modality="optical",
    units="surface_reflectance",
    channels=tuple(
        ChannelSpec(
            name, "reflectance", native, 10.0, wavelength_nm=wavelength, psf_sigma_pixels=psf
        )
        for name, wavelength, native, psf in (
            ("B02", 492.4, 10.0, 0.55),
            ("B03", 559.8, 10.0, 0.55),
            ("B04", 664.6, 10.0, 0.55),
            ("B05", 704.1, 20.0, 1.05),
            ("B06", 740.5, 20.0, 1.05),
            ("B07", 782.8, 20.0, 1.05),
            ("B08", 832.8, 10.0, 0.55),
            ("B8A", 864.7, 20.0, 1.05),
            ("B11", 1613.7, 20.0, 1.05),
            ("B12", 2202.4, 20.0, 1.05),
        )
    ),
)

SENTINEL1 = SensorSpec(
    name="sentinel-1",
    modality="sar",
    units="decibel_backscatter",
    channels=(
        ChannelSpec("VV", "backscatter_db", 10.0, 10.0, frequency_ghz=5.405, polarization="VV"),
        ChannelSpec("VH", "backscatter_db", 10.0, 10.0, frequency_ghz=5.405, polarization="VH"),
    ),
)

_REGISTRY = {spec.name: spec for spec in (SENTINEL1, SENTINEL2)}


def register_sensor(spec: SensorSpec, *, replace: bool = False) -> None:
    if spec.name in _REGISTRY and not replace:
        raise ValueError(f"sensor already registered: {spec.name}")
    if not spec.channels:
        raise ValueError("a sensor must contain at least one channel")
    _REGISTRY[spec.name] = spec


def get_sensor(name: str) -> SensorSpec:
    try:
        return _REGISTRY[name]
    except KeyError as error:
        raise KeyError(f"unknown sensor {name!r}; registered={sorted(_REGISTRY)}") from error
