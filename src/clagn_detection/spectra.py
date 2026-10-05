"""Read one-dimensional SDSS spec FITS files without hiding quality information."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

FLUX_UNIT = "1e-17 erg s-1 cm-2 Angstrom-1"


@dataclass(frozen=True)
class Spectrum:
    wavelength_obs: np.ndarray
    flux: np.ndarray
    ivar: np.ndarray
    bad: np.ndarray
    identifier: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arrays = (
            np.asarray(self.wavelength_obs, dtype=float),
            np.asarray(self.flux, dtype=float),
            np.asarray(self.ivar, dtype=float),
            np.asarray(self.bad, dtype=bool),
        )
        if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1:
            raise ValueError("wavelength, flux, ivar, and bad must be equal-length 1D arrays")
        if len(arrays[0]) < 3 or not np.all(np.isfinite(arrays[0])):
            raise ValueError("wavelength needs at least three finite values")
        if np.any(arrays[0] <= 0) or np.any(np.diff(arrays[0]) <= 0):
            raise ValueError("observed wavelengths must be positive and strictly increasing")
        for name, array in zip(("wavelength_obs", "flux", "ivar", "bad"), arrays):
            object.__setattr__(self, name, array)

    @property
    def good(self) -> np.ndarray:
        return (
            ~self.bad
            & np.isfinite(self.flux)
            & np.isfinite(self.ivar)
            & (self.ivar > 0)
        )

    def wavelength_rest(self, redshift: float) -> np.ndarray:
        if not np.isfinite(redshift) or redshift < 0:
            raise ValueError("redshift must be finite and nonnegative")
        return self.wavelength_obs / (1.0 + redshift)


def read_sdss_spectrum(path: str | Path, *, reject_or_bits: int = 0) -> Spectrum:
    """Read an SDSS individual spec FITS file.

    The default rejects zero-IVAR pixels and nonzero AND_MASK pixels. OR_MASK
    can be filtered with explicitly chosen bits; rejecting every OR_MASK bit
    by default would discard many otherwise usable pixels. The SDSS flux
    values are kept in their native observed-frame per-Angstrom units.
    """
    path = Path(path)
    with fits.open(path, memmap=True) as hdus:
        if len(hdus) < 2 or hdus[1].data is None:
            raise ValueError(f"{path} has no SDSS spectral table in HDU 1")
        table = hdus[1].data
        columns = {name.lower(): name for name in table.names}
        missing = {"flux", "loglam", "ivar"} - columns.keys()
        if missing:
            raise ValueError(f"{path} is missing spectral columns: {sorted(missing)}")
        flux = np.asarray(table[columns["flux"]], dtype=float).copy()
        loglam = np.asarray(table[columns["loglam"]], dtype=float).copy()
        ivar = np.asarray(table[columns["ivar"]], dtype=float).copy()
        bad = np.zeros(flux.size, dtype=bool)
        if "and_mask" in columns:
            bad |= np.asarray(table[columns["and_mask"]]) != 0
        if reject_or_bits:
            if reject_or_bits < 0:
                raise ValueError("reject_or_bits must be nonnegative")
            if "or_mask" not in columns:
                raise ValueError("OR_MASK column is absent")
            bad |= (np.asarray(table[columns["or_mask"]], dtype=np.int64) & reject_or_bits) != 0
        header = hdus[0].header
        metadata = {
            key: str(header[key])
            for key in ("PLATEID", "MJD", "FIBERID", "RUN2D")
            if key in header
        }
    return Spectrum(10.0**loglam, flux, ivar, bad, path.stem, metadata)
