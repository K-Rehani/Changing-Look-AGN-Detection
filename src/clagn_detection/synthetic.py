"""Deterministic synthetic example; no simulated value is an SDSS result."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from .spectra import Spectrum


def _gaussian(x: np.ndarray, center: float, sigma: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def make_example(redshift: float = 0.15, seed: int = 31) -> tuple[Spectrum, Spectrum]:
    """Make a pair with a disappearing broad Balmer component and offset grids."""
    rng = np.random.default_rng(seed)
    spectra = []
    for epoch, offset in ((0, 0.0), (1, 0.27)):
        x = np.arange(2650.0 + offset, 6900.0, 1.25)
        continuum = 9 + 0.0004 * (x - 4861)
        flux = continuum.copy()
        for center, narrow_amp in ((4861.333, 2.3), (6562.819, 2.5)):
            flux += _gaussian(x, center, 2.2, narrow_amp)
        flux += _gaussian(x, 6548.05, 2.2, 0.7)
        flux += _gaussian(x, 6583.45, 2.2, 2.07)
        flux += _gaussian(x, 5006.843, 2.0, 1.8)
        flux += _gaussian(x, 2798.75, 12.0, 1.5)
        if epoch == 0:
            flux += _gaussian(x, 4861.333, 22.0, 2.8)
            flux += _gaussian(x, 6562.819, 30.0, 3.5)
        noise = 0.18
        flux += rng.normal(0, noise, size=x.size)
        bad = np.zeros(x.size, dtype=bool)
        bad[(x > 5300) & (x < 5305)] = True
        spectra.append(Spectrum(
            x * (1 + redshift), flux, np.full(x.size, 1 / noise**2),
            bad, f"SYNTHETIC_epoch_{epoch}", {"SYNTHETIC": "true"},
        ))
    return spectra[0], spectra[1]


def write_example_fits(spectrum: Spectrum, path: str | Path) -> Path:
    """Write a small FITS file matching the fields used by the SDSS reader."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        fits.Column(name="flux", format="E", array=spectrum.flux.astype("f4")),
        fits.Column(name="loglam", format="D", array=np.log10(spectrum.wavelength_obs)),
        fits.Column(name="ivar", format="E", array=spectrum.ivar.astype("f4")),
        fits.Column(name="and_mask", format="J", array=spectrum.bad.astype("i4")),
        fits.Column(name="or_mask", format="J", array=spectrum.bad.astype("i4")),
    ]
    primary = fits.PrimaryHDU()
    primary.header["SYNTH"] = (True, "Synthetic example; not observed SDSS data")
    fits.HDUList([primary, fits.BinTableHDU.from_columns(columns)]).writeto(
        path, overwrite=True
    )
    return path
