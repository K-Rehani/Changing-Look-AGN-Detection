"""Catalog and FITS boundary checks."""

import csv

import numpy as np
from astropy.io import fits
import pytest

from clagn_detection.catalog import read_pairs, select_pairs, write_pairs
from clagn_detection.spectra import read_sdss_spectrum
from clagn_detection.synthetic import make_example, write_example_fits


def test_catalog_uses_oldest_and_latest_valid_observations(tmp_path):
    columns = [
        fits.Column(name="SDSS_NAME", format="18A", array=["J0000+0000", "J0001+0001"]),
        fits.Column(name="Z", format="E", array=[0.3, 1.2]),
        fits.Column(name="NSPEC", format="J", array=[2, 1]),
        fits.Column(name="PLATE", format="J", array=[10, 20]),
        fits.Column(name="MJD", format="J", array=[57000, 57000]),
        fits.Column(name="FIBERID", format="J", array=[11, 21]),
        fits.Column(name="PLATE_DUPLICATE", format="3J", array=[[12, 13, -1], [22, -1, -1]]),
        fits.Column(name="MJD_DUPLICATE", format="3J", array=[[56000, 58000, -1], [58000, -1, -1]]),
        fits.Column(name="FIBERID_DUPLICATE", format="3J", array=[[12, 13, -1], [22, -1, -1]]),
    ]
    path = tmp_path / "tiny-catalog.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(path)
    pairs = list(select_pairs(path))
    assert len(pairs) == 1
    assert (pairs[0].old.plate, pairs[0].old.mjd) == (12, 56000)
    assert (pairs[0].new.plate, pairs[0].new.mjd) == (13, 58000)
    assert pairs[0].rest_days == pytest.approx(2000 / 1.3, rel=1e-7)
    manifest = tmp_path / "pairs.csv"
    assert write_pairs(manifest, iter(pairs)) == 1
    assert list(read_pairs(manifest)) == pairs
    with manifest.open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 1


def test_sdss_like_reader_roundtrip_and_bad_pixels(tmp_path):
    old, _ = make_example()
    path = write_example_fits(old, tmp_path / "example.fits")
    restored = read_sdss_spectrum(path)
    assert np.allclose(restored.wavelength_obs, old.wavelength_obs)
    assert np.allclose(restored.flux, old.flux, rtol=1e-6)
    assert np.array_equal(restored.good, old.good)
