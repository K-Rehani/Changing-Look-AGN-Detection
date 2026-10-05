"""Behavioral checks using known spectra, not survey detections."""

import numpy as np
import pytest

from clagn_detection.comparison import compare_spectra
from clagn_detection.lines import CoverageError, LINES, measure_line
from clagn_detection.spectra import Spectrum
from clagn_detection.synthetic import make_example


def test_disappearing_broad_balmer_lines_are_flagged_on_offset_grids():
    old, new = make_example()
    assert not np.array_equal(old.wavelength_obs, new.wavelength_obs)
    result = compare_spectra(old, new, 0.15)
    assert result.broad_transition_review
    for name in ("Hbeta", "Halpha"):
        assert result.broad_fits[name]["old"]["detected"]
        assert not result.broad_fits[name]["new"]["detected"]
        assert result.measurements[name]["delta_flux_snr"] < -5


def test_continuum_offset_does_not_imitate_line_transition():
    old, _ = make_example()
    new = Spectrum(old.wavelength_obs.copy(), old.flux + 2.0,
                   old.ivar.copy(), old.bad.copy(), "bright-continuum")
    result = compare_spectra(old, new, 0.15)
    assert not result.broad_transition_review
    assert abs(result.measurements["Hbeta"]["delta_flux_snr"]) < 0.01
    assert abs(result.measurements["Halpha"]["delta_flux_snr"]) < 0.01


@pytest.mark.parametrize("z", [0.0, 0.6])
def test_line_flux_and_rest_ew_respect_redshift_and_units(z):
    x = np.arange(4670.0, 5110.0, 0.5)
    amplitude, sigma, continuum = 3.0, 12.0, 5.0
    # A fixed rest-frame equivalent width has observed f_lambda reduced by 1+z.
    flux = (continuum + amplitude * np.exp(-0.5 * ((x - 4861.333) / sigma)**2)) / (1 + z)
    spectrum = Spectrum(x * (1 + z), flux, np.full(x.size, 1e4),
                        np.zeros(x.size, dtype=bool))
    measured = measure_line(spectrum, z, LINES["Hbeta"])
    expected = amplitude * sigma * np.sqrt(2 * np.pi)
    assert measured.flux == pytest.approx(expected, rel=0.003)
    assert measured.ew_rest == pytest.approx(expected / continuum, rel=0.003)


def test_masked_line_and_missing_red_coverage_fail_locally():
    old, new = make_example()
    x = old.wavelength_rest(0.15)
    bad = old.bad | ((x > 4830) & (x < 4890))
    masked = Spectrum(old.wavelength_obs, old.flux, old.ivar, bad)
    with pytest.raises(CoverageError):
        measure_line(masked, 0.15, LINES["Hbeta"])
    short = Spectrum(old.wavelength_obs[x < 5300], old.flux[x < 5300],
                     old.ivar[x < 5300], old.bad[x < 5300])
    result = compare_spectra(short, new, 0.15)
    assert "error" in result.measurements["Halpha"]
    assert "delta_flux_snr" in result.measurements["Hbeta"]


def test_zero_inverse_variance_cannot_contribute_to_line_measurement():
    old, _ = make_example()
    x = old.wavelength_rest(0.15)
    ivar = old.ivar.copy()
    ivar[(x > 4840) & (x < 4880)] = 0
    with pytest.raises(CoverageError):
        measure_line(Spectrum(old.wavelength_obs, old.flux, ivar, old.bad),
                     0.15, LINES["Hbeta"])
