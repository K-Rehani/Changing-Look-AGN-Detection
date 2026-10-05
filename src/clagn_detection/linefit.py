"""Exploratory broad Balmer component fits for human-review prioritization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .lines import CoverageError, Line, _window_samples
from .spectra import Spectrum

C_KM_S = 299792.458
FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))
MAX_BROAD_SIGMA = 70.0  # rest-frame Angstroms; fits at this bound are rejected


@dataclass(frozen=True)
class BroadFit:
    name: str
    broad_flux: float
    broad_flux_error: float | None
    broad_flux_snr: float | None
    fwhm_km_s: float
    delta_bic: float
    detected: bool
    n_pixels: int


def _gaussian(x: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _model(x: np.ndarray, p: np.ndarray, line: Line, with_broad: bool) -> np.ndarray:
    a, b, amp_n, shift_n, sigma_n = p[:5]
    flux = a + b * (x - line.center)
    flux = flux + amp_n * _gaussian(x, line.center + shift_n, sigma_n)
    next_index = 5
    if line.name == "Halpha":
        amp_nii = p[5]
        flux = flux + amp_nii * _gaussian(x, 6583.45 + shift_n, sigma_n)
        flux = flux + (amp_nii / 2.96) * _gaussian(x, 6548.05 + shift_n, sigma_n)
        next_index = 6
    if with_broad:
        amp_b, shift_b, sigma_b = p[next_index:next_index + 3]
        flux = flux + amp_b * _gaussian(x, line.center + shift_b, sigma_b)
    return flux


def fit_broad_balmer(spectrum: Spectrum, redshift: float, line: Line) -> BroadFit:
    """Compare a narrow-only model with a narrow-plus-broad model.

    Fits use formal IVAR weights. The covariance estimate assumes independent
    pixels and is only a prioritization diagnostic; it does not cover
    calibration, Fe II, host continuum, or model-selection uncertainties.
    """
    if line.name not in {"Hbeta", "Halpha"}:
        raise ValueError("broad-component fits are implemented for Hbeta and Halpha")
    x = spectrum.wavelength_rest(redshift)
    step = float(np.median(np.diff(x)))
    parts = [
        _window_samples(x, spectrum.good, window, step)
        for window in (line.left, line.window, line.right)
    ]
    indices = np.unique(np.concatenate(parts))
    if indices.size < 35:
        raise CoverageError("too few valid pixels for broad-line fit")
    xx, yy = x[indices], spectrum.flux[indices]
    root_ivar = np.sqrt(spectrum.ivar[indices])
    cont_indices = np.concatenate((parts[0], parts[2]))
    a0 = float(np.median(spectrum.flux[cont_indices]))
    peak = max(0.1, float(np.max(yy[(xx >= line.center - 8) & (xx <= line.center + 8)]) - a0))
    base = [a0, 0.0, peak, 0.0, 2.0]
    lower = [-np.inf, -np.inf, 0.0, -8.0, 0.8]
    upper = [np.inf, np.inf, np.inf, 8.0, 8.0]
    if line.name == "Halpha":
        base.append(max(0.1, peak / 3))
        lower.append(0.0)
        upper.append(np.inf)

    def solve(initial: list[float], lo: list[float], hi: list[float],
              with_broad: bool):
        return least_squares(
            lambda p: (_model(xx, p, line, with_broad) - yy) * root_ivar,
            initial, bounds=(lo, hi), max_nfev=2000,
        )

    narrow = solve(base, lower, upper, False)
    sigma_min = line.center * 1000.0 / (C_KM_S * FWHM_FACTOR)
    broad = solve(
        list(narrow.x) + [max(0.1, peak / 3), 0.0, max(sigma_min * 1.5, 12.0)],
        lower + [0.0, -25.0, sigma_min],
        upper + [np.inf, 25.0, MAX_BROAD_SIGMA],
        True,
    )
    chi2_n = float(np.sum(narrow.fun**2))
    chi2_b = float(np.sum(broad.fun**2))
    delta_bic = chi2_n - chi2_b - (len(broad.x) - len(narrow.x)) * np.log(indices.size)
    amp, _, sigma = broad.x[-3:]
    broad_flux = float(amp * sigma * np.sqrt(2 * np.pi) * (1 + redshift))
    fwhm = float(FWHM_FACTOR * sigma / line.center * C_KM_S)
    error = None
    snr = None
    try:
        normal = broad.jac.T @ broad.jac
        if np.linalg.cond(normal) < 1e12:
            covariance = np.linalg.inv(normal) * max(
                1.0, chi2_b / max(1, indices.size - len(broad.x))
            )
            gradient = np.zeros(len(broad.x))
            factor = np.sqrt(2 * np.pi) * (1 + redshift)
            gradient[-3] = sigma * factor
            gradient[-1] = amp * factor
            variance = float(gradient @ covariance @ gradient)
            if np.isfinite(variance) and variance > 0:
                error = float(np.sqrt(variance))
                snr = broad_flux / error
    except np.linalg.LinAlgError:
        pass
    detected = bool(
        broad.success and narrow.success and delta_bic > 10
        and snr is not None and snr >= 5
        and sigma < MAX_BROAD_SIGMA * 0.98
        and fwhm >= 1000
    )
    return BroadFit(line.name, broad_flux, error, snr, fwhm, delta_bic,
                    detected, indices.size)
