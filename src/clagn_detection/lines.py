"""Local continuum and integrated line measurements.

Flux density remains in observed-frame SDSS units. Windows are specified in
rest-frame Angstroms. Line flux is integrated over observed wavelength, so
its numerical unit is 1e-17 erg s-1 cm-2 for standard SDSS spec files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectra import Spectrum


@dataclass(frozen=True)
class Line:
    name: str
    center: float
    window: tuple[float, float]
    left: tuple[float, float]
    right: tuple[float, float]


# Broad-line and continuum windows adapted from Guo et al. (2024), ApJS 270,
# 26, Table 3. H-alpha includes narrow H-alpha and [N II]; an integrated
# H-alpha measurement is not an isolated broad-line measurement.
LINES = {
    "Hbeta": Line("Hbeta", 4861.333, (4780, 4940), (4730, 4770), (5030, 5080)),
    "Halpha": Line("Halpha", 6562.819, (6450, 6700), (6420, 6450), (6750, 6790)),
    "MgII": Line("MgII", 2798.75, (2750, 2850), (2680, 2720), (2880, 2920)),
    # A diagnostic for relative calibration, not an automatic flux correction.
    "OIII5007": Line("OIII5007", 5006.843, (4996, 5018), (4970, 4990), (5025, 5045)),
}


class CoverageError(ValueError):
    """A measurement window is not covered by enough usable pixels."""


@dataclass(frozen=True)
class LineMeasurement:
    name: str
    flux: float
    flux_error: float
    ew_rest: float | None
    continuum: float
    continuum_error: float
    n_line_pixels: int
    n_continuum_pixels: int

    @property
    def flux_snr(self) -> float:
        return self.flux / self.flux_error


def _window_samples(x: np.ndarray, good: np.ndarray, window: tuple[float, float],
                    step: float, min_points: int = 5) -> np.ndarray:
    in_window = (x >= window[0]) & (x <= window[1])
    valid = in_window & good
    indices = np.flatnonzero(valid)
    if (
        indices.size < min_points
        or x[0] > window[0] + step
        or x[-1] < window[1] - step
        or x[indices[0]] > window[0] + 2.5 * step
        or x[indices[-1]] < window[1] - 2.5 * step
        or np.any(np.diff(x[indices]) > 3 * step)
    ):
        raise CoverageError(f"insufficient unmasked coverage in {window} Angstrom")
    return indices


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    if x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("integration grid must be strictly increasing")
    w = np.empty_like(x)
    w[0] = (x[1] - x[0]) / 2
    w[-1] = (x[-1] - x[-2]) / 2
    if x.size > 2:
        w[1:-1] = (x[2:] - x[:-2]) / 2
    return w


def measure_line(spectrum: Spectrum, redshift: float, line: Line) -> LineMeasurement:
    x = spectrum.wavelength_rest(redshift)
    good = spectrum.good
    if np.count_nonzero(good) < 20:
        raise CoverageError("spectrum has too few valid pixels")
    step = float(np.median(np.diff(x)))
    if step <= 0 or not np.isfinite(step):
        raise CoverageError("invalid wavelength spacing")
    left = _window_samples(x, good, line.left, step)
    right = _window_samples(x, good, line.right, step)
    indices = _window_samples(x, good, line.window, step)
    cont = np.concatenate((left, right))
    center_offset = x[cont] - line.center
    design = np.column_stack((np.ones(cont.size), center_offset))
    weights = spectrum.ivar[cont]
    normal = design.T @ (weights[:, None] * design)
    if np.linalg.cond(normal) > 1e12:
        raise CoverageError("continuum fit is ill-conditioned")
    covariance = np.linalg.inv(normal)
    coefficients = covariance @ (design.T @ (weights * spectrum.flux[cont]))
    residual = spectrum.flux[cont] - design @ coefficients
    reduced_chi2 = np.sum(weights * residual**2) / max(1, cont.size - 2)
    covariance *= max(1.0, float(reduced_chi2))

    x_line = x[indices]
    w_obs = _trapezoid_weights(x_line) * (1.0 + redshift)
    offsets = x_line - line.center
    baseline = coefficients[0] + coefficients[1] * offsets
    flux = float(np.dot(w_obs, spectrum.flux[indices] - baseline))
    line_variance = np.sum(w_obs**2 / spectrum.ivar[indices])
    continuum_vector = np.array((np.sum(w_obs), np.dot(w_obs, offsets)))
    variance = float(line_variance + continuum_vector @ covariance @ continuum_vector)
    if not np.isfinite(variance) or variance <= 0:
        raise CoverageError("line-flux uncertainty is invalid")
    continuum = float(coefficients[0])
    continuum_error = float(np.sqrt(covariance[0, 0]))
    ew_rest = flux / ((1.0 + redshift) * continuum) if continuum > 0 else None
    return LineMeasurement(
        line.name, flux, float(np.sqrt(variance)), ew_rest,
        continuum, continuum_error, indices.size, cont.size,
    )
