"""Diagnostic plots; interpolation here is only for visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .comparison import Comparison
from .lines import LINES
from .spectra import Spectrum


def _interpolate_valid(x: np.ndarray, y: np.ndarray, valid: np.ndarray,
                       target: np.ndarray) -> np.ndarray:
    """Interpolate only between adjacent valid pixels and leave masked gaps blank."""
    result = np.full_like(target, np.nan, dtype=float)
    right = np.searchsorted(x, target, side="left")
    in_range = (right > 0) & (right < x.size)
    idx = np.flatnonzero(in_range)
    lo, hi = right[idx] - 1, right[idx]
    adjacent = valid[lo] & valid[hi]
    idx, lo, hi = idx[adjacent], lo[adjacent], hi[adjacent]
    fraction = (target[idx] - x[lo]) / (x[hi] - x[lo])
    result[idx] = y[lo] * (1 - fraction) + y[hi] * fraction
    return result


def plot_comparison(old: Spectrum, new: Spectrum, result: Comparison,
                    output: str | Path) -> Path:
    x1, x2 = old.wavelength_rest(result.redshift), new.wavelength_rest(result.redshift)
    lo, hi = max(x1[0], x2[0]), min(x1[-1], x2[-1])
    if hi <= lo:
        raise ValueError("spectra have no overlapping rest-frame wavelength range")
    step = max(float(np.median(np.diff(x1))), float(np.median(np.diff(x2))))
    common = np.arange(lo, hi, step)
    if common.size < 3:
        raise ValueError("overlap is too small for a diagnostic plot")
    f1 = _interpolate_valid(x1, old.flux, old.good, common)
    f2 = _interpolate_valid(x2, new.flux, new.good, common)
    fig, (upper, lower) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True,
    )
    upper.plot(x1, np.where(old.good, old.flux, np.nan), lw=0.75, label="earlier")
    upper.plot(x2, np.where(new.good, new.flux, np.nan), lw=0.75, label="later")
    lower.plot(common, f2 - f1, color="#5b3a9e", lw=0.75)
    lower.axhline(0, color="0.4", lw=0.8)
    for name, line in LINES.items():
        if name == "OIII5007" or not (lo <= line.center <= hi):
            continue
        for axis in (upper, lower):
            axis.axvspan(*line.window, color="steelblue", alpha=0.07)
        lower.text(line.center, 0.98, name, transform=lower.get_xaxis_transform(),
                   ha="center", va="top", fontsize=8)
    upper.set_ylabel("SDSS flux density\n(1e-17 erg s-1 cm-2 A-1)")
    lower.set_ylabel("later - earlier")
    lower.set_xlabel("Rest wavelength (Angstrom)")
    upper.legend(loc="upper right")
    title = f"{old.identifier} / {new.identifier}, z={result.redshift:.4f}"
    if result.broad_transition_review:
        title += " | broad-line review flag"
    upper.set_title(title)
    upper.set_xlim(lo, hi)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
