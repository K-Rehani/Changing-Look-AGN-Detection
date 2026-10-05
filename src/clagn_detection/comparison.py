"""Compare two epochs and produce review priorities, never automatic confirmations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .linefit import BroadFit, fit_broad_balmer
from .lines import CoverageError, LINES, LineMeasurement, measure_line
from .spectra import Spectrum


@dataclass
class Comparison:
    old_id: str
    new_id: str
    redshift: float
    measurements: dict[str, dict]
    broad_fits: dict[str, dict]
    priority_score: float
    broad_transition_review: bool
    reasons: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                return None
            if isinstance(value, (np.integer, np.bool_)):
                return value.item()
            return value
        return clean(asdict(self))


def _line_pair(old: LineMeasurement, new: LineMeasurement) -> dict:
    delta = new.flux - old.flux
    error = math.hypot(old.flux_error, new.flux_error)
    return {
        "old": asdict(old),
        "new": asdict(new),
        "delta_flux": delta,
        "delta_flux_error": error,
        "delta_flux_snr": delta / error,
    }


def _broad_pair(old: BroadFit, new: BroadFit) -> dict:
    result = {"old": asdict(old), "new": asdict(new), "change_snr": None}
    if old.broad_flux_error is not None and new.broad_flux_error is not None:
        result["change_snr"] = (new.broad_flux - old.broad_flux) / math.hypot(
            old.broad_flux_error, new.broad_flux_error
        )
    return result


def compare_spectra(old: Spectrum, new: Spectrum, redshift: float) -> Comparison:
    if not np.isfinite(redshift) or redshift < 0:
        raise ValueError("redshift must be finite and nonnegative")
    measurements: dict[str, dict] = {}
    broad_fits: dict[str, dict] = {}
    warnings: list[str] = []
    reasons: list[str] = []
    scores: list[float] = []
    for name, line in LINES.items():
        try:
            first = measure_line(old, redshift, line)
            last = measure_line(new, redshift, line)
            measurements[name] = _line_pair(first, last)
            if name != "OIII5007":
                scores.append(abs(measurements[name]["delta_flux_snr"]))
            if name in {"Hbeta", "Halpha"}:
                try:
                    broad_fits[name] = _broad_pair(
                        fit_broad_balmer(old, redshift, line),
                        fit_broad_balmer(new, redshift, line),
                    )
                except (CoverageError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    broad_fits[name] = {"error": str(exc)}
        except (CoverageError, ValueError, np.linalg.LinAlgError) as exc:
            measurements[name] = {"error": str(exc)}
    oiii = measurements.get("OIII5007", {})
    if "delta_flux" in oiii:
        first, last = oiii["old"], oiii["new"]
        if min(first["flux"] / first["flux_error"],
               last["flux"] / last["flux_error"]) >= 5:
            mean_flux = (abs(first["flux"]) + abs(last["flux"])) / 2
            if (mean_flux > 0 and abs(oiii["delta_flux"]) / mean_flux > 0.3
                    and abs(oiii["delta_flux_snr"]) >= 3):
                warnings.append(
                    "[O III] 5007 differs by >30%; inspect relative calibration, "
                    "aperture effects, and intrinsic variability."
                )
    review = False
    for name in ("Hbeta", "Halpha"):
        fit = broad_fits.get(name, {})
        line_result = measurements.get(name, {})
        if "old" not in fit or "delta_flux_snr" not in line_result:
            continue
        transition = fit["old"]["detected"] != fit["new"]["detected"]
        change_snr = fit["change_snr"]
        if transition and change_snr is not None and abs(change_snr) >= 3:
            review = True
            reasons.append(
                f"{name}: broad-component fit changes between epochs "
                f"(formal broad-flux change {change_snr:.1f} sigma)"
            )
            scores.append(abs(change_snr))
    if not any("delta_flux_snr" in item for item in measurements.values()):
        warnings.append("No line has usable coverage in both epochs.")
    warnings.append(
        "Model fits and formal errors do not include correlated noise, host/Fe II "
        "mismatch, or relative calibration systematics; review flags need inspection."
    )
    return Comparison(old.identifier, new.identifier, redshift, measurements,
                      broad_fits, max(scores, default=0.0), review, reasons, warnings)
