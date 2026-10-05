"""Optional SDSS retrieval with a local cache and a per-file provenance log."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time

from .catalog import Observation


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_spectrum(observation: Observation, cache_dir: str | Path, *,
                   release: int = 17, retries: int = 2) -> tuple[Path, bool]:
    """Fetch exactly one plate/MJD/fiber spectrum through astroquery.

    Returns (path, was_downloaded). A missing or failed download never creates
    a cache entry. The raw returned FITS HDUs are preserved unchanged.
    """
    try:
        from astroquery.sdss import SDSS
    except ImportError as exc:
        raise RuntimeError("Install the optional dependency: pip install '.[sdss]'") from exc
    if retries < 0 or release < 8:
        raise ValueError("retries must be >= 0 and SDSS release must be >= 8")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / observation.filename
    if target.is_file() and target.stat().st_size:
        return target, False
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            hdus_list = SDSS.get_spectra(
                plate=observation.plate, mjd=observation.mjd,
                fiberID=observation.fiber, data_release=release,
                cache=True, show_progress=False, timeout=60,
            )
            if not hdus_list or len(hdus_list) != 1:
                raise RuntimeError("SDSS returned zero or multiple spectra for one identifier")
            temporary = target.with_suffix(".fits.part")
            try:
                hdus_list[0].writeto(temporary, overwrite=True, checksum=True)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
                hdus_list[0].close()
            return target, True
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"could not fetch {observation.filename}: {last_error}")
