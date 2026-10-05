"""Select oldest and latest repeat observations from the DR16Q quasar catalog."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from astropy.io import fits


@dataclass(frozen=True, order=True)
class Observation:
    mjd: int
    plate: int
    fiber: int

    @property
    def filename(self) -> str:
        return f"spec-{self.plate:04d}-{self.mjd}-{self.fiber:04d}.fits"


@dataclass(frozen=True)
class Pair:
    name: str
    redshift: float
    old: Observation
    new: Observation

    @property
    def rest_days(self) -> float:
        return (self.new.mjd - self.old.mjd) / (1.0 + self.redshift)

    def to_row(self) -> dict[str, str | int | float]:
        return {
            "name": self.name, "redshift": self.redshift,
            "old_plate": self.old.plate, "old_mjd": self.old.mjd,
            "old_fiber": self.old.fiber, "new_plate": self.new.plate,
            "new_mjd": self.new.mjd, "new_fiber": self.new.fiber,
            "rest_days": self.rest_days,
        }


PAIR_FIELDS = [
    "name", "redshift", "old_plate", "old_mjd", "old_fiber",
    "new_plate", "new_mjd", "new_fiber", "rest_days",
]


def _identifier(value: object) -> str:
    return value.decode("ascii", errors="replace").strip() if isinstance(value, bytes) else str(value).strip()


def observations_from_row(row: object) -> list[Observation]:
    """Collect the primary observation and every valid DR16Q duplicate triple."""
    columns = {name.lower(): name for name in row.array.names}
    needed = {
        "plate", "mjd", "fiberid",
        "plate_duplicate", "mjd_duplicate", "fiberid_duplicate",
    }
    if not needed <= columns.keys():
        raise ValueError(f"DR16Q columns missing: {sorted(needed - columns.keys())}")
    triples = [(row[columns["plate"]], row[columns["mjd"]], row[columns["fiberid"]])]
    triples.extend(zip(
        np.atleast_1d(row[columns["plate_duplicate"]]),
        np.atleast_1d(row[columns["mjd_duplicate"]]),
        np.atleast_1d(row[columns["fiberid_duplicate"]]),
    ))
    unique = {
        Observation(int(mjd), int(plate), int(fiber))
        for plate, mjd, fiber in triples
        if int(plate) > 0 and int(mjd) > 0 and int(fiber) > 0
    }
    return sorted(unique)


def select_pairs(catalog: str | Path, max_z: float = 0.8, limit: int | None = None) -> Iterator[Pair]:
    """Yield one widest-baseline pair per selected DR16Q object, in catalog order.

    FITS access is memory mapped. Only Z and NSPEC are searched as full columns;
    the large duplicate arrays are accessed for the selected rows.
    """
    if max_z <= 0 or not np.isfinite(max_z):
        raise ValueError("max_z must be positive and finite")
    with fits.open(catalog, memmap=True) as hdus:
        if len(hdus) < 2:
            raise ValueError("expected the DR16Q catalog in HDU 1")
        data = hdus[1].data
        cols = {name.lower(): name for name in data.names}
        missing = {"z", "nspec", "sdss_name"} - cols.keys()
        if missing:
            raise ValueError(f"catalog columns missing: {sorted(missing)}")
        selected = np.flatnonzero(
            np.isfinite(data[cols["z"]])
            & (data[cols["z"]] > 0)
            & (data[cols["z"]] < max_z)
            & (data[cols["nspec"]] >= 1)
        )
        yielded = 0
        for index in selected:
            row = data[int(index)]
            observations = observations_from_row(row)
            if len(observations) < 2 or observations[0].mjd == observations[-1].mjd:
                continue
            yield Pair(
                _identifier(row[cols["sdss_name"]]),
                float(row[cols["z"]]),
                observations[0],
                observations[-1],
            )
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def write_pairs(path: str | Path, pairs: Iterator[Pair]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair.to_row())
            count += 1
    return count


def read_pairs(path: str | Path) -> Iterator[Pair]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(PAIR_FIELDS) <= set(reader.fieldnames or []):
            raise ValueError(f"manifest needs columns: {PAIR_FIELDS}")
        for row in reader:
            yield Pair(
                row["name"], float(row["redshift"]),
                Observation(int(row["old_mjd"]), int(row["old_plate"]), int(row["old_fiber"])),
                Observation(int(row["new_mjd"]), int(row["new_plate"]), int(row["new_fiber"])),
            )
