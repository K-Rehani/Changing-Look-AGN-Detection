"""Command line entry points for catalog selection, caching, and analysis."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .catalog import read_pairs, select_pairs, write_pairs
from .comparison import compare_spectra
from .fetch import fetch_spectrum, sha256_file
from .plotting import plot_comparison
from .spectra import read_sdss_spectrum
from .synthetic import make_example, write_example_fits


def _save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _compare(args: argparse.Namespace) -> int:
    old = read_sdss_spectrum(args.earlier)
    new = read_sdss_spectrum(args.later)
    result = compare_spectra(old, new, args.z)
    _save_json(args.out_json, result.to_dict())
    if args.plot:
        plot_comparison(old, new, result, args.plot)
    print(f"Review flag: {result.broad_transition_review}; "
          f"priority score: {result.priority_score:.2f}")
    return 0


def _select(args: argparse.Namespace) -> int:
    count = write_pairs(args.out, select_pairs(args.catalog, args.max_z, args.limit))
    print(f"Wrote {count} repeat-spectrum pairs to {args.out}")
    return 0


def _fetch(args: argparse.Namespace) -> int:
    cache = Path(args.cache_dir)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    failures = 0
    count = 0
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "filename", "release", "retrieved_at_utc", "sha256", "status", "error",
        ])
        writer.writeheader()
        for pair in read_pairs(args.manifest):
            if args.limit is not None and count >= args.limit:
                break
            count += 1
            for obs in (pair.old, pair.new):
                if obs in seen:
                    continue
                seen.add(obs)
                row = {"filename": obs.filename, "release": args.release,
                       "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                       "sha256": "", "status": "", "error": ""}
                try:
                    path, downloaded = fetch_spectrum(
                        obs, cache, release=args.release, retries=args.retries
                    )
                    row["sha256"] = sha256_file(path)
                    row["status"] = "downloaded" if downloaded else "cached"
                except Exception as exc:
                    failures += 1
                    row["status"] = "failed"
                    row["error"] = str(exc)
                writer.writerow(row)
                stream.flush()
    print(f"Processed {count} pairs, {len(seen)} unique spectra; "
          f"{failures} failures. Log: {log_path}")
    return 1 if failures else 0


def _analyze(args: argparse.Namespace) -> int:
    cache = Path(args.cache_dir)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, pair in enumerate(read_pairs(args.manifest)):
        if args.limit is not None and index >= args.limit:
            break
        row = {"name": pair.name, "redshift": pair.redshift,
               "rest_days": pair.rest_days, "old_file": pair.old.filename,
               "new_file": pair.new.filename, "priority_score": "",
               "broad_transition_review": "", "status": "", "error": ""}
        try:
            old = read_sdss_spectrum(cache / pair.old.filename)
            new = read_sdss_spectrum(cache / pair.new.filename)
            result = compare_spectra(old, new, pair.redshift)
            row.update(priority_score=f"{result.priority_score:.4f}",
                       broad_transition_review=str(result.broad_transition_review),
                       status="analyzed")
            if args.details_dir:
                _save_json(Path(args.details_dir) / f"{index:06d}.json",
                           {"pair": pair.to_row(), "analysis": result.to_dict()})
            if args.plots_dir and (args.plot_all or result.broad_transition_review):
                plot_comparison(old, new, result,
                                Path(args.plots_dir) / f"{index:06d}.png")
        except (OSError, ValueError, KeyError, IndexError) as exc:
            row["status"], row["error"] = "failed", str(exc)
        rows.append(row)
    fields = ["name", "redshift", "rest_days", "old_file", "new_file",
              "priority_score", "broad_transition_review", "status", "error"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pair outcomes to {output}; "
          f"{sum(r['status'] == 'failed' for r in rows)} failed")
    return 1 if any(r["status"] == "failed" for r in rows) else 0


def _demo(args: argparse.Namespace) -> int:
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    old, new = make_example()
    old_path = write_example_fits(old, output / "SYNTHETIC_earlier.fits")
    new_path = write_example_fits(new, output / "SYNTHETIC_later.fits")
    old, new = read_sdss_spectrum(old_path), read_sdss_spectrum(new_path)
    result = compare_spectra(old, new, 0.15)
    _save_json(output / "SYNTHETIC_result.json", result.to_dict())
    plot_comparison(old, new, result, output / "SYNTHETIC_comparison.png")
    print(f"Synthetic demonstration written to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clagn", description="Prioritize repeat optical AGN spectra for review"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select-pairs", help="select repeat observations in DR16Q_v4")
    select.add_argument("catalog", type=Path)
    select.add_argument("--out", type=Path, required=True)
    select.add_argument("--max-z", type=float, default=0.8)
    select.add_argument("--limit", type=int)
    select.set_defaults(func=_select)

    fetch = sub.add_parser("fetch", help="download listed spectra to a local cache")
    fetch.add_argument("manifest", type=Path)
    fetch.add_argument("--cache-dir", type=Path, required=True)
    fetch.add_argument("--log", type=Path, default=Path("results/fetch-log.csv"))
    fetch.add_argument("--release", type=int, default=17)
    fetch.add_argument("--retries", type=int, default=2)
    fetch.add_argument("--limit", type=int, help="maximum number of pairs")
    fetch.set_defaults(func=_fetch)

    analyze = sub.add_parser("analyze", help="analyze cached spectra in a manifest")
    analyze.add_argument("manifest", type=Path)
    analyze.add_argument("--cache-dir", type=Path, required=True)
    analyze.add_argument("--out", type=Path, required=True)
    analyze.add_argument("--details-dir", type=Path)
    analyze.add_argument("--plots-dir", type=Path)
    analyze.add_argument("--plot-all", action="store_true")
    analyze.add_argument("--limit", type=int)
    analyze.set_defaults(func=_analyze)

    compare = sub.add_parser("compare", help="compare two local SDSS spec FITS files")
    compare.add_argument("earlier", type=Path)
    compare.add_argument("later", type=Path)
    compare.add_argument("--z", type=float, required=True, help="shared catalog redshift")
    compare.add_argument("--out-json", type=Path, required=True)
    compare.add_argument("--plot", type=Path)
    compare.set_defaults(func=_compare)

    demo = sub.add_parser("demo", help="run a synthetic broad-line transition example")
    demo.add_argument("--out-dir", type=Path, default=Path("results/demo"))
    demo.set_defaults(func=_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"clagn: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
