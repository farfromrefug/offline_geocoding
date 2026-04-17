"""Command-line interface for offline-geocoding.

Commands
--------
import   – Build a geocoding database from a Photon JSONL(.zst) dump.
search   – Forward-geocoding / full-text place search.
reverse  – Reverse geocoding (nearest place to a coordinate).
stats    – Show database statistics (row counts + per-table disk usage).

Examples
--------
# Build a database for English + French, filtered to a .poly region:
    offline-geocoding import \\
        --input  /data/photon-dump.jsonl.zst \\
        --output /data/geocoding.db \\
        --languages en,fr \\
        --poly    /data/europe.poly \\
        --workers 8

# Import only natural peaks and all tourism features:
    offline-geocoding import ... --tag-include /data/include.json

# Compress R-tree to 0.1° grid cells:
    offline-geocoding import ... --grid-size 0.1

# Search:
    offline-geocoding search --db /data/geocoding.db "Eiffel Tower"
    offline-geocoding search --db /data/geocoding.db "Paris" \\
        --bbox 2.0,48.5,3.0,49.2 --limit 5

# Reverse geocoding:
    offline-geocoding reverse --db /data/geocoding.db --lat 48.8566 --lon 2.3522

# Database statistics:
    offline-geocoding stats --db /data/geocoding.db
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import List, Optional, Tuple

import click

from . import __version__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """Parse ``'min_lon,min_lat,max_lon,max_lat'`` into a float tuple."""
    parts = bbox_str.split(",")
    if len(parts) != 4:
        raise click.BadParameter(
            "bbox must be four comma-separated floats: min_lon,min_lat,max_lon,max_lat"
        )
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise click.BadParameter(f"Invalid bbox value: {exc}") from exc


def _parse_languages(lang_str: str) -> List[str]:
    langs = [l.strip() for l in lang_str.split(",") if l.strip()]
    if not langs:
        raise click.BadParameter("At least one language code must be specified.")
    return langs


def _print_result(result: dict, index: int) -> None:
    """Pretty-print one search result."""
    click.echo(f"  [{index}] {result.get('name') or '(unnamed)'}")
    click.echo(f"       lat={result['lat']:.6f}  lon={result['lon']:.6f}")
    if result.get("address_type"):
        click.echo(f"       type={result['address_type']}", nl=False)
        if result.get("osm_key") and result.get("osm_value"):
            click.echo(f"  osm={result['osm_key']}={result['osm_value']}", nl=False)
        click.echo()
    addr = result.get("address", {})
    addr_parts = []
    for k in ("city", "state", "country"):
        val = addr.get(k)
        if val:
            addr_parts.append(val)
    if addr_parts:
        click.echo(f"       address: {', '.join(addr_parts)}")
    if result.get("importance") is not None:
        click.echo(f"       importance={result['importance']:.4f}")
    if result.get("distance_deg") is not None:
        click.echo(f"       distance={result['distance_deg']:.6f}°")
    if result.get("categories"):
        click.echo(f"       categories: {', '.join(result['categories'])}")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(__version__, prog_name="offline-geocoding")
def main() -> None:
    """Offline geocoding tool – build and query SQLite databases from Photon dumps."""


# ---------------------------------------------------------------------------
# import command
# ---------------------------------------------------------------------------

@main.command("import")
@click.option(
    "--input", "-i", "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to the Photon dump file (.jsonl or .jsonl.zst).",
)
@click.option(
    "--output", "-o", "output_path",
    required=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Path to the output SQLite database (created or overwritten).",
)
@click.option(
    "--languages", "-l",
    default="en",
    show_default=True,
    help="Comma-separated list of ISO 639-1 language codes to import "
         "(e.g. 'en,fr,de').  The bare OSM 'name' tag is always kept.",
)
@click.option(
    "--poly", "-p",
    "poly_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Optional .poly file to spatially filter imported places.",
)
@click.option(
    "--workers", "-w",
    default=0,
    show_default=True,
    type=int,
    help="Number of worker processes (0 = use all available CPUs).",
)
@click.option(
    "--batch-size", "-b",
    default=2000,
    show_default=True,
    type=int,
    help="Number of JSON lines per worker task.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option(
    "--single-thread", "single_thread",
    is_flag=True,
    default=False,
    help=(
        "Run the import on a single thread without worker sub-processes. "
        "Simpler, avoids the merge step, and defers index creation to the end "
        "for faster bulk-insert performance.  Recommended when you have few CPUs "
        "or want predictable, reproducible behaviour."
    ),
)
@click.option(
    "--store-extra",
    is_flag=True,
    default=False,
    help=(
        "Store the 'extra' JSON blob (tags, wikipedia links, etc.) from each "
        "place in the database.  Significantly increases database size.  "
        "Disabled by default."
    ),
)
@click.option(
    "--tag-filter",
    "tag_filter_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help=(
        "Path to a JSON file specifying OSM tag combinations to *skip* during "
        "import (block-list).  File must be a JSON array of objects with "
        "'osm_key' and optional 'osm_value' fields.  When not specified a "
        "built-in default filter is applied.  Pass an empty array file to "
        "disable filtering."
    ),
)
@click.option(
    "--tag-include",
    "tag_include_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help=(
        "Path to a JSON file specifying OSM tag combinations to *keep* during "
        "import (allow-list).  Only places matching at least one entry are "
        "imported.  Use null/absent 'osm_value' to include all values for a "
        "key.  Applied after --tag-filter.  Example entry to include all "
        "natural features: {\"osm_key\": \"natural\"}."
    ),
)
@click.option(
    "--grid-size",
    "grid_size",
    default=0.0,
    show_default=True,
    type=float,
    help=(
        "R-tree grid cell size in degrees.  When > 0, places within the same "
        "grid cell share a single R-tree entry, reducing R-tree table size for "
        "dense datasets.  For example, 0.1 creates ~11 km grid cells.  "
        "Reverse queries remain correct but may return a few extra candidates "
        "near cell borders.  Default 0 = one R-tree entry per place."
    ),
)
def cmd_import(
    input_path: str,
    output_path: str,
    languages: str,
    poly_file: Optional[str],
    workers: int,
    batch_size: int,
    verbose: bool,
    single_thread: bool,
    store_extra: bool,
    tag_filter_path: Optional[str],
    tag_include_path: Optional[str],
    grid_size: float,
) -> None:
    """Import a Photon JSONL(.zst) dump into a geocoding SQLite database."""
    _setup_logging(verbose)
    langs = _parse_languages(languages)
    click.echo(
        f"Importing '{input_path}' → '{output_path}' "
        f"[languages: {langs}, workers: {workers or 'auto'}]"
    )
    if poly_file:
        click.echo(f"  poly filter: {poly_file}")

    from .importer import import_database, load_tag_filter, load_tag_include_filter, _DEFAULT_TAG_FILTER

    tag_filter = None
    if tag_filter_path is not None:
        tag_filter = load_tag_filter(tag_filter_path)
        click.echo(f"  tag filter (block): {tag_filter_path} ({len(tag_filter)} rules)")
    else:
        tag_filter = _DEFAULT_TAG_FILTER
        click.echo(f"  tag filter (block): built-in default ({len(tag_filter)} rules)")

    tag_include = None
    if tag_include_path is not None:
        tag_include = load_tag_include_filter(tag_include_path)
        click.echo(f"  tag include (allow): {tag_include_path} ({len(tag_include)} rules)")

    if grid_size > 0:
        click.echo(f"  R-tree grid size: {grid_size}°")

    import_database(
        input_path=input_path,
        output_path=output_path,
        languages=langs,
        poly_file=poly_file,
        num_workers=workers,
        batch_size=batch_size,
        show_progress=True,
        single_thread=single_thread,
        tag_filter=tag_filter,
        tag_include=tag_include,
        store_extra=store_extra,
        grid_size=grid_size,
    )
    click.echo("Done.")


# ---------------------------------------------------------------------------
# search command
# ---------------------------------------------------------------------------

@main.command("search")
@click.argument("query")
@click.option(
    "--db", "-d", "db_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to the geocoding SQLite database.",
)
@click.option(
    "--languages", "-l",
    default=None,
    help="Comma-separated preferred display language codes "
         "(defaults to languages stored in the database).",
)
@click.option(
    "--bbox",
    default=None,
    help="Bounding box filter: min_lon,min_lat,max_lon,max_lat (WGS84).",
)
@click.option(
    "--limit", "-n",
    default=10,
    show_default=True,
    type=int,
    help="Maximum number of results.",
)
@click.option(
    "--offset",
    default=0,
    show_default=True,
    type=int,
    help="Pagination offset.",
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output results as JSON.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cmd_search(
    query: str,
    db_path: str,
    languages: Optional[str],
    bbox: Optional[str],
    limit: int,
    offset: int,
    output_json: bool,
    verbose: bool,
) -> None:
    """Forward-geocoding: search for places matching QUERY."""
    _setup_logging(verbose)
    langs = _parse_languages(languages) if languages else None
    bbox_tuple = _parse_bbox(bbox) if bbox else None

    from .query import search as do_search

    results = do_search(
        db_path,
        query,
        languages=langs,
        bbox=bbox_tuple,
        limit=limit,
        offset=offset,
    )

    if output_json:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        if not results:
            click.echo("No results found.")
        else:
            click.echo(f"Found {len(results)} result(s) for '{query}':")
            for i, r in enumerate(results, 1):
                _print_result(r, i)


# ---------------------------------------------------------------------------
# reverse command
# ---------------------------------------------------------------------------

@main.command("reverse")
@click.option(
    "--db", "-d", "db_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to the geocoding SQLite database.",
)
@click.option("--lat", required=True, type=float, help="Latitude (WGS84).")
@click.option("--lon", required=True, type=float, help="Longitude (WGS84).")
@click.option(
    "--radius", "-r",
    default=0.1,
    show_default=True,
    type=float,
    help="Search radius in degrees (≈111 km per degree).",
)
@click.option(
    "--limit", "-n",
    default=5,
    show_default=True,
    type=int,
    help="Maximum number of results.",
)
@click.option(
    "--languages", "-l",
    default=None,
    help="Comma-separated preferred display language codes.",
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output results as JSON.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cmd_reverse(
    db_path: str,
    lat: float,
    lon: float,
    radius: float,
    limit: int,
    languages: Optional[str],
    output_json: bool,
    verbose: bool,
) -> None:
    """Reverse geocoding: find nearest places to (lat, lon)."""
    _setup_logging(verbose)
    langs = _parse_languages(languages) if languages else None

    from .query import reverse as do_reverse

    results = do_reverse(
        db_path,
        lat=lat,
        lon=lon,
        radius_deg=radius,
        limit=limit,
        languages=langs,
    )

    if output_json:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        if not results:
            click.echo(f"No places found within {radius}° of ({lat}, {lon}).")
        else:
            click.echo(
                f"Nearest {len(results)} place(s) to ({lat}, {lon}) "
                f"within {radius}°:"
            )
            for i, r in enumerate(results, 1):
                _print_result(r, i)


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------

@main.command("stats")
@click.option(
    "--db", "-d", "db_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to the geocoding SQLite database.",
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output as JSON.",
)
@click.option(
    "--sizes",
    "show_sizes",
    is_flag=True,
    default=False,
    help="Also show per-table disk usage (requires dbstat virtual table).",
)
def cmd_stats(db_path: str, output_json: bool, show_sizes: bool) -> None:
    """Show row-count (and optionally disk-usage) statistics for a geocoding database."""
    from .query import stats as do_stats, get_languages, table_sizes as do_table_sizes

    counts = do_stats(db_path)
    langs = get_languages(db_path)
    sizes = do_table_sizes(db_path) if show_sizes else []

    if output_json:
        out: dict = {"languages": langs, "counts": counts}
        if sizes:
            out["table_sizes"] = sizes
        click.echo(json.dumps(out, indent=2))
    else:
        click.echo(f"Database: {db_path}")
        click.echo(f"Languages: {', '.join(langs) if langs else '(unknown)'}")
        click.echo("Row counts:")
        for table, count in counts.items():
            click.echo(f"  {table:<22} {count:>12,}")
        if sizes:
            click.echo("Disk usage (by table/index):")
            for entry in sizes:
                size_kb = entry["bytes"] / 1024
                click.echo(
                    f"  {entry['table']:<30} {entry['pages']:>8} pages  "
                    f"{size_kb:>10.1f} KB"
                )
