"""Import a Photon JSONL.ZST dump into the offline geocoding SQLite database.

Architecture
------------
* A **reader** in the main process decompresses the ``.zst`` file and splits
  it into line batches.
* A ``multiprocessing.Pool`` of **worker** processes parses JSON, filters by
  the .poly file (optional), and extracts structured place data.
* A **writer** in the main process receives processed batches (in order) and
  performs all SQLite writes inside large transactions for maximum throughput.

Worker processes only handle pure-Python data, so there are no thread-safety
or pickling issues with SQLite connections.

Name-tag handling
-----------------
Given ``--languages en,fr``:

* ``name``           → stored with lang ``'default'`` (fallback for all langs)
* ``name:en``        → stored with lang ``'en'``
* ``name:fr``        → stored with lang ``'fr'``
* ``alt_name``       → stored with lang ``'default'``, kind ``'alt'``
* ``alt_name:en``    → stored with lang ``'en'``, kind ``'alt'``
* Names with language suffixes *not* in the supported set are dropped.

Address-tag handling mirrors the same pattern for
``city``, ``state``, ``county``, ``district``, ``locality``, ``street``,
``country``, and ``other``.
"""

from __future__ import annotations

import gzip
import json
import logging
import multiprocessing as mp
import os
import sqlite3
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from .poly_filter import PolyFilter
from .schema import compress_json, create_database

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OSM name-tag prefixes → internal "kind" code.
_NAME_KINDS: Dict[str, str] = {
    "name": "name",
    "alt_name": "alt",
    "old_name": "old",
    "int_name": "int",
    "loc_name": "loc",
    "short_name": "short",
    "official_name": "official",
}

# Valid address component types (from the photon spec).
_ADDR_TYPES: Set[str] = {
    "country", "state", "county", "city",
    "district", "locality", "street", "other",
}

# Default batch size (number of raw JSON lines per worker task).
_DEFAULT_BATCH = 500

# SQLite write commit every N *places* (balance throughput vs. durability).
_COMMIT_EVERY = 10_000


# ---------------------------------------------------------------------------
# Worker-side helpers (must be importable at module level for pickling)
# ---------------------------------------------------------------------------

def _scalar(value: Any) -> Optional[str]:
    """Return a stripped string or ``None`` for list/None/empty values."""
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value else None


def _extract_names(
    name_dict: Optional[Dict],
    lang_set: Set[str],
) -> List[Tuple[str, str, str]]:
    """Extract ``(value, lang, kind)`` triples from a photon *name* dict.

    Parameters
    ----------
    name_dict:
        The ``name`` field of a photon place entry.
    lang_set:
        Set of ISO language codes the caller wants to keep.

    Returns
    -------
    Deduplicated list of ``(name_value, lang, kind)`` triples where
    *lang* is ``'default'`` for the bare OSM ``name`` tag and *kind*
    is one of the ``_NAME_KINDS`` values.
    """
    if not name_dict:
        return []

    results: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for key, raw_value in name_dict.items():
        value = _scalar(raw_value)
        if value is None:
            continue

        for prefix, kind in _NAME_KINDS.items():
            if key == prefix:
                entry = (value, "default", kind)
                if entry not in seen:
                    seen.add(entry)
                    results.append(entry)
                break
            if key.startswith(prefix + ":"):
                lang = key[len(prefix) + 1:]
                if lang in lang_set:
                    entry = (value, lang, kind)
                    if entry not in seen:
                        seen.add(entry)
                        results.append(entry)
                break

    return results


def _extract_addresses(
    addr_dict: Optional[Dict],
    lang_set: Set[str],
) -> List[Tuple[str, str, str]]:
    """Extract ``(addr_type, lang, value)`` triples from a photon *address* dict.

    Parameters
    ----------
    addr_dict:
        The ``address`` field of a photon place entry.
    lang_set:
        Set of ISO language codes the caller wants to keep.
    """
    if not addr_dict:
        return []

    results: List[Tuple[str, str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for key, raw_value in addr_dict.items():
        value = _scalar(raw_value)
        if value is None:
            continue

        if ":" in key:
            addr_type, lang = key.split(":", 1)
            if lang not in lang_set:
                continue
        else:
            addr_type = key
            lang = "default"

        if addr_type not in _ADDR_TYPES:
            continue

        entry = (addr_type, lang, value)
        if entry not in seen:
            seen.add(entry)
            results.append(entry)

    return results


def _parse_place_entry(
    entry: Dict,
    lang_set: Set[str],
    poly_filter: Optional[PolyFilter],
) -> Optional[Dict]:
    """Parse one place sub-entry from a photon ``Place`` object.

    Returns ``None`` if the entry is invalid or filtered out.
    """
    centroid = entry.get("centroid")
    if not centroid or len(centroid) < 2:
        return None

    try:
        lon = float(centroid[0])
        lat = float(centroid[1])
    except (TypeError, ValueError):
        return None

    if poly_filter is not None and not poly_filter.contains(lon, lat):
        return None

    # Bounding box – fall back to point when not provided.
    bbox = entry.get("bbox")
    if bbox and len(bbox) >= 4:
        try:
            bbox_min_lon = float(bbox[0])
            bbox_min_lat = float(bbox[1])
            bbox_max_lon = float(bbox[2])
            bbox_max_lat = float(bbox[3])
        except (TypeError, ValueError):
            bbox_min_lon = bbox_max_lon = lon
            bbox_min_lat = bbox_max_lat = lat
    else:
        bbox_min_lon = bbox_max_lon = lon
        bbox_min_lat = bbox_max_lat = lat

    names = _extract_names(entry.get("name"), lang_set)
    addresses = _extract_addresses(entry.get("address"), lang_set)

    raw_cats = entry.get("categories", [])
    categories: List[str] = [
        c for c in (raw_cats if isinstance(raw_cats, list) else [])
        if isinstance(c, str) and c.strip()
    ]

    extra = entry.get("extra")
    extra_blob: Optional[bytes] = compress_json(extra) if extra else None

    cc = entry.get("country_code")
    country_code: Optional[str] = cc.lower() if isinstance(cc, str) and cc else None

    importance_raw = entry.get("importance")
    try:
        importance = float(importance_raw) if importance_raw is not None else 0.0
    except (TypeError, ValueError):
        importance = 0.0

    return {
        "photon_id": str(entry.get("place_id", "")),
        "osm_type": entry.get("object_type"),
        "osm_id": entry.get("object_id"),
        "osm_key": entry.get("osm_key"),
        "osm_value": entry.get("osm_value"),
        "address_type": entry.get("address_type"),
        "importance": importance,
        "country_code": country_code,
        "postcode": _scalar(entry.get("postcode")),
        "housenumber": _scalar(entry.get("housenumber")),
        "lat": lat,
        "lon": lon,
        "bbox_min_lon": bbox_min_lon,
        "bbox_min_lat": bbox_min_lat,
        "bbox_max_lon": bbox_max_lon,
        "bbox_max_lat": bbox_max_lat,
        "extra": extra_blob,
        "names": names,
        "addresses": addresses,
        "categories": categories,
        "addresslines": entry.get("addresslines", []),
    }


def _process_chunk(args: Tuple) -> List[Dict]:
    """Worker entry point: parse a batch of raw JSON lines.

    Parameters (packed in *args* for ``Pool.imap`` compatibility)
    -------------------------------------------------------------
    lines : List[str]
    languages : List[str]
    poly_state : Optional[list]
        Serialised ``PolyFilter`` state (``None`` for no filter).
    """
    lines: List[str]
    languages: List[str]
    poly_state: Optional[list]
    lines, languages, poly_state = args

    poly_filter: Optional[PolyFilter] = None
    if poly_state is not None:
        pf = PolyFilter([])
        pf.__setstate__(poly_state)
        poly_filter = pf

    lang_set = set(languages)
    results: List[Dict] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        obj_type = obj.get("type")
        content = obj.get("content")

        if obj_type == "NominatimDumpFile":
            results.append({"type": "header", "data": content})

        elif obj_type == "CountryInfo":
            if isinstance(content, list):
                results.append({"type": "countries", "data": content})

        elif obj_type == "Place":
            if not isinstance(content, list):
                continue
            for entry in content:
                if not isinstance(entry, dict):
                    continue
                parsed = _parse_place_entry(entry, lang_set, poly_filter)
                if parsed is not None:
                    results.append({"type": "place", "data": parsed})

    return results


# ---------------------------------------------------------------------------
# Writer-side helpers
# ---------------------------------------------------------------------------

class _StringCache:
    """In-process cache mapping string values to their ``strings.id``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cache: Dict[str, int] = {}

    def get_ids(self, values: Sequence[str]) -> Dict[str, int]:
        """Return ``{value: id}`` for all *values*, inserting new ones."""
        missing = [v for v in values if v not in self._cache]
        if missing:
            self._conn.executemany(
                "INSERT OR IGNORE INTO strings(value) VALUES (?)",
                [(v,) for v in missing],
            )
            # SQLite's default SQLITE_LIMIT_VARIABLE_NUMBER is 999; stay
            # safely below it so we can pass N values in a single query.
            chunk = 900
            for i in range(0, len(missing), chunk):
                part = missing[i: i + chunk]
                ph = ",".join("?" * len(part))
                rows = self._conn.execute(
                    f"SELECT value, id FROM strings WHERE value IN ({ph})", part
                ).fetchall()
                for val, sid in rows:
                    self._cache[val] = sid
        return {v: self._cache[v] for v in values if v in self._cache}


class _CategoryCache:
    """In-process cache mapping category names to their ``categories.id``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cache: Dict[str, int] = {}

    def get_id(self, name: str) -> int:
        if name not in self._cache:
            self._conn.execute(
                "INSERT OR IGNORE INTO categories(name) VALUES (?)", (name,)
            )
            row = self._conn.execute(
                "SELECT id FROM categories WHERE name = ?", (name,)
            ).fetchone()
            if row:
                self._cache[name] = row[0]
        return self._cache[name]


class _CountryCache:
    """In-process cache mapping country codes to their ``countries.id``."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cache: Dict[str, int] = {}

    def get_id(self, code: str) -> Optional[int]:
        if not code:
            return None
        if code not in self._cache:
            self._conn.execute(
                "INSERT OR IGNORE INTO countries(code, names) VALUES (?, X'')",
                (code,),
            )
            row = self._conn.execute(
                "SELECT id FROM countries WHERE code = ?", (code,)
            ).fetchone()
            if row:
                self._cache[code] = row[0]
        return self._cache.get(code)

    def update_names(self, code: str, names_blob: bytes) -> None:
        self._conn.execute(
            "UPDATE countries SET names = ? WHERE code = ?",
            (names_blob, code),
        )


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------

def _write_place(
    conn: sqlite3.Connection,
    place: Dict,
    strings: _StringCache,
    cats: _CategoryCache,
    countries: _CountryCache,
    photon_to_db: Dict[str, Tuple[int, Optional[str], Optional[str]]],
    languages: List[str],
) -> None:
    """Insert one parsed place into all relevant tables."""
    country_id = countries.get_id(place["country_code"] or "")

    cur = conn.execute(
        """
        INSERT INTO places (
            photon_id, osm_type, osm_id, osm_key, osm_value,
            address_type, importance, country_id, postcode, housenumber,
            lat, lon, bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat,
            extra
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            place["photon_id"],
            place["osm_type"],
            place["osm_id"],
            place["osm_key"],
            place["osm_value"],
            place["address_type"],
            place["importance"],
            country_id,
            place["postcode"],
            place["housenumber"],
            place["lat"],
            place["lon"],
            place["bbox_min_lon"],
            place["bbox_min_lat"],
            place["bbox_max_lon"],
            place["bbox_max_lat"],
            place["extra"],
        ),
    )
    db_id: int = cur.lastrowid  # type: ignore[assignment]

    # R-tree entry.
    conn.execute(
        "INSERT INTO places_rtree VALUES (?,?,?,?,?)",
        (
            db_id,
            place["bbox_min_lat"],
            place["bbox_max_lat"],
            place["bbox_min_lon"],
            place["bbox_max_lon"],
        ),
    )

    # Resolve addresslines → extra address components.
    extra_addresses: List[Tuple[str, str, str]] = []
    for ref in place.get("addresslines", []):
        ref_id = str(ref.get("place_id", ""))
        if ref_id and ref_id in photon_to_db:
            _ref_db_id, ref_addr_type, ref_name = photon_to_db[ref_id]
            if ref.get("isaddress") and ref_addr_type and ref_name:
                extra_addresses.append((ref_addr_type, "default", ref_name))

    # Combine explicit addresses with addresslines-derived ones.
    all_addresses = list(place["addresses"]) + extra_addresses

    # --- String deduplication ------------------------------------------------
    all_name_values = [v for v, _l, _k in place["names"]]
    all_addr_values = [v for _t, _l, v in all_addresses]
    all_string_values = list(set(all_name_values + all_addr_values))

    if all_string_values:
        sid_map = strings.get_ids(all_string_values)
    else:
        sid_map = {}

    # --- place_names ---------------------------------------------------------
    if place["names"]:
        name_rows: List[Tuple] = []
        seen_pn: Set[Tuple] = set()
        for value, lang, kind in place["names"]:
            sid = sid_map.get(value)
            if sid is None:
                continue
            row = (db_id, sid, lang, kind)
            if row not in seen_pn:
                seen_pn.add(row)
                name_rows.append(row)
        if name_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO place_names(place_id, string_id, lang, kind) "
                "VALUES (?,?,?,?)",
                name_rows,
            )

    # --- place_addresses -----------------------------------------------------
    if all_addresses:
        addr_rows: List[Tuple] = []
        seen_pa: Set[Tuple] = set()
        for addr_type, lang, value in all_addresses:
            sid = sid_map.get(value)
            if sid is None:
                continue
            row = (db_id, sid, addr_type, lang)
            if row not in seen_pa:
                seen_pa.add(row)
                addr_rows.append(row)
        if addr_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO place_addresses"
                "(place_id, string_id, addr_type, lang) VALUES (?,?,?,?)",
                addr_rows,
            )

    # --- place_categories ----------------------------------------------------
    for cat in place["categories"]:
        cid = cats.get_id(cat)
        conn.execute(
            "INSERT OR IGNORE INTO place_categories(place_id, category_id) "
            "VALUES (?,?)",
            (db_id, cid),
        )

    # --- FTS5 ----------------------------------------------------------------
    fts_names = " ".join(
        sorted(set(v for v, _l, _k in place["names"]))
    )
    extra_address_parts = [v for _t, _l, v in extra_addresses]
    fts_address_parts = [v for _t, _l, v in place["addresses"]] + extra_address_parts
    if place.get("postcode"):
        fts_names += " " + place["postcode"]
    if place.get("housenumber"):
        fts_names += " " + place["housenumber"]
    fts_address = " ".join(sorted(set(fts_address_parts)))

    conn.execute(
        "INSERT INTO places_fts(place_id, names, address) VALUES (?,?,?)",
        (db_id, fts_names.strip(), fts_address.strip()),
    )

    # --- Update photon_to_db cache for addresslines resolution ---------------
    primary_name: Optional[str] = next(
        (v for v, l, k in place["names"] if l == "default" and k == "name"),
        next((v for v, _l, k in place["names"] if k == "name"), None),
    )
    photon_id = place["photon_id"]
    if photon_id:
        photon_to_db[photon_id] = (db_id, place["address_type"], primary_name)


def _write_countries(
    conn: sqlite3.Connection,
    country_list: List[Dict],
    countries: _CountryCache,
    languages: List[str],
) -> None:
    """Upsert CountryInfo records into the ``countries`` table."""
    lang_set = set(languages)
    for entry in country_list:
        code = entry.get("country_code", "").lower()
        if not code:
            continue
        raw_names: Dict = entry.get("name", {})
        # Filter to supported languages + default.
        filtered: Dict[str, str] = {}
        for k, v in raw_names.items():
            val = _scalar(v)
            if val is None:
                continue
            if k == "name":
                filtered["default"] = val
            elif ":" in k and k.split(":", 1)[1] in lang_set:
                filtered[k.split(":", 1)[1]] = val
        blob = gzip.compress(
            json.dumps(filtered, ensure_ascii=False).encode(), compresslevel=6
        )
        conn.execute(
            "INSERT INTO countries(code, names) VALUES (?,?) "
            "ON CONFLICT(code) DO UPDATE SET names = excluded.names",
            (code, blob),
        )
        row = conn.execute(
            "SELECT id FROM countries WHERE code = ?", (code,)
        ).fetchone()
        if row:
            countries._cache[code] = row[0]


# ---------------------------------------------------------------------------
# Line reader / batcher
# ---------------------------------------------------------------------------

def _open_input(path: str):
    """Return a text-line iterator over *path* (plain or .zst compressed)."""
    if path.endswith(".zst"):
        if zstd is None:
            raise RuntimeError(
                "The 'zstandard' package is required for .zst files. "
                "Install it with: pip install zstandard"
            )
        dctx = zstd.ZstdDecompressor()
        fh = open(path, "rb")
        reader = dctx.stream_reader(fh)
        import io
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace"), fh
    else:
        fh = open(path, "r", encoding="utf-8", errors="replace")
        return fh, fh


def _line_batches(
    text_iter,
    batch_size: int = _DEFAULT_BATCH,
) -> Iterator[List[str]]:
    """Yield lists of at most *batch_size* lines from *text_iter*."""
    batch: List[str] = []
    for line in text_iter:
        batch.append(line)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Public import entry point
# ---------------------------------------------------------------------------

def import_database(
    input_path: str,
    output_path: str,
    languages: List[str],
    poly_file: Optional[str] = None,
    num_workers: int = 0,
    batch_size: int = _DEFAULT_BATCH,
    show_progress: bool = True,
) -> None:
    """Import a Photon JSONL(.zst) dump into *output_path*.

    Parameters
    ----------
    input_path:
        Path to the photon dump file (``*.jsonl`` or ``*.jsonl.zst``).
    output_path:
        Path to the SQLite database to create/overwrite.
    languages:
        List of ISO 639-1 language codes to retain (e.g. ``['en', 'fr']``).
        The bare OSM ``name`` tag is always kept as ``'default'``.
    poly_file:
        Optional path to a ``.poly`` filter file.
    num_workers:
        Number of worker processes.  ``0`` → use all available CPUs.
    batch_size:
        Number of JSON lines per worker task.
    show_progress:
        Display a ``tqdm`` progress bar when *tqdm* is installed.
    """
    if not languages:
        raise ValueError("At least one language must be specified.")

    n_workers = num_workers if num_workers > 0 else max(1, (mp.cpu_count() or 1))
    log.info("Using %d worker process(es) + 1 writer (main)", n_workers)

    # Build poly filter serialisation (None → no filter).
    poly_state: Optional[list] = None
    if poly_file:
        pf = PolyFilter.from_file(poly_file)
        poly_state = pf.__getstate__()
        log.info("Loaded poly filter from %s (%d polygon(s))", poly_file, len(poly_state))

    conn = create_database(output_path)

    # Store import metadata.
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?,?)",
        ("languages", json.dumps(languages)),
    )
    conn.commit()

    strings = _StringCache(conn)
    cats = _CategoryCache(conn)
    countries = _CountryCache(conn)
    photon_to_db: Dict[str, Tuple[int, Optional[str], Optional[str]]] = {}

    total_places = 0
    total_skipped = 0

    text_iter, fh = _open_input(input_path)

    # Progress bar (optional).
    pbar = None
    if show_progress:
        if tqdm is not None:
            pbar = tqdm(unit=" places", desc="Importing", dynamic_ncols=True)
        else:
            log.info("tqdm not installed – no progress bar")

    try:
        batch_args = (
            (batch, languages, poly_state)
            for batch in _line_batches(text_iter, batch_size)
        )

        conn.execute("BEGIN")
        places_since_commit = 0

        with mp.Pool(processes=n_workers) as pool:
            for batch_results in pool.imap(_process_chunk, batch_args, chunksize=2):
                for item in batch_results:
                    item_type = item["type"]

                    if item_type == "header":
                        data = item["data"]
                        if isinstance(data, dict):
                            conn.execute(
                                "INSERT OR REPLACE INTO metadata(key, value) "
                                "VALUES (?,?)",
                                ("dump_version", data.get("version", "")),
                            )
                            conn.execute(
                                "INSERT OR REPLACE INTO metadata(key, value) "
                                "VALUES (?,?)",
                                ("generator", data.get("generator", "")),
                            )

                    elif item_type == "countries":
                        _write_countries(conn, item["data"], countries, languages)

                    elif item_type == "place":
                        try:
                            _write_place(
                                conn,
                                item["data"],
                                strings,
                                cats,
                                countries,
                                photon_to_db,
                                languages,
                            )
                            total_places += 1
                            places_since_commit += 1
                            if pbar is not None:
                                pbar.update(1)
                        except sqlite3.Error as exc:
                            log.debug("Skipping place due to DB error: %s", exc)
                            total_skipped += 1

                # Commit periodically to avoid huge transactions.
                if places_since_commit >= _COMMIT_EVERY:
                    conn.commit()
                    conn.execute("BEGIN")
                    places_since_commit = 0

        # Final commit.
        conn.commit()

    finally:
        if pbar is not None:
            pbar.close()
        try:
            text_iter.close()  # type: ignore[union-attr]
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass

    # Post-import optimisation.
    log.info("Running ANALYZE and VACUUM …")
    conn.execute("ANALYZE")
    conn.execute("VACUUM")
    conn.close()

    log.info(
        "Import complete: %d places inserted, %d skipped.",
        total_places,
        total_skipped,
    )
