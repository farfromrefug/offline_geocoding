"""Import a Photon JSONL.ZST dump into the offline geocoding SQLite database.

Parallel Architecture
---------------------
Reading the compressed input is inherently sequential, but JSON parsing and
filtering are CPU-bound.  We therefore use a **producer–consumer** pattern:

1. **Reader** (main process)
   - Opens and streams the ``.zst`` (or plain ``.jsonl``) file.
   - Groups raw text lines into *batches* and puts them on a bounded
     ``multiprocessing.Queue``.

2. **Worker processes** (N, one per available CPU by default)
   - Each worker owns its own private SQLite *worker database*.
   - Workers consume batches from the queue, parse JSON, apply the poly
     filter, and write directly to their local database.
   - Because each worker writes to a separate file there is **no write
     contention** – the bottleneck is eliminated.

3. **Merge** (main process, after all workers finish)
   - The main process opens the final database and ATTACHes each worker
     database in turn.
   - Strings and categories are inserted with ``INSERT OR IGNORE`` (the
     UNIQUE constraint deduplicates by value/name).
   - Temporary mapping tables (``_smap``, ``_cmap``, ``_pmap``) translate
     worker-local IDs to the globally assigned IDs in the final database.
   - Place names, addresses, categories, FTS data, and R-tree entries are
     inserted using JOIN on those mapping tables.

ID consistency across databases
--------------------------------
``langs``, ``name_kinds``, and ``addr_types`` are pre-seeded with
**fixed IDs** at database creation time.  Because all worker databases and
the final database are created with the same language list (and therefore the
same ``build_lang_ids`` result), these IDs are identical everywhere — no
remapping is needed for them during the merge step.

Only ``strings`` (via ``_smap``) and ``categories`` (via ``_cmap``) need
ID remapping.  ``places.osm_key_id``, ``places.osm_value_id``,
``places.postcode_id``, and ``places.hn_id`` are string IDs and are also
remapped via ``_smap``.

R-tree centroid-only storage
-----------------------------
``rtree_data`` staging table stores only the centroid ``(id, lat, lon)``.
During merge the centroid is inserted as a degenerate R-tree bounding box
``(min_lat=lat, max_lat=lat, min_lon=lon, max_lon=lon)``, which always
satisfies the R-tree constraint ``min <= max``.  Storing only the centroid
and ignoring the bbox entirely eliminates the infamous
``IntegrityError: rtree constraint failed: places_rtree.(min_lat<=max_lat)``
that can occur when OSM bbox data contains swapped or equal coordinates.

FTS5 contentless table
-----------------------
The final database uses ``content=''`` (contentless) FTS5.  Only the trigram
index is stored — the text content is NOT persisted in FTS5 shadow tables.
Queries retrieve the matching ``rowid`` (which equals ``place_id``) and join
``places``.  Workers still use a plain ``fts_data`` staging table for
efficient aggregation.
"""

from __future__ import annotations

import gzip
import json
import logging
import multiprocessing as mp
import os
import sqlite3
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from .poly_filter import PolyFilter
from .schema import (
    ADDR_TYPE_IDS,
    IMPORTANCE_SCALE,
    LAT_LON_SCALE,
    NAME_KIND_IDS,
    build_lang_ids,
    compress_json,
    create_database,
    create_indexes,
    create_worker_database,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OSM name-tag prefixes -> internal "kind" code (must match NAME_KIND_IDS).
_NAME_KINDS: Dict[str, str] = {
    "name":          "name",
    "alt_name":      "alt",
    "old_name":      "old",
    "int_name":      "int",
    "loc_name":      "loc",
    "short_name":    "short",
    "official_name": "official",
}

# Valid address component types (from the photon spec, must match ADDR_TYPE_IDS).
_ADDR_TYPES: Set[str] = set(ADDR_TYPE_IDS.keys())

# Default batch size (number of raw JSON lines per dispatch).
_DEFAULT_BATCH = 2_000

# Number of places between transaction commits within a worker.
_COMMIT_EVERY = 50_000

# Sentinel value put on the work queue to tell a worker to shut down.
_SENTINEL = None

# Each worker is allocated this many consecutive place IDs.
_PLACES_PER_WORKER = 50_000_000

# Maximum number of parameters in a single SQLite statement.
_SQL_PARAM_CHUNK = 900


# ---------------------------------------------------------------------------
# Pure parsing helpers (called inside worker processes)
# ---------------------------------------------------------------------------

def _scalar(value: Any) -> Optional[str]:
    """Return a stripped string, or ``None`` for list/None/empty."""
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
    """Extract ``(value, lang, kind)`` triples from a photon *name* dict."""
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
    """Extract ``(addr_type, lang, value)`` triples from a photon *address* dict."""
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
        "photon_id":    str(entry.get("place_id", "")),
        "osm_id":       entry.get("object_id"),
        "osm_key":      _scalar(entry.get("osm_key")),
        "osm_value":    _scalar(entry.get("osm_value")),
        "address_type": _scalar(entry.get("address_type")),
        "importance":   importance,
        "country_code": country_code,
        "postcode":     _scalar(entry.get("postcode")),
        "housenumber":  _scalar(entry.get("housenumber")),
        "lat":          lat,
        "lon":          lon,
        "extra":        extra_blob,
        "names":        names,
        "addresses":    addresses,
        "categories":   categories,
        "addresslines": entry.get("addresslines", []),
    }


# ---------------------------------------------------------------------------
# Worker-side string / category cache
# ---------------------------------------------------------------------------

class _LocalStringCache:
    """Cache string -> local integer id within a worker database."""

    __slots__ = ("_conn", "_cache")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cache: Dict[str, int] = {}

    def get_id(self, value: str) -> int:
        """Intern a single string and return its local id."""
        sid = self._cache.get(value)
        if sid is None:
            self._conn.execute(
                "INSERT OR IGNORE INTO strings(value) VALUES (?)", (value,)
            )
            row = self._conn.execute(
                "SELECT id FROM strings WHERE value = ?", (value,)
            ).fetchone()
            sid = row[0]
            self._cache[value] = sid
        return sid

    def intern_batch(self, values: List[str]) -> None:
        """Intern a list of strings in bulk."""
        missing = [v for v in values if v not in self._cache]
        if not missing:
            return

        self._conn.executemany(
            "INSERT OR IGNORE INTO strings(value) VALUES (?)",
            ((v,) for v in missing),
        )

        for i in range(0, len(missing), _SQL_PARAM_CHUNK):
            chunk = missing[i : i + _SQL_PARAM_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT value, id FROM strings WHERE value IN ({placeholders})",
                chunk,
            ).fetchall()
            for val, sid in rows:
                self._cache[val] = sid


class _LocalCategoryCache:
    """Cache category name -> local integer id within a worker database."""

    __slots__ = ("_conn", "_cache")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cache: Dict[str, int] = {}

    def get_id(self, name: str) -> int:
        cid = self._cache.get(name)
        if cid is None:
            self._conn.execute(
                "INSERT OR IGNORE INTO categories(name) VALUES (?)", (name,)
            )
            row = self._conn.execute(
                "SELECT id FROM categories WHERE name = ?", (name,)
            ).fetchone()
            cid = row[0]
            self._cache[name] = cid
        return cid

    def intern_batch(self, names: List[str]) -> None:
        """Intern a list of category names in bulk."""
        missing = [n for n in names if n not in self._cache]
        if not missing:
            return
        self._conn.executemany(
            "INSERT OR IGNORE INTO categories(name) VALUES (?)",
            ((n,) for n in missing),
        )
        for i in range(0, len(missing), _SQL_PARAM_CHUNK):
            chunk = missing[i : i + _SQL_PARAM_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT name, id FROM categories WHERE name IN ({placeholders})",
                chunk,
            ).fetchall()
            for cat_name, cid in rows:
                self._cache[cat_name] = cid


# ---------------------------------------------------------------------------
# Worker-side write helpers
# ---------------------------------------------------------------------------

def _write_countries_worker(
    conn: sqlite3.Connection,
    country_list: List[Dict],
    lang_set: Set[str],
    lang_ids: Dict[str, int],
    strings: _LocalStringCache,
) -> None:
    """Upsert CountryInfo records into the worker ``countries`` + ``country_names`` tables."""
    for entry in country_list:
        code = _scalar(entry.get("country_code")) or ""
        code = code.lower()
        if not code:
            continue

        # Ensure country row exists.
        conn.execute(
            "INSERT OR IGNORE INTO countries(code) VALUES (?)",
            (code,),
        )

        raw_names: Dict = entry.get("name", {}) or {}
        filtered: Dict[str, str] = {}
        for k, v in raw_names.items():
            val = _scalar(v)
            if val is None:
                continue
            if k == "name":
                filtered["default"] = val
            elif ":" in k and k.split(":", 1)[1] in lang_set:
                filtered[k.split(":", 1)[1]] = val

        if not filtered:
            continue

        # Intern name strings in bulk.
        strings.intern_batch(list(filtered.values()))

        # Insert one row per language into country_names.
        for lang, name_val in filtered.items():
            lid = lang_ids.get(lang)
            sid = strings._cache.get(name_val)
            if lid is None or sid is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO country_names(code, string_id, lang_id)"
                " VALUES (?,?,?)",
                (code, sid, lid),
            )


def _write_places_batch(
    conn: sqlite3.Connection,
    parsed_places: List[Dict],
    strings: _LocalStringCache,
    cats: _LocalCategoryCache,
    photon_to_db: Dict[str, Tuple[int, Optional[str], Optional[str]]],
    lang_ids: Dict[str, int],
    staging: bool = True,
) -> None:
    """Bulk-write a list of already-parsed places into the worker database."""
    if not parsed_places:
        return

    # ---- 0. Resolve addresslines ------------------------------------------
    all_addresses_per_place: List[List[Tuple[str, str, str]]] = []
    for place in parsed_places:
        extra_addresses: List[Tuple[str, str, str]] = []
        for ref in place.get("addresslines", []):
            ref_pid = str(ref.get("place_id", ""))
            if ref_pid and ref_pid in photon_to_db:
                _ref_db_id, ref_addr_type, ref_name = photon_to_db[ref_pid]
                if ref.get("isaddress") and ref_addr_type and ref_name:
                    extra_addresses.append((ref_addr_type, "default", ref_name))
        all_addresses = list(place["addresses"]) + extra_addresses
        all_addresses_per_place.append(all_addresses)

        primary_name: Optional[str] = next(
            (v for v, l, k in place["names"] if l == "default" and k == "name"),
            next((v for v, _l, k in place["names"] if k == "name"), None),
        )
        photon_id = place["photon_id"]
        if photon_id:
            photon_to_db[photon_id] = (
                place["_place_id"],
                place["address_type"],
                primary_name,
            )

    # ---- 1. Ensure country codes exist (placeholder rows) ------------------
    country_codes: Set[str] = {
        p["country_code"]
        for p in parsed_places
        if p["country_code"] is not None
    }
    if country_codes:
        conn.executemany(
            "INSERT OR IGNORE INTO countries(code) VALUES (?)",
            ((code,) for code in country_codes),
        )

    # ---- 2. Bulk-intern all strings and categories -------------------------
    all_strings: List[str] = []
    all_cat_names: List[str] = []
    for place, all_addresses in zip(parsed_places, all_addresses_per_place):
        all_strings.extend(v for v, _l, _k in place["names"])
        all_strings.extend(v for _t, _l, v in all_addresses)
        # Also intern the place's own TEXT fields that become string IDs.
        for fld in ("osm_key", "osm_value", "postcode", "housenumber"):
            val = place.get(fld)
            if val:
                all_strings.append(val)
        all_cat_names.extend(place["categories"])

    strings.intern_batch(list(dict.fromkeys(all_strings)))
    cats.intern_batch(list(dict.fromkeys(all_cat_names)))

    # ---- 3. Bulk-insert place rows ----------------------------------------
    place_rows: List[Tuple] = []
    for place in parsed_places:
        osm_key_id   = strings._cache.get(place["osm_key"])   if place.get("osm_key")   else None
        osm_value_id = strings._cache.get(place["osm_value"]) if place.get("osm_value") else None
        addr_type_id = ADDR_TYPE_IDS.get(place["address_type"]) if place.get("address_type") else None
        postcode_id  = strings._cache.get(place["postcode"])   if place.get("postcode")   else None
        hn_id        = strings._cache.get(place["housenumber"]) if place.get("housenumber") else None

        place_rows.append((
            place["_place_id"],
            place["photon_id"],
            place["osm_id"],
            osm_key_id,
            osm_value_id,
            addr_type_id,
            round(place["importance"] * IMPORTANCE_SCALE),
            place["country_code"],
            postcode_id,
            hn_id,
            round(place["lat"] * LAT_LON_SCALE),
            round(place["lon"] * LAT_LON_SCALE),
            place["extra"],
        ))
    conn.executemany(
        """
        INSERT OR IGNORE INTO places (
            id, photon_id, osm_id, osm_key_id, osm_value_id,
            addr_type_id, importance, country_code, postcode_id, hn_id,
            lat, lon, extra
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        place_rows,
    )

    # ---- 4. Bulk-insert child rows ----------------------------------------
    name_rows:  List[Tuple] = []
    addr_rows:  List[Tuple] = []
    cat_rows:   List[Tuple] = []
    fts_rows:   List[Tuple] = []
    rtree_rows: List[Tuple] = []

    seen_pn: Set[Tuple] = set()
    seen_pa: Set[Tuple] = set()

    for place, all_addresses in zip(parsed_places, all_addresses_per_place):
        pid = place["_place_id"]

        for value, lang, kind in place["names"]:
            sid    = strings._cache.get(value)
            lid    = lang_ids.get(lang)
            kid    = NAME_KIND_IDS.get(kind)
            if sid is None or lid is None or kid is None:
                continue
            row = (pid, sid, lid, kid)
            if row not in seen_pn:
                seen_pn.add(row)
                name_rows.append(row)

        for addr_type, lang, value in all_addresses:
            sid   = strings._cache.get(value)
            at_id = ADDR_TYPE_IDS.get(addr_type)
            lid   = lang_ids.get(lang)
            if sid is None or at_id is None or lid is None:
                continue
            row = (pid, sid, at_id, lid)
            if row not in seen_pa:
                seen_pa.add(row)
                addr_rows.append(row)

        for cat in place["categories"]:
            cid = cats._cache.get(cat)
            if cid is None:
                continue
            cat_rows.append((pid, cid))

        fts_names = " ".join(sorted({v for v, _l, _k in place["names"]}))
        if place.get("postcode"):
            fts_names += " " + place["postcode"]
        if place.get("housenumber"):
            fts_names += " " + place["housenumber"]
        fts_addr = " ".join(sorted({v for _t, _l, v in all_addresses}))
        fts_rows.append((pid, fts_names.strip(), fts_addr.strip()))

        # Centroid only – no bbox.  min == max satisfies the R-tree constraint.
        rtree_rows.append((
            pid,
            round(place["lat"] * LAT_LON_SCALE),
            round(place["lon"] * LAT_LON_SCALE),
        ))

    if name_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO place_names(place_id, string_id, lang_id, kind_id)"
            " VALUES (?,?,?,?)",
            name_rows,
        )
    if addr_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO place_addresses"
            "(place_id, string_id, addr_type_id, lang_id) VALUES (?,?,?,?)",
            addr_rows,
        )
    if cat_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO place_categories(place_id, category_id)"
            " VALUES (?,?)",
            cat_rows,
        )
    if fts_rows:
        if staging:
            conn.executemany(
                "INSERT OR REPLACE INTO fts_data(place_id, names, address)"
                " VALUES (?,?,?)",
                fts_rows,
            )
        else:
            # Contentless FTS5: rowid = place_id.
            conn.executemany(
                "INSERT INTO places_fts(rowid, names, address) VALUES (?,?,?)",
                fts_rows,
            )
    if rtree_rows:
        if staging:
            conn.executemany(
                "INSERT OR REPLACE INTO rtree_data(id, lat, lon) VALUES (?,?,?)",
                rtree_rows,
            )
        else:
            # Centroid point: min == max for both lat and lon.
            scale = float(LAT_LON_SCALE)
            conn.executemany(
                "INSERT INTO places_rtree(id, min_lat, max_lat, min_lon, max_lon)"
                " VALUES (?,?,?,?,?)",
                [
                    (r[0], r[1] / scale, r[1] / scale, r[2] / scale, r[2] / scale)
                    for r in rtree_rows
                ],
            )


# ---------------------------------------------------------------------------
# Worker process entry point
# ---------------------------------------------------------------------------

def _worker_main(
    work_queue,
    db_path: str,
    languages: List[str],
    poly_state: Optional[list],
    worker_id: int,
    place_id_base: int,
) -> None:
    """Worker process: drain *work_queue* and write parsed results to *db_path*."""
    poly_filter: Optional[PolyFilter] = None
    if poly_state is not None:
        pf = PolyFilter([])
        pf.__setstate__(poly_state)
        poly_filter = pf

    lang_set: Set[str] = set(languages)
    lang_ids: Dict[str, int] = build_lang_ids(languages)

    conn = create_worker_database(db_path, languages)
    strings = _LocalStringCache(conn)
    cats = _LocalCategoryCache(conn)
    photon_to_db: Dict[str, Tuple[int, Optional[str], Optional[str]]] = {}
    local_counter = 0

    conn.execute("BEGIN")

    while True:
        batch = work_queue.get()
        if batch is _SENTINEL:
            break

        batch_country_infos: List[List[Dict]] = []
        parsed_places: List[Dict] = []

        for raw_line in batch:
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
                if isinstance(content, dict):
                    conn.execute(
                        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?,?)",
                        ("dump_version", content.get("version", "")),
                    )

            elif obj_type == "CountryInfo":
                if isinstance(content, list):
                    batch_country_infos.append(content)

            elif obj_type == "Place":
                if not isinstance(content, list):
                    continue
                for entry in content:
                    if not isinstance(entry, dict):
                        continue
                    parsed = _parse_place_entry(entry, lang_set, poly_filter)
                    if parsed is None:
                        continue
                    place_id = place_id_base + local_counter + 1
                    local_counter += 1
                    parsed["_place_id"] = place_id
                    parsed_places.append(parsed)

        for country_list in batch_country_infos:
            _write_countries_worker(conn, country_list, lang_set, lang_ids, strings)

        if parsed_places:
            try:
                _write_places_batch(
                    conn, parsed_places, strings, cats, photon_to_db, lang_ids
                )
            except sqlite3.Error as exc:
                log.warning("Worker %d batch write failed: %s", worker_id, exc)
                for place in parsed_places:
                    try:
                        _write_places_batch(
                            conn, [place], strings, cats, photon_to_db, lang_ids
                        )
                    except sqlite3.Error as exc2:
                        log.debug(
                            "Worker %d skipping place %s: %s",
                            worker_id, place.get("photon_id"), exc2,
                        )

        if local_counter % _COMMIT_EVERY == 0 and local_counter > 0:
            conn.execute("COMMIT")
            conn.execute("BEGIN")

    conn.execute("COMMIT")
    conn.close()
    log.debug("Worker %d done: %d places", worker_id, local_counter)


# ---------------------------------------------------------------------------
# Merge step
# ---------------------------------------------------------------------------

def _merge_worker_db(
    final_conn: sqlite3.Connection,
    worker_path: str,
) -> int:
    """Merge one worker database into the final database.

    Uses ATTACH DATABASE and SQL-driven temp mapping tables so that:

    * ``strings`` and ``categories`` are globally deduplicated.
    * ``countries`` are inserted by TEXT primary key (code); ``country_names``
      are upserted using remapped string IDs.
    * ``places`` receive new globally-sequential IDs via the ROW_NUMBER trick.
    * All dependent tables are inserted with remapped IDs via JOIN.
    * R-tree entries are inserted as centroid points (min == max) to satisfy
      the SQLite R-tree constraint unconditionally.

    Returns the number of places merged from this worker.
    """
    final_conn.execute("PRAGMA foreign_keys = OFF")
    final_conn.execute("ATTACH DATABASE ? AS w", (worker_path,))
    try:
        final_conn.execute("BEGIN")

        # 1. Strings - INSERT OR IGNORE deduplicates by value.
        final_conn.execute(
            "INSERT OR IGNORE INTO strings(value) SELECT value FROM w.strings"
        )

        # 2. String ID mapping: worker local id -> final global id.
        final_conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _smap(wid INTEGER, fid INTEGER)"
        )
        final_conn.execute("DELETE FROM _smap")
        final_conn.execute(
            """
            INSERT INTO _smap(wid, fid)
            SELECT ws.id, ms.id
            FROM w.strings ws
            JOIN main.strings ms ON ms.value = ws.value
            """
        )
        final_conn.execute(
            "CREATE INDEX IF NOT EXISTS _idx_smap_wid ON _smap(wid)"
        )

        # 3. Categories - same pattern.
        final_conn.execute(
            "INSERT OR IGNORE INTO categories(name) SELECT name FROM w.categories"
        )
        final_conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _cmap(wid INTEGER, fid INTEGER)"
        )
        final_conn.execute("DELETE FROM _cmap")
        final_conn.execute(
            """
            INSERT INTO _cmap(wid, fid)
            SELECT wc.id, mc.id
            FROM w.categories wc
            JOIN main.categories mc ON mc.name = wc.name
            """
        )
        final_conn.execute(
            "CREATE INDEX IF NOT EXISTS _idx_cmap_wid ON _cmap(wid)"
        )

        # 4. Countries - INSERT OR IGNORE by TEXT primary key (no remap needed).
        final_conn.execute(
            "INSERT OR IGNORE INTO countries(code) SELECT code FROM w.countries"
        )

        # 5. Metadata - upsert.
        final_conn.execute(
            """
            INSERT OR REPLACE INTO metadata(key, value)
            SELECT key, value FROM w.metadata
            """
        )

        # 6. Ensure all country_codes referenced by the worker's places exist.
        final_conn.execute(
            """
            INSERT OR IGNORE INTO countries(code)
            SELECT DISTINCT country_code
            FROM w.places
            WHERE country_code IS NOT NULL
            """
        )

        # 7. Country names - remap string_id via _smap; lang_id is pre-seeded (no remap).
        #    OR REPLACE to overwrite placeholder-only rows with real names.
        final_conn.execute(
            """
            INSERT OR REPLACE INTO country_names(code, string_id, lang_id)
            SELECT cn.code, sm.fid, cn.lang_id
            FROM w.country_names cn
            JOIN _smap sm ON sm.wid = cn.string_id
            """
        )

        # 8. Places - insert without explicit id so SQLite assigns new sequential IDs.
        base_row = final_conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM places"
        ).fetchone()
        base_id: int = base_row[0]

        worker_count_row = final_conn.execute(
            "SELECT COUNT(*) FROM w.places"
        ).fetchone()
        worker_place_count: int = worker_count_row[0]

        if worker_place_count > 0:
            # Remap string IDs for osm_key_id, osm_value_id, postcode_id, hn_id.
            # addr_type_id is pre-seeded (same IDs everywhere) – no remap.
            # lang_id / kind_id are also pre-seeded – no remap.
            final_conn.execute(
                """
                INSERT INTO places(
                    photon_id, osm_id, osm_key_id, osm_value_id, addr_type_id,
                    importance, country_code, postcode_id, hn_id, lat, lon, extra
                )
                SELECT
                    wp.photon_id,
                    wp.osm_id,
                    sk.fid,
                    sv.fid,
                    wp.addr_type_id,
                    wp.importance,
                    wp.country_code,
                    sp.fid,
                    sh.fid,
                    wp.lat,
                    wp.lon,
                    wp.extra
                FROM w.places wp
                LEFT JOIN _smap sk ON sk.wid = wp.osm_key_id
                LEFT JOIN _smap sv ON sv.wid = wp.osm_value_id
                LEFT JOIN _smap sp ON sp.wid = wp.postcode_id
                LEFT JOIN _smap sh ON sh.wid = wp.hn_id
                ORDER BY wp.id
                """
            )

            # 9. Place ID mapping using ROW_NUMBER.
            final_conn.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _pmap(wid INTEGER, fid INTEGER)"
            )
            final_conn.execute("DELETE FROM _pmap")
            final_conn.execute(
                f"""
                WITH src AS (
                    SELECT id AS wid,
                           ROW_NUMBER() OVER (ORDER BY id) AS rn
                    FROM w.places
                )
                INSERT INTO _pmap(wid, fid)
                SELECT src.wid, {base_id} + src.rn
                FROM src
                """
            )
            final_conn.execute(
                "CREATE INDEX IF NOT EXISTS _idx_pmap_wid ON _pmap(wid)"
            )

            # 10. place_names - remap place_id and string_id.
            #     lang_id and kind_id are pre-seeded (no remap).
            final_conn.execute(
                """
                INSERT OR IGNORE INTO place_names(place_id, string_id, lang_id, kind_id)
                SELECT pm.fid, sm.fid, pn.lang_id, pn.kind_id
                FROM w.place_names pn
                JOIN _pmap pm ON pm.wid = pn.place_id
                JOIN _smap sm ON sm.wid = pn.string_id
                """
            )

            # 11. place_addresses - remap place_id and string_id.
            #     addr_type_id and lang_id are pre-seeded (no remap).
            final_conn.execute(
                """
                INSERT OR IGNORE INTO place_addresses(place_id, string_id, addr_type_id, lang_id)
                SELECT pm.fid, sm.fid, pa.addr_type_id, pa.lang_id
                FROM w.place_addresses pa
                JOIN _pmap pm ON pm.wid = pa.place_id
                JOIN _smap sm ON sm.wid = pa.string_id
                """
            )

            # 12. place_categories - remap place_id and category_id.
            final_conn.execute(
                """
                INSERT OR IGNORE INTO place_categories(place_id, category_id)
                SELECT pm.fid, cm.fid
                FROM w.place_categories pc
                JOIN _pmap pm ON pm.wid = pc.place_id
                JOIN _cmap cm ON cm.wid = pc.category_id
                """
            )

            # 13. FTS5 (contentless) - rowid = place_id.
            final_conn.execute(
                """
                INSERT INTO places_fts(rowid, names, address)
                SELECT pm.fid, fd.names, fd.address
                FROM w.fts_data fd
                JOIN _pmap pm ON pm.wid = fd.place_id
                """
            )

            # 14. R-tree - centroid point: min == max for both lat and lon.
            #     This always satisfies the R-tree constraint (min <= max).
            final_conn.execute(
                f"""
                INSERT INTO places_rtree(id, min_lat, max_lat, min_lon, max_lon)
                SELECT pm.fid,
                       rd.lat * 1.0 / {LAT_LON_SCALE},
                       rd.lat * 1.0 / {LAT_LON_SCALE},
                       rd.lon * 1.0 / {LAT_LON_SCALE},
                       rd.lon * 1.0 / {LAT_LON_SCALE}
                FROM w.rtree_data rd
                JOIN _pmap pm ON pm.wid = rd.id
                """
            )

        final_conn.execute("COMMIT")

    except Exception:
        try:
            final_conn.execute("ROLLBACK")
        except Exception:
            pass
        raise

    finally:
        try:
            final_conn.execute("DETACH DATABASE w")
        except Exception:
            pass
        final_conn.execute("PRAGMA foreign_keys = ON")

    return worker_place_count


# ---------------------------------------------------------------------------
# Input reader helpers
# ---------------------------------------------------------------------------

def _open_input(path: str):
    """Return ``(text_iter, raw_fh)`` for *path* (plain or .zst compressed)."""
    if path.endswith(".zst"):
        if zstd is None:
            raise RuntimeError(
                "The 'zstandard' package is required for .zst files. "
                "Install it with: pip install zstandard"
            )
        dctx = zstd.ZstdDecompressor()
        fh = open(path, "rb")
        import io
        reader = dctx.stream_reader(fh)
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
# Single-threaded import (no worker sub-processes, no merge)
# ---------------------------------------------------------------------------

def _import_single_thread(
    input_path: str,
    output_path: str,
    languages: List[str],
    poly_file: Optional[str],
    batch_size: int,
    show_progress: bool,
) -> None:
    """Import directly into the final database on a single thread."""
    poly_filter: Optional[PolyFilter] = None
    if poly_file:
        pf = PolyFilter.from_file(poly_file)
        poly_filter = pf
        log.info(
            "Loaded poly filter from %s (%d polygon(s))",
            poly_file,
            len(pf.__getstate__()),
        )

    lang_set: Set[str] = set(languages)
    lang_ids: Dict[str, int] = build_lang_ids(languages)

    if os.path.exists(output_path):
        os.unlink(output_path)

    conn = create_database(output_path, languages=languages, with_indexes=False)

    conn.execute("BEGIN")
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?,?)",
        ("languages", json.dumps(languages)),
    )
    conn.execute("COMMIT")

    strings = _LocalStringCache(conn)
    cats = _LocalCategoryCache(conn)
    photon_to_db: Dict[str, Tuple[int, Optional[str], Optional[str]]] = {}
    total_places = 0

    conn.execute("BEGIN")

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(unit=" batches", desc="Reading+parsing", dynamic_ncols=True)

    text_iter, raw_fh = _open_input(input_path)
    try:
        for batch in _line_batches(text_iter, batch_size):
            batch_country_infos: List[List[Dict]] = []
            parsed_places: List[Dict] = []

            for raw_line in batch:
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
                    if isinstance(content, dict):
                        conn.execute(
                            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?,?)",
                            ("dump_version", content.get("version", "")),
                        )

                elif obj_type == "CountryInfo":
                    if isinstance(content, list):
                        batch_country_infos.append(content)

                elif obj_type == "Place":
                    if not isinstance(content, list):
                        continue
                    for entry in content:
                        if not isinstance(entry, dict):
                            continue
                        parsed = _parse_place_entry(entry, lang_set, poly_filter)
                        if parsed is None:
                            continue
                        total_places += 1
                        parsed["_place_id"] = total_places
                        parsed_places.append(parsed)

            for country_list in batch_country_infos:
                _write_countries_worker(conn, country_list, lang_set, lang_ids, strings)

            if parsed_places:
                try:
                    _write_places_batch(
                        conn, parsed_places, strings, cats, photon_to_db,
                        lang_ids, staging=False,
                    )
                except sqlite3.Error as exc:
                    log.warning("Batch write failed: %s", exc)
                    for place in parsed_places:
                        try:
                            _write_places_batch(
                                conn, [place], strings, cats, photon_to_db,
                                lang_ids, staging=False,
                            )
                        except sqlite3.Error as exc2:
                            log.debug(
                                "Skipping place %s: %s",
                                place.get("photon_id"), exc2,
                            )

            if pbar is not None:
                pbar.update(1)

            if total_places % _COMMIT_EVERY == 0 and total_places > 0:
                conn.execute("COMMIT")
                conn.execute("BEGIN")

    finally:
        if pbar is not None:
            pbar.close()
        try:
            text_iter.close()
        except Exception:
            pass
        try:
            raw_fh.close()
        except Exception:
            pass

    conn.execute("COMMIT")

    log.info("Creating indexes...")
    create_indexes(conn)

    log.info("Running ANALYZE...")
    conn.execute("ANALYZE")
    conn.close()

    log.info("Import complete (single-thread): %d places imported.", total_places)


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
    single_thread: bool = False,
) -> None:
    """Import a Photon JSONL(.zst) dump into *output_path*."""
    if not languages:
        raise ValueError("At least one language must be specified.")

    if single_thread:
        log.info("Single-thread mode: running import without worker processes.")
        _import_single_thread(
            input_path=input_path,
            output_path=output_path,
            languages=languages,
            poly_file=poly_file,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        return

    n_workers = num_workers if num_workers > 0 else max(1, (mp.cpu_count() or 1))
    log.info("Using %d worker process(es)", n_workers)

    poly_state: Optional[list] = None
    if poly_file:
        pf = PolyFilter.from_file(poly_file)
        poly_state = pf.__getstate__()
        log.info(
            "Loaded poly filter from %s (%d polygon(s))",
            poly_file,
            len(poly_state),
        )

    out_dir = os.path.dirname(os.path.abspath(output_path))
    worker_db_paths: List[str] = [
        os.path.join(out_dir, f"_worker_{i}_tmp.db")
        for i in range(n_workers)
    ]

    for p in worker_db_paths:
        if os.path.exists(p):
            os.unlink(p)

    # ---------- Phase 1: parallel import into worker databases ---------------

    work_queue: mp.Queue = mp.Queue(maxsize=n_workers * 4)

    workers: List[mp.Process] = []
    for i in range(n_workers):
        p = mp.Process(
            target=_worker_main,
            args=(
                work_queue,
                worker_db_paths[i],
                languages,
                poly_state,
                i,
                i * _PLACES_PER_WORKER,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    text_iter, raw_fh = _open_input(input_path)
    batches_sent = 0
    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(unit=" batches", desc="Reading+parsing", dynamic_ncols=True)

    try:
        for batch in _line_batches(text_iter, batch_size):
            work_queue.put(batch)
            batches_sent += 1
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
        try:
            text_iter.close()
        except Exception:
            pass
        try:
            raw_fh.close()
        except Exception:
            pass

    for _ in workers:
        work_queue.put(_SENTINEL)

    log.info("All %d batches dispatched, waiting for workers...", batches_sent)
    for p in workers:
        p.join()
    log.info("All workers finished.")

    # ---------- Phase 2: merge worker databases into final database ----------

    if os.path.exists(output_path):
        os.unlink(output_path)
    final_conn = create_database(output_path, languages=languages)

    final_conn.execute("BEGIN")
    final_conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?,?)",
        ("languages", json.dumps(languages)),
    )
    final_conn.execute("COMMIT")

    total_places = 0
    for i, worker_path in enumerate(worker_db_paths):
        if not os.path.exists(worker_path):
            log.warning("Worker %d database not found: %s", i, worker_path)
            continue
        log.info("Merging worker %d database (%s)...", i, worker_path)
        n = _merge_worker_db(final_conn, worker_path)
        total_places += n
        log.info("  merged %d places (running total: %d)", n, total_places)

    log.info("Running ANALYZE...")
    final_conn.execute("ANALYZE")
    final_conn.close()

    for p in worker_db_paths:
        try:
            if os.path.exists(p):
                os.unlink(p)
        except OSError as exc:
            log.warning("Could not remove worker DB %s: %s", p, exc)

    log.info("Import complete: %d places imported.", total_places)
