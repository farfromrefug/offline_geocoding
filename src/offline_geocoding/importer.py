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
     inserted using JOIN on those mapping tables – ensuring referential
     integrity and complete cross-worker deduplication.

Name-tag handling
-----------------
Given ``--languages en,fr``:

* ``name``          -> lang ``'default'`` (OSM bare name, always kept)
* ``name:en``       -> lang ``'en'``
* ``name:fr``       -> lang ``'fr'``
* ``alt_name``      -> lang ``'default'``, kind ``'alt'``
* ``alt_name:en``   -> lang ``'en'``, kind ``'alt'``
* Other language suffixes are dropped.

Address-tag handling mirrors the same pattern for all ``_ADDR_TYPES``.

Addresslines
------------
``addresslines`` references are resolved within each worker's accumulated
state (``photon_to_db`` dict).  Cross-worker references are silently dropped
– this is an acceptable limitation since most critical address data lives in
the ``address`` field of each place entry.
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
from .schema import compress_json, create_database, create_worker_database

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OSM name-tag prefixes -> internal "kind" code.
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

# Default batch size (number of raw JSON lines per dispatch).
_DEFAULT_BATCH = 500

# Number of places between transaction commits within a worker.
_COMMIT_EVERY = 5_000

# Sentinel value put on the work queue to tell a worker to shut down.
_SENTINEL = None

# Each worker is allocated this many consecutive place IDs.
# 50 million per worker -> supports up to 127 workers (63-bit SQLite INTEGER).
_PLACES_PER_WORKER = 50_000_000


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
    """Extract ``(value, lang, kind)`` triples from a photon *name* dict.

    Parameters
    ----------
    name_dict:
        The ``name`` field of a photon place entry.
    lang_set:
        Set of ISO language codes to retain.

    Returns
    -------
    Deduplicated list of ``(name_value, lang, kind)`` triples where *lang*
    is ``'default'`` for the bare OSM ``name`` tag and *kind* is one of the
    ``_NAME_KINDS`` values.
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
        Set of ISO language codes to retain.
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

    # Bounding box - fall back to point bbox when not provided.
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


# ---------------------------------------------------------------------------
# Worker-side string / category cache
# ---------------------------------------------------------------------------

class _LocalStringCache:
    """Cache string -> local integer id within a worker database.

    Inserts new strings with ``INSERT OR IGNORE`` so the UNIQUE constraint
    on ``strings.value`` guarantees deduplication within this worker.
    The mapping ``value -> id`` is kept in memory to avoid repeated lookups.
    """

    __slots__ = ("_conn", "_cache")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cache: Dict[str, int] = {}

    def get_id(self, value: str) -> int:
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


# ---------------------------------------------------------------------------
# Worker-side write helpers
# ---------------------------------------------------------------------------

def _write_countries_worker(
    conn: sqlite3.Connection,
    country_list: List[Dict],
    lang_set: Set[str],
) -> None:
    """Upsert CountryInfo records into the worker ``countries`` table."""
    for entry in country_list:
        code = _scalar(entry.get("country_code")) or ""
        code = code.lower()
        if not code:
            continue
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
        blob = (
            gzip.compress(
                json.dumps(filtered, ensure_ascii=False).encode(), compresslevel=6
            )
            if filtered
            else b""
        )
        conn.execute(
            "INSERT INTO countries(code, names) VALUES (?,?) "
            "ON CONFLICT(code) DO UPDATE SET names = excluded.names "
            "WHERE excluded.names != X''",
            (code, blob),
        )


def _write_place_worker(
    conn: sqlite3.Connection,
    place: Dict,
    place_id: int,
    strings: _LocalStringCache,
    cats: _LocalCategoryCache,
    photon_to_db: Dict[str, Tuple[int, Optional[str], Optional[str]]],
) -> None:
    """Insert one parsed place into the worker database."""
    # Resolve addresslines -> extra address components (within this worker).
    extra_addresses: List[Tuple[str, str, str]] = []
    for ref in place.get("addresslines", []):
        ref_pid = str(ref.get("place_id", ""))
        if ref_pid and ref_pid in photon_to_db:
            _ref_db_id, ref_addr_type, ref_name = photon_to_db[ref_pid]
            if ref.get("isaddress") and ref_addr_type and ref_name:
                extra_addresses.append((ref_addr_type, "default", ref_name))

    all_addresses: List[Tuple[str, str, str]] = list(place["addresses"]) + extra_addresses

    conn.execute(
        """
        INSERT INTO places (
            id, photon_id, osm_type, osm_id, osm_key, osm_value,
            address_type, importance, country_code, postcode, housenumber,
            lat, lon, bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat,
            extra
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            place_id,
            place["photon_id"],
            place["osm_type"],
            place["osm_id"],
            place["osm_key"],
            place["osm_value"],
            place["address_type"],
            place["importance"],
            place["country_code"],
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

    # Intern all strings needed by this place in one batch.
    all_str_values = list(
        {v for v, _l, _k in place["names"]}
        | {v for _t, _l, v in all_addresses}
    )
    sid_map: Dict[str, int] = {v: strings.get_id(v) for v in all_str_values}

    # place_names
    seen_pn: Set[Tuple] = set()
    name_rows: List[Tuple] = []
    for value, lang, kind in place["names"]:
        sid = sid_map.get(value)
        if sid is None:
            continue
        row = (place_id, sid, lang, kind)
        if row not in seen_pn:
            seen_pn.add(row)
            name_rows.append(row)
    if name_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO place_names(place_id, string_id, lang, kind) "
            "VALUES (?,?,?,?)",
            name_rows,
        )

    # place_addresses
    seen_pa: Set[Tuple] = set()
    addr_rows: List[Tuple] = []
    for addr_type, lang, value in all_addresses:
        sid = sid_map.get(value)
        if sid is None:
            continue
        row = (place_id, sid, addr_type, lang)
        if row not in seen_pa:
            seen_pa.add(row)
            addr_rows.append(row)
    if addr_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO place_addresses"
            "(place_id, string_id, addr_type, lang) VALUES (?,?,?,?)",
            addr_rows,
        )

    # place_categories
    for cat in place["categories"]:
        cid = cats.get_id(cat)
        conn.execute(
            "INSERT OR IGNORE INTO place_categories(place_id, category_id) "
            "VALUES (?,?)",
            (place_id, cid),
        )

    # FTS staging
    fts_names_parts = sorted({v for v, _l, _k in place["names"]})
    if place.get("postcode"):
        fts_names_parts.append(place["postcode"])
    if place.get("housenumber"):
        fts_names_parts.append(place["housenumber"])
    fts_addr_parts = sorted({v for _t, _l, v in all_addresses})

    conn.execute(
        "INSERT OR REPLACE INTO fts_data(place_id, names, address) VALUES (?,?,?)",
        (place_id, " ".join(fts_names_parts), " ".join(fts_addr_parts)),
    )

    # R-tree staging
    conn.execute(
        "INSERT OR REPLACE INTO rtree_data"
        "(id, min_lat, max_lat, min_lon, max_lon) VALUES (?,?,?,?,?)",
        (
            place_id,
            place["bbox_min_lat"],
            place["bbox_max_lat"],
            place["bbox_min_lon"],
            place["bbox_max_lon"],
        ),
    )

    # Update addresslines resolution map for subsequent places in this worker.
    primary_name: Optional[str] = next(
        (v for v, l, k in place["names"] if l == "default" and k == "name"),
        next((v for v, _l, k in place["names"] if k == "name"), None),
    )
    photon_id = place["photon_id"]
    if photon_id:
        photon_to_db[photon_id] = (place_id, place["address_type"], primary_name)


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
    """Worker process: drain *work_queue* and write parsed results to *db_path*.

    Parameters
    ----------
    work_queue:
        Shared input queue.  ``None`` is the sentinel that signals shutdown.
    db_path:
        Path for this worker's private SQLite database.
    languages:
        Supported language codes.
    poly_state:
        Serialised PolyFilter state, or ``None`` for no spatial filter.
    worker_id:
        Zero-based worker index (used only for logging).
    place_id_base:
        The first place ID this worker may assign.  Each worker uses a
        non-overlapping range so merge requires no place-ID remapping.
    """
    # Restore poly filter (pickling-safe).
    poly_filter: Optional[PolyFilter] = None
    if poly_state is not None:
        pf = PolyFilter([])
        pf.__setstate__(poly_state)
        poly_filter = pf

    lang_set: Set[str] = set(languages)
    conn = create_worker_database(db_path)
    strings = _LocalStringCache(conn)
    cats = _LocalCategoryCache(conn)
    photon_to_db: Dict[str, Tuple[int, Optional[str], Optional[str]]] = {}
    local_counter = 0

    conn.execute("BEGIN")

    while True:
        batch = work_queue.get()
        if batch is _SENTINEL:
            break

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
                    _write_countries_worker(conn, content, lang_set)

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
                    try:
                        _write_place_worker(
                            conn, parsed, place_id, strings, cats, photon_to_db
                        )
                    except sqlite3.Error as exc:
                        log.debug("Worker %d skipping place: %s", worker_id, exc)

        # Periodic commit to bound transaction size.
        if local_counter % _COMMIT_EVERY == 0 and local_counter > 0:
            conn.commit()
            conn.execute("BEGIN")

    conn.commit()
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

    * ``strings`` and ``categories`` are globally deduplicated (INSERT OR IGNORE
      on the UNIQUE value/name column).
    * ``countries`` are upserted by their TEXT primary key (``code``).
    * ``places`` receive new globally-sequential IDs via the ROW_NUMBER trick.
    * All dependent tables (place_names, place_addresses, place_categories,
      fts_data, rtree_data) are inserted with remapped IDs via JOIN.

    Returns the number of places merged from this worker.
    """
    final_conn.execute("ATTACH DATABASE ? AS w", (worker_path,))
    final_conn.execute("BEGIN")

    # 1. Strings - INSERT OR IGNORE deduplicates by value.
    final_conn.execute(
        "INSERT OR IGNORE INTO strings(value) SELECT value FROM w.strings"
    )

    # 2. String ID mapping: worker local id -> final global id (join by value).
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

    # 4. Countries - upsert by TEXT primary key (code); no integer remap needed.
    #    SQLite's new-style upsert (ON CONFLICT DO UPDATE) does not work with
    #    INSERT...SELECT from an ATTACH'd database, so we use a two-step approach:
    #    INSERT OR IGNORE to add new countries, then UPDATE to overwrite names
    #    from the worker only when the worker has non-empty name data.
    final_conn.execute(
        "INSERT OR IGNORE INTO countries(code, names) SELECT code, names FROM w.countries"
    )
    final_conn.execute(
        """
        UPDATE countries
        SET names = (
            SELECT wc.names FROM w.countries wc WHERE wc.code = countries.code
        )
        WHERE code IN (SELECT code FROM w.countries WHERE names != X'')
        """
    )

    # 5. Metadata - upsert (all workers write the same dump_version value).
    final_conn.execute(
        """
        INSERT OR REPLACE INTO metadata(key, value)
        SELECT key, value FROM w.metadata
        """
    )

    # 6. Places - insert without explicit id so SQLite assigns new sequential IDs.
    #    We capture the current max id before insertion to build the mapping.
    base_row = final_conn.execute(
        "SELECT COALESCE(MAX(id), 0) FROM places"
    ).fetchone()
    base_id: int = base_row[0]

    worker_count_row = final_conn.execute(
        "SELECT COUNT(*) FROM w.places"
    ).fetchone()
    worker_place_count: int = worker_count_row[0]

    if worker_place_count > 0:
        final_conn.execute(
            """
            INSERT INTO places(
                photon_id, osm_type, osm_id, osm_key, osm_value,
                address_type, importance, country_code, postcode, housenumber,
                lat, lon, bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat,
                extra
            )
            SELECT
                photon_id, osm_type, osm_id, osm_key, osm_value,
                address_type, importance, country_code, postcode, housenumber,
                lat, lon, bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat,
                extra
            FROM w.places
            ORDER BY id
            """
        )

        # 7. Place ID mapping using ROW_NUMBER.
        #    Rows were inserted in w.places ORDER BY id, so the k-th inserted
        #    row (1-based) got id = base_id + k.
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

        # 8. place_names - remap place_id and string_id.
        final_conn.execute(
            """
            INSERT OR IGNORE INTO place_names(place_id, string_id, lang, kind)
            SELECT pm.fid, sm.fid, pn.lang, pn.kind
            FROM w.place_names pn
            JOIN _pmap pm ON pm.wid = pn.place_id
            JOIN _smap sm ON sm.wid = pn.string_id
            """
        )

        # 9. place_addresses - remap place_id and string_id.
        final_conn.execute(
            """
            INSERT OR IGNORE INTO place_addresses(place_id, string_id, addr_type, lang)
            SELECT pm.fid, sm.fid, pa.addr_type, pa.lang
            FROM w.place_addresses pa
            JOIN _pmap pm ON pm.wid = pa.place_id
            JOIN _smap sm ON sm.wid = pa.string_id
            """
        )

        # 10. place_categories - remap place_id and category_id.
        final_conn.execute(
            """
            INSERT OR IGNORE INTO place_categories(place_id, category_id)
            SELECT pm.fid, cm.fid
            FROM w.place_categories pc
            JOIN _pmap pm ON pm.wid = pc.place_id
            JOIN _cmap cm ON cm.wid = pc.category_id
            """
        )

        # 11. FTS5 - remap place_id; insert into virtual table.
        final_conn.execute(
            """
            INSERT INTO places_fts(place_id, names, address)
            SELECT pm.fid, fd.names, fd.address
            FROM w.fts_data fd
            JOIN _pmap pm ON pm.wid = fd.place_id
            """
        )

        # 12. R-tree - remap id.
        final_conn.execute(
            """
            INSERT INTO places_rtree(id, min_lat, max_lat, min_lon, max_lon)
            SELECT pm.fid, rd.min_lat, rd.max_lat, rd.min_lon, rd.max_lon
            FROM w.rtree_data rd
            JOIN _pmap pm ON pm.wid = rd.id
            """
        )

    final_conn.commit()
    final_conn.execute("DETACH DATABASE w")

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
        Number of worker processes.  ``0`` uses all available CPUs.
    batch_size:
        Number of JSON lines per work-queue item.
    show_progress:
        Display a ``tqdm`` progress bar when *tqdm* is installed.
    """
    if not languages:
        raise ValueError("At least one language must be specified.")

    n_workers = num_workers if num_workers > 0 else max(1, (mp.cpu_count() or 1))
    log.info("Using %d worker process(es)", n_workers)

    # Build poly filter serialisation (None -> no filter).
    poly_state: Optional[list] = None
    if poly_file:
        pf = PolyFilter.from_file(poly_file)
        poly_state = pf.__getstate__()
        log.info(
            "Loaded poly filter from %s (%d polygon(s))",
            poly_file,
            len(poly_state),
        )

    # Determine worker database paths (same directory as the output file).
    out_dir = os.path.dirname(os.path.abspath(output_path))
    worker_db_paths: List[str] = [
        os.path.join(out_dir, f"_worker_{i}_tmp.db")
        for i in range(n_workers)
    ]

    # Remove any stale worker databases from a previous failed run.
    for p in worker_db_paths:
        if os.path.exists(p):
            os.unlink(p)

    # ---------- Phase 1: parallel import into worker databases ---------------

    # Bounded queue: prevents the reader from buffering the whole file in RAM.
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
            work_queue.put(batch)  # blocks when queue is full -> back-pressure
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

    # Send shutdown sentinel to each worker.
    for _ in workers:
        work_queue.put(_SENTINEL)

    log.info("All %d batches dispatched, waiting for workers...", batches_sent)
    for p in workers:
        p.join()
    log.info("All workers finished.")

    # ---------- Phase 2: merge worker databases into final database ----------

    if os.path.exists(output_path):
        os.unlink(output_path)
    final_conn = create_database(output_path)

    # Store language metadata in the final database.
    final_conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?,?)",
        ("languages", json.dumps(languages)),
    )
    final_conn.commit()

    total_places = 0
    for i, worker_path in enumerate(worker_db_paths):
        if not os.path.exists(worker_path):
            log.warning("Worker %d database not found: %s", i, worker_path)
            continue
        log.info("Merging worker %d database (%s)...", i, worker_path)
        n = _merge_worker_db(final_conn, worker_path)
        total_places += n
        log.info("  merged %d places (running total: %d)", n, total_places)

    # Post-merge optimisation.
    log.info("Running ANALYZE...")
    final_conn.execute("ANALYZE")
    final_conn.close()

    # Clean up worker databases.
    for p in worker_db_paths:
        try:
            if os.path.exists(p):
                os.unlink(p)
        except OSError as exc:
            log.warning("Could not remove worker DB %s: %s", p, exc)

    log.info("Import complete: %d places imported.", total_places)
