"""Search and reverse-geocoding query functions.

All functions accept either a file-system path to an SQLite database or an
open :class:`sqlite3.Connection`.

Search (geocoding)
------------------
Uses the FTS5 ``places_fts`` virtual table with the *trigram* tokeniser for
fast fuzzy substring matching.  An optional bounding box restricts results
geographically.

Reverse geocoding
-----------------
Uses the R-tree ``places_rtree`` virtual table to find candidates within a
square radius, then ranks them by Euclidean distance.

Examples
--------
>>> import offline_geocoding.query as q

# --- Forward search --------------------------------------------------
>>> results = q.search("db.sqlite", "Paris", languages=["en", "fr"], limit=5)
>>> for r in results:
...     print(r["name"], r["lat"], r["lon"])

# --- Bounding-box search ---------------------------------------------
>>> bbox = (2.0, 48.5, 3.0, 49.2)   # (min_lon, min_lat, max_lon, max_lat)
>>> results = q.search("db.sqlite", "Rivoli", bbox=bbox, limit=10)

# --- Reverse geocoding -----------------------------------------------
>>> results = q.reverse("db.sqlite", lat=48.8566, lon=2.3522, radius_deg=0.05)
>>> for r in results:
...     print(r["name"], r["distance_deg"])
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

from .schema import apply_pragmas, decompress_json

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

_DbArg = Union[str, sqlite3.Connection]


def _conn(db: _DbArg) -> Tuple[sqlite3.Connection, bool]:
    """Return ``(connection, should_close)``."""
    if isinstance(db, str):
        c = sqlite3.connect(db, check_same_thread=False)
        c.row_factory = sqlite3.Row
        apply_pragmas(c)
        return c, True
    return db, False


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _get_supported_languages(conn: sqlite3.Connection) -> List[str]:
    """Read the language list stored during import."""
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'languages'"
    ).fetchone()
    if row:
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _resolve_country_name(
    conn: sqlite3.Connection,
    country_code: Optional[str],
    languages: List[str],
    cache: Dict[str, Optional[str]],
) -> Optional[str]:
    """Return the localised country name for *country_code*.

    Results are memoised in *cache* (keyed by country_code + language list
    fingerprint) to avoid re-decompressing the same blob for every result in
    a search response.

    The country ``names`` column is a gzip-compressed JSON object mapping
    language code -> localised name, e.g.::

        {"default": "France", "en": "France", "fr": "France"}

    Lookup strategy:
    1. Try each language in *languages* in order.
    2. Fall back to the ``"default"`` entry (the bare OSM ``name`` tag).
    """
    if not country_code:
        return None

    # Build a cache key that incorporates the requested language order.
    lang_key = f"{country_code}|{','.join(languages)}"
    if lang_key in cache:
        return cache[lang_key]

    c_row = conn.execute(
        "SELECT names FROM countries WHERE code = ?",
        (country_code,),
    ).fetchone()

    country_name: Optional[str] = None
    if c_row and c_row["names"]:
        try:
            cnames = decompress_json(c_row["names"])
            for lang in languages:
                if lang in cnames:
                    country_name = cnames[lang]
                    break
            if country_name is None:
                country_name = cnames.get("default")
        except Exception:
            pass

    cache[lang_key] = country_name
    return country_name


def _format_place(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    languages: Optional[List[str]] = None,
    country_cache: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """Convert a ``places`` table row into a result dictionary.

    Parameters
    ----------
    conn:
        Open database connection.
    row:
        A row from the ``places`` table (or joined result).
    languages:
        Preferred language order for name resolution.  Defaults to the
        languages stored in the database.
    country_cache:
        Optional shared dict used to cache decompressed country name blobs
        across multiple calls.  Pass the same dict for all results in a
        single query to avoid redundant decompression.
    """
    place_id: int = row["id"]

    if languages is None:
        languages = _get_supported_languages(conn)

    if country_cache is None:
        country_cache = {}

    # Build {(lang, kind): name_value} from place_names + strings.
    name_rows = conn.execute(
        """
        SELECT s.value, pn.lang, pn.kind
        FROM place_names pn
        JOIN strings s ON s.id = pn.string_id
        WHERE pn.place_id = ?
        """,
        (place_id,),
    ).fetchall()

    names_by_lang_kind: Dict[Tuple[str, str], str] = {
        (r["lang"], r["kind"]): r["value"] for r in name_rows
    }
    all_name_values = [r["value"] for r in name_rows]

    # Resolve display name: prefer requested languages in order, fall back
    # to 'default'.
    display_name: Optional[str] = None
    for lang in (languages or []):
        display_name = names_by_lang_kind.get((lang, "name"))
        if display_name:
            break
    if display_name is None:
        display_name = names_by_lang_kind.get(("default", "name"))
    if display_name is None and all_name_values:
        display_name = all_name_values[0]

    # Address components.
    addr_rows = conn.execute(
        """
        SELECT s.value, pa.addr_type, pa.lang
        FROM place_addresses pa
        JOIN strings s ON s.id = pa.string_id
        WHERE pa.place_id = ?
        ORDER BY pa.addr_type, pa.lang
        """,
        (place_id,),
    ).fetchall()

    address: Dict[str, str] = {}
    for ar in addr_rows:
        key = (
            ar["addr_type"]
            if ar["lang"] == "default"
            else f"{ar['addr_type']}:{ar['lang']}"
        )
        address[key] = ar["value"]

    # Categories.
    cat_rows = conn.execute(
        """
        SELECT c.name
        FROM place_categories pc
        JOIN categories c ON c.id = pc.category_id
        WHERE pc.place_id = ?
        """,
        (place_id,),
    ).fetchall()
    category_list = [r["name"] for r in cat_rows]

    # Country name – looked up by the TEXT country_code column (no integer FK).
    # Uses the per-call cache to avoid redundant blob decompression.
    country_code: Optional[str] = row["country_code"]
    country_name = _resolve_country_name(
        conn, country_code, languages or [], country_cache
    )

    # Extra tags.
    extra: Optional[Dict] = None
    if row["extra"]:
        try:
            extra = decompress_json(row["extra"])
        except Exception:
            pass

    result: Dict[str, Any] = {
        "id": place_id,
        "name": display_name,
        "all_names": all_name_values,
        "osm_type": row["osm_type"],
        "osm_id": row["osm_id"],
        "osm_key": row["osm_key"],
        "osm_value": row["osm_value"],
        "address_type": row["address_type"],
        "importance": row["importance"],
        "lat": row["lat"],
        "lon": row["lon"],
        "bbox": (
            row["bbox_min_lon"],
            row["bbox_min_lat"],
            row["bbox_max_lon"],
            row["bbox_max_lat"],
        )
        if row["bbox_min_lon"] is not None
        else None,
        "postcode": row["postcode"],
        "housenumber": row["housenumber"],
        "country_code": country_code,
        "country_name": country_name,
        "address": address,
        "categories": category_list,
        "extra": extra,
    }
    return result


# ---------------------------------------------------------------------------
# Forward search (geocoding)
# ---------------------------------------------------------------------------

def search(
    db: _DbArg,
    query: str,
    languages: Optional[List[str]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    limit: int = 10,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Full-text search with optional bounding-box filter.

    Parameters
    ----------
    db:
        Path to the SQLite database or an open connection.
    query:
        Search string.  Trigram FTS supports substring matching – partial
        words and typos are handled without explicit wildcards.
    languages:
        Preferred language order for display names.  Defaults to the
        languages stored during import.
    bbox:
        Optional bounding box ``(min_lon, min_lat, max_lon, max_lat)`` in
        WGS84 degrees.  Only places whose centroid falls within the box are
        returned.
    limit:
        Maximum number of results.
    offset:
        Pagination offset.

    Returns
    -------
    List of place dictionaries ordered by FTS rank × importance.
    """
    if not query or not query.strip():
        return []

    conn, should_close = _conn(db)
    try:
        # Escape FTS5 special characters to avoid query-syntax errors.
        fts_query = _escape_fts(query)

        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            sql = """
                SELECT p.*
                FROM places_fts fts
                JOIN places p ON p.id = fts.place_id
                WHERE places_fts MATCH ?
                  AND p.lat BETWEEN ? AND ?
                  AND p.lon BETWEEN ? AND ?
                ORDER BY fts.rank, p.importance DESC
                LIMIT ? OFFSET ?
            """
            params = (
                fts_query,
                min_lat, max_lat,
                min_lon, max_lon,
                limit, offset,
            )
        else:
            sql = """
                SELECT p.*
                FROM places_fts fts
                JOIN places p ON p.id = fts.place_id
                WHERE places_fts MATCH ?
                ORDER BY fts.rank, p.importance DESC
                LIMIT ? OFFSET ?
            """
            params = (fts_query, limit, offset)

        rows = conn.execute(sql, params).fetchall()
        country_cache: Dict[str, Optional[str]] = {}
        return [_format_place(conn, r, languages, country_cache) for r in rows]
    finally:
        if should_close:
            conn.close()


def search_column(
    db: _DbArg,
    query: str,
    column: str = "names",
    languages: Optional[List[str]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    limit: int = 10,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """FTS search restricted to a single FTS column (``'names'`` or ``'address'``).

    Useful when you want to match only the name or only the address part.
    """
    if column not in ("names", "address"):
        raise ValueError("column must be 'names' or 'address'")
    return search(
        db,
        f"{column}:{_escape_fts(query)}",
        languages=languages,
        bbox=bbox,
        limit=limit,
        offset=offset,
    )


# ---------------------------------------------------------------------------
# Reverse geocoding
# ---------------------------------------------------------------------------

def reverse(
    db: _DbArg,
    lat: float,
    lon: float,
    radius_deg: float = 0.1,
    limit: int = 5,
    languages: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Find the nearest places to ``(lat, lon)``.

    Uses the R-tree index for a fast bounding-box pre-filter, then ranks
    candidates by Euclidean distance (in degrees).

    Parameters
    ----------
    db:
        Path to the SQLite database or an open connection.
    lat, lon:
        Query coordinates in WGS84 degrees.
    radius_deg:
        Search radius in degrees (≈111 km per degree of latitude).
    limit:
        Maximum number of results.
    languages:
        Preferred language order for display names.

    Returns
    -------
    List of place dictionaries with an extra ``distance_deg`` key, ordered
    by ascending distance.
    """
    conn, should_close = _conn(db)
    try:
        min_lat = lat - radius_deg
        max_lat = lat + radius_deg
        min_lon = lon - radius_deg
        max_lon = lon + radius_deg

        sql = """
            SELECT p.*,
                   ((p.lat - ?) * (p.lat - ?) + (p.lon - ?) * (p.lon - ?)) AS dist_sq
            FROM places_rtree rt
            JOIN places p ON p.id = rt.id
            WHERE rt.min_lat <= ? AND rt.max_lat >= ?
              AND rt.min_lon <= ? AND rt.max_lon >= ?
            ORDER BY dist_sq
            LIMIT ?
        """
        params = (
            lat, lat, lon, lon,
            max_lat, min_lat,
            max_lon, min_lon,
            limit,
        )

        rows = conn.execute(sql, params).fetchall()
        results: List[Dict[str, Any]] = []
        country_cache: Dict[str, Optional[str]] = {}
        for row in rows:
            place = _format_place(conn, row, languages, country_cache)
            dist_sq = row["dist_sq"]
            place["distance_deg"] = math.sqrt(dist_sq) if dist_sq >= 0 else 0.0
            results.append(place)
        return results
    finally:
        if should_close:
            conn.close()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _escape_fts(query: str) -> str:
    """Wrap *query* in double-quotes to prevent FTS5 syntax errors.

    For trigram FTS, quoted phrases still perform substring matching.
    """
    # Remove any existing quotes to avoid nesting issues.
    clean = query.replace('"', ' ').strip()
    if not clean:
        return '""'
    return f'"{clean}"'


def get_languages(db: _DbArg) -> List[str]:
    """Return the supported language codes stored in *db*."""
    conn, should_close = _conn(db)
    try:
        return _get_supported_languages(conn)
    finally:
        if should_close:
            conn.close()


def stats(db: _DbArg) -> Dict[str, int]:
    """Return basic row-count statistics for *db*."""
    conn, should_close = _conn(db)
    try:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "places",
                "strings",
                "categories",
                "countries",
                "place_names",
                "place_addresses",
                "place_categories",
            )
        }
    finally:
        if should_close:
            conn.close()
