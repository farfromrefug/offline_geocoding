"""Search and reverse-geocoding query functions.

All functions accept either a file-system path to an SQLite database or an
open :class:`sqlite3.Connection`.

Search (geocoding)
------------------
Uses the FTS5 ``places_fts`` virtual table (contentless, trigram tokeniser)
for fast fuzzy substring matching.  Matches return ``rowid`` (== ``place_id``)
which is joined to ``places``.  An optional bounding box restricts results
geographically.

Reverse geocoding
-----------------
Uses the R-tree ``places_rtree`` virtual table to find candidates within a
square radius (centroid-to-centroid), then ranks them by Euclidean distance.

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

from .schema import (
    ADDR_TYPE_IDS,
    IMPORTANCE_SCALE,
    LAT_LON_SCALE,
    NAME_KIND_IDS,
    apply_pragmas,
)

# ---------------------------------------------------------------------------
# Precomputed reverse-lookup dicts (no DB query needed)
# ---------------------------------------------------------------------------

#: Integer kind_id -> kind string (e.g. 1 -> "name")
_KIND_BY_ID: Dict[int, str] = {v: k for k, v in NAME_KIND_IDS.items()}

#: Integer addr_type_id -> addr_type string (e.g. 4 -> "city")
_ADDR_TYPE_BY_ID: Dict[int, str] = {v: k for k, v in ADDR_TYPE_IDS.items()}

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

    Looks up ``country_names`` joined with ``strings`` and ``langs``.
    Results are memoised in *cache*.
    """
    if not country_code:
        return None

    lang_key = f"{country_code}|{','.join(languages)}"
    if lang_key in cache:
        return cache[lang_key]

    rows = conn.execute(
        """
        SELECT s.value AS name, l.code AS lang
        FROM country_names cn
        JOIN strings s ON s.id = cn.string_id
        JOIN langs l   ON l.id = cn.lang_id
        WHERE cn.code = ?
        """,
        (country_code,),
    ).fetchall()

    names: Dict[str, str] = {r["lang"]: r["name"] for r in rows}

    country_name: Optional[str] = None
    for lang in languages:
        if lang in names:
            country_name = names[lang]
            break
    if country_name is None:
        country_name = names.get("default")

    cache[lang_key] = country_name
    return country_name


def _resolve_string(
    conn: sqlite3.Connection,
    string_id: Optional[int],
) -> Optional[str]:
    """Look up a single string by its integer ID."""
    if string_id is None:
        return None
    row = conn.execute(
        "SELECT value FROM strings WHERE id = ?", (string_id,)
    ).fetchone()
    return row[0] if row else None


def _resolve_osm_tag_token(
    conn: sqlite3.Connection,
    tag_id: Optional[int],
) -> Optional[str]:
    """Look up an OSM tag token string by its ``osm_tags.id``."""
    if tag_id is None:
        return None
    row = conn.execute(
        "SELECT token FROM osm_tags WHERE id = ?", (tag_id,)
    ).fetchone()
    return row[0] if row else None


def _format_place(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    languages: Optional[List[str]] = None,
    country_cache: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """Convert a ``places`` table row into a result dictionary."""
    place_id: int = row["id"]

    if languages is None:
        languages = _get_supported_languages(conn)

    if country_cache is None:
        country_cache = {}

    # Build {(lang, kind_str): name_value} from place_names + strings + langs.
    name_rows = conn.execute(
        """
        SELECT s.value AS value, l.code AS lang, pn.kind_id
        FROM place_names pn
        JOIN strings s ON s.id = pn.string_id
        JOIN langs   l ON l.id = pn.lang_id
        WHERE pn.place_id = ?
        """,
        (place_id,),
    ).fetchall()

    names_by_lang_kind: Dict[Tuple[str, str], str] = {}
    all_name_values: List[str] = []
    for r in name_rows:
        kind_str = _KIND_BY_ID.get(r["kind_id"], "name")
        names_by_lang_kind[(r["lang"], kind_str)] = r["value"]
        all_name_values.append(r["value"])

    # Resolve display name.
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
        SELECT s.value AS value, pa.addr_type_id, l.code AS lang
        FROM place_addresses pa
        JOIN strings   s ON s.id = pa.string_id
        JOIN langs     l ON l.id = pa.lang_id
        WHERE pa.place_id = ?
        ORDER BY pa.addr_type_id, pa.lang_id
        """,
        (place_id,),
    ).fetchall()

    address: Dict[str, str] = {}
    for ar in addr_rows:
        at_str = _ADDR_TYPE_BY_ID.get(ar["addr_type_id"], "other")
        key = at_str if ar["lang"] == "default" else f"{at_str}:{ar['lang']}"
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

    # Country name from country_names table.
    country_code: Optional[str] = row["country_code"]
    country_name = _resolve_country_name(
        conn, country_code, languages or [], country_cache
    )

    # Resolve string-ID columns.
    osm_key     = _resolve_osm_tag_token(conn, row["osm_key_id"])
    osm_value   = _resolve_osm_tag_token(conn, row["osm_value_id"])
    postcode    = _resolve_string(conn, row["postcode_id"])
    housenumber = _resolve_string(conn, row["hn_id"])
    address_type = _ADDR_TYPE_BY_ID.get(row["addr_type_id"]) if row["addr_type_id"] else None

    # Extra tags.
    extra: Optional[Dict] = None
    if row["extra"]:
        try:
            from .schema import decompress_json
            extra = decompress_json(row["extra"])
        except Exception:
            pass

    result: Dict[str, Any] = {
        "id":           place_id,
        "name":         display_name,
        "all_names":    all_name_values,
        "osm_id":       row["osm_id"],
        "osm_key":      osm_key,
        "osm_value":    osm_value,
        "address_type": address_type,
        # Stored as INTEGER (×IMPORTANCE_SCALE); convert back to float.
        "importance":   row["importance"] / IMPORTANCE_SCALE,
        # Stored as INTEGER (×LAT_LON_SCALE); convert back to float degrees.
        "lat":          row["lat"] / LAT_LON_SCALE,
        "lon":          row["lon"] / LAT_LON_SCALE,
        "postcode":     postcode,
        "housenumber":  housenumber,
        "country_code": country_code,
        "country_name": country_name,
        "address":      address,
        "categories":   category_list,
        "extra":        extra,
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
        Search string.  Trigram FTS supports substring matching.
    languages:
        Preferred language order for display names.
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
        fts_query = _escape_fts(query)

        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            sql = """
                SELECT p.*
                FROM places_fts fts
                JOIN places p ON p.id = fts.rowid
                WHERE places_fts MATCH ?
                  AND p.lat BETWEEN ? AND ?
                  AND p.lon BETWEEN ? AND ?
                ORDER BY fts.rank, p.importance DESC
                LIMIT ? OFFSET ?
            """
            params = (
                fts_query,
                round(min_lat * LAT_LON_SCALE), round(max_lat * LAT_LON_SCALE),
                round(min_lon * LAT_LON_SCALE), round(max_lon * LAT_LON_SCALE),
                limit, offset,
            )
        else:
            sql = """
                SELECT p.*
                FROM places_fts fts
                JOIN places p ON p.id = fts.rowid
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
    """FTS search restricted to a single FTS column (``'names'`` or ``'address'``)."""
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

    The R-tree stores centroid points (min == max for lat/lon), so the query
    ``min_lat <= max_lat_query AND max_lat >= min_lat_query`` correctly finds
    all centroids within the radius box.

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
    """
    conn, should_close = _conn(db)
    try:
        min_lat = lat - radius_deg
        max_lat = lat + radius_deg
        min_lon = lon - radius_deg
        max_lon = lon + radius_deg

        lat_int = round(lat * LAT_LON_SCALE)
        lon_int = round(lon * LAT_LON_SCALE)

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
            lat_int, lat_int, lon_int, lon_int,
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
            place["distance_deg"] = math.sqrt(dist_sq) / LAT_LON_SCALE if dist_sq >= 0 else 0.0
            results.append(place)
        return results
    finally:
        if should_close:
            conn.close()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _escape_fts(query: str) -> str:
    """Wrap *query* in double-quotes to prevent FTS5 syntax errors."""
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
                "osm_tags",
                "osm_tag_names",
                "countries",
                "country_names",
                "place_names",
                "place_addresses",
                "place_categories",
                "place_osm_tags",
            )
        }
    finally:
        if should_close:
            conn.close()
