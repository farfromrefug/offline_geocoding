"""SQLite database schema for offline geocoding.

Design Goals
------------
* **Zero cross-worker duplication** – strings and categories are interned and
  identified by integer primary key.  Multiple worker processes each build
  their own temporary database; a SQL-driven merge step deduplicates across
  workers using temp mapping tables.
* **Compact storage** – shared ``strings`` table means "Paris" is stored once
  even if thousands of places reference it as a city or name.
* **Language-aware queries** – country names stored as a gzip-compressed JSON
  map ``{lang: name}``; all other name/address variants stored as rows in
  ``place_names`` / ``place_addresses`` with an explicit ``lang`` column.
* **Fast search** – FTS5 with trigram tokeniser for fuzzy substring matching;
  R-tree for reverse geocoding.

Tables (final database)
-----------------------
metadata          – key/value store for import settings
countries         – ISO country codes keyed by ``code TEXT PRIMARY KEY``
                    + gzip-compressed JSON name map  (lang → name)
strings           – interned text values; INTEGER PRIMARY KEY (AUTOINCREMENT)
categories        – interned OSM category strings; INTEGER PRIMARY KEY
places            – one row per photon place entry; references country by code
place_names       – N:M place ↔ string with lang + kind columns
place_addresses   – N:M place ↔ string with addr_type + lang columns
place_categories  – N:M place ↔ category
places_fts        – FTS5 virtual table, trigram tokeniser (fuzzy search)
places_rtree      – R-tree spatial index for reverse geocoding

Worker databases (temporary, one per worker process)
-----------------------------------------------------
Same as above *minus* the virtual tables (FTS5 / R-tree).
Instead two plain tables hold the data that will be loaded into the virtual
tables during the final merge step:

fts_data   – (place_id, names TEXT, address TEXT)
rtree_data – (id, min_lat, max_lat, min_lon, max_lon)
"""

import gzip
import json
import logging
import sqlite3
from typing import Any, List

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage scale constants
# ---------------------------------------------------------------------------

#: lat/lon is stored as ``round(degrees * LAT_LON_SCALE)`` (INTEGER).
#: 1 000 000 gives ~0.11 m precision at the equator.
LAT_LON_SCALE: int = 1_000_000

#: importance is stored as ``round(value * IMPORTANCE_SCALE)`` (INTEGER).
#: 2 decimal digits are sufficient for scoring.
IMPORTANCE_SCALE: int = 100

# ---------------------------------------------------------------------------
# Schema DDL – shared between final and worker databases
# ---------------------------------------------------------------------------

_PRAGMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA temp_store   = MEMORY;
PRAGMA cache_size   = -65536;
PRAGMA mmap_size    = 268435456;
PRAGMA foreign_keys = ON;
"""

# Worker databases are disposable: use synchronous=OFF (safe – if the process
# crashes the whole file is discarded) and foreign_keys=OFF (constraint checking
# is pure overhead when the DB is rebuilt from scratch every import run).
_WORKER_PRAGMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = OFF;
PRAGMA temp_store   = MEMORY;
PRAGMA cache_size   = -131072;
PRAGMA mmap_size    = 268435456;
PRAGMA foreign_keys = OFF;
"""

# Tables present in BOTH the final database and every worker database.
_COMMON_TABLES = """
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per country.
-- ``code`` is the ISO 3166-1 alpha-2 code used as a natural key.
-- ``names`` is a gzip-compressed JSON object mapping lang -> localised name,
-- e.g. {"default": "France", "en": "France", "fr": "France"}.
-- Queries decompress this blob in Python and look up by language code,
-- so country names support the same multi-language query as place names.
CREATE TABLE IF NOT EXISTS countries (
    code  TEXT PRIMARY KEY,
    names BLOB NOT NULL DEFAULT X''
);

-- Interned text strings.  Every unique address component, place name, etc.
-- is stored exactly once here.  ``id`` is an AUTOINCREMENT integer assigned
-- locally within each worker; the merge step remaps these to globally unique
-- IDs in the final database.
CREATE TABLE IF NOT EXISTS strings (
    id    INTEGER PRIMARY KEY,
    value TEXT UNIQUE NOT NULL
);

-- Interned OSM category strings (e.g. "amenity.restaurant").
-- Same AUTOINCREMENT + merge-remap approach as strings.
CREATE TABLE IF NOT EXISTS categories (
    id   INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

-- Core place record.
-- ``country_code`` references countries(code) directly – no integer FK
-- indirection – so the merge step does not need to remap country IDs.
-- ``extra`` is a gzip-compressed JSON blob of arbitrary extra OSM tags.
-- ``importance`` is stored as round(value * IMPORTANCE_SCALE) (INTEGER).
-- ``lat`` / ``lon`` are stored as round(degrees * LAT_LON_SCALE) (INTEGER).
CREATE TABLE IF NOT EXISTS places (
    id           INTEGER PRIMARY KEY,
    photon_id    TEXT,
    osm_type     TEXT,
    osm_id       INTEGER,
    osm_key      TEXT,
    osm_value    TEXT,
    address_type TEXT,
    importance   INTEGER NOT NULL DEFAULT 0,
    country_code TEXT    REFERENCES countries(code),
    postcode     TEXT,
    housenumber  TEXT,
    lat          INTEGER NOT NULL,
    lon          INTEGER NOT NULL,
    bbox_min_lon REAL,
    bbox_min_lat REAL,
    bbox_max_lon REAL,
    bbox_max_lat REAL,
    extra        BLOB
);

-- Multilingual names for a place.
-- lang: 'default' for the bare OSM ``name`` tag, otherwise ISO language code.
-- kind: 'name' | 'alt' | 'old' | 'int' | 'loc' | 'short' | 'official'
CREATE TABLE IF NOT EXISTS place_names (
    place_id  INTEGER NOT NULL REFERENCES places(id)  ON DELETE CASCADE,
    string_id INTEGER NOT NULL REFERENCES strings(id),
    lang      TEXT    NOT NULL,
    kind      TEXT    NOT NULL DEFAULT 'name',
    PRIMARY KEY (place_id, string_id, lang, kind)
);

-- Address components for a place.
-- addr_type: 'country' | 'state' | 'county' | 'city' | 'district' |
--            'locality' | 'street' | 'other'
-- lang: 'default' | ISO language code
CREATE TABLE IF NOT EXISTS place_addresses (
    place_id  INTEGER NOT NULL REFERENCES places(id)  ON DELETE CASCADE,
    string_id INTEGER NOT NULL REFERENCES strings(id),
    addr_type TEXT    NOT NULL,
    lang      TEXT    NOT NULL,
    PRIMARY KEY (place_id, addr_type, lang)
);

-- Categories assigned to a place.
CREATE TABLE IF NOT EXISTS place_categories (
    place_id    INTEGER NOT NULL REFERENCES places(id)      ON DELETE CASCADE,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    PRIMARY KEY (place_id, category_id)
);
"""

# Worker-only tables that hold data destined for the final virtual tables.
_WORKER_PLAIN_TABLES = """
-- Staging table for FTS5 data; populated in each worker, merged into
-- the final ``places_fts`` virtual table during the merge step.
CREATE TABLE IF NOT EXISTS fts_data (
    place_id INTEGER PRIMARY KEY,
    names    TEXT NOT NULL DEFAULT '',
    address  TEXT NOT NULL DEFAULT ''
);

-- Staging table for R-tree data; same lifecycle as fts_data.
-- Coordinates stored as round(degrees * LAT_LON_SCALE) to save space;
-- converted back to REAL when inserted into the places_rtree virtual table.
CREATE TABLE IF NOT EXISTS rtree_data (
    id      INTEGER PRIMARY KEY,
    min_lat INTEGER NOT NULL,
    max_lat INTEGER NOT NULL,
    min_lon INTEGER NOT NULL,
    max_lon INTEGER NOT NULL
);
"""

# FTS5 with trigram tokeniser – enables efficient substring / fuzzy search.
# `names`   : space-separated concatenation of all name variants (all langs).
# `address` : space-separated concatenation of all address tokens (all langs).
_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS places_fts USING fts5(
    place_id UNINDEXED,
    names,
    address,
    tokenize = 'trigram case_sensitive 0'
);
"""

# R-tree virtual table for fast bounding-box reverse geocoding.
_RTREE = """
CREATE VIRTUAL TABLE IF NOT EXISTS places_rtree USING rtree(
    id,
    min_lat, max_lat,
    min_lon, max_lon
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_places_country    ON places(country_code);
CREATE INDEX IF NOT EXISTS idx_places_osm        ON places(osm_type, osm_id);
CREATE INDEX IF NOT EXISTS idx_places_importance ON places(importance DESC);
CREATE INDEX IF NOT EXISTS idx_places_photon_id  ON places(photon_id);
CREATE INDEX IF NOT EXISTS idx_place_names_place ON place_names(place_id);
CREATE INDEX IF NOT EXISTS idx_place_names_lang  ON place_names(lang, kind);
CREATE INDEX IF NOT EXISTS idx_place_addr_place  ON place_addresses(place_id);
CREATE INDEX IF NOT EXISTS idx_place_cat_place   ON place_categories(place_id);
"""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _check_sqlite_version() -> None:
    """Raise ``RuntimeError`` if SQLite is too old for the trigram tokeniser."""
    sqlite_ver = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
    if sqlite_ver < (3, 34, 0):
        raise RuntimeError(
            f"SQLite {sqlite3.sqlite_version} does not support the FTS5 trigram "
            "tokeniser. Please upgrade to SQLite >= 3.34.0."
        )


def create_database(path: str, with_indexes: bool = True) -> sqlite3.Connection:
    """Create (or open) the **final** geocoding database and apply its schema.

    The final database has FTS5 and R-tree virtual tables in addition to all
    common tables.

    Parameters
    ----------
    path:
        Filesystem path for the SQLite file.
    with_indexes:
        When ``True`` (default) the secondary B-tree indexes are created
        immediately.  Pass ``False`` to defer index creation to the end of the
        import (via :func:`create_indexes`) for faster bulk-insert performance.

    Returns an open :class:`sqlite3.Connection`.
    """
    _check_sqlite_version()
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    _apply_pragma(conn)

    for block in (_COMMON_TABLES, _FTS, _RTREE):
        for stmt in _split_statements(block):
            conn.execute(stmt)

    if with_indexes:
        create_indexes(conn)

    return conn


def create_indexes(conn: sqlite3.Connection) -> None:
    """Create (or ensure existence of) all secondary B-tree indexes.

    Safe to call multiple times – every statement uses ``IF NOT EXISTS``.
    Commit the connection after calling this function if needed.
    """
    for stmt in _split_statements(_INDEXES):
        conn.execute(stmt)


def create_worker_database(path: str) -> sqlite3.Connection:
    """Create a **worker** temporary database used during parallel import.

    Worker databases contain the same relational tables as the final database
    but use plain ``fts_data`` / ``rtree_data`` tables instead of virtual
    tables.  This avoids virtual-table overhead during high-throughput writes
    and ensures that ATTACH-based merging works reliably.

    ``PRAGMA foreign_keys = OFF`` is intentional: worker DBs are disposable and
    constraint checking is pure overhead on the hot write path.  Integrity is
    still guaranteed via the merge SQL which JOINs on mapping tables.

    Returns an open :class:`sqlite3.Connection`.
    """
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    for stmt in _split_statements(_WORKER_PRAGMA):
        conn.execute(stmt)

    for block in (_COMMON_TABLES, _WORKER_PLAIN_TABLES):
        for stmt in _split_statements(block):
            conn.execute(stmt)

    return conn


def apply_pragmas(conn: sqlite3.Connection) -> None:
    """Re-apply per-connection PRAGMAs (call after every new connection)."""
    _apply_pragma(conn)


def _apply_pragma(conn: sqlite3.Connection) -> None:
    for stmt in _split_statements(_PRAGMA):
        conn.execute(stmt)


def compress_json(obj: Any) -> bytes:
    """JSON-encode *obj* and gzip-compress the result."""
    return gzip.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8"), compresslevel=6)


def decompress_json(data: bytes) -> Any:
    """Decompress *data* (gzip) and JSON-decode it."""
    return json.loads(gzip.decompress(data).decode("utf-8"))


def _split_statements(sql: str):
    """Yield non-empty SQL statements, correctly handling comments and strings.

    Splits on ``;`` that appear outside of SQL single-line comments
    (``-- …``), block comments (``/* … */``), and quoted string literals.
    """
    stmt_chars: List[str] = []
    i = 0
    n = len(sql)

    while i < n:
        ch = sql[i]

        # -- single-line comment
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            # Consume to end of line (include newline so formatting is kept).
            while i < n and sql[i] != "\n":
                stmt_chars.append(sql[i])
                i += 1

        # /* block comment */
        elif ch == "/" and i + 1 < n and sql[i + 1] == "*":
            stmt_chars.append(ch)
            i += 1
            while i < n:
                stmt_chars.append(sql[i])
                if sql[i] == "*" and i + 1 < n and sql[i + 1] == "/":
                    stmt_chars.append(sql[i + 1])
                    i += 2
                    break
                i += 1

        # single-quoted string literal
        elif ch == "'":
            stmt_chars.append(ch)
            i += 1
            while i < n:
                stmt_chars.append(sql[i])
                if sql[i] == "'" and i + 1 < n and sql[i + 1] == "'":
                    # escaped quote
                    stmt_chars.append(sql[i + 1])
                    i += 2
                elif sql[i] == "'":
                    i += 1
                    break
                else:
                    i += 1

        # double-quoted identifier
        elif ch == '"':
            stmt_chars.append(ch)
            i += 1
            while i < n:
                stmt_chars.append(sql[i])
                if sql[i] == '"' and i + 1 < n and sql[i + 1] == '"':
                    stmt_chars.append(sql[i + 1])
                    i += 2
                elif sql[i] == '"':
                    i += 1
                    break
                else:
                    i += 1

        # statement terminator
        elif ch == ";":
            stmt = "".join(stmt_chars).strip()
            if stmt:
                yield stmt
            stmt_chars = []
            i += 1

        else:
            stmt_chars.append(ch)
            i += 1

    # Yield any trailing statement (no trailing semicolon).
    stmt = "".join(stmt_chars).strip()
    if stmt:
        yield stmt
