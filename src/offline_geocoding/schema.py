"""SQLite database schema for offline geocoding.

Tables
------
metadata          – key/value store for import settings
countries         – deduplicated ISO country codes + gzip-compressed name map
strings           – deduplicated name/address strings (normalised, shared)
categories        – deduplicated OSM category strings
places            – main place table (one row per photon place entry)
place_names       – N:M: place <-> string, with lang + kind columns
place_addresses   – N:M: place <-> string, with addr_type + lang columns
place_categories  – N:M: place <-> category
places_fts        – FTS5 virtual table with trigram tokeniser (fuzzy search)
places_rtree      – R-tree spatial index for reverse geocoding
"""

import gzip
import json
import logging
import sqlite3
from typing import Any, List

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_PRAGMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA temp_store   = MEMORY;
PRAGMA cache_size   = -65536;
PRAGMA mmap_size    = 268435456;
PRAGMA foreign_keys = ON;
"""

_TABLES = """
CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Deduplicated country records.
-- `names` is a gzip-compressed JSON object mapping lang -> localised name.
CREATE TABLE IF NOT EXISTS countries (
    id   INTEGER PRIMARY KEY,
    code TEXT UNIQUE NOT NULL,
    names BLOB NOT NULL DEFAULT X''
);

-- Deduplicated text strings shared by place_names and place_addresses.
CREATE TABLE IF NOT EXISTS strings (
    id    INTEGER PRIMARY KEY,
    value TEXT UNIQUE NOT NULL
);

-- Deduplicated OSM category strings (e.g. "amenity.restaurant").
CREATE TABLE IF NOT EXISTS categories (
    id   INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

-- Core place record.
-- centroid is stored as (lat, lon) and bbox likewise (WGS84).
-- `extra` is a gzip-compressed JSON blob of arbitrary extra tags.
CREATE TABLE IF NOT EXISTS places (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    photon_id    TEXT,
    osm_type     TEXT,
    osm_id       INTEGER,
    osm_key      TEXT,
    osm_value    TEXT,
    address_type TEXT,
    importance   REAL    NOT NULL DEFAULT 0.0,
    country_id   INTEGER REFERENCES countries(id),
    postcode     TEXT,
    housenumber  TEXT,
    lat          REAL    NOT NULL,
    lon          REAL    NOT NULL,
    bbox_min_lon REAL,
    bbox_min_lat REAL,
    bbox_max_lon REAL,
    bbox_max_lat REAL,
    extra        BLOB
);

-- Multilingual names for a place.
-- lang: 'default' for the bare `name` tag, otherwise an ISO language code.
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
CREATE INDEX IF NOT EXISTS idx_places_country    ON places(country_id);
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

def create_database(path: str) -> sqlite3.Connection:
    """Create (or open) the geocoding database at *path* and apply the schema.

    Returns an open :class:`sqlite3.Connection`.
    """
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Check FTS5 trigram availability (added in SQLite 3.34.0).
    sqlite_ver = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
    if sqlite_ver < (3, 34, 0):
        raise RuntimeError(
            f"SQLite {sqlite3.sqlite_version} does not support the FTS5 trigram "
            "tokeniser. Please upgrade to SQLite >= 3.34.0."
        )

    for stmt in _split_statements(_PRAGMA):
        conn.execute(stmt)

    for block in (_TABLES, _FTS, _RTREE, _INDEXES):
        for stmt in _split_statements(block):
            conn.execute(stmt)

    conn.commit()
    return conn


def apply_pragmas(conn: sqlite3.Connection) -> None:
    """Re-apply per-connection PRAGMAs (call after every new connection)."""
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
