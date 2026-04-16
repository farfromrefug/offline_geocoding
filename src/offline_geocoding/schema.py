"""SQLite database schema for offline geocoding.

Design Goals
------------
* **Zero cross-worker duplication** – strings and categories are interned and
  identified by integer primary key.  Multiple worker processes each build
  their own temporary database; a SQL-driven merge step deduplicates across
  workers using temp mapping tables.
* **Compact storage** – shared ``strings`` table means "Paris" is stored once
  even if thousands of places reference it as a city or name.  ``langs``,
  ``name_kinds``, and ``addr_types`` are tiny lookup tables whose IDs replace
  repeated TEXT columns throughout the schema.
* **Language-aware queries** – all name/address variants stored as rows in
  ``place_names`` / ``place_addresses`` with ``lang_id`` referencing the
  ``langs`` table.  Country names stored in ``country_names`` the same way.
  OSM tag tokens (keys and values from categories) translated via
  ``osm_tag_names`` and indexed in FTS5.
* **Fast search** – FTS5 contentless table (no ``_content`` shadow table) with
  trigram tokeniser for fuzzy substring matching; R-tree for reverse geocoding.
  Category tokens and their translations are included in the FTS ``names``
  column so that ``"toto restaurant"`` or ``"toto natural peak"`` both work.

Normalisation rules
-------------------
* ``langs`` / ``name_kinds`` / ``addr_types`` are **pre-seeded** at database
  creation time with fixed IDs from the constants ``NAME_KIND_IDS`` and
  ``ADDR_TYPE_IDS``.  ``langs`` gets IDs assigned from sorted({"default"} |
  user_languages).  Because all worker databases and the final database are
  created with the same language list and the same pre-seeded tables, their
  IDs are **always identical** — no remapping is needed for these tables
  during the merge step.
* ``strings`` (via ``_smap``), ``categories`` (via ``_cmap``), and
  ``osm_tags`` (via ``_tmap``) need ID remapping during merge.

Tables (final database)
-----------------------
metadata         – key/value store for import settings
langs            – interned language codes; INTEGER PK, pre-seeded
name_kinds       – interned name-kind strings ('name', 'alt', ...); pre-seeded
addr_types       – interned address-type strings ('city', 'country', ...); pre-seeded
strings          – interned text values; INTEGER PRIMARY KEY (AUTOINCREMENT)
categories       – interned OSM category strings; INTEGER PRIMARY KEY
osm_tags         – interned OSM tag tokens; id PK, token TEXT, ctx TEXT UNIQUE
osm_tag_names    – localised labels for tag tokens; references osm_tags + strings + langs
countries        – ISO country codes keyed by ``code TEXT PRIMARY KEY``
country_names    – localised country names; references strings + langs
places           – one row per photon place entry; normalised columns
place_names      – N:M place ↔ string with lang_id + kind_id columns
place_addresses  – N:M place ↔ string with addr_type_id + lang_id columns
place_categories – N:M place ↔ category
place_osm_tags   – N:M place ↔ osm_tag (all tokens from split categories)
places_fts       – FTS5 contentless virtual table, trigram tokeniser
places_rtree     – R-tree spatial index; stores centroid as (min==max) point

Worker databases (temporary, one per worker process)
-----------------------------------------------------
Same as above *minus* the virtual tables (FTS5 / R-tree).
Instead two plain tables hold the data that will be loaded into the virtual
tables during the final merge step:

fts_data   – (place_id, names TEXT, address TEXT)
rtree_data – (id, lat INTEGER, lon INTEGER)   -- centroid only
"""

import gzip
import json
import logging
import sqlite3
from typing import Any, Dict, List, Sequence

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
# Fixed normalisation mappings
# ---------------------------------------------------------------------------

#: Maps OSM name-kind string to its pre-seeded integer ID.
#: These IDs are the same in every worker DB and the final DB.
NAME_KIND_IDS: Dict[str, int] = {
    "name":     1,
    "alt":      2,
    "old":      3,
    "int":      4,
    "loc":      5,
    "short":    6,
    "official": 7,
}

#: Maps address-type string to its pre-seeded integer ID.
#: These IDs are the same in every worker DB and the final DB.
ADDR_TYPE_IDS: Dict[str, int] = {
    "country":  1,
    "state":    2,
    "county":   3,
    "city":     4,
    "district": 5,
    "locality": 6,
    "street":   7,
    "other":    8,
}


def build_lang_ids(languages: Sequence[str]) -> Dict[str, int]:
    """Return ``{lang_code: id}`` for ``'default'`` plus every language in
    *languages*.

    IDs are assigned to sorted codes (ascending), starting at 1.  Because all
    databases in an import run are created with the same *languages* list,
    they all produce the same mapping — no ID remapping is needed during merge.
    """
    codes = sorted({"default"} | set(languages))
    return {code: i + 1 for i, code in enumerate(codes)}


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

-- Interned language codes.  Seeded at DB creation time from the user's
-- language list + 'default'.  Same IDs in all worker DBs and the final DB.
CREATE TABLE IF NOT EXISTS langs (
    id   INTEGER PRIMARY KEY,
    code TEXT UNIQUE NOT NULL
);

-- Interned name-kind strings ('name', 'alt', 'old', ...).
-- Pre-seeded with fixed IDs from NAME_KIND_IDS.
CREATE TABLE IF NOT EXISTS name_kinds (
    id   INTEGER PRIMARY KEY,
    kind TEXT UNIQUE NOT NULL
);

-- Interned address-type strings ('country', 'city', 'street', ...).
-- Pre-seeded with fixed IDs from ADDR_TYPE_IDS.
CREATE TABLE IF NOT EXISTS addr_types (
    id   INTEGER PRIMARY KEY,
    type TEXT UNIQUE NOT NULL
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

-- Interned OSM tag tokens: individual parts from split categories.
-- ``token`` is the raw string (e.g., "natural", "peak", "amenity").
-- ``ctx`` is the translation lookup key and is UNIQUE:
--   - for a key token: ctx == token  (e.g., "natural")
--   - for a value token: ctx == "{parent_key}={token}"  (e.g., "natural=peak")
-- Merge-remapped via ``_tmap`` (same pattern as strings / categories).
CREATE TABLE IF NOT EXISTS osm_tags (
    id    INTEGER PRIMARY KEY,
    token TEXT NOT NULL,
    ctx   TEXT UNIQUE NOT NULL
);

-- Localised (translated) labels for OSM tag tokens.
-- ``string_id`` references strings(id): the translated label text.
-- ``lang_id``   references langs(id).
-- (tag_id, lang_id) PRIMARY KEY – one translation per language per token.
CREATE TABLE IF NOT EXISTS osm_tag_names (
    tag_id    INTEGER NOT NULL REFERENCES osm_tags(id),
    string_id INTEGER NOT NULL REFERENCES strings(id),
    lang_id   INTEGER NOT NULL REFERENCES langs(id),
    PRIMARY KEY (tag_id, lang_id)
);

-- One row per country.
-- ``code`` is the ISO 3166-1 alpha-2 code used as a natural key.
-- Country names are stored in ``country_names`` (like place_names).
CREATE TABLE IF NOT EXISTS countries (
    code TEXT PRIMARY KEY
);

-- Localised country names.
-- Same structure as place_names: string_id references strings(id),
-- lang_id references langs(id).  This makes country names queryable
-- via FTS5 like any other address component.
CREATE TABLE IF NOT EXISTS country_names (
    code      TEXT    NOT NULL REFERENCES countries(code) ON DELETE CASCADE,
    string_id INTEGER NOT NULL REFERENCES strings(id),
    lang_id   INTEGER NOT NULL REFERENCES langs(id),
    PRIMARY KEY (code, lang_id)
);

-- Core place record.  Heavily normalised: TEXT columns replaced by INTEGER
-- foreign keys wherever the cardinality is bounded.
-- ``importance`` stored as round(value * IMPORTANCE_SCALE) (INTEGER).
-- ``lat`` / ``lon`` stored as round(degrees * LAT_LON_SCALE) (INTEGER).
-- ``osm_key_id`` / ``osm_value_id`` reference osm_tags(id), NOT strings(id).
-- They share the same interning table as split category parts.
CREATE TABLE IF NOT EXISTS places (
    id           INTEGER PRIMARY KEY,
    photon_id    TEXT,
    osm_id       INTEGER,
    osm_key_id   INTEGER REFERENCES osm_tags(id),
    osm_value_id INTEGER REFERENCES osm_tags(id),
    addr_type_id INTEGER REFERENCES addr_types(id),
    importance   INTEGER NOT NULL DEFAULT 0,
    country_code TEXT    REFERENCES countries(code),
    postcode_id  INTEGER REFERENCES strings(id),
    hn_id        INTEGER REFERENCES strings(id),
    lat          INTEGER NOT NULL,
    lon          INTEGER NOT NULL,
    extra        BLOB
);

-- Multilingual names for a place.
-- lang_id → langs(id); kind_id → name_kinds(id).
CREATE TABLE IF NOT EXISTS place_names (
    place_id  INTEGER NOT NULL REFERENCES places(id)     ON DELETE CASCADE,
    string_id INTEGER NOT NULL REFERENCES strings(id),
    lang_id   INTEGER NOT NULL REFERENCES langs(id),
    kind_id   INTEGER NOT NULL REFERENCES name_kinds(id),
    PRIMARY KEY (place_id, string_id, lang_id, kind_id)
);

-- Address components for a place.
-- addr_type_id → addr_types(id); lang_id → langs(id).
CREATE TABLE IF NOT EXISTS place_addresses (
    place_id     INTEGER NOT NULL REFERENCES places(id)     ON DELETE CASCADE,
    string_id    INTEGER NOT NULL REFERENCES strings(id),
    addr_type_id INTEGER NOT NULL REFERENCES addr_types(id),
    lang_id      INTEGER NOT NULL REFERENCES langs(id),
    PRIMARY KEY (place_id, addr_type_id, lang_id)
);

-- Categories assigned to a place.
CREATE TABLE IF NOT EXISTS place_categories (
    place_id    INTEGER NOT NULL REFERENCES places(id)      ON DELETE CASCADE,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    PRIMARY KEY (place_id, category_id)
);

-- OSM tag tokens assigned to a place (from split category parts).
-- Enables efficient "filter by tag token" queries and drives FTS indexing.
CREATE TABLE IF NOT EXISTS place_osm_tags (
    place_id INTEGER NOT NULL REFERENCES places(id)  ON DELETE CASCADE,
    tag_id   INTEGER NOT NULL REFERENCES osm_tags(id),
    PRIMARY KEY (place_id, tag_id)
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

-- Staging table for R-tree data.
-- Only the centroid (lat, lon) is stored; the R-tree virtual table receives
-- a degenerate bounding box where min == max (a point), which always satisfies
-- the SQLite R-tree constraint min_lat <= max_lat / min_lon <= max_lon.
-- Coordinates are stored as round(degrees * LAT_LON_SCALE) (INTEGER).
CREATE TABLE IF NOT EXISTS rtree_data (
    id  INTEGER PRIMARY KEY,
    lat INTEGER NOT NULL,
    lon INTEGER NOT NULL
);
"""

# FTS5 contentless virtual table.
# ``content=''`` means the FTS5 shadow ``_content`` table is NOT populated –
# names/address text are NOT stored twice.  The trigram index (_data) is
# built from the text passed during INSERT and is used for MATCH queries.
# Queries retrieve the matching ``rowid`` (== place_id) and join ``places``.
_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS places_fts USING fts5(
    names,
    address,
    content='',
    tokenize = 'trigram case_sensitive 0'
);
"""

# R-tree virtual table for fast bounding-box reverse geocoding.
# Centroid stored as a degenerate box: min_lat == max_lat, min_lon == max_lon.
_RTREE = """
CREATE VIRTUAL TABLE IF NOT EXISTS places_rtree USING rtree(
    id,
    min_lat, max_lat,
    min_lon, max_lon
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_places_country    ON places(country_code);
CREATE INDEX IF NOT EXISTS idx_places_osm        ON places(osm_id);
CREATE INDEX IF NOT EXISTS idx_places_importance ON places(importance DESC);
CREATE INDEX IF NOT EXISTS idx_places_photon_id  ON places(photon_id);
CREATE INDEX IF NOT EXISTS idx_place_names_place ON place_names(place_id);
CREATE INDEX IF NOT EXISTS idx_place_names_lang  ON place_names(lang_id, kind_id);
CREATE INDEX IF NOT EXISTS idx_place_addr_place  ON place_addresses(place_id);
CREATE INDEX IF NOT EXISTS idx_place_cat_place   ON place_categories(place_id);
CREATE INDEX IF NOT EXISTS idx_country_names     ON country_names(code);
CREATE INDEX IF NOT EXISTS idx_osm_tags_ctx       ON osm_tags(ctx);
CREATE INDEX IF NOT EXISTS idx_osm_tag_names_tag  ON osm_tag_names(tag_id);
CREATE INDEX IF NOT EXISTS idx_place_osm_tags_place ON place_osm_tags(place_id);
CREATE INDEX IF NOT EXISTS idx_place_osm_tags_tag   ON place_osm_tags(tag_id);
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


def _seed_fixed_tables(
    conn: sqlite3.Connection,
    languages: Sequence[str],
) -> None:
    """Pre-seed ``langs``, ``name_kinds``, and ``addr_types`` with fixed IDs.

    Safe to call multiple times (``INSERT OR IGNORE``).  Must be called
    inside an active transaction or with autocommit.
    """
    # langs: sorted({"default"} | set(languages)), IDs start at 1.
    for code, lid in build_lang_ids(languages).items():
        conn.execute(
            "INSERT OR IGNORE INTO langs(id, code) VALUES (?, ?)", (lid, code)
        )

    # name_kinds: fixed mapping from NAME_KIND_IDS.
    for kind, kid in NAME_KIND_IDS.items():
        conn.execute(
            "INSERT OR IGNORE INTO name_kinds(id, kind) VALUES (?, ?)", (kid, kind)
        )

    # addr_types: fixed mapping from ADDR_TYPE_IDS.
    for atype, aid in ADDR_TYPE_IDS.items():
        conn.execute(
            "INSERT OR IGNORE INTO addr_types(id, type) VALUES (?, ?)", (aid, atype)
        )


def create_database(
    path: str,
    languages: Sequence[str] = (),
    with_indexes: bool = True,
) -> sqlite3.Connection:
    """Create (or open) the **final** geocoding database and apply its schema.

    The final database has FTS5 and R-tree virtual tables in addition to all
    common tables.

    Parameters
    ----------
    path:
        Filesystem path for the SQLite file.
    languages:
        ISO language codes to pre-seed into the ``langs`` table.  Should be
        the same list that is passed to worker processes so that
        ``lang_id`` values are consistent across all databases.
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

    _seed_fixed_tables(conn, languages)

    if with_indexes:
        create_indexes(conn)

    return conn


def create_indexes(conn: sqlite3.Connection) -> None:
    """Create (or ensure existence of) all secondary B-tree indexes.

    Safe to call multiple times – every statement uses ``IF NOT EXISTS``.
    """
    for stmt in _split_statements(_INDEXES):
        conn.execute(stmt)


def create_worker_database(
    path: str,
    languages: Sequence[str] = (),
) -> sqlite3.Connection:
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

    _seed_fixed_tables(conn, languages)

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
